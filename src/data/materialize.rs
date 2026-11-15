use crate::ast::{Item, ItemKind, StructDef, Type, TypeKind};
use crate::data::layout::{CanonicalTypeTag, LogicalFieldDesc, LogicalLayoutV1};
use crate::data::{DataValue, Document, ExactNumber};
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum MaterializeError {
    MissingField { ordinal: u16, name: String },
    TypeMismatch { ordinal: u16, name: String, expected: CanonicalTypeTag },
    NullWhereNotNullable { ordinal: u16, name: String },
    NominalTypeMismatch { expected: String, received: String },
    ExtraFieldInStrictLayout(String),
    DuplicateField(String),
    WidthOverflow { ordinal: u16, name: String, expected: CanonicalTypeTag, raw: String },
    TrailingJunk { offset: usize },
    MalformedInput(String),
    Cancelled { initialized_count: usize },
    BufferTooSmall { required: usize, actual: usize },
    UnalignedBuffer { required_align: usize, ptr_addr: usize },
    UninitializedSlot { ordinal: u16, name: String },
}

impl fmt::Display for MaterializeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MaterializeError::MissingField { ordinal, name } => {
                write!(f, "Missing required field '{}' at ordinal {}", name, ordinal)
            }
            MaterializeError::TypeMismatch { ordinal, name, expected } => {
                write!(
                    f,
                    "Type mismatch for field '{}' at ordinal {}: expected {:?}",
                    name, ordinal, expected
                )
            }
            MaterializeError::NullWhereNotNullable { ordinal, name } => {
                write!(
                    f,
                    "Null value not permitted for non-nullable field '{}' at ordinal {}",
                    name, ordinal
                )
            }
            MaterializeError::NominalTypeMismatch { expected, received } => {
                write!(
                    f,
                    "Nominal type mismatch: expected '{}', received '{}'",
                    expected, received
                )
            }
            MaterializeError::ExtraFieldInStrictLayout(k) => {
                write!(f, "Unrecognized field '{}' in strict struct layout", k)
            }
            MaterializeError::DuplicateField(k) => {
                write!(f, "Duplicate field '{}' in strict struct layout", k)
            }
            MaterializeError::WidthOverflow { ordinal, name, expected, raw } => {
                write!(
                    f,
                    "Width overflow for field '{}' at ordinal {} (expected {:?}): '{}'",
                    name, ordinal, expected, raw
                )
            }
            MaterializeError::TrailingJunk { offset } => {
                write!(f, "Trailing junk after struct payload at byte offset {}", offset)
            }
            MaterializeError::MalformedInput(msg) => {
                write!(f, "Malformed input during direct struct decode: {}", msg)
            }
            MaterializeError::Cancelled { initialized_count } => {
                write!(
                    f,
                    "Struct decode cancelled after initializing {} field(s)",
                    initialized_count
                )
            }
            MaterializeError::BufferTooSmall { required, actual } => {
                write!(
                    f,
                    "Contiguous struct buffer too small: required {} bytes, got {}",
                    required, actual
                )
            }
            MaterializeError::UnalignedBuffer { required_align, ptr_addr } => {
                write!(
                    f,
                    "Contiguous struct buffer unaligned: required {}-byte alignment for {:#x}",
                    required_align, ptr_addr
                )
            }
            MaterializeError::UninitializedSlot { ordinal, name } => {
                write!(f, "Uninitialized slot '{}' at ordinal {}", name, ordinal)
            }
        }
    }
}

impl std::error::Error for MaterializeError {}

/// Materialized struct representation with fixed slots indexed directly by declaration ordinal.
/// Conforms to Gate-SCHEMA-09: Direct VM slot decoding decodes fields by ordinal without hash lookups.
#[derive(Debug, Clone, PartialEq)]
pub struct MaterializedStruct {
    pub layout: LogicalLayoutV1,
    pub slots: Vec<DataValue>, // strictly indexed by field.ordinal (0..N-1)
}

impl MaterializedStruct {
    pub fn new(layout: LogicalLayoutV1, slots: Vec<DataValue>) -> Self {
        Self { layout, slots }
    }

    /// O(1) direct slot read by declaration-order ordinal (Gate-SCHEMA-09)
    #[inline(always)]
    pub fn get_slot(&self, ordinal: usize) -> Option<&DataValue> {
        self.slots.get(ordinal)
    }

    /// O(1) direct slot write by declaration-order ordinal
    #[inline(always)]
    pub fn set_slot(&mut self, ordinal: usize, val: DataValue) -> Result<(), MaterializeError> {
        if ordinal < self.slots.len() {
            self.slots[ordinal] = val;
            Ok(())
        } else {
            Err(MaterializeError::MissingField {
                ordinal: ordinal as u16,
                name: format!("slot_{}", ordinal),
            })
        }
    }

    /// Dematerialize back into an ordered Document
    pub fn to_document(&self) -> Document {
        let mut doc = Document::with_capacity(self.layout.fields.len());
        for field in &self.layout.fields {
            let val = self.slots.get(field.ordinal as usize).cloned().unwrap_or(DataValue::Null);
            let _ = doc.try_insert(field.name.clone(), val);
        }
        doc
    }

    /// Serialize directly to BSON bytes
    pub fn to_bson(&self) -> Vec<u8> {
        let doc = self.to_document();
        crate::data::bson::to_vec(&doc)
    }

    /// Serialize directly to JSON string
    pub fn to_json(&self) -> Result<String, crate::data::json::JsonError> {
        let doc = self.to_document();
        crate::data::json::to_string(&DataValue::Document(doc))
    }

    /// Convert slots to DB query parameters
    pub fn to_db_params(&self) -> Vec<(String, DataValue)> {
        let mut params = Vec::with_capacity(self.layout.fields.len());
        for field in &self.layout.fields {
            let val = self.slots.get(field.ordinal as usize).cloned().unwrap_or(DataValue::Null);
            params.push((field.name.clone(), val));
        }
        params
    }
}

/// Helper to convert AST Type to CanonicalTypeTag and nullability/optionality flags.
pub fn type_to_canonical_tag(ty: &Type) -> (CanonicalTypeTag, bool, bool) {
    match &ty.kind {
        TypeKind::Int => (CanonicalTypeTag::Int64, false, false),
        TypeKind::Float => (CanonicalTypeTag::Float64, false, false),
        TypeKind::String => (CanonicalTypeTag::StringSlice, false, false),
        TypeKind::Bool => (CanonicalTypeTag::Bool, false, false),
        TypeKind::Option(inner) => {
            let (tag, _, _) = type_to_canonical_tag(inner);
            (tag, true, true)
        }
        TypeKind::Named(name) => match name.as_str() {
            "ObjectId" => (CanonicalTypeTag::ObjectId, false, false),
            "DateTimeUtc" => (CanonicalTypeTag::DateTimeUtc, false, false),
            "Decimal128" => (CanonicalTypeTag::Decimal128, false, false),
            "Binary" | "Bytes" => (CanonicalTypeTag::BinarySlice, false, false),
            _ => {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                std::hash::Hash::hash(name, &mut hasher);
                (
                    CanonicalTypeTag::NestedStruct(std::hash::Hasher::finish(&mut hasher)),
                    false,
                    false,
                )
            }
        },
        _ => (CanonicalTypeTag::StringSlice, false, false),
    }
}

/// Convert an AST StructDef into a LogicalLayoutV1
pub fn struct_def_to_layout(def: &StructDef) -> LogicalLayoutV1 {
    let mut fields = Vec::with_capacity(def.fields.len());
    for (i, f) in def.fields.iter().enumerate() {
        let (tag, is_nullable, is_optional) = type_to_canonical_tag(&f.ty);
        fields.push(
            LogicalFieldDesc::new(i as u16, &f.name, tag).with_nullability(is_nullable, is_optional),
        );
    }
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    std::hash::Hash::hash(&def.name, &mut hasher);
    let type_id = std::hash::Hasher::finish(&hasher);
    LogicalLayoutV1::new(type_id, &def.name, fields)
}

/// Direct materialization into ItemKind::Struct slots from an AST StructDef.
pub fn materialize_into_struct(
    def: &StructDef,
    doc: &Document,
) -> Result<MaterializedStruct, MaterializeError> {
    let layout = struct_def_to_layout(def);
    materialize_from_document(&layout, doc)
}

/// Materialize from an AST Item (verifies ItemKind::Struct).
pub fn materialize_from_item(
    item: &Item,
    doc: &Document,
) -> Result<MaterializedStruct, MaterializeError> {
    match &item.kind {
        ItemKind::Struct(def) => materialize_into_struct(def, doc),
        _ => Err(MaterializeError::NominalTypeMismatch {
            expected: "ItemKind::Struct".to_string(),
            received: format!("{:?}", item.kind),
        }),
    }
}

/// Materialize a Document into fixed slots according to declaration-order ordinals.
/// Rejects partial initialization on failure and cleans up in reverse order.
/// Enforces:
/// 1. Nominal type check if document contains nominal type markers (_type, __type, $type)
/// 2. Missing required field -> MissingField (fails closed)
/// 3. Required nullable field that is absent -> MissingField, not null (fails closed)
/// 4. Null value where field is not nullable -> NullWhereNotNullable (fails closed)
/// 5. Value type mismatch -> TypeMismatch (fails closed)
pub fn materialize_from_document(
    layout: &LogicalLayoutV1,
    doc: &Document,
) -> Result<MaterializedStruct, MaterializeError> {
    // 1. Nominal type verification
    if let Some(tag_val) = doc.get("_type").or_else(|| doc.get("__type")).or_else(|| doc.get("$type")) {
        if let DataValue::String(s) = tag_val {
            if s != &layout.type_name {
                return Err(MaterializeError::NominalTypeMismatch {
                    expected: layout.type_name.clone(),
                    received: s.clone(),
                });
            }
        }
    }

    // Check for extra fields in strict layout
    for (key, _) in doc.entries() {
        if key == "_type" || key == "__type" || key == "$type" {
            continue;
        }
        if !layout.fields.iter().any(|f| &f.name == key) {
            return Err(MaterializeError::ExtraFieldInStrictLayout(key.clone()));
        }
    }

    let max_ordinal = layout
        .fields
        .iter()
        .map(|f| f.ordinal as usize)
        .max()
        .unwrap_or(0);
    let mut slots = vec![DataValue::Null; max_ordinal + 1];

    for field in &layout.fields {
        match doc.get(&field.name) {
            Some(val) => {
                if !matches_tag(val, field.tag) {
                    if field.is_nullable && matches!(val, DataValue::Null) {
                        slots[field.ordinal as usize] = DataValue::Null;
                    } else if matches!(val, DataValue::Null) {
                        // Cleanup allocated slots on failure
                        slots.clear();
                        return Err(MaterializeError::NullWhereNotNullable {
                            ordinal: field.ordinal,
                            name: field.name.clone(),
                        });
                    } else {
                        // Cleanup allocated slots on failure
                        slots.clear();
                        return Err(MaterializeError::TypeMismatch {
                            ordinal: field.ordinal,
                            name: field.name.clone(),
                            expected: field.tag,
                        });
                    }
                } else {
                    slots[field.ordinal as usize] = val.clone();
                }
            }
            None => {
                if field.is_optional {
                    slots[field.ordinal as usize] = DataValue::Null;
                } else {
                    // Required field (including required nullable fields) must be present in doc
                    slots.clear();
                    return Err(MaterializeError::MissingField {
                        ordinal: field.ordinal,
                        name: field.name.clone(),
                    });
                }
            }
        }
    }

    Ok(MaterializedStruct::new(layout.clone(), slots))
}

fn matches_tag(val: &DataValue, tag: CanonicalTypeTag) -> bool {
    match (val, tag) {
        (DataValue::Bool(_), CanonicalTypeTag::Bool) => true,
        (DataValue::Number(ExactNumber::Int(_)), CanonicalTypeTag::Int64) => true,
        (DataValue::Number(ExactNumber::UInt(_)), CanonicalTypeTag::UInt64) => true,
        (DataValue::Number(ExactNumber::Float(_)), CanonicalTypeTag::Float64) => true,
        (DataValue::Number(ExactNumber::Decimal128(_)), CanonicalTypeTag::Decimal128) => true,
        (DataValue::ObjectId(_), CanonicalTypeTag::ObjectId) => true,
        (DataValue::DateTimeUtc(_), CanonicalTypeTag::DateTimeUtc) => true,
        (DataValue::String(_), CanonicalTypeTag::StringSlice) => true,
        (DataValue::Binary(_), CanonicalTypeTag::BinarySlice) => true,
        (DataValue::Document(_), CanonicalTypeTag::NestedStruct(_)) => true,
        _ => false,
    }
}

/// Instrumentation counters for Gate-SCHEMA-09 verification.
/// Separates decode key dispatch from post-commit field lookup and tracks DOM/heap allocations.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LayoutPerfCounters {
    pub dom_node_constructions: u64,
    pub intermediate_map_allocations: u64,
    pub decode_key_dispatches: u64,
    pub field_read_name_lookups: u64,
    pub field_read_hash_lookups: u64,
    pub field_read_allocations: u64,
    pub partial_init_cleanups: u64,
    pub owned_slice_allocations: u64,
    pub owned_slice_frees: u64,
}

/// Aligned contiguous byte buffer for C-compatible struct layout storage.
#[derive(Debug)]
pub struct AlignedStorage {
    ptr: std::ptr::NonNull<u8>,
    layout: std::alloc::Layout,
}

impl AlignedStorage {
    /// Allocates a zero-initialized contiguous buffer with at least `align` alignment.
    pub fn new(size: usize, align: usize) -> Self {
        let alloc_size = size.max(1);
        let alloc_align = align.max(1).next_power_of_two();
        let layout = std::alloc::Layout::from_size_align(alloc_size, alloc_align)
            .expect("valid aligned storage layout");
        // SAFETY: `alloc_size >= 1` and `alloc_align` is a power of two.
        let raw = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr = std::ptr::NonNull::new(raw).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Self { ptr, layout }
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.layout.size()
    }

    #[inline(always)]
    pub fn align(&self) -> usize {
        self.layout.align()
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `self.ptr` is valid for `self.layout.size()` bytes for the lifetime of `self`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.layout.size()) }
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `self.ptr` is uniquely owned and valid for `self.layout.size()` bytes.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.layout.size()) }
    }
}

impl Drop for AlignedStorage {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` was allocated with `std::alloc::alloc_zeroed(self.layout)`.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

/// Owned payload backing a managed slice or nested struct inside `ContiguousStruct`.
#[derive(Debug)]
pub enum OwnedFieldPayload {
    Bytes(Box<[u8]>),
    Nested(Box<ContiguousStruct>),
}

/// Contiguous C-compatible struct instance backed by `AlignedStorage`.
/// Fields are laid out at `LogicalFieldDesc::byte_offset` with C ABI alignment and tail padding.
/// Post-commit field reads use `(ordinal, byte_offset)` directly with 0 name/hash lookups and 0 allocations.
#[derive(Debug)]
pub struct ContiguousStruct {
    pub layout: LogicalLayoutV1,
    storage: AlignedStorage,
    init_mask: Vec<bool>,
    null_mask: Vec<bool>,
    init_order: Vec<u16>,
    owned_slots: Vec<Option<OwnedFieldPayload>>,
}

impl ContiguousStruct {
    /// Allocates an uninitialized contiguous C-compatible buffer for `layout`.
    pub fn new_uninit(layout: LogicalLayoutV1) -> Self {
        let n = layout.fields.len();
        let storage = AlignedStorage::new(layout.size_bytes, layout.align_bytes);
        let mut owned_slots = Vec::with_capacity(n);
        for _ in 0..n {
            owned_slots.push(None);
        }
        Self {
            layout,
            storage,
            init_mask: vec![false; n],
            null_mask: vec![false; n],
            init_order: Vec::with_capacity(n),
            owned_slots,
        }
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *const u8 {
        self.storage.as_ptr()
    }

    #[inline(always)]
    pub fn raw_bytes(&self) -> &[u8] {
        let size = self.layout.size_bytes.min(self.storage.len());
        &self.storage.as_slice()[..size]
    }

    #[inline(always)]
    pub fn is_initialized(&self, ordinal: u16) -> bool {
        self.init_mask.get(ordinal as usize).copied().unwrap_or(false)
    }

    #[inline(always)]
    pub fn is_null(&self, ordinal: u16) -> bool {
        self.null_mask.get(ordinal as usize).copied().unwrap_or(false)
    }

    pub fn initialized_count(&self) -> usize {
        self.init_order.len()
    }

    /// Cleans up any partially initialized fields in reverse initialization order.
    pub fn cleanup_partial(&mut self, mut counters: Option<&mut LayoutPerfCounters>) -> usize {
        let mut cleaned = 0usize;
        while let Some(ord) = self.init_order.pop() {
            let idx = ord as usize;
            if let Some(payload) = self.owned_slots.get_mut(idx).and_then(|s| s.take()) {
                if matches!(payload, OwnedFieldPayload::Bytes(_)) {
                    if let Some(c) = counters.as_deref_mut() {
                        c.owned_slice_frees += 1;
                    }
                }
                drop(payload);
            }
            if let Some(flag) = self.init_mask.get_mut(idx) {
                *flag = false;
            }
            if let Some(flag) = self.null_mask.get_mut(idx) {
                *flag = false;
            }
            cleaned += 1;
            if let Some(c) = counters.as_deref_mut() {
                c.partial_init_cleanups += 1;
            }
        }
        cleaned
    }

    #[inline(always)]
    fn check_read_slot(&self, ordinal: u16, byte_offset: usize, width: usize) -> Result<bool, MaterializeError> {
        let idx = ordinal as usize;
        if !self.init_mask.get(idx).copied().unwrap_or(false) {
            return Err(MaterializeError::UninitializedSlot {
                ordinal,
                name: format!("slot_{}", ordinal),
            });
        }
        if self.null_mask.get(idx).copied().unwrap_or(false) {
            return Ok(true);
        }
        if byte_offset.saturating_add(width) > self.storage.len() {
            return Err(MaterializeError::BufferTooSmall {
                required: byte_offset.saturating_add(width),
                actual: self.storage.len(),
            });
        }
        Ok(false)
    }

    #[inline(always)]
    pub fn read_bool_at(
        &self,
        ordinal: u16,
        byte_offset: usize,
        _counters: Option<&mut LayoutPerfCounters>,
    ) -> Result<Option<bool>, MaterializeError> {
        if self.check_read_slot(ordinal, byte_offset, 1)? {
            return Ok(None);
        }
        Ok(Some(self.storage.as_slice()[byte_offset] != 0))
    }

    #[inline(always)]
    pub fn read_i64_at(
        &self,
        ordinal: u16,
        byte_offset: usize,
        _counters: Option<&mut LayoutPerfCounters>,
    ) -> Result<Option<i64>, MaterializeError> {
        if self.check_read_slot(ordinal, byte_offset, 8)? {
            return Ok(None);
        }
        let bytes: [u8; 8] = self.storage.as_slice()[byte_offset..byte_offset + 8]
            .try_into()
            .expect("8-byte slice");
        Ok(Some(i64::from_ne_bytes(bytes)))
    }

    #[inline(always)]
    pub fn read_u64_at(
        &self,
        ordinal: u16,
        byte_offset: usize,
        _counters: Option<&mut LayoutPerfCounters>,
    ) -> Result<Option<u64>, MaterializeError> {
        if self.check_read_slot(ordinal, byte_offset, 8)? {
            return Ok(None);
        }
        let bytes: [u8; 8] = self.storage.as_slice()[byte_offset..byte_offset + 8]
            .try_into()
            .expect("8-byte slice");
        Ok(Some(u64::from_ne_bytes(bytes)))
    }

    #[inline(always)]
    pub fn read_f64_at(
        &self,
        ordinal: u16,
        byte_offset: usize,
        _counters: Option<&mut LayoutPerfCounters>,
    ) -> Result<Option<f64>, MaterializeError> {
        if self.check_read_slot(ordinal, byte_offset, 8)? {
            return Ok(None);
        }
        let bytes: [u8; 8] = self.storage.as_slice()[byte_offset..byte_offset + 8]
            .try_into()
            .expect("8-byte slice");
        Ok(Some(f64::from_ne_bytes(bytes)))
    }

    #[inline(always)]
    pub fn read_datetime_utc_at(
        &self,
        ordinal: u16,
        byte_offset: usize,
        counters: Option<&mut LayoutPerfCounters>,
    ) -> Result<Option<i64>, MaterializeError> {
        self.read_i64_at(ordinal, byte_offset, counters)
    }

    #[inline(always)]
    pub fn read_object_id_at(
        &self,
        ordinal: u16,
        byte_offset: usize,
        _counters: Option<&mut LayoutPerfCounters>,
    ) -> Result<Option<[u8; 12]>, MaterializeError> {
        if self.check_read_slot(ordinal, byte_offset, 12)? {
            return Ok(None);
        }
        let bytes: [u8; 12] = self.storage.as_slice()[byte_offset..byte_offset + 12]
            .try_into()
            .expect("12-byte slice");
        Ok(Some(bytes))
    }

    #[inline(always)]
    pub fn read_decimal128_at(
        &self,
        ordinal: u16,
        byte_offset: usize,
        _counters: Option<&mut LayoutPerfCounters>,
    ) -> Result<Option<i128>, MaterializeError> {
        if self.check_read_slot(ordinal, byte_offset, 16)? {
            return Ok(None);
        }
        let bytes: [u8; 16] = self.storage.as_slice()[byte_offset..byte_offset + 16]
            .try_into()
            .expect("16-byte slice");
        Ok(Some(i128::from_ne_bytes(bytes)))
    }

    #[inline(always)]
    pub fn read_str_at(
        &self,
        ordinal: u16,
        byte_offset: usize,
        _counters: Option<&mut LayoutPerfCounters>,
    ) -> Result<Option<&str>, MaterializeError> {
        let slice_width = self.layout.target_abi.pointer_width * 2 + 8;
        if self.check_read_slot(ordinal, byte_offset, slice_width)? {
            return Ok(None);
        }
        match self.owned_slots.get(ordinal as usize).and_then(|o| o.as_ref()) {
            Some(OwnedFieldPayload::Bytes(b)) => {
                let s = std::str::from_utf8(b).map_err(|e| {
                    MaterializeError::MalformedInput(format!("Invalid UTF-8 in slot {}: {}", ordinal, e))
                })?;
                Ok(Some(s))
            }
            _ => Err(MaterializeError::UninitializedSlot {
                ordinal,
                name: format!("slot_{}", ordinal),
            }),
        }
    }

    #[inline(always)]
    pub fn read_bytes_at(
        &self,
        ordinal: u16,
        byte_offset: usize,
        _counters: Option<&mut LayoutPerfCounters>,
    ) -> Result<Option<&[u8]>, MaterializeError> {
        let slice_width = self.layout.target_abi.pointer_width * 2 + 8;
        if self.check_read_slot(ordinal, byte_offset, slice_width)? {
            return Ok(None);
        }
        match self.owned_slots.get(ordinal as usize).and_then(|o| o.as_ref()) {
            Some(OwnedFieldPayload::Bytes(b)) => Ok(Some(b.as_ref())),
            _ => Err(MaterializeError::UninitializedSlot {
                ordinal,
                name: format!("slot_{}", ordinal),
            }),
        }
    }

    #[inline(always)]
    pub fn read_nested_at(
        &self,
        ordinal: u16,
        _counters: Option<&mut LayoutPerfCounters>,
    ) -> Result<Option<&ContiguousStruct>, MaterializeError> {
        if self.check_read_slot(ordinal, 0, 0)? {
            return Ok(None);
        }
        match self.owned_slots.get(ordinal as usize).and_then(|o| o.as_ref()) {
            Some(OwnedFieldPayload::Nested(n)) => Ok(Some(n.as_ref())),
            _ => Err(MaterializeError::UninitializedSlot {
                ordinal,
                name: format!("slot_{}", ordinal),
            }),
        }
    }

    fn write_bytes_at(&mut self, byte_offset: usize, bytes: &[u8]) {
        self.storage.as_mut_slice()[byte_offset..byte_offset + bytes.len()].copy_from_slice(bytes);
    }

    fn write_slice_header(&mut self, byte_offset: usize, ptr_addr: usize, len: usize, owner_id: u64) {
        let pw = self.layout.target_abi.pointer_width;
        if pw == 4 {
            let p32 = (ptr_addr as u32).to_ne_bytes();
            let l32 = (len as u32).to_ne_bytes();
            let o64 = owner_id.to_ne_bytes();
            self.write_bytes_at(byte_offset, &p32);
            self.write_bytes_at(byte_offset + 4, &l32);
            self.write_bytes_at(byte_offset + 8, &o64);
        } else {
            let p64 = (ptr_addr as u64).to_ne_bytes();
            let l64 = (len as u64).to_ne_bytes();
            let o64 = owner_id.to_ne_bytes();
            self.write_bytes_at(byte_offset, &p64);
            self.write_bytes_at(byte_offset + 8, &l64);
            self.write_bytes_at(byte_offset + 16, &o64);
        }
    }

    /// Converts this contiguous struct into a `MaterializedStruct` (for interoperability).
    pub fn to_materialized(&self) -> MaterializedStruct {
        let mut slots = Vec::with_capacity(self.layout.fields.len());
        for field in &self.layout.fields {
            let ord = field.ordinal;
            let off = field.byte_offset;
            if self.is_null(ord) || !self.is_initialized(ord) {
                slots.push(DataValue::Null);
                continue;
            }
            let val = match field.tag {
                CanonicalTypeTag::Bool => {
                    DataValue::Bool(self.read_bool_at(ord, off, None).ok().flatten().unwrap_or(false))
                }
                CanonicalTypeTag::Int64 => {
                    DataValue::Number(ExactNumber::Int(
                        self.read_i64_at(ord, off, None).ok().flatten().unwrap_or(0),
                    ))
                }
                CanonicalTypeTag::UInt64 => {
                    DataValue::Number(ExactNumber::UInt(
                        self.read_u64_at(ord, off, None).ok().flatten().unwrap_or(0),
                    ))
                }
                CanonicalTypeTag::Float64 => {
                    DataValue::Number(ExactNumber::Float(
                        self.read_f64_at(ord, off, None).ok().flatten().unwrap_or(0.0),
                    ))
                }
                CanonicalTypeTag::Decimal128 => {
                    DataValue::Number(ExactNumber::Decimal128(crate::data::Decimal128::new(
                        self.read_decimal128_at(ord, off, None)
                            .ok()
                            .flatten()
                            .unwrap_or(0)
                            .to_ne_bytes(),
                    )))
                }
                CanonicalTypeTag::DateTimeUtc => {
                    DataValue::DateTimeUtc(crate::data::DateTimeUtc::from_millis(
                        self.read_datetime_utc_at(ord, off, None).ok().flatten().unwrap_or(0),
                    ))
                }
                CanonicalTypeTag::ObjectId => {
                    DataValue::ObjectId(crate::data::ObjectId::new(
                        self.read_object_id_at(ord, off, None).ok().flatten().unwrap_or([0u8; 12]),
                    ))
                }
                CanonicalTypeTag::StringSlice => DataValue::String(
                    self.read_str_at(ord, off, None)
                        .ok()
                        .flatten()
                        .unwrap_or("")
                        .to_string(),
                ),
                CanonicalTypeTag::BinarySlice => DataValue::Binary(
                    self.read_bytes_at(ord, off, None)
                        .ok()
                        .flatten()
                        .unwrap_or(&[])
                        .to_vec(),
                ),
                CanonicalTypeTag::NestedStruct(_) => {
                    if let Ok(Some(nested)) = self.read_nested_at(ord, None) {
                        DataValue::Document(nested.to_materialized().to_document())
                    } else {
                        DataValue::Null
                    }
                }
            };
            slots.push(val);
        }
        MaterializedStruct::new(self.layout.clone(), slots)
    }
}

impl Drop for ContiguousStruct {
    fn drop(&mut self) {
        let _ = self.cleanup_partial(None);
    }
}

/// Options for direct JSON-to-contiguous-struct decoding.
#[derive(Default)]
pub struct ContiguousDecodeOptions<'a> {
    pub cancel_check: Option<&'a dyn Fn() -> bool>,
    pub nested_layouts: Option<&'a std::collections::HashMap<u64, LogicalLayoutV1>>,
    pub require_nominal_tag: bool,
}

/// Direct JSON object decoder into a contiguous C-compatible `ContiguousStruct`.
/// Performs 0 intermediate `Document`/`Map`/`DataValue` DOM constructions.
pub fn decode_json_into_contiguous(
    layout: &LogicalLayoutV1,
    input: &[u8],
    counters: &mut LayoutPerfCounters,
    opts: &ContiguousDecodeOptions<'_>,
) -> Result<ContiguousStruct, MaterializeError> {
    let mut parser = DirectJsonDecoder {
        input,
        pos: 0,
        counters,
        opts,
    };
    let result = parser.decode_object(layout)?;
    parser.skip_ws();
    if parser.pos < parser.input.len() {
        let mut res = result;
        res.cleanup_partial(Some(parser.counters));
        return Err(MaterializeError::TrailingJunk { offset: parser.pos });
    }
    Ok(result)
}

struct DirectJsonDecoder<'a, 'b> {
    input: &'a [u8],
    pos: usize,
    counters: &'a mut LayoutPerfCounters,
    opts: &'a ContiguousDecodeOptions<'b>,
}

impl<'a, 'b> DirectJsonDecoder<'a, 'b> {
    fn check_cancelled(&mut self, target: &mut ContiguousStruct) -> Result<(), MaterializeError> {
        if let Some(cb) = self.opts.cancel_check {
            if cb() {
                let initialized_count = target.cleanup_partial(Some(self.counters));
                return Err(MaterializeError::Cancelled { initialized_count });
            }
        }
        Ok(())
    }

    fn skip_ws(&mut self) {
        while self.pos < self.input.len() {
            match self.input[self.pos] {
                b' ' | b'\t' | b'\r' | b'\n' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn next_byte(&mut self) -> Option<u8> {
        let b = self.input.get(self.pos).copied()?;
        self.pos += 1;
        Some(b)
    }

    fn parse_json_string(&mut self) -> Result<String, MaterializeError> {
        self.skip_ws();
        if self.next_byte() != Some(b'"') {
            return Err(MaterializeError::MalformedInput(format!(
                "Expected '\"' at offset {}",
                self.pos
            )));
        }
        let start = self.pos;
        let mut has_escape = false;
        while self.pos < self.input.len() {
            let b = self.input[self.pos];
            if b == b'"' {
                let slice = &self.input[start..self.pos];
                self.pos += 1;
                if !has_escape {
                    let s = std::str::from_utf8(slice).map_err(|e| {
                        MaterializeError::MalformedInput(format!("Invalid UTF-8: {}", e))
                    })?;
                    return Ok(s.to_string());
                } else {
                    return unescape_json_bytes(slice);
                }
            } else if b == b'\\' {
                has_escape = true;
                self.pos += 1;
                if self.pos >= self.input.len() {
                    return Err(MaterializeError::MalformedInput("Unterminated escape".into()));
                }
                self.pos += 1;
            } else if b < 0x20 {
                return Err(MaterializeError::MalformedInput(
                    "Unescaped control character in JSON string".into(),
                ));
            } else {
                self.pos += 1;
            }
        }
        Err(MaterializeError::MalformedInput("Unterminated JSON string".into()))
    }

    fn decode_object(&mut self, layout: &LogicalLayoutV1) -> Result<ContiguousStruct, MaterializeError> {
        let mut target = ContiguousStruct::new_uninit(layout.clone());
        self.skip_ws();
        if self.next_byte() != Some(b'{') {
            return Err(MaterializeError::MalformedInput(format!(
                "Expected '{{' at offset {}",
                self.pos
            )));
        }

        let field_count = layout.fields.len();
        let mut seen_fields = vec![false; field_count];
        let mut seen_nominal_tag = false;

        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
        } else {
            loop {
                if let Err(e) = self.check_cancelled(&mut target) {
                    return Err(e);
                }
                let key = match self.parse_json_string() {
                    Ok(k) => k,
                    Err(e) => {
                        target.cleanup_partial(Some(self.counters));
                        return Err(e);
                    }
                };
                self.skip_ws();
                if self.next_byte() != Some(b':') {
                    target.cleanup_partial(Some(self.counters));
                    return Err(MaterializeError::MalformedInput(format!(
                        "Expected ':' after key '{}' at offset {}",
                        key, self.pos
                    )));
                }
                if let Err(e) = self.check_cancelled(&mut target) {
                    return Err(e);
                }

                if key == "_type" || key == "__type" || key == "$type" {
                    if seen_nominal_tag {
                        target.cleanup_partial(Some(self.counters));
                        return Err(MaterializeError::DuplicateField(key));
                    }
                    seen_nominal_tag = true;
                    let tag_str = match self.parse_json_string() {
                        Ok(s) => s,
                        Err(_) => {
                            target.cleanup_partial(Some(self.counters));
                            return Err(MaterializeError::NominalTypeMismatch {
                                expected: layout.type_name.clone(),
                                received: "<non-string>".into(),
                            });
                        }
                    };
                    if tag_str != layout.type_name {
                        target.cleanup_partial(Some(self.counters));
                        return Err(MaterializeError::NominalTypeMismatch {
                            expected: layout.type_name.clone(),
                            received: tag_str,
                        });
                    }
                } else {
                    self.counters.decode_key_dispatches += 1;
                    let field = match layout.fields.iter().find(|f| f.name == key) {
                        Some(f) => f,
                        None => {
                            target.cleanup_partial(Some(self.counters));
                            return Err(MaterializeError::ExtraFieldInStrictLayout(key));
                        }
                    };
                    let ord_idx = field.ordinal as usize;
                    if seen_fields[ord_idx] {
                        target.cleanup_partial(Some(self.counters));
                        return Err(MaterializeError::DuplicateField(key));
                    }
                    seen_fields[ord_idx] = true;

                    if let Err(e) = self.decode_field_value(field, &mut target) {
                        target.cleanup_partial(Some(self.counters));
                        return Err(e);
                    }
                }

                self.skip_ws();
                match self.next_byte() {
                    Some(b',') => {
                        self.skip_ws();
                        continue;
                    }
                    Some(b'}') => break,
                    other => {
                        target.cleanup_partial(Some(self.counters));
                        return Err(MaterializeError::MalformedInput(format!(
                            "Expected ',' or '}}' after field, got {:?} at offset {}",
                            other, self.pos
                        )));
                    }
                }
            }
        }

        if self.opts.require_nominal_tag && !seen_nominal_tag {
            target.cleanup_partial(Some(self.counters));
            return Err(MaterializeError::NominalTypeMismatch {
                expected: layout.type_name.clone(),
                received: "<missing>".into(),
            });
        }

        for field in &layout.fields {
            let ord_idx = field.ordinal as usize;
            if !seen_fields[ord_idx] {
                if field.is_optional {
                    target.null_mask[ord_idx] = true;
                    target.init_mask[ord_idx] = true;
                    target.init_order.push(field.ordinal);
                } else {
                    target.cleanup_partial(Some(self.counters));
                    return Err(MaterializeError::MissingField {
                        ordinal: field.ordinal,
                        name: field.name.clone(),
                    });
                }
            }
        }

        Ok(target)
    }

    fn decode_field_value(
        &mut self,
        field: &LogicalFieldDesc,
        target: &mut ContiguousStruct,
    ) -> Result<(), MaterializeError> {
        self.skip_ws();
        let ord = field.ordinal;
        let ord_idx = ord as usize;

        if self.input[self.pos..].starts_with(b"null")
            && !self.input.get(self.pos + 4).is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_')
        {
            self.pos += 4;
            if field.is_nullable {
                target.null_mask[ord_idx] = true;
                target.init_mask[ord_idx] = true;
                target.init_order.push(ord);
                return Ok(());
            } else {
                return Err(MaterializeError::NullWhereNotNullable {
                    ordinal: ord,
                    name: field.name.clone(),
                });
            }
        }

        match field.tag {
            CanonicalTypeTag::Bool => {
                if self.input[self.pos..].starts_with(b"true") {
                    self.pos += 4;
                    target.write_bytes_at(field.byte_offset, &[1u8]);
                } else if self.input[self.pos..].starts_with(b"false") {
                    self.pos += 5;
                    target.write_bytes_at(field.byte_offset, &[0u8]);
                } else {
                    return Err(MaterializeError::TypeMismatch {
                        ordinal: ord,
                        name: field.name.clone(),
                        expected: field.tag,
                    });
                }
            }
            CanonicalTypeTag::Int64 | CanonicalTypeTag::DateTimeUtc => {
                let raw = self.scan_number_token(ord, &field.name, field.tag)?;
                if raw.contains('.') || raw.contains('e') || raw.contains('E') {
                    return Err(MaterializeError::TypeMismatch {
                        ordinal: ord,
                        name: field.name.clone(),
                        expected: field.tag,
                    });
                }
                let val = raw.parse::<i64>().map_err(|_| MaterializeError::WidthOverflow {
                    ordinal: ord,
                    name: field.name.clone(),
                    expected: field.tag,
                    raw: raw.to_string(),
                })?;
                target.write_bytes_at(field.byte_offset, &val.to_ne_bytes());
            }
            CanonicalTypeTag::UInt64 => {
                let raw = self.scan_number_token(ord, &field.name, field.tag)?;
                if raw.contains('.') || raw.contains('e') || raw.contains('E') {
                    return Err(MaterializeError::TypeMismatch {
                        ordinal: ord,
                        name: field.name.clone(),
                        expected: field.tag,
                    });
                }
                let val = raw.parse::<u64>().map_err(|_| MaterializeError::WidthOverflow {
                    ordinal: ord,
                    name: field.name.clone(),
                    expected: field.tag,
                    raw: raw.to_string(),
                })?;
                target.write_bytes_at(field.byte_offset, &val.to_ne_bytes());
            }
            CanonicalTypeTag::Float64 => {
                let raw = self.scan_number_token(ord, &field.name, field.tag)?;
                let val = raw.parse::<f64>().map_err(|_| MaterializeError::WidthOverflow {
                    ordinal: ord,
                    name: field.name.clone(),
                    expected: field.tag,
                    raw: raw.to_string(),
                })?;
                if !val.is_finite() {
                    return Err(MaterializeError::WidthOverflow {
                        ordinal: ord,
                        name: field.name.clone(),
                        expected: field.tag,
                        raw: raw.to_string(),
                    });
                }
                target.write_bytes_at(field.byte_offset, &val.to_ne_bytes());
            }
            CanonicalTypeTag::Decimal128 => {
                let raw = self.scan_number_token(ord, &field.name, field.tag)?;
                let val = raw.parse::<i128>().map_err(|_| MaterializeError::WidthOverflow {
                    ordinal: ord,
                    name: field.name.clone(),
                    expected: field.tag,
                    raw: raw.to_string(),
                })?;
                target.write_bytes_at(field.byte_offset, &val.to_ne_bytes());
            }
            CanonicalTypeTag::ObjectId => {
                if self.peek() != Some(b'"') {
                    return Err(MaterializeError::TypeMismatch {
                        ordinal: ord,
                        name: field.name.clone(),
                        expected: field.tag,
                    });
                }
                let hex = self.parse_json_string()?;
                if hex.len() != 24 {
                    return Err(MaterializeError::TypeMismatch {
                        ordinal: ord,
                        name: field.name.clone(),
                        expected: field.tag,
                    });
                }
                let mut oid = [0u8; 12];
                for i in 0..12 {
                    oid[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).map_err(|_| {
                        MaterializeError::TypeMismatch {
                            ordinal: ord,
                            name: field.name.clone(),
                            expected: field.tag,
                        }
                    })?;
                }
                target.write_bytes_at(field.byte_offset, &oid);
            }
            CanonicalTypeTag::StringSlice => {
                if self.peek() != Some(b'"') {
                    return Err(MaterializeError::TypeMismatch {
                        ordinal: ord,
                        name: field.name.clone(),
                        expected: field.tag,
                    });
                }
                let s = self.parse_json_string()?;
                let boxed: Box<[u8]> = s.into_bytes().into_boxed_slice();
                let ptr_addr = boxed.as_ptr() as usize;
                let len = boxed.len();
                self.counters.owned_slice_allocations += 1;
                target.write_slice_header(field.byte_offset, ptr_addr, len, ord as u64 + 1);
                target.owned_slots[ord_idx] = Some(OwnedFieldPayload::Bytes(boxed));
            }
            CanonicalTypeTag::BinarySlice => {
                if self.peek() != Some(b'"') {
                    return Err(MaterializeError::TypeMismatch {
                        ordinal: ord,
                        name: field.name.clone(),
                        expected: field.tag,
                    });
                }
                let s = self.parse_json_string()?;
                let boxed: Box<[u8]> = s.into_bytes().into_boxed_slice();
                let ptr_addr = boxed.as_ptr() as usize;
                let len = boxed.len();
                self.counters.owned_slice_allocations += 1;
                target.write_slice_header(field.byte_offset, ptr_addr, len, ord as u64 + 1);
                target.owned_slots[ord_idx] = Some(OwnedFieldPayload::Bytes(boxed));
            }
            CanonicalTypeTag::NestedStruct(nested_id) => {
                if self.peek() != Some(b'{') {
                    return Err(MaterializeError::TypeMismatch {
                        ordinal: ord,
                        name: field.name.clone(),
                        expected: field.tag,
                    });
                }
                let nested_layout = self
                    .opts
                    .nested_layouts
                    .and_then(|m| m.get(&nested_id))
                    .ok_or_else(|| {
                        MaterializeError::MalformedInput(format!(
                            "Missing nested layout {} for field '{}'",
                            nested_id, field.name
                        ))
                    })?;
                let nested_struct = self.decode_object(nested_layout)?;
                target.owned_slots[ord_idx] = Some(OwnedFieldPayload::Nested(Box::new(nested_struct)));
            }
        }

        target.init_mask[ord_idx] = true;
        target.init_order.push(ord);
        Ok(())
    }

    fn scan_number_token(
        &mut self,
        ordinal: u16,
        name: &str,
        expected: CanonicalTypeTag,
    ) -> Result<&'a str, MaterializeError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        let mut has_digit = false;
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() {
                has_digit = true;
                self.pos += 1;
            } else {
                break;
            }
        }
        if !has_digit {
            return Err(MaterializeError::TypeMismatch {
                ordinal,
                name: name.to_string(),
                expected,
            });
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            while let Some(b) = self.peek() {
                if b.is_ascii_digit() {
                    self.pos += 1;
                } else {
                    break;
                }
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            while let Some(b) = self.peek() {
                if b.is_ascii_digit() {
                    self.pos += 1;
                } else {
                    break;
                }
            }
        }
        std::str::from_utf8(&self.input[start..self.pos])
            .map_err(|e| MaterializeError::MalformedInput(format!("Invalid UTF-8 number: {}", e)))
    }
}

fn unescape_json_bytes(slice: &[u8]) -> Result<String, MaterializeError> {
    let mut out = Vec::with_capacity(slice.len());
    let mut i = 0;
    while i < slice.len() {
        if slice[i] == b'\\' {
            i += 1;
            if i >= slice.len() {
                return Err(MaterializeError::MalformedInput("Incomplete escape".into()));
            }
            match slice[i] {
                b'"' => out.push(b'"'),
                b'\\' => out.push(b'\\'),
                b'/' => out.push(b'/'),
                b'b' => out.push(0x08),
                b'f' => out.push(0x0c),
                b'n' => out.push(b'\n'),
                b'r' => out.push(b'\r'),
                b't' => out.push(b'\t'),
                b'u' => {
                    if i + 4 >= slice.len() {
                        return Err(MaterializeError::MalformedInput("Invalid \\u escape".into()));
                    }
                    let hex = std::str::from_utf8(&slice[i + 1..i + 5])
                        .map_err(|_| MaterializeError::MalformedInput("Invalid \\u hex".into()))?;
                    let code = u16::from_str_radix(hex, 16)
                        .map_err(|_| MaterializeError::MalformedInput("Invalid \\u hex".into()))?;
                    let ch = char::from_u32(code as u32)
                        .ok_or_else(|| MaterializeError::MalformedInput("Invalid unicode scalar".into()))?;
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    i += 4;
                }
                other => {
                    return Err(MaterializeError::MalformedInput(format!(
                        "Unsupported JSON escape '\\{}'",
                        other as char
                    )));
                }
            }
            i += 1;
        } else {
            out.push(slice[i]);
            i += 1;
        }
    }
    String::from_utf8(out)
        .map_err(|e| MaterializeError::MalformedInput(format!("Invalid UTF-8 after unescape: {}", e)))
}
