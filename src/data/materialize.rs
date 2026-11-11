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
