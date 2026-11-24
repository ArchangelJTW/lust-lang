//! Gate-SCHEMA-09 Struct Binding Plan, Wire Aliases, Mutate-Then-Encode, and
//! ModelContract / DbWritePlan / Recording Adapter (RFC Sections 6.7, 6.8, 6.8.1, 6.8.2).
//!
//! Explicitly scoped as a pure contract & fake recording adapter for parameter ordering,
//! stale-plan rejection, generated-ID insert, sparse-patch NULL semantics, and lifecycle
//! hooks — without claiming live database integration.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::hash::{Hash, Hasher};

use crate::data::layout::{CanonicalTypeTag, LogicalLayoutV1};
use crate::data::materialize::{
    ContiguousDecodeOptions, ContiguousStruct, LayoutPerfCounters, MaterializeError,
    decode_json_into_contiguous,
};
use crate::data::{DataValue, Document, ExactNumber};

/// Wire-key alias mapping (`wire_key -> canonical_lust_field_name`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WireAliasMap {
    pub wire_to_field: BTreeMap<String, String>,
    pub field_to_wire: BTreeMap<String, String>,
}

impl WireAliasMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a wire alias for a canonical Lust field name, rejecting alias collisions.
    pub fn add_alias(
        &mut self,
        field_name: impl Into<String>,
        wire_key: impl Into<String>,
    ) -> Result<(), ModelContractError> {
        let field = field_name.into();
        let wire = wire_key.into();
        if let Some(existing) = self.wire_to_field.get(&wire) {
            if existing != &field {
                return Err(ModelContractError::AliasCollision {
                    wire_key: wire,
                    first_field: existing.clone(),
                    second_field: field,
                });
            }
        }
        self.wire_to_field.insert(wire.clone(), field.clone());
        self.field_to_wire.insert(field, wire);
        Ok(())
    }

    pub fn canonical_field_for<'a>(&'a self, wire_key: &'a str) -> &'a str {
        self.wire_to_field
            .get(wire_key)
            .map(String::as_str)
            .unwrap_or(wire_key)
    }

    pub fn wire_key_for<'a>(&'a self, field_name: &'a str) -> &'a str {
        self.field_to_wire
            .get(field_name)
            .map(String::as_str)
            .unwrap_or(field_name)
    }
}

/// Origin of a `StructBindingPlan` (RFC Gate-SCHEMA-09: derived / overridden / import-bound).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BindingOrigin {
    Derived,
    Overridden,
    ImportBound,
}

/// Compiled struct binding plan tying nominal layout identity, schema fingerprint,
/// wire aliases, and record-level pure constraints together.
#[derive(Debug, Clone)]
pub struct StructBindingPlan {
    pub origin: BindingOrigin,
    pub layout: LogicalLayoutV1,
    pub schema_fingerprint: u64,
    pub layout_fingerprint: u64,
    pub aliases: WireAliasMap,
    pub non_negative_int_fields: Vec<u16>,
}

impl StructBindingPlan {
    pub fn new(
        origin: BindingOrigin,
        layout: LogicalLayoutV1,
        schema_fingerprint: u64,
        aliases: WireAliasMap,
    ) -> Result<Self, ModelContractError> {
        // Reject alias collisions against declared field names.
        for (wire, target) in &aliases.wire_to_field {
            if layout.field_by_name(target).is_none() {
                return Err(ModelContractError::UnknownField(target.clone()));
            }
            if wire != target && layout.field_by_name(wire).is_some() {
                return Err(ModelContractError::AliasCollision {
                    wire_key: wire.clone(),
                    first_field: wire.clone(),
                    second_field: target.clone(),
                });
            }
        }
        let layout_fingerprint = compute_layout_fingerprint(&layout, schema_fingerprint, &aliases);
        Ok(Self {
            origin,
            layout,
            schema_fingerprint,
            layout_fingerprint,
            aliases,
            non_negative_int_fields: Vec::new(),
        })
    }

    pub fn require_non_negative_int(mut self, ordinal: u16) -> Self {
        self.non_negative_int_fields.push(ordinal);
        self
    }

    /// Validates record-level pure constraints on a decoded `ContiguousStruct`.
    pub fn validate_record_constraints(
        &self,
        cs: &ContiguousStruct,
    ) -> Result<(), ModelContractError> {
        if cs.layout.type_id != self.layout.type_id
            || cs.layout.type_name != self.layout.type_name
            || cs.layout.size_bytes != self.layout.size_bytes
        {
            return Err(ModelContractError::LayoutMismatch {
                expected: self.layout.type_name.clone(),
                received: cs.layout.type_name.clone(),
            });
        }
        for &ord in &self.non_negative_int_fields {
            let field = self
                .layout
                .field_by_ordinal(ord)
                .ok_or_else(|| ModelContractError::UnknownField(format!("ordinal_{ord}")))?;
            if let Ok(Some(v)) = cs.read_i64_at(ord, field.byte_offset, None) {
                if v < 0 {
                    return Err(ModelContractError::ConstraintViolation {
                        field: field.name.clone(),
                        reason: format!("expected non-negative i64, got {v}"),
                    });
                }
            }
        }
        Ok(())
    }

    /// Revalidates a mutated `MaterializedStruct` (or document) before reverse encoding to JSON.
    pub fn encode_validated_json(
        &self,
        doc: &Document,
    ) -> Result<String, ModelContractError> {
        self.revalidate_document(doc)?;
        // Emit fields in declared layout order using canonical wire names (aliases applied).
        let mut out = Document::with_capacity(self.layout.fields.len());
        for field in &self.layout.fields {
            let wire_key = self.aliases.wire_key_for(&field.name);
            let val = doc.get(&field.name).cloned().unwrap_or(DataValue::Null);
            out.try_insert(wire_key.to_string(), val)
                .map_err(|e| ModelContractError::SerializationError(e.to_string()))?;
        }
        crate::data::json::to_string(&DataValue::Document(out))
            .map_err(|e| ModelContractError::SerializationError(format!("{e:?}")))
    }

    /// Revalidates a mutated document before reverse encoding to BSON bytes.
    pub fn encode_validated_bson(
        &self,
        doc: &Document,
    ) -> Result<Vec<u8>, ModelContractError> {
        self.revalidate_document(doc)?;
        let mut out = Document::with_capacity(self.layout.fields.len());
        for field in &self.layout.fields {
            let wire_key = self.aliases.wire_key_for(&field.name);
            let val = doc.get(&field.name).cloned().unwrap_or(DataValue::Null);
            out.try_insert(wire_key.to_string(), val)
                .map_err(|e| ModelContractError::SerializationError(e.to_string()))?;
        }
        Ok(crate::data::bson::to_vec(&out))
    }

    fn revalidate_document(&self, doc: &Document) -> Result<(), ModelContractError> {
        for field in &self.layout.fields {
            match doc.get(&field.name) {
                None if !field.is_optional => {
                    return Err(ModelContractError::ConstraintViolation {
                        field: field.name.clone(),
                        reason: "missing required field".to_string(),
                    });
                }
                Some(DataValue::Null) if !field.is_nullable => {
                    return Err(ModelContractError::ConstraintViolation {
                        field: field.name.clone(),
                        reason: "null not permitted for non-nullable field".to_string(),
                    });
                }
                Some(DataValue::Number(ExactNumber::Int(v)))
                    if self.non_negative_int_fields.contains(&field.ordinal) && *v < 0 =>
                {
                    return Err(ModelContractError::ConstraintViolation {
                        field: field.name.clone(),
                        reason: format!("expected non-negative i64, got {v}"),
                    });
                }
                _ => {}
            }
        }
        Ok(())
    }
}

fn compute_layout_fingerprint(
    layout: &LogicalLayoutV1,
    schema_fingerprint: u64,
    aliases: &WireAliasMap,
) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    layout.type_id.hash(&mut h);
    layout.type_name.hash(&mut h);
    layout.size_bytes.hash(&mut h);
    layout.align_bytes.hash(&mut h);
    schema_fingerprint.hash(&mut h);
    for f in &layout.fields {
        f.ordinal.hash(&mut h);
        f.name.hash(&mut h);
        f.tag.hash(&mut h);
        f.byte_offset.hash(&mut h);
        f.byte_size.hash(&mut h);
        f.is_nullable.hash(&mut h);
        f.is_optional.hash(&mut h);
    }
    for (k, v) in &aliases.wire_to_field {
        k.hash(&mut h);
        v.hash(&mut h);
    }
    h.finish()
}

/// Decodes JSON bytes directly into a `ContiguousStruct` using `plan`, resolving wire aliases
/// during key scan (without constructing any intermediate DOM/Map) and rejecting duplicate
/// aliases (e.g., both canonical field name and alias present in the same JSON object).
pub fn decode_json_with_plan(
    plan: &StructBindingPlan,
    input: &[u8],
    counters: &mut LayoutPerfCounters,
    opts: &ContiguousDecodeOptions<'_>,
) -> Result<ContiguousStruct, MaterializeError> {
    if plan.aliases.wire_to_field.is_empty() {
        return decode_json_into_contiguous(&plan.layout, input, counters, opts);
    }

    // Rewrite top-level wire alias keys in a streaming byte pass (zero DOM / DataValue objects)
    // so the canonical direct contiguous decoder checks duplicates, types, overflows, and
    // trailing junk with identical cleanup guarantees.
    let rewritten = rewrite_top_level_wire_aliases(input, &plan.aliases)?;
    decode_json_into_contiguous(&plan.layout, &rewritten, counters, opts)
}

fn rewrite_top_level_wire_aliases(
    input: &[u8],
    aliases: &WireAliasMap,
) -> Result<Vec<u8>, MaterializeError> {
    let mut out = Vec::with_capacity(input.len() + 16);
    let mut pos = 0usize;
    let len = input.len();
    let mut depth = 0usize;
    let mut expecting_key = false;

    while pos < len {
        let b = input[pos];
        match b {
            b' ' | b'\t' | b'\r' | b'\n' => {
                out.push(b);
                pos += 1;
            }
            b'{' => {
                depth += 1;
                expecting_key = depth == 1;
                out.push(b);
                pos += 1;
            }
            b'}' => {
                depth = depth.saturating_sub(1);
                expecting_key = false;
                out.push(b);
                pos += 1;
            }
            b',' if depth == 1 => {
                expecting_key = true;
                out.push(b);
                pos += 1;
            }
            b':' if depth == 1 => {
                expecting_key = false;
                out.push(b);
                pos += 1;
            }
            b'"' => {
                let start = pos;
                pos += 1;
                while pos < len {
                    if input[pos] == b'\\' {
                        pos += 2;
                    } else if input[pos] == b'"' {
                        pos += 1;
                        break;
                    } else {
                        pos += 1;
                    }
                }
                let raw_token = &input[start..pos.min(len)];
                if depth == 1 && expecting_key && raw_token.len() >= 2 && raw_token[raw_token.len() - 1] == b'"' {
                    let key_bytes = &raw_token[1..raw_token.len() - 1];
                    if let Ok(key_str) = std::str::from_utf8(key_bytes) {
                        let canonical = aliases.canonical_field_for(key_str);
                        out.push(b'"');
                        out.extend_from_slice(canonical.as_bytes());
                        out.push(b'"');
                        continue;
                    }
                }
                out.extend_from_slice(raw_token);
            }
            _ => {
                out.push(b);
                pos += 1;
            }
        }
    }
    Ok(out)
}

/// Generated-field policy (RFC Section 6.8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GeneratedIdPolicy {
    None,
    DatabaseIdentity,
    DatabaseDefault,
}

/// Write operation kind for `DbWritePlan`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DbOperation {
    Insert,
    UpdateFull,
    Patch,
}

/// Sparse patch field state distinguishing Missing vs Null vs concrete value.
#[derive(Debug, Clone, PartialEq)]
pub enum PatchField {
    Missing,
    Null,
    Value(DataValue),
}

/// Separate typed InsertInput omitting database-generated fields (never an uninitialized `T`).
#[derive(Debug, Clone)]
pub struct InsertInput {
    pub values: BTreeMap<String, DataValue>,
}

impl InsertInput {
    pub fn new() -> Self {
        Self {
            values: BTreeMap::new(),
        }
    }

    pub fn with_field(mut self, name: impl Into<String>, value: DataValue) -> Self {
        self.values.insert(name.into(), value);
        self
    }
}

impl Default for InsertInput {
    fn default() -> Self {
        Self::new()
    }
}

/// Sparse PatchInput separating Missing (do not write) from Null (bind SQL NULL).
#[derive(Debug, Clone)]
pub struct PatchInput {
    pub pk_value: DataValue,
    pub fields: BTreeMap<String, PatchField>,
}

impl PatchInput {
    pub fn new(pk_value: DataValue) -> Self {
        Self {
            pk_value,
            fields: BTreeMap::new(),
        }
    }

    pub fn with_patch(mut self, name: impl Into<String>, field: PatchField) -> Self {
        self.fields.insert(name.into(), field);
        self
    }
}

/// Errors produced by `ModelContract`, `DbWritePlan`, or lifecycle hooks before any DB I/O.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelContractError {
    AliasCollision {
        wire_key: String,
        first_field: String,
        second_field: String,
    },
    UnknownField(String),
    LayoutMismatch {
        expected: String,
        received: String,
    },
    ConstraintViolation {
        field: String,
        reason: String,
    },
    StaleWritePlan {
        expected_fingerprint: u64,
        plan_fingerprint: u64,
    },
    NullOnNonNullableColumn(String),
    MissingRequiredWriteField(String),
    ReadOnlyFieldWriteRejected(String),
    HookFailed(String),
    HookCancelled(String),
    SerializationError(String),
}

impl fmt::Display for ModelContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ModelContractError {}

/// Compiled write projection plan pinned to a `ModelContract` fingerprint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DbWritePlan {
    pub operation: DbOperation,
    pub table: String,
    pub ordered_fields: Vec<String>,
    pub ordered_columns: Vec<String>,
    pub db_types: Vec<CanonicalTypeTag>,
    pub placeholders: Vec<String>,
    pub model_fingerprint: u64,
}

/// Explicit FIFO lifecycle hooks for `ModelContract` (RFC Section 6.8.2).
pub struct ModelLifecycleHooks {
    pub validate_record: Vec<Box<dyn Fn(&Document) -> Result<(), String>>>,
    pub before_save: Vec<Box<dyn Fn(&mut Document) -> Result<bool, String>>>,
    pub after_load: Vec<Box<dyn Fn(&mut Document) -> Result<(), String>>>,
}

impl Default for ModelLifecycleHooks {
    fn default() -> Self {
        Self {
            validate_record: Vec::new(),
            before_save: Vec::new(),
            after_load: Vec::new(),
        }
    }
}

/// Immutable Data Mapper `ModelContract` (RFC Section 6.8).
pub struct ModelContract {
    pub table: String,
    pub primary_key: Vec<String>,
    pub generated: HashMap<String, GeneratedIdPolicy>,
    pub read_only: Vec<String>,
    pub columns: HashMap<String, String>,
    pub binding: StructBindingPlan,
    pub model_fingerprint: u64,
    pub hooks: ModelLifecycleHooks,
}

impl ModelContract {
    pub fn new(
        table: impl Into<String>,
        primary_key: Vec<String>,
        generated: HashMap<String, GeneratedIdPolicy>,
        read_only: Vec<String>,
        columns: HashMap<String, String>,
        binding: StructBindingPlan,
    ) -> Self {
        let table_s = table.into();
        let mut h = std::collections::hash_map::DefaultHasher::new();
        table_s.hash(&mut h);
        primary_key.hash(&mut h);
        binding.layout_fingerprint.hash(&mut h);
        let mut gen_sorted: Vec<_> = generated.iter().collect();
        gen_sorted.sort_by_key(|(k, _)| *k);
        for (k, v) in gen_sorted {
            k.hash(&mut h);
            v.hash(&mut h);
        }
        let mut cols_sorted: Vec<_> = columns.iter().collect();
        cols_sorted.sort_by_key(|(k, _)| *k);
        for (k, v) in cols_sorted {
            k.hash(&mut h);
            v.hash(&mut h);
        }
        read_only.hash(&mut h);
        let model_fingerprint = h.finish();

        Self {
            table: table_s,
            primary_key,
            generated,
            read_only,
            columns,
            binding,
            model_fingerprint,
            hooks: ModelLifecycleHooks::default(),
        }
    }

    pub fn column_name_for<'a>(&'a self, field_name: &'a str) -> &'a str {
        self.columns
            .get(field_name)
            .map(String::as_str)
            .unwrap_or(field_name)
    }

    /// Compiles an INSERT write plan excluding `DatabaseIdentity` / `read_only` columns.
    pub fn compile_insert_plan(&self) -> DbWritePlan {
        let mut ordered_fields = Vec::new();
        let mut ordered_columns = Vec::new();
        let mut db_types = Vec::new();
        let mut placeholders = Vec::new();

        for field in &self.binding.layout.fields {
            if matches!(
                self.generated.get(&field.name),
                Some(GeneratedIdPolicy::DatabaseIdentity)
            ) || self.read_only.contains(&field.name)
            {
                continue;
            }
            ordered_fields.push(field.name.clone());
            ordered_columns.push(self.column_name_for(&field.name).to_string());
            db_types.push(field.tag);
            placeholders.push(format!("${}", placeholders.len() + 1));
        }

        DbWritePlan {
            operation: DbOperation::Insert,
            table: self.table.clone(),
            ordered_fields,
            ordered_columns,
            db_types,
            placeholders,
            model_fingerprint: self.model_fingerprint,
        }
    }

    /// Compiles a sparse PATCH write plan containing ONLY present (non-Missing) columns.
    pub fn compile_patch_plan(&self, patch: &PatchInput) -> Result<DbWritePlan, ModelContractError> {
        let mut ordered_fields = Vec::new();
        let mut ordered_columns = Vec::new();
        let mut db_types = Vec::new();
        let mut placeholders = Vec::new();

        for field in &self.binding.layout.fields {
            let Some(patch_state) = patch.fields.get(&field.name) else {
                continue;
            };
            match patch_state {
                PatchField::Missing => continue,
                PatchField::Null => {
                    if self.read_only.contains(&field.name) {
                        return Err(ModelContractError::ReadOnlyFieldWriteRejected(
                            field.name.clone(),
                        ));
                    }
                    if !field.is_nullable {
                        return Err(ModelContractError::NullOnNonNullableColumn(
                            field.name.clone(),
                        ));
                    }
                }
                PatchField::Value(_) => {
                    if self.read_only.contains(&field.name) {
                        return Err(ModelContractError::ReadOnlyFieldWriteRejected(
                            field.name.clone(),
                        ));
                    }
                }
            }
            ordered_fields.push(field.name.clone());
            ordered_columns.push(self.column_name_for(&field.name).to_string());
            db_types.push(field.tag);
            placeholders.push(format!("${}", placeholders.len() + 1));
        }

        // Append PK predicate placeholder at the end in deterministic order.
        for pk in &self.primary_key {
            let pk_field = self
                .binding
                .layout
                .field_by_name(pk)
                .ok_or_else(|| ModelContractError::UnknownField(pk.clone()))?;
            ordered_fields.push(pk.clone());
            ordered_columns.push(self.column_name_for(pk).to_string());
            db_types.push(pk_field.tag);
            placeholders.push(format!("${}", placeholders.len() + 1));
        }

        Ok(DbWritePlan {
            operation: DbOperation::Patch,
            table: self.table.clone(),
            ordered_fields,
            ordered_columns,
            db_types,
            placeholders,
            model_fingerprint: self.model_fingerprint,
        })
    }

    fn verify_plan_fresh(&self, plan: &DbWritePlan) -> Result<(), ModelContractError> {
        if plan.model_fingerprint != self.model_fingerprint {
            return Err(ModelContractError::StaleWritePlan {
                expected_fingerprint: self.model_fingerprint,
                plan_fingerprint: plan.model_fingerprint,
            });
        }
        Ok(())
    }

    /// Produces ordered DB parameters for an `InsertInput` following `plan`, running pure
    /// `validate_record` and `before_save` hooks before returning parameters (zero I/O on error).
    pub fn prepare_insert_params(
        &self,
        input: &InsertInput,
        plan: &DbWritePlan,
        hook_trace: &mut Vec<String>,
        swap_params_mutant: bool,
    ) -> Result<Vec<(String, DataValue)>, ModelContractError> {
        self.verify_plan_fresh(plan)?;

        let mut staging = Document::new();
        for f_name in &plan.ordered_fields {
            let field_desc = self
                .binding
                .layout
                .field_by_name(f_name)
                .ok_or_else(|| ModelContractError::UnknownField(f_name.clone()))?;
            match input.values.get(f_name) {
                Some(DataValue::Null) if !field_desc.is_nullable => {
                    return Err(ModelContractError::NullOnNonNullableColumn(f_name.clone()));
                }
                Some(v) => {
                    let _ = staging.try_insert(f_name.clone(), v.clone());
                }
                None if field_desc.is_optional => {
                    let _ = staging.try_insert(f_name.clone(), DataValue::Null);
                }
                None => {
                    return Err(ModelContractError::MissingRequiredWriteField(f_name.clone()));
                }
            }
        }

        for (i, v) in self.hooks.validate_record.iter().enumerate() {
            hook_trace.push(format!("validate_record[{i}]"));
            v(&staging).map_err(ModelContractError::HookFailed)?;
        }
        for (i, h) in self.hooks.before_save.iter().enumerate() {
            hook_trace.push(format!("before_save[{i}]"));
            let cont = h(&mut staging).map_err(ModelContractError::HookFailed)?;
            if !cont {
                return Err(ModelContractError::HookCancelled(format!("before_save[{i}]")));
            }
        }
        // Revalidate after hooks before emitting params.
        for (i, v) in self.hooks.validate_record.iter().enumerate() {
            hook_trace.push(format!("revalidate_record[{i}]"));
            v(&staging).map_err(ModelContractError::HookFailed)?;
        }

        let mut params = Vec::with_capacity(plan.ordered_fields.len());
        for (f_name, col_name) in plan.ordered_fields.iter().zip(plan.ordered_columns.iter()) {
            let val = staging.get(f_name).cloned().unwrap_or(DataValue::Null);
            params.push((col_name.clone(), val));
        }

        if swap_params_mutant && params.len() >= 2 {
            let v0 = params[0].1.clone();
            let v1 = params[1].1.clone();
            params[0].1 = v1;
            params[1].1 = v0;
        }
        Ok(params)
    }

    /// Produces ordered DB parameters for a sparse `PatchInput` following `plan`.
    pub fn prepare_patch_params(
        &self,
        patch: &PatchInput,
        plan: &DbWritePlan,
    ) -> Result<Vec<(String, DataValue)>, ModelContractError> {
        self.verify_plan_fresh(plan)?;
        let mut params = Vec::with_capacity(plan.ordered_fields.len());
        let pk_start = plan.ordered_fields.len().saturating_sub(self.primary_key.len());

        for (idx, (f_name, col_name)) in plan
            .ordered_fields
            .iter()
            .zip(plan.ordered_columns.iter())
            .enumerate()
        {
            if idx >= pk_start {
                params.push((col_name.clone(), patch.pk_value.clone()));
                continue;
            }
            match patch.fields.get(f_name) {
                Some(PatchField::Value(v)) => params.push((col_name.clone(), v.clone())),
                Some(PatchField::Null) => params.push((col_name.clone(), DataValue::Null)),
                _ => {
                    return Err(ModelContractError::MissingRequiredWriteField(f_name.clone()));
                }
            }
        }
        Ok(params)
    }

    /// Runs `after_load` hooks on a decoded persisted row in deterministic FIFO order.
    pub fn apply_after_load(
        &self,
        mut doc: Document,
        hook_trace: &mut Vec<String>,
    ) -> Result<Document, ModelContractError> {
        for (i, h) in self.hooks.after_load.iter().enumerate() {
            hook_trace.push(format!("after_load[{i}]"));
            h(&mut doc).map_err(ModelContractError::HookFailed)?;
        }
        for (i, v) in self.hooks.validate_record.iter().enumerate() {
            hook_trace.push(format!("post_load_validate[{i}]"));
            v(&doc).map_err(ModelContractError::HookFailed)?;
        }
        Ok(doc)
    }
}

/// Recorded write call captured by `RecordingDbAdapter`.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordedDbWrite {
    pub operation: DbOperation,
    pub table: String,
    pub columns: Vec<String>,
    pub placeholders: Vec<String>,
    pub params: Vec<(String, DataValue)>,
}

/// In-memory fake recording adapter for Gate-SCHEMA-09 verification.
/// Verifies ordered parameters, generated-ID RETURNING decode, and zero I/O on pre-write
/// failures — explicitly without claiming live database driver integration.
#[derive(Debug, Default)]
pub struct RecordingDbAdapter {
    pub io_calls: usize,
    pub next_generated_id: i64,
    pub writes: Vec<RecordedDbWrite>,
}

impl RecordingDbAdapter {
    pub fn new(start_id: i64) -> Self {
        Self {
            io_calls: 0,
            next_generated_id: start_id,
            writes: Vec::new(),
        }
    }

    /// Executes an insert through `contract` + `plan`, returning a full persisted `Document`
    /// with the database-generated ID populated via simulated `RETURNING`.
    pub fn execute_insert(
        &mut self,
        contract: &ModelContract,
        input: &InsertInput,
        plan: &DbWritePlan,
        hook_trace: &mut Vec<String>,
        swap_params_mutant: bool,
    ) -> Result<Document, ModelContractError> {
        let params =
            contract.prepare_insert_params(input, plan, hook_trace, swap_params_mutant)?;
        // Only now does I/O occur.
        self.io_calls += 1;
        let gen_id = self.next_generated_id;
        self.next_generated_id += 1;
        self.writes.push(RecordedDbWrite {
            operation: plan.operation,
            table: plan.table.clone(),
            columns: plan.ordered_columns.clone(),
            placeholders: plan.placeholders.clone(),
            params: params.clone(),
        });

        // Build full persisted row as if decoded from DB RETURNING.
        let mut persisted = Document::new();
        for pk in &contract.primary_key {
            let _ = persisted.try_insert(pk.clone(), DataValue::Number(ExactNumber::Int(gen_id)));
        }
        for (field_name, (_, val)) in plan.ordered_fields.iter().zip(params.into_iter()) {
            let _ = persisted.try_insert(field_name.clone(), val);
        }
        contract.apply_after_load(persisted, hook_trace)
    }

    /// Executes a sparse patch through `contract` + `plan`, recording exact ordered params.
    pub fn execute_patch(
        &mut self,
        contract: &ModelContract,
        patch: &PatchInput,
        plan: &DbWritePlan,
    ) -> Result<(), ModelContractError> {
        let params = contract.prepare_patch_params(patch, plan)?;
        self.io_calls += 1;
        self.writes.push(RecordedDbWrite {
            operation: plan.operation,
            table: plan.table.clone(),
            columns: plan.ordered_columns.clone(),
            placeholders: plan.placeholders.clone(),
            params,
        });
        Ok(())
    }
}
