//! JSON Schema 2020-12 / OpenAPI 3.1 interoperability layer for Lust (DS-90, DS-92).
//!
//! Handles dialect-aware export and import of schemas with strict fail-closed portability,
//! bounded reference resolution, cycle detection, and explicit projection support.

use crate::data::{DataValue, Document, ExactNumber};
use crate::schema::builder::Schema;
use crate::schema::ir::{ObjectFieldIR, SchemaIR};
use crate::schema::pattern::{export_regex_pattern, import_regex_pattern, PatternMode};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Target JSON Schema dialect for export/import.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JsonSchemaDialect {
    #[default]
    Draft2020_12,
    OpenApi3_1,
}

impl JsonSchemaDialect {
    pub fn schema_uri(&self) -> Option<&'static str> {
        match self {
            JsonSchemaDialect::Draft2020_12 => Some("https://json-schema.org/draft/2020-12/schema"),
            JsonSchemaDialect::OpenApi3_1 => None,
        }
    }
}

/// Errors occurring during schema export to JSON Schema / OpenAPI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaExportError {
    NonPortableConstraint(String),
    NonPortableTransform(String),
    NonPortableFormat(String),
    UnsupportedRepresentation(String),
    NonPortablePattern(String),
    UnsupportedKeyword(String),
}

impl fmt::Display for SchemaExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchemaExportError::NonPortableConstraint(msg) => write!(f, "Non-portable constraint: {}", msg),
            SchemaExportError::NonPortableTransform(msg) => write!(f, "Non-portable transform: {}", msg),
            SchemaExportError::NonPortableFormat(msg) => write!(f, "Non-portable format: {}", msg),
            SchemaExportError::UnsupportedRepresentation(msg) => write!(f, "Unsupported representation: {}", msg),
            SchemaExportError::NonPortablePattern(msg) => write!(f, "Non-portable pattern: {}", msg),
            SchemaExportError::UnsupportedKeyword(msg) => write!(f, "Unsupported keyword: {}", msg),
        }
    }
}

impl std::error::Error for SchemaExportError {}

/// Errors occurring during schema import from JSON Schema / OpenAPI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaImportError {
    UnsupportedRemoteRef(String),
    UnresolvedRef(String),
    CyclicRef(String),
    BudgetExceeded(String),
    UnsupportedKeyword(String),
    InvalidSchema(String),
    NonPortablePattern(String),
}

impl fmt::Display for SchemaImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchemaImportError::UnsupportedRemoteRef(r) => write!(f, "Unsupported remote $ref rejected: {}", r),
            SchemaImportError::UnresolvedRef(r) => write!(f, "Unresolved local $ref: {}", r),
            SchemaImportError::CyclicRef(r) => write!(f, "Cyclic $ref detected: {}", r),
            SchemaImportError::BudgetExceeded(msg) => write!(f, "Schema import budget exceeded: {}", msg),
            SchemaImportError::UnsupportedKeyword(k) => write!(f, "Unsupported JSON Schema keyword: {}", k),
            SchemaImportError::InvalidSchema(msg) => write!(f, "Invalid JSON Schema document: {}", msg),
            SchemaImportError::NonPortablePattern(msg) => write!(f, "Non-portable pattern: {}", msg),
        }
    }
}

impl std::error::Error for SchemaImportError {}

/// Options controlling JSON Schema export.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ExportOptions {
    pub dialect: JsonSchemaDialect,
}

/// Options controlling JSON Schema import.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportOptions {
    pub max_depth: usize,
    pub max_nodes: usize,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self {
            max_depth: 64,
            max_nodes: 1000,
        }
    }
}

/// Mode of explicit schema projection (DS-92).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionMode {
    Input,
    Output,
}

/// Record of an omitted constraint during schema projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OmittedConstraint {
    pub path: Vec<String>,
    pub constraint: String,
    pub reason: String,
}

/// Result of an explicit schema projection.
#[derive(Debug, Clone, PartialEq)]
pub struct SchemaProjection {
    pub document: Document,
    pub omissions: Vec<OmittedConstraint>,
    pub projection: ProjectionMode,
    pub equivalent: bool,
}

/// Portable JSON Schema wrapper.
#[derive(Debug, Clone, PartialEq)]
pub struct JsonSchemaPortable {
    pub document: Document,
    pub dialect: JsonSchemaDialect,
}

impl JsonSchemaPortable {
    pub fn to_json_schema(&self) -> Document {
        self.document.clone()
    }
}

/// Export a schema to a JSON Schema `Document` strictly failing closed on non-portable constructs.
pub fn to_json_schema(schema: &Schema, options: Option<&ExportOptions>) -> Result<Document, SchemaExportError> {
    let dialect = options.map(|o| o.dialect).unwrap_or_default();
    let mut exporter = SchemaExporter {
        dialect,
        path: Vec::new(),
        allow_omissions: false,
        omissions: Vec::new(),
    };
    let mut doc = exporter.export_ir(&schema.ir)?;
    if let Some(uri) = dialect.schema_uri() {
        let _ = doc.try_insert("$schema", DataValue::String(uri.to_string()));
    }
    Ok(doc)
}

/// Export a schema to a validated portable JSON Schema document.
pub fn as_json_schema_portable(
    schema: &Schema,
    options: Option<&ExportOptions>,
) -> Result<JsonSchemaPortable, SchemaExportError> {
    let dialect = options.map(|o| o.dialect).unwrap_or_default();
    let doc = to_json_schema(schema, options)?;
    Ok(JsonSchemaPortable {
        document: doc,
        dialect,
    })
}

/// Export an explicit JSON Schema projection (Input or Output), recording all omissions (DS-92).
pub fn export_json_schema_projection(
    schema: &Schema,
    projection: ProjectionMode,
) -> Result<SchemaProjection, SchemaExportError> {
    if projection == ProjectionMode::Output {
        // Output projection fails if schema contains uninvertible transforms
        if contains_transform(&schema.ir) {
            return Err(SchemaExportError::NonPortableTransform(
                "Output projection requires explicit output schema; pure transform stage is non-invertible".to_string(),
            ));
        }
    }

    let mut exporter = SchemaExporter {
        dialect: JsonSchemaDialect::Draft2020_12,
        path: Vec::new(),
        allow_omissions: true,
        omissions: Vec::new(),
    };

    let mut doc = exporter.export_ir(&schema.ir)?;
    let _ = doc.try_insert(
        "$schema",
        DataValue::String("https://json-schema.org/draft/2020-12/schema".to_string()),
    );

    let equivalent = exporter.omissions.is_empty();
    Ok(SchemaProjection {
        document: doc,
        omissions: exporter.omissions,
        projection,
        equivalent,
    })
}

fn contains_transform(ir: &SchemaIR) -> bool {
    match ir {
        SchemaIR::Transform { .. } => true,
        SchemaIR::Refine { inner, .. } => contains_transform(inner),
        SchemaIR::Array { item, .. } => contains_transform(item),
        SchemaIR::Object { fields, .. } => fields.iter().any(|f| contains_transform(&f.schema)),
        SchemaIR::Record { key, value } => contains_transform(key) || contains_transform(value),
        SchemaIR::Union(vars) | SchemaIR::DiscriminatedUnion { variants: vars, .. } => {
            vars.iter().any(contains_transform)
        }
        SchemaIR::Optional(inner) | SchemaIR::Nullable(inner) | SchemaIR::Pattern { inner, .. } => {
            contains_transform(inner)
        }
        _ => false,
    }
}

struct SchemaExporter {
    dialect: JsonSchemaDialect,
    path: Vec<String>,
    allow_omissions: bool,
    omissions: Vec<OmittedConstraint>,
}

impl SchemaExporter {
    fn export_ir(&mut self, ir: &SchemaIR) -> Result<Document, SchemaExportError> {
        let mut doc = Document::new();
        match ir {
            SchemaIR::Any => {
                // Empty object represents any value in JSON Schema
            }
            SchemaIR::Bool => {
                doc.try_insert("type", DataValue::String("boolean".to_string())).unwrap();
            }
            SchemaIR::Int { min, max } => {
                doc.try_insert("type", DataValue::String("integer".to_string())).unwrap();
                if let Some(m) = min {
                    doc.try_insert("minimum", DataValue::Number(ExactNumber::Int(*m))).unwrap();
                }
                if let Some(m) = max {
                    doc.try_insert("maximum", DataValue::Number(ExactNumber::Int(*m))).unwrap();
                }
            }
            SchemaIR::Float { min, max } => {
                doc.try_insert("type", DataValue::String("number".to_string())).unwrap();
                if let Some(m) = min {
                    doc.try_insert("minimum", DataValue::Number(ExactNumber::Float(*m))).unwrap();
                }
                if let Some(m) = max {
                    doc.try_insert("maximum", DataValue::Number(ExactNumber::Float(*m))).unwrap();
                }
            }
            SchemaIR::String { min_len, max_len, regex } => {
                doc.try_insert("type", DataValue::String("string".to_string())).unwrap();
                if let Some(min) = min_len {
                    doc.try_insert("minLength", DataValue::Number(ExactNumber::Int(*min as i64))).unwrap();
                }
                if let Some(max) = max_len {
                    doc.try_insert("maxLength", DataValue::Number(ExactNumber::Int(*max as i64))).unwrap();
                }
                if let Some(pat) = regex {
                    let exported_pat = export_regex_pattern(pat)?;
                    doc.try_insert("pattern", DataValue::String(exported_pat)).unwrap();
                }
            }
            SchemaIR::Pattern { inner, pattern, mode } => {
                doc = self.export_ir(inner)?;
                let exported_pat = match mode {
                    PatternMode::Search => pattern.clone(),
                    PatternMode::FullMatch => export_regex_pattern(pattern)?,
                };
                doc.try_insert("pattern", DataValue::String(exported_pat)).unwrap();
            }
            SchemaIR::Binary { min_len, max_len } => {
                if !self.allow_omissions {
                    return Err(SchemaExportError::UnsupportedRepresentation(
                        "Extended binary representation cannot be strictly exported to standard JSON Schema".to_string(),
                    ));
                }
                self.omissions.push(OmittedConstraint {
                    path: self.path.clone(),
                    constraint: "binary".to_string(),
                    reason: "Extended binary representation projected to string with contentEncoding".to_string(),
                });
                doc.try_insert("type", DataValue::String("string".to_string())).unwrap();
                doc.try_insert("contentEncoding", DataValue::String("base64".to_string())).unwrap();
                if let Some(min) = min_len {
                    doc.try_insert("minLength", DataValue::Number(ExactNumber::Int(*min as i64))).unwrap();
                }
                if let Some(max) = max_len {
                    doc.try_insert("maxLength", DataValue::Number(ExactNumber::Int(*max as i64))).unwrap();
                }
            }
            SchemaIR::ObjectId | SchemaIR::DateTimeUtc | SchemaIR::Decimal128 => {
                let name = match ir {
                    SchemaIR::ObjectId => "ObjectId",
                    SchemaIR::DateTimeUtc => "DateTimeUtc",
                    SchemaIR::Decimal128 => "Decimal128",
                    _ => unreachable!(),
                };
                if !self.allow_omissions {
                    return Err(SchemaExportError::UnsupportedRepresentation(format!(
                        "Extended BSON type '{}' cannot be strictly exported to standard JSON Schema",
                        name
                    )));
                }
                self.omissions.push(OmittedConstraint {
                    path: self.path.clone(),
                    constraint: name.to_string(),
                    reason: format!("Extended BSON type '{}' projected to string format annotation", name),
                });
                doc.try_insert("type", DataValue::String("string".to_string())).unwrap();
                doc.try_insert("format", DataValue::String(name.to_lowercase())).unwrap();
            }
            SchemaIR::Literal(val) => {
                doc.try_insert("const", val.clone()).unwrap();
            }
            SchemaIR::Array { item, min_items, max_items } => {
                doc.try_insert("type", DataValue::String("array".to_string())).unwrap();
                self.path.push("[]".to_string());
                let item_doc = self.export_ir(item)?;
                self.path.pop();
                doc.try_insert("items", DataValue::Document(item_doc)).unwrap();
                if let Some(min) = min_items {
                    doc.try_insert("minItems", DataValue::Number(ExactNumber::Int(*min as i64))).unwrap();
                }
                if let Some(max) = max_items {
                    doc.try_insert("maxItems", DataValue::Number(ExactNumber::Int(*max as i64))).unwrap();
                }
            }
            SchemaIR::Object { fields, strict } => {
                doc.try_insert("type", DataValue::String("object".to_string())).unwrap();
                let mut props = Document::new();
                let mut required = Vec::new();

                for field in fields {
                    self.path.push(field.name.clone());
                    let field_doc = self.export_ir(&field.schema)?;
                    self.path.pop();
                    props.try_insert(&field.name, DataValue::Document(field_doc)).unwrap();

                    if !field.optional {
                        required.push(DataValue::String(field.name.clone()));
                    }
                }

                doc.try_insert("properties", DataValue::Document(props)).unwrap();
                if !required.is_empty() {
                    doc.try_insert("required", DataValue::Array(required)).unwrap();
                }
                // .strict() exports "additionalProperties": false, passthrough exports true
                doc.try_insert("additionalProperties", DataValue::Bool(!*strict)).unwrap();
            }
            SchemaIR::Record { value, .. } => {
                doc.try_insert("type", DataValue::String("object".to_string())).unwrap();
                self.path.push("[*]".to_string());
                let val_doc = self.export_ir(value)?;
                self.path.pop();
                doc.try_insert("additionalProperties", DataValue::Document(val_doc)).unwrap();
            }
            SchemaIR::Union(variants) => {
                let mut any_of = Vec::new();
                for (i, v) in variants.iter().enumerate() {
                    self.path.push(format!("<union:{}>", i));
                    let v_doc = self.export_ir(v)?;
                    self.path.pop();
                    any_of.push(DataValue::Document(v_doc));
                }
                doc.try_insert("anyOf", DataValue::Array(any_of)).unwrap();
            }
            SchemaIR::DiscriminatedUnion { discriminator, variants } => {
                let mut one_of = Vec::new();
                for (i, v) in variants.iter().enumerate() {
                    self.path.push(format!("<discrim:{}>", i));
                    let v_doc = self.export_ir(v)?;
                    self.path.pop();
                    one_of.push(DataValue::Document(v_doc));
                }
                doc.try_insert("oneOf", DataValue::Array(one_of)).unwrap();

                if self.dialect == JsonSchemaDialect::OpenApi3_1 {
                    let mut disc_doc = Document::new();
                    disc_doc.try_insert("propertyName", DataValue::String(discriminator.clone())).unwrap();
                    doc.try_insert("discriminator", DataValue::Document(disc_doc)).unwrap();
                }
            }
            SchemaIR::Optional(inner) => {
                return self.export_ir(inner);
            }
            SchemaIR::Nullable(inner) => {
                let inner_doc = self.export_ir(inner)?;
                let mut null_doc = Document::new();
                null_doc.try_insert("type", DataValue::String("null".to_string())).unwrap();
                let any_of = vec![DataValue::Document(inner_doc), DataValue::Document(null_doc)];
                doc.try_insert("anyOf", DataValue::Array(any_of)).unwrap();
            }
            SchemaIR::Refine { inner, message, .. } => {
                if !self.allow_omissions {
                    return Err(SchemaExportError::NonPortableConstraint(format!(
                        "Pure refine callback{} at path '{}' cannot be exported to JSON Schema",
                        message.as_ref().map(|m| format!(" ('{}')", m)).unwrap_or_default(),
                        self.path.join(".")
                    )));
                }
                self.omissions.push(OmittedConstraint {
                    path: self.path.clone(),
                    constraint: "refine".to_string(),
                    reason: message.clone().unwrap_or_else(|| "Pure refine callback omitted".to_string()),
                });
                return self.export_ir(inner);
            }
            SchemaIR::Transform { inner, .. } => {
                if !self.allow_omissions {
                    return Err(SchemaExportError::NonPortableTransform(format!(
                        "Transform stage at path '{}' cannot be exported to declarative JSON Schema",
                        self.path.join(".")
                    )));
                }
                self.omissions.push(OmittedConstraint {
                    path: self.path.clone(),
                    constraint: "transform".to_string(),
                    reason: "Transform stage omitted in input projection".to_string(),
                });
                return self.export_ir(inner);
            }
            SchemaIR::InvalidRegex { pattern, error } => {
                return Err(SchemaExportError::NonPortablePattern(format!(
                    "Invalid regex '{}': {}",
                    pattern, error
                )));
            }
        }
        Ok(doc)
    }
}

/// Import a JSON Schema `Document` into a Lust `Schema` (DS-90).
pub fn schema_from_json_schema(
    document: &Document,
    options: Option<&ImportOptions>,
) -> Result<Schema, SchemaImportError> {
    let opts = options.cloned().unwrap_or_default();
    let mut importer = SchemaImporter::new(document, opts);
    let ir = importer.import_root(document)?;
    Ok(Schema::new(ir))
}

struct SchemaImporter<'a> {
    defs: HashMap<String, Document>,
    ref_stack: HashSet<String>,
    depth: usize,
    nodes: usize,
    max_depth: usize,
    max_nodes: usize,
    _root: &'a Document,
}

impl<'a> SchemaImporter<'a> {
    fn new(root: &'a Document, options: ImportOptions) -> Self {
        let mut defs = HashMap::new();
        Self::collect_defs(root, &mut defs);
        Self {
            defs,
            ref_stack: HashSet::new(),
            depth: 0,
            nodes: 0,
            max_depth: options.max_depth,
            max_nodes: options.max_nodes,
            _root: root,
        }
    }

    fn collect_defs(doc: &Document, defs: &mut HashMap<String, Document>) {
        for key in ["$defs", "definitions"] {
            if let Some(DataValue::Document(d)) = doc.get(key) {
                for (name, val) in d.iter() {
                    if let DataValue::Document(sub) = val {
                        defs.insert(format!("#/{}/{}", key, name), sub.clone());
                    }
                }
            }
        }
    }

    fn check_budget(&mut self) -> Result<(), SchemaImportError> {
        self.nodes += 1;
        if self.nodes > self.max_nodes {
            return Err(SchemaImportError::BudgetExceeded(format!(
                "Node budget of {} exceeded",
                self.max_nodes
            )));
        }
        if self.depth > self.max_depth {
            return Err(SchemaImportError::BudgetExceeded(format!(
                "Nesting depth limit of {} exceeded",
                self.max_depth
            )));
        }
        Ok(())
    }

    fn import_root(&mut self, doc: &Document) -> Result<SchemaIR, SchemaImportError> {
        self.check_budget()?;
        self.depth += 1;
        let res = self.import_document(doc);
        self.depth -= 1;
        res
    }

    fn import_document(&mut self, doc: &Document) -> Result<SchemaIR, SchemaImportError> {
        // Check for $ref
        if let Some(DataValue::String(r)) = doc.get("$ref") {
            if r.starts_with("http://") || r.starts_with("https://") || r.starts_with("file://") {
                return Err(SchemaImportError::UnsupportedRemoteRef(r.clone()));
            }
            if self.ref_stack.contains(r) {
                return Err(SchemaImportError::CyclicRef(r.clone()));
            }

            let target_doc = self.defs.get(r).cloned().ok_or_else(|| {
                SchemaImportError::UnresolvedRef(r.clone())
            })?;

            self.ref_stack.insert(r.clone());
            let imported = self.import_root(&target_doc);
            self.ref_stack.remove(r);
            return imported;
        }

        // Check for const
        if let Some(val) = doc.get("const") {
            return Ok(SchemaIR::Literal(val.clone()));
        }

        // Check for enum
        if let Some(DataValue::Array(items)) = doc.get("enum") {
            let variants = items.iter().map(|item| SchemaIR::Literal(item.clone())).collect();
            return Ok(SchemaIR::Union(variants));
        }

        // Check for oneOf
        if let Some(DataValue::Array(branches)) = doc.get("oneOf") {
            let mut variants = Vec::new();
            for b in branches {
                if let DataValue::Document(b_doc) = b {
                    variants.push(self.import_root(b_doc)?);
                }
            }

            // Check if OpenAPI discriminator is present
            if let Some(DataValue::Document(disc_doc)) = doc.get("discriminator") {
                if let Some(DataValue::String(prop)) = disc_doc.get("propertyName") {
                    return Ok(SchemaIR::DiscriminatedUnion {
                        discriminator: prop.clone(),
                        variants,
                    });
                }
            }

            return Ok(SchemaIR::Union(variants));
        }

        // Check for anyOf
        if let Some(DataValue::Array(branches)) = doc.get("anyOf") {
            let mut has_null = false;
            let mut non_null_branches = Vec::new();

            for b in branches {
                if let DataValue::Document(b_doc) = b {
                    if let Some(DataValue::String(t)) = b_doc.get("type") {
                        if t == "null" {
                            has_null = true;
                            continue;
                        }
                    }
                    non_null_branches.push(self.import_root(b_doc)?);
                }
            }

            if has_null && non_null_branches.len() == 1 {
                return Ok(SchemaIR::Nullable(Box::new(non_null_branches.remove(0))));
            }

            let mut all_variants = non_null_branches;
            if has_null {
                all_variants.push(SchemaIR::Literal(DataValue::Null));
            }
            return Ok(SchemaIR::Union(all_variants));
        }

        // Check type keyword
        let ty = match doc.get("type") {
            Some(DataValue::String(s)) => s.as_str(),
            Some(DataValue::Array(types)) => {
                // If type array contains "null" plus one other type, treat as Nullable
                let mut has_null = false;
                let mut other_types = Vec::new();
                for t in types {
                    if let DataValue::String(ts) = t {
                        if ts == "null" {
                            has_null = true;
                        } else {
                            other_types.push(ts.clone());
                        }
                    }
                }
                if has_null && other_types.len() == 1 {
                    let mut single_doc = doc.clone();
                    single_doc.try_insert("type", DataValue::String(other_types[0].clone())).unwrap();
                    let inner = self.import_root(&single_doc)?;
                    return Ok(SchemaIR::Nullable(Box::new(inner)));
                }
                return Err(SchemaImportError::UnsupportedKeyword(
                    "Multi-type arrays without null are not supported in subset 1".to_string(),
                ));
            }
            None => {
                // Untyped schema document with properties => treat as object
                if doc.get("properties").is_some() {
                    "object"
                } else if doc.get("items").is_some() {
                    "array"
                } else {
                    return Ok(SchemaIR::Any);
                }
            }
            _ => return Err(SchemaImportError::InvalidSchema("Invalid 'type' value".to_string())),
        };

        match ty {
            "boolean" => Ok(SchemaIR::Bool),
            "integer" => {
                let min = self.extract_int_bound(doc, "minimum", "exclusiveMinimum")?;
                let max = self.extract_int_bound(doc, "maximum", "exclusiveMaximum")?;
                Ok(SchemaIR::Int { min, max })
            }
            "number" => {
                let min = self.extract_float_bound(doc, "minimum", "exclusiveMinimum")?;
                let max = self.extract_float_bound(doc, "maximum", "exclusiveMaximum")?;
                Ok(SchemaIR::Float { min, max })
            }
            "string" => {
                let min_len = doc.get("minLength").and_then(|v| match v {
                    DataValue::Number(ExactNumber::Int(i)) if *i >= 0 => Some(*i as usize),
                    _ => None,
                });
                let max_len = doc.get("maxLength").and_then(|v| match v {
                    DataValue::Number(ExactNumber::Int(i)) if *i >= 0 => Some(*i as usize),
                    _ => None,
                });

                let string_ir = SchemaIR::String {
                    min_len,
                    max_len,
                    regex: None,
                };

                // JSON Schema pattern uses search-mode semantics (contains_pattern)
                if let Some(DataValue::String(pat)) = doc.get("pattern") {
                    let imported_pat = import_regex_pattern(pat)?;
                    Ok(SchemaIR::Pattern {
                        inner: Box::new(string_ir),
                        pattern: imported_pat,
                        mode: PatternMode::Search,
                    })
                } else {
                    Ok(string_ir)
                }
            }
            "null" => Ok(SchemaIR::Literal(DataValue::Null)),
            "array" => {
                let item = if let Some(DataValue::Document(item_doc)) = doc.get("items") {
                    Box::new(self.import_root(item_doc)?)
                } else {
                    Box::new(SchemaIR::Any)
                };
                let min_items = doc.get("minItems").and_then(|v| match v {
                    DataValue::Number(ExactNumber::Int(i)) if *i >= 0 => Some(*i as usize),
                    _ => None,
                });
                let max_items = doc.get("maxItems").and_then(|v| match v {
                    DataValue::Number(ExactNumber::Int(i)) if *i >= 0 => Some(*i as usize),
                    _ => None,
                });
                Ok(SchemaIR::Array {
                    item,
                    min_items,
                    max_items,
                })
            }
            "object" => {
                let mut fields = Vec::new();
                let required_set: HashSet<String> = if let Some(DataValue::Array(reqs)) = doc.get("required") {
                    reqs.iter()
                        .filter_map(|r| match r {
                            DataValue::String(s) => Some(s.clone()),
                            _ => None,
                        })
                        .collect()
                } else {
                    HashSet::new()
                };

                if let Some(DataValue::Document(props)) = doc.get("properties") {
                    for (field_name, field_val) in props.iter() {
                        if let DataValue::Document(field_doc) = field_val {
                            let schema = self.import_root(field_doc)?;
                            let optional = !required_set.contains(field_name);
                            fields.push(ObjectFieldIR {
                                name: field_name.clone(),
                                schema,
                                optional,
                                default: None,
                            });
                        }
                    }
                }

                // In JSON Schema import: absent additionalProperties means allow (passthrough),
                // while "additionalProperties": false explicitly means strict!
                let strict = match doc.get("additionalProperties") {
                    Some(DataValue::Bool(false)) => true,
                    _ => false,
                };

                Ok(SchemaIR::Object { fields, strict })
            }
            other => Err(SchemaImportError::UnsupportedKeyword(format!(
                "Type '{}' is not supported in LustJsonSchema2020Subset1",
                other
            ))),
        }
    }

    fn extract_int_bound(
        &self,
        doc: &Document,
        inclusive_key: &str,
        exclusive_key: &str,
    ) -> Result<Option<i64>, SchemaImportError> {
        if let Some(v) = doc.get(inclusive_key) {
            match v {
                DataValue::Number(ExactNumber::Int(i)) => Ok(Some(*i)),
                _ => Err(SchemaImportError::InvalidSchema(format!("Expected integer for '{}'", inclusive_key))),
            }
        } else if let Some(v) = doc.get(exclusive_key) {
            match v {
                DataValue::Number(ExactNumber::Int(i)) => Ok(Some(i.saturating_add(1))),
                _ => Err(SchemaImportError::InvalidSchema(format!("Expected integer for '{}'", exclusive_key))),
            }
        } else {
            Ok(None)
        }
    }

    fn extract_float_bound(
        &self,
        doc: &Document,
        inclusive_key: &str,
        exclusive_key: &str,
    ) -> Result<Option<f64>, SchemaImportError> {
        if let Some(v) = doc.get(inclusive_key) {
            match v {
                DataValue::Number(ExactNumber::Float(f)) => Ok(Some(*f)),
                DataValue::Number(ExactNumber::Int(i)) => Ok(Some(*i as f64)),
                _ => Err(SchemaImportError::InvalidSchema(format!("Expected number for '{}'", inclusive_key))),
            }
        } else if let Some(v) = doc.get(exclusive_key) {
            match v {
                DataValue::Number(ExactNumber::Float(f)) => Ok(Some(*f)),
                DataValue::Number(ExactNumber::Int(i)) => Ok(Some(*i as f64)),
                _ => Err(SchemaImportError::InvalidSchema(format!("Expected number for '{}'", exclusive_key))),
            }
        } else {
            Ok(None)
        }
    }
}
