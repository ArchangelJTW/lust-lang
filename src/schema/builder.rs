use std::sync::Arc;
use crate::data::DataValue;
use crate::schema::error::{ValidationError, ValidationErrors};
use crate::schema::ir::{ObjectFieldIR, RefinePredicate, SchemaIR, TransformFn};
use crate::schema::validate::validate_ir;

#[derive(Debug, Clone, PartialEq)]
pub struct Schema {
    pub ir: SchemaIR,
}

impl Schema {
    pub fn new(ir: SchemaIR) -> Self {
        Self { ir }
    }

    pub fn parse(&self, value: &DataValue) -> Result<DataValue, ValidationErrors> {
        let mut path = Vec::new();
        let mut errors = Vec::new();
        if let Some(val) = validate_ir(&self.ir, value, &mut path, &mut errors) {
            if errors.is_empty() {
                return Ok(val);
            }
        }
        if errors.is_empty() {
            errors.push(ValidationError::new(
                path,
                "Validation failed",
                "validation_failed",
            ));
        }
        Err(ValidationErrors::new(errors))
    }

    pub fn validate(&self, value: &DataValue) -> crate::schema::Effect<DataValue, ValidationErrors> {
        match self.parse(value) {
            Ok(val) => crate::schema::Effect::success(val),
            Err(errs) => crate::schema::Effect::failure(errs),
        }
    }

    pub fn validate_single(&self, value: &DataValue) -> crate::schema::Effect<DataValue, ValidationError> {
        match self.parse(value) {
            Ok(val) => crate::schema::Effect::success(val),
            Err(errs) => crate::schema::Effect::failure(errs.into()),
        }
    }

    pub fn optional(self) -> Self {
        Schema::new(SchemaIR::Optional(Box::new(self.ir)))
    }

    pub fn nullable(self) -> Self {
        Schema::new(SchemaIR::Nullable(Box::new(self.ir)))
    }

    pub fn refine<F>(self, predicate: F) -> Self
    where
        F: Fn(&DataValue) -> bool + Send + Sync + 'static,
    {
        Schema::new(SchemaIR::Refine {
            inner: Box::new(self.ir),
            predicate: RefinePredicate {
                f: Arc::new(predicate),
            },
            message: None,
        })
    }

    pub fn refine_with_message<F>(self, predicate: F, message: impl Into<String>) -> Self
    where
        F: Fn(&DataValue) -> bool + Send + Sync + 'static,
    {
        Schema::new(SchemaIR::Refine {
            inner: Box::new(self.ir),
            predicate: RefinePredicate {
                f: Arc::new(predicate),
            },
            message: Some(message.into()),
        })
    }

    pub fn transform<F>(self, transform_fn: F) -> Self
    where
        F: Fn(DataValue) -> DataValue + Send + Sync + 'static,
    {
        Schema::new(SchemaIR::Transform {
            inner: Box::new(self.ir),
            transform: TransformFn {
                f: Arc::new(transform_fn),
            },
        })
    }

    pub fn to_json_schema(&self) -> Result<crate::data::Document, crate::schema::interop::SchemaExportError> {
        crate::schema::interop::to_json_schema(self, None)
    }

    pub fn to_json_schema_with_options(
        &self,
        options: &crate::schema::interop::ExportOptions,
    ) -> Result<crate::data::Document, crate::schema::interop::SchemaExportError> {
        crate::schema::interop::to_json_schema(self, Some(options))
    }

    pub fn as_json_schema_portable(
        &self,
        options: Option<&crate::schema::interop::ExportOptions>,
    ) -> Result<crate::schema::interop::JsonSchemaPortable, crate::schema::interop::SchemaExportError> {
        crate::schema::interop::as_json_schema_portable(self, options)
    }

    pub fn export_json_schema_projection(
        &self,
        projection: crate::schema::interop::ProjectionMode,
    ) -> Result<crate::schema::interop::SchemaProjection, crate::schema::interop::SchemaExportError> {
        crate::schema::interop::export_json_schema_projection(self, projection)
    }

    pub fn min(mut self, min: usize) -> Self {
        match &mut self.ir {
            SchemaIR::String { min_len, .. } => {
                *min_len = Some(min);
            }
            SchemaIR::Int { min: m, .. } => {
                *m = Some(min as i64);
            }
            SchemaIR::Float { min: m, .. } => {
                *m = Some(min as f64);
            }
            SchemaIR::Array { min_items, .. } => {
                *min_items = Some(min);
            }
            SchemaIR::Binary { min_len, .. } => {
                *min_len = Some(min);
            }
            _ => {}
        }
        self
    }

    pub fn max(mut self, max: usize) -> Self {
        match &mut self.ir {
            SchemaIR::String { max_len, .. } => {
                *max_len = Some(max);
            }
            SchemaIR::Int { max: m, .. } => {
                *m = Some(max as i64);
            }
            SchemaIR::Float { max: m, .. } => {
                *m = Some(max as f64);
            }
            SchemaIR::Array { max_items, .. } => {
                *max_items = Some(max);
            }
            SchemaIR::Binary { max_len, .. } => {
                *max_len = Some(max);
            }
            _ => {}
        }
        self
    }

    pub fn compile(self) -> SchemaIR {
        self.ir
    }

    pub fn to_ir(&self) -> &SchemaIR {
        &self.ir
    }
}

pub struct StringBuilder {
    min_len: Option<usize>,
    max_len: Option<usize>,
    regex: Option<String>,
}

impl StringBuilder {
    pub fn new() -> Self {
        Self {
            min_len: None,
            max_len: None,
            regex: None,
        }
    }

    pub fn min(mut self, len: usize) -> Self {
        self.min_len = Some(len);
        self
    }

    pub fn max(mut self, len: usize) -> Self {
        self.max_len = Some(len);
        self
    }

    pub fn regex(mut self, pattern: impl Into<String>) -> Self {
        self.regex = Some(pattern.into());
        self
    }

    pub fn build(self) -> Schema {
        if let Some(ref pattern) = self.regex {
            if let Err(e) = crate::regex::BoundedRegex::new(pattern) {
                return Schema::new(SchemaIR::InvalidRegex {
                    pattern: pattern.clone(),
                    error: e.to_string(),
                });
            }
        }
        Schema::new(SchemaIR::String {
            min_len: self.min_len,
            max_len: self.max_len,
            regex: self.regex,
        })
    }

    pub fn compile(self) -> SchemaIR {
        self.build().ir
    }

    pub fn refine<F>(self, predicate: F) -> Schema
    where
        F: Fn(&DataValue) -> bool + Send + Sync + 'static,
    {
        self.build().refine(predicate)
    }

    pub fn refine_with_message<F>(self, predicate: F, message: impl Into<String>) -> Schema
    where
        F: Fn(&DataValue) -> bool + Send + Sync + 'static,
    {
        self.build().refine_with_message(predicate, message)
    }

    pub fn transform<F>(self, transform_fn: F) -> Schema
    where
        F: Fn(DataValue) -> DataValue + Send + Sync + 'static,
    {
        self.build().transform(transform_fn)
    }

    pub fn optional(self) -> Schema {
        self.build().optional()
    }

    pub fn nullable(self) -> Schema {
        self.build().nullable()
    }
}

impl From<StringBuilder> for Schema {
    fn from(b: StringBuilder) -> Self {
        b.build()
    }
}

pub struct IntBuilder {
    min: Option<i64>,
    max: Option<i64>,
}

impl IntBuilder {
    pub fn new() -> Self {
        Self { min: None, max: None }
    }

    pub fn min(mut self, val: i64) -> Self {
        self.min = Some(val);
        self
    }

    pub fn max(mut self, val: i64) -> Self {
        self.max = Some(val);
        self
    }

    pub fn build(self) -> Schema {
        Schema::new(SchemaIR::Int {
            min: self.min,
            max: self.max,
        })
    }

    pub fn compile(self) -> SchemaIR {
        self.build().ir
    }

    pub fn refine<F>(self, predicate: F) -> Schema
    where
        F: Fn(&DataValue) -> bool + Send + Sync + 'static,
    {
        self.build().refine(predicate)
    }

    pub fn refine_with_message<F>(self, predicate: F, message: impl Into<String>) -> Schema
    where
        F: Fn(&DataValue) -> bool + Send + Sync + 'static,
    {
        self.build().refine_with_message(predicate, message)
    }

    pub fn transform<F>(self, transform_fn: F) -> Schema
    where
        F: Fn(DataValue) -> DataValue + Send + Sync + 'static,
    {
        self.build().transform(transform_fn)
    }

    pub fn optional(self) -> Schema {
        self.build().optional()
    }

    pub fn nullable(self) -> Schema {
        self.build().nullable()
    }
}

impl From<IntBuilder> for Schema {
    fn from(b: IntBuilder) -> Self {
        b.build()
    }
}

pub struct FloatBuilder {
    min: Option<f64>,
    max: Option<f64>,
}

impl FloatBuilder {
    pub fn new() -> Self {
        Self { min: None, max: None }
    }

    pub fn min(mut self, val: f64) -> Self {
        self.min = Some(val);
        self
    }

    pub fn max(mut self, val: f64) -> Self {
        self.max = Some(val);
        self
    }

    pub fn build(self) -> Schema {
        Schema::new(SchemaIR::Float {
            min: self.min,
            max: self.max,
        })
    }

    pub fn compile(self) -> SchemaIR {
        self.build().ir
    }

    pub fn refine<F>(self, predicate: F) -> Schema
    where
        F: Fn(&DataValue) -> bool + Send + Sync + 'static,
    {
        self.build().refine(predicate)
    }

    pub fn refine_with_message<F>(self, predicate: F, message: impl Into<String>) -> Schema
    where
        F: Fn(&DataValue) -> bool + Send + Sync + 'static,
    {
        self.build().refine_with_message(predicate, message)
    }

    pub fn transform<F>(self, transform_fn: F) -> Schema
    where
        F: Fn(DataValue) -> DataValue + Send + Sync + 'static,
    {
        self.build().transform(transform_fn)
    }

    pub fn optional(self) -> Schema {
        self.build().optional()
    }

    pub fn nullable(self) -> Schema {
        self.build().nullable()
    }
}

impl From<FloatBuilder> for Schema {
    fn from(b: FloatBuilder) -> Self {
        b.build()
    }
}

pub struct ArrayBuilder {
    item: SchemaIR,
    min_items: Option<usize>,
    max_items: Option<usize>,
}

impl ArrayBuilder {
    pub fn new(item: Schema) -> Self {
        Self {
            item: item.ir,
            min_items: None,
            max_items: None,
        }
    }

    pub fn min(mut self, count: usize) -> Self {
        self.min_items = Some(count);
        self
    }

    pub fn max(mut self, count: usize) -> Self {
        self.max_items = Some(count);
        self
    }

    pub fn build(self) -> Schema {
        Schema::new(SchemaIR::Array {
            item: Box::new(self.item),
            min_items: self.min_items,
            max_items: self.max_items,
        })
    }

    pub fn compile(self) -> SchemaIR {
        self.build().ir
    }

    pub fn refine<F>(self, predicate: F) -> Schema
    where
        F: Fn(&DataValue) -> bool + Send + Sync + 'static,
    {
        self.build().refine(predicate)
    }

    pub fn refine_with_message<F>(self, predicate: F, message: impl Into<String>) -> Schema
    where
        F: Fn(&DataValue) -> bool + Send + Sync + 'static,
    {
        self.build().refine_with_message(predicate, message)
    }

    pub fn transform<F>(self, transform_fn: F) -> Schema
    where
        F: Fn(DataValue) -> DataValue + Send + Sync + 'static,
    {
        self.build().transform(transform_fn)
    }

    pub fn optional(self) -> Schema {
        self.build().optional()
    }

    pub fn nullable(self) -> Schema {
        self.build().nullable()
    }
}

impl From<ArrayBuilder> for Schema {
    fn from(b: ArrayBuilder) -> Self {
        b.build()
    }
}

pub fn unpack_optional(ir: SchemaIR) -> (SchemaIR, bool) {
    match ir {
        SchemaIR::Optional(inner) => {
            let (unwrapped, _) = unpack_optional(*inner);
            (unwrapped, true)
        }
        SchemaIR::Nullable(inner) => {
            let (unwrapped, is_opt) = unpack_optional(*inner);
            (SchemaIR::Nullable(Box::new(unwrapped)), is_opt)
        }
        SchemaIR::Refine { inner, predicate, message } => {
            let (unwrapped, is_opt) = unpack_optional(*inner);
            (SchemaIR::Refine { inner: Box::new(unwrapped), predicate, message }, is_opt)
        }
        SchemaIR::Pattern { inner, pattern, mode } => {
            let (unwrapped, is_opt) = unpack_optional(*inner);
            (SchemaIR::Pattern { inner: Box::new(unwrapped), pattern, mode }, is_opt)
        }
        SchemaIR::Transform { inner, transform } => {
            let (unwrapped, is_opt) = unpack_optional(*inner);
            (SchemaIR::Transform { inner: Box::new(unwrapped), transform }, is_opt)
        }
        other => (other, false),
    }
}

pub struct ObjectBuilder {
    fields: Vec<ObjectFieldIR>,
    strict: bool,
}

impl ObjectBuilder {
    pub fn new() -> Self {
        Self {
            fields: Vec::new(),
            strict: false,
        }
    }

    pub fn field(mut self, name: impl Into<String>, schema: impl Into<Schema>) -> Self {
        let sch = schema.into();
        let (schema_ir, optional) = unpack_optional(sch.ir);
        self.fields.push(ObjectFieldIR {
            name: name.into(),
            schema: schema_ir,
            optional,
            default: None,
        });
        self
    }

    pub fn field_with_default(
        mut self,
        name: impl Into<String>,
        schema: impl Into<Schema>,
        default_val: DataValue,
    ) -> Self {
        let sch = schema.into();
        self.fields.push(ObjectFieldIR {
            name: name.into(),
            schema: sch.ir,
            optional: true,
            default: Some(default_val),
        });
        self
    }

    pub fn strict(mut self) -> Self {
        self.strict = true;
        self
    }

    pub fn build(self) -> Schema {
        Schema::new(SchemaIR::Object {
            fields: self.fields,
            strict: self.strict,
        })
    }

    pub fn compile(self) -> SchemaIR {
        self.build().ir
    }

    pub fn refine<F>(self, predicate: F) -> Schema
    where
        F: Fn(&DataValue) -> bool + Send + Sync + 'static,
    {
        self.build().refine(predicate)
    }

    pub fn refine_with_message<F>(self, predicate: F, message: impl Into<String>) -> Schema
    where
        F: Fn(&DataValue) -> bool + Send + Sync + 'static,
    {
        self.build().refine_with_message(predicate, message)
    }

    pub fn transform<F>(self, transform_fn: F) -> Schema
    where
        F: Fn(DataValue) -> DataValue + Send + Sync + 'static,
    {
        self.build().transform(transform_fn)
    }

    pub fn optional(self) -> Schema {
        self.build().optional()
    }

    pub fn nullable(self) -> Schema {
        self.build().nullable()
    }
}

impl From<ObjectBuilder> for Schema {
    fn from(b: ObjectBuilder) -> Self {
        b.build()
    }
}

pub struct BinaryBuilder {
    min_len: Option<usize>,
    max_len: Option<usize>,
}

impl BinaryBuilder {
    pub fn new() -> Self {
        Self {
            min_len: None,
            max_len: None,
        }
    }

    pub fn min(mut self, len: usize) -> Self {
        self.min_len = Some(len);
        self
    }

    pub fn max(mut self, len: usize) -> Self {
        self.max_len = Some(len);
        self
    }

    pub fn build(self) -> Schema {
        Schema::new(SchemaIR::Binary {
            min_len: self.min_len,
            max_len: self.max_len,
        })
    }

    pub fn compile(self) -> SchemaIR {
        self.build().ir
    }

    pub fn refine<F>(self, predicate: F) -> Schema
    where
        F: Fn(&DataValue) -> bool + Send + Sync + 'static,
    {
        self.build().refine(predicate)
    }

    pub fn refine_with_message<F>(self, predicate: F, message: impl Into<String>) -> Schema
    where
        F: Fn(&DataValue) -> bool + Send + Sync + 'static,
    {
        self.build().refine_with_message(predicate, message)
    }

    pub fn transform<F>(self, transform_fn: F) -> Schema
    where
        F: Fn(DataValue) -> DataValue + Send + Sync + 'static,
    {
        self.build().transform(transform_fn)
    }

    pub fn optional(self) -> Schema {
        self.build().optional()
    }

    pub fn nullable(self) -> Schema {
        self.build().nullable()
    }
}

impl From<BinaryBuilder> for Schema {
    fn from(b: BinaryBuilder) -> Self {
        b.build()
    }
}

pub struct SchemaFactory;

#[allow(non_upper_case_globals)]
pub static s: SchemaFactory = SchemaFactory;

impl SchemaFactory {
    pub fn string(&self) -> StringBuilder {
        StringBuilder::new()
    }

    pub fn int(&self) -> IntBuilder {
        IntBuilder::new()
    }

    pub fn float(&self) -> FloatBuilder {
        FloatBuilder::new()
    }

    pub fn bool(&self) -> Schema {
        Schema::new(SchemaIR::Bool)
    }

    pub fn binary(&self) -> BinaryBuilder {
        BinaryBuilder::new()
    }

    pub fn object_id(&self) -> Schema {
        Schema::new(SchemaIR::ObjectId)
    }

    pub fn datetime_utc(&self) -> Schema {
        Schema::new(SchemaIR::DateTimeUtc)
    }

    pub fn decimal128(&self) -> Schema {
        Schema::new(SchemaIR::Decimal128)
    }

    pub fn any(&self) -> Schema {
        Schema::new(SchemaIR::Any)
    }

    pub fn literal(&self, val: impl Into<DataValue>) -> Schema {
        Schema::new(SchemaIR::Literal(val.into()))
    }

    pub fn array(&self, item: impl Into<Schema>) -> ArrayBuilder {
        ArrayBuilder::new(item.into())
    }

    pub fn object(&self) -> ObjectBuilder {
        ObjectBuilder::new()
    }

    pub fn record(&self, key: impl Into<Schema>, value: impl Into<Schema>) -> Schema {
        Schema::new(SchemaIR::Record {
            key: Box::new(key.into().ir),
            value: Box::new(value.into().ir),
        })
    }

    pub fn union(&self, variants: Vec<Schema>) -> Schema {
        Schema::new(SchemaIR::Union(variants.into_iter().map(|sch| sch.ir).collect()))
    }

    pub fn discriminated_union(&self, discriminator: impl Into<String>, variants: Vec<Schema>) -> Schema {
        Schema::new(SchemaIR::DiscriminatedUnion {
            discriminator: discriminator.into(),
            variants: variants.into_iter().map(|sch| sch.ir).collect(),
        })
    }

    pub fn r#struct(&self, name: impl Into<String>) -> crate::schema::binding::StructSchemaBuilder {
        crate::schema::binding::StructSchemaBuilder::new(name)
    }

    pub fn derive_schema<T: crate::schema::binding::DeriveSchemaTarget>(&self, target: T) -> Schema {
        target.derive_schema()
    }

    pub fn schema_from_json_schema(
        &self,
        doc: &crate::data::Document,
        options: Option<&crate::schema::interop::ImportOptions>,
    ) -> Result<Schema, crate::schema::interop::SchemaImportError> {
        crate::schema::interop::schema_from_json_schema(doc, options)
    }
}
