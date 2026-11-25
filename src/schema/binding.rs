use crate::ast::{Item, ItemKind, StructDef};
use crate::data::layout::{CanonicalTypeTag, LogicalFieldDesc, LogicalLayoutV1, TargetAbi};
use crate::data::materialize::{
    materialize_from_document, struct_def_to_layout, MaterializeError, MaterializedStruct,
};
use crate::data::{DataValue, Document};
use crate::schema::builder::Schema;
use crate::schema::ir::{ObjectFieldIR, SchemaIR};
use std::ops::Deref;

#[derive(Debug, Clone, PartialEq)]
pub struct StructSchemaBinding {
    pub layout: LogicalLayoutV1,
    pub schema: Schema,
}

impl StructSchemaBinding {
    pub fn new(layout: LogicalLayoutV1) -> Self {
        let schema = derive_schema_from_layout(&layout);
        Self { layout, schema }
    }

    pub fn from_layout(layout: LogicalLayoutV1) -> Self {
        Self::new(layout)
    }

    pub fn from_struct_def(def: &StructDef) -> Self {
        let layout = struct_def_to_layout(def);
        Self::new(layout)
    }

    pub fn from_item(item: &Item) -> Result<Self, MaterializeError> {
        match &item.kind {
            ItemKind::Struct(def) => Ok(Self::from_struct_def(def)),
            _ => Err(MaterializeError::NominalTypeMismatch {
                expected: "ItemKind::Struct".to_string(),
                received: format!("{:?}", item.kind),
            }),
        }
    }

    pub fn get_field_by_ordinal(&self, ordinal: u16) -> Option<&LogicalFieldDesc> {
        self.layout.fields.iter().find(|f| f.ordinal == ordinal)
    }

    pub fn get_field_by_name(&self, name: &str) -> Option<&LogicalFieldDesc> {
        self.layout.fields.iter().find(|f| f.name == name)
    }

    pub fn materialize(&self, doc: &Document) -> Result<MaterializedStruct, MaterializeError> {
        materialize_from_document(&self.layout, doc)
    }
}

impl Deref for StructSchemaBinding {
    type Target = Schema;

    fn deref(&self) -> &Self::Target {
        &self.schema
    }
}

impl From<StructSchemaBinding> for Schema {
    fn from(binding: StructSchemaBinding) -> Self {
        binding.schema
    }
}

pub fn derive_schema_from_layout(layout: &LogicalLayoutV1) -> Schema {
    let mut fields = Vec::with_capacity(layout.fields.len());

    for field in &layout.fields {
        let mut field_schema = match field.tag {
            CanonicalTypeTag::Bool => SchemaIR::Bool,
            CanonicalTypeTag::Int64 | CanonicalTypeTag::UInt64 => {
                SchemaIR::Int { min: None, max: None }
            }
            CanonicalTypeTag::Float64 => SchemaIR::Float { min: None, max: None },
            CanonicalTypeTag::StringSlice => SchemaIR::String {
                min_len: None,
                max_len: None,
                regex: None,
            },
            CanonicalTypeTag::BinarySlice => SchemaIR::Binary {
                min_len: None,
                max_len: None,
            },
            CanonicalTypeTag::ObjectId => SchemaIR::ObjectId,
            CanonicalTypeTag::DateTimeUtc => SchemaIR::DateTimeUtc,
            CanonicalTypeTag::Decimal128 => SchemaIR::Decimal128,
            CanonicalTypeTag::NestedStruct(_) => SchemaIR::Any,
        };

        if field.is_nullable {
            field_schema = SchemaIR::Nullable(Box::new(field_schema));
        }

        fields.push(ObjectFieldIR {
            name: field.name.clone(),
            schema: field_schema,
            optional: field.is_optional,
            default: None,
        });
    }

    Schema::new(SchemaIR::Object {
        fields,
        strict: true,
    })
}

pub fn derive_schema_from_struct_def(def: &StructDef) -> Schema {
    let layout = struct_def_to_layout(def);
    derive_schema_from_layout(&layout)
}

pub fn derive_schema_from_item(item: &Item) -> Result<Schema, MaterializeError> {
    match &item.kind {
        ItemKind::Struct(def) => Ok(derive_schema_from_struct_def(def)),
        _ => Err(MaterializeError::NominalTypeMismatch {
            expected: "ItemKind::Struct".to_string(),
            received: format!("{:?}", item.kind),
        }),
    }
}

pub trait DeriveSchemaTarget {
    fn derive_schema(&self) -> Schema;
}

impl DeriveSchemaTarget for LogicalLayoutV1 {
    fn derive_schema(&self) -> Schema {
        derive_schema_from_layout(self)
    }
}

impl DeriveSchemaTarget for &LogicalLayoutV1 {
    fn derive_schema(&self) -> Schema {
        derive_schema_from_layout(self)
    }
}

impl DeriveSchemaTarget for StructDef {
    fn derive_schema(&self) -> Schema {
        derive_schema_from_struct_def(self)
    }
}

impl DeriveSchemaTarget for &StructDef {
    fn derive_schema(&self) -> Schema {
        derive_schema_from_struct_def(self)
    }
}

impl DeriveSchemaTarget for Item {
    fn derive_schema(&self) -> Schema {
        derive_schema_from_item(self).unwrap_or_else(|e| panic!("{}", e))
    }
}

impl DeriveSchemaTarget for &Item {
    fn derive_schema(&self) -> Schema {
        derive_schema_from_item(self).unwrap_or_else(|e| panic!("{}", e))
    }
}

pub fn derive_schema<T: DeriveSchemaTarget>(target: T) -> Schema {
    target.derive_schema()
}

/// Fluent builder for nominal struct schemas with direct VM slot bindings
pub struct StructSchemaBuilder {
    name: String,
    fields: Vec<LogicalFieldDesc>,
    schema_fields: Vec<ObjectFieldIR>,
    strict: bool,
}

impl StructSchemaBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            fields: Vec::new(),
            schema_fields: Vec::new(),
            strict: true,
        }
    }

    pub fn field(mut self, name: impl Into<String>, schema: impl Into<Schema>) -> Self {
        let name_str = name.into();
        let sch = schema.into();
        let ordinal = self.fields.len() as u16;

        let (tag, is_nullable, is_optional) = match &sch.ir {
            SchemaIR::Nullable(inner) => match &**inner {
                SchemaIR::Optional(innermost) => (ir_to_canonical_tag(innermost), true, true),
                _ => (ir_to_canonical_tag(inner), true, false),
            },
            SchemaIR::Optional(inner) => match &**inner {
                SchemaIR::Nullable(innermost) => (ir_to_canonical_tag(innermost), true, true),
                _ => (ir_to_canonical_tag(inner), false, true),
            },
            other => (ir_to_canonical_tag(other), false, false),
        };

        let field_desc = LogicalFieldDesc::new(ordinal, &name_str, tag)
            .with_nullability(is_nullable, is_optional);
        self.fields.push(field_desc);

        let (schema_ir, optional) = crate::schema::builder::unpack_optional(sch.ir);

        self.schema_fields.push(ObjectFieldIR {
            name: name_str,
            schema: schema_ir,
            optional,
            default: None,
        });

        self
    }

    pub fn nullable_field(mut self, name: impl Into<String>, schema: impl Into<Schema>) -> Self {
        let name_str = name.into();
        let sch = schema.into();
        let ordinal = self.fields.len() as u16;
        let tag = ir_to_canonical_tag(&sch.ir);

        let field_desc = LogicalFieldDesc::new(ordinal, &name_str, tag)
            .with_nullability(true, false);
        self.fields.push(field_desc);

        self.schema_fields.push(ObjectFieldIR {
            name: name_str,
            schema: SchemaIR::Nullable(Box::new(sch.ir)),
            optional: false,
            default: None,
        });

        self
    }

    pub fn optional_field(mut self, name: impl Into<String>, schema: impl Into<Schema>) -> Self {
        let name_str = name.into();
        let sch = schema.into();
        let ordinal = self.fields.len() as u16;
        let tag = ir_to_canonical_tag(&sch.ir);

        let field_desc = LogicalFieldDesc::new(ordinal, &name_str, tag)
            .with_nullability(true, true);
        self.fields.push(field_desc);

        self.schema_fields.push(ObjectFieldIR {
            name: name_str,
            schema: SchemaIR::Optional(Box::new(sch.ir)),
            optional: true,
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
        let name_str = name.into();
        let sch = schema.into();
        let ordinal = self.fields.len() as u16;
        let tag = ir_to_canonical_tag(&sch.ir);

        let field_desc = LogicalFieldDesc::new(ordinal, &name_str, tag)
            .with_nullability(true, true);
        self.fields.push(field_desc);

        self.schema_fields.push(ObjectFieldIR {
            name: name_str,
            schema: sch.ir,
            optional: true,
            default: Some(default_val),
        });

        self
    }

    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    pub fn build(self) -> StructSchemaBinding {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&self.name, &mut hasher);
        let type_id = std::hash::Hasher::finish(&hasher);

        let layout = LogicalLayoutV1::new_with_abi(
            type_id,
            self.name,
            self.fields,
            TargetAbi::x86_64(),
        );

        let schema = Schema::new(SchemaIR::Object {
            fields: self.schema_fields,
            strict: self.strict,
        });

        StructSchemaBinding { layout, schema }
    }

    pub fn compile(self) -> SchemaIR {
        self.build().schema.ir
    }
}

impl From<StructSchemaBuilder> for Schema {
    fn from(b: StructSchemaBuilder) -> Self {
        b.build().schema
    }
}

impl From<StructSchemaBuilder> for StructSchemaBinding {
    fn from(b: StructSchemaBuilder) -> Self {
        b.build()
    }
}

fn ir_to_canonical_tag(ir: &SchemaIR) -> CanonicalTypeTag {
    match ir {
        SchemaIR::Bool => CanonicalTypeTag::Bool,
        SchemaIR::Int { .. } => CanonicalTypeTag::Int64,
        SchemaIR::Float { .. } => CanonicalTypeTag::Float64,
        SchemaIR::String { .. } => CanonicalTypeTag::StringSlice,
        SchemaIR::Binary { .. } => CanonicalTypeTag::BinarySlice,
        SchemaIR::ObjectId => CanonicalTypeTag::ObjectId,
        SchemaIR::DateTimeUtc => CanonicalTypeTag::DateTimeUtc,
        SchemaIR::Decimal128 => CanonicalTypeTag::Decimal128,
        SchemaIR::Nullable(inner) | SchemaIR::Optional(inner) => ir_to_canonical_tag(inner),
        SchemaIR::Refine { inner, .. } | SchemaIR::Transform { inner, .. } => {
            ir_to_canonical_tag(inner)
        }
        _ => CanonicalTypeTag::StringSlice,
    }
}
