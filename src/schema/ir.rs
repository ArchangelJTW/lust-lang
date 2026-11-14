use std::fmt;
use std::sync::Arc;
use crate::data::DataValue;
use crate::schema::pattern::PatternMode;

#[derive(Clone)]
pub struct RefinePredicate {
    pub f: Arc<dyn Fn(&DataValue) -> bool + Send + Sync>,
}

impl fmt::Debug for RefinePredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RefinePredicate(<fn>)")
    }
}

impl PartialEq for RefinePredicate {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.f, &other.f)
    }
}

#[derive(Clone)]
pub struct TransformFn {
    pub f: Arc<dyn Fn(DataValue) -> DataValue + Send + Sync>,
}

impl fmt::Debug for TransformFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TransformFn(<fn>)")
    }
}

impl PartialEq for TransformFn {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.f, &other.f)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObjectFieldIR {
    pub name: String,
    pub schema: SchemaIR,
    pub optional: bool,
    pub default: Option<DataValue>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SchemaIR {
    Any,
    String {
        min_len: Option<usize>,
        max_len: Option<usize>,
        regex: Option<String>,
    },
    Int {
        min: Option<i64>,
        max: Option<i64>,
    },
    Float {
        min: Option<f64>,
        max: Option<f64>,
    },
    Bool,
    Binary {
        min_len: Option<usize>,
        max_len: Option<usize>,
    },
    ObjectId,
    DateTimeUtc,
    Decimal128,
    Literal(DataValue),
    Array {
        item: Box<SchemaIR>,
        min_items: Option<usize>,
        max_items: Option<usize>,
    },
    Object {
        fields: Vec<ObjectFieldIR>,
        strict: bool,
    },
    Record {
        key: Box<SchemaIR>,
        value: Box<SchemaIR>,
    },
    Union(Vec<SchemaIR>),
    DiscriminatedUnion {
        discriminator: String,
        variants: Vec<SchemaIR>,
    },
    Optional(Box<SchemaIR>),
    Nullable(Box<SchemaIR>),
    Pattern {
        inner: Box<SchemaIR>,
        pattern: String,
        mode: PatternMode,
    },
    Refine {
        inner: Box<SchemaIR>,
        predicate: RefinePredicate,
        message: Option<String>,
    },
    Transform {
        inner: Box<SchemaIR>,
        transform: TransformFn,
    },
    InvalidRegex {
        pattern: String,
        error: String,
    },
}
