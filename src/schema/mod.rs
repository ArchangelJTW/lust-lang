//! Fluent schema definition and validation engine for Lust.
//!
//! # Examples
//!
//! ```rust
//! use lust::schema::s;
//! use lust::data::{DataValue, Document, ExactNumber};
//!
//! let user_schema = s.object()
//!     .field("id", s.int().min(1).build())
//!     .field("name", s.string().min(1).max(50).build())
//!     .build();
//!
//! let mut doc = Document::new();
//! doc.try_insert("id", DataValue::Number(ExactNumber::Int(42))).unwrap();
//! doc.try_insert("name", DataValue::String("Alice".to_string())).unwrap();
//!
//! assert!(user_schema.parse(&DataValue::Document(doc)).is_ok());
//! ```

pub mod binding;
pub mod builder;
pub mod error;
pub mod interop;
pub mod ir;
pub mod pattern;
pub mod pipeline;
pub mod validate;

pub use binding::{
    derive_schema, derive_schema_from_item, derive_schema_from_layout,
    derive_schema_from_struct_def, DeriveSchemaTarget, StructSchemaBinding, StructSchemaBuilder,
};
pub use builder::{s, ArrayBuilder, BinaryBuilder, FloatBuilder, IntBuilder, ObjectBuilder, Schema, StringBuilder};
pub use error::{ValidationError, ValidationErrorKind, ValidationErrors};
pub use interop::{
    as_json_schema_portable, export_json_schema_projection, schema_from_json_schema, to_json_schema,
    ExportOptions, ImportOptions, JsonSchemaDialect, JsonSchemaPortable, OmittedConstraint,
    ProjectionMode, SchemaExportError, SchemaImportError, SchemaProjection,
};
pub use ir::{ObjectFieldIR, RefinePredicate, SchemaIR, TransformFn};
pub use pattern::{export_regex_pattern, import_regex_pattern, matches_full, matches_pattern, matches_search, PatternMode};
pub use crate::effect_abi::{Cause, Exit};

/// Effect representation for schema validation returning Exit<T, E> with requirements R
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Effect<T = crate::data::DataValue, E = ValidationErrors, R = ()> {
    pub exit: crate::effect_abi::Exit<T, E>,
    _req: std::marker::PhantomData<fn() -> R>,
}

impl<T, E: std::fmt::Debug, R> Effect<T, E, R> {
    pub fn success(val: T) -> Self {
        Self {
            exit: crate::effect_abi::Exit::Success(val),
            _req: std::marker::PhantomData,
        }
    }

    pub fn failure(err: E) -> Self {
        Self {
            exit: crate::effect_abi::Exit::Failure(crate::effect_abi::Cause::Fail(err)),
            _req: std::marker::PhantomData,
        }
    }

    pub fn is_success(&self) -> bool {
        self.exit.is_success()
    }

    pub fn is_failure(&self) -> bool {
        self.exit.is_failure()
    }

    pub fn into_exit(self) -> crate::effect_abi::Exit<T, E> {
        self.exit
    }

    pub fn into_result(self) -> Result<T, E> {
        match self.exit {
            crate::effect_abi::Exit::Success(v) => Ok(v),
            crate::effect_abi::Exit::Failure(crate::effect_abi::Cause::Fail(e)) => Err(e),
            crate::effect_abi::Exit::Failure(other) => panic!("Effect defect or interruption: {:?}", other),
        }
    }
}

impl<T, E, R> std::ops::Deref for Effect<T, E, R> {
    type Target = crate::effect_abi::Exit<T, E>;
    fn deref(&self) -> &Self::Target {
        &self.exit
    }
}

impl<T, E, R> From<crate::effect_abi::Exit<T, E>> for Effect<T, E, R> {
    fn from(exit: crate::effect_abi::Exit<T, E>) -> Self {
        Self {
            exit,
            _req: std::marker::PhantomData,
        }
    }
}
