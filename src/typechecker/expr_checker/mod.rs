pub(super) use super::TypeChecker;
pub(super) use crate::typechecker::type_env;
pub(super) use crate::{ast::*, error::Result};
mod collections;
mod entry;
mod operations;
mod patterns;
