pub(super) use super::{Chunk, Function, Instruction, Register, Value};
pub(super) use crate::ast::{
    BinaryOp, ExprKind, ExternItem, Item, ItemKind, Literal, Stmt, StmtKind, UnaryOp,
};
use crate::config::LustConfig;
pub(super) use crate::number::LustInt;
pub(super) use crate::{Expr, LustError, Result};
pub(super) use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
pub(super) use hashbrown::{HashMap, HashSet};
mod closures;
mod expressions;
mod methods;
mod module;
mod patterns;
mod registers;
mod statements;
pub struct Compiler {
    pub(super) functions: Vec<Function>,
    pub(super) function_table: HashMap<String, usize>,
    pub(super) trait_impls: Vec<(String, String)>,
    pub(super) trait_names: HashSet<String>,
    pub(super) current_function: usize,
    pub(super) scopes: Vec<Scope>,
    pub(super) loop_contexts: Vec<LoopContext>,
    pub(super) next_register: Register,
    pub(super) max_register: Register,
    pub(super) current_line: usize,
    pub(super) imports_by_module: HashMap<String, crate::modules::ModuleImports>,
    pub(super) current_module: Option<String>,
    pub(super) entry_module: Option<String>,
    pub(super) module_locals: HashMap<String, HashSet<String>>,
    pub(super) current_function_name: Option<String>,
    pub(super) extern_function_aliases: HashMap<String, String>,
    pub(super) stdlib_symbols: HashSet<String>,
}

#[derive(Debug, Clone)]
pub(super) struct Scope {
    pub(super) locals: HashMap<String, (Register, bool)>,
    pub(super) depth: usize,
}

#[derive(Debug, Clone)]
pub(super) struct LoopContext {
    pub(super) continue_target: Option<usize>,
    pub(super) continue_jumps: Vec<usize>,
    pub(super) break_jumps: Vec<usize>,
}

impl Compiler {
    pub fn new() -> Self {
        let mut compiler = Self {
            functions: Vec::new(),
            function_table: HashMap::new(),
            trait_impls: Vec::new(),
            trait_names: HashSet::new(),
            current_function: 0,
            scopes: Vec::new(),
            loop_contexts: Vec::new(),
            next_register: 0,
            max_register: 0,
            current_line: 0,
            imports_by_module: HashMap::new(),
            current_module: None,
            entry_module: None,
            module_locals: HashMap::new(),
            current_function_name: None,
            extern_function_aliases: HashMap::new(),
            stdlib_symbols: HashSet::new(),
        };
        compiler.configure_stdlib(&LustConfig::default());
        compiler
    }

    pub fn set_imports_by_module(&mut self, map: HashMap<String, crate::modules::ModuleImports>) {
        self.imports_by_module = map;
    }

    pub fn set_entry_module(&mut self, module: impl Into<String>) {
        self.entry_module = Some(module.into());
    }

    pub fn get_trait_impls(&self) -> &[(String, String)] {
        &self.trait_impls
    }

    pub fn configure_stdlib(&mut self, config: &LustConfig) {
        self.stdlib_symbols.clear();
        self.stdlib_symbols.extend(
            ["print", "println", "type", "tostring", "task"]
                .into_iter()
                .map(String::from),
        );
        for module in config.enabled_modules() {
            match module {
                "io" | "os" => {
                    self.stdlib_symbols.insert(module.to_string());
                }

                _ => {}
            }
        }
    }

    pub(super) fn is_stdlib_symbol(&self, name: &str) -> bool {
        self.stdlib_symbols.contains(name)
    }

    pub(super) fn record_extern_function(&mut self, name: &str) {
        let runtime_name = name.to_string();
        self.extern_function_aliases
            .entry(runtime_name.clone())
            .or_insert(runtime_name.clone());
        let module_name = self
            .current_module
            .clone()
            .or_else(|| self.entry_module.clone());
        if let Some(module) = module_name {
            if !name.contains('.') {
                let qualified = format!("{}.{}", module, name);
                self.extern_function_aliases
                    .entry(qualified)
                    .or_insert(runtime_name);
            }
        }
    }

    pub(super) fn describe_expr_kind(kind: &ExprKind) -> &'static str {
        match kind {
            ExprKind::Literal(_) => "literal expression",
            ExprKind::Identifier(_) => "identifier expression",
            ExprKind::Binary { .. } => "binary expression",
            ExprKind::Unary { .. } => "unary expression",
            ExprKind::Call { .. } => "function call",
            ExprKind::MethodCall { .. } => "method call",
            ExprKind::FieldAccess { .. } => "field access",
            ExprKind::Index { .. } => "index access",
            ExprKind::Array(_) => "array literal",
            ExprKind::Map(_) => "map literal",
            ExprKind::Tuple(_) => "tuple literal",
            ExprKind::StructLiteral { .. } => "struct literal",
            ExprKind::EnumConstructor { .. } => "enum constructor",
            ExprKind::Lambda { .. } => "lambda expression",
            ExprKind::Paren(_) => "parenthesized expression",
            ExprKind::Cast { .. } => "cast expression",
            ExprKind::TypeCheck { .. } => "`is` type check",
            ExprKind::IsPattern { .. } => "`is` pattern expression",
            ExprKind::If { .. } => "`if` expression",
            ExprKind::Block(_) => "block expression",
            ExprKind::Return(_) => "return expression",
            ExprKind::Range { .. } => "range expression",
        }
    }

    pub(super) fn type_to_string(type_kind: &crate::ast::TypeKind) -> String {
        use crate::ast::TypeKind;
        match type_kind {
            TypeKind::Int => "int".to_string(),
            TypeKind::Float => "float".to_string(),
            TypeKind::String => "string".to_string(),
            TypeKind::Bool => "bool".to_string(),
            TypeKind::Named(name) => name.clone(),
            TypeKind::Array(inner) => format!("Array<{}>", Self::type_to_string(&inner.kind)),
            TypeKind::Map(key, val) => format!(
                "Map<{}, {}>",
                Self::type_to_string(&key.kind),
                Self::type_to_string(&val.kind)
            ),
            TypeKind::Table => "Table".to_string(),
            TypeKind::Option(inner) => format!("Option<{}>", Self::type_to_string(&inner.kind)),
            TypeKind::Result(ok, err) => format!(
                "Result<{}, {}>",
                Self::type_to_string(&ok.kind),
                Self::type_to_string(&err.kind)
            ),
            TypeKind::Function {
                params,
                return_type,
            } => {
                let param_strs: Vec<String> = params
                    .iter()
                    .map(|p| Self::type_to_string(&p.kind))
                    .collect();
                format!(
                    "function({}) -> {}",
                    param_strs.join(", "),
                    Self::type_to_string(&return_type.kind)
                )
            }

            TypeKind::Tuple(elements) => {
                let element_strs: Vec<String> = elements
                    .iter()
                    .map(|t| Self::type_to_string(&t.kind))
                    .collect();
                format!("Tuple<{}>", element_strs.join(", "))
            }

            TypeKind::Generic(name) => name.clone(),
            TypeKind::GenericInstance { name, type_args } => {
                let arg_strs: Vec<String> = type_args
                    .iter()
                    .map(|t| Self::type_to_string(&t.kind))
                    .collect();
                format!("{}<{}>", name, arg_strs.join(", "))
            }

            TypeKind::Unknown => "unknown".to_string(),
            TypeKind::Union(types) => {
                let type_strs: Vec<String> = types
                    .iter()
                    .map(|t| Self::type_to_string(&t.kind))
                    .collect();
                format!("{}", type_strs.join(" | "))
            }

            TypeKind::Unit => "()".to_string(),
            TypeKind::Infer => "_".to_string(),
            TypeKind::Ref(inner) => format!("&{}", Self::type_to_string(&inner.kind)),
            TypeKind::MutRef(inner) => format!("&mut {}", Self::type_to_string(&inner.kind)),
            TypeKind::Pointer { mutable, pointee } => {
                if *mutable {
                    format!("*mut {}", Self::type_to_string(&pointee.kind))
                } else {
                    format!("*{}", Self::type_to_string(&pointee.kind))
                }
            }

            TypeKind::Trait(name) => name.clone(),
            TypeKind::TraitBound(traits) => traits.join(" + "),
        }
    }

    fn module_context_name(&self) -> Option<&str> {
        self.current_module
            .as_deref()
            .or_else(|| self.entry_module.as_deref())
    }

    fn is_builtin_type_name(name: &str) -> bool {
        matches!(
            name,
            "int"
                | "float"
                | "string"
                | "bool"
                | "unknown"
                | "Table"
                | "Array"
                | "Map"
                | "Option"
                | "Result"
                | "Iterator"
                | "Task"
                | "TaskStatus"
                | "TaskInfo"
        )
    }

    pub(super) fn resolve_type_name(&self, name: &str) -> String {
        if let Some((head, tail)) = name.split_once('.') {
            if let Some(module) = self.module_context_name() {
                if let Some(imports) = self.imports_by_module.get(module) {
                    if let Some(real_module) = imports.module_aliases.get(head) {
                        if tail.is_empty() {
                            return real_module.clone();
                        } else {
                            return format!("{}.{}", real_module, tail);
                        }
                    }
                }
            }

            return name.to_string();
        }

        if Self::is_builtin_type_name(name) {
            return name.to_string();
        }

        if let Some(module) = self.module_context_name() {
            if let Some(imports) = self.imports_by_module.get(module) {
                if let Some(fq) = imports.type_aliases.get(name) {
                    return fq.clone();
                }
            }

            return format!("{}.{}", module, name);
        }

        name.to_string()
    }
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}
