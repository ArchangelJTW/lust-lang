pub(super) use super::{Chunk, Function, Instruction, Register, Value};
pub(super) use crate::ast::{
    BinaryOp, ExprKind, ExternItem, Item, ItemKind, Literal, Span, Stmt, StmtKind, Type, TypeKind,
    UnaryOp,
};
use crate::config::LustConfig;
pub(super) use crate::number::LustInt;
pub(super) use crate::number::NumericType;
use crate::typechecker::FunctionSignature;
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
    pub(super) extern_value_aliases: HashMap<String, String>,
    pub(super) stdlib_symbols: HashSet<String>,
    option_coercions: HashMap<String, HashSet<Span>>,
    checked_array_indices: HashMap<String, HashSet<Span>>,
    numeric_types: HashMap<String, HashMap<Span, NumericType>>,
    function_signatures: HashMap<String, FunctionSignature>,
    minimal_runtime_types: bool,
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
            trait_names: ["ToString".to_string(), "HashKey".to_string()]
                .into_iter()
                .collect(),
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
            extern_value_aliases: HashMap::new(),
            stdlib_symbols: HashSet::new(),
            option_coercions: HashMap::new(),
            checked_array_indices: HashMap::new(),
            numeric_types: HashMap::new(),
            function_signatures: HashMap::new(),
            minimal_runtime_types: false,
        };
        compiler.configure_stdlib(&LustConfig::default());
        compiler
    }

    pub fn set_minimal_runtime_types(&mut self, enabled: bool) {
        self.minimal_runtime_types = enabled;
    }

    /// Apply all relevant config settings to the compiler
    pub fn configure(&mut self, config: &LustConfig) {
        self.minimal_runtime_types = config.minimal_runtime_types();
        self.configure_stdlib(config);
    }

    pub(super) fn new_function(
        &self,
        name: impl Into<String>,
        param_count: u8,
        is_method: bool,
    ) -> Function {
        if self.minimal_runtime_types {
            Function::new_minimal(name, param_count, is_method)
        } else {
            Function::new(name, param_count, is_method)
        }
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
            [
                "print",
                "println",
                "type",
                "tostring",
                "unpack",
                "select",
                "task",
                "lua",
                "array",
                "map",
                "math",
                "string",
                "error",
                "assert",
                "tonumber",
                "pairs",
                "ipairs",
                "setmetatable",
            ]
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

    pub fn set_option_coercions(&mut self, map: HashMap<String, HashSet<Span>>) {
        self.option_coercions = map;
    }

    pub fn set_checked_array_indices(&mut self, map: HashMap<String, HashSet<Span>>) {
        self.checked_array_indices = map;
    }

    pub fn set_numeric_types(&mut self, map: HashMap<String, HashMap<Span, NumericType>>) {
        self.numeric_types = map;
    }

    pub(super) fn numeric_type(&self, span: Span) -> Option<NumericType> {
        self.numeric_types
            .get(self.current_module.as_deref().unwrap_or(""))?
            .get(&span)
            .copied()
    }

    pub fn set_function_signatures(&mut self, signatures: HashMap<String, FunctionSignature>) {
        self.function_signatures = signatures;
    }

    pub fn take_function_signatures(&mut self) -> HashMap<String, FunctionSignature> {
        core::mem::take(&mut self.function_signatures)
    }

    pub(super) fn is_stdlib_symbol(&self, name: &str) -> bool {
        self.stdlib_symbols.contains(name)
    }

    pub(super) fn should_wrap_option(&self, span: Span) -> bool {
        let module = self.current_module.as_deref().unwrap_or("");
        self.option_coercions
            .get(module)
            .map_or(false, |set| set.contains(&span))
    }

    pub(super) fn is_checked_array_index(&self, span: Span) -> bool {
        let module = self.current_module.as_deref().unwrap_or("");
        self.checked_array_indices
            .get(module)
            .map_or(false, |set| set.contains(&span))
    }

    fn assign_signature_by_name(&mut self, func_idx: usize, name: &str) {
        if let Some(signature) = self.function_signatures.get(name).cloned() {
            self.functions[func_idx].set_signature(signature);
        }
    }

    fn try_set_lambda_signature(
        &mut self,
        func_idx: usize,
        params: &[(String, Option<Type>)],
        return_type: &Option<Type>,
    ) {
        if let Some(signature) = Self::lambda_signature(params, return_type) {
            self.functions[func_idx].set_signature(signature);
        }
    }

    fn lambda_signature(
        params: &[(String, Option<Type>)],
        return_type: &Option<Type>,
    ) -> Option<FunctionSignature> {
        let mut param_types = Vec::with_capacity(params.len());
        for (_, ty) in params {
            if let Some(ty) = ty {
                param_types.push(ty.clone());
            } else {
                return None;
            }
        }

        // An omitted lambda return annotation is inferred by the typechecker; it does not
        // declare unit. The compiler does not retain that per-expression inference here.
        let return_type = return_type
            .clone()
            .unwrap_or_else(|| Type::new(TypeKind::Unknown, Span::dummy()));
        Some(FunctionSignature {
            params: param_types,
            return_type,
            is_method: false,
            type_params: Vec::new(),
            trait_bounds: Vec::new(),
        })
    }

    pub fn record_extern_value(&mut self, name: &str) {
        let runtime_name = name.to_string();
        self.extern_value_aliases
            .entry(runtime_name.clone())
            .or_insert(runtime_name.clone());
        let module_name = self
            .current_module
            .clone()
            .or_else(|| self.entry_module.clone());
        if let Some(module) = module_name {
            if !name.contains('.') {
                let qualified = format!("{}.{}", module, name);
                self.extern_value_aliases
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
                    "function({}): {}",
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
                | "Array"
                | "Map"
                | "Option"
                | "Result"
                | "Iterator"
                | "Task"
                | "TaskStatus"
                | "TaskInfo"
                | "IndexError"
                | "LuaValue"
                | "LuaTable"
                | "LuaFunction"
                | "LuaUserdata"
                | "LuaThread"
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{intern::Interner, Lexer, Parser, TypeChecker, VM};

    fn compile_typed(source: &str, low_memory: bool) -> Vec<Function> {
        let mut interner = Interner::new();
        let tokens = Lexer::new(source, &mut interner).tokenize().unwrap();
        let items = Parser::new(tokens).parse().unwrap();
        let config = LustConfig::default()
            .with_low_memory_mode(low_memory)
            .with_minimal_runtime_types(low_memory);
        let mut checker = TypeChecker::with_config(&config);
        checker.check_module(&items).unwrap();
        let mut compiler = Compiler::new();
        compiler.configure(&config);
        compiler.set_numeric_types(checker.take_numeric_types());
        compiler.set_function_signatures(checker.take_function_signatures());
        compiler.compile_module(&items).unwrap()
    }

    #[test]
    fn numeric_lowering_uses_expression_types_in_both_memory_modes() {
        for low_memory in [false, true] {
            let functions = compile_typed(
                r#"
function integer(n: int): bool
    local value = n + 1 + 2
    value += 3
    return value < n * 4
end
function floating(n: float): float
    local value = n + 1.0
    value = -value
    return value * 2.0
end
function mixed(n: int): float
    return n + 1 + 0.5
end
function dynamic(n: unknown): unknown
    return n + 1
end
function generic<T>(n: T, other: T): bool
    return n == other
end
"#,
                low_memory,
            );
            let ops = |name: &str| {
                &functions
                    .iter()
                    .find(|f| f.name == name)
                    .unwrap()
                    .chunk
                    .instructions
            };
            assert!(ops("integer")
                .iter()
                .any(|op| matches!(op, Instruction::AddInt(d, l, _) if d == l)));
            assert!(ops("integer")
                .iter()
                .any(|op| matches!(op, Instruction::LtInt(..))));
            assert!(ops("integer")
                .iter()
                .any(|op| matches!(op, Instruction::MulInt(..))));
            assert!(!ops("integer")
                .iter()
                .any(|op| matches!(op, Instruction::Add(..))));
            assert!(ops("floating")
                .iter()
                .any(|op| matches!(op, Instruction::AddFloat(..))));
            assert!(ops("floating")
                .iter()
                .any(|op| matches!(op, Instruction::NegFloat(..))));
            assert!(ops("floating")
                .iter()
                .any(|op| matches!(op, Instruction::MulFloat(..))));
            assert!(ops("mixed")
                .iter()
                .any(|op| matches!(op, Instruction::AddInt(..))));
            assert!(ops("mixed")
                .iter()
                .any(|op| matches!(op, Instruction::Add(..))));
            assert!(ops("dynamic")
                .iter()
                .any(|op| matches!(op, Instruction::Add(..))));
            assert!(ops("generic")
                .iter()
                .any(|op| matches!(op, Instruction::Eq(..))));

            let mut vm = VM::new();
            vm.load_functions(functions);
            assert_eq!(
                vm.call("integer", vec![Value::Int(4)]).unwrap(),
                Value::Bool(true)
            );
            assert_eq!(
                vm.call("floating", vec![Value::Float(2.0)]).unwrap(),
                Value::Float(-6.0)
            );
            assert_eq!(
                vm.call("mixed", vec![Value::Int(2)]).unwrap(),
                Value::Float(3.5)
            );
            assert_eq!(
                vm.call("dynamic", vec![Value::Float(2.5)]).unwrap(),
                Value::Float(3.5)
            );
        }
    }

    #[test]
    fn typed_assignments_preserve_copies_and_reused_slots() {
        let functions = compile_typed(
            r#"
function copies(n: int): int
    local first = n + 1
    local second = first
    second += 2
    return first * 10 + second
end
function scopes(n: int): float
    local total = 0
    if n > 0 then
        local value: int = n + 1
        total = value
    end
    local value: float = 1.5
    value = value + 2.5
    return value
end
"#,
            false,
        );
        let mut vm = VM::new();
        vm.load_functions(functions);
        assert_eq!(
            vm.call("copies", vec![Value::Int(2)]).unwrap(),
            Value::Int(35)
        );
        for n in [0, 2] {
            assert_eq!(
                vm.call("scopes", vec![Value::Int(n)]).unwrap(),
                Value::Float(4.0)
            );
        }
    }

    #[test]
    fn numeric_result_fusion_does_not_cross_control_flow() {
        let mut compiler = Compiler::new();
        compiler.functions.push(Function::new("branch", 0, false));
        compiler.emit(Instruction::Jump(1), 1);
        compiler.emit(Instruction::AddInt(2, 0, 1), 1);
        compiler.move_result(0, 2, 0);
        assert_eq!(
            compiler.current_chunk().instructions.last(),
            Some(&Instruction::Move(0, 2))
        );
    }

    #[test]
    fn omitted_lambda_return_type_is_dynamic_at_runtime() {
        let signature = Compiler::lambda_signature(
            &[(
                "value".to_string(),
                Some(Type::new(TypeKind::Int, Span::dummy())),
            )],
            &None,
        )
        .unwrap();

        assert!(matches!(signature.return_type.kind, TypeKind::Unknown));
    }
}
