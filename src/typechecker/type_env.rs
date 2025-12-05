use crate::{
    ast::*,
    builtins::{self, BuiltinFunction},
    config::LustConfig,
    error::{LustError, Result},
};
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::fmt;
use hashbrown::{HashMap, HashSet};
pub struct TypeEnv {
    scopes: Vec<HashMap<String, Type>>,
    refinements: Vec<HashMap<String, Type>>,
    generic_instances: HashMap<String, HashMap<String, Type>>,
    functions: HashMap<String, FunctionSignature>,
    structs: HashMap<String, StructDef>,
    enums: HashMap<String, EnumDef>,
    traits: HashMap<String, TraitDef>,
    type_aliases: HashMap<String, (Vec<String>, Type)>,
    impls: Vec<ImplBlock>,
    builtin_types: HashSet<String>,
}

#[derive(Debug, Clone)]
pub struct FunctionSignature {
    pub params: Vec<Type>,
    pub return_type: Type,
    pub is_method: bool,
}

impl fmt::Display for FunctionSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let params = self
            .params
            .iter()
            .map(|param| param.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "function({}) -> {}", params, self.return_type)
    }
}

impl TypeEnv {
    pub fn new() -> Self {
        Self::with_config(&LustConfig::default())
    }

    fn register_builtin_function_slice(&mut self, functions: &[BuiltinFunction], span: Span) {
        for builtin in functions {
            self.functions
                .insert(builtin.name.to_string(), builtin.to_signature(span));
        }
    }

    pub fn with_config(config: &LustConfig) -> Self {
        let mut env = Self {
            scopes: vec![HashMap::new()],
            refinements: vec![HashMap::new()],
            generic_instances: HashMap::new(),
            functions: HashMap::new(),
            structs: HashMap::new(),
            enums: HashMap::new(),
            traits: HashMap::new(),
            type_aliases: HashMap::new(),
            impls: Vec::new(),
            builtin_types: HashSet::new(),
        };
        env.register_builtins(config);
        env
    }

    fn register_builtins(&mut self, config: &LustConfig) {
        let dummy_span = Span::new(0, 0, 0, 0);
        self.register_builtin_type("Task");
        self.register_builtin_type("TaskStatus");
        self.register_builtin_type("TaskInfo");
        let task_status_type = Type::new(TypeKind::Named("TaskStatus".to_string()), dummy_span);
        let unknown_type = Type::new(TypeKind::Unknown, dummy_span);
        let option_unknown_type =
            Type::new(TypeKind::Option(Box::new(unknown_type.clone())), dummy_span);
        let string_type = Type::new(TypeKind::String, dummy_span);
        let option_string_type =
            Type::new(TypeKind::Option(Box::new(string_type.clone())), dummy_span);
        let task_info_struct = StructDef {
            name: "TaskInfo".to_string(),
            type_params: vec![],
            trait_bounds: vec![],
            fields: vec![
                StructField {
                    name: "state".to_string(),
                    ty: task_status_type.clone(),
                    visibility: Visibility::Public,
                    ownership: FieldOwnership::Strong,
                    weak_target: None,
                },
                StructField {
                    name: "last_yield".to_string(),
                    ty: option_unknown_type.clone(),
                    visibility: Visibility::Public,
                    ownership: FieldOwnership::Strong,
                    weak_target: None,
                },
                StructField {
                    name: "last_result".to_string(),
                    ty: option_unknown_type.clone(),
                    visibility: Visibility::Public,
                    ownership: FieldOwnership::Strong,
                    weak_target: None,
                },
                StructField {
                    name: "error".to_string(),
                    ty: option_string_type.clone(),
                    visibility: Visibility::Public,
                    ownership: FieldOwnership::Strong,
                    weak_target: None,
                },
            ],
            visibility: Visibility::Public,
        };
        self.structs
            .insert("TaskInfo".to_string(), task_info_struct);
        self.register_builtin_function_slice(builtins::base_functions(), dummy_span);
        self.register_builtin_function_slice(builtins::task_functions(), dummy_span);
        if let Some(global_scope) = self.scopes.first_mut() {
            global_scope.insert("task".to_string(), Type::new(TypeKind::Unknown, dummy_span));
        }

        let task_status_enum = EnumDef {
            name: "TaskStatus".to_string(),
            type_params: vec![],
            trait_bounds: vec![],
            variants: vec![
                EnumVariant {
                    name: "Ready".to_string(),
                    fields: None,
                },
                EnumVariant {
                    name: "Running".to_string(),
                    fields: None,
                },
                EnumVariant {
                    name: "Yielded".to_string(),
                    fields: None,
                },
                EnumVariant {
                    name: "Completed".to_string(),
                    fields: None,
                },
                EnumVariant {
                    name: "Failed".to_string(),
                    fields: None,
                },
                EnumVariant {
                    name: "Stopped".to_string(),
                    fields: None,
                },
            ],
            visibility: Visibility::Public,
        };
        self.enums
            .insert("TaskStatus".to_string(), task_status_enum);
        if config.is_module_enabled("io") {
            if let Some(global_scope) = self.scopes.first_mut() {
                global_scope.insert("io".to_string(), Type::new(TypeKind::Unknown, dummy_span));
            }

            self.register_builtin_function_slice(builtins::io_functions(), dummy_span);
        }

        if config.is_module_enabled("os") {
            if let Some(global_scope) = self.scopes.first_mut() {
                global_scope.insert("os".to_string(), Type::new(TypeKind::Unknown, dummy_span));
            }

            self.register_builtin_function_slice(builtins::os_functions(), dummy_span);
        }

        let option_enum = EnumDef {
            name: "Option".to_string(),
            type_params: vec!["T".to_string()],
            trait_bounds: vec![],
            variants: vec![
                EnumVariant {
                    name: "Some".to_string(),
                    fields: Some(vec![Type::new(
                        TypeKind::Generic("T".to_string()),
                        dummy_span,
                    )]),
                },
                EnumVariant {
                    name: "None".to_string(),
                    fields: None,
                },
            ],
            visibility: Visibility::Public,
        };
        self.enums.insert("Option".to_string(), option_enum);
        let result_enum = EnumDef {
            name: "Result".to_string(),
            type_params: vec!["T".to_string(), "E".to_string()],
            trait_bounds: vec![],
            variants: vec![
                EnumVariant {
                    name: "Ok".to_string(),
                    fields: Some(vec![Type::new(
                        TypeKind::Generic("T".to_string()),
                        dummy_span,
                    )]),
                },
                EnumVariant {
                    name: "Err".to_string(),
                    fields: Some(vec![Type::new(
                        TypeKind::Generic("E".to_string()),
                        dummy_span,
                    )]),
                },
            ],
            visibility: Visibility::Public,
        };
        self.enums.insert("Result".to_string(), result_enum);
        let to_string_trait = TraitDef {
            name: "ToString".to_string(),
            type_params: vec![],
            methods: vec![TraitMethod {
                name: "to_string".to_string(),
                type_params: vec![],
                params: vec![FunctionParam {
                    name: "self".to_string(),
                    ty: Type::new(TypeKind::Unknown, dummy_span),
                    is_self: true,
                }],
                return_type: Some(Type::new(TypeKind::String, dummy_span)),
                default_impl: None,
            }],
            visibility: Visibility::Public,
        };
        self.traits.insert("ToString".to_string(), to_string_trait);
        let hash_key_trait = TraitDef {
            name: "HashKey".to_string(),
            type_params: vec![],
            methods: vec![TraitMethod {
                name: "to_hashkey".to_string(),
                type_params: vec![],
                params: vec![FunctionParam {
                    name: "self".to_string(),
                    ty: Type::new(TypeKind::Unknown, dummy_span),
                    is_self: true,
                }],
                return_type: Some(Type::new(TypeKind::Unknown, dummy_span)),
                default_impl: None,
            }],
            visibility: Visibility::Public,
        };
        self.traits.insert("HashKey".to_string(), hash_key_trait);
        let int_to_string_impl = ImplBlock {
            type_params: vec![],
            trait_name: Some("ToString".to_string()),
            target_type: Type::new(TypeKind::Int, dummy_span),
            methods: vec![],
            where_clause: vec![],
        };
        self.impls.push(int_to_string_impl);
        let float_to_string_impl = ImplBlock {
            type_params: vec![],
            trait_name: Some("ToString".to_string()),
            target_type: Type::new(TypeKind::Float, dummy_span),
            methods: vec![],
            where_clause: vec![],
        };
        self.impls.push(float_to_string_impl);
        let bool_to_string_impl = ImplBlock {
            type_params: vec![],
            trait_name: Some("ToString".to_string()),
            target_type: Type::new(TypeKind::Bool, dummy_span),
            methods: vec![],
            where_clause: vec![],
        };
        self.impls.push(bool_to_string_impl);
        let string_to_string_impl = ImplBlock {
            type_params: vec![],
            trait_name: Some("ToString".to_string()),
            target_type: Type::new(TypeKind::String, dummy_span),
            methods: vec![],
            where_clause: vec![],
        };
        self.impls.push(string_to_string_impl);
    }

    fn register_builtin_type(&mut self, name: &str) {
        self.builtin_types.insert(name.to_string());
    }

    pub fn is_builtin_type(&self, name: &str) -> bool {
        self.builtin_types.contains(name)
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
        self.refinements.push(HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        self.scopes.pop();
        self.refinements.pop();
    }

    pub fn declare_variable(&mut self, name: String, ty: Type) -> Result<()> {
        let scope = self
            .scopes
            .last_mut()
            .expect("Type environment has no scope");
        scope.insert(name, ty);
        Ok(())
    }

    pub fn lookup_variable(&self, name: &str) -> Option<Type> {
        for refinement_scope in self.refinements.iter().rev() {
            if let Some(ty) = refinement_scope.get(name) {
                return Some(ty.clone());
            }
        }

        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(ty.clone());
            }
        }

        None
    }

    pub fn refine_variable_type(&mut self, name: String, refined_type: Type) {
        if let Some(refinement_scope) = self.refinements.last_mut() {
            refinement_scope.insert(name, refined_type);
        }
    }

    pub fn record_generic_instance(
        &mut self,
        var_name: String,
        type_param: String,
        concrete_type: Type,
    ) {
        self.generic_instances
            .entry(var_name)
            .or_insert_with(HashMap::new)
            .insert(type_param, concrete_type);
    }

    pub fn lookup_generic_param(&self, var_name: &str, type_param: &str) -> Option<Type> {
        self.generic_instances
            .get(var_name)?
            .get(type_param)
            .cloned()
    }

    pub fn register_function(&mut self, name: String, sig: FunctionSignature) -> Result<()> {
        if self.functions.contains_key(&name) {
            return Err(LustError::TypeError {
                message: format!("Function '{}' is already defined", name),
            });
        }

        self.functions.insert(name, sig);
        Ok(())
    }

    pub fn register_or_update_function(
        &mut self,
        name: String,
        sig: FunctionSignature,
    ) -> Result<()> {
        if let Some(existing) = self.functions.get_mut(&name) {
            if existing.params != sig.params || existing.return_type != sig.return_type {
                return Err(LustError::TypeError {
                    message: format!(
                        "Function '{}' is already defined with a different signature",
                        name
                    ),
                });
            }

            if sig.is_method && !existing.is_method {
                existing.is_method = true;
            }
            return Ok(());
        }

        self.functions.insert(name, sig);
        Ok(())
    }

    pub fn lookup_function(&self, name: &str) -> Option<&FunctionSignature> {
        self.functions.get(name)
    }

    pub fn function_signatures(&self) -> HashMap<String, FunctionSignature> {
        self.functions.clone()
    }

    pub fn struct_definitions(&self) -> HashMap<String, StructDef> {
        self.structs.clone()
    }

    pub fn enum_definitions(&self) -> HashMap<String, EnumDef> {
        self.enums.clone()
    }

    pub fn register_struct(&mut self, s: &StructDef) -> Result<()> {
        if self.structs.contains_key(&s.name) {
            return Err(LustError::TypeError {
                message: format!("Struct '{}' is already defined", s.name),
            });
        }

        self.structs.insert(s.name.clone(), s.clone());
        Ok(())
    }

    pub fn lookup_struct(&self, name: &str) -> Option<&StructDef> {
        self.structs.get(name)
    }

    pub fn register_enum(&mut self, e: &EnumDef) -> Result<()> {
        if self.enums.contains_key(&e.name) {
            return Err(LustError::TypeError {
                message: format!("Enum '{}' is already defined", e.name),
            });
        }

        self.enums.insert(e.name.clone(), e.clone());
        Ok(())
    }

    pub fn lookup_enum(&self, name: &str) -> Option<&EnumDef> {
        self.enums.get(name)
    }

    pub fn register_trait(&mut self, t: &TraitDef) -> Result<()> {
        if self.traits.contains_key(&t.name) {
            return Err(LustError::TypeError {
                message: format!("Trait '{}' is already defined", t.name),
            });
        }

        self.traits.insert(t.name.clone(), t.clone());
        Ok(())
    }

    pub fn lookup_trait(&self, name: &str) -> Option<&TraitDef> {
        self.traits.get(name)
    }

    pub fn register_type_alias(
        &mut self,
        name: String,
        type_params: Vec<String>,
        target: Type,
    ) -> Result<()> {
        if self.type_aliases.contains_key(&name) {
            return Err(LustError::TypeError {
                message: format!("Type alias '{}' is already defined", name),
            });
        }

        self.type_aliases.insert(name, (type_params, target));
        Ok(())
    }

    pub fn register_impl(&mut self, impl_block: &ImplBlock) {
        self.impls.push(impl_block.clone());
    }

    pub fn lookup_method(&self, type_name: &str, method_name: &str) -> Option<&FunctionDef> {
        for impl_block in &self.impls {
            if let TypeKind::Named(name) = &impl_block.target_type.kind {
                if name == type_name {
                    for method in &impl_block.methods {
                        if method.name.ends_with(&format!(":{}", method_name))
                            || method.name == method_name
                        {
                            return Some(method);
                        }
                    }
                }
            }
        }

        None
    }

    pub fn lookup_struct_field(&self, struct_name: &str, field_name: &str) -> Option<Type> {
        let struct_def = self.lookup_struct(struct_name)?;
        for field in &struct_def.fields {
            if field.name == field_name {
                return Some(field.ty.clone());
            }
        }

        None
    }

    pub fn type_implements_trait(&self, ty: &Type, trait_name: &str) -> bool {
        for impl_block in &self.impls {
            if let Some(impl_trait_name) = &impl_block.trait_name {
                if impl_trait_name == trait_name {
                    if self.types_match(&impl_block.target_type, ty) {
                        return true;
                    }
                }
            }
        }

        false
    }

    fn types_match(&self, type1: &Type, type2: &Type) -> bool {
        match (&type1.kind, &type2.kind) {
            (TypeKind::Int, TypeKind::Int) => true,
            (TypeKind::Float, TypeKind::Float) => true,
            (TypeKind::Bool, TypeKind::Bool) => true,
            (TypeKind::String, TypeKind::String) => true,
            (TypeKind::Named(n1), TypeKind::Named(n2)) => n1 == n2,
            (TypeKind::Array(t1), TypeKind::Array(t2)) => self.types_match(t1, t2),
            (TypeKind::Map(k1, v1), TypeKind::Map(k2, v2)) => {
                self.types_match(k1, k2) && self.types_match(v1, v2)
            }

            (TypeKind::Option(t1), TypeKind::Option(t2)) => self.types_match(t1, t2),
            (TypeKind::Result(ok1, err1), TypeKind::Result(ok2, err2)) => {
                self.types_match(ok1, ok2) && self.types_match(err1, err2)
            }

            _ => false,
        }
    }
}
