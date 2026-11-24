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
    impls: Vec<ImplBlock>,
    builtin_types: HashSet<String>,
    constants: HashMap<String, Type>,
    services: HashMap<String, Option<Type>>,
    type_aliases: HashMap<String, Type>,
}

#[derive(Debug, Clone)]
pub struct FunctionSignature {
    pub params: Vec<Type>,
    pub return_type: Type,
    pub is_method: bool,
    pub type_params: Vec<String>,
    pub trait_bounds: Vec<TraitBound>,
    pub effects: Option<crate::ast::ReturnSpec>,
}

impl FunctionSignature {
    pub fn return_spec(&self) -> crate::ast::ReturnSpec {
        if let Some(effects) = &self.effects {
            effects.clone()
        } else {
            crate::ast::ReturnSpec {
                return_type: Some(self.return_type.clone()),
                error_type: None,
                requirements: Vec::new(),
            }
        }
    }

    pub fn success_type(&self) -> Type {
        if let Some(effects) = &self.effects {
            effects.return_type.clone().unwrap_or_else(|| Type::new(TypeKind::Unit, Span::dummy()))
        } else {
            self.return_type.clone()
        }
    }

    pub fn error_type(&self) -> Option<Type> {
        self.return_spec().error_type
    }

    pub fn requirements(&self) -> Vec<Type> {
        self.return_spec().requirements
    }
}

impl fmt::Display for FunctionSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let params = self
            .params
            .iter()
            .map(|param| param.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "function({}): {}", params, self.return_type)
    }
}

impl Default for TypeEnv {
    fn default() -> Self {
        Self::new()
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
            impls: Vec::new(),
            builtin_types: HashSet::new(),
            constants: HashMap::new(),
            services: HashMap::new(),
            type_aliases: HashMap::new(),
        };
        env.register_builtins(config);
        env
    }

    fn register_builtins(&mut self, config: &LustConfig) {
        let dummy_span = Span::new(0, 0, 0, 0);
        self.register_builtin_type("Task");
        self.register_builtin_type("TaskStatus");
        self.register_builtin_type("TaskInfo");
        self.register_builtin_type("IndexError");
        self.register_builtin_type("Iterator");
        self.register_builtin_type("LuaValue");
        self.register_builtin_type("LuaTable");
        self.register_builtin_type("LuaFunction");
        self.register_builtin_type("LuaUserdata");
        self.register_builtin_type("LuaThread");
        self.register_builtin_type("Array");
        self.register_builtin_type("Map");
        self.register_builtin_type("Table");
        self.register_builtin_type("LuaTable");
        self.register_builtin_type("std.table.Table");
        self.register_builtin_type("std.table.LuaTable");
        self.register_builtin_type("table.Table");
        self.register_builtin_type("table.LuaTable");
        self.register_builtin_type("Regex");
        self.register_builtin_type("Schema");
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
            doc: None,
        };
        self.structs
            .insert("TaskInfo".to_string(), task_info_struct);
        self.structs.insert(
            "IndexError".to_string(),
            StructDef {
                name: "IndexError".to_string(),
                type_params: vec![],
                trait_bounds: vec![],
                fields: vec![
                    StructField {
                        name: "index".to_string(),
                        ty: Type::new(TypeKind::Int, dummy_span),
                        visibility: Visibility::Public,
                        ownership: FieldOwnership::Strong,
                        weak_target: None,
                    },
                    StructField {
                        name: "length".to_string(),
                        ty: Type::new(TypeKind::Int, dummy_span),
                        visibility: Visibility::Public,
                        ownership: FieldOwnership::Strong,
                        weak_target: None,
                    },
                ],
                visibility: Visibility::Public,
                doc: None,
            },
        );
        self.structs.insert(
            "LuaTable".to_string(),
            StructDef {
                name: "LuaTable".to_string(),
                type_params: vec![],
                trait_bounds: vec![],
                fields: vec![
                    StructField {
                        name: "table".to_string(),
                        ty: Type::new(
                            TypeKind::Map(
                                Box::new(Type::new(
                                    TypeKind::Named("LuaValue".to_string()),
                                    dummy_span,
                                )),
                                Box::new(Type::new(
                                    TypeKind::Named("LuaValue".to_string()),
                                    dummy_span,
                                )),
                            ),
                            dummy_span,
                        ),
                        visibility: Visibility::Public,
                        ownership: FieldOwnership::Strong,
                        weak_target: None,
                    },
                    StructField {
                        name: "metamethods".to_string(),
                        ty: Type::new(
                            TypeKind::Map(
                                Box::new(Type::new(TypeKind::String, dummy_span)),
                                Box::new(Type::new(
                                    TypeKind::Named("LuaValue".to_string()),
                                    dummy_span,
                                )),
                            ),
                            dummy_span,
                        ),
                        visibility: Visibility::Public,
                        ownership: FieldOwnership::Strong,
                        weak_target: None,
                    },
                ],
                visibility: Visibility::Public,
                doc: None,
            },
        );
        for name in ["LuaFunction", "LuaThread"] {
            self.structs.insert(
                name.to_string(),
                StructDef {
                    name: name.to_string(),
                    type_params: vec![],
                    trait_bounds: vec![],
                    fields: vec![StructField {
                        name: "handle".to_string(),
                        ty: Type::new(TypeKind::Int, dummy_span),
                        visibility: Visibility::Public,
                        ownership: FieldOwnership::Strong,
                        weak_target: None,
                    }],
                    visibility: Visibility::Public,
                    doc: None,
                },
            );
        }
        for name in ["Table", "std.table.Table", "table.Table"] {
            self.structs.insert(
                name.to_string(),
                StructDef {
                    name: name.to_string(),
                    type_params: vec!["T".to_string()],
                    trait_bounds: vec![],
                    fields: vec![],
                    visibility: Visibility::Public,
                    doc: None,
                },
            );
        }
        for name in ["std.table.LuaTable", "table.LuaTable"] {
            if let Some(lua_table_def) = self.structs.get("LuaTable").cloned() {
                self.structs.insert(name.to_string(), lua_table_def);
            }
        }
        self.structs.insert(
            "LuaUserdata".to_string(),
            StructDef {
                name: "LuaUserdata".to_string(),
                type_params: vec![],
                trait_bounds: vec![],
                fields: vec![
                    StructField {
                        name: "handle".to_string(),
                        ty: Type::new(TypeKind::Int, dummy_span),
                        visibility: Visibility::Public,
                        ownership: FieldOwnership::Strong,
                        weak_target: None,
                    },
                    StructField {
                        name: "ptr".to_string(),
                        ty: Type::new(TypeKind::Int, dummy_span),
                        visibility: Visibility::Public,
                        ownership: FieldOwnership::Strong,
                        weak_target: None,
                    },
                    StructField {
                        name: "state".to_string(),
                        ty: Type::new(TypeKind::Int, dummy_span),
                        visibility: Visibility::Public,
                        ownership: FieldOwnership::Strong,
                        weak_target: None,
                    },
                    StructField {
                        name: "metamethods".to_string(),
                        ty: Type::new(
                            TypeKind::Map(
                                Box::new(Type::new(TypeKind::String, dummy_span)),
                                Box::new(Type::new(
                                    TypeKind::Named("LuaValue".to_string()),
                                    dummy_span,
                                )),
                            ),
                            dummy_span,
                        ),
                        visibility: Visibility::Public,
                        ownership: FieldOwnership::Strong,
                        weak_target: None,
                    },
                ],
                visibility: Visibility::Public,
                doc: None,
            },
        );
        match config.prelude_mode() {
            crate::config::PreludeMode::Pure => {
                // In Pure mode, base functions and ambient global modules are not registered into global scope.
            }
            crate::config::PreludeMode::Custom { auto_globals, .. } => {
                for builtin in builtins::base_functions() {
                    if auto_globals.contains(builtin.name) {
                        self.functions
                            .insert(builtin.name.to_string(), builtin.to_signature(dummy_span));
                    }
                }
                if let Some(global_scope) = self.scopes.first_mut() {
                    for mod_name in ["array", "table", "map", "math", "string", "task", "lua", "rx", "view"] {
                        if auto_globals.contains(mod_name) {
                            global_scope.insert(mod_name.to_string(), Type::new(TypeKind::Unknown, dummy_span));
                        }
                    }
                }
            }
            crate::config::PreludeMode::Standard => {
                self.register_builtin_function_slice(builtins::base_functions(), dummy_span);
                if let Some(global_scope) = self.scopes.first_mut() {
                    global_scope.insert(
                        "array".to_string(),
                        Type::new(TypeKind::Unknown, dummy_span),
                    );
                    global_scope.insert(
                        "table".to_string(),
                        Type::new(TypeKind::Unknown, dummy_span),
                    );
                    global_scope.insert("map".to_string(), Type::new(TypeKind::Unknown, dummy_span));
                    global_scope.insert("math".to_string(), Type::new(TypeKind::Unknown, dummy_span));
                    global_scope.insert(
                        "string".to_string(),
                        Type::new(TypeKind::Unknown, dummy_span),
                    );
                    global_scope.insert("task".to_string(), Type::new(TypeKind::Unknown, dummy_span));
                    global_scope.insert("lua".to_string(), Type::new(TypeKind::Unknown, dummy_span));
                    global_scope.insert("rx".to_string(), Type::new(TypeKind::Unknown, dummy_span));
                    global_scope.insert("view".to_string(), Type::new(TypeKind::Unknown, dummy_span));
                }
            }
        }
        self.register_builtin_function_slice(builtins::array_functions(), dummy_span);
        self.register_builtin_function_slice(builtins::table_functions(), dummy_span);
        self.register_builtin_function_slice(builtins::map_functions(), dummy_span);
        self.register_builtin_function_slice(builtins::math_functions(), dummy_span);
        self.register_builtin_function_slice(builtins::string_functions(), dummy_span);
        self.register_builtin_function_slice(builtins::task_functions(), dummy_span);
        self.register_builtin_function_slice(builtins::lua_functions(), dummy_span);
        self.register_builtin_function_slice(builtins::rx_functions(), dummy_span);
        self.register_builtin_function_slice(builtins::test_functions(), dummy_span);

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
            doc: None,
        };
        self.enums
            .insert("TaskStatus".to_string(), task_status_enum);
        let lua_value_enum = EnumDef {
            name: "LuaValue".to_string(),
            type_params: vec![],
            trait_bounds: vec![],
            variants: vec![
                EnumVariant {
                    name: "Nil".to_string(),
                    fields: None,
                },
                EnumVariant {
                    name: "Bool".to_string(),
                    fields: Some(vec![Type::new(TypeKind::Bool, dummy_span)]),
                },
                EnumVariant {
                    name: "Int".to_string(),
                    fields: Some(vec![Type::new(TypeKind::Int, dummy_span)]),
                },
                EnumVariant {
                    name: "Float".to_string(),
                    fields: Some(vec![Type::new(TypeKind::Float, dummy_span)]),
                },
                EnumVariant {
                    name: "String".to_string(),
                    fields: Some(vec![Type::new(TypeKind::String, dummy_span)]),
                },
                EnumVariant {
                    name: "Table".to_string(),
                    fields: Some(vec![Type::new(
                        TypeKind::Named("LuaTable".to_string()),
                        dummy_span,
                    )]),
                },
                EnumVariant {
                    name: "Function".to_string(),
                    fields: Some(vec![Type::new(
                        TypeKind::Named("LuaFunction".to_string()),
                        dummy_span,
                    )]),
                },
                EnumVariant {
                    name: "Userdata".to_string(),
                    fields: Some(vec![Type::new(
                        TypeKind::Named("LuaUserdata".to_string()),
                        dummy_span,
                    )]),
                },
                EnumVariant {
                    name: "Thread".to_string(),
                    fields: Some(vec![Type::new(
                        TypeKind::Named("LuaThread".to_string()),
                        dummy_span,
                    )]),
                },
                EnumVariant {
                    name: "LightUserdata".to_string(),
                    fields: Some(vec![Type::new(TypeKind::Unknown, dummy_span)]),
                },
            ],
            visibility: Visibility::Public,
            doc: None,
        };
        self.enums.insert("LuaValue".to_string(), lua_value_enum);
        if config.prelude_mode() == crate::config::PreludeMode::Standard {
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
        } else {
            self.register_builtin_function_slice(builtins::io_functions(), dummy_span);
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
            doc: None,
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
            doc: None,
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
                doc: None,
            }],
            visibility: Visibility::Public,
            doc: None,
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
                doc: None,
            }],
            visibility: Visibility::Public,
            doc: None,
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
        let luavalue_to_string_impl = ImplBlock {
            type_params: vec![],
            trait_name: Some("ToString".to_string()),
            target_type: Type::new(TypeKind::Named("LuaValue".to_string()), dummy_span),
            methods: vec![],
            where_clause: vec![],
        };
        self.impls.push(luavalue_to_string_impl);
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

    pub fn resolve_type(&self, ty: &Type) -> Type {
        match &ty.kind {
            TypeKind::GenericInstance { name, type_args }
                if name == "Table" || name == "std.table.Table" || name == "table.Table" =>
            {
                match type_args.len() {
                    1 => Type::new(
                        TypeKind::Array(Box::new(self.resolve_type(&type_args[0]))),
                        ty.span,
                    ),
                    2 => Type::new(
                        TypeKind::Map(
                            Box::new(self.resolve_type(&type_args[0])),
                            Box::new(self.resolve_type(&type_args[1])),
                        ),
                        ty.span,
                    ),
                    _ => ty.clone(),
                }
            }
            TypeKind::Named(name)
                if name == "Table" || name == "std.table.Table" || name == "table.Table" =>
            {
                Type::new(
                    TypeKind::Array(Box::new(Type::new(TypeKind::Unknown, ty.span))),
                    ty.span,
                )
            }
            TypeKind::Named(name)
                if name == "std.table.LuaTable" || name == "table.LuaTable" =>
            {
                Type::new(TypeKind::Named("LuaTable".to_string()), ty.span)
            }
            TypeKind::Array(inner) => {
                Type::new(TypeKind::Array(Box::new(self.resolve_type(inner))), ty.span)
            }
            TypeKind::Map(k, v) => Type::new(
                TypeKind::Map(
                    Box::new(self.resolve_type(k)),
                    Box::new(self.resolve_type(v)),
                ),
                ty.span,
            ),
            TypeKind::Option(inner) => {
                Type::new(TypeKind::Option(Box::new(self.resolve_type(inner))), ty.span)
            }
            TypeKind::Result(ok, err) => Type::new(
                TypeKind::Result(
                    Box::new(self.resolve_type(ok)),
                    Box::new(self.resolve_type(err)),
                ),
                ty.span,
            ),
            TypeKind::Ref(inner) => {
                Type::new(TypeKind::Ref(Box::new(self.resolve_type(inner))), ty.span)
            }
            TypeKind::MutRef(inner) => {
                Type::new(TypeKind::MutRef(Box::new(self.resolve_type(inner))), ty.span)
            }
            TypeKind::Tuple(types) => Type::new(
                TypeKind::Tuple(types.iter().map(|t| self.resolve_type(t)).collect()),
                ty.span,
            ),
            TypeKind::GenericInstance { name, type_args } => Type::new(
                TypeKind::GenericInstance {
                    name: name.clone(),
                    type_args: type_args.iter().map(|t| self.resolve_type(t)).collect(),
                },
                ty.span,
            ),
            TypeKind::Function {
                params,
                return_type,
            } => Type::new(
                TypeKind::Function {
                    params: params.iter().map(|p| self.resolve_type(p)).collect(),
                    return_type: Box::new(self.resolve_type(return_type)),
                },
                ty.span,
            ),
            _ => ty.clone(),
        }
    }

    pub fn declare_variable(&mut self, name: String, ty: Type) -> Result<()> {
        let ty = self.resolve_type(&ty);
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
        let refined_type = self.resolve_type(&refined_type);
        if let Some(refinement_scope) = self.refinements.last_mut() {
            refinement_scope.insert(name, refined_type);
        }
    }

    pub fn register_constant(&mut self, name: String, ty: Type) -> Result<()> {
        let ty = self.resolve_type(&ty);
        if let Some(existing) = self.constants.get(&name) {
            if existing != &ty {
                return Err(LustError::TypeError {
                    message: format!(
                        "Constant '{}' is already defined with a different type",
                        name
                    ),
                });
            }

            return Ok(());
        }

        self.constants.insert(name, ty);
        Ok(())
    }

    pub fn lookup_constant(&self, name: &str) -> Option<Type> {
        self.constants.get(name).cloned()
    }

    pub fn record_generic_instance(
        &mut self,
        var_name: String,
        type_param: String,
        concrete_type: Type,
    ) {
        let concrete_type = self.resolve_type(&concrete_type);
        self.generic_instances
            .entry(var_name)
            .or_default()
            .insert(type_param, concrete_type);
    }

    pub fn lookup_generic_param(&self, var_name: &str, type_param: &str) -> Option<Type> {
        self.generic_instances
            .get(var_name)?
            .get(type_param)
            .cloned()
    }

    pub fn register_function(&mut self, name: String, mut sig: FunctionSignature) -> Result<()> {
        if self.functions.contains_key(&name) {
            return Err(LustError::TypeError {
                message: format!("Function '{}' is already defined", name),
            });
        }
        sig.params = sig.params.iter().map(|p| self.resolve_type(p)).collect();
        sig.return_type = self.resolve_type(&sig.return_type);

        self.functions.insert(name, sig);
        Ok(())
    }

    pub fn register_or_update_function(
        &mut self,
        name: String,
        mut sig: FunctionSignature,
    ) -> Result<()> {
        sig.params = sig.params.iter().map(|p| self.resolve_type(p)).collect();
        sig.return_type = self.resolve_type(&sig.return_type);
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

    pub fn remove_function(&mut self, name: &str) {
        self.functions.remove(name);
    }

    pub fn function_signatures(&self) -> HashMap<String, FunctionSignature> {
        self.functions.clone()
    }

    pub fn take_function_signatures(&mut self) -> HashMap<String, FunctionSignature> {
        core::mem::take(&mut self.functions)
    }

    pub fn struct_definitions(&self) -> HashMap<String, StructDef> {
        self.structs.clone()
    }

    pub fn take_struct_definitions(&mut self) -> HashMap<String, StructDef> {
        core::mem::take(&mut self.structs)
    }

    pub fn enum_definitions(&self) -> HashMap<String, EnumDef> {
        self.enums.clone()
    }

    pub fn take_enum_definitions(&mut self) -> HashMap<String, EnumDef> {
        core::mem::take(&mut self.enums)
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

    pub fn register_type_alias(&mut self, name: String, ty: Type) {
        self.type_aliases.insert(name, ty);
    }

    pub fn lookup_type_alias(&self, name: &str) -> Option<&Type> {
        self.type_aliases.get(name)
    }

    pub fn register_service(&mut self, name: String, target_type: Option<Type>) -> Result<()> {
        if self.services.contains_key(&name) {
            return Err(LustError::TypeError {
                message: format!("Service '{}' is already defined", name),
            });
        }

        self.services.insert(name, target_type);
        Ok(())
    }

    pub fn lookup_service(&self, name: &str) -> Option<&Option<Type>> {
        self.services.get(name)
    }

    pub fn is_service(&self, name: &str) -> bool {
        self.services.contains_key(name)
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

    pub fn register_impl(&mut self, impl_block: &ImplBlock) -> Result<()> {
        for existing in &self.impls {
            if let Some(trait_name) = &impl_block.trait_name
                && existing.trait_name.as_ref() == Some(trait_name)
                && self.types_overlap(&existing.target_type, &impl_block.target_type)
            {
                return Err(LustError::TypeError {
                    message: format!(
                        "Conflicting implementations of trait '{}' for type '{}'",
                        trait_name, impl_block.target_type
                    ),
                });
            }

            if self.types_overlap(&existing.target_type, &impl_block.target_type) {
                for method in &impl_block.methods {
                    let method_name = method.name.rsplit(':').next().unwrap_or(&method.name);
                    if existing.methods.iter().any(|existing_method| {
                        existing_method
                            .name
                            .rsplit(':')
                            .next()
                            .unwrap_or(&existing_method.name)
                            == method_name
                    }) {
                        return Err(LustError::TypeError {
                            message: format!(
                                "Method '{}' has conflicting implementations for type '{}'",
                                method_name, impl_block.target_type
                            ),
                        });
                    }
                }
            }
        }
        self.impls.push(impl_block.clone());
        Ok(())
    }

    pub fn lookup_method(&self, type_name: &str, method_name: &str) -> Option<&FunctionDef> {
        let effective_name = if type_name == "Table"
            || type_name == "std.table.Table"
            || type_name == "table.Table"
        {
            "Array"
        } else {
            type_name
        };
        for impl_block in &self.impls {
            if let TypeKind::Named(name) | TypeKind::GenericInstance { name, .. } =
                &impl_block.target_type.kind
                && (name == type_name || name == effective_name)
            {
                for method in &impl_block.methods {
                    if method.name.ends_with(&format!(":{}", method_name))
                        || method.name == method_name
                    {
                        return Some(method);
                    }
                }
            }
        }

        None
    }

    pub fn lookup_method_impl(
        &self,
        type_name: &str,
        method_name: &str,
    ) -> Option<(&ImplBlock, &FunctionDef)> {
        let effective_name = if type_name == "Table"
            || type_name == "std.table.Table"
            || type_name == "table.Table"
        {
            "Array"
        } else {
            type_name
        };
        for impl_block in &self.impls {
            let target_name = match &impl_block.target_type.kind {
                TypeKind::Named(name) | TypeKind::GenericInstance { name, .. } => name.as_str(),
                _ => continue,
            };
            if target_name != type_name && target_name != effective_name {
                continue;
            }
            if let Some(method) = impl_block.methods.iter().find(|method| {
                method.name.ends_with(&format!(":{}", method_name)) || method.name == method_name
            }) {
                return Some((impl_block, method));
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
        let resolved_ty = self.resolve_type(ty);
        for impl_block in &self.impls {
            if let Some(impl_trait_name) = &impl_block.trait_name
                && impl_trait_name == trait_name
            {
                let mut bindings = HashMap::new();
                if self.match_type_pattern(&impl_block.target_type, &resolved_ty, &mut bindings) {
                    let bounds_satisfied = impl_block.where_clause.iter().all(|bound| {
                        bindings.get(&bound.type_param).is_some_and(|concrete| {
                            bound
                                .traits
                                .iter()
                                .all(|required| self.type_implements_trait(concrete, required))
                        })
                    });
                    if bounds_satisfied {
                        return true;
                    }
                }
            }
        }

        false
    }

    fn match_type_pattern(
        &self,
        pattern: &Type,
        actual: &Type,
        bindings: &mut HashMap<String, Type>,
    ) -> bool {
        let pattern_resolved = self.resolve_type(pattern);
        let actual_resolved = self.resolve_type(actual);
        let pattern = &pattern_resolved;
        let actual = &actual_resolved;
        match (&pattern.kind, &actual.kind) {
            (TypeKind::Generic(name), _) => {
                if let Some(existing) = bindings.get(name) {
                    self.types_match(existing, actual)
                } else {
                    bindings.insert(name.clone(), actual.clone());
                    true
                }
            }
            (TypeKind::Array(left), TypeKind::Array(right))
            | (TypeKind::Option(left), TypeKind::Option(right))
            | (TypeKind::Ref(left), TypeKind::Ref(right))
            | (TypeKind::MutRef(left), TypeKind::MutRef(right)) => {
                self.match_type_pattern(left, right, bindings)
            }
            (TypeKind::Map(lk, lv), TypeKind::Map(rk, rv))
            | (TypeKind::Result(lk, lv), TypeKind::Result(rk, rv)) => {
                self.match_type_pattern(lk, rk, bindings)
                    && self.match_type_pattern(lv, rv, bindings)
            }
            (
                TypeKind::GenericInstance {
                    name: left_name,
                    type_args: left_args,
                },
                TypeKind::GenericInstance {
                    name: right_name,
                    type_args: right_args,
                },
            ) => {
                left_name == right_name
                    && left_args.len() == right_args.len()
                    && left_args
                        .iter()
                        .zip(right_args)
                        .all(|(left, right)| self.match_type_pattern(left, right, bindings))
            }
            _ => self.types_match(pattern, actual),
        }
    }

    pub fn types_match(&self, type1: &Type, type2: &Type) -> bool {
        let t1_resolved = self.resolve_type(type1);
        let t2_resolved = self.resolve_type(type2);
        let type1 = &t1_resolved;
        let type2 = &t2_resolved;
        if type1.kind == type2.kind {
            return true;
        }
        match (&type1.kind, &type2.kind) {
            (TypeKind::Generic(_), _) | (_, TypeKind::Generic(_)) => true,
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
            (
                TypeKind::GenericInstance {
                    name: name1,
                    type_args: args1,
                },
                TypeKind::GenericInstance {
                    name: name2,
                    type_args: args2,
                },
            ) => {
                name1 == name2
                    && args1.len() == args2.len()
                    && args1
                        .iter()
                        .zip(args2)
                        .all(|(left, right)| self.types_match(left, right))
            }

            _ => false,
        }
    }

    fn types_overlap(&self, left: &Type, right: &Type) -> bool {
        let l_resolved = self.resolve_type(left);
        let r_resolved = self.resolve_type(right);
        let left = &l_resolved;
        let right = &r_resolved;
        if let (
            TypeKind::GenericInstance {
                name: left_name, ..
            },
            TypeKind::GenericInstance {
                name: right_name, ..
            },
        ) = (&left.kind, &right.kind)
            && left_name == right_name
        {
            return true;
        }
        self.types_match(left, right)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_type_aliases() {
        let env = TypeEnv::new();
        let dummy_span = Span::new(0, 0, 0, 0);

        // Table<int> -> Array<int>
        let table_int = Type::new(
            TypeKind::GenericInstance {
                name: "Table".to_string(),
                type_args: vec![Type::new(TypeKind::Int, dummy_span)],
            },
            dummy_span,
        );
        let resolved_table_int = env.resolve_type(&table_int);
        assert_eq!(
            resolved_table_int.kind,
            TypeKind::Array(Box::new(Type::new(TypeKind::Int, dummy_span)))
        );

        // Table<string, int> -> Map<string, int>
        let table_map = Type::new(
            TypeKind::GenericInstance {
                name: "Table".to_string(),
                type_args: vec![
                    Type::new(TypeKind::String, dummy_span),
                    Type::new(TypeKind::Int, dummy_span),
                ],
            },
            dummy_span,
        );
        let resolved_table_map = env.resolve_type(&table_map);
        assert_eq!(
            resolved_table_map.kind,
            TypeKind::Map(
                Box::new(Type::new(TypeKind::String, dummy_span)),
                Box::new(Type::new(TypeKind::Int, dummy_span))
            )
        );

        // LuaTable remains LuaTable
        let lua_table = Type::new(TypeKind::Named("LuaTable".to_string()), dummy_span);
        let resolved_lua_table = env.resolve_type(&lua_table);
        assert_eq!(resolved_lua_table.kind, TypeKind::Named("LuaTable".to_string()));

        // types_match between Table<int> and Array<int>
        let array_int = Type::new(
            TypeKind::Array(Box::new(Type::new(TypeKind::Int, dummy_span))),
            dummy_span,
        );
        assert!(env.types_match(&table_int, &array_int));
        assert!(env.types_match(&array_int, &table_int));

        // types_match between Table<string, int> and Map<string, int>
        let map_str_int = Type::new(
            TypeKind::Map(
                Box::new(Type::new(TypeKind::String, dummy_span)),
                Box::new(Type::new(TypeKind::Int, dummy_span)),
            ),
            dummy_span,
        );
        assert!(env.types_match(&table_map, &map_str_int));
        assert!(env.types_match(&map_str_int, &table_map));

        // std.table.Table<int> -> Array<int>
        let std_table_int = Type::new(
            TypeKind::GenericInstance {
                name: "std.table.Table".to_string(),
                type_args: vec![Type::new(TypeKind::Int, dummy_span)],
            },
            dummy_span,
        );
        let resolved_std_table_int = env.resolve_type(&std_table_int);
        assert_eq!(
            resolved_std_table_int.kind,
            TypeKind::Array(Box::new(Type::new(TypeKind::Int, dummy_span)))
        );
        assert!(env.types_match(&std_table_int, &array_int));

        // std.table.Table<string, int> -> Map<string, int>
        let std_table_map = Type::new(
            TypeKind::GenericInstance {
                name: "std.table.Table".to_string(),
                type_args: vec![
                    Type::new(TypeKind::String, dummy_span),
                    Type::new(TypeKind::Int, dummy_span),
                ],
            },
            dummy_span,
        );
        let resolved_std_table_map = env.resolve_type(&std_table_map);
        assert_eq!(
            resolved_std_table_map.kind,
            TypeKind::Map(
                Box::new(Type::new(TypeKind::String, dummy_span)),
                Box::new(Type::new(TypeKind::Int, dummy_span))
            )
        );
        assert!(env.types_match(&std_table_map, &map_str_int));

        // std.table.LuaTable -> LuaTable
        let std_lua_table = Type::new(TypeKind::Named("std.table.LuaTable".to_string()), dummy_span);
        let resolved_std_lua_table = env.resolve_type(&std_lua_table);
        assert_eq!(resolved_std_lua_table.kind, TypeKind::Named("LuaTable".to_string()));
    }
}
