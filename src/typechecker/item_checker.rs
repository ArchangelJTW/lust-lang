use super::{type_env::FunctionSignature, TypeChecker};
use crate::{ast::*, error::Result};
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
impl TypeChecker {
    pub(super) fn check_item(&mut self, item: &Item) -> Result<()> {
        match &item.kind {
            ItemKind::Script(stmts) => {
                let prev_return = self.current_function_return_type.clone();
                let script_return = Type::new(TypeKind::Unit, TypeChecker::dummy_span());
                self.current_function_return_type = Some(script_return);
                for stmt in stmts {
                    self.check_stmt(stmt)?;
                }

                self.current_function_return_type = prev_return;
                Ok(())
            }

            ItemKind::Function(func) => self.check_function(func),
            ItemKind::Struct(_) => Ok(()),
            ItemKind::Enum(_) => Ok(()),
            ItemKind::Trait(_) => Ok(()),
            ItemKind::Impl(impl_block) => self.check_impl(impl_block),
            ItemKind::TypeAlias { .. } => Ok(()),
            ItemKind::Module { items, .. } => {
                for item in items {
                    self.check_item(item)?;
                }

                Ok(())
            }

            ItemKind::Use { .. } => Ok(()),
            ItemKind::Const { name, ty, value } => self.check_const(name, ty, value),
            ItemKind::Static {
                name, ty, value, ..
            } => self.check_static(name, ty, value),
            ItemKind::Extern { items, .. } => self.check_extern(items),
        }
    }

    fn check_function(&mut self, func: &FunctionDef) -> Result<()> {
        let canonical_param_types: Vec<Type> = func
            .params
            .iter()
            .map(|p| self.canonicalize_type(&p.ty))
            .collect();
        let return_type = func
            .return_type
            .as_ref()
            .map(|ty| self.canonicalize_type(ty))
            .unwrap_or(Type::new(TypeKind::Unit, TypeChecker::dummy_span()));
        let sig = FunctionSignature {
            params: canonical_param_types.clone(),
            return_type: return_type.clone(),
            is_method: func.is_method,
        };
        let mut resolved_self_type: Option<String> = None;
        if func.is_method {
            if let Some(colon_pos) = func.name.find(':') {
                let type_name = &func.name[..colon_pos];
                let resolved = self.resolve_type_key(type_name);
                resolved_self_type = Some(resolved.clone());
                let impl_block = ImplBlock {
                    type_params: vec![],
                    trait_name: None,
                    target_type: Type::new(TypeKind::Named(resolved), TypeChecker::dummy_span()),
                    methods: vec![func.clone()],
                    where_clause: vec![],
                };
                self.env.register_impl(&impl_block);
            }
        }

        if self.env.lookup_function(&func.name).is_none() {
            self.env.register_function(func.name.clone(), sig)?;
        }

        let prev_trait_bounds = self.current_trait_bounds.clone();
        for bound in &func.trait_bounds {
            self.current_trait_bounds
                .insert(bound.type_param.clone(), bound.traits.clone());
        }

        self.env.push_scope();
        if func.is_method && !func.params.iter().any(|p| p.is_self) {
            if let Some(resolved) = resolved_self_type.as_ref().cloned() {
                let self_type = Type::new(TypeKind::Named(resolved), TypeChecker::dummy_span());
                self.env.declare_variable("self".to_string(), self_type)?;
            }
        }

        for (param, ty) in func.params.iter().zip(canonical_param_types.iter()) {
            self.env.declare_variable(param.name.clone(), ty.clone())?;
        }

        let prev_return_type = self.current_function_return_type.clone();
        self.current_function_return_type = Some(return_type.clone());
        for stmt in &func.body {
            self.check_stmt(stmt)?;
        }

        if !func.body.is_empty() {
            if let Some(last_stmt) = func.body.last() {
                match &last_stmt.kind {
                    StmtKind::Return(_) => {}
                    StmtKind::Expr(expr) => {
                        let expr_type = self.check_expr(expr)?;
                        self.unify(&return_type, &expr_type)?;
                    }
                    _ => {}
                }
            }
        }

        self.current_function_return_type = prev_return_type;
        self.current_trait_bounds = prev_trait_bounds;
        self.env.pop_scope();
        Ok(())
    }

    fn check_impl(&mut self, impl_block: &ImplBlock) -> Result<()> {
        let type_name = if let TypeKind::Named(name) = &impl_block.target_type.kind {
            let key = self.resolve_type_key(name);
            if self.env.lookup_struct(&key).is_none() && self.env.lookup_enum(&key).is_none() {
                return Err(self.type_error(format!(
                    "Cannot implement methods for undefined type '{}'",
                    name
                )));
            }

            key
        } else {
            return Err(self.type_error("Impl block target must be a named type".to_string()));
        };
        if let Some(trait_name) = &impl_block.trait_name {
            let resolved_trait = self.resolve_type_key(trait_name);
            let trait_def = self
                .env
                .lookup_trait(&resolved_trait)
                .ok_or_else(|| self.type_error(format!("Undefined trait '{}'", trait_name)))?
                .clone();
            for trait_method in &trait_def.methods {
                let impl_method = impl_block.methods.iter().find(|m| {
                    m.name == trait_method.name
                        || m.name.ends_with(&format!(":{}", trait_method.name))
                });
                let impl_method = match impl_method {
                    Some(m) => m,
                    None => {
                        return Err(self.type_error(format!(
                            "Trait '{}' requires method '{}' to be implemented for type '{}'",
                            trait_name, trait_method.name, type_name
                        )));
                    }
                };
                let trait_params: Vec<_> =
                    trait_method.params.iter().filter(|p| !p.is_self).collect();
                let impl_params: Vec<_> =
                    impl_method.params.iter().filter(|p| !p.is_self).collect();
                if trait_params.len() != impl_params.len() {
                    return Err(self.type_error(format!(
                        "Method '{}' in impl for '{}' has {} parameters, but trait '{}' requires {}",
                        trait_method.name, type_name, impl_params.len(), trait_name, trait_params.len()
                    )));
                }

                for (trait_param, impl_param) in trait_params.iter().zip(impl_params.iter()) {
                    if !self.types_compatible(&trait_param.ty, &impl_param.ty) {
                        return Err(self.type_error(format!(
                            "Method '{}' parameter '{}' has type '{}', but trait requires '{}'",
                            trait_method.name, impl_param.name, impl_param.ty, trait_param.ty
                        )));
                    }
                }

                let trait_return = trait_method
                    .return_type
                    .clone()
                    .unwrap_or(Type::new(TypeKind::Unit, TypeChecker::dummy_span()));
                let impl_return = impl_method
                    .return_type
                    .clone()
                    .unwrap_or(Type::new(TypeKind::Unit, TypeChecker::dummy_span()));
                if !self.types_compatible(&trait_return, &impl_return) {
                    return Err(self.type_error(format!(
                        "Method '{}' returns '{}', but trait '{}' requires '{}'",
                        trait_method.name, impl_return, trait_name, trait_return
                    )));
                }
            }
        }

        let mut impl_block_q = impl_block.clone();
        impl_block_q.target_type = Type::new(
            TypeKind::Named(type_name.clone()),
            impl_block.target_type.span,
        );
        for method in &mut impl_block_q.methods {
            for param in &mut method.params {
                let canonical = self.canonicalize_type(&param.ty);
                param.ty = canonical;
            }

            if let Some(ret_ty) = method.return_type.clone() {
                method.return_type = Some(self.canonicalize_type(&ret_ty));
            }
        }

        self.env.register_impl(&impl_block_q);
        for method in &impl_block.methods {
            let mut method_with_mangled_name = method.clone();
            let has_self = method.params.iter().any(|p| p.is_self || p.name == "self");
            let mangled_name = if method.name.contains(':') || method.name.contains('.') {
                method.name.clone()
            } else if has_self {
                format!("{}:{}", type_name, method.name)
            } else {
                format!("{}.{}", type_name, method.name)
            };
            method_with_mangled_name.name = mangled_name;
            method_with_mangled_name.is_method = has_self;
            for param in &mut method_with_mangled_name.params {
                if param.is_self && matches!(param.ty.kind, TypeKind::Infer) {
                    param.ty = impl_block.target_type.clone();
                }
            }

            self.check_function(&method_with_mangled_name)?;
        }

        Ok(())
    }

    fn check_const(&mut self, name: &str, ty: &Type, value: &Expr) -> Result<()> {
        let value_type = self.check_expr(value)?;
        self.unify(ty, &value_type)?;
        self.env.declare_variable(name.to_string(), ty.clone())?;
        Ok(())
    }

    fn check_static(&mut self, name: &str, ty: &Type, value: &Expr) -> Result<()> {
        let value_type = self.check_expr(value)?;
        self.unify(ty, &value_type)?;
        self.env.declare_variable(name.to_string(), ty.clone())?;
        Ok(())
    }

    fn check_extern(&mut self, items: &[ExternItem]) -> Result<()> {
        for item in items {
            match item {
                ExternItem::Function {
                    name,
                    params,
                    return_type,
                } => {
                    let canonical_params: Vec<Type> =
                        params.iter().map(|ty| self.canonicalize_type(ty)).collect();
                    let canonical_return = return_type
                        .clone()
                        .map(|ty| self.canonicalize_type(&ty))
                        .unwrap_or(Type::new(TypeKind::Unit, TypeChecker::dummy_span()));
                    let sig = FunctionSignature {
                        params: canonical_params.clone(),
                        return_type: canonical_return.clone(),
                        is_method: false,
                    };
                    self.register_external_function((name.clone(), sig.clone()))?;
                    if let Some((_struct_name_raw, method_name)) = name.split_once(':') {
                        if let Some(self_ty) = canonical_params.first() {
                            let canonical_self = self_ty.clone();
                            if matches!(
                                canonical_self.kind,
                                TypeKind::Named(_) | TypeKind::GenericInstance { .. }
                            ) {
                                let struct_name = match &canonical_self.kind {
                                    TypeKind::Named(name) => name.clone(),
                                    TypeKind::GenericInstance { name, .. } => name.clone(),
                                    _ => unreachable!(),
                                };
                                let mut method_params: Vec<FunctionParam> = Vec::new();
                                method_params.push(FunctionParam {
                                    name: "self".to_string(),
                                    ty: canonical_self.clone(),
                                    is_self: true,
                                });
                                for (idx, ty) in canonical_params.iter().enumerate().skip(1) {
                                    method_params.push(FunctionParam {
                                        name: format!("arg{}", idx),
                                        ty: ty.clone(),
                                        is_self: false,
                                    });
                                }
                                let method_def = FunctionDef {
                                    name: format!("{}:{}", struct_name, method_name),
                                    type_params: Vec::new(),
                                    trait_bounds: Vec::new(),
                                    params: method_params,
                                    return_type: Some(canonical_return.clone()),
                                    body: Vec::new(),
                                    is_method: true,
                                    visibility: Visibility::Public,
                                };
                                let impl_block = ImplBlock {
                                    type_params: Vec::new(),
                                    trait_name: None,
                                    target_type: canonical_self.clone(),
                                    methods: vec![method_def],
                                    where_clause: Vec::new(),
                                };
                                self.register_external_impl(impl_block)?;
                            }
                        }
                    }
                }

                ExternItem::Const { name, ty } => {
                    self.register_external_constant(name.clone(), ty.clone())?;
                }

                ExternItem::Struct(_) | ExternItem::Enum(_) => {
                    // Type definitions are registered earlier during collection.
                }
            }
        }

        Ok(())
    }
}
