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
            ItemKind::Struct(def) => self.check_struct_definition(def),
            ItemKind::Enum(def) => self.check_enum_definition(def),
            ItemKind::Trait(def) => self.check_trait_definition(def),
            ItemKind::Impl(impl_block) => self.check_impl(impl_block),
            ItemKind::TypeAlias {
                name,
                type_params,
                target,
            } => {
                self.validate_type_alias_cycle(&self.resolve_type_key(name))?;
                self.push_type_params(type_params)?;
                let canonical = self.canonicalize_type(target);
                self.validate_type(&canonical)?;
                self.pop_type_params();
                Ok(())
            }
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
        self.push_type_params(&func.type_params)?;
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
        for param in &canonical_param_types {
            self.validate_type(param)?;
        }
        self.validate_type(&return_type)?;
        self.validate_trait_bounds(&func.type_params, &func.trait_bounds)?;
        let sig = FunctionSignature {
            params: canonical_param_types.clone(),
            return_type: return_type.clone(),
            is_method: func.is_method,
            type_params: func.type_params.clone(),
            trait_bounds: self.canonicalize_trait_bounds(&func.trait_bounds),
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
                    target_type: Type::new(
                        TypeKind::Named(resolved.clone()),
                        TypeChecker::dummy_span(),
                    ),
                    methods: vec![func.clone()],
                    where_clause: vec![],
                };
                let method_name = func.name.rsplit(':').next().unwrap_or(&func.name);
                if self.env.lookup_method(&resolved, method_name).is_none() {
                    self.env.register_impl(&impl_block)?;
                }
            }
        }

        if self.env.lookup_function(&func.name).is_none() {
            self.env.register_function(func.name.clone(), sig)?;
        }

        let prev_trait_bounds = self.current_trait_bounds.clone();
        for bound in self.canonicalize_trait_bounds(&func.trait_bounds) {
            self.current_trait_bounds
                .insert(bound.type_param, bound.traits);
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
        self.pop_type_params();
        Ok(())
    }

    fn check_struct_definition(&mut self, def: &StructDef) -> Result<()> {
        self.push_type_params(&def.type_params)?;
        self.validate_trait_bounds(&def.type_params, &def.trait_bounds)?;
        for field in &def.fields {
            self.validate_type(&self.canonicalize_type(&field.ty))?;
            if let Some(target) = &field.weak_target {
                self.validate_type(&self.canonicalize_type(target))?;
            }
        }
        self.pop_type_params();
        Ok(())
    }

    fn check_enum_definition(&mut self, def: &EnumDef) -> Result<()> {
        self.push_type_params(&def.type_params)?;
        self.validate_trait_bounds(&def.type_params, &def.trait_bounds)?;
        for variant in &def.variants {
            if let Some(fields) = &variant.fields {
                for field in fields {
                    self.validate_type(&self.canonicalize_type(field))?;
                }
            }
        }
        self.pop_type_params();
        Ok(())
    }

    fn check_trait_definition(&mut self, def: &TraitDef) -> Result<()> {
        if !def.type_params.is_empty() {
            return Err(
                self.type_error(format!("Generic trait '{}' is not supported yet", def.name))
            );
        }
        for method in &def.methods {
            if !method.type_params.is_empty() {
                return Err(self.type_error(format!(
                    "Generic trait method '{}.{}' is not supported yet",
                    def.name, method.name
                )));
            }
            if method.default_impl.is_some() {
                return Err(self.type_error(format!(
                    "Default trait method '{}.{}' is not supported yet",
                    def.name, method.name
                )));
            }
            let self_params: Vec<_> = method.params.iter().filter(|param| param.is_self).collect();
            if self_params.len() != 1
                || !method.params.first().is_some_and(|param| param.is_self)
                || !matches!(self_params[0].ty.kind, TypeKind::Unknown)
                || self_params[0].ty.span.start_line > 0
            {
                return Err(self.type_error(format!(
                    "Trait method '{}.{}' must begin with exactly one unannotated self parameter",
                    def.name, method.name
                )));
            }
            for param in &method.params {
                self.validate_type(&self.canonicalize_type(&param.ty))?;
            }
            if let Some(return_type) = &method.return_type {
                self.validate_type(&self.canonicalize_type(return_type))?;
            }
        }
        Ok(())
    }

    fn check_impl(&mut self, impl_block: &ImplBlock) -> Result<()> {
        self.push_type_params(&impl_block.type_params)?;
        let raw_target_name = match &impl_block.target_type.kind {
            TypeKind::Named(name) | TypeKind::GenericInstance { name, .. } => Some(name),
            _ => None,
        };
        if raw_target_name.is_some_and(|name| {
            self.env
                .lookup_type_alias(&self.resolve_type_key(name))
                .is_some()
        }) {
            return Err(self.type_error(
                "Impl targets cannot be type aliases; use the underlying nominal type".to_string(),
            ));
        }
        self.validate_trait_bounds(&impl_block.type_params, &impl_block.where_clause)?;
        if !impl_block.where_clause.is_empty() {
            return Err(self.type_error(
                "Conditional generic impls are not supported with runtime-erased type arguments"
                    .to_string(),
            ));
        }
        let previous_impl_bounds = self.current_trait_bounds.clone();
        let canonical_impl_bounds = self.canonicalize_trait_bounds(&impl_block.where_clause);
        for bound in &canonical_impl_bounds {
            self.current_trait_bounds
                .insert(bound.type_param.clone(), bound.traits.clone());
        }
        let canonical_target = self.canonicalize_type(&impl_block.target_type);
        if let TypeKind::GenericInstance { name, type_args } = &canonical_target.kind {
            let declaration = self
                .env
                .lookup_struct(name)
                .map(|def| (&def.type_params, &def.trait_bounds))
                .or_else(|| {
                    self.env
                        .lookup_enum(name)
                        .map(|def| (&def.type_params, &def.trait_bounds))
                });
            if let Some((declared_params, declared_bounds)) = declaration {
                for bound in declared_bounds {
                    if let Some(index) = declared_params
                        .iter()
                        .position(|param| param == &bound.type_param)
                    {
                        if let Some(Type {
                            kind: TypeKind::Generic(actual_param),
                            ..
                        }) = type_args.get(index)
                        {
                            let traits: Vec<String> = bound
                                .traits
                                .iter()
                                .map(|name| self.resolve_type_key(name))
                                .collect();
                            self.current_trait_bounds
                                .entry(actual_param.clone())
                                .or_default()
                                .extend(traits);
                        }
                    }
                }
            }
        }
        self.validate_type(&canonical_target)?;
        if let TypeKind::GenericInstance { type_args, .. } = &canonical_target.kind {
            let universal_target = type_args.len() == impl_block.type_params.len()
                && type_args.iter().zip(&impl_block.type_params).all(
                    |(arg, param)| matches!(&arg.kind, TypeKind::Generic(name) if name == param),
                );
            if !universal_target {
                return Err(self.type_error(
                    "Specialized generic impls are not supported with runtime-erased type arguments"
                        .to_string(),
                ));
            }
        }
        let type_name = if let TypeKind::Named(name) | TypeKind::GenericInstance { name, .. } =
            &canonical_target.kind
        {
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
        let mut impl_block_q = impl_block.clone();
        if let Some(trait_name) = &impl_block.trait_name {
            impl_block_q.trait_name = Some(self.resolve_type_key(trait_name));
        }
        impl_block_q.target_type = canonical_target.clone();
        impl_block_q.where_clause = self.canonicalize_trait_bounds(&impl_block.where_clause);
        for method in &mut impl_block_q.methods {
            self.push_type_params(&method.type_params)?;
            for param in &mut method.params {
                param.ty = self.canonicalize_type(&param.ty);
            }
            if let Some(ret_ty) = method.return_type.clone() {
                method.return_type = Some(self.canonicalize_type(&ret_ty));
            }
            method.trait_bounds = self.canonicalize_trait_bounds(&method.trait_bounds);
            self.pop_type_params();
        }

        if let Some(trait_name) = &impl_block.trait_name {
            let resolved_trait = self.resolve_type_key(trait_name);
            let trait_def = self
                .env
                .lookup_trait(&resolved_trait)
                .ok_or_else(|| self.type_error(format!("Undefined trait '{}'", trait_name)))?
                .clone();
            for trait_method in &trait_def.methods {
                let impl_method = impl_block_q.methods.iter().find(|m| {
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
                if !impl_method.type_params.is_empty() || !impl_method.trait_bounds.is_empty() {
                    return Err(self.type_error(format!(
                        "Trait method implementation '{}.{}' cannot introduce generic parameters or bounds",
                        type_name, trait_method.name
                    )));
                }
                let trait_params: Vec<_> =
                    trait_method.params.iter().filter(|p| !p.is_self).collect();
                let impl_params: Vec<_> =
                    impl_method.params.iter().filter(|p| !p.is_self).collect();
                let trait_has_self = trait_method.params.iter().any(|param| param.is_self);
                let impl_has_self = impl_method.params.iter().any(|param| param.is_self);
                if trait_has_self != impl_has_self {
                    return Err(self.type_error(format!(
                        "Method '{}' in impl for '{}' must {}a self receiver",
                        trait_method.name,
                        type_name,
                        if trait_has_self { "have " } else { "not have " }
                    )));
                }
                if let Some(self_param) = impl_method.params.iter().find(|param| param.is_self) {
                    if !matches!(self_param.ty.kind, TypeKind::Infer) {
                        return Err(self.type_error(format!(
                            "Method '{}' in impl for '{}' must use an unannotated self parameter",
                            trait_method.name, type_name
                        )));
                    }
                }
                if trait_params.len() != impl_params.len() {
                    return Err(self.type_error(format!(
                        "Method '{}' in impl for '{}' has {} parameters, but trait '{}' requires {}",
                        trait_method.name, type_name, impl_params.len(), trait_name, trait_params.len()
                    )));
                }

                for (trait_param, impl_param) in trait_params.iter().zip(impl_params.iter()) {
                    let trait_param_type = self.canonicalize_type(&trait_param.ty);
                    if !self.types_equal(&trait_param_type, &impl_param.ty) {
                        return Err(self.type_error(format!(
                            "Method '{}' parameter '{}' has type '{}', but trait requires '{}'",
                            trait_method.name, impl_param.name, impl_param.ty, trait_param_type
                        )));
                    }
                }

                let trait_return = trait_method
                    .return_type
                    .clone()
                    .map(|ty| self.canonicalize_type(&ty))
                    .unwrap_or(Type::new(TypeKind::Unit, TypeChecker::dummy_span()));
                let impl_return = impl_method
                    .return_type
                    .clone()
                    .unwrap_or(Type::new(TypeKind::Unit, TypeChecker::dummy_span()));
                if !matches!(trait_return.kind, TypeKind::Unknown)
                    && !self.types_equal(&trait_return, &impl_return)
                {
                    return Err(self.type_error(format!(
                        "Method '{}' returns '{}', but trait '{}' requires '{}'",
                        trait_method.name, impl_return, trait_name, trait_return
                    )));
                }
            }
        }

        self.env.register_impl(&impl_block_q)?;
        for method in &impl_block.methods {
            let mut method_with_mangled_name = method.clone();
            let has_self = method.params.iter().any(|p| p.is_self || p.name == "self");
            if !has_self && (!impl_block.type_params.is_empty() || !method.type_params.is_empty()) {
                return Err(self.type_error(format!(
                    "Generic static method '{}.{}' is not supported yet",
                    type_name, method.name
                )));
            }
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
                    param.ty = canonical_target.clone();
                }
            }

            self.check_function(&method_with_mangled_name)?;
        }

        self.current_trait_bounds = previous_impl_bounds;
        self.pop_type_params();
        Ok(())
    }

    fn check_const(&mut self, name: &str, ty: &Type, value: &Expr) -> Result<()> {
        let ty = self.canonicalize_type(ty);
        self.validate_type(&ty)?;
        let value_type = self.check_expr(value)?;
        self.unify(&ty, &value_type)?;
        self.env.declare_variable(name.to_string(), ty)?;
        Ok(())
    }

    fn check_static(&mut self, name: &str, ty: &Type, value: &Expr) -> Result<()> {
        let ty = self.canonicalize_type(ty);
        self.validate_type(&ty)?;
        let value_type = self.check_expr(value)?;
        self.unify(&ty, &value_type)?;
        self.env.declare_variable(name.to_string(), ty)?;
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
                        type_params: Vec::new(),
                        trait_bounds: Vec::new(),
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
