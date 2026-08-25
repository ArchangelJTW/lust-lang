mod expr_checker;
mod item_checker;
mod stmt_checker;
mod type_env;
use crate::modules::{LoadedModule, ModuleImports};
use crate::{
    ast::*,
    config::LustConfig,
    error::{LustError, Result},
};
pub(super) use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::mem;
use hashbrown::{HashMap, HashSet};
pub use type_env::FunctionSignature;
pub use type_env::TypeEnv;
pub struct TypeChecker {
    env: TypeEnv,
    current_function_return_type: Option<Type>,
    in_loop: bool,
    pending_generic_instances: Option<HashMap<String, Type>>,
    expected_lambda_signature: Option<(Vec<Type>, Option<Type>)>,
    current_trait_bounds: HashMap<String, Vec<String>>,
    type_param_scopes: Vec<HashSet<String>>,
    current_module: Option<String>,
    imports_by_module: HashMap<String, ModuleImports>,
    expr_types_by_module: HashMap<String, HashMap<Span, Type>>,
    variable_types_by_module: HashMap<String, HashMap<Span, Type>>,
    short_circuit_info: HashMap<String, HashMap<Span, ShortCircuitInfo>>,
    checked_array_indices: HashMap<String, HashSet<Span>>,
    low_memory_mode: bool,
}

pub struct TypeCollection {
    pub expr_types: HashMap<String, HashMap<Span, Type>>,
    pub variable_types: HashMap<String, HashMap<Span, Type>>,
}

#[derive(Clone, Debug)]
struct ShortCircuitInfo {
    truthy: Option<Type>,
    falsy: Option<Type>,
    option_inner: Option<Type>,
}

impl TypeChecker {
    pub fn new() -> Self {
        Self::with_config(&LustConfig::default())
    }

    pub fn with_config(config: &LustConfig) -> Self {
        Self {
            env: TypeEnv::with_config(config),
            current_function_return_type: None,
            in_loop: false,
            pending_generic_instances: None,
            expected_lambda_signature: None,
            current_trait_bounds: HashMap::new(),
            type_param_scopes: Vec::new(),
            current_module: None,
            imports_by_module: HashMap::new(),
            expr_types_by_module: HashMap::new(),
            variable_types_by_module: HashMap::new(),
            short_circuit_info: HashMap::new(),
            checked_array_indices: HashMap::new(),
            low_memory_mode: config.low_memory_mode(),
        }
    }

    /// Configure the typechecker with a LustConfig
    pub fn configure(&mut self, config: &LustConfig) {
        self.low_memory_mode = config.low_memory_mode();
    }

    fn dummy_span() -> Span {
        Span::new(0, 0, 0, 0)
    }

    pub fn check_module(&mut self, items: &[Item]) -> Result<()> {
        for item in items {
            self.register_type_definition(item)?;
        }

        self.validate_struct_cycles()?;
        self.env.push_scope();
        self.register_module_init_locals(items)?;
        for item in items {
            self.check_item(item)?;
        }

        self.env.pop_scope();
        Ok(())
    }

    pub fn check_program(&mut self, modules: &[LoadedModule]) -> Result<()> {
        for m in modules {
            self.current_module = Some(m.path.clone());
            for item in &m.items {
                self.register_type_definition(item)?;
            }
        }

        self.validate_struct_cycles()?;
        for m in modules {
            self.current_module = Some(m.path.clone());
            self.env.push_scope();
            self.register_module_init_locals(&m.items)?;
            for item in &m.items {
                self.check_item(item)?;
            }

            self.env.pop_scope();
        }

        self.current_module = None;
        Ok(())
    }

    fn validate_struct_cycles(&self) -> Result<()> {
        use hashbrown::{HashMap, HashSet};
        let struct_defs = self.env.struct_definitions();
        if struct_defs.is_empty() {
            return Ok(());
        }

        let mut simple_to_full: HashMap<String, Vec<String>> = HashMap::new();
        for name in struct_defs.keys() {
            let simple = name.rsplit('.').next().unwrap_or(name).to_string();
            simple_to_full.entry(simple).or_default().push(name.clone());
        }

        let mut struct_has_weak: HashMap<String, bool> = HashMap::new();
        for (name, def) in &struct_defs {
            let has_weak = def
                .fields
                .iter()
                .any(|field| matches!(field.ownership, FieldOwnership::Weak));
            struct_has_weak.insert(name.clone(), has_weak);
        }

        let mut graph: HashMap<String, Vec<String>> = HashMap::new();
        for (name, def) in &struct_defs {
            let module_prefix = name.rsplit_once('.').map(|(module, _)| module.to_string());
            let mut edges: HashSet<String> = HashSet::new();
            for field in &def.fields {
                if matches!(field.ownership, FieldOwnership::Weak) {
                    let target = field.weak_target.as_ref().ok_or_else(|| {
                        self.type_error(format!(
                            "Field '{}.{}' is marked as 'ref' but has no target type",
                            name, field.name
                        ))
                    })?;
                    let target_name = if let TypeKind::Named(inner)
                    | TypeKind::GenericInstance { name: inner, .. } = &target.kind
                    {
                        inner
                    } else {
                        return Err(self.type_error(format!(
                            "Field '{}.{}' uses 'ref' but only struct types are supported",
                            name, field.name
                        )));
                    };
                    let resolved = self.resolve_struct_name_for_cycle(
                        target_name.as_str(),
                        module_prefix.as_deref(),
                        &struct_defs,
                        &simple_to_full,
                    );
                    if resolved.is_none() {
                        return Err(self.type_error(format!(
                            "Field '{}.{}' uses 'ref' but '{}' is not a known struct type",
                            name, field.name, target_name
                        )));
                    }

                    continue;
                }

                self.collect_strong_struct_targets(
                    &field.ty,
                    module_prefix.as_deref(),
                    &struct_defs,
                    &simple_to_full,
                    &mut edges,
                );
            }

            graph.insert(name.clone(), edges.into_iter().collect());
        }

        fn dfs(
            node: &str,
            graph: &HashMap<String, Vec<String>>,
            visited: &mut HashSet<String>,
            on_stack: &mut HashSet<String>,
            stack: &mut Vec<String>,
        ) -> Option<Vec<String>> {
            visited.insert(node.to_string());
            on_stack.insert(node.to_string());
            stack.push(node.to_string());
            if let Some(neighbors) = graph.get(node) {
                for neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        if let Some(cycle) = dfs(neighbor, graph, visited, on_stack, stack) {
                            return Some(cycle);
                        }
                    } else if on_stack.contains(neighbor) {
                        if let Some(pos) = stack.iter().position(|n| n == neighbor) {
                            let mut cycle = stack[pos..].to_vec();
                            cycle.push(neighbor.clone());
                            return Some(cycle);
                        }
                    }
                }
            }

            stack.pop();
            on_stack.remove(node);
            None
        }

        let mut visited: HashSet<String> = HashSet::new();
        let mut on_stack: HashSet<String> = HashSet::new();
        let mut stack: Vec<String> = Vec::new();
        for name in struct_defs.keys() {
            if !visited.contains(name) {
                if let Some(cycle) = dfs(name, &graph, &mut visited, &mut on_stack, &mut stack) {
                    let contains_weak = cycle
                        .iter()
                        .any(|node| struct_has_weak.get(node).copied().unwrap_or(false));
                    if contains_weak {
                        continue;
                    }

                    // let description = cycle.join(" -> ");
                    break;
                    // return Err(self.type_error(format!(
                    //     "Strong ownership cycle detected: {}. Mark at least one field as 'ref' to break the cycle.",
                    //     description
                    // )));
                }
            }
        }

        Ok(())
    }

    fn collect_strong_struct_targets(
        &self,
        ty: &Type,
        parent_module: Option<&str>,
        struct_defs: &HashMap<String, StructDef>,
        simple_to_full: &HashMap<String, Vec<String>>,
        out: &mut HashSet<String>,
    ) {
        match &ty.kind {
            TypeKind::Named(name) => {
                if let Some(resolved) = self.resolve_struct_name_for_cycle(
                    name,
                    parent_module,
                    struct_defs,
                    simple_to_full,
                ) {
                    out.insert(resolved);
                }
            }

            TypeKind::Array(inner)
            | TypeKind::Ref(inner)
            | TypeKind::MutRef(inner)
            | TypeKind::Option(inner) => {
                self.collect_strong_struct_targets(
                    inner,
                    parent_module,
                    struct_defs,
                    simple_to_full,
                    out,
                );
            }

            TypeKind::Map(key, value) => {
                self.collect_strong_struct_targets(
                    key,
                    parent_module,
                    struct_defs,
                    simple_to_full,
                    out,
                );
                self.collect_strong_struct_targets(
                    value,
                    parent_module,
                    struct_defs,
                    simple_to_full,
                    out,
                );
            }

            TypeKind::Tuple(elements) | TypeKind::Union(elements) => {
                for element in elements {
                    self.collect_strong_struct_targets(
                        element,
                        parent_module,
                        struct_defs,
                        simple_to_full,
                        out,
                    );
                }
            }

            TypeKind::Result(ok, err) => {
                self.collect_strong_struct_targets(
                    ok,
                    parent_module,
                    struct_defs,
                    simple_to_full,
                    out,
                );
                self.collect_strong_struct_targets(
                    err,
                    parent_module,
                    struct_defs,
                    simple_to_full,
                    out,
                );
            }

            TypeKind::GenericInstance { type_args, .. } => {
                for arg in type_args {
                    self.collect_strong_struct_targets(
                        arg,
                        parent_module,
                        struct_defs,
                        simple_to_full,
                        out,
                    );
                }
            }

            _ => {}
        }
    }

    fn resolve_struct_name_for_cycle(
        &self,
        name: &str,
        parent_module: Option<&str>,
        struct_defs: &HashMap<String, StructDef>,
        simple_to_full: &HashMap<String, Vec<String>>,
    ) -> Option<String> {
        if struct_defs.contains_key(name) {
            return Some(name.to_string());
        }

        if name.contains('.') {
            return None;
        }

        if let Some(candidates) = simple_to_full.get(name) {
            if candidates.len() == 1 {
                return Some(candidates[0].clone());
            }

            if let Some(module) = parent_module {
                for candidate in candidates {
                    if let Some((candidate_module, _)) = candidate.rsplit_once('.') {
                        if candidate_module == module {
                            return Some(candidate.clone());
                        }
                    }
                }
            }
        }

        None
    }

    pub fn set_imports_by_module(&mut self, map: HashMap<String, ModuleImports>) {
        self.imports_by_module = map;
    }

    pub fn take_type_info(&mut self) -> TypeCollection {
        TypeCollection {
            expr_types: mem::take(&mut self.expr_types_by_module),
            variable_types: mem::take(&mut self.variable_types_by_module),
        }
    }

    pub fn take_option_coercions(&mut self) -> HashMap<String, HashSet<Span>> {
        let mut result: HashMap<String, HashSet<Span>> = HashMap::new();
        let info = mem::take(&mut self.short_circuit_info);
        for (module, entries) in info {
            let mut spans: HashSet<Span> = HashSet::new();
            for (span, entry) in entries {
                if entry.option_inner.is_some() {
                    spans.insert(span);
                }
            }
            if !spans.is_empty() {
                result.insert(module, spans);
            }
        }

        result
    }

    pub fn take_checked_array_indices(&mut self) -> HashMap<String, HashSet<Span>> {
        mem::take(&mut self.checked_array_indices)
    }

    pub fn function_signatures(&self) -> HashMap<String, type_env::FunctionSignature> {
        self.env.function_signatures()
    }

    pub fn take_function_signatures(&mut self) -> HashMap<String, type_env::FunctionSignature> {
        self.env.take_function_signatures()
    }

    pub fn struct_definitions(&self) -> HashMap<String, StructDef> {
        self.env.struct_definitions()
    }

    pub fn take_struct_definitions(&mut self) -> HashMap<String, StructDef> {
        self.env.take_struct_definitions()
    }

    pub fn enum_definitions(&self) -> HashMap<String, EnumDef> {
        self.env.enum_definitions()
    }

    pub fn take_enum_definitions(&mut self) -> HashMap<String, EnumDef> {
        self.env.take_enum_definitions()
    }

    fn register_module_init_locals(&mut self, items: &[Item]) -> Result<()> {
        let module = match &self.current_module {
            Some(m) => m.clone(),
            None => return Ok(()),
        };
        let init_name = format!("__init@{}", module);
        for item in items {
            if let ItemKind::Function(func) = &item.kind {
                if func.name == init_name {
                    for stmt in &func.body {
                        if let StmtKind::Local {
                            bindings,
                            ref mutable,
                            initializer,
                        } = &stmt.kind
                        {
                            self.check_local_stmt(
                                bindings.as_slice(),
                                *mutable,
                                initializer.as_ref().map(|values| values.as_slice()),
                            )?;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub fn resolve_function_key(&self, name: &str) -> String {
        if name.contains('.') || name.contains(':') {
            return name.to_string();
        }

        if let Some(module) = &self.current_module {
            if let Some(imports) = self.imports_by_module.get(module) {
                if let Some(fq) = imports.function_aliases.get(name) {
                    return fq.clone();
                }
            }

            let qualified = format!("{}.{}", module, name);
            if self.env.lookup_function(&qualified).is_some() {
                return qualified;
            }

            if self.env.lookup_function(name).is_some() {
                return name.to_string();
            }

            return qualified;
        }

        name.to_string()
    }

    pub fn resolve_value_key(&self, name: &str) -> String {
        if name.contains('.') || name.contains(':') {
            return name.to_string();
        }

        if let Some(module) = &self.current_module {
            if let Some(imports) = self.imports_by_module.get(module) {
                if let Some(fq) = imports.function_aliases.get(name) {
                    return fq.clone();
                }
            }

            return format!("{}.{}", module, name);
        }

        name.to_string()
    }

    pub fn resolve_module_alias(&self, alias: &str) -> Option<String> {
        if let Some(module) = &self.current_module {
            if let Some(imports) = self.imports_by_module.get(module) {
                if let Some(m) = imports.module_aliases.get(alias) {
                    return Some(m.clone());
                }
            }
        }

        None
    }

    pub fn register_external_struct(&mut self, mut def: StructDef) -> Result<()> {
        def.name = self.resolve_type_key(&def.name);
        self.push_type_params(&def.type_params)?;
        for field in &mut def.fields {
            field.ty = self.canonicalize_type(&field.ty);
            if let Some(target) = &field.weak_target {
                field.weak_target = Some(self.canonicalize_type(target));
            }
        }
        self.pop_type_params();
        self.env.register_struct(&def)
    }

    pub fn register_external_enum(&mut self, mut def: EnumDef) -> Result<()> {
        def.name = self.resolve_type_key(&def.name);
        self.push_type_params(&def.type_params)?;
        for variant in &mut def.variants {
            if let Some(fields) = &mut variant.fields {
                for field in fields {
                    *field = self.canonicalize_type(field);
                }
            }
        }
        self.pop_type_params();
        self.env.register_enum(&def)
    }

    pub fn register_external_trait(&mut self, mut def: TraitDef) -> Result<()> {
        def.name = self.resolve_type_key(&def.name);
        self.push_type_params(&def.type_params)?;
        for method in &mut def.methods {
            self.push_type_params(&method.type_params)?;
            for param in &mut method.params {
                param.ty = self.canonicalize_type(&param.ty);
            }
            if let Some(ret) = method.return_type.clone() {
                method.return_type = Some(self.canonicalize_type(&ret));
            }
            self.pop_type_params();
        }
        self.pop_type_params();
        self.env.register_trait(&def)
    }

    pub fn register_external_function(
        &mut self,
        (name, mut signature): (String, FunctionSignature),
    ) -> Result<()> {
        self.push_type_params(&signature.type_params)?;
        signature.params = signature
            .params
            .into_iter()
            .map(|ty| self.canonicalize_type(&ty))
            .collect();
        signature.return_type = self.canonicalize_type(&signature.return_type);
        signature.trait_bounds = self.canonicalize_trait_bounds(&signature.trait_bounds);
        for param in &signature.params {
            self.validate_type(param)?;
        }
        self.validate_type(&signature.return_type)?;
        self.validate_trait_bounds(&signature.type_params, &signature.trait_bounds)?;
        self.pop_type_params();
        let canonical = self.resolve_type_key(&name);
        self.env.register_or_update_function(canonical, signature)
    }

    pub fn register_external_constant(&mut self, name: String, ty: Type) -> Result<()> {
        let canonical_ty = self.canonicalize_type(&ty);
        let canonical_name = self.resolve_value_key(&name);
        self.env.register_constant(canonical_name, canonical_ty)
    }

    pub fn register_external_impl(&mut self, mut impl_block: ImplBlock) -> Result<()> {
        self.push_type_params(&impl_block.type_params)?;
        impl_block.target_type = self.canonicalize_type(&impl_block.target_type);
        self.validate_type(&impl_block.target_type)?;
        self.validate_trait_bounds(&impl_block.type_params, &impl_block.where_clause)?;
        if !impl_block.where_clause.is_empty() {
            return Err(self.type_error(
                "Conditional external impls are not supported with runtime-erased type arguments"
                    .to_string(),
            ));
        }
        if let TypeKind::GenericInstance { type_args, .. } = &impl_block.target_type.kind {
            let universal_target = type_args.len() == impl_block.type_params.len()
                && type_args.iter().zip(&impl_block.type_params).all(|(arg, param)| {
                    matches!(&arg.kind, TypeKind::Generic(name) if name == param)
                });
            if !universal_target {
                return Err(self.type_error(
                    "Specialized external generic impls are not supported with runtime-erased type arguments"
                        .to_string(),
                ));
            }
        }
        impl_block.where_clause = self.canonicalize_trait_bounds(&impl_block.where_clause);
        if let Some(trait_name) = &impl_block.trait_name {
            let resolved = self.resolve_type_key(trait_name);
            if self.env.lookup_trait(&resolved).is_none() {
                return Err(self.type_error(format!("Undefined trait '{}'", trait_name)));
            }
            impl_block.trait_name = Some(resolved);
        }
        for method in &mut impl_block.methods {
            self.push_type_params(&method.type_params)?;
            for param in &mut method.params {
                param.ty = self.canonicalize_type(&param.ty);
            }
            if let Some(ret) = method.return_type.clone() {
                method.return_type = Some(self.canonicalize_type(&ret));
            }
            method.trait_bounds = self.canonicalize_trait_bounds(&method.trait_bounds);
            self.pop_type_params();
        }
        self.pop_type_params();

        if let Some(trait_name) = &impl_block.trait_name {
            let trait_def = self.env.lookup_trait(trait_name).unwrap().clone();
            for trait_method in &trait_def.methods {
                let impl_method = impl_block
                    .methods
                    .iter()
                    .find(|method| method.name.rsplit(':').next() == Some(&trait_method.name))
                    .ok_or_else(|| {
                        self.type_error(format!(
                            "Trait '{}' requires method '{}'",
                            trait_name, trait_method.name
                        ))
                    })?;
                if !impl_method.type_params.is_empty() || !impl_method.trait_bounds.is_empty() {
                    return Err(self.type_error(format!(
                        "Trait method implementation '{}' cannot introduce generic parameters or bounds",
                        trait_method.name
                    )));
                }
                let trait_has_self = trait_method.params.iter().any(|param| param.is_self);
                let impl_has_self = impl_method.params.iter().any(|param| param.is_self);
                if trait_has_self != impl_has_self {
                    return Err(self.type_error(format!(
                        "Method '{}' has an incompatible self receiver",
                        trait_method.name
                    )));
                }
                let trait_params: Vec<_> = trait_method
                    .params
                    .iter()
                    .filter(|param| !param.is_self)
                    .collect();
                let impl_params: Vec<_> = impl_method
                    .params
                    .iter()
                    .filter(|param| !param.is_self)
                    .collect();
                if trait_params.len() != impl_params.len() {
                    return Err(self.type_error(format!(
                        "Method '{}' has {} parameter(s), but trait '{}' requires {}",
                        trait_method.name,
                        impl_params.len(),
                        trait_name,
                        trait_params.len()
                    )));
                }
                for (expected, actual) in trait_params.iter().zip(impl_params) {
                    let expected_type = self.canonicalize_type(&expected.ty);
                    if !self.types_equal(&expected_type, &actual.ty) {
                        return Err(self.type_error(format!(
                            "Method '{}' parameter type '{}' does not match trait type '{}'",
                            trait_method.name, actual.ty, expected_type
                        )));
                    }
                }
                let expected_return = trait_method
                    .return_type
                    .as_ref()
                    .map(|ty| self.canonicalize_type(ty))
                    .unwrap_or(Type::new(TypeKind::Unit, Self::dummy_span()));
                let actual_return = impl_method
                    .return_type
                    .clone()
                    .unwrap_or(Type::new(TypeKind::Unit, Self::dummy_span()));
                if !matches!(expected_return.kind, TypeKind::Unknown)
                    && !self.types_equal(&expected_return, &actual_return)
                {
                    return Err(self.type_error(format!(
                        "Method '{}' return type '{}' does not match trait type '{}'",
                        trait_method.name, actual_return, expected_return
                    )));
                }
            }
        }

        let type_name = match &impl_block.target_type.kind {
            TypeKind::Named(name) => self.resolve_type_key(name),
            TypeKind::GenericInstance { name, .. } => self.resolve_type_key(name),
            _ => {
                return Err(self.type_error(
                    "Impl target must be a named type when registering from Rust".to_string(),
                ))
            }
        };

        self.env.register_impl(&impl_block)?;
        for method in &impl_block.methods {
            let params: Vec<Type> = method.params.iter().map(|p| p.ty.clone()).collect();
            let return_type = method
                .return_type
                .clone()
                .unwrap_or(Type::new(TypeKind::Unit, Span::dummy()));
            let has_self = method.params.iter().any(|p| p.is_self);
            let canonical_name = if method.name.contains(':') || method.name.contains('.') {
                self.resolve_type_key(&method.name)
            } else if has_self {
                format!("{}:{}", type_name, method.name)
            } else {
                format!("{}.{}", type_name, method.name)
            };
            #[cfg(all(debug_assertions, feature = "std"))]
            eprintln!(
                "register_external_impl canonical method {} (has_self={})",
                canonical_name, has_self
            );
            let signature = FunctionSignature {
                params,
                return_type,
                is_method: has_self,
                type_params: method.type_params.clone(),
                trait_bounds: method.trait_bounds.clone(),
            };
            self.env
                .register_or_update_function(canonical_name, signature)?;
        }

        Ok(())
    }

    pub fn resolve_type_key(&self, name: &str) -> String {
        if let Some((head, tail)) = name.split_once('.') {
            if let Some(module) = &self.current_module {
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

        if self.env.lookup_struct(name).is_some()
            || self.env.lookup_enum(name).is_some()
            || self.env.lookup_trait(name).is_some()
        {
            return name.to_string();
        }

        if self.env.is_builtin_type(name) {
            return name.to_string();
        }

        if let Some(module) = &self.current_module {
            if let Some(imports) = self.imports_by_module.get(module) {
                if let Some(fq) = imports.type_aliases.get(name) {
                    return fq.clone();
                }
            }

            return format!("{}.{}", module, name);
        }

        name.to_string()
    }

    fn register_type_definition(&mut self, item: &Item) -> Result<()> {
        match &item.kind {
            ItemKind::Struct(s) => {
                let mut s2 = s.clone();
                if let Some(module) = &self.current_module {
                    if !s2.name.contains('.') {
                        s2.name = format!("{}.{}", module, s2.name);
                    }
                }

                self.push_type_params(&s2.type_params)?;
                for field in &mut s2.fields {
                    field.ty = self.canonicalize_type(&field.ty);
                    if let Some(target) = &field.weak_target {
                        field.weak_target = Some(self.canonicalize_type(target));
                    }
                }
                s2.trait_bounds = self.canonicalize_trait_bounds(&s2.trait_bounds);
                self.pop_type_params();

                self.env.register_struct(&s2)?;
            }

            ItemKind::Enum(e) => {
                let mut e2 = e.clone();
                if let Some(module) = &self.current_module {
                    if !e2.name.contains('.') {
                        e2.name = format!("{}.{}", module, e2.name);
                    }
                }

                self.push_type_params(&e2.type_params)?;
                for variant in &mut e2.variants {
                    if let Some(fields) = &mut variant.fields {
                        for field in fields {
                            *field = self.canonicalize_type(field);
                        }
                    }
                }
                e2.trait_bounds = self.canonicalize_trait_bounds(&e2.trait_bounds);
                self.pop_type_params();

                self.env.register_enum(&e2)?;
            }

            ItemKind::Trait(t) => {
                let mut t2 = t.clone();
                if let Some(module) = &self.current_module {
                    if !t2.name.contains('.') {
                        t2.name = format!("{}.{}", module, t2.name);
                    }
                }

                self.push_type_params(&t2.type_params)?;
                for method in &mut t2.methods {
                    self.push_type_params(&method.type_params)?;
                    for param in &mut method.params {
                        param.ty = self.canonicalize_type(&param.ty);
                    }
                    if let Some(ret) = method.return_type.clone() {
                        method.return_type = Some(self.canonicalize_type(&ret));
                    }
                    self.pop_type_params();
                }
                self.pop_type_params();

                self.env.register_trait(&t2)?;
            }

            ItemKind::TypeAlias {
                name,
                type_params,
                target,
            } => {
                let qname = if let Some(module) = &self.current_module {
                    if name.contains('.') {
                        name.clone()
                    } else {
                        format!("{}.{}", module, name)
                    }
                } else {
                    name.clone()
                };
                self.push_type_params(type_params)?;
                let canonical_target = self.canonicalize_type(target);
                self.pop_type_params();
                self.env.register_type_alias(
                    qname,
                    type_params.clone(),
                    canonical_target,
                )?;
            }

            ItemKind::Extern { items, .. } => {
                for ext in items {
                    match ext {
                        ExternItem::Struct(def) => {
                            self.register_external_struct(def.clone())?;
                        }
                        ExternItem::Enum(def) => {
                            self.register_external_enum(def.clone())?;
                        }
                        ExternItem::Const { name, ty } => {
                            let key = self.resolve_value_key(name);
                            self.env
                                .register_constant(key, self.canonicalize_type(ty))?;
                        }
                        ExternItem::Function { .. } => {}
                    }
                }
            }

            _ => {}
        }

        Ok(())
    }

    fn type_error(&self, message: String) -> LustError {
        LustError::TypeError { message }
    }

    fn type_error_at(&self, message: String, span: Span) -> LustError {
        if span.start_line > 0 {
            LustError::TypeErrorWithSpan {
                message,
                line: span.start_line,
                column: span.start_col,
                module: self.current_module.clone(),
            }
        } else {
            LustError::TypeError { message }
        }
    }

    fn types_equal(&self, t1: &Type, t2: &Type) -> bool {
        t1.kind == t2.kind
    }

    fn push_type_params(&mut self, params: &[String]) -> Result<()> {
        let mut scope = HashSet::new();
        for param in params {
            if !scope.insert(param.clone()) {
                return Err(self.type_error(format!(
                    "Type parameter '{}' is declared more than once",
                    param
                )));
            }
        }
        self.type_param_scopes.push(scope);
        Ok(())
    }

    fn pop_type_params(&mut self) {
        self.type_param_scopes.pop();
    }

    fn is_type_param_in_scope(&self, name: &str) -> bool {
        self.type_param_scopes
            .iter()
            .rev()
            .any(|scope| scope.contains(name))
    }

    fn canonicalize_trait_bounds(&self, bounds: &[TraitBound]) -> Vec<TraitBound> {
        bounds
            .iter()
            .map(|bound| TraitBound {
                type_param: bound.type_param.clone(),
                traits: bound
                    .traits
                    .iter()
                    .map(|name| self.resolve_type_key(name))
                    .collect(),
            })
            .collect()
    }

    pub fn canonicalize_type(&self, ty: &Type) -> Type {
        let mut expanding_aliases = HashSet::new();
        self.canonicalize_type_inner(ty, &mut expanding_aliases)
    }

    fn canonicalize_type_inner(
        &self,
        ty: &Type,
        expanding_aliases: &mut HashSet<String>,
    ) -> Type {
        use crate::ast::TypeKind as TK;
        match &ty.kind {
            TK::Named(name) if !name.contains('.') && self.is_type_param_in_scope(name) => {
                Type::new(TK::Generic(name.clone()), ty.span)
            }
            TK::Named(name) => {
                let resolved = self.resolve_type_key(name);
                if let Some((params, target)) = self.env.lookup_type_alias(&resolved) {
                    if params.is_empty() && expanding_aliases.insert(resolved.clone()) {
                        let expanded = self.canonicalize_type_inner(target, expanding_aliases);
                        expanding_aliases.remove(&resolved);
                        return expanded;
                    }
                }
                if self.env.lookup_trait(&resolved).is_some() {
                    Type::new(TK::Trait(resolved), ty.span)
                } else {
                    Type::new(TK::Named(resolved), ty.span)
                }
            }
            TK::Array(inner) => {
                Type::new(
                    TK::Array(Box::new(
                        self.canonicalize_type_inner(inner, expanding_aliases),
                    )),
                    ty.span,
                )
            }

            TK::Tuple(elements) => Type::new(
                TK::Tuple(
                    elements
                        .iter()
                        .map(|t| self.canonicalize_type_inner(t, expanding_aliases))
                        .collect(),
                ),
                ty.span,
            ),
            TK::Function {
                params,
                return_type,
            } => Type::new(
                TK::Function {
                    params: params
                        .iter()
                        .map(|param| self.canonicalize_type_inner(param, expanding_aliases))
                        .collect(),
                    return_type: Box::new(
                        self.canonicalize_type_inner(return_type, expanding_aliases),
                    ),
                },
                ty.span,
            ),
            TK::Option(inner) => {
                Type::new(
                    TK::Option(Box::new(
                        self.canonicalize_type_inner(inner, expanding_aliases),
                    )),
                    ty.span,
                )
            }

            TK::Result(ok, err) => Type::new(
                TK::Result(
                    Box::new(self.canonicalize_type_inner(ok, expanding_aliases)),
                    Box::new(self.canonicalize_type_inner(err, expanding_aliases)),
                ),
                ty.span,
            ),
            TK::Map(k, v) => Type::new(
                TK::Map(
                    Box::new(self.canonicalize_type_inner(k, expanding_aliases)),
                    Box::new(self.canonicalize_type_inner(v, expanding_aliases)),
                ),
                ty.span,
            ),
            TK::Ref(inner) => Type::new(
                TK::Ref(Box::new(
                    self.canonicalize_type_inner(inner, expanding_aliases),
                )),
                ty.span,
            ),
            TK::MutRef(inner) => {
                Type::new(
                    TK::MutRef(Box::new(
                        self.canonicalize_type_inner(inner, expanding_aliases),
                    )),
                    ty.span,
                )
            }

            TK::Pointer { mutable, pointee } => Type::new(
                TK::Pointer {
                    mutable: *mutable,
                    pointee: Box::new(
                        self.canonicalize_type_inner(pointee, expanding_aliases),
                    ),
                },
                ty.span,
            ),
            TK::GenericInstance { name, type_args } => {
                let resolved = self.resolve_type_key(name);
                let canonical_args: Vec<Type> = type_args
                    .iter()
                    .map(|arg| self.canonicalize_type_inner(arg, expanding_aliases))
                    .collect();
                if let Some((params, target)) = self.env.lookup_type_alias(&resolved) {
                    if params.len() == canonical_args.len()
                        && expanding_aliases.insert(resolved.clone())
                    {
                        let bindings = params
                            .iter()
                            .cloned()
                            .zip(canonical_args)
                            .collect();
                        let substituted = self.substitute_type(target, &bindings);
                        let expanded =
                            self.canonicalize_type_inner(&substituted, expanding_aliases);
                        expanding_aliases.remove(&resolved);
                        return expanded;
                    }
                }
                Type::new(
                    TK::GenericInstance {
                        name: resolved,
                        type_args: canonical_args,
                    },
                    ty.span,
                )
            }
            TK::Union(types) => Type::new(
                TK::Union(
                    types
                        .iter()
                        .map(|ty| self.canonicalize_type_inner(ty, expanding_aliases))
                        .collect(),
                ),
                ty.span,
            ),
            TK::Trait(name) => Type::new(TK::Trait(self.resolve_type_key(name)), ty.span),
            TK::TraitBound(traits) => Type::new(
                TK::TraitBound(
                    traits
                        .iter()
                        .map(|name| self.resolve_type_key(name))
                        .collect(),
                ),
                ty.span,
            ),
            _ => ty.clone(),
        }
    }

    fn infer_type_arguments(
        &self,
        expected: &Type,
        actual: &Type,
        type_params: &[String],
        bindings: &mut HashMap<String, Type>,
    ) -> Result<()> {
        if let TypeKind::Generic(name) = &expected.kind {
            if type_params.iter().any(|param| param == name) {
                if let Some(bound) = bindings.get(name) {
                    return self.unify(bound, actual);
                }
                bindings.insert(name.clone(), self.canonicalize_type(actual));
                return Ok(());
            }
        }

        match (&expected.kind, &actual.kind) {
            (TypeKind::Array(expected), TypeKind::Array(actual))
            | (TypeKind::Option(expected), TypeKind::Option(actual))
            | (TypeKind::Ref(expected), TypeKind::Ref(actual))
            | (TypeKind::MutRef(expected), TypeKind::MutRef(actual)) => {
                self.infer_type_arguments(expected, actual, type_params, bindings)
            }
            (TypeKind::Map(expected_key, expected_value), TypeKind::Map(actual_key, actual_value))
            | (
                TypeKind::Result(expected_key, expected_value),
                TypeKind::Result(actual_key, actual_value),
            ) => {
                self.infer_type_arguments(expected_key, actual_key, type_params, bindings)?;
                self.infer_type_arguments(expected_value, actual_value, type_params, bindings)
            }
            (TypeKind::Tuple(expected), TypeKind::Tuple(actual))
            | (TypeKind::Union(expected), TypeKind::Union(actual)) => {
                if expected.len() != actual.len() {
                    return Err(self.type_error(format!(
                        "Type arity mismatch: expected {} element(s), got {}",
                        expected.len(),
                        actual.len()
                    )));
                }
                for (expected, actual) in expected.iter().zip(actual.iter()) {
                    self.infer_type_arguments(expected, actual, type_params, bindings)?;
                }
                Ok(())
            }
            (
                TypeKind::Function {
                    params: expected_params,
                    return_type: expected_return,
                },
                TypeKind::Function {
                    params: actual_params,
                    return_type: actual_return,
                },
            ) => {
                if expected_params.len() != actual_params.len() {
                    return self.unify(expected, actual);
                }
                for (expected, actual) in expected_params.iter().zip(actual_params.iter()) {
                    self.infer_type_arguments(expected, actual, type_params, bindings)?;
                }
                self.infer_type_arguments(
                    expected_return,
                    actual_return,
                    type_params,
                    bindings,
                )
            }
            (
                TypeKind::Pointer {
                    mutable: expected_mutable,
                    pointee: expected_pointee,
                },
                TypeKind::Pointer {
                    mutable: actual_mutable,
                    pointee: actual_pointee,
                },
            ) if expected_mutable == actual_mutable => self.infer_type_arguments(
                expected_pointee,
                actual_pointee,
                type_params,
                bindings,
            ),
            (
                TypeKind::GenericInstance {
                    name: expected_name,
                    type_args: expected_args,
                },
                TypeKind::GenericInstance {
                    name: actual_name,
                    type_args: actual_args,
                },
            ) if expected_name == actual_name && expected_args.len() == actual_args.len() => {
                for (expected, actual) in expected_args.iter().zip(actual_args.iter()) {
                    self.infer_type_arguments(expected, actual, type_params, bindings)?;
                }
                Ok(())
            }
            _ => self.unify(expected, actual),
        }
    }

    fn bind_type_argument(
        &self,
        name: &str,
        concrete: &Type,
        bindings: &mut HashMap<String, Type>,
    ) -> Result<()> {
        if let Some(existing) = bindings.get(name) {
            self.unify(existing, concrete)
        } else {
            bindings.insert(name.to_string(), concrete.clone());
            Ok(())
        }
    }

    fn validate_type(&self, ty: &Type) -> Result<()> {
        match &ty.kind {
            TypeKind::Named(name) => {
                let generic_arity = self
                    .env
                    .lookup_struct(name)
                    .map(|def| def.type_params.len())
                    .or_else(|| self.env.lookup_enum(name).map(|def| def.type_params.len()))
                    .or_else(|| {
                        self.env
                            .lookup_type_alias(name)
                            .map(|(params, _)| params.len())
                    })
                    .unwrap_or(0);
                if generic_arity > 0 {
                    return Err(self.type_error_at(
                        format!(
                            "Generic type '{}' requires {} type argument(s)",
                            name, generic_arity
                        ),
                        ty.span,
                    ));
                }
                if self.env.lookup_struct(name).is_none()
                    && self.env.lookup_enum(name).is_none()
                    && self.env.lookup_type_alias(name).is_none()
                    && !self.env.is_builtin_type(name)
                {
                    return Err(self.type_error_at(format!("Undefined type '{}'", name), ty.span));
                }
            }
            TypeKind::Generic(name) => {
                if !self.is_type_param_in_scope(name) {
                    return Err(self.type_error_at(
                        format!("Undeclared type parameter '{}'", name),
                        ty.span,
                    ));
                }
            }
            TypeKind::Array(inner)
            | TypeKind::Option(inner)
            | TypeKind::Ref(inner)
            | TypeKind::MutRef(inner) => self.validate_type(inner)?,
            TypeKind::Map(key, value) | TypeKind::Result(key, value) => {
                self.validate_type(key)?;
                self.validate_type(value)?;
            }
            TypeKind::Function {
                params,
                return_type,
            } => {
                for param in params {
                    self.validate_type(param)?;
                }
                self.validate_type(return_type)?;
            }
            TypeKind::Tuple(elements) | TypeKind::Union(elements) => {
                for element in elements {
                    self.validate_type(element)?;
                }
            }
            TypeKind::Pointer { pointee, .. } => self.validate_type(pointee)?,
            TypeKind::GenericInstance { name, type_args } => {
                let (arity, bounds) = if let Some(def) = self.env.lookup_struct(name) {
                    (def.type_params.len(), Some((&def.type_params, &def.trait_bounds)))
                } else if let Some(def) = self.env.lookup_enum(name) {
                    (def.type_params.len(), Some((&def.type_params, &def.trait_bounds)))
                } else if let Some((params, _)) = self.env.lookup_type_alias(name) {
                    (params.len(), None)
                } else {
                    return Err(self
                        .type_error_at(format!("Undefined generic type '{}'", name), ty.span));
                };
                if arity != type_args.len() {
                    return Err(self.type_error_at(
                        format!(
                            "Type '{}' expects {} type argument(s), got {}",
                            name,
                            arity,
                            type_args.len()
                        ),
                        ty.span,
                    ));
                }
                for arg in type_args {
                    self.validate_type(arg)?;
                }
                if let Some((type_params, trait_bounds)) = bounds {
                    let bindings = type_params
                        .iter()
                        .cloned()
                        .zip(type_args.iter().cloned())
                        .collect();
                    self.validate_generic_bindings(type_params, trait_bounds, &bindings)?;
                }
            }
            TypeKind::Trait(name) => {
                if self.env.lookup_trait(name).is_none() {
                    return Err(self.type_error_at(format!("Undefined trait '{}'", name), ty.span));
                }
            }
            TypeKind::TraitBound(names) => {
                for name in names {
                    if self.env.lookup_trait(name).is_none() {
                        return Err(
                            self.type_error_at(format!("Undefined trait '{}'", name), ty.span)
                        );
                    }
                }
            }
            TypeKind::Int
            | TypeKind::Float
            | TypeKind::String
            | TypeKind::Bool
            | TypeKind::Unknown
            | TypeKind::Unit
            | TypeKind::Infer => {}
        }
        Ok(())
    }

    fn validate_trait_bounds(
        &self,
        type_params: &[String],
        bounds: &[TraitBound],
    ) -> Result<()> {
        for bound in bounds {
            if !type_params.iter().any(|param| param == &bound.type_param) {
                return Err(self.type_error(format!(
                    "Trait bound references undeclared type parameter '{}'",
                    bound.type_param
                )));
            }
            for trait_name in &bound.traits {
                let resolved = self.resolve_type_key(trait_name);
                if self.env.lookup_trait(&resolved).is_none() {
                    return Err(self.type_error(format!(
                        "Undefined trait '{}' in bound for '{}'",
                        trait_name, bound.type_param
                    )));
                }
            }
        }
        Ok(())
    }

    fn validate_type_alias_cycle(&self, name: &str) -> Result<()> {
        fn visit_type(
            checker: &TypeChecker,
            ty: &Type,
            visiting: &mut HashSet<String>,
        ) -> bool {
            match &ty.kind {
                TypeKind::Named(name) | TypeKind::GenericInstance { name, .. }
                    if checker.env.lookup_type_alias(name).is_some() =>
                {
                    if !visiting.insert(name.clone()) {
                        return true;
                    }
                    let (_, target) = checker.env.lookup_type_alias(name).unwrap();
                    let cyclic = visit_type(checker, target, visiting);
                    visiting.remove(name);
                    if cyclic {
                        return true;
                    }
                    if let TypeKind::GenericInstance { type_args, .. } = &ty.kind {
                        return type_args
                            .iter()
                            .any(|arg| visit_type(checker, arg, visiting));
                    }
                    false
                }
                TypeKind::Array(inner)
                | TypeKind::Option(inner)
                | TypeKind::Ref(inner)
                | TypeKind::MutRef(inner) => visit_type(checker, inner, visiting),
                TypeKind::Map(key, value) | TypeKind::Result(key, value) => {
                    visit_type(checker, key, visiting)
                        || visit_type(checker, value, visiting)
                }
                TypeKind::Function {
                    params,
                    return_type,
                } => {
                    params
                        .iter()
                        .any(|param| visit_type(checker, param, visiting))
                        || visit_type(checker, return_type, visiting)
                }
                TypeKind::Tuple(elements) | TypeKind::Union(elements) => elements
                    .iter()
                    .any(|element| visit_type(checker, element, visiting)),
                TypeKind::Pointer { pointee, .. } => visit_type(checker, pointee, visiting),
                TypeKind::GenericInstance { type_args, .. } => type_args
                    .iter()
                    .any(|arg| visit_type(checker, arg, visiting)),
                _ => false,
            }
        }

        let Some((_, target)) = self.env.lookup_type_alias(name) else {
            return Ok(());
        };
        let mut visiting = HashSet::new();
        visiting.insert(name.to_string());
        if visit_type(self, target, &mut visiting) {
            return Err(self.type_error(format!(
                "Recursive type alias '{}' is not supported",
                name
            )));
        }
        Ok(())
    }

    fn substitute_type(&self, ty: &Type, bindings: &HashMap<String, Type>) -> Type {
        let kind = match &ty.kind {
            TypeKind::Generic(name) => {
                if let Some(bound) = bindings.get(name) {
                    return bound.clone();
                }
                TypeKind::Generic(name.clone())
            }
            TypeKind::Array(inner) => {
                TypeKind::Array(Box::new(self.substitute_type(inner, bindings)))
            }
            TypeKind::Map(key, value) => TypeKind::Map(
                Box::new(self.substitute_type(key, bindings)),
                Box::new(self.substitute_type(value, bindings)),
            ),
            TypeKind::Function {
                params,
                return_type,
            } => TypeKind::Function {
                params: params
                    .iter()
                    .map(|param| self.substitute_type(param, bindings))
                    .collect(),
                return_type: Box::new(self.substitute_type(return_type, bindings)),
            },
            TypeKind::Tuple(elements) => TypeKind::Tuple(
                elements
                    .iter()
                    .map(|element| self.substitute_type(element, bindings))
                    .collect(),
            ),
            TypeKind::Option(inner) => {
                TypeKind::Option(Box::new(self.substitute_type(inner, bindings)))
            }
            TypeKind::Result(ok, err) => TypeKind::Result(
                Box::new(self.substitute_type(ok, bindings)),
                Box::new(self.substitute_type(err, bindings)),
            ),
            TypeKind::Ref(inner) => {
                TypeKind::Ref(Box::new(self.substitute_type(inner, bindings)))
            }
            TypeKind::MutRef(inner) => {
                TypeKind::MutRef(Box::new(self.substitute_type(inner, bindings)))
            }
            TypeKind::Pointer { mutable, pointee } => TypeKind::Pointer {
                mutable: *mutable,
                pointee: Box::new(self.substitute_type(pointee, bindings)),
            },
            TypeKind::GenericInstance { name, type_args } => TypeKind::GenericInstance {
                name: name.clone(),
                type_args: type_args
                    .iter()
                    .map(|arg| self.substitute_type(arg, bindings))
                    .collect(),
            },
            TypeKind::Union(types) => TypeKind::Union(
                types
                    .iter()
                    .map(|ty| self.substitute_type(ty, bindings))
                    .collect(),
            ),
            _ => ty.kind.clone(),
        };
        Type::new(kind, ty.span)
    }

    fn validate_generic_call(
        &self,
        signature: &FunctionSignature,
        bindings: &HashMap<String, Type>,
    ) -> Result<()> {
        self.validate_generic_bindings(
            &signature.type_params,
            &signature.trait_bounds,
            bindings,
        )
    }

    fn validate_generic_bindings(
        &self,
        type_params: &[String],
        trait_bounds: &[TraitBound],
        bindings: &HashMap<String, Type>,
    ) -> Result<()> {
        for type_param in type_params {
            if !bindings.contains_key(type_param) {
                return Err(self.type_error(format!(
                    "Cannot infer type parameter '{}' from this call",
                    type_param
                )));
            }
        }

        for bound in trait_bounds {
            let concrete = bindings.get(&bound.type_param).ok_or_else(|| {
                self.type_error(format!(
                    "Cannot infer bounded type parameter '{}' from this call",
                    bound.type_param
                ))
            })?;
            for trait_name in &bound.traits {
                if !self.type_satisfies_trait(concrete, trait_name) {
                    return Err(self.type_error(format!(
                        "Type '{}' does not implement required trait '{}'",
                        concrete, trait_name
                    )));
                }
            }
        }
        Ok(())
    }

    fn instantiate_nominal_type(
        &self,
        name: String,
        type_params: &[String],
        trait_bounds: &[TraitBound],
        bindings: &HashMap<String, Type>,
        span: Span,
    ) -> Result<Type> {
        if type_params.is_empty() {
            return Ok(Type::new(TypeKind::Named(name), span));
        }
        self.validate_generic_bindings(type_params, trait_bounds, bindings)?;
        let mut type_args = Vec::with_capacity(type_params.len());
        for type_param in type_params {
            type_args.push(bindings[type_param].clone());
        }
        Ok(Type::new(
            TypeKind::GenericInstance { name, type_args },
            span,
        ))
    }

    fn type_satisfies_trait(&self, ty: &Type, trait_name: &str) -> bool {
        match &ty.kind {
            TypeKind::Generic(type_param) => self
                .current_trait_bounds
                .get(type_param)
                .is_some_and(|bounds| bounds.iter().any(|bound| bound == trait_name)),
            TypeKind::Trait(actual) => actual == trait_name,
            _ => self.env.type_implements_trait(ty, trait_name),
        }
    }

    fn unify(&self, expected: &Type, actual: &Type) -> Result<()> {
        let span = if actual.span.start_line > 0 {
            Some(actual.span)
        } else if expected.span.start_line > 0 {
            Some(expected.span)
        } else {
            None
        };
        self.unify_at(expected, actual, span)
    }

    fn unify_invariant(&self, expected: &Type, actual: &Type) -> Result<()> {
        if matches!(expected.kind, TypeKind::Unknown | TypeKind::Infer)
            || matches!(actual.kind, TypeKind::Unknown | TypeKind::Infer)
            || self.types_equal(expected, actual)
        {
            Ok(())
        } else {
            Err(self.type_error_at(
                format!("Type mismatch: expected '{}', got '{}'", expected, actual),
                if actual.span.start_line > 0 {
                    actual.span
                } else {
                    expected.span
                },
            ))
        }
    }

    fn unify_at(&self, expected: &Type, actual: &Type, span: Option<Span>) -> Result<()> {
        if matches!(expected.kind, TypeKind::Unknown) || matches!(actual.kind, TypeKind::Unknown) {
            return Ok(());
        }

        if matches!(expected.kind, TypeKind::Infer) || matches!(actual.kind, TypeKind::Infer) {
            return Ok(());
        }

        if self.is_lua_multi_return(expected) || self.is_lua_multi_return(actual) {
            return Ok(());
        }

        if matches!(&expected.kind, TypeKind::Named(name) if name == "LuaValue")
            || matches!(&actual.kind, TypeKind::Named(name) if name == "LuaValue")
        {
            return Ok(());
        }

        match (&expected.kind, &actual.kind) {
            (TypeKind::Trait(expected_trait), TypeKind::Trait(actual_trait))
                if expected_trait == actual_trait =>
            {
                return Ok(())
            }
            (TypeKind::Trait(trait_name), _) if self.type_satisfies_trait(actual, trait_name) => {
                return Ok(())
            }
            (TypeKind::Union(expected_types), TypeKind::Union(actual_types)) => {
                if expected_types.len() != actual_types.len() {
                    return Err(self.type_error(format!(
                        "Union types have different number of members: expected {}, got {}",
                        expected_types.len(),
                        actual_types.len()
                    )));
                }

                for exp_type in expected_types {
                    let mut found = false;
                    for act_type in actual_types {
                        if self.types_equal(exp_type, act_type) {
                            found = true;
                            break;
                        }
                    }

                    if !found {
                        return Err(match span {
                            Some(s) => self.type_error_at(
                                format!(
                                    "Union type member '{}' not found in actual union",
                                    exp_type
                                ),
                                s,
                            ),
                            None => self.type_error(format!(
                                "Union type member '{}' not found in actual union",
                                exp_type
                            )),
                        });
                    }
                }

                return Ok(());
            }

            (TypeKind::Union(expected_types), _) => {
                for union_member in expected_types {
                    if self.unify(union_member, actual).is_ok() {
                        return Ok(());
                    }
                }

                return Err(match span {
                    Some(s) => self.type_error_at(
                        format!("Type '{}' is not compatible with union type", actual),
                        s,
                    ),
                    None => self.type_error(format!(
                        "Type '{}' is not compatible with union type",
                        actual
                    )),
                });
            }

            (_, TypeKind::Union(actual_types)) => {
                for union_member in actual_types {
                    self.unify(expected, union_member)?;
                }

                return Ok(());
            }

            _ => {}
        }

        match (&expected.kind, &actual.kind) {
            (TypeKind::Tuple(expected_elems), TypeKind::Tuple(actual_elems)) => {
                if expected_elems.len() != actual_elems.len() {
                    return Err(match span {
                        Some(s) => self.type_error_at(
                            format!(
                                "Tuple length mismatch: expected {} element(s), got {}",
                                expected_elems.len(),
                                actual_elems.len()
                            ),
                            s,
                        ),
                        None => self.type_error(format!(
                            "Tuple length mismatch: expected {} element(s), got {}",
                            expected_elems.len(),
                            actual_elems.len()
                        )),
                    });
                }

                for (exp_elem, act_elem) in expected_elems.iter().zip(actual_elems.iter()) {
                    self.unify(exp_elem, act_elem)?;
                }

                return Ok(());
            }

            (TypeKind::Tuple(_), _) | (_, TypeKind::Tuple(_)) => {
                return Err(match span {
                    Some(s) => self.type_error_at(
                        format!("Tuple type is not compatible with type '{}'", actual),
                        s,
                    ),
                    None => self.type_error(format!(
                        "Tuple type is not compatible with type '{}'",
                        actual
                    )),
                })
            }

            (TypeKind::Named(name), TypeKind::Array(_))
            | (TypeKind::Array(_), TypeKind::Named(name))
                if name == "Array" =>
            {
                return Ok(());
            }

            (TypeKind::Array(exp_el), TypeKind::Array(act_el)) => {
                if matches!(exp_el.kind, TypeKind::Unknown | TypeKind::Infer)
                    || matches!(act_el.kind, TypeKind::Unknown | TypeKind::Infer)
                {
                    return Ok(());
                } else {
                    return self.unify_invariant(exp_el, act_el);
                }
            }

            (TypeKind::Map(exp_key, exp_value), TypeKind::Map(act_key, act_value)) => {
                self.unify_invariant(exp_key, act_key)?;
                return self.unify_invariant(exp_value, act_value);
            }

            (TypeKind::Named(name), TypeKind::Option(_))
            | (TypeKind::Option(_), TypeKind::Named(name))
                if name == "Option" =>
            {
                return Ok(());
            }

            (TypeKind::Option(exp_inner), TypeKind::Option(act_inner)) => {
                if matches!(exp_inner.kind, TypeKind::Unknown | TypeKind::Infer)
                    || matches!(act_inner.kind, TypeKind::Unknown | TypeKind::Infer)
                {
                    return Ok(());
                } else {
                    return self.unify(exp_inner, act_inner);
                }
            }

            (TypeKind::Named(name), TypeKind::Result(_, _))
            | (TypeKind::Result(_, _), TypeKind::Named(name))
                if name == "Result" =>
            {
                return Ok(());
            }

            (TypeKind::Result(exp_ok, exp_err), TypeKind::Result(act_ok, act_err)) => {
                if matches!(exp_ok.kind, TypeKind::Unknown | TypeKind::Infer)
                    || matches!(act_ok.kind, TypeKind::Unknown | TypeKind::Infer)
                {
                    if matches!(exp_err.kind, TypeKind::Unknown | TypeKind::Infer)
                        || matches!(act_err.kind, TypeKind::Unknown | TypeKind::Infer)
                    {
                        return Ok(());
                    } else {
                        return self.unify(exp_err, act_err);
                    }
                } else {
                    self.unify(exp_ok, act_ok)?;
                    return self.unify(exp_err, act_err);
                }
            }

            _ => {}
        }

        if self.types_equal(expected, actual) {
            Ok(())
        } else {
            Err(match span {
                Some(s) => self.type_error_at(
                    format!("Type mismatch: expected '{}', got '{}'", expected, actual),
                    s,
                ),
                None => self.type_error(format!(
                    "Type mismatch: expected '{}', got '{}'",
                    expected, actual
                )),
            })
        }
    }

    fn types_compatible(&self, expected: &Type, actual: &Type) -> bool {
        if matches!(expected.kind, TypeKind::Unknown) || matches!(actual.kind, TypeKind::Unknown) {
            return true;
        }

        if matches!(expected.kind, TypeKind::Infer) || matches!(actual.kind, TypeKind::Infer) {
            return true;
        }

        match (&expected.kind, &actual.kind) {
            (TypeKind::Generic(_), TypeKind::Generic(_)) => return true,
            (TypeKind::Generic(_), _) | (_, TypeKind::Generic(_)) => return true,
            _ => {}
        }

        match (&expected.kind, &actual.kind) {
            (TypeKind::Array(e1), TypeKind::Array(e2)) => {
                return self.types_compatible(e1, e2);
            }

            (TypeKind::Named(name), TypeKind::Array(_))
            | (TypeKind::Array(_), TypeKind::Named(name))
                if name == "Array" =>
            {
                return true;
            }

            _ => {}
        }

        match (&expected.kind, &actual.kind) {
            (TypeKind::Map(k1, v1), TypeKind::Map(k2, v2)) => {
                return self.types_compatible(k1, k2) && self.types_compatible(v1, v2);
            }

            _ => {}
        }

        match (&expected.kind, &actual.kind) {
            (TypeKind::Option(t1), TypeKind::Option(t2)) => {
                return self.types_compatible(t1, t2);
            }

            (TypeKind::Named(name), TypeKind::Option(_))
            | (TypeKind::Option(_), TypeKind::Named(name))
                if name == "Option" =>
            {
                return true;
            }

            _ => {}
        }

        match (&expected.kind, &actual.kind) {
            (TypeKind::Result(ok1, err1), TypeKind::Result(ok2, err2)) => {
                return self.types_compatible(ok1, ok2) && self.types_compatible(err1, err2);
            }

            (TypeKind::Named(name), TypeKind::Result(_, _))
            | (TypeKind::Result(_, _), TypeKind::Named(name))
                if name == "Result" =>
            {
                return true;
            }

            _ => {}
        }

        match (&expected.kind, &actual.kind) {
            (
                TypeKind::Function {
                    params: p1,
                    return_type: r1,
                },
                TypeKind::Function {
                    params: p2,
                    return_type: r2,
                },
            ) => {
                if p1.len() != p2.len() {
                    return false;
                }

                for (t1, t2) in p1.iter().zip(p2.iter()) {
                    if !self.types_compatible(t1, t2) {
                        return false;
                    }
                }

                return self.types_compatible(r1, r2);
            }

            _ => {}
        }

        self.types_equal(expected, actual)
    }

    fn unify_with_bounds(&self, expected: &Type, actual: &Type) -> Result<()> {
        if let TypeKind::Generic(type_param) = &expected.kind {
            if let Some(trait_names) = self.current_trait_bounds.get(type_param) {
                for trait_name in trait_names {
                    if !self.env.type_implements_trait(actual, trait_name) {
                        return Err(self.type_error(format!(
                            "Type '{}' does not implement required trait '{}'",
                            actual, trait_name
                        )));
                    }
                }

                return Ok(());
            }

            return Ok(());
        }

        self.unify(expected, actual)
    }

    fn is_lua_multi_return(&self, ty: &Type) -> bool {
        if let TypeKind::Array(inner) = &ty.kind {
            return matches!(inner.kind, TypeKind::Unknown)
                || matches!(&inner.kind, TypeKind::Named(name) if name == "LuaValue");
        }
        false
    }

    fn record_short_circuit_info(&mut self, span: Span, info: &ShortCircuitInfo) {
        if self.low_memory_mode {
            return; // Skip recording in low memory mode
        }
        let truthy = info.truthy.as_ref().map(|ty| self.canonicalize_type(ty));
        let falsy = info.falsy.as_ref().map(|ty| self.canonicalize_type(ty));
        let option_inner = info
            .option_inner
            .as_ref()
            .map(|ty| self.canonicalize_type(ty));
        let module_key = self.current_module_key();
        self.short_circuit_info
            .entry(module_key)
            .or_default()
            .insert(
                span,
                ShortCircuitInfo {
                    truthy,
                    falsy,
                    option_inner,
                },
            );
    }

    fn short_circuit_profile(&self, expr: &Expr, ty: &Type) -> ShortCircuitInfo {
        let module_key = self
            .current_module
            .as_ref()
            .map(String::as_str)
            .unwrap_or("");
        if let Some(module_map) = self.short_circuit_info.get(module_key) {
            if let Some(info) = module_map.get(&expr.span) {
                return info.clone();
            }
        }

        ShortCircuitInfo {
            truthy: if self.type_can_be_truthy(ty) {
                Some(self.canonicalize_type(ty))
            } else {
                None
            },
            falsy: self.extract_falsy_type(ty),
            option_inner: None,
        }
    }

    fn current_module_key(&self) -> String {
        self.current_module
            .as_ref()
            .cloned()
            .unwrap_or_else(|| "".to_string())
    }

    fn clear_option_for_span(&mut self, span: Span) {
        let module_key = self.current_module_key();
        if let Some(module_map) = self.short_circuit_info.get_mut(&module_key) {
            if let Some(info) = module_map.get_mut(&span) {
                info.option_inner = None;
            }
        }
    }

    fn type_can_be_truthy(&self, ty: &Type) -> bool {
        match &ty.kind {
            TypeKind::Union(members) => {
                members.iter().any(|member| self.type_can_be_truthy(member))
            }
            TypeKind::Bool => true,
            TypeKind::Unknown => true,
            _ => true,
        }
    }

    fn type_can_be_falsy(&self, ty: &Type) -> bool {
        match &ty.kind {
            TypeKind::Union(members) => members.iter().any(|member| self.type_can_be_falsy(member)),
            TypeKind::Bool => true,
            TypeKind::Unknown => true,
            TypeKind::Option(_) => true,
            _ => false,
        }
    }

    fn extract_falsy_type(&self, ty: &Type) -> Option<Type> {
        match &ty.kind {
            TypeKind::Bool => Some(Type::new(TypeKind::Bool, ty.span)),
            TypeKind::Unknown => Some(Type::new(TypeKind::Unknown, ty.span)),
            TypeKind::Option(inner) => Some(Type::new(
                TypeKind::Option(Box::new(self.canonicalize_type(inner))),
                ty.span,
            )),
            TypeKind::Union(members) => {
                let mut parts = Vec::new();
                for member in members {
                    if let Some(part) = self.extract_falsy_type(member) {
                        parts.push(part);
                    }
                }
                self.merge_optional_types(parts)
            }
            _ => None,
        }
    }

    fn merge_optional_types(&self, types: Vec<Type>) -> Option<Type> {
        if types.is_empty() {
            return None;
        }

        Some(self.make_union_from_types(types))
    }

    fn make_union_from_types(&self, types: Vec<Type>) -> Type {
        let mut flat: Vec<Type> = Vec::new();
        for ty in types {
            let canonical = self.canonicalize_type(&ty);
            match &canonical.kind {
                TypeKind::Union(members) => {
                    for member in members {
                        self.push_unique_type(&mut flat, member.clone());
                    }
                }
                _ => self.push_unique_type(&mut flat, canonical),
            }
        }

        match flat.len() {
            0 => Type::new(TypeKind::Unknown, Self::dummy_span()),
            1 => flat.into_iter().next().unwrap(),
            _ => Type::new(TypeKind::Union(flat), Self::dummy_span()),
        }
    }

    fn push_unique_type(&self, list: &mut Vec<Type>, candidate: Type) {
        if !list
            .iter()
            .any(|existing| self.types_equal(existing, &candidate))
        {
            list.push(candidate);
        }
    }

    fn combine_truthy_falsy(&self, truthy: Option<Type>, falsy: Option<Type>) -> Type {
        match (truthy, falsy) {
            (Some(t), Some(f)) => self.make_union_from_types(vec![t, f]),
            (Some(t), None) => t,
            (None, Some(f)) => f,
            (None, None) => Type::new(TypeKind::Unknown, Self::dummy_span()),
        }
    }

    fn is_bool_like(&self, ty: &Type) -> bool {
        match &ty.kind {
            TypeKind::Bool => true,
            TypeKind::Union(members) => members.iter().all(|member| self.is_bool_like(member)),
            _ => false,
        }
    }

    fn option_inner_type<'a>(&self, ty: &'a Type) -> Option<&'a Type> {
        match &ty.kind {
            TypeKind::Option(inner) => Some(inner.as_ref()),
            TypeKind::Union(members) => {
                for member in members {
                    if let Some(inner) = self.option_inner_type(member) {
                        return Some(inner);
                    }
                }
                None
            }
            _ => None,
        }
    }

    fn should_optionize(&self, left: &Type, right: &Type) -> bool {
        self.is_bool_like(left)
            && !self.is_bool_like(right)
            && self.option_inner_type(right).is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{intern::Interner, lexer::Lexer, parser::Parser};
    #[cfg(feature = "std")]
    use std::sync::Mutex;

    #[cfg(feature = "std")]
    static CHECK_LOCK: Mutex<()> = Mutex::new(());

    fn check(source: &str) -> Result<()> {
        #[cfg(feature = "std")]
        let _guard = CHECK_LOCK.lock().unwrap();
        let mut interner = Interner::new();
        let mut lexer = Lexer::new(source, &mut interner);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(tokens);
        let items = parser.parse()?;
        TypeChecker::new().check_module(&items)
    }

    #[test]
    fn single_letter_nominal_type_is_not_a_generic() {
        check(
            "struct K\n  x: int\nend\n\
             local value: K = K { x = 7 }\n\
             local x: int = value.x\n",
        )
        .unwrap();
    }

    #[test]
    fn generic_function_infers_and_substitutes_multi_letter_parameter() {
        check(
            "function identity<Element>(value: Element): Element\n\
               return value\n\
             end\n\
             local number: int = identity(42)\n\
             local text: string = identity(\"hello\")\n",
        )
        .unwrap();
    }

    #[test]
    fn generic_function_infers_nested_parameter() {
        check(
            "function first<Element>(values: Array<Element>): Element\n\
               return values[0]:unwrap()\n\
             end\n\
             local number: int = first([42])\n",
        )
        .unwrap();
    }

    #[test]
    fn generic_struct_infers_type_and_substitutes_fields() {
        check(
            "struct Box<Item>\n\
               value: Item\n\
             end\n\
             local boxed: Box<int> = Box { value = 42 }\n\
             local value: int = boxed.value\n",
        )
        .unwrap();
    }

    #[test]
    fn generic_impl_substitutes_receiver_and_method_types() {
        check(
            "struct Box<Item>\n\
               value: Item\n\
             end\n\
             impl<Item> Box<Item>\n\
               function get(self): Item\n\
                 return self.value\n\
               end\n\
               function replace<Next>(self, value: Next): Next\n\
                 return value\n\
               end\n\
             end\n\
             local boxed: Box<int> = Box { value = 42 }\n\
             local value: int = boxed:get()\n\
             local text: string = boxed:replace<string>(\"hello\")\n",
        )
        .unwrap();
    }

    #[test]
    fn explicit_function_type_arguments_bind_uninferred_parameters() {
        check(
            "function tagged<Tag>(value: int): int\n\
               return value\n\
             end\n\
             local value: int = tagged<string>(42)\n",
        )
        .unwrap();

        let error = check(
            "function tagged<Tag>(value: int): int\n\
               return value\n\
             end\n\
             local value = tagged(42)\n",
        )
        .unwrap_err();
        assert!(error.to_string().contains("Cannot infer type parameter 'Tag'"));
    }

    #[test]
    fn generic_aliases_expand_with_scoped_parameters() {
        check(
            "type Pair<Item> = (Item, Item)\n\
             type Count = int\n\
             local pair: Pair<int> = (1, 2)\n\
             local count: Count = 3\n",
        )
        .unwrap();
    }

    #[test]
    fn recursive_type_aliases_are_rejected_without_recursing() {
        let direct = check("type Loop = Loop\n").unwrap_err();
        assert!(direct.to_string().contains("Recursive type alias"));

        let mutual = check("type Left = Right\ntype Right = Left\n").unwrap_err();
        assert!(mutual.to_string().contains("Recursive type alias"));
    }

    #[test]
    fn optional_generic_fields_prefer_the_full_option_shape() {
        check(
            "struct Holder<Item>\n\
               value: Option<Item>\n\
             end\n\
             local holder = Holder { value = Option.Some(\"hello\") }\n\
             local text: string = holder.value:unwrap()\n",
        )
        .unwrap();
    }

    #[test]
    fn generic_unit_variants_accept_explicit_or_contextual_arguments() {
        check(
            "enum Maybe<Item>\n\
               None\n\
             end\n\
             local explicit: Maybe<int> = Maybe.None<int>()\n\
             local contextual: Maybe<string> = Maybe.None\n",
        )
        .unwrap();
    }

    #[test]
    fn specialized_generic_impls_are_rejected_under_erasure() {
        let error = check(
            "struct Box<Item>\n\
               value: Item\n\
             end\n\
             impl Box<int>\n\
               function get(self): int\n\
                 return self.value\n\
               end\n\
             end\n",
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("Specialized generic impls are not supported"));
    }

    #[test]
    fn generic_impl_cannot_weaken_a_trait_method_signature() {
        let error = check(
            "trait IntSource\n\
               function get(self): int\n\
             end\n\
             struct Box<Item>\n\
               value: Item\n\
             end\n\
             impl<Item> IntSource for Box<Item>\n\
               function get(self): Item\n\
                 return self.value\n\
               end\n\
             end\n",
        )
        .unwrap_err();
        assert!(error.to_string().contains("returns 'Item'"));
    }

    #[test]
    fn invalid_generic_annotations_are_rejected_without_initializers() {
        let error = check(
            "struct Box<Item>\n\
               value: Item\n\
             end\n\
             local value: Box<int, string>\n",
        )
        .unwrap_err();
        assert!(error.to_string().contains("expects 1 type argument"));
    }

    #[test]
    fn spaced_comparisons_are_not_parsed_as_generic_calls() {
        check(
            "local a: int = 1\n\
             local b: int = 2\n\
             local c: int = 3\n\
             local result: bool = a < b and c > (b)\n",
        )
        .unwrap();
    }

    #[test]
    fn conditional_generic_impl_is_rejected_under_erasure() {
        let error = check(
            "struct Box<Item>\n\
               value: Item\n\
             end\n\
             impl<Item: ToString> Box<Item>\n\
               function text(self): string\n\
                 return self.value:to_string()\n\
               end\n\
             end\n\
             local boxed: Box<int> = Box { value = 42 }\n\
             local text: string = boxed:text()\n",
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("Conditional generic impls are not supported"));
    }

    #[test]
    fn undeclared_type_parameters_are_rejected_as_undefined_types() {
        let error = check(
            "function broken(value: Missing): Missing\n\
               return value\n\
             end\n",
        )
        .unwrap_err();
        assert!(error.to_string().contains("Undefined type 'Missing'"));
    }

    #[test]
    fn generic_enum_constructor_infers_type_arguments() {
        check(
            "enum Boxed<Item>\n\
               Value(Item)\n\
             end\n\
             local boxed: Boxed<int> = Boxed.Value(42)\n",
        )
        .unwrap();
    }

    #[test]
    fn repeated_generic_parameter_must_have_one_type() {
        let error = check(
            "function choose<T>(left: T, right: T): T\n\
               return left\n\
             end\n\
             local value = choose(1, \"wrong\")\n",
        )
        .unwrap_err();
        assert!(error.to_string().contains("Argument 2"));
    }

    #[test]
    fn generic_trait_bounds_are_enforced_at_call_site() {
        let valid =
            "trait Drawable\n\
               function draw(self): string\n\
             end\n\
             struct Circle\n\
             end\n\
             impl Drawable for Circle\n\
               function draw(self): string\n\
                 return \"circle\"\n\
               end\n\
             end\n\
             function render<Item: Drawable>(item: Item): string\n\
               return item:draw()\n\
             end\n";
        check(&format!("{}local output: string = render(Circle {{}})\n", valid)).unwrap();

        let error = check(&format!("{}local output = render(1)\n", valid)).unwrap_err();
        assert!(error.to_string().contains("does not implement required trait"));
    }

    #[test]
    fn bare_trait_name_accepts_any_implementor() {
        check(
            "trait Drawable\n\
               function draw(self): string\n\
             end\n\
             struct Circle\n\
             end\n\
             impl Drawable for Circle\n\
               function draw(self): string\n\
                 return \"circle\"\n\
               end\n\
             end\n\
             local drawable: Drawable = Circle {}\n\
             local output: string = drawable:draw()\n",
        )
        .unwrap();
    }

    #[test]
    fn mutable_containers_remain_invariant_over_trait_values() {
        let declarations =
            "trait Drawable\n\
               function draw(self): string\n\
             end\n\
             struct Circle\n\
             end\n\
             impl Drawable for Circle\n\
               function draw(self): string\n\
                 return \"circle\"\n\
               end\n\
             end\n";
        check(&format!(
            "{}local values: Array<Drawable> = [Circle {{}}]\n",
            declarations
        ))
        .unwrap();

        let error = check(&format!(
            "{}local circles: Array<Circle> = [Circle {{}}]\n\
             local values: Array<Drawable> = circles\n",
            declarations
        ))
        .unwrap_err();
        assert!(error.to_string().contains("expected 'Drawable'"));
    }
}
