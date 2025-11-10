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
    current_module: Option<String>,
    imports_by_module: HashMap<String, ModuleImports>,
    expr_types_by_module: HashMap<String, HashMap<Span, Type>>,
    variable_types_by_module: HashMap<String, HashMap<Span, Type>>,
    short_circuit_info: HashMap<String, HashMap<Span, ShortCircuitInfo>>,
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
            current_module: None,
            imports_by_module: HashMap::new(),
            expr_types_by_module: HashMap::new(),
            variable_types_by_module: HashMap::new(),
            short_circuit_info: HashMap::new(),
        }
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
                    let target_name = if let TypeKind::Named(inner) = &target.kind {
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

                    let description = cycle.join(" -> ");
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

    pub fn function_signatures(&self) -> HashMap<String, type_env::FunctionSignature> {
        self.env.function_signatures()
    }

    pub fn struct_definitions(&self) -> HashMap<String, StructDef> {
        self.env.struct_definitions()
    }

    pub fn enum_definitions(&self) -> HashMap<String, EnumDef> {
        self.env.enum_definitions()
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
        for field in &mut def.fields {
            field.ty = self.canonicalize_type(&field.ty);
            if let Some(target) = &field.weak_target {
                field.weak_target = Some(self.canonicalize_type(target));
            }
        }
        self.env.register_struct(&def)
    }

    pub fn register_external_enum(&mut self, mut def: EnumDef) -> Result<()> {
        def.name = self.resolve_type_key(&def.name);
        for variant in &mut def.variants {
            if let Some(fields) = &mut variant.fields {
                for field in fields {
                    *field = self.canonicalize_type(field);
                }
            }
        }
        self.env.register_enum(&def)
    }

    pub fn register_external_trait(&mut self, mut def: TraitDef) -> Result<()> {
        def.name = self.resolve_type_key(&def.name);
        for method in &mut def.methods {
            for param in &mut method.params {
                param.ty = self.canonicalize_type(&param.ty);
            }
            if let Some(ret) = method.return_type.clone() {
                method.return_type = Some(self.canonicalize_type(&ret));
            }
        }
        self.env.register_trait(&def)
    }

    pub fn register_external_function(
        &mut self,
        (name, mut signature): (String, FunctionSignature),
    ) -> Result<()> {
        signature.params = signature
            .params
            .into_iter()
            .map(|ty| self.canonicalize_type(&ty))
            .collect();
        signature.return_type = self.canonicalize_type(&signature.return_type);
        let canonical = self.resolve_type_key(&name);
        self.env.register_or_update_function(canonical, signature)
    }

    pub fn register_external_impl(&mut self, mut impl_block: ImplBlock) -> Result<()> {
        impl_block.target_type = self.canonicalize_type(&impl_block.target_type);
        if let Some(trait_name) = &impl_block.trait_name {
            impl_block.trait_name = Some(self.resolve_type_key(trait_name));
        }
        for method in &mut impl_block.methods {
            for param in &mut method.params {
                param.ty = self.canonicalize_type(&param.ty);
            }
            if let Some(ret) = method.return_type.clone() {
                method.return_type = Some(self.canonicalize_type(&ret));
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

        self.env.register_impl(&impl_block);
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
            #[cfg(debug_assertions)]
            eprintln!(
                "register_external_impl canonical method {} (has_self={})",
                canonical_name, has_self
            );
            let signature = FunctionSignature {
                params,
                return_type,
                is_method: has_self,
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

                self.env.register_struct(&s2)?;
            }

            ItemKind::Enum(e) => {
                let mut e2 = e.clone();
                if let Some(module) = &self.current_module {
                    if !e2.name.contains('.') {
                        e2.name = format!("{}.{}", module, e2.name);
                    }
                }

                self.env.register_enum(&e2)?;
            }

            ItemKind::Trait(t) => {
                let mut t2 = t.clone();
                if let Some(module) = &self.current_module {
                    if !t2.name.contains('.') {
                        t2.name = format!("{}.{}", module, t2.name);
                    }
                }

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
                self.env
                    .register_type_alias(qname, type_params.clone(), target.clone())?;
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

    pub fn canonicalize_type(&self, ty: &Type) -> Type {
        use crate::ast::TypeKind as TK;
        match &ty.kind {
            TK::Named(name) => Type::new(TK::Named(self.resolve_type_key(name)), ty.span),
            TK::Array(inner) => {
                Type::new(TK::Array(Box::new(self.canonicalize_type(inner))), ty.span)
            }

            TK::Tuple(elements) => Type::new(
                TK::Tuple(elements.iter().map(|t| self.canonicalize_type(t)).collect()),
                ty.span,
            ),
            TK::Option(inner) => {
                Type::new(TK::Option(Box::new(self.canonicalize_type(inner))), ty.span)
            }

            TK::Result(ok, err) => Type::new(
                TK::Result(
                    Box::new(self.canonicalize_type(ok)),
                    Box::new(self.canonicalize_type(err)),
                ),
                ty.span,
            ),
            TK::Map(k, v) => Type::new(
                TK::Map(
                    Box::new(self.canonicalize_type(k)),
                    Box::new(self.canonicalize_type(v)),
                ),
                ty.span,
            ),
            TK::Ref(inner) => Type::new(TK::Ref(Box::new(self.canonicalize_type(inner))), ty.span),
            TK::MutRef(inner) => {
                Type::new(TK::MutRef(Box::new(self.canonicalize_type(inner))), ty.span)
            }

            TK::Pointer { mutable, pointee } => Type::new(
                TK::Pointer {
                    mutable: *mutable,
                    pointee: Box::new(self.canonicalize_type(pointee)),
                },
                ty.span,
            ),
            _ => ty.clone(),
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

    fn unify_at(&self, expected: &Type, actual: &Type, span: Option<Span>) -> Result<()> {
        if matches!(expected.kind, TypeKind::Unknown) || matches!(actual.kind, TypeKind::Unknown) {
            return Ok(());
        }

        if matches!(expected.kind, TypeKind::Infer) || matches!(actual.kind, TypeKind::Infer) {
            return Ok(());
        }

        match (&expected.kind, &actual.kind) {
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
                    return self.unify(exp_el, act_el);
                }
            }

            (TypeKind::Map(exp_key, exp_value), TypeKind::Map(act_key, act_value)) => {
                self.unify(exp_key, act_key)?;
                return self.unify(exp_value, act_value);
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

    fn record_short_circuit_info(&mut self, span: Span, info: &ShortCircuitInfo) {
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
