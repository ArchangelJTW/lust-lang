use super::*;
impl Compiler {
    pub fn compile_module(&mut self, items: &[Item]) -> Result<Vec<Function>> {
        let mut script_stmts: Vec<Stmt> = Vec::new();
        for item in items {
            match &item.kind {
                ItemKind::Module { name, items } => {
                    let prev = self.current_module.clone();
                    self.current_module = Some(name.clone());
                    for ni in items {
                        match &ni.kind {
                            ItemKind::Script(stmts) => {
                                if let Some(module) = self.module_scope_name() {
                                    let locals_entry =
                                        self.module_locals.entry(module.to_string()).or_default();
                                    for name in Self::collect_top_level_locals(stmts) {
                                        locals_entry.insert(name);
                                    }
                                }
                                script_stmts.extend_from_slice(stmts);
                            }

                            ItemKind::Function(func_def) => {
                                let func_idx = self.functions.len();
                                let function = self.new_function(
                                    &func_def.name,
                                    func_def.params.len() as u8,
                                    func_def.is_method,
                                );
                                self.function_table.insert(func_def.name.clone(), func_idx);
                                self.functions.push(function);
                                self.assign_signature_by_name(func_idx, &func_def.name);
                            }

                            ItemKind::Struct(struct_def) => {
                                self.register_ast_struct_def(struct_def);
                            }
                            ItemKind::Enum(_) => {}
                            ItemKind::Trait(trait_def) => {
                                self.trait_names.insert(trait_def.name.clone());
                                self.trait_names
                                    .insert(self.resolve_type_name(&trait_def.name));
                            }

                            ItemKind::Impl(impl_block) => {
                                let type_name = match &impl_block.target_type.kind {
                                    crate::ast::TypeKind::Named(name)
                                    | crate::ast::TypeKind::GenericInstance { name, .. } => {
                                        self.resolve_type_name(name)
                                    }

                                    _ => {
                                        return Err(LustError::CompileError(
                                            "Impl block target must be a named type".to_string(),
                                        ));
                                    }
                                };
                                if let Some(trait_name) = &impl_block.trait_name {
                                    let resolved_trait =
                                        if trait_name == "ToString" || trait_name == "HashKey" {
                                            trait_name.clone()
                                        } else {
                                            self.resolve_type_name(trait_name)
                                        };
                                    self.trait_impls.push((type_name.clone(), resolved_trait));
                                }

                                for method in &impl_block.methods {
                                    let func_idx = self.functions.len();
                                    let has_self =
                                        method.params.iter().any(|p| p.is_self || p.name == "self");
                                    let mangled_name =
                                        if method.name.contains(':') || method.name.contains('.') {
                                            method.name.clone()
                                        } else if has_self {
                                            format!("{}:{}", type_name, method.name)
                                        } else {
                                            format!("{}.{}", type_name, method.name)
                                        };
                                    let function = self.new_function(
                                        &mangled_name,
                                        method.params.len() as u8,
                                        has_self,
                                    );
                                    self.function_table.insert(mangled_name.clone(), func_idx);
                                    self.functions.push(function);
                                    self.assign_signature_by_name(func_idx, &mangled_name);
                                }
                            }

                            ItemKind::Extern { items, .. } => {
                                for ext in items {
                                    match ext {
                                        ExternItem::Function { name, .. }
                                        | ExternItem::Const { name, .. } => {
                                            self.record_extern_value(name);
                                        }
                                        ExternItem::Struct(_) | ExternItem::Enum(_) => {}
                                    }
                                }
                            }

                            ItemKind::Use { .. } => {}
                            ItemKind::Module { .. } => {}
                            ItemKind::Service(_) => {}
                            ItemKind::Test(_) | ItemKind::Suite(_) => {}
                            ItemKind::Schema(_) | ItemKind::TypeAlias { .. } => {}
                        }
                    }

                    self.current_module = prev;
                }

                ItemKind::Script(stmts) => {
                    if let Some(module) = self.module_scope_name() {
                        let locals_entry =
                            self.module_locals.entry(module.to_string()).or_default();
                        for name in Self::collect_top_level_locals(stmts) {
                            locals_entry.insert(name);
                        }
                    }
                    script_stmts.extend_from_slice(stmts);
                }

                ItemKind::Function(func_def) => {
                    let func_idx = self.functions.len();
                    let function = self.new_function(
                        &func_def.name,
                        func_def.params.len() as u8,
                        func_def.is_method,
                    );
                    self.function_table.insert(func_def.name.clone(), func_idx);
                    self.functions.push(function);
                    self.assign_signature_by_name(func_idx, &func_def.name);
                }

                ItemKind::Struct(struct_def) => {
                    self.register_ast_struct_def(struct_def);
                }
                ItemKind::Enum(_) => {}
                ItemKind::Trait(trait_def) => {
                    self.trait_names.insert(trait_def.name.clone());
                    self.trait_names
                        .insert(self.resolve_type_name(&trait_def.name));
                }

                ItemKind::Impl(impl_block) => {
                    let type_name = match &impl_block.target_type.kind {
                        crate::ast::TypeKind::Named(name)
                        | crate::ast::TypeKind::GenericInstance { name, .. } => {
                            self.resolve_type_name(name)
                        }
                        _ => {
                            return Err(LustError::CompileError(
                                "Impl block target must be a named type".to_string(),
                            ));
                        }
                    };
                    if let Some(trait_name) = &impl_block.trait_name {
                        let resolved_trait = if trait_name == "ToString" || trait_name == "HashKey"
                        {
                            trait_name.clone()
                        } else {
                            self.resolve_type_name(trait_name)
                        };
                        self.trait_impls.push((type_name.clone(), resolved_trait));
                    }

                    for method in &impl_block.methods {
                        let func_idx = self.functions.len();
                        let has_self = method.params.iter().any(|p| p.is_self || p.name == "self");
                        let mangled_name = if method.name.contains(':') || method.name.contains('.')
                        {
                            method.name.clone()
                        } else if has_self {
                            format!("{}:{}", type_name, method.name)
                        } else {
                            format!("{}.{}", type_name, method.name)
                        };
                        let function =
                            self.new_function(&mangled_name, method.params.len() as u8, has_self);
                        self.function_table.insert(mangled_name.clone(), func_idx);
                        self.functions.push(function);
                        self.assign_signature_by_name(func_idx, &mangled_name);
                    }
                }

                ItemKind::Extern { items, .. } => {
                    for ext in items {
                        match ext {
                            ExternItem::Function { name, .. } | ExternItem::Const { name, .. } => {
                                self.record_extern_value(name);
                            }
                            ExternItem::Struct(_) | ExternItem::Enum(_) => {}
                        }
                    }
                }

                ItemKind::Use { .. } => {}
                ItemKind::Service(_) => {}
                ItemKind::Test(_) | ItemKind::Suite(_) => {}
                ItemKind::Schema(_) | ItemKind::TypeAlias { .. } => {}
            }
        }

        let script_func_idx = self.functions.len();
        let script_func = self.new_function("__script", 0, false);
        self.function_table
            .insert("__script".to_string(), script_func_idx);
        self.functions.push(script_func);
        self.assign_signature_by_name(script_func_idx, "__script");
        let mut pending_mir: Vec<(usize, crate::mir::ir::MirFunction)> = Vec::new();
        let mut func_idx = 0;
        for item in items {
            match &item.kind {
                ItemKind::Module { name, items } => {
                    let prev = self.current_module.clone();
                    self.current_module = Some(name.clone());
                    for ni in items {
                        match &ni.kind {
                            ItemKind::Function(func_def) => {
                                if let Some(mir_fn) =
                                    self.compile_function_with_owner(func_idx, func_def, None)?
                                {
                                    pending_mir.push((func_idx, mir_fn));
                                }
                                func_idx += 1;
                            }

                            ItemKind::Impl(impl_block) => {
                                let owner_type = match &impl_block.target_type.kind {
                                    crate::ast::TypeKind::Named(name)
                                    | crate::ast::TypeKind::GenericInstance { name, .. } => {
                                        Some(self.resolve_type_name(name))
                                    }
                                    _ => None,
                                };
                                for method in &impl_block.methods {
                                    if let Some(mir_fn) = self.compile_function_with_owner(
                                        func_idx,
                                        method,
                                        owner_type.as_deref(),
                                    )? {
                                        pending_mir.push((func_idx, mir_fn));
                                    }
                                    func_idx += 1;
                                }
                            }

                            ItemKind::Module { .. } => {}
                            _ => {}
                        }
                    }

                    self.current_module = prev;
                }

                ItemKind::Function(func_def) => {
                    if let Some(mir_fn) =
                        self.compile_function_with_owner(func_idx, func_def, None)?
                    {
                        pending_mir.push((func_idx, mir_fn));
                    }
                    func_idx += 1;
                }

                ItemKind::Impl(impl_block) => {
                    let owner_type = match &impl_block.target_type.kind {
                        crate::ast::TypeKind::Named(name)
                        | crate::ast::TypeKind::GenericInstance { name, .. } => {
                            Some(self.resolve_type_name(name))
                        }
                        _ => None,
                    };
                    for method in &impl_block.methods {
                        if let Some(mir_fn) = self.compile_function_with_owner(
                            func_idx,
                            method,
                            owner_type.as_deref(),
                        )? {
                            pending_mir.push((func_idx, mir_fn));
                        }
                        func_idx += 1;
                    }
                }

                _ => {}
            }
        }

        {
            let fake_func = crate::ast::FunctionDef {
                name: "__script".to_string(),
                type_params: vec![],
                trait_bounds: vec![],
                params: vec![],
                return_type: None,
                body: script_stmts,
                is_method: false,
                visibility: crate::ast::Visibility::Private,
                doc: None,
                effects: None,
            };
            let prev = self.current_module.clone();
            if let Some(entry) = &self.entry_module {
                self.current_module = Some(entry.clone());
            }

            if let Some(mir_fn) =
                self.compile_function_with_owner(script_func_idx, &fake_func, None)?
            {
                pending_mir.push((script_func_idx, mir_fn));
            }
            self.assign_signature_by_name(script_func_idx, &fake_func.name);
            self.current_module = prev;
        }

        if self.use_mir {
            // In mixed MIR + AST->Bytecode modules (`use_mir && !strict_mir`), keep only the
            // closed call-graph subset of MIR functions whose module-local callees also lowered
            // to MIR. Otherwise a MIR-VM caller would emit `LoadGlobal` / `GetIndex` against an
            // AST->Bytecode callee that returns `Value::Struct` and is not registered as a global.
            let mut valid_mir_names: HashSet<String> = pending_mir
                .iter()
                .map(|(_, f)| f.signature.name.clone())
                .collect();
            if !self.strict_mir {
                loop {
                    let mut dropped = Vec::new();
                    for (_, mir_fn) in &pending_mir {
                        if !valid_mir_names.contains(&mir_fn.signature.name) {
                            continue;
                        }
                        let calls_non_mir_local = mir_fn.blocks.iter().any(|b| {
                            matches!(
                                &b.terminator,
                                Some(crate::mir::ir::Terminator::Call { callee, .. })
                                    if self.function_table.contains_key(callee.as_str())
                                        && !valid_mir_names.contains(callee.as_str())
                            )
                        });
                        if calls_non_mir_local {
                            dropped.push(mir_fn.signature.name.clone());
                        }
                    }
                    if dropped.is_empty() {
                        break;
                    }
                    for name in dropped {
                        valid_mir_names.remove(&name);
                    }
                }
            }

            let mir_struct_layouts = self.build_struct_layouts_for_mir();
            for (f_idx, mir_fn) in pending_mir {
                if !valid_mir_names.contains(&mir_fn.signature.name) {
                    continue;
                }
                match crate::mir::MirToBytecodeCompiler::compile_with_field_slots_and_callee_indices(
                    &mir_fn,
                    &self.function_table,
                    &mir_struct_layouts,
                ) {
                    Ok(mut bytecode_fn) => {
                        bytecode_fn.is_method = self.functions[f_idx].is_method;
                        bytecode_fn.signature = self.functions[f_idx].signature.clone();
                        bytecode_fn.trusted_signature = if bytecode_fn.signature.is_some() {
                            bytecode_fn.compute_trusted_signature()
                        } else {
                            self.functions[f_idx].trusted_signature
                        };
                        self.functions[f_idx] = bytecode_fn;
                        self.mir_functions.push(mir_fn);
                    }
                    Err(err) => {
                        if self.strict_mir {
                            return Err(LustError::CompileError(format!(
                                "Strict MIR compilation failed for function '{}': {err}",
                                mir_fn.signature.name
                            )));
                        }
                    }
                }
            }
        } else {
            for (_, mir_fn) in pending_mir {
                self.mir_functions.push(mir_fn);
            }
        }

        for function in &mut self.functions {
            function.chunk.compute_bb_gas_weights();
            function.chunk.build_packed_code();
        }

        Ok(self.functions.clone())
    }

    pub(super) fn collect_top_level_locals(stmts: &[Stmt]) -> Vec<String> {
        let mut names = Vec::new();
        for stmt in stmts {
            if let StmtKind::Local { bindings, .. } = &stmt.kind {
                for binding in bindings {
                    names.push(binding.name.clone());
                }
            }
        }

        names
    }

    pub(super) fn module_scope_name(&self) -> Option<&str> {
        if let Some(module) = &self.current_module {
            Some(module.as_str())
        } else if let Some(entry) = &self.entry_module {
            Some(entry.as_str())
        } else {
            Some("__root")
        }
    }

    pub(super) fn looks_like_type_name(name: &str) -> bool {
        name.chars()
            .next()
            .map(|c| c.is_ascii_uppercase())
            .unwrap_or(false)
    }

    pub(super) fn is_initializer_context(&self) -> bool {
        if let Some(name) = self.current_function_name.as_deref() {
            name == "__script" || name.starts_with("__init@")
        } else {
            false
        }
    }

    pub(super) fn current_scope_is_top_level(&self) -> bool {
        matches!(self.scopes.last(), Some(scope) if scope.depth == 0)
    }

    pub(super) fn is_module_level_identifier(&self, name: &str) -> bool {
        if let Some(module) = self.module_scope_name()
            && let Some(locals) = self.module_locals.get(module)
        {
            return locals.contains(name);
        }

        false
    }

    pub(super) fn should_sync_module_local(&self, name: &str) -> bool {
        self.is_initializer_context()
            && self.current_scope_is_top_level()
            && self.is_module_level_identifier(name)
    }

    pub(super) fn emit_store_module_global(&mut self, name: &str, src_reg: Register) {
        if let Some(module) = self.module_scope_name() {
            let key = format!("{}::{}", module, name);
            let name_idx = self.add_string_constant(&key);
            self.emit(Instruction::StoreGlobal(name_idx, src_reg), 0);
        }
    }

    pub(super) fn emit_load_module_global(&mut self, name: &str, dest_reg: Register) -> Result<()> {
        if let Some(module) = self.module_scope_name() {
            let key = format!("{}::{}", module, name);
            let const_idx = self.add_string_constant(&key);
            self.emit(Instruction::LoadGlobal(dest_reg, const_idx), 0);
            Ok(())
        } else {
            Err(LustError::CompileError(format!(
                "Undefined variable: {}",
                name
            )))
        }
    }

    fn build_struct_layouts_for_mir(
        &self,
    ) -> HashMap<String, alloc::rc::Rc<crate::bytecode::StructLayout>> {
        let mut layouts = HashMap::with_capacity(self.struct_defs.len());
        for (name, def) in &self.struct_defs {
            let field_names: Vec<alloc::rc::Rc<String>> = def
                .fields
                .iter()
                .map(|field| alloc::rc::Rc::new(field.name.clone()))
                .collect();
            let field_storage: Vec<crate::bytecode::FieldStorage> = def
                .fields
                .iter()
                .map(|field| match field.ownership {
                    crate::ast::FieldOwnership::Weak => crate::bytecode::FieldStorage::Weak,
                    crate::ast::FieldOwnership::Strong => crate::bytecode::FieldStorage::Strong,
                })
                .collect();
            let field_types: Vec<crate::ast::Type> =
                def.fields.iter().map(|field| field.ty.clone()).collect();
            let weak_targets: Vec<Option<crate::ast::Type>> = def
                .fields
                .iter()
                .map(|field| field.weak_target.clone())
                .collect();
            let layout = alloc::rc::Rc::new(crate::bytecode::StructLayout::new(
                def.name.clone(),
                field_names,
                field_storage,
                field_types,
                weak_targets,
            ));
            layouts.insert(name.clone(), layout);
        }
        layouts
    }

    fn collect_struct_defs_for_mir(&self) -> Vec<crate::ast::StructDef> {
        let mut defs: Vec<crate::ast::StructDef> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        // 1. Prioritize simple names belonging to the current module scope or imported type aliases.
        if let Some(module) = self.module_scope_name() {
            if let Some(imports) = self.imports_by_module.get(module) {
                for (alias, fq) in &imports.type_aliases {
                    if let Some(def) = self.struct_defs.get(fq)
                        && seen.insert(alias.clone())
                    {
                        let mut aliased = def.clone();
                        aliased.name = alias.clone();
                        defs.push(aliased);
                    }
                }
            }
            for (key, def) in &self.struct_defs {
                let simple = key.rsplit('.').next().unwrap_or(key.as_str());
                if self.resolve_type_name(simple) == *key && seen.insert(simple.to_string()) {
                    let mut scoped = def.clone();
                    scoped.name = simple.to_string();
                    defs.push(scoped);
                }
            }
        }

        // 2. Register all known struct definitions under their exact keys (qualified & simple).
        for (key, def) in &self.struct_defs {
            if seen.insert(key.clone()) {
                let mut keyed = def.clone();
                keyed.name = key.clone();
                defs.push(keyed);
            }
        }

        // 3. Fallback simple names for any remaining definitions not yet registered.
        for (key, def) in &self.struct_defs {
            let simple = key.rsplit('.').next().unwrap_or(key.as_str());
            if seen.insert(simple.to_string()) {
                let mut named = def.clone();
                named.name = simple.to_string();
                defs.push(named);
            }
        }

        defs
    }

    fn resolve_mir_type(&self, ty: &mut crate::mir::ir::MirType) {
        use crate::mir::ir::MirType;
        match ty {
            MirType::Struct(name) => {
                let resolved = self.resolve_type_name(name);
                if self.struct_defs.contains_key(&resolved) {
                    *name = resolved;
                }
            }
            MirType::Array(inner) => self.resolve_mir_type(inner),
            MirType::Option(inner) => self.resolve_mir_type(inner),
            MirType::Tuple(elems) => {
                for elem in elems {
                    self.resolve_mir_type(elem);
                }
            }
            MirType::Map(k, v) => {
                self.resolve_mir_type(k);
                self.resolve_mir_type(v);
            }
            MirType::Function {
                params,
                return_type,
            } => {
                for p in params {
                    self.resolve_mir_type(p);
                }
                self.resolve_mir_type(return_type);
            }
            MirType::Effect {
                return_type,
                error_type,
            } => {
                self.resolve_mir_type(return_type);
                self.resolve_mir_type(error_type);
            }
            _ => {}
        }
    }

    fn resolve_mir_struct_types(&self, mir_fn: &mut crate::mir::ir::MirFunction) {
        for param_ty in &mut mir_fn.signature.param_types {
            self.resolve_mir_type(param_ty);
        }
        self.resolve_mir_type(&mut mir_fn.signature.return_type);
        for local in &mut mir_fn.locals {
            self.resolve_mir_type(&mut local.ty);
        }
        for block in &mut mir_fn.blocks {
            for stmt in &mut block.statements {
                if let crate::mir::ir::Statement::Assign(_, rval, _) = stmt {
                    match rval {
                        crate::mir::ir::Rvalue::Aggregate(ty, _)
                        | crate::mir::ir::Rvalue::Cast(_, ty) => {
                            self.resolve_mir_type(ty);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    fn resolve_mir_callees(&self, mir_fn: &mut crate::mir::ir::MirFunction) {
        let module_name = self.module_scope_name();
        let imports = module_name.and_then(|m| self.imports_by_module.get(m));

        for block in &mut mir_fn.blocks {
            let Some(crate::mir::ir::Terminator::Call { callee, .. }) = &mut block.terminator
            else {
                continue;
            };

            if let Some((ty_part, method_part)) = callee.split_once(':') {
                if self.function_table.contains_key(callee.as_str()) {
                    continue;
                }
                let resolved_ty = self.resolve_type_name(ty_part);
                let cand = format!("{resolved_ty}:{method_part}");
                if self.function_table.contains_key(&cand) {
                    *callee = cand;
                    continue;
                }
            }

            if !callee.contains('.') {
                if let Some(fq) = imports
                    .and_then(|imp| imp.function_aliases.get(callee.as_str()))
                    && self.function_table.contains_key(fq)
                {
                    *callee = fq.clone();
                    continue;
                }
                if let Some(m) = module_name {
                    let cand = format!("{m}.{callee}");
                    if self.function_table.contains_key(&cand) {
                        *callee = cand;
                    }
                }
            } else if !self.function_table.contains_key(callee.as_str())
                && let Some((mod_alias, fn_name)) = callee.split_once('.')
            {
                if let Some(real_mod) =
                    imports.and_then(|imp| imp.module_aliases.get(mod_alias))
                {
                    let cand = format!("{real_mod}.{fn_name}");
                    if self.function_table.contains_key(&cand) {
                        *callee = cand;
                        continue;
                    }
                }
                let resolved_ty = self.resolve_type_name(mod_alias);
                let cand = format!("{resolved_ty}.{fn_name}");
                if self.function_table.contains_key(&cand) {
                    *callee = cand;
                }
            }
        }
    }

    pub(super) fn compile_function_with_owner(
        &mut self,
        func_idx: usize,
        func_def: &crate::ast::FunctionDef,
        owner_struct: Option<&str>,
    ) -> Result<Option<crate::mir::ir::MirFunction>> {
        let mut lowered_mir: Option<crate::mir::ir::MirFunction> = None;
        if self.use_mir || self.collect_mir {
            let scope_structs = self.collect_struct_defs_for_mir();
            let struct_refs: Vec<&crate::ast::StructDef> = scope_structs.iter().collect();
            match crate::mir::AstLowering::lower_function_with_structs_and_owner(
                func_def,
                owner_struct,
                &struct_refs,
            ) {
                Ok((mut mir_fn, _ops)) => {
                    mir_fn.signature.name = self.functions[func_idx].name.clone();
                    self.resolve_mir_struct_types(&mut mir_fn);
                    self.resolve_mir_callees(&mut mir_fn);
                    match crate::mir::MirVerifier::verify_fn(&mir_fn) {
                        Ok(_) => {
                            crate::mir::MirOptimizer::optimize_function(&mut mir_fn);
                            lowered_mir = Some(mir_fn);
                        }
                        Err(errors) => {
                            if self.strict_mir {
                                return Err(LustError::CompileError(format!(
                                    "Strict MIR verification failed for function '{}': {:?}",
                                    func_def.name, errors
                                )));
                            }
                        }
                    }
                }
                Err(err) => {
                    if self.strict_mir {
                        return Err(LustError::CompileError(format!(
                            "Strict MIR lowering failed for function '{}': {err}",
                            func_def.name
                        )));
                    }
                }
            }
        }

        self.current_function = func_idx;
        self.next_register = 0;
        self.max_register = 0;
        self.scopes.clear();
        let prev_function_name = self.current_function_name.clone();
        self.current_function_name = Some(func_def.name.clone());
        let mut scope = Scope {
            locals: HashMap::new(),
            upvalues: HashMap::new(),
            effect_thunks: HashMap::new(),
            depth: 0,
            cleanups: Vec::new(),
            is_thunk_boundary: false,
        };
        if func_def.is_method {
            scope.locals.insert("self".to_string(), (0, false));
            self.next_register = 1;
        }

        for param in &func_def.params {
            let reg = self.allocate_register();
            scope.locals.insert(param.name.clone(), (reg, false));
            self.register_type(reg, param.ty.kind.clone());
        }

        self.scopes.push(scope);
        let has_value_return = func_def
            .return_type
            .as_ref()
            .map_or(false, |t| !matches!(t.kind, TypeKind::Unit))
            || func_def
                .effects
                .as_ref()
                .and_then(|e| e.return_type.as_ref())
                .map_or(false, |t| !matches!(t.kind, TypeKind::Unit));

        let body_len = func_def.body.len();
        for (i, stmt) in func_def.body.iter().enumerate() {
            let is_last = i + 1 == body_len;
            if is_last && has_value_return && matches!(&stmt.kind, StmtKind::Expr(_)) {
                if let StmtKind::Expr(expr) = &stmt.kind {
                    let ret_stmt = Stmt {
                        kind: StmtKind::Return(vec![expr.clone()]),
                        span: stmt.span.clone(),
                    };
                    self.compile_stmt(&ret_stmt)?;
                    self.reset_temp_registers();
                    continue;
                }
            }
            self.compile_stmt(stmt)?;
            self.reset_temp_registers();
        }

        let last_instr = self.current_chunk().instructions.last();
        if !matches!(last_instr, Some(Instruction::Return(_))) {
            self.emit_all_defers()?;
            self.emit(Instruction::Return(255), 0);
        }

        self.functions[func_idx].set_register_count(self.max_register + 1);
        self.scopes.pop();
        self.current_function_name = prev_function_name;
        Ok(lowered_mir)
    }
}
