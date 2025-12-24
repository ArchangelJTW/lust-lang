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
                                let function = Function::new(
                                    &func_def.name,
                                    func_def.params.len() as u8,
                                    func_def.is_method,
                                );
                                self.function_table.insert(func_def.name.clone(), func_idx);
                                self.functions.push(function);
                                self.assign_signature_by_name(func_idx, &func_def.name);
                            }

                            ItemKind::Struct(_) | ItemKind::Enum(_) => {}
                            ItemKind::Trait(trait_def) => {
                                self.trait_names.insert(trait_def.name.clone());
                            }

                            ItemKind::Impl(impl_block) => {
                                let type_name = match &impl_block.target_type.kind {
                                    crate::ast::TypeKind::Named(name) => {
                                        self.resolve_type_name(name)
                                    }

                                    _ => {
                                        return Err(LustError::CompileError(
                                            "Impl block target must be a named type".to_string(),
                                        ))
                                    }
                                };
                                if let Some(trait_name) = &impl_block.trait_name {
                                    let resolved_trait = self.resolve_type_name(trait_name);
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
                                    let function = Function::new(
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

                            ItemKind::TypeAlias { .. }
                            | ItemKind::Use { .. }
                            | ItemKind::Const { .. }
                            | ItemKind::Static { .. } => {}
                            ItemKind::Module { .. } => {}
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
                    let function = Function::new(
                        &func_def.name,
                        func_def.params.len() as u8,
                        func_def.is_method,
                    );
                    self.function_table.insert(func_def.name.clone(), func_idx);
                    self.functions.push(function);
                    self.assign_signature_by_name(func_idx, &func_def.name);
                }

                ItemKind::Struct(_) | ItemKind::Enum(_) => {}
                ItemKind::Trait(trait_def) => {
                    self.trait_names.insert(trait_def.name.clone());
                }

                ItemKind::Impl(impl_block) => {
                    let type_name = match &impl_block.target_type.kind {
                        crate::ast::TypeKind::Named(name) => self.resolve_type_name(name),
                        _ => {
                            return Err(LustError::CompileError(
                                "Impl block target must be a named type".to_string(),
                            ));
                        }
                    };
                    if let Some(trait_name) = &impl_block.trait_name {
                        let resolved_trait = self.resolve_type_name(trait_name);
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
                            Function::new(&mangled_name, method.params.len() as u8, has_self);
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

                ItemKind::TypeAlias { .. }
                | ItemKind::Use { .. }
                | ItemKind::Const { .. }
                | ItemKind::Static { .. } => {}
            }
        }

        let script_func_idx = self.functions.len();
        let script_func = Function::new("__script", 0, false);
        self.function_table
            .insert("__script".to_string(), script_func_idx);
        self.functions.push(script_func);
        self.assign_signature_by_name(script_func_idx, "__script");
        let mut func_idx = 0;
        for item in items {
            match &item.kind {
                ItemKind::Module { name, items } => {
                    let prev = self.current_module.clone();
                    self.current_module = Some(name.clone());
                    for ni in items {
                        match &ni.kind {
                            ItemKind::Function(func_def) => {
                                self.compile_function(func_idx, func_def)?;
                                func_idx += 1;
                            }

                            ItemKind::Impl(impl_block) => {
                                for method in &impl_block.methods {
                                    self.compile_function(func_idx, method)?;
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
                    self.compile_function(func_idx, func_def)?;
                    func_idx += 1;
                }

                ItemKind::Impl(impl_block) => {
                    for method in &impl_block.methods {
                        self.compile_function(func_idx, method)?;
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
            };
            let prev = self.current_module.clone();
            if let Some(entry) = &self.entry_module {
                self.current_module = Some(entry.clone());
            }

            self.compile_function(script_func_idx, &fake_func)?;
            self.assign_signature_by_name(script_func_idx, &fake_func.name);
            self.current_module = prev;
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

    pub(super) fn is_module_init_function(
        func_def: &crate::ast::FunctionDef,
        module: &str,
    ) -> bool {
        func_def.name == format!("__init@{}", module)
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
        if let Some(module) = self.module_scope_name() {
            if let Some(locals) = self.module_locals.get(module) {
                return locals.contains(name);
            }
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

    pub(super) fn compile_function(
        &mut self,
        func_idx: usize,
        func_def: &crate::ast::FunctionDef,
    ) -> Result<()> {
        self.current_function = func_idx;
        self.next_register = 0;
        self.max_register = 0;
        self.scopes.clear();
        let prev_function_name = self.current_function_name.clone();
        self.current_function_name = Some(func_def.name.clone());
        let mut scope = Scope {
            locals: HashMap::new(),
            depth: 0,
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
        for stmt in &func_def.body {
            self.compile_stmt(stmt)?;
            self.reset_temp_registers();
        }

        let last_instr = self.current_chunk().instructions.last();
        if !matches!(last_instr, Some(Instruction::Return(_))) {
            self.emit(Instruction::Return(255), 0);
        }

        self.functions[func_idx].set_register_count(self.max_register + 1);
        self.scopes.pop();
        self.current_function_name = prev_function_name;
        Ok(())
    }
}
