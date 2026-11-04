use super::*;
impl Compiler {
    pub(super) fn compile_stmt(&mut self, stmt: &Stmt) -> Result<()> {
        self.current_line = stmt.span.start_line;
        match &stmt.kind {
            StmtKind::Local {
                bindings,
                initializer,
            } => {
                if let Some(values) = initializer {
                    if bindings.len() == 1 && values.len() == 1 {
                        let mut init_expr = &values[0];
                        while let ExprKind::Paren(inner) = &init_expr.kind {
                            init_expr = inner;
                        }
                        if let ExprKind::EffectBlock(stmts) = &init_expr.kind {
                            let thunk = self.build_effect_thunk(&bindings[0].name, stmts)?;
                            if let Some(scope) = self.scopes.last_mut() {
                                scope
                                    .effect_thunks
                                    .insert(bindings[0].name.clone(), thunk);
                            }
                            return Ok(());
                        }
                        if let ExprKind::Identifier(name) = &init_expr.kind {
                            if self.lookup_effect_thunk(name).is_some() {
                                return Err(LustError::CompileError(format!(
                                    "Effect thunk '{name}' cannot be aliased or copied"
                                )));
                            }
                        }
                    }
                }

                let mut binding_regs = Vec::new();
                for (idx, binding) in bindings.iter().enumerate() {
                    let reg = self.next_local_slot();
                    if let Some(type_ann) = &binding.type_annotation {
                        self.register_type(reg, type_ann.kind.clone());
                    } else if let Some(values) = initializer {
                        if let Some(init_expr) = values.get(idx) {
                            match &init_expr.kind {
                                ExprKind::Literal(Literal::Integer(_)) => {
                                    self.register_type(reg, TypeKind::Int);
                                }
                                ExprKind::Identifier(src_name) => {
                                    if let Ok(src_reg) = self.resolve_local(src_name) {
                                        if let Some(tk) = self.functions[self.current_function]
                                            .register_types
                                            .get(&src_reg)
                                            .cloned()
                                        {
                                            self.register_type(reg, tk);
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    binding_regs.push((binding.name.clone(), reg));
                }

                let sync_module_local =
                    self.is_initializer_context() && self.current_scope_is_top_level();
                let module_scope = if sync_module_local {
                    self.module_scope_name().map(|s| s.to_string())
                } else {
                    None
                };

                if let Some(values) = initializer {
                    if bindings.len() == 1 && values.len() == 1 {
                        let expression_start = self.current_chunk().instructions.len();
                        let value_reg = self.compile_expr(&values[0])?;
                        let (_name, target_reg) = &binding_regs[0];
                        self.move_result(*target_reg, value_reg, expression_start);
                        self.free_register(value_reg);
                    } else {
                        let tuple_reg = self.compile_exprs_to_tuple(values)?;
                        for (index, (_name, target_reg)) in binding_regs.iter().enumerate() {
                            let value_reg = self.allocate_register();
                            self.emit(Instruction::TupleGet(value_reg, tuple_reg, index as u8), 0);
                            self.emit(Instruction::Move(*target_reg, value_reg), 0);
                            self.free_register(value_reg);
                        }

                        self.free_register(tuple_reg);
                    }
                } else {
                    for (_name, target_reg) in &binding_regs {
                        self.emit(Instruction::LoadNil(*target_reg), 0);
                    }
                }

                if let Some(scope) = self.scopes.last_mut() {
                    for (name, reg) in &binding_regs {
                        scope.locals.insert(name.clone(), (*reg, false));
                    }
                }

                if let Some(module) = module_scope {
                    {
                        let locals_entry = self.module_locals.entry(module.clone()).or_default();
                        for (name, _) in &binding_regs {
                            locals_entry.insert(name.clone());
                        }
                    }
                    for (name, reg) in &binding_regs {
                        self.emit_store_module_global(name, *reg);
                    }
                }
            }

            StmtKind::Assign { targets, values } => {
                for target in targets {
                    if let ExprKind::Identifier(name) = &target.kind {
                        if self.lookup_effect_thunk(name).is_some() {
                            return Err(LustError::CompileError(format!(
                                "Effect thunk '{name}' cannot be reassigned or mutated"
                            )));
                        }
                    }
                }
                for val in values {
                    if let ExprKind::Identifier(name) = &val.kind {
                        if self.lookup_effect_thunk(name).is_some() {
                            return Err(LustError::CompileError(format!(
                                "Effect thunk '{name}' cannot be aliased or copied"
                            )));
                        }
                    }
                }

                if targets.len() == 1 && values.len() == 1 {
                    if let (
                        ExprKind::Identifier(target_name),
                        ExprKind::Binary {
                            op: BinaryOp::Concat,
                            left,
                            right,
                        },
                    ) = (&targets[0].kind, &values[0].kind)
                        && let ExprKind::Identifier(left_name) = &left.kind
                        && left_name == target_name
                        && let Ok(target_reg) = self.resolve_local(target_name)
                    {
                        let rhs_reg = self.compile_expr(right)?;
                        self.emit(Instruction::Concat(target_reg, target_reg, rhs_reg), 0);
                        if self.should_sync_module_local(target_name) {
                            self.emit_store_module_global(target_name, target_reg);
                        }

                        self.free_register(rhs_reg);
                        return Ok(());
                    }

                    let expression_start = self.current_chunk().instructions.len();
                    let value_reg = self.compile_expr(&values[0])?;
                    self.assign_value_to_target(&targets[0], value_reg, expression_start)?;
                    self.free_register(value_reg);
                } else {
                    let tuple_reg = self.compile_exprs_to_tuple(values)?;
                    for (index, target) in targets.iter().enumerate() {
                        let value_reg = self.allocate_register();
                        let expression_start =
                            self.emit(Instruction::TupleGet(value_reg, tuple_reg, index as u8), 0);
                        self.assign_value_to_target(target, value_reg, expression_start)?;
                        self.free_register(value_reg);
                    }

                    self.free_register(tuple_reg);
                }
            }

            StmtKind::CompoundAssign { target, op, value } => {
                if let ExprKind::Identifier(name) = &target.kind {
                    if self.lookup_effect_thunk(name).is_some() {
                        return Err(LustError::CompileError(format!(
                            "Effect thunk '{name}' cannot be reassigned or mutated"
                        )));
                    }
                }
                match &target.kind {
                    ExprKind::Identifier(name) => {
                        let (target_reg, target_is_local, target_name) =
                            if let Ok(local_reg) = self.resolve_local(name) {
                                (local_reg, true, name.clone())
                            } else if self.is_module_level_identifier(name) {
                                let reg = self.allocate_register();
                                self.emit_load_module_global(name, reg)?;
                                (reg, false, name.clone())
                            } else {
                                return Err(LustError::CompileError(
                                    "Invalid compound assignment target".to_string(),
                                ));
                            };
                        let value_reg = self.compile_expr(value)?;
                        let result_reg = self.allocate_register();
                        let expression_start = self.current_chunk().instructions.len();
                        self.compile_binary_op(
                            *op,
                            result_reg,
                            target_reg,
                            value_reg,
                            (target.span, value.span),
                        )?;
                        if target_is_local {
                            if result_reg != target_reg {
                                self.move_result(target_reg, result_reg, expression_start);
                                self.free_register(result_reg);
                            }

                            if self.should_sync_module_local(&target_name) {
                                self.emit_store_module_global(&target_name, target_reg);
                            }

                            self.free_register(value_reg);
                        } else {
                            self.emit_store_module_global(&target_name, result_reg);
                            self.free_register(result_reg);
                            self.free_register(value_reg);
                            self.free_register(target_reg);
                        }
                    }

                    ExprKind::FieldAccess { object, field } => {
                        let obj_reg = self.compile_expr(object)?;
                        let field_idx = self.add_string_constant(field);
                        let cur_reg = self.allocate_register();
                        self.emit(Instruction::GetField(cur_reg, obj_reg, field_idx), 0);
                        let val_reg = self.compile_expr(value)?;
                        let res_reg = self.allocate_register();
                        self.compile_binary_op(
                            *op,
                            res_reg,
                            cur_reg,
                            val_reg,
                            (target.span, value.span),
                        )?;
                        self.emit(Instruction::SetField(obj_reg, field_idx, res_reg), 0);
                        self.free_register(res_reg);
                        self.free_register(val_reg);
                        self.free_register(cur_reg);
                        self.free_register(obj_reg);
                    }

                    ExprKind::Index { object, index } => {
                        let obj_reg = self.compile_expr(object)?;
                        let idx_reg = self.compile_expr(index)?;
                        let cur_reg = self.allocate_register();
                        self.emit(Instruction::GetIndex(cur_reg, obj_reg, idx_reg), 0);
                        let val_reg = self.compile_expr(value)?;
                        let res_reg = self.allocate_register();
                        self.compile_binary_op(
                            *op,
                            res_reg,
                            cur_reg,
                            val_reg,
                            (target.span, value.span),
                        )?;
                        self.emit(Instruction::SetIndex(obj_reg, idx_reg, res_reg), 0);
                        self.free_register(res_reg);
                        self.free_register(val_reg);
                        self.free_register(cur_reg);
                        self.free_register(idx_reg);
                        self.free_register(obj_reg);
                    }

                    _ => {
                        return Err(LustError::CompileError(
                            "Invalid compound assignment target".to_string(),
                        ));
                    }
                }
            }

            StmtKind::Expr(expr) => {
                let reg = self.compile_expr(expr)?;
                self.free_register(reg);
            }

            StmtKind::If {
                condition,
                then_block,
                elseif_branches,
                else_block,
            } => {
                self.compile_if_stmt(condition, then_block, elseif_branches, else_block)?;
            }

            StmtKind::While { condition, body } => {
                self.compile_while_loop(condition, body)?;
            }

            StmtKind::Return(values) => {
                if self.thunk_return_target.is_some() {
                    if values.len() != 1 {
                        return Err(LustError::CompileError(
                            "Effect thunk return must return exactly one value".into(),
                        ));
                    }
                    let target_reg = self.thunk_return_target.as_ref().unwrap().0;
                    let expr_reg = self.compile_expr(&values[0])?;
                    self.emit(Instruction::Move(target_reg, expr_reg), 0);
                    self.free_register(expr_reg);
                    let jump_pos = self.emit(Instruction::Jump(0), 0);
                    if let Some((_, jumps)) = &mut self.thunk_return_target {
                        jumps.push(jump_pos);
                    }
                    return Ok(());
                }

                for val in values {
                    if let ExprKind::Identifier(name) = &val.kind {
                        if self.lookup_effect_thunk(name).is_some() {
                            return Err(LustError::CompileError(format!(
                                "Effect thunk '{name}' cannot escape or be used as a value"
                            )));
                        }
                    }
                }

                let ret_reg = if values.is_empty() {
                    255
                } else if values.len() == 1 {
                    let expr_reg = self.compile_expr(&values[0])?;
                    let dedicated_reg = self.next_local_slot();
                    if let Some(scope) = self.scopes.first_mut() {
                        scope.locals.insert(format!("(return_{})", dedicated_reg), (dedicated_reg, false));
                    }
                    self.emit(Instruction::Move(dedicated_reg, expr_reg), 0);
                    self.free_register(expr_reg);
                    dedicated_reg
                } else {
                    let tuple_reg = self.compile_exprs_to_tuple(values)?;
                    let dedicated_reg = self.next_local_slot();
                    if let Some(scope) = self.scopes.first_mut() {
                        scope.locals.insert(format!("(return_{})", dedicated_reg), (dedicated_reg, false));
                    }
                    self.emit(Instruction::Move(dedicated_reg, tuple_reg), 0);
                    self.free_register(tuple_reg);
                    dedicated_reg
                };

                self.emit_all_defers()?;
                self.emit(Instruction::Return(ret_reg), 0);
            }

            StmtKind::Block(stmts) => {
                self.begin_scope();
                for stmt in stmts {
                    self.compile_stmt(stmt)?;
                    self.reset_temp_registers();
                }

                self.end_scope()?;
            }

            StmtKind::Break => {
                if self.loop_contexts.is_empty() {
                    return Err(LustError::CompileError(
                        "'break' outside of loop".to_string(),
                    ));
                }

                let jump_pos = self.emit(Instruction::Jump(0), 0);
                let loop_ctx = self.loop_contexts.last_mut().unwrap();
                loop_ctx.break_jumps.push(jump_pos);
            }

            StmtKind::Continue => {
                let continue_target = self
                    .loop_contexts
                    .last()
                    .ok_or_else(|| {
                        LustError::CompileError("'continue' outside of loop".to_string())
                    })?
                    .continue_target;
                if let Some(target) = continue_target {
                    self.emit_jump_back_to(target);
                } else {
                    let jump_pos = self.emit(Instruction::Jump(0), 0);
                    self.loop_contexts
                        .last_mut()
                        .unwrap()
                        .continue_jumps
                        .push(jump_pos);
                }
            }

            StmtKind::ForNumeric {
                variable,
                start,
                end,
                step,
                body,
            } => {
                self.compile_for_numeric_loop(variable, start, end, step.as_ref(), body)?;
            }

            StmtKind::ForIn {
                variables,
                iterator,
                body,
            } => {
                self.compile_for_in_loop(variables, iterator, body)?;
            }

            StmtKind::Defer(s) => {
                if let Some(scope) = self.scopes.last_mut() {
                    scope.cleanups.push(ScopeCleanup {
                        kind: CleanupKind::Defer,
                        stmt: s.clone(),
                    });
                }
            }

            StmtKind::ErrDefer(s) => {
                if let Some(scope) = self.scopes.last_mut() {
                    scope.cleanups.push(ScopeCleanup {
                        kind: CleanupKind::ErrDefer,
                        stmt: s.clone(),
                    });
                }
            }

            StmtKind::Scoped(stmts) => {
                self.begin_scope();
                for s in stmts {
                    self.compile_stmt(s)?;
                }
                self.end_scope()?;
            }
        }

        // Reset temporary registers after each statement to prevent accumulation
        self.reset_temp_registers();

        Ok(())
    }

    pub(super) fn compile_if_stmt(
        &mut self,
        condition: &Expr,
        then_block: &[Stmt],
        elseif_branches: &[(Expr, Vec<Stmt>)],
        else_block: &Option<Vec<Stmt>>,
    ) -> Result<()> {
        self.begin_scope();
        let mut tested_scrutinee = None;
        let cond_reg = if let Some((obj_expr, idx_expr, binding_name)) =
            self.try_unwrap_checked_array_pattern(condition)
        {
            let obj_reg = self.compile_expr(obj_expr)?;
            let idx_reg = self.compile_expr(idx_expr)?;
            let val_reg = if let Some(name) = &binding_name {
                self.bind_local_variable(name)
            } else {
                self.allocate_register()
            };
            let cond_reg = self.allocate_register();
            self.emit(
                Instruction::ArrayIndexOk(val_reg, cond_reg, obj_reg, idx_reg),
                0,
            );
            self.free_register(obj_reg);
            self.free_register(idx_reg);
            if binding_name.is_none() {
                self.free_register(val_reg);
            }
            cond_reg
        } else if let Some((expr, pattern)) = Self::direct_is_pattern(condition) {
            if self.extract_all_pattern_bindings(condition).is_empty() {
                self.compile_expr(condition)?
            } else {
                let scrutinee_reg = self.compile_expr(expr)?;
                let result_reg = self.compile_is_pattern(scrutinee_reg, pattern)?;
                tested_scrutinee = Some((scrutinee_reg, pattern));
                result_reg
            }
        } else {
            self.compile_expr(condition)?
        };
        let jump_to_else = self.emit(Instruction::JumpIfNot(cond_reg, 0), 0);
        self.free_register(cond_reg);
        if let Some((enum_reg, pattern)) = tested_scrutinee {
            self.bind_pattern_variables(enum_reg, pattern)?;
            self.free_register(enum_reg);
        }
        self.reset_temp_registers();

        for stmt in then_block {
            self.compile_stmt(stmt)?;
        }

        self.end_scope()?;
        let mut jumps_to_end = vec![self.emit(Instruction::Jump(0), 0)];
        let next_branch_start = self.current_chunk().instructions.len();
        self.current_chunk_mut()
            .patch_jump(jump_to_else, next_branch_start);
        for (elseif_cond, elseif_body) in elseif_branches {
            self.begin_scope();
            let mut tested_scrutinee = None;
            let elseif_cond_reg = if let Some((obj_expr, idx_expr, binding_name)) =
                self.try_unwrap_checked_array_pattern(elseif_cond)
            {
                let obj_reg = self.compile_expr(obj_expr)?;
                let idx_reg = self.compile_expr(idx_expr)?;
                let val_reg = if let Some(name) = &binding_name {
                    self.bind_local_variable(name)
                } else {
                    self.allocate_register()
                };
                let cond_reg = self.allocate_register();
                self.emit(
                    Instruction::ArrayIndexOk(val_reg, cond_reg, obj_reg, idx_reg),
                    0,
                );
                self.free_register(obj_reg);
                self.free_register(idx_reg);
                if binding_name.is_none() {
                    self.free_register(val_reg);
                }
                cond_reg
            } else if let Some((expr, pattern)) = Self::direct_is_pattern(elseif_cond) {
                if self.extract_all_pattern_bindings(elseif_cond).is_empty() {
                    self.compile_expr(elseif_cond)?
                } else {
                    let scrutinee_reg = self.compile_expr(expr)?;
                    let result_reg = self.compile_is_pattern(scrutinee_reg, pattern)?;
                    tested_scrutinee = Some((scrutinee_reg, pattern));
                    result_reg
                }
            } else {
                self.compile_expr(elseif_cond)?
            };
            let jump_to_next = self.emit(Instruction::JumpIfNot(elseif_cond_reg, 0), 0);
            self.free_register(elseif_cond_reg);
            if let Some((enum_reg, pattern)) = tested_scrutinee {
                self.bind_pattern_variables(enum_reg, pattern)?;
                self.free_register(enum_reg);
            }
            self.reset_temp_registers();

            for stmt in elseif_body {
                self.compile_stmt(stmt)?;
            }

            self.end_scope()?;
            jumps_to_end.push(self.emit(Instruction::Jump(0), 0));
            let next_start = self.current_chunk().instructions.len();
            self.current_chunk_mut()
                .patch_jump(jump_to_next, next_start);
        }

        if let Some(else_stmts) = else_block {
            self.begin_scope();
            self.reset_temp_registers();
            for stmt in else_stmts {
                self.compile_stmt(stmt)?;
            }

            self.end_scope()?;
        }

        let end_pos = self.current_chunk().instructions.len();
        for jump in jumps_to_end {
            self.current_chunk_mut().patch_jump(jump, end_pos);
        }

        Ok(())
    }

    pub(super) fn compile_while_loop(&mut self, condition: &Expr, body: &[Stmt]) -> Result<()> {
        let loop_start = self.current_chunk().instructions.len();
        let cond_reg = self.compile_expr(condition)?;
        let jump_to_end = self.emit(Instruction::JumpIfNot(cond_reg, 0), 0);
        self.free_register(cond_reg);
        self.loop_contexts.push(LoopContext {
            continue_target: Some(loop_start),
            continue_jumps: Vec::new(),
            break_jumps: Vec::new(),
        });
        self.begin_scope();
        for stmt in body {
            self.compile_stmt(stmt)?;
        }

        self.end_scope()?;
        let loop_ctx = self.loop_contexts.pop().unwrap();
        self.emit_jump_back_to(loop_start);
        let end_pos = self.current_chunk().instructions.len();
        self.current_chunk_mut().patch_jump(jump_to_end, end_pos);
        for break_jump in loop_ctx.break_jumps {
            self.current_chunk_mut().patch_jump(break_jump, end_pos);
        }

        Ok(())
    }

    /// The value of an integer-literal expression, if it is one.
    ///
    /// Used to settle a numeric `for` loop's comparison direction at compile
    /// time.  Only the forms a step is actually written in are folded: a literal,
    /// a negated literal, and either wrapped in parentheses.
    fn const_int_expr(expr: &Expr) -> Option<LustInt> {
        match &expr.kind {
            ExprKind::Literal(Literal::Integer(value)) => Some(*value),
            ExprKind::Paren(inner) => Self::const_int_expr(inner),
            ExprKind::Unary {
                op: UnaryOp::Neg,
                operand,
            } => Self::const_int_expr(operand).map(|value| -value),
            _ => None,
        }
    }

    pub(super) fn compile_for_numeric_loop(
        &mut self,
        variable: &str,
        start: &Expr,
        end: &Expr,
        step: Option<&Expr>,
        body: &[Stmt],
    ) -> Result<()> {
        self.begin_scope();
        let int_loop = self.numeric_type(start.span) == Some(NumericType::Int)
            && self.numeric_type(end.span) == Some(NumericType::Int)
            && step.is_none_or(|step| self.numeric_type(step.span) == Some(NumericType::Int));
        let specialize = |instruction: Instruction| {
            if int_loop {
                instruction.specialize_numeric(NumericType::Int)
            } else {
                instruction
            }
        };
        let var_reg = self.next_local_slot();
        let start_reg = self.compile_expr(start)?;
        if start_reg != var_reg {
            self.emit(Instruction::Move(var_reg, start_reg), 0);
        }

        self.free_register(start_reg);
        self.register_type(var_reg, crate::ast::TypeKind::Int);
        if let Some(scope) = self.scopes.last_mut() {
            scope.locals.insert(variable.to_string(), (var_reg, false));
        }

        let end_reg = self.compile_expr(end)?;
        let step_reg = if let Some(step_expr) = step {
            self.compile_expr(step_expr)?
        } else {
            let reg = self.allocate_register();
            let const_idx = self.add_int_const(1);
            self.emit(Instruction::LoadConst(reg, const_idx), 0);
            reg
        };
        if let Some(scope) = self.scopes.last_mut() {
            scope
                .locals
                .insert("(for limit)".to_string(), (end_reg, false));
            scope
                .locals
                .insert("(for step)".to_string(), (step_reg, false));
        }

        // The loop test depends on the sign of the step: an ascending loop runs
        // while `var <= end`, a descending one while `var >= end`.  This used to
        // emit `Le` unconditionally, so every descending loop
        // (`for i = 100, 1, -1`) tested `100 <= 1`, failed on entry and ran zero
        // iterations.
        //
        // A literal step is resolved at compile time, which keeps the common case
        // at one comparison per iteration.  Only a step whose sign is unknown
        // until runtime pays for the dynamic form below.
        let static_step = match step {
            Some(step) => Self::const_int_expr(step),
            None => Some(1),
        };

        let loop_start = self.current_chunk().instructions.len();
        let cond_reg = self.allocate_register();
        match static_step {
            Some(value) if value < 0 => {
                self.emit(specialize(Instruction::Ge(cond_reg, var_reg, end_reg)), 0);
            }
            Some(_) => {
                self.emit(specialize(Instruction::Le(cond_reg, var_reg, end_reg)), 0);
            }
            None => {
                // cond = (step < 0) ? (var >= end) : (var <= end), spelled with
                // the instructions available:
                //   cond = (neg AND ge) OR ((NOT neg) AND le)
                let zero_reg = self.allocate_register();
                let const_idx = self.add_int_const(0);
                self.emit(Instruction::LoadConst(zero_reg, const_idx), 0);
                let neg_reg = self.allocate_register();
                self.emit(specialize(Instruction::Lt(neg_reg, step_reg, zero_reg)), 0);
                let pos_reg = self.allocate_register();
                self.emit(Instruction::Not(pos_reg, neg_reg), 0);

                let ge_reg = self.allocate_register();
                self.emit(specialize(Instruction::Ge(ge_reg, var_reg, end_reg)), 0);
                let le_reg = self.allocate_register();
                self.emit(specialize(Instruction::Le(le_reg, var_reg, end_reg)), 0);
                let neg_and_ge_reg = self.allocate_register();
                self.emit(Instruction::And(neg_and_ge_reg, neg_reg, ge_reg), 0);
                let pos_and_le_reg = self.allocate_register();
                self.emit(Instruction::And(pos_and_le_reg, pos_reg, le_reg), 0);
                self.emit(Instruction::Or(cond_reg, neg_and_ge_reg, pos_and_le_reg), 0);

                self.free_register(pos_and_le_reg);
                self.free_register(neg_and_ge_reg);
                self.free_register(le_reg);
                self.free_register(ge_reg);
                self.free_register(pos_reg);
                self.free_register(neg_reg);
                self.free_register(zero_reg);
            }
        }
        let jump_to_end = self.emit(Instruction::JumpIfNot(cond_reg, 0), 0);
        self.free_register(cond_reg);
        self.loop_contexts.push(LoopContext {
            continue_target: None,
            continue_jumps: Vec::new(),
            break_jumps: Vec::new(),
        });
        for stmt in body {
            self.compile_stmt(stmt)?;
        }

        let loop_ctx = self.loop_contexts.pop().unwrap();
        let increment_pos = self.current_chunk().instructions.len();
        let temp_reg = self.allocate_register();
        let increment = self.emit(specialize(Instruction::Add(temp_reg, var_reg, step_reg)), 0);
        self.move_result(var_reg, temp_reg, increment);
        self.free_register(temp_reg);
        for continue_jump in loop_ctx.continue_jumps {
            self.current_chunk_mut()
                .patch_jump(continue_jump, increment_pos);
        }

        self.emit_jump_back_to(loop_start);
        let end_pos = self.current_chunk().instructions.len();
        self.current_chunk_mut().patch_jump(jump_to_end, end_pos);
        for break_jump in loop_ctx.break_jumps {
            self.current_chunk_mut().patch_jump(break_jump, end_pos);
        }

        self.free_register(end_reg);
        self.free_register(step_reg);
        self.end_scope()?;
        Ok(())
    }

    pub(super) fn compile_for_in_loop(
        &mut self,
        variables: &[String],
        iterator: &Expr,
        body: &[Stmt],
    ) -> Result<()> {
        // `array.iter(...)` evaluates to an Iterator, so it must use the
        // generic `Iterator::next()` loop below rather than the indexed array
        // fast path. Preserve the existing receiver fast path for method-call
        // iterators and all other expressions.
        let is_array_iter_call = matches!(
            &iterator.kind,
            crate::ast::ExprKind::Call { callee, .. }
                if matches!(
                    &callee.kind,
                    crate::ast::ExprKind::FieldAccess { object, field }
                        if field == "iter"
                            && matches!(&object.kind, crate::ast::ExprKind::Identifier(name) if name == "array")
                )
        );
        let array_receiver_expr: Option<&Expr> = match &iterator.kind {
            _ if is_array_iter_call => None,
            crate::ast::ExprKind::MethodCall {
                receiver, method, ..
            } if method == "iter" => Some(receiver),
            _ => Some(iterator),
        };
        if variables.len() == 1
            && let ExprKind::Range { start, end, inclusive } = &iterator.kind
        {
            self.begin_scope();
            let start_reg = self.compile_expr(start)?;
            let end_reg = self.compile_expr(end)?;
            let var_reg = self.next_local_slot();
            self.emit(Instruction::Move(var_reg, start_reg), 0);
            self.free_register(start_reg);
            if let Some(scope) = self.scopes.last_mut() {
                scope.locals.insert(variables[0].clone(), (var_reg, false));
            }

            let loop_start = self.current_chunk().instructions.len();
            let cond_reg = self.allocate_register();
            if *inclusive {
                self.emit(Instruction::LeInt(cond_reg, var_reg, end_reg), 0);
            } else {
                self.emit(Instruction::LtInt(cond_reg, var_reg, end_reg), 0);
            }
            let jump_to_end = self.emit(Instruction::JumpIfNot(cond_reg, 0), 0);
            self.free_register(cond_reg);
            self.loop_contexts.push(LoopContext {
                continue_target: None,
                continue_jumps: Vec::new(),
                break_jumps: Vec::new(),
            });

            for stmt in body {
                self.compile_stmt(stmt)?;
            }

            let loop_ctx = self.loop_contexts.pop().unwrap();
            let increment_pos = self.current_chunk().instructions.len();
            let one_reg = self.allocate_register();
            let one_idx = self.add_int_const(1);
            self.emit(Instruction::LoadConst(one_reg, one_idx), 0);
            let tmp_reg = self.allocate_register();
            let increment = self.emit(Instruction::AddInt(tmp_reg, var_reg, one_reg), 0);
            self.move_result(var_reg, tmp_reg, increment);
            self.free_register(tmp_reg);
            self.free_register(one_reg);
            for continue_jump in loop_ctx.continue_jumps {
                self.current_chunk_mut()
                    .patch_jump(continue_jump, increment_pos);
            }

            self.emit_jump_back_to(loop_start);
            let end_pos = self.current_chunk().instructions.len();
            self.current_chunk_mut().patch_jump(jump_to_end, end_pos);
            for b in loop_ctx.break_jumps {
                self.current_chunk_mut().patch_jump(b, end_pos);
            }

            self.free_register(end_reg);
            self.free_register(var_reg);
            self.end_scope()?;
            return Ok(());
        }

        if variables.len() == 1
            && let Some(arr_expr) = array_receiver_expr
        {
            self.begin_scope();
            let array_reg = self.compile_expr(arr_expr)?;
            let elem_reg = self.allocate_register();
            if let Some(scope) = self.scopes.last_mut() {
                scope.locals.insert(variables[0].clone(), (elem_reg, false));
            }

            let i_reg = self.next_local_slot();
            let zero_idx = self.add_int_const(0);
            self.emit(Instruction::LoadConst(i_reg, zero_idx), 0);
            if let Some(scope) = self.scopes.last_mut() {
                scope
                    .locals
                    .insert("(for index)".to_string(), (i_reg, false));
            }

            let len_reg = self.next_local_slot();
            self.emit(Instruction::ArrayLen(len_reg, array_reg), 0);
            if let Some(scope) = self.scopes.last_mut() {
                scope
                    .locals
                    .insert("(for length)".to_string(), (len_reg, false));
            }

            let loop_start = self.current_chunk().instructions.len();
            let cond_reg = self.allocate_register();
            self.emit(Instruction::LtInt(cond_reg, i_reg, len_reg), 0);
            let jump_to_end = self.emit(Instruction::JumpIfNot(cond_reg, 0), 0);
            self.free_register(cond_reg);
            self.loop_contexts.push(LoopContext {
                continue_target: None,
                continue_jumps: Vec::new(),
                break_jumps: Vec::new(),
            });
            self.emit(Instruction::GetIndex(elem_reg, array_reg, i_reg), 0);
            for stmt in body {
                self.compile_stmt(stmt)?;
            }

            let loop_ctx = self.loop_contexts.pop().unwrap();
            let increment_pos = self.current_chunk().instructions.len();
            let one_reg = self.allocate_register();
            let one_idx = self.add_int_const(1);
            self.emit(Instruction::LoadConst(one_reg, one_idx), 0);
            let tmp_reg = self.allocate_register();
            let increment = self.emit(Instruction::AddInt(tmp_reg, i_reg, one_reg), 0);
            self.move_result(i_reg, tmp_reg, increment);
            self.free_register(tmp_reg);
            self.free_register(one_reg);
            for continue_jump in loop_ctx.continue_jumps {
                self.current_chunk_mut()
                    .patch_jump(continue_jump, increment_pos);
            }

            self.emit_jump_back_to(loop_start);
            let end_pos = self.current_chunk().instructions.len();
            self.current_chunk_mut().patch_jump(jump_to_end, end_pos);
            for b in loop_ctx.break_jumps {
                self.current_chunk_mut().patch_jump(b, end_pos);
            }

            self.free_register(elem_reg);
            self.free_register(len_reg);
            self.free_register(i_reg);
            self.free_register(array_reg);
            self.end_scope()?;
            return Ok(());
        }

        self.begin_scope();
        let iter_reg = if let crate::ast::ExprKind::MethodCall { method, .. } = &iterator.kind {
            if method == "iter" {
                self.compile_expr(iterator)?
            } else {
                let iter_src_reg = self.compile_expr(iterator)?;
                let iter_method_idx = self.add_string_constant("iter");
                let iter_reg = self.allocate_register();
                self.emit(
                    Instruction::CallMethod(iter_src_reg, iter_method_idx, 0, 0, iter_reg),
                    0,
                );
                self.free_register(iter_src_reg);
                iter_reg
            }
        } else {
            let iter_src_reg = self.compile_expr(iterator)?;
            let iter_method_idx = self.add_string_constant("iter");
            let iter_reg = self.allocate_register();
            self.emit(
                Instruction::CallMethod(iter_src_reg, iter_method_idx, 0, 0, iter_reg),
                0,
            );
            self.free_register(iter_src_reg);
            iter_reg
        };
        if let Some(scope) = self.scopes.last_mut() {
            scope
                .locals
                .insert("(for iterator)".to_string(), (iter_reg, false));
        }

        let loop_start = self.current_chunk().instructions.len();
        let next_method_idx = self.add_string_constant("next");
        let opt_reg = self.allocate_register();
        self.emit(
            Instruction::CallMethod(iter_reg, next_method_idx, 0, 0, opt_reg),
            0,
        );
        let enum_name_idx = self.add_string_constant("Option");
        let none_variant_idx = self.add_string_constant("None");
        let is_none_reg = self.allocate_register();
        self.emit(
            Instruction::IsEnumVariant(is_none_reg, opt_reg, enum_name_idx, none_variant_idx),
            0,
        );
        let jump_to_end = self.emit(Instruction::JumpIf(is_none_reg, 0), 0);
        self.free_register(is_none_reg);
        self.loop_contexts.push(LoopContext {
            continue_target: None,
            continue_jumps: Vec::new(),
            break_jumps: Vec::new(),
        });
        let unwrap_method_idx = self.add_string_constant("unwrap");
        let temp_val_reg = self.allocate_register();
        self.emit(
            Instruction::CallMethod(opt_reg, unwrap_method_idx, 0, 0, temp_val_reg),
            0,
        );
        if variables.len() == 1 {
            let var_reg = self.next_local_slot();
            self.emit(Instruction::Move(var_reg, temp_val_reg), 0);
            if let Some(scope) = self.scopes.last_mut() {
                scope.locals.insert(variables[0].clone(), (var_reg, false));
            }
        } else {
            for (i, var) in variables.iter().enumerate() {
                let var_reg = self.next_local_slot();
                let idx_reg = self.allocate_register();
                let idx_const = self.add_int_const(i as LustInt);
                self.emit(Instruction::LoadConst(idx_reg, idx_const), 0);
                self.emit(Instruction::GetIndex(var_reg, temp_val_reg, idx_reg), 0);
                self.free_register(idx_reg);
                if let Some(scope) = self.scopes.last_mut() {
                    scope.locals.insert(var.clone(), (var_reg, false));
                }
            }
        }

        for stmt in body {
            self.compile_stmt(stmt)?;
        }

        let loop_ctx = self.loop_contexts.pop().unwrap();
        for continue_jump in loop_ctx.continue_jumps {
            self.current_chunk_mut()
                .patch_jump(continue_jump, loop_start);
        }

        self.emit_jump_back_to(loop_start);
        let end_pos = self.current_chunk().instructions.len();
        self.current_chunk_mut().patch_jump(jump_to_end, end_pos);
        for break_jump in loop_ctx.break_jumps {
            self.current_chunk_mut().patch_jump(break_jump, end_pos);
        }

        self.free_register(iter_reg);
        self.end_scope()?;
        Ok(())
    }
}
