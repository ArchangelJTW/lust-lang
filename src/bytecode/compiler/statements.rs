use super::*;
impl Compiler {
    pub(super) fn compile_stmt(&mut self, stmt: &Stmt) -> Result<()> {
        self.current_line = stmt.span.start_line;
        match &stmt.kind {
            StmtKind::Local {
                bindings,
                mutable,
                initializer,
            } => {
                let mut binding_regs = Vec::new();
                for binding in bindings {
                    let reg = self.next_local_slot();
                    if let Some(type_ann) = &binding.type_annotation {
                        self.register_type(reg, type_ann.kind.clone());
                    }

                    if let Some(scope) = self.scopes.last_mut() {
                        scope.locals.insert(binding.name.clone(), (reg, *mutable));
                    }

                    binding_regs.push((binding.name.clone(), reg));
                }

                if let Some(values) = initializer {
                    if bindings.len() == 1 && values.len() == 1 {
                        let value_reg = self.compile_expr(&values[0])?;
                        let (name, target_reg) = &binding_regs[0];
                        if value_reg != *target_reg {
                            self.emit(Instruction::Move(*target_reg, value_reg), 0);
                        }

                        if self.should_sync_module_local(name) {
                            self.emit_store_module_global(name, *target_reg);
                        }

                        self.free_register(value_reg);
                    } else {
                        let tuple_reg = self.compile_exprs_to_tuple(values)?;
                        for (index, (name, target_reg)) in binding_regs.iter().enumerate() {
                            let value_reg = self.allocate_register();
                            self.emit(Instruction::TupleGet(value_reg, tuple_reg, index as u8), 0);
                            self.emit(Instruction::Move(*target_reg, value_reg), 0);
                            if self.should_sync_module_local(name) {
                                self.emit_store_module_global(name, *target_reg);
                            }

                            self.free_register(value_reg);
                        }

                        self.free_register(tuple_reg);
                    }
                } else {
                    for (name, target_reg) in &binding_regs {
                        self.emit(Instruction::LoadNil(*target_reg), 0);
                        if self.should_sync_module_local(name) {
                            self.emit_store_module_global(name, *target_reg);
                        }
                    }
                }
            }

            StmtKind::Assign { targets, values } => {
                if targets.len() == 1 && values.len() == 1 {
                    if let (
                        ExprKind::Identifier(target_name),
                        ExprKind::Binary {
                            op: BinaryOp::Concat,
                            left,
                            right,
                        },
                    ) = (&targets[0].kind, &values[0].kind)
                    {
                        if let ExprKind::Identifier(left_name) = &left.kind {
                            if left_name == target_name {
                                if let Ok(target_reg) = self.resolve_local(target_name) {
                                    let rhs_reg = self.compile_expr(right)?;
                                    self.emit(
                                        Instruction::Concat(target_reg, target_reg, rhs_reg),
                                        0,
                                    );
                                    if self.should_sync_module_local(target_name) {
                                        self.emit_store_module_global(target_name, target_reg);
                                    }

                                    self.free_register(rhs_reg);
                                    return Ok(());
                                }
                            }
                        }
                    }

                    let value_reg = self.compile_expr(&values[0])?;
                    self.assign_value_to_target(&targets[0], value_reg)?;
                    self.free_register(value_reg);
                } else {
                    let tuple_reg = self.compile_exprs_to_tuple(values)?;
                    for (index, target) in targets.iter().enumerate() {
                        let value_reg = self.allocate_register();
                        self.emit(Instruction::TupleGet(value_reg, tuple_reg, index as u8), 0);
                        self.assign_value_to_target(target, value_reg)?;
                        self.free_register(value_reg);
                    }

                    self.free_register(tuple_reg);
                }
            }

            StmtKind::CompoundAssign { target, op, value } => {
                let (target_reg, target_is_local, target_name) = match &target.kind {
                    ExprKind::Identifier(name) => {
                        if let Ok(local_reg) = self.resolve_local(name) {
                            (local_reg, true, name.clone())
                        } else if self.is_module_level_identifier(name) {
                            let reg = self.allocate_register();
                            self.emit_load_module_global(&name, reg)?;
                            (reg, false, name.clone())
                        } else {
                            return Err(LustError::CompileError(
                                "Invalid compound assignment target".to_string(),
                            ));
                        }
                    }

                    _ => {
                        return Err(LustError::CompileError(
                            "Invalid compound assignment target".to_string(),
                        ))
                    }
                };
                let value_reg = self.compile_expr(value)?;
                let result_reg = self.allocate_register();
                self.compile_binary_op(*op, result_reg, target_reg, value_reg)?;
                if target_is_local {
                    if result_reg != target_reg {
                        self.emit(Instruction::Move(target_reg, result_reg), 0);
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
                if values.is_empty() {
                    self.emit(Instruction::Return(255), 0);
                } else if values.len() == 1 {
                    let reg = self.compile_expr(&values[0])?;
                    self.emit(Instruction::Return(reg), 0);
                } else {
                    let tuple_reg = self.compile_exprs_to_tuple(values)?;
                    self.emit(Instruction::Return(tuple_reg), 0);
                }
            }

            StmtKind::Block(stmts) => {
                self.begin_scope();
                for stmt in stmts {
                    self.compile_stmt(stmt)?;
                    self.reset_temp_registers();
                }

                self.end_scope();
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
        }

        Ok(())
    }

    pub(super) fn compile_if_stmt(
        &mut self,
        condition: &Expr,
        then_block: &[Stmt],
        elseif_branches: &[(Expr, Vec<Stmt>)],
        else_block: &Option<Vec<Stmt>>,
    ) -> Result<()> {
        let is_standalone_pattern = matches!(&condition.kind, ExprKind::IsPattern { .. });
        let pattern_bindings: Vec<(&Expr, &crate::ast::Pattern)> = if is_standalone_pattern {
            self.extract_all_pattern_bindings(condition)
        } else {
            Vec::new()
        };
        self.begin_scope();
        let cond_reg = self.compile_expr(condition)?;
        let jump_to_else = self.emit(Instruction::JumpIfNot(cond_reg, 0), 0);
        self.free_register(cond_reg);
        self.reset_temp_registers();
        for (scrutinee_expr, pattern) in pattern_bindings {
            let enum_reg = self.compile_expr(scrutinee_expr)?;
            self.bind_pattern_variables(enum_reg, pattern)?;
            self.free_register(enum_reg);
        }

        for stmt in then_block {
            self.compile_stmt(stmt)?;
        }

        self.end_scope();
        let mut jumps_to_end = vec![self.emit(Instruction::Jump(0), 0)];
        let next_branch_start = self.current_chunk().instructions.len();
        self.current_chunk_mut()
            .patch_jump(jump_to_else, next_branch_start);
        for (elseif_cond, elseif_body) in elseif_branches {
            let is_standalone_pattern = matches!(&elseif_cond.kind, ExprKind::IsPattern { .. });
            let elseif_pattern_bindings = if is_standalone_pattern {
                self.extract_all_pattern_bindings(elseif_cond)
            } else {
                Vec::new()
            };
            let elseif_cond_reg = self.compile_expr(elseif_cond)?;
            let jump_to_next = self.emit(Instruction::JumpIfNot(elseif_cond_reg, 0), 0);
            self.free_register(elseif_cond_reg);
            self.begin_scope();
            self.reset_temp_registers();
            for (scrutinee_expr, pattern) in elseif_pattern_bindings {
                let enum_reg = self.compile_expr(scrutinee_expr)?;
                self.bind_pattern_variables(enum_reg, pattern)?;
                self.free_register(enum_reg);
            }

            for stmt in elseif_body {
                self.compile_stmt(stmt)?;
            }

            self.end_scope();
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

            self.end_scope();
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

        self.end_scope();
        let loop_ctx = self.loop_contexts.pop().unwrap();
        self.emit_jump_back_to(loop_start);
        let end_pos = self.current_chunk().instructions.len();
        self.current_chunk_mut().patch_jump(jump_to_end, end_pos);
        for break_jump in loop_ctx.break_jumps {
            self.current_chunk_mut().patch_jump(break_jump, end_pos);
        }

        Ok(())
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
                .insert(format!("(for limit)"), (end_reg, false));
            scope
                .locals
                .insert(format!("(for step)"), (step_reg, false));
        }

        let loop_start = self.current_chunk().instructions.len();
        let cond_reg = self.allocate_register();
        self.emit(Instruction::Le(cond_reg, var_reg, end_reg), 0);
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
        self.emit(Instruction::Add(temp_reg, var_reg, step_reg), 0);
        self.emit(Instruction::Move(var_reg, temp_reg), 0);
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
        self.end_scope();
        Ok(())
    }

    pub(super) fn compile_for_in_loop(
        &mut self,
        variables: &[String],
        iterator: &Expr,
        body: &[Stmt],
    ) -> Result<()> {
        let array_receiver_expr: Option<&Expr> = match &iterator.kind {
            crate::ast::ExprKind::MethodCall {
                receiver, method, ..
            } if method == "iter" => Some(receiver),
            _ => Some(iterator),
        };
        if variables.len() == 1 {
            if let Some(arr_expr) = array_receiver_expr {
                self.begin_scope();
                let array_reg = self.compile_expr(arr_expr)?;
                let elem_reg = self.next_local_slot();
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
                self.emit(Instruction::Lt(cond_reg, i_reg, len_reg), 0);
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
                self.emit(Instruction::Add(tmp_reg, i_reg, one_reg), 0);
                self.emit(Instruction::Move(i_reg, tmp_reg), 0);
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

                self.free_register(len_reg);
                self.free_register(i_reg);
                self.free_register(array_reg);
                self.end_scope();
                return Ok(());
            }
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
                .insert(format!("(for iterator)"), (iter_reg, false));
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
                let idx_const = self.add_int_const(i as i64);
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
        self.end_scope();
        Ok(())
    }
}
