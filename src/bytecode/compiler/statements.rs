use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum HoistedScalarConst {
    Int(LustInt),
    FloatBits(crate::number::FloatBits),
}

impl HoistedScalarConst {
    fn scope_key(self) -> String {
        match self {
            Self::Int(v) => Compiler::hoisted_int_key(v),
            Self::FloatBits(bits) => Compiler::hoisted_float_key(bits),
        }
    }
}

impl Compiler {
    pub fn compile_stmt(&mut self, stmt: &Stmt) -> Result<()> {
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
                                        if let Some(tk) = self
                                            .functions
                                            .get(self.current_function)
                                            .and_then(|f| f.register_types.get(&src_reg).cloned())
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
                        if let Some(upvalue_idx) = self.lookup_upvalue(target_name) {
                            self.emit(Instruction::StoreUpvalue(upvalue_idx, target_reg), 0);
                        }
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
                        let compound_imm = if !self.loop_contexts.is_empty()
                            && self.numeric_type(target.span) == Some(NumericType::Int)
                            && self.numeric_type(value.span) == Some(NumericType::Int)
                        {
                            match *op {
                                BinaryOp::Add => Self::const_int_literal(value)
                                    .and_then(|v| i8::try_from(v).ok()),
                                BinaryOp::Sub => Self::const_int_literal(value)
                                    .and_then(|v| v.checked_neg())
                                    .and_then(|neg| i8::try_from(neg).ok()),
                                _ => None,
                            }
                        } else {
                            None
                        };
                        if let Some(imm) = compound_imm {
                            if target_is_local {
                                self.emit(Instruction::AddIntImm(target_reg, target_reg, imm), 0);
                                if let Some(upvalue_idx) = self.lookup_upvalue(&target_name) {
                                    self.emit(Instruction::StoreUpvalue(upvalue_idx, target_reg), 0);
                                }
                                if self.should_sync_module_local(&target_name) {
                                    self.emit_store_module_global(&target_name, target_reg);
                                }
                            } else {
                                let result_reg = self.allocate_register();
                                self.emit(Instruction::AddIntImm(result_reg, target_reg, imm), 0);
                                self.emit_store_module_global(&target_name, result_reg);
                                self.free_register(result_reg);
                                self.free_register(target_reg);
                            }
                        } else {
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

                                if let Some(upvalue_idx) = self.lookup_upvalue(&target_name) {
                                    self.emit(Instruction::StoreUpvalue(upvalue_idx, target_reg), 0);
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
                if !self.try_compile_array_void_call(expr)? {
                    let reg = self.compile_expr(expr)?;
                    self.free_register(reg);
                }
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
                if self.functions.is_empty() {
                    return Err(LustError::CompileError(
                        "'return' outside of function".to_string(),
                    ));
                }

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

                let is_static_err = values.len() == 1 && match &values[0].kind {
                    ExprKind::EnumConstructor { variant, .. } => variant == "Err",
                    ExprKind::Call { callee, .. } => {
                        if let ExprKind::FieldAccess { object, field } = &callee.kind {
                            if let ExprKind::Identifier(type_name) = &object.kind {
                                type_name == "Result" && field == "Err"
                            } else {
                                false
                            }
                        } else if let ExprKind::Identifier(fn_name) = &callee.kind {
                            fn_name == "Err"
                        } else {
                            false
                        }
                    }
                    _ => false,
                };

                let has_cleanups = self.scopes.iter().any(|s| !s.cleanups.is_empty());

                let (ret_reg, should_free_ret) = if values.is_empty() {
                    (255, false)
                } else if values.len() == 1 {
                    let expr_reg = self.compile_expr(&values[0])?;
                    if !has_cleanups {
                        (expr_reg, true)
                    } else {
                        let dedicated_reg = self.next_local_slot();
                        if let Some(scope) = self.scopes.first_mut() {
                            scope.locals.insert(format!("(return_{})", dedicated_reg), (dedicated_reg, false));
                        }
                        self.emit(Instruction::Move(dedicated_reg, expr_reg), 0);
                        self.free_register(expr_reg);
                        (dedicated_reg, false)
                    }
                } else {
                    let tuple_reg = self.compile_exprs_to_tuple(values)?;
                    if !has_cleanups {
                        (tuple_reg, true)
                    } else {
                        let dedicated_reg = self.next_local_slot();
                        if let Some(scope) = self.scopes.first_mut() {
                            scope.locals.insert(format!("(return_{})", dedicated_reg), (dedicated_reg, false));
                        }
                        self.emit(Instruction::Move(dedicated_reg, tuple_reg), 0);
                        self.free_register(tuple_reg);
                        (dedicated_reg, false)
                    }
                };

                if has_cleanups {
                    self.emit_defers_for_return(ret_reg, is_static_err)?;
                }
                self.emit(Instruction::Return(ret_reg), 0);
                if should_free_ret {
                    self.free_register(ret_reg);
                }
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
        let mut jumps_to_end = Vec::new();
        if !elseif_branches.is_empty() || else_block.is_some() {
            jumps_to_end.push(self.emit(Instruction::Jump(0), 0));
        }
        let next_branch_start = self.current_chunk().instructions.len();
        self.current_chunk_mut()
            .patch_jump(jump_to_else, next_branch_start);
        for (idx, (elseif_cond, elseif_body)) in elseif_branches.iter().enumerate() {
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
            let is_last_elseif = idx + 1 == elseif_branches.len();
            if !is_last_elseif || else_block.is_some() {
                jumps_to_end.push(self.emit(Instruction::Jump(0), 0));
            }
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

    fn is_simple_expr_for_const_hoisting(expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Literal(_)
            | ExprKind::RegexLiteral { .. }
            | ExprKind::Identifier(_) => true,
            ExprKind::Lambda { .. } | ExprKind::EffectBlock(_) => false,
            ExprKind::MethodCall {
                receiver,
                method,
                args,
                ..
            } => {
                if method == "reduce" {
                    return false;
                }
                Self::is_simple_expr_for_const_hoisting(receiver)
                    && args.iter().all(Self::is_simple_expr_for_const_hoisting)
            }
            ExprKind::Call { callee, args, .. } => {
                if let ExprKind::FieldAccess { field, .. } = &callee.kind
                    && field == "reduce"
                {
                    return false;
                }
                Self::is_simple_expr_for_const_hoisting(callee)
                    && args.iter().all(Self::is_simple_expr_for_const_hoisting)
            }
            ExprKind::Binary { left, right, .. } => {
                Self::is_simple_expr_for_const_hoisting(left)
                    && Self::is_simple_expr_for_const_hoisting(right)
            }
            ExprKind::Unary { operand, .. }
            | ExprKind::FieldAccess {
                object: operand, ..
            }
            | ExprKind::Paren(operand)
            | ExprKind::Cast { expr: operand, .. }
            | ExprKind::TypeCheck { expr: operand, .. }
            | ExprKind::IsPattern { expr: operand, .. } => {
                Self::is_simple_expr_for_const_hoisting(operand)
            }
            ExprKind::Index { object, index } => {
                Self::is_simple_expr_for_const_hoisting(object)
                    && Self::is_simple_expr_for_const_hoisting(index)
            }
            ExprKind::Range { start, end, .. } => {
                Self::is_simple_expr_for_const_hoisting(start)
                    && Self::is_simple_expr_for_const_hoisting(end)
            }
            ExprKind::Array(elems)
            | ExprKind::Tuple(elems)
            | ExprKind::EnumConstructor { args: elems, .. }
            | ExprKind::Return(elems) => elems.iter().all(Self::is_simple_expr_for_const_hoisting),
            ExprKind::Map(entries) => entries.iter().all(|(k, v)| {
                Self::is_simple_expr_for_const_hoisting(k)
                    && Self::is_simple_expr_for_const_hoisting(v)
            }),
            ExprKind::StructLiteral { fields, .. } => fields
                .iter()
                .all(|f| Self::is_simple_expr_for_const_hoisting(&f.value)),
            ExprKind::Block(stmts) => stmts.iter().all(Self::is_simple_stmt_for_const_hoisting),
            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                Self::is_simple_expr_for_const_hoisting(condition)
                    && Self::is_simple_expr_for_const_hoisting(then_branch)
                    && else_branch
                        .as_deref()
                        .is_none_or(Self::is_simple_expr_for_const_hoisting)
            }
        }
    }

    fn is_simple_stmt_for_const_hoisting(stmt: &Stmt) -> bool {
        match &stmt.kind {
            StmtKind::Defer(_) | StmtKind::ErrDefer(_) => false,
            StmtKind::Break | StmtKind::Continue => true,
            StmtKind::Local { initializer, .. } => initializer
                .as_ref()
                .is_none_or(|vals| vals.iter().all(Self::is_simple_expr_for_const_hoisting)),
            StmtKind::Assign { targets, values } => {
                targets.iter().all(Self::is_simple_expr_for_const_hoisting)
                    && values.iter().all(Self::is_simple_expr_for_const_hoisting)
            }
            StmtKind::CompoundAssign { target, value, .. } => {
                Self::is_simple_expr_for_const_hoisting(target)
                    && Self::is_simple_expr_for_const_hoisting(value)
            }
            StmtKind::Expr(expr) => Self::is_simple_expr_for_const_hoisting(expr),
            StmtKind::Return(values) => values.iter().all(Self::is_simple_expr_for_const_hoisting),
            StmtKind::Block(stmts) | StmtKind::Scoped(stmts) => {
                stmts.iter().all(Self::is_simple_stmt_for_const_hoisting)
            }
            StmtKind::If {
                condition,
                then_block,
                elseif_branches,
                else_block,
            } => {
                Self::is_simple_expr_for_const_hoisting(condition)
                    && then_block.iter().all(Self::is_simple_stmt_for_const_hoisting)
                    && elseif_branches.iter().all(|(cond, blk)| {
                        Self::is_simple_expr_for_const_hoisting(cond)
                            && blk.iter().all(Self::is_simple_stmt_for_const_hoisting)
                    })
                    && else_block
                        .as_ref()
                        .is_none_or(|blk| blk.iter().all(Self::is_simple_stmt_for_const_hoisting))
            }
            StmtKind::While { condition, body } => {
                Self::is_simple_expr_for_const_hoisting(condition)
                    && body.iter().all(Self::is_simple_stmt_for_const_hoisting)
            }
            StmtKind::ForNumeric {
                start,
                end,
                step,
                body,
                ..
            } => {
                Self::is_simple_expr_for_const_hoisting(start)
                    && Self::is_simple_expr_for_const_hoisting(end)
                    && step
                        .as_ref()
                        .is_none_or(Self::is_simple_expr_for_const_hoisting)
                    && body.iter().all(Self::is_simple_stmt_for_const_hoisting)
            }
            StmtKind::ForIn { iterator, body, .. } => {
                Self::is_simple_expr_for_const_hoisting(iterator)
                    && body.iter().all(Self::is_simple_stmt_for_const_hoisting)
            }
        }
    }

    fn collect_hoistable_consts_in_expr(
        &self,
        expr: &Expr,
        seen: &mut HashSet<HoistedScalarConst>,
        out: &mut Vec<HoistedScalarConst>,
    ) {
        const MAX_HOISTED_WHILE_CONSTS: usize = 16;
        if out.len() >= MAX_HOISTED_WHILE_CONSTS {
            return;
        }
        match &expr.kind {
            ExprKind::Literal(Literal::Integer(v)) => {
                let key = HoistedScalarConst::Int(*v);
                if seen.insert(key) {
                    out.push(key);
                }
            }
            ExprKind::Literal(Literal::Float(f)) => {
                let key = HoistedScalarConst::FloatBits(f.to_bits());
                if seen.insert(key) {
                    out.push(key);
                }
            }
            ExprKind::Binary { left, op, right } => {
                if self.numeric_type(left.span) == Some(NumericType::Int)
                    && self.numeric_type(right.span) == Some(NumericType::Int)
                    && let Some((operand_expr, _)) = Self::try_int_add_imm(*op, left, right)
                {
                    self.collect_hoistable_consts_in_expr(operand_expr, seen, out);
                    return;
                }
                self.collect_hoistable_consts_in_expr(left, seen, out);
                self.collect_hoistable_consts_in_expr(right, seen, out);
            }
            ExprKind::Unary { operand, .. }
            | ExprKind::FieldAccess {
                object: operand, ..
            }
            | ExprKind::Paren(operand)
            | ExprKind::Cast { expr: operand, .. }
            | ExprKind::TypeCheck { expr: operand, .. }
            | ExprKind::IsPattern { expr: operand, .. } => {
                self.collect_hoistable_consts_in_expr(operand, seen, out);
            }
            ExprKind::Index { object, index } => {
                self.collect_hoistable_consts_in_expr(object, seen, out);
                self.collect_hoistable_consts_in_expr(index, seen, out);
            }
            ExprKind::Call { callee, args, .. } => {
                self.collect_hoistable_consts_in_expr(callee, seen, out);
                for arg in args {
                    self.collect_hoistable_consts_in_expr(arg, seen, out);
                }
            }
            ExprKind::MethodCall { receiver, args, .. } => {
                self.collect_hoistable_consts_in_expr(receiver, seen, out);
                for arg in args {
                    self.collect_hoistable_consts_in_expr(arg, seen, out);
                }
            }
            ExprKind::Array(elems)
            | ExprKind::Tuple(elems)
            | ExprKind::EnumConstructor { args: elems, .. }
            | ExprKind::Return(elems) => {
                for elem in elems {
                    self.collect_hoistable_consts_in_expr(elem, seen, out);
                }
            }
            ExprKind::Map(entries) => {
                for (k, v) in entries {
                    self.collect_hoistable_consts_in_expr(k, seen, out);
                    self.collect_hoistable_consts_in_expr(v, seen, out);
                }
            }
            ExprKind::StructLiteral { fields, .. } => {
                for f in fields {
                    self.collect_hoistable_consts_in_expr(&f.value, seen, out);
                }
            }
            ExprKind::Range { start, end, .. } => {
                self.collect_hoistable_consts_in_expr(start, seen, out);
                self.collect_hoistable_consts_in_expr(end, seen, out);
            }
            ExprKind::Block(stmts) => {
                for stmt in stmts {
                    self.collect_hoistable_consts_in_stmt(stmt, seen, out);
                }
            }
            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.collect_hoistable_consts_in_expr(condition, seen, out);
                self.collect_hoistable_consts_in_expr(then_branch, seen, out);
                if let Some(else_expr) = else_branch {
                    self.collect_hoistable_consts_in_expr(else_expr, seen, out);
                }
            }
            ExprKind::Literal(_)
            | ExprKind::RegexLiteral { .. }
            | ExprKind::Identifier(_)
            | ExprKind::Lambda { .. }
            | ExprKind::EffectBlock(_) => {}
        }
    }

    fn collect_hoistable_consts_in_stmt(
        &self,
        stmt: &Stmt,
        seen: &mut HashSet<HoistedScalarConst>,
        out: &mut Vec<HoistedScalarConst>,
    ) {
        const MAX_HOISTED_WHILE_CONSTS: usize = 16;
        if out.len() >= MAX_HOISTED_WHILE_CONSTS {
            return;
        }
        match &stmt.kind {
            StmtKind::Local { initializer, .. } => {
                if let Some(vals) = initializer {
                    for val in vals {
                        self.collect_hoistable_consts_in_expr(val, seen, out);
                    }
                }
            }
            StmtKind::Assign { targets, values } => {
                for target in targets {
                    self.collect_hoistable_consts_in_expr(target, seen, out);
                }
                for val in values {
                    self.collect_hoistable_consts_in_expr(val, seen, out);
                }
            }
            StmtKind::CompoundAssign { target, op, value } => {
                let folded_imm = matches!(target.kind, ExprKind::Identifier(_))
                    && self.numeric_type(target.span) == Some(NumericType::Int)
                    && self.numeric_type(value.span) == Some(NumericType::Int)
                    && match *op {
                        BinaryOp::Add => Self::const_int_literal(value)
                            .and_then(|v| i8::try_from(v).ok())
                            .is_some(),
                        BinaryOp::Sub => Self::const_int_literal(value)
                            .and_then(|v| v.checked_neg())
                            .and_then(|neg| i8::try_from(neg).ok())
                            .is_some(),
                        _ => false,
                    };
                self.collect_hoistable_consts_in_expr(target, seen, out);
                if !folded_imm {
                    self.collect_hoistable_consts_in_expr(value, seen, out);
                }
            }
            StmtKind::Expr(expr) => self.collect_hoistable_consts_in_expr(expr, seen, out),
            StmtKind::Return(values) => {
                for val in values {
                    self.collect_hoistable_consts_in_expr(val, seen, out);
                }
            }
            StmtKind::Block(stmts) | StmtKind::Scoped(stmts) => {
                for s in stmts {
                    self.collect_hoistable_consts_in_stmt(s, seen, out);
                }
            }
            StmtKind::If {
                condition,
                then_block,
                elseif_branches,
                else_block,
            } => {
                self.collect_hoistable_consts_in_expr(condition, seen, out);
                for s in then_block {
                    self.collect_hoistable_consts_in_stmt(s, seen, out);
                }
                for (cond, blk) in elseif_branches {
                    self.collect_hoistable_consts_in_expr(cond, seen, out);
                    for s in blk {
                        self.collect_hoistable_consts_in_stmt(s, seen, out);
                    }
                }
                if let Some(blk) = else_block {
                    for s in blk {
                        self.collect_hoistable_consts_in_stmt(s, seen, out);
                    }
                }
            }
            StmtKind::While { condition, body } => {
                self.collect_hoistable_consts_in_expr(condition, seen, out);
                for s in body {
                    self.collect_hoistable_consts_in_stmt(s, seen, out);
                }
            }
            StmtKind::ForNumeric {
                start,
                end,
                step,
                body,
                ..
            } => {
                self.collect_hoistable_consts_in_expr(start, seen, out);
                self.collect_hoistable_consts_in_expr(end, seen, out);
                if let Some(step_expr) = step {
                    self.collect_hoistable_consts_in_expr(step_expr, seen, out);
                }
                for s in body {
                    self.collect_hoistable_consts_in_stmt(s, seen, out);
                }
            }
            StmtKind::ForIn { iterator, body, .. } => {
                self.collect_hoistable_consts_in_expr(iterator, seen, out);
                for s in body {
                    self.collect_hoistable_consts_in_stmt(s, seen, out);
                }
            }
            StmtKind::Break | StmtKind::Continue | StmtKind::Defer(_) | StmtKind::ErrDefer(_) => {}
        }
    }

    fn compile_while_condition_branch(&mut self, condition: &Expr) -> Result<usize> {
        let mut cond_expr = condition;
        while let ExprKind::Paren(inner) = &cond_expr.kind {
            cond_expr = inner;
        }
        if let ExprKind::Binary { left, op, right } = &cond_expr.kind
            && matches!(op, BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge)
            && self.numeric_type(left.span) == Some(NumericType::Int)
            && self.numeric_type(right.span) == Some(NumericType::Int)
        {
            self.current_line = cond_expr.span.start_line;
            let left_reg = self.compile_expr(left)?;
            let saved_freereg = self.next_register;
            let right_reg = self.compile_expr(right)?;
            if self.next_register < saved_freereg {
                self.next_register = saved_freereg;
            }
            let branch_instr = match op {
                BinaryOp::Lt => Instruction::BranchLtInt(left_reg, right_reg, 0),
                BinaryOp::Le => Instruction::BranchLeInt(left_reg, right_reg, 0),
                BinaryOp::Gt => Instruction::BranchLtInt(right_reg, left_reg, 0),
                BinaryOp::Ge => Instruction::BranchLeInt(right_reg, left_reg, 0),
                _ => unreachable!(),
            };
            let jump_to_end = self.emit(branch_instr, 0);
            self.free_register(right_reg);
            self.free_register(left_reg);
            return Ok(jump_to_end);
        }

        let cond_reg = self.compile_expr(condition)?;
        let jump_to_end = self.emit(Instruction::JumpIfNot(cond_reg, 0), 0);
        self.free_register(cond_reg);
        Ok(jump_to_end)
    }

    pub(super) fn compile_while_loop(&mut self, condition: &Expr, body: &[Stmt]) -> Result<()> {
        let mut hoisted_scope_active = false;
        let base_before = (self.max_local_register_index() + 1).max(0) as usize;
        if base_before < 192
            && Self::is_simple_expr_for_const_hoisting(condition)
            && body.iter().all(Self::is_simple_stmt_for_const_hoisting)
        {
            let mut seen = HashSet::new();
            let mut candidates = Vec::new();
            self.collect_hoistable_consts_in_expr(condition, &mut seen, &mut candidates);
            for stmt in body {
                self.collect_hoistable_consts_in_stmt(stmt, &mut seen, &mut candidates);
            }
            if !candidates.is_empty() {
                self.begin_scope();
                hoisted_scope_active = true;
                let mut first_reg: Option<Register> = None;
                for candidate in candidates {
                    let cur_base = (self.max_local_register_index() + 1).max(0) as usize;
                    if cur_base >= 200 {
                        break;
                    }
                    let key = candidate.scope_key();
                    if self.resolve_local(&key).is_ok() {
                        continue;
                    }
                    let reg = self.next_local_slot();
                    let const_idx = match candidate {
                        HoistedScalarConst::Int(v) => self.add_int_const(v),
                        HoistedScalarConst::FloatBits(bits) => self.add_constant(Value::Float(
                            crate::number::LustFloat::from_bits(bits),
                        )),
                    };
                    self.emit(Instruction::LoadConst(reg, const_idx), 0);
                    if first_reg.is_none() {
                        first_reg = Some(reg);
                    }
                    if let Some(scope) = self.scopes.last_mut() {
                        scope.locals.insert(key, (reg, false));
                    }
                }
                if let Some(reg) = first_reg {
                    if let Some(scope) = self.scopes.last_mut() {
                        scope
                            .locals
                            .insert("(hoist_active)".to_string(), (reg, false));
                    }
                } else {
                    self.end_scope()?;
                    hoisted_scope_active = false;
                }
                self.reset_temp_registers();
            }
        }

        let loop_start = self.current_chunk().instructions.len();
        let jump_to_end = self.compile_while_condition_branch(condition)?;
        self.reset_temp_registers();
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

        if hoisted_scope_active {
            self.end_scope()?;
            self.reset_temp_registers();
        }

        Ok(())
    }

    /// The value of an integer-literal expression, if it is one.
    ///
    /// Used to settle a numeric `for` loop's comparison direction at compile
    /// time.  Only the forms a step is actually written in are folded: a literal,
    /// a negated literal, and either wrapped in parentheses.
    fn const_int_expr(expr: &Expr) -> Option<LustInt> {
        Self::const_int_literal(expr)
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
        let jump_to_end = if int_loop
            && let Some(value) = static_step
        {
            if value < 0 {
                self.emit(Instruction::BranchLeInt(end_reg, var_reg, 0), 0)
            } else {
                self.emit(Instruction::BranchLeInt(var_reg, end_reg, 0), 0)
            }
        } else {
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
            let jump = self.emit(Instruction::JumpIfNot(cond_reg, 0), 0);
            self.free_register(cond_reg);
            jump
        };
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
        if int_loop
            && let Some(step_val) = static_step
            && let Ok(imm) = i8::try_from(step_val)
        {
            self.emit(Instruction::AddIntImm(var_reg, var_reg, imm), 0);
        } else {
            let temp_reg = self.allocate_register();
            let increment = self.emit(specialize(Instruction::Add(temp_reg, var_reg, step_reg)), 0);
            self.move_result(var_reg, temp_reg, increment);
            self.free_register(temp_reg);
        }
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
        let array_receiver_expr: Option<&Expr> = match &iterator.kind {
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
            let jump_to_end = if *inclusive {
                self.emit(Instruction::BranchLeInt(var_reg, end_reg, 0), 0)
            } else {
                self.emit(Instruction::BranchLtInt(var_reg, end_reg, 0), 0)
            };
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
            self.emit(Instruction::AddIntImm(var_reg, var_reg, 1), 0);
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
            let jump_to_end = self.emit(Instruction::BranchLtInt(i_reg, len_reg, 0), 0);
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
            self.emit(Instruction::AddIntImm(i_reg, i_reg, 1), 0);
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
