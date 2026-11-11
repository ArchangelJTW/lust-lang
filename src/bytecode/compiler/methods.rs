use super::*;
impl Compiler {
    pub(super) fn compile_map_method(
        &mut self,
        array_expr: &Expr,
        lambda_expr: &Expr,
    ) -> Result<Register> {
        let array_reg = self.compile_expr(array_expr)?;
        let lambda_reg = self.compile_expr(lambda_expr)?;
        let result_reg = self.allocate_register();
        let dummy_elem_reg = self.allocate_register();
        self.emit(Instruction::NewArray(result_reg, dummy_elem_reg, 0), 0);
        let len_reg = self.allocate_register();
        self.emit(Instruction::ArrayLen(len_reg, array_reg), 0);
        let i_reg = self.allocate_register();
        let zero_const_idx = self.add_int_const(0);
        self.emit(Instruction::LoadConst(i_reg, zero_const_idx), 0);
        let end_reg = self.allocate_register();
        let one_const_idx = self.add_int_const(1);
        let one_reg = self.allocate_register();
        self.emit(Instruction::LoadConst(one_reg, one_const_idx), 0);
        self.emit(Instruction::SubInt(end_reg, len_reg, one_reg), 0);
        let loop_watermark = self.next_register;
        let loop_start = self.current_chunk().instructions.len();
        let cond_reg = self.allocate_register();
        self.emit(Instruction::LeInt(cond_reg, i_reg, end_reg), 0);
        let jump_to_end = self.emit(Instruction::JumpIfNot(cond_reg, 0), 0);
        let elem_reg = self.allocate_register();
        self.emit(Instruction::GetIndex(elem_reg, array_reg, i_reg), 0);
        let mapped_reg = self.allocate_register();
        self.emit(Instruction::Call(lambda_reg, elem_reg, 1, mapped_reg), 0);
        let push_method_idx = self.add_string_constant("push");
        let push_result_reg = self.allocate_register();
        self.emit(
            Instruction::CallMethod(result_reg, push_method_idx, mapped_reg, 1, push_result_reg),
            0,
        );
        let inc_reg = self.allocate_register();
        let one_reg2 = self.allocate_register();
        self.emit(Instruction::LoadConst(one_reg2, one_const_idx), 0);
        self.emit(Instruction::AddInt(inc_reg, i_reg, one_reg2), 0);
        self.emit(Instruction::Move(i_reg, inc_reg), 0);
        self.next_register = loop_watermark;
        self.emit_jump_back_to(loop_start);
        let end_pos = self.current_chunk().instructions.len();
        self.current_chunk_mut().patch_jump(jump_to_end, end_pos);
        Ok(result_reg)
    }

    pub(super) fn compile_filter_method(
        &mut self,
        array_expr: &Expr,
        predicate_expr: &Expr,
    ) -> Result<Register> {
        let array_reg = self.compile_expr(array_expr)?;
        let predicate_reg = self.compile_expr(predicate_expr)?;
        let result_reg = self.allocate_register();
        let dummy_elem_reg = self.allocate_register();
        self.emit(Instruction::NewArray(result_reg, dummy_elem_reg, 0), 0);
        let len_reg = self.allocate_register();
        self.emit(Instruction::ArrayLen(len_reg, array_reg), 0);
        let i_reg = self.allocate_register();
        let zero_const_idx = self.add_int_const(0);
        self.emit(Instruction::LoadConst(i_reg, zero_const_idx), 0);
        let end_reg = self.allocate_register();
        let one_const_idx = self.add_int_const(1);
        let one_reg = self.allocate_register();
        self.emit(Instruction::LoadConst(one_reg, one_const_idx), 0);
        self.emit(Instruction::SubInt(end_reg, len_reg, one_reg), 0);
        let loop_watermark = self.next_register;
        let loop_start = self.current_chunk().instructions.len();
        let cond_reg = self.allocate_register();
        self.emit(Instruction::LeInt(cond_reg, i_reg, end_reg), 0);
        let jump_to_end = self.emit(Instruction::JumpIfNot(cond_reg, 0), 0);
        let elem_reg = self.allocate_register();
        self.emit(Instruction::GetIndex(elem_reg, array_reg, i_reg), 0);
        let pred_result_reg = self.allocate_register();
        self.emit(
            Instruction::Call(predicate_reg, elem_reg, 1, pred_result_reg),
            0,
        );
        let jump_to_next_iter = self.emit(Instruction::JumpIfNot(pred_result_reg, 0), 0);
        let push_method_idx = self.add_string_constant("push");
        let push_result_reg = self.allocate_register();
        self.emit(
            Instruction::CallMethod(result_reg, push_method_idx, elem_reg, 1, push_result_reg),
            0,
        );
        let next_iter_pos = self.current_chunk().instructions.len();
        self.current_chunk_mut()
            .patch_jump(jump_to_next_iter, next_iter_pos);
        let inc_reg = self.allocate_register();
        let one_reg2 = self.allocate_register();
        self.emit(Instruction::LoadConst(one_reg2, one_const_idx), 0);
        self.emit(Instruction::AddInt(inc_reg, i_reg, one_reg2), 0);
        self.emit(Instruction::Move(i_reg, inc_reg), 0);
        self.next_register = loop_watermark;
        self.emit_jump_back_to(loop_start);
        let end_pos = self.current_chunk().instructions.len();
        self.current_chunk_mut().patch_jump(jump_to_end, end_pos);
        Ok(result_reg)
    }

    pub(super) fn compile_reduce_method(
        &mut self,
        array_expr: &Expr,
        init_expr: &Expr,
        reducer_expr: &Expr,
    ) -> Result<Register> {
        let array_reg = self.compile_expr(array_expr)?;
        let acc_reg = self.compile_expr(init_expr)?;
        let reducer_reg = self.compile_expr(reducer_expr)?;
        let len_reg = self.allocate_register();
        self.emit(Instruction::ArrayLen(len_reg, array_reg), 0);
        let i_reg = self.allocate_register();
        let zero_const_idx = self.add_int_const(0);
        self.emit(Instruction::LoadConst(i_reg, zero_const_idx), 0);
        let end_reg = self.allocate_register();
        let one_const_idx = self.add_int_const(1);
        let one_reg = self.allocate_register();
        self.emit(Instruction::LoadConst(one_reg, one_const_idx), 0);
        self.emit(Instruction::SubInt(end_reg, len_reg, one_reg), 0);
        let loop_watermark = self.next_register;
        let loop_start = self.current_chunk().instructions.len();
        let cond_reg = self.allocate_register();
        self.emit(Instruction::LeInt(cond_reg, i_reg, end_reg), 0);
        let jump_to_end = self.emit(Instruction::JumpIfNot(cond_reg, 0), 0);
        let elem_reg = self.allocate_register();
        self.emit(Instruction::GetIndex(elem_reg, array_reg, i_reg), 0);
        let arg_base = self.next_register;
        let arg1_reg = self.allocate_register();
        let arg2_reg = self.allocate_register();
        if acc_reg != arg1_reg {
            self.emit(Instruction::Move(arg1_reg, acc_reg), 0);
        }

        if elem_reg != arg2_reg {
            self.emit(Instruction::Move(arg2_reg, elem_reg), 0);
        }

        let new_acc_reg = self.allocate_register();
        self.emit(Instruction::Call(reducer_reg, arg_base, 2, new_acc_reg), 0);
        self.emit(Instruction::Move(acc_reg, new_acc_reg), 0);
        let inc_reg = self.allocate_register();
        let one_reg2 = self.allocate_register();
        self.emit(Instruction::LoadConst(one_reg2, one_const_idx), 0);
        self.emit(Instruction::AddInt(inc_reg, i_reg, one_reg2), 0);
        self.emit(Instruction::Move(i_reg, inc_reg), 0);
        self.next_register = loop_watermark;
        self.emit_jump_back_to(loop_start);
        let end_pos = self.current_chunk().instructions.len();
        self.current_chunk_mut().patch_jump(jump_to_end, end_pos);
        Ok(acc_reg)
    }

    pub(super) fn build_effect_thunk(
        &mut self,
        name: &str,
        stmts: &[Stmt],
    ) -> Result<BytecodeEffectThunk> {
        let mut locally_bound = HashSet::new();
        let mut free_vars = HashSet::new();
        validate_and_find_captures_in_stmts(stmts, &mut locally_bound, &mut free_vars)?;

        let mut sorted_vars: Vec<String> = free_vars.into_iter().collect();
        sorted_vars.sort();

        let mut captures = Vec::new();
        for var_name in sorted_vars {
            if self.lookup_effect_thunk(&var_name).is_some() {
                return Err(LustError::CompileError(format!(
                    "Effect thunk '{var_name}' cannot be captured in nested effect"
                )));
            }
            if let Ok(src_reg) = self.resolve_local(&var_name) {
                let loc_ty = self
                    .functions
                    .get(self.current_function)
                    .and_then(|f| f.register_types.get(&src_reg));
                let is_int = match loc_ty {
                    Some(TypeKind::Int) => true,
                    Some(TypeKind::Named(s)) if s == "int" || s == "i64" => true,
                    _ => false,
                };
                if !is_int {
                    return Err(LustError::CompileError(format!(
                        "Unsupported capture type for '{var_name}': only read-only integers are supported, found {loc_ty:?}"
                    )));
                }

                let snapshot_reg = self.next_local_slot();
                if let Some(scope) = self.scopes.last_mut() {
                    scope.locals.insert(
                        format!("(__thunk_cap_{}_{})", name, var_name),
                        (snapshot_reg, false),
                    );
                }
                self.register_type(snapshot_reg, TypeKind::Int);
                self.emit(Instruction::Move(snapshot_reg, src_reg), 0);
                captures.push((var_name, snapshot_reg));
            }
        }

        Ok(BytecodeEffectThunk {
            stmts: stmts.to_vec(),
            captures,
        })
    }

    pub(super) fn compile_effect_run_sync(
        &mut self,
        thunk: &BytecodeEffectThunk,
    ) -> Result<Register> {
        let result_reg = self.next_local_slot();
        if let Some(scope) = self.scopes.last_mut() {
            scope
                .locals
                .insert(format!("(thunk_res_{})", result_reg), (result_reg, false));
        }
        self.register_type(result_reg, TypeKind::Int);
        self.emit(Instruction::LoadNil(result_reg), 0);
        let saved_thunk_return = self.thunk_return_target.take();
        self.thunk_return_target = Some((result_reg, Vec::new()));

        self.begin_thunk_scope(&thunk.captures);
        for stmt in &thunk.stmts {
            self.compile_stmt(stmt)?;
            self.reset_temp_registers();
        }
        self.end_scope()?;

        let (_, return_jumps) = self.thunk_return_target.take().unwrap();
        self.thunk_return_target = saved_thunk_return;

        let end_pos = self.current_chunk().instructions.len();
        for jump in return_jumps {
            self.current_chunk_mut().patch_jump(jump, end_pos);
        }

        self.reset_temp_registers();
        Ok(result_reg)
    }
}

fn validate_and_find_captures_in_expr(
    expr: &Expr,
    locally_bound: &HashSet<String>,
    free_vars: &mut HashSet<String>,
) -> Result<()> {
    match &expr.kind {
        ExprKind::Literal(_) => Ok(()),
        ExprKind::Identifier(name) => {
            if !locally_bound.contains(name) {
                free_vars.insert(name.clone());
            }
            Ok(())
        }
        ExprKind::Binary { left, right, .. } => {
            validate_and_find_captures_in_expr(left, locally_bound, free_vars)?;
            validate_and_find_captures_in_expr(right, locally_bound, free_vars)?;
            Ok(())
        }
        ExprKind::Unary { operand, .. } => {
            validate_and_find_captures_in_expr(operand, locally_bound, free_vars)
        }
        ExprKind::Call { callee, args, .. } => {
            validate_and_find_captures_in_expr(callee, locally_bound, free_vars)?;
            for a in args {
                validate_and_find_captures_in_expr(a, locally_bound, free_vars)?;
            }
            Ok(())
        }
        ExprKind::MethodCall { receiver, args, .. } => {
            validate_and_find_captures_in_expr(receiver, locally_bound, free_vars)?;
            for a in args {
                validate_and_find_captures_in_expr(a, locally_bound, free_vars)?;
            }
            Ok(())
        }
        ExprKind::FieldAccess { object, .. } => {
            validate_and_find_captures_in_expr(object, locally_bound, free_vars)
        }
        ExprKind::Index { object, index } => {
            validate_and_find_captures_in_expr(object, locally_bound, free_vars)?;
            validate_and_find_captures_in_expr(index, locally_bound, free_vars)?;
            Ok(())
        }
        ExprKind::Array(elems) | ExprKind::Tuple(elems) => {
            for e in elems {
                validate_and_find_captures_in_expr(e, locally_bound, free_vars)?;
            }
            Ok(())
        }
        ExprKind::Map(pairs) => {
            for (k, v) in pairs {
                validate_and_find_captures_in_expr(k, locally_bound, free_vars)?;
                validate_and_find_captures_in_expr(v, locally_bound, free_vars)?;
            }
            Ok(())
        }
        ExprKind::StructLiteral { fields, .. } => {
            for f in fields {
                validate_and_find_captures_in_expr(&f.value, locally_bound, free_vars)?;
            }
            Ok(())
        }
        ExprKind::EnumConstructor { args, .. } => {
            for a in args {
                validate_and_find_captures_in_expr(a, locally_bound, free_vars)?;
            }
            Ok(())
        }
        ExprKind::Lambda { .. } => Err(LustError::CompileError(
            "Lambdas are not supported inside effect thunk".into(),
        )),
        ExprKind::Paren(inner)
        | ExprKind::Cast { expr: inner, .. }
        | ExprKind::TypeCheck { expr: inner, .. } => {
            validate_and_find_captures_in_expr(inner, locally_bound, free_vars)
        }
        ExprKind::IsPattern { expr: inner, .. } => {
            validate_and_find_captures_in_expr(inner, locally_bound, free_vars)
        }
        ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            validate_and_find_captures_in_expr(condition, locally_bound, free_vars)?;
            validate_and_find_captures_in_expr(then_branch, locally_bound, free_vars)?;
            if let Some(eb) = else_branch {
                validate_and_find_captures_in_expr(eb, locally_bound, free_vars)?;
            }
            Ok(())
        }
        ExprKind::Block(stmts) => {
            let mut inner_bound = locally_bound.clone();
            validate_and_find_captures_in_stmts(stmts, &mut inner_bound, free_vars)
        }
        ExprKind::EffectBlock(_) => Err(LustError::CompileError(
            "Nested effect blocks are not supported".into(),
        )),
        ExprKind::Return(exprs) => {
            for e in exprs {
                validate_and_find_captures_in_expr(e, locally_bound, free_vars)?;
            }
            Ok(())
        }
        ExprKind::Range { start, end, .. } => {
            validate_and_find_captures_in_expr(start, locally_bound, free_vars)?;
            validate_and_find_captures_in_expr(end, locally_bound, free_vars)?;
            Ok(())
        }
    }
}

fn validate_and_find_captures_in_stmts(
    stmts: &[Stmt],
    locally_bound: &mut HashSet<String>,
    free_vars: &mut HashSet<String>,
) -> Result<()> {
    for stmt in stmts {
        match &stmt.kind {
            StmtKind::Local {
                bindings,
                initializer,
            } => {
                if let Some(inits) = initializer {
                    for i in inits {
                        validate_and_find_captures_in_expr(i, locally_bound, free_vars)?;
                    }
                }
                for b in bindings {
                    locally_bound.insert(b.name.clone());
                }
            }
            StmtKind::Assign { targets, values } => {
                for v in values {
                    validate_and_find_captures_in_expr(v, locally_bound, free_vars)?;
                }
                for t in targets {
                    if let ExprKind::Identifier(name) = &t.kind {
                        if !locally_bound.contains(name) {
                            return Err(LustError::CompileError(format!(
                                "Write to captured variable '{name}' is not supported in effect thunk"
                            )));
                        }
                    } else {
                        validate_and_find_captures_in_expr(t, locally_bound, free_vars)?;
                    }
                }
            }
            StmtKind::CompoundAssign { target, op: _, value } => {
                validate_and_find_captures_in_expr(value, locally_bound, free_vars)?;
                if let ExprKind::Identifier(name) = &target.kind {
                    if !locally_bound.contains(name) {
                        return Err(LustError::CompileError(format!(
                            "Write to captured variable '{name}' is not supported in effect thunk"
                        )));
                    }
                } else {
                    validate_and_find_captures_in_expr(target, locally_bound, free_vars)?;
                }
            }
            StmtKind::Expr(e) => {
                validate_and_find_captures_in_expr(e, locally_bound, free_vars)?;
            }
            StmtKind::If {
                condition,
                then_block,
                elseif_branches,
                else_block,
            } => {
                validate_and_find_captures_in_expr(condition, locally_bound, free_vars)?;
                let mut then_bound = locally_bound.clone();
                validate_and_find_captures_in_stmts(then_block, &mut then_bound, free_vars)?;
                for (c, body) in elseif_branches {
                    validate_and_find_captures_in_expr(c, locally_bound, free_vars)?;
                    let mut elif_bound = locally_bound.clone();
                    validate_and_find_captures_in_stmts(body, &mut elif_bound, free_vars)?;
                }
                if let Some(eb) = else_block {
                    let mut else_bound = locally_bound.clone();
                    validate_and_find_captures_in_stmts(eb, &mut else_bound, free_vars)?;
                }
            }
            StmtKind::While { condition, body } => {
                validate_and_find_captures_in_expr(condition, locally_bound, free_vars)?;
                let mut body_bound = locally_bound.clone();
                validate_and_find_captures_in_stmts(body, &mut body_bound, free_vars)?;
            }
            StmtKind::ForNumeric {
                variable,
                start,
                end,
                step,
                body,
            } => {
                validate_and_find_captures_in_expr(start, locally_bound, free_vars)?;
                validate_and_find_captures_in_expr(end, locally_bound, free_vars)?;
                if let Some(s) = step {
                    validate_and_find_captures_in_expr(s, locally_bound, free_vars)?;
                }
                let mut body_bound = locally_bound.clone();
                body_bound.insert(variable.clone());
                validate_and_find_captures_in_stmts(body, &mut body_bound, free_vars)?;
            }
            StmtKind::ForIn {
                variables,
                iterator,
                body,
            } => {
                validate_and_find_captures_in_expr(iterator, locally_bound, free_vars)?;
                let mut body_bound = locally_bound.clone();
                for v in variables {
                    body_bound.insert(v.clone());
                }
                validate_and_find_captures_in_stmts(body, &mut body_bound, free_vars)?;
            }
            StmtKind::Return(exprs) => {
                if exprs.is_empty() {
                    return Err(LustError::CompileError(
                        "Effect thunk return must return an integer value".into(),
                    ));
                }
                for e in exprs {
                    validate_and_find_captures_in_expr(e, locally_bound, free_vars)?;
                }
            }
            StmtKind::Break | StmtKind::Continue => {}
            StmtKind::Block(inner) | StmtKind::Scoped(inner) => {
                let mut inner_bound = locally_bound.clone();
                validate_and_find_captures_in_stmts(inner, &mut inner_bound, free_vars)?;
            }
            StmtKind::Defer(_) | StmtKind::ErrDefer(_) => {
                return Err(LustError::CompileError(
                    "Cleanups (defer/errdefer) are not supported inside effect thunk".into(),
                ));
            }
        }
    }
    Ok(())
}
