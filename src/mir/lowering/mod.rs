use crate::ast::{
    BinaryOp, Expr, ExprKind, FunctionDef, Literal, Stmt, StmtKind, Type, TypeKind, UnaryOp,
};
use crate::mir::ir::*;
use alloc::{boxed::Box, format, string::String, vec, vec::Vec};
use hashbrown::HashMap;

pub struct AstLowering<'a> {
    current_fn: &'a mut MirFunction,
    current_block: BlockId,
    scopes: Vec<HashMap<String, LocalId>>,
    loop_stack: Vec<(BlockId, BlockId)>, // (header/continue_block, exit_block)
}

impl<'a> AstLowering<'a> {
    pub fn new(func: &'a mut MirFunction) -> Self {
        let entry = func.entry;
        let mut scope = HashMap::new();
        // Register parameters in the root scope
        for (idx, p_id) in func.params.iter().enumerate() {
            let p_name = func.locals[p_id.0]
                .name
                .clone()
                .unwrap_or_else(|| format!("arg{idx}"));
            scope.insert(p_name, *p_id);
        }

        Self {
            current_fn: func,
            current_block: entry,
            scopes: vec![scope],
            loop_stack: Vec::new(),
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn insert_var(&mut self, name: String, local_id: LocalId) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, local_id);
        }
    }

    fn lookup_var(&self, name: &str) -> Option<LocalId> {
        for scope in self.scopes.iter().rev() {
            if let Some(id) = scope.get(name) {
                return Some(*id);
            }
        }
        None
    }

    pub fn lower_function(func_def: &FunctionDef) -> MirFunction {
        let sig = FunctionSignature {
            name: func_def.name.clone(),
            param_types: func_def
                .params
                .iter()
                .map(|p| mir_type_from_ast(&p.ty))
                .collect(),
            return_type: func_def
                .return_type
                .as_ref()
                .map(mir_type_from_ast)
                .unwrap_or(MirType::Unit),
            is_public: true,
        };

        // Create MirFunction with default entry block
        let mut mir_fn = MirFunction::new(sig, crate::ast::Span::dummy());

        // Update parameter names if defined in func_def
        for (i, p) in func_def.params.iter().enumerate() {
            let p_id = mir_fn.params[i];
            mir_fn.locals[p_id.0].name = Some(p.name.clone());
        }

        {
            let mut lowering = AstLowering::new(&mut mir_fn);
            for stmt in &func_def.body {
                lowering.lower_statement(stmt);
            }

            // Ensure current block has a terminator
            let cur_block = lowering.current_block;
            if lowering.current_fn.blocks[cur_block.0].terminator.is_none() {
                lowering.current_fn.blocks[cur_block.0].terminator =
                    Some(Terminator::Return(None, crate::ast::Span::dummy()));
            }
        }

        mir_fn
    }

    pub fn lower_script(name: &str, stmts: &[Stmt]) -> MirFunction {
        let sig = FunctionSignature {
            name: name.into(),
            param_types: vec![],
            return_type: MirType::Unit,
            is_public: true,
        };

        let mut mir_fn = MirFunction::new(sig, crate::ast::Span::dummy());
        {
            let mut lowering = AstLowering::new(&mut mir_fn);
            for stmt in stmts {
                lowering.lower_statement(stmt);
            }

            let cur_block = lowering.current_block;
            if lowering.current_fn.blocks[cur_block.0].terminator.is_none() {
                lowering.current_fn.blocks[cur_block.0].terminator =
                    Some(Terminator::Return(None, crate::ast::Span::dummy()));
            }
        }

        mir_fn
    }

    pub fn lower_statement(&mut self, stmt: &Stmt) {
        if self.current_fn.blocks[self.current_block.0].terminator.is_some() {
            let dead_bb = self.current_fn.new_block();
            self.current_fn.blocks[dead_bb.0].terminator = Some(Terminator::Unreachable(stmt.span));
            self.current_block = dead_bb;
        }

        match &stmt.kind {
            StmtKind::Local { bindings, initializer } => {
                let init_operands = if let Some(inits) = initializer {
                    inits.iter().map(|expr| self.lower_expr(expr)).collect()
                } else {
                    Vec::new()
                };

                for (idx, b) in bindings.iter().enumerate() {
                    let init_op = init_operands.get(idx).cloned().unwrap_or(Operand::Constant(MirConstant::Int(0)));
                    let inferred_ty = match &init_op {
                        Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => {
                            self.current_fn.locals[loc.0].ty.clone()
                        }
                        Operand::Constant(c) => match c {
                            MirConstant::Int(_) => MirType::Int,
                            MirConstant::Float(_) => MirType::Float,
                            MirConstant::Bool(_) => MirType::Bool,
                            MirConstant::String(_) => MirType::String,
                            MirConstant::Unit => MirType::Unit,
                        },
                        _ => MirType::Int,
                    };
                    let ty = b
                        .type_annotation
                        .as_ref()
                        .map(mir_type_from_ast)
                        .unwrap_or(inferred_ty);
                    let local_id = self.current_fn.add_local(
                        Some(b.name.clone()),
                        ty,
                        true,
                        b.span,
                    );
                    self.insert_var(b.name.clone(), local_id);

                    self.emit_stmt(Statement::StorageLive(local_id, b.span));
                    self.emit_stmt(Statement::Assign(Place::Local(local_id), Rvalue::Use(init_op), b.span));
                }
            }
            StmtKind::Assign { targets, values } => {
                for (target, val) in targets.iter().zip(values.iter()) {
                    let rval_op = self.lower_expr(val);
                    if let Some(place) = self.lower_place(target) {
                        self.emit_stmt(Statement::Assign(place, Rvalue::Use(rval_op), stmt.span));
                    }
                }
            }
            StmtKind::CompoundAssign { target, op, value } => {
                let mir_op = mir_binop_from_ast(*op);
                if let Some(place) = self.lower_place(target) {
                    let lhs_op = Operand::Copy(place.clone());
                    let rhs_op = self.lower_expr(value);
                    self.emit_stmt(Statement::Assign(
                        place,
                        Rvalue::BinaryOp(mir_op, lhs_op, rhs_op),
                        stmt.span,
                    ));
                }
            }
            StmtKind::Expr(expr) => {
                self.lower_expr(expr);
            }
            StmtKind::If {
                condition,
                then_block,
                elseif_branches,
                else_block,
            } => {
                self.lower_if(condition, then_block, elseif_branches, else_block.as_deref(), stmt.span);
            }
            StmtKind::While { condition, body } => {
                let header = self.current_fn.new_block();
                let body_block = self.current_fn.new_block();
                let exit_block = self.current_fn.new_block();

                // Branch from current block to header
                self.terminate_block(Terminator::Goto(header, stmt.span));

                // Header: evaluate condition
                self.current_block = header;
                self.emit_stmt(Statement::ResourceCheck(ResourceKind::Gas, 1, stmt.span));
                let cond_op = self.lower_expr(condition);
                self.terminate_block(Terminator::Cond {
                    cond: cond_op,
                    then_block: body_block,
                    else_block: exit_block,
                    span: stmt.span,
                });

                // Body block
                self.current_block = body_block;
                self.loop_stack.push((header, exit_block));
                self.push_scope();
                for s in body {
                    self.lower_statement(s);
                }
                self.pop_scope();
                self.loop_stack.pop();

                // Loop back to header
                if self.current_fn.blocks[self.current_block.0].terminator.is_none() {
                    self.terminate_block(Terminator::Goto(header, stmt.span));
                }

                self.current_block = exit_block;
            }
            StmtKind::Return(exprs) => {
                let ret_op = if let Some(first) = exprs.first() {
                    Some(self.lower_expr(first))
                } else {
                    None
                };
                self.terminate_block(Terminator::Return(ret_op, stmt.span));
            }
            StmtKind::ForNumeric {
                variable,
                start,
                end,
                step,
                body,
            } => {
                let start_op = self.lower_expr(start);
                let end_op = self.lower_expr(end);
                let step_op = if let Some(step_expr) = step {
                    self.lower_expr(step_expr)
                } else {
                    Operand::Constant(MirConstant::Int(1))
                };

                let static_step = match step {
                    Some(s) => const_int_from_expr(s),
                    None => Some(1),
                };

                let end_local = self.current_fn.add_local(None, MirType::Int, false, stmt.span);
                self.emit_stmt(Statement::StorageLive(end_local, stmt.span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(end_local),
                    Rvalue::Use(end_op),
                    stmt.span,
                ));

                let step_local = self.current_fn.add_local(None, MirType::Int, false, stmt.span);
                self.emit_stmt(Statement::StorageLive(step_local, stmt.span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(step_local),
                    Rvalue::Use(step_op),
                    stmt.span,
                ));

                let var_local = self.current_fn.add_local(
                    Some(variable.clone()),
                    MirType::Int,
                    true,
                    stmt.span,
                );
                self.emit_stmt(Statement::StorageLive(var_local, stmt.span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(var_local),
                    Rvalue::Use(start_op),
                    stmt.span,
                ));

                let header_bb = self.current_fn.new_block();
                let body_bb = self.current_fn.new_block();
                let step_bb = self.current_fn.new_block();
                let exit_bb = self.current_fn.new_block();

                self.terminate_block(Terminator::Goto(header_bb, stmt.span));

                // Header block
                self.current_block = header_bb;
                self.emit_stmt(Statement::ResourceCheck(ResourceKind::Gas, 1, stmt.span));
                let cond_temp = self.current_fn.add_local(None, MirType::Bool, true, stmt.span);
                self.emit_stmt(Statement::StorageLive(cond_temp, stmt.span));

                let cond_rval = match static_step {
                    Some(val) if val < 0 => Rvalue::BinaryOp(
                        MirBinaryOp::GtEq,
                        Operand::Copy(Place::Local(var_local)),
                        Operand::Copy(Place::Local(end_local)),
                    ),
                    Some(_) => Rvalue::BinaryOp(
                        MirBinaryOp::LtEq,
                        Operand::Copy(Place::Local(var_local)),
                        Operand::Copy(Place::Local(end_local)),
                    ),
                    None => {
                        let is_neg_temp = self.current_fn.add_local(None, MirType::Bool, true, stmt.span);
                        self.emit_stmt(Statement::StorageLive(is_neg_temp, stmt.span));
                        self.emit_stmt(Statement::Assign(
                            Place::Local(is_neg_temp),
                            Rvalue::BinaryOp(
                                MirBinaryOp::Lt,
                                Operand::Copy(Place::Local(step_local)),
                                Operand::Constant(MirConstant::Int(0)),
                            ),
                            stmt.span,
                        ));
                        let ge_temp = self.current_fn.add_local(None, MirType::Bool, true, stmt.span);
                        self.emit_stmt(Statement::StorageLive(ge_temp, stmt.span));
                        self.emit_stmt(Statement::Assign(
                            Place::Local(ge_temp),
                            Rvalue::BinaryOp(
                                MirBinaryOp::GtEq,
                                Operand::Copy(Place::Local(var_local)),
                                Operand::Copy(Place::Local(end_local)),
                            ),
                            stmt.span,
                        ));
                        let le_temp = self.current_fn.add_local(None, MirType::Bool, true, stmt.span);
                        self.emit_stmt(Statement::StorageLive(le_temp, stmt.span));
                        self.emit_stmt(Statement::Assign(
                            Place::Local(le_temp),
                            Rvalue::BinaryOp(
                                MirBinaryOp::LtEq,
                                Operand::Copy(Place::Local(var_local)),
                                Operand::Copy(Place::Local(end_local)),
                            ),
                            stmt.span,
                        ));
                        let neg_and_ge = self.current_fn.add_local(None, MirType::Bool, true, stmt.span);
                        self.emit_stmt(Statement::StorageLive(neg_and_ge, stmt.span));
                        self.emit_stmt(Statement::Assign(
                            Place::Local(neg_and_ge),
                            Rvalue::BinaryOp(
                                MirBinaryOp::And,
                                Operand::Copy(Place::Local(is_neg_temp)),
                                Operand::Copy(Place::Local(ge_temp)),
                            ),
                            stmt.span,
                        ));
                        let not_neg = self.current_fn.add_local(None, MirType::Bool, true, stmt.span);
                        self.emit_stmt(Statement::StorageLive(not_neg, stmt.span));
                        self.emit_stmt(Statement::Assign(
                            Place::Local(not_neg),
                            Rvalue::UnaryOp(MirUnaryOp::Not, Operand::Copy(Place::Local(is_neg_temp))),
                            stmt.span,
                        ));
                        let pos_and_le = self.current_fn.add_local(None, MirType::Bool, true, stmt.span);
                        self.emit_stmt(Statement::StorageLive(pos_and_le, stmt.span));
                        self.emit_stmt(Statement::Assign(
                            Place::Local(pos_and_le),
                            Rvalue::BinaryOp(
                                MirBinaryOp::And,
                                Operand::Copy(Place::Local(not_neg)),
                                Operand::Copy(Place::Local(le_temp)),
                            ),
                            stmt.span,
                        ));
                        Rvalue::BinaryOp(
                            MirBinaryOp::Or,
                            Operand::Copy(Place::Local(neg_and_ge)),
                            Operand::Copy(Place::Local(pos_and_le)),
                        )
                    }
                };

                self.emit_stmt(Statement::Assign(Place::Local(cond_temp), cond_rval, stmt.span));
                self.terminate_block(Terminator::Cond {
                    cond: Operand::Move(Place::Local(cond_temp)),
                    then_block: body_bb,
                    else_block: exit_bb,
                    span: stmt.span,
                });

                // Body block
                self.current_block = body_bb;
                self.push_scope();
                self.insert_var(variable.clone(), var_local);
                self.loop_stack.push((step_bb, exit_bb));
                for s in body {
                    self.lower_statement(s);
                }
                self.loop_stack.pop();
                self.pop_scope();
                if self.current_fn.blocks[self.current_block.0].terminator.is_none() {
                    self.terminate_block(Terminator::Goto(step_bb, stmt.span));
                }

                // Step block
                self.current_block = step_bb;
                let next_var_temp = self.current_fn.add_local(None, MirType::Int, true, stmt.span);
                self.emit_stmt(Statement::StorageLive(next_var_temp, stmt.span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(next_var_temp),
                    Rvalue::BinaryOp(
                        MirBinaryOp::Add,
                        Operand::Copy(Place::Local(var_local)),
                        Operand::Copy(Place::Local(step_local)),
                    ),
                    stmt.span,
                ));
                self.emit_stmt(Statement::Assign(
                    Place::Local(var_local),
                    Rvalue::Use(Operand::Move(Place::Local(next_var_temp))),
                    stmt.span,
                ));
                self.terminate_block(Terminator::Goto(header_bb, stmt.span));

                self.current_block = exit_bb;
            }
            StmtKind::Break => {
                if let Some((_, exit_bb)) = self.loop_stack.last().cloned() {
                    self.terminate_block(Terminator::Goto(exit_bb, stmt.span));
                }
            }
            StmtKind::Continue => {
                if let Some((cont_bb, _)) = self.loop_stack.last().cloned() {
                    self.terminate_block(Terminator::Goto(cont_bb, stmt.span));
                }
            }
            StmtKind::Block(stmts) => {
                self.push_scope();
                for s in stmts {
                    self.lower_statement(s);
                }
                self.pop_scope();
            }
            _ => {
                // Unsupported statement logged as debug/noop for basic scalar phase
            }
        }
    }

    fn lower_if(
        &mut self,
        condition: &Expr,
        then_block: &[Stmt],
        elseif_branches: &[(Expr, Vec<Stmt>)],
        else_block: Option<&[Stmt]>,
        span: crate::ast::Span,
    ) {
        let cond_op = self.lower_expr(condition);
        let then_bb = self.current_fn.new_block();
        let merge_bb = self.current_fn.new_block();

        let next_cond_bb = if !elseif_branches.is_empty() || else_block.is_some() {
            self.current_fn.new_block()
        } else {
            merge_bb
        };

        self.terminate_block(Terminator::Cond {
            cond: cond_op,
            then_block: then_bb,
            else_block: next_cond_bb,
            span,
        });

        // Lower Then Block
        self.current_block = then_bb;
        self.push_scope();
        for s in then_block {
            self.lower_statement(s);
        }
        self.pop_scope();
        if self.current_fn.blocks[self.current_block.0].terminator.is_none() {
            self.terminate_block(Terminator::Goto(merge_bb, span));
        }

        // Lower ElseIf branches if present
        let mut cur_else_bb = next_cond_bb;
        for (i, (elif_cond, elif_body)) in elseif_branches.iter().enumerate() {
            if cur_else_bb == merge_bb {
                break;
            }
            self.current_block = cur_else_bb;
            let elif_op = self.lower_expr(elif_cond);
            let elif_then = self.current_fn.new_block();
            let next_elif = if i + 1 < elseif_branches.len() || else_block.is_some() {
                self.current_fn.new_block()
            } else {
                merge_bb
            };

            self.terminate_block(Terminator::Cond {
                cond: elif_op,
                then_block: elif_then,
                else_block: next_elif,
                span,
            });

            self.current_block = elif_then;
            self.push_scope();
            for s in elif_body {
                self.lower_statement(s);
            }
            self.pop_scope();
            if self.current_fn.blocks[self.current_block.0].terminator.is_none() {
                self.terminate_block(Terminator::Goto(merge_bb, span));
            }

            cur_else_bb = next_elif;
        }

        // Lower Else block if present
        if let Some(eb) = else_block {
            if cur_else_bb != merge_bb {
                self.current_block = cur_else_bb;
                self.push_scope();
                for s in eb {
                    self.lower_statement(s);
                }
                self.pop_scope();
                if self.current_fn.blocks[self.current_block.0].terminator.is_none() {
                    self.terminate_block(Terminator::Goto(merge_bb, span));
                }
            }
        }

        self.current_block = merge_bb;
    }

    pub fn lower_expr(&mut self, expr: &Expr) -> Operand {
        match &expr.kind {
            ExprKind::Literal(lit) => match lit {
                Literal::Integer(i) => Operand::Constant(MirConstant::Int(*i)),
                Literal::Float(f) => Operand::Constant(MirConstant::Float(*f)),
                Literal::Bool(b) => Operand::Constant(MirConstant::Bool(*b)),
                Literal::String(s) => Operand::Constant(MirConstant::String(s.clone())),
            },
            ExprKind::Identifier(name) => {
                if let Some(loc_id) = self.lookup_var(name) {
                    Operand::Copy(Place::Local(loc_id))
                } else {
                    Operand::Constant(MirConstant::Int(0))
                }
            }
            ExprKind::Paren(inner) => self.lower_expr(inner),
            ExprKind::Binary { left, op, right } => {
                if *op == BinaryOp::And {
                    let res_local = self.current_fn.add_local(None, MirType::Bool, true, expr.span);
                    self.emit_stmt(Statement::StorageLive(res_local, expr.span));
                    let lhs_op = self.lower_expr(left);
                    self.emit_stmt(Statement::Assign(
                        Place::Local(res_local),
                        Rvalue::Use(lhs_op.clone()),
                        expr.span,
                    ));

                    let rhs_bb = self.current_fn.new_block();
                    let merge_bb = self.current_fn.new_block();

                    self.terminate_block(Terminator::Cond {
                        cond: lhs_op,
                        then_block: rhs_bb,
                        else_block: merge_bb,
                        span: expr.span,
                    });

                    self.current_block = rhs_bb;
                    let rhs_op = self.lower_expr(right);
                    self.emit_stmt(Statement::Assign(
                        Place::Local(res_local),
                        Rvalue::Use(rhs_op),
                        expr.span,
                    ));
                    if self.current_fn.blocks[self.current_block.0].terminator.is_none() {
                        self.terminate_block(Terminator::Goto(merge_bb, expr.span));
                    }

                    self.current_block = merge_bb;
                    return Operand::Move(Place::Local(res_local));
                }

                if *op == BinaryOp::Or {
                    let res_local = self.current_fn.add_local(None, MirType::Bool, true, expr.span);
                    self.emit_stmt(Statement::StorageLive(res_local, expr.span));
                    let lhs_op = self.lower_expr(left);
                    self.emit_stmt(Statement::Assign(
                        Place::Local(res_local),
                        Rvalue::Use(lhs_op.clone()),
                        expr.span,
                    ));

                    let rhs_bb = self.current_fn.new_block();
                    let merge_bb = self.current_fn.new_block();

                    self.terminate_block(Terminator::Cond {
                        cond: lhs_op,
                        then_block: merge_bb,
                        else_block: rhs_bb,
                        span: expr.span,
                    });

                    self.current_block = rhs_bb;
                    let rhs_op = self.lower_expr(right);
                    self.emit_stmt(Statement::Assign(
                        Place::Local(res_local),
                        Rvalue::Use(rhs_op),
                        expr.span,
                    ));
                    if self.current_fn.blocks[self.current_block.0].terminator.is_none() {
                        self.terminate_block(Terminator::Goto(merge_bb, expr.span));
                    }

                    self.current_block = merge_bb;
                    return Operand::Move(Place::Local(res_local));
                }

                let lhs = self.lower_expr(left);
                let rhs = self.lower_expr(right);
                let mir_op = mir_binop_from_ast(*op);
                let res_ty = match mir_op {
                    MirBinaryOp::Eq | MirBinaryOp::NotEq | MirBinaryOp::Lt | MirBinaryOp::LtEq | MirBinaryOp::Gt | MirBinaryOp::GtEq => MirType::Bool,
                    _ => MirType::Int,
                };
                let temp = self.current_fn.add_local(None, res_ty, true, expr.span);
                self.emit_stmt(Statement::StorageLive(temp, expr.span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(temp),
                    Rvalue::BinaryOp(mir_op, lhs, rhs),
                    expr.span,
                ));
                Operand::Move(Place::Local(temp))
            }
            ExprKind::Unary { op, operand } => {
                let inner = self.lower_expr(operand);
                let mir_op = match op {
                    UnaryOp::Neg => MirUnaryOp::Neg,
                    UnaryOp::Not => MirUnaryOp::Not,
                };
                let temp = self.current_fn.add_local(None, MirType::Int, true, expr.span);
                self.emit_stmt(Statement::StorageLive(temp, expr.span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(temp),
                    Rvalue::UnaryOp(mir_op, inner),
                    expr.span,
                ));
                Operand::Move(Place::Local(temp))
            }
            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let res_local = self.current_fn.add_local(None, MirType::Int, true, expr.span);
                self.emit_stmt(Statement::StorageLive(res_local, expr.span));

                let cond_op = self.lower_expr(condition);
                let then_bb = self.current_fn.new_block();
                let else_bb = self.current_fn.new_block();
                let merge_bb = self.current_fn.new_block();

                self.terminate_block(Terminator::Cond {
                    cond: cond_op,
                    then_block: then_bb,
                    else_block: else_bb,
                    span: expr.span,
                });

                self.current_block = then_bb;
                let then_op = self.lower_expr(then_branch);
                self.emit_stmt(Statement::Assign(
                    Place::Local(res_local),
                    Rvalue::Use(then_op),
                    expr.span,
                ));
                if self.current_fn.blocks[self.current_block.0].terminator.is_none() {
                    self.terminate_block(Terminator::Goto(merge_bb, expr.span));
                }

                self.current_block = else_bb;
                if let Some(eb) = else_branch {
                    let else_op = self.lower_expr(eb);
                    self.emit_stmt(Statement::Assign(
                        Place::Local(res_local),
                        Rvalue::Use(else_op),
                        expr.span,
                    ));
                }
                if self.current_fn.blocks[self.current_block.0].terminator.is_none() {
                    self.terminate_block(Terminator::Goto(merge_bb, expr.span));
                }

                self.current_block = merge_bb;
                Operand::Move(Place::Local(res_local))
            }
            ExprKind::IsPattern { expr: scrutinee, pattern } => {
                let scrutinee_op = self.lower_expr(scrutinee);
                match pattern {
                    crate::ast::Pattern::Wildcard | crate::ast::Pattern::Identifier(_) => {
                        Operand::Constant(MirConstant::Bool(true))
                    }
                    crate::ast::Pattern::Literal(lit) => {
                        let lit_op = match lit {
                            Literal::Integer(i) => Operand::Constant(MirConstant::Int(*i)),
                            Literal::Float(f) => Operand::Constant(MirConstant::Float(*f)),
                            Literal::Bool(b) => Operand::Constant(MirConstant::Bool(*b)),
                            Literal::String(s) => Operand::Constant(MirConstant::String(s.clone())),
                        };
                        let temp = self.current_fn.add_local(None, MirType::Bool, true, expr.span);
                        self.emit_stmt(Statement::StorageLive(temp, expr.span));
                        self.emit_stmt(Statement::Assign(
                            Place::Local(temp),
                            Rvalue::BinaryOp(MirBinaryOp::Eq, scrutinee_op, lit_op),
                            expr.span,
                        ));
                        Operand::Move(Place::Local(temp))
                    }
                    _ => Operand::Constant(MirConstant::Bool(false)),
                }
            }
            ExprKind::Array(elements) => {
                let lowered_elems: Vec<Operand> = elements.iter().map(|e| self.lower_expr(e)).collect();
                let elem_ty = MirType::Int;
                let arr_ty = MirType::Array(Box::new(elem_ty));
                let temp = self.current_fn.add_local(None, arr_ty.clone(), true, expr.span);
                self.emit_stmt(Statement::StorageLive(temp, expr.span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(temp),
                    Rvalue::Aggregate(arr_ty, lowered_elems),
                    expr.span,
                ));
                Operand::Move(Place::Local(temp))
            }
            ExprKind::Index { object, index } => {
                let obj_op = self.lower_expr(object);
                let idx_op = self.lower_expr(index);

                let base_place = match obj_op {
                    Operand::Copy(p) | Operand::Move(p) => p,
                    Operand::Constant(_) => {
                        let temp = self.current_fn.add_local(None, MirType::Array(Box::new(MirType::Int)), true, expr.span);
                        self.emit_stmt(Statement::StorageLive(temp, expr.span));
                        self.emit_stmt(Statement::Assign(Place::Local(temp), Rvalue::Use(obj_op), expr.span));
                        Place::Local(temp)
                    }
                };
                let idx_local = match idx_op {
                    Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => loc,
                    _ => {
                        let temp = self.current_fn.add_local(None, MirType::Int, true, expr.span);
                        self.emit_stmt(Statement::StorageLive(temp, expr.span));
                        self.emit_stmt(Statement::Assign(Place::Local(temp), Rvalue::Use(idx_op), expr.span));
                        temp
                    }
                };
                let elem_place = Place::Index(Box::new(base_place), idx_local);
                let val_temp = self.current_fn.add_local(None, MirType::Int, true, expr.span);
                self.emit_stmt(Statement::StorageLive(val_temp, expr.span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(val_temp),
                    Rvalue::Use(Operand::Copy(elem_place)),
                    expr.span,
                ));
                Operand::Move(Place::Local(val_temp))
            }
            ExprKind::MethodCall { receiver, method, args, .. } if method == "len" && args.is_empty() => {
                let rec_op = self.lower_expr(receiver);
                let temp = self.current_fn.add_local(None, MirType::Int, true, expr.span);
                self.emit_stmt(Statement::StorageLive(temp, expr.span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(temp),
                    Rvalue::UnaryOp(MirUnaryOp::Len, rec_op),
                    expr.span,
                ));
                Operand::Move(Place::Local(temp))
            }
            ExprKind::MethodCall { receiver, method, args, .. } if method == "unwrap" && args.is_empty() => {
                self.lower_expr(receiver)
            }
            ExprKind::Call { callee, args, .. }
                if args.is_empty()
                    && matches!(&callee.kind, ExprKind::FieldAccess { field, .. } if field == "len") =>
            {
                if let ExprKind::FieldAccess { object, .. } = &callee.kind {
                    let rec_op = self.lower_expr(object);
                    let temp = self.current_fn.add_local(None, MirType::Int, true, expr.span);
                    self.emit_stmt(Statement::StorageLive(temp, expr.span));
                    self.emit_stmt(Statement::Assign(
                        Place::Local(temp),
                        Rvalue::UnaryOp(MirUnaryOp::Len, rec_op),
                        expr.span,
                    ));
                    Operand::Move(Place::Local(temp))
                } else {
                    Operand::Constant(MirConstant::Int(0))
                }
            }
            ExprKind::Call { callee, args, .. }
                if args.len() == 1
                    && matches!(&callee.kind, ExprKind::Identifier(name) if name == "len") =>
            {
                let rec_op = self.lower_expr(&args[0]);
                let temp = self.current_fn.add_local(None, MirType::Int, true, expr.span);
                self.emit_stmt(Statement::StorageLive(temp, expr.span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(temp),
                    Rvalue::UnaryOp(MirUnaryOp::Len, rec_op),
                    expr.span,
                ));
                Operand::Move(Place::Local(temp))
            }
            ExprKind::FieldAccess { object, field } if field == "len" => {
                let rec_op = self.lower_expr(object);
                let temp = self.current_fn.add_local(None, MirType::Int, true, expr.span);
                self.emit_stmt(Statement::StorageLive(temp, expr.span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(temp),
                    Rvalue::UnaryOp(MirUnaryOp::Len, rec_op),
                    expr.span,
                ));
                Operand::Move(Place::Local(temp))
            }
            ExprKind::Call { callee, args, .. } => {
                let callee_name = match &callee.kind {
                    ExprKind::Identifier(name) => name.clone(),
                    _ => "unknown_callee".into(),
                };
                let lowered_args: Vec<Operand> = args.iter().map(|a| self.lower_expr(a)).collect();
                let temp = self.current_fn.add_local(None, MirType::Int, true, expr.span);
                let succ_bb = self.current_fn.new_block();

                self.emit_stmt(Statement::StorageLive(temp, expr.span));
                self.terminate_block(Terminator::Call {
                    callee: callee_name,
                    args: lowered_args,
                    destination: Place::Local(temp),
                    success: succ_bb,
                    error: None,
                    span: expr.span,
                });

                self.current_block = succ_bb;
                Operand::Move(Place::Local(temp))
            }
            _ => Operand::Constant(MirConstant::Unit),
        }
    }

    fn lower_place(&mut self, expr: &Expr) -> Option<Place> {
        match &expr.kind {
            ExprKind::Identifier(name) => self.lookup_var(name).map(Place::Local),
            ExprKind::Paren(inner) => self.lower_place(inner),
            ExprKind::Index { object, index } => {
                let base_place = self.lower_place(object)?;
                let idx_op = self.lower_expr(index);
                let idx_local = match idx_op {
                    Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => loc,
                    _ => {
                        let temp = self.current_fn.add_local(None, MirType::Int, true, expr.span);
                        self.emit_stmt(Statement::StorageLive(temp, expr.span));
                        self.emit_stmt(Statement::Assign(Place::Local(temp), Rvalue::Use(idx_op), expr.span));
                        temp
                    }
                };
                Some(Place::Index(Box::new(base_place), idx_local))
            }
            _ => None,
        }
    }

    fn emit_stmt(&mut self, stmt: Statement) {
        self.current_fn.blocks[self.current_block.0]
            .statements
            .push(stmt);
    }

    fn terminate_block(&mut self, term: Terminator) {
        self.current_fn.blocks[self.current_block.0].terminator = Some(term);
    }
}

fn const_int_from_expr(expr: &Expr) -> Option<i64> {
    match &expr.kind {
        ExprKind::Literal(Literal::Integer(val)) => Some(*val),
        ExprKind::Paren(inner) => const_int_from_expr(inner),
        ExprKind::Unary {
            op: UnaryOp::Neg,
            operand,
        } => const_int_from_expr(operand).map(|v| -v),
        _ => None,
    }
}

pub fn mir_type_from_ast(ty: &Type) -> MirType {
    match &ty.kind {
        TypeKind::Int => MirType::Int,
        TypeKind::Float => MirType::Float,
        TypeKind::Bool => MirType::Bool,
        TypeKind::String => MirType::String,
        TypeKind::Unit => MirType::Unit,
        TypeKind::Named(s) => MirType::Struct(s.clone()),
        TypeKind::Tuple(types) => MirType::Tuple(types.iter().map(mir_type_from_ast).collect()),
        TypeKind::Array(inner) => MirType::Array(Box::new(mir_type_from_ast(inner))),
        TypeKind::Map(k, v) => {
            MirType::Map(Box::new(mir_type_from_ast(k)), Box::new(mir_type_from_ast(v)))
        }
        _ => MirType::Int,
    }
}

pub fn mir_binop_from_ast(op: BinaryOp) -> MirBinaryOp {
    match op {
        BinaryOp::Add => MirBinaryOp::Add,
        BinaryOp::Sub => MirBinaryOp::Sub,
        BinaryOp::Mul => MirBinaryOp::Mul,
        BinaryOp::Div => MirBinaryOp::Div,
        BinaryOp::Mod => MirBinaryOp::Mod,
        BinaryOp::Pow => MirBinaryOp::Pow,
        BinaryOp::Eq => MirBinaryOp::Eq,
        BinaryOp::Ne => MirBinaryOp::NotEq,
        BinaryOp::Lt => MirBinaryOp::Lt,
        BinaryOp::Le => MirBinaryOp::LtEq,
        BinaryOp::Gt => MirBinaryOp::Gt,
        BinaryOp::Ge => MirBinaryOp::GtEq,
        BinaryOp::And => MirBinaryOp::And,
        BinaryOp::Or => MirBinaryOp::Or,
        _ => MirBinaryOp::Add,
    }
}
