use super::*;
use crate::ast::StructLiteralField;
impl Compiler {
    pub(super) fn compile_expr(&mut self, expr: &Expr) -> Result<Register> {
        self.current_line = expr.span.start_line;
        match &expr.kind {
            ExprKind::Literal(lit) => {
                let reg = self.allocate_register();
                match lit {
                    Literal::Integer(i) => {
                        let const_idx = self.add_int_const(*i);
                        self.emit(Instruction::LoadConst(reg, const_idx), 0);
                    }

                    Literal::Float(f) => {
                        let const_idx = self.add_constant(Value::Float(*f));
                        self.emit(Instruction::LoadConst(reg, const_idx), 0);
                    }

                    Literal::String(s) => {
                        let const_idx = self.add_constant(Value::string(s.clone()));
                        self.emit(Instruction::LoadConst(reg, const_idx), 0);
                    }

                    Literal::Bool(b) => {
                        self.emit(Instruction::LoadBool(reg, *b), 0);
                    }
                }

                Ok(reg)
            }

            ExprKind::Identifier(name) => {
                if let Ok(source_reg) = self.resolve_local(name) {
                    return Ok(source_reg);
                }

                if self.is_module_level_identifier(name) {
                    let reg = self.allocate_register();
                    self.emit_load_module_global(name, reg)?;
                    return Ok(reg);
                }

                let mut lookup_name = name.clone();
                if let Some(module) = &self.current_module {
                    if let Some(imports) = self.imports_by_module.get(module) {
                        if let Some(fq) = imports.function_aliases.get(name) {
                            lookup_name = fq.clone();
                        }
                    }
                }

                if let Some(&func_idx) = self.function_table.get(&lookup_name) {
                    let reg = self.allocate_register();
                    let const_idx = self.add_constant(Value::Function(func_idx));
                    self.emit(Instruction::LoadConst(reg, const_idx), 0);
                    return Ok(reg);
                }

                if self.is_stdlib_symbol(name) {
                    let reg = self.allocate_register();
                    let name_idx = self.add_string_constant(name);
                    self.emit(Instruction::LoadGlobal(reg, name_idx), 0);
                    return Ok(reg);
                }

                if let Some(runtime_name) = self
                    .extern_function_aliases
                    .get(&lookup_name)
                    .cloned()
                    .or_else(|| self.extern_function_aliases.get(name).cloned())
                {
                    let reg = self.allocate_register();
                    let name_idx = self.add_string_constant(&runtime_name);
                    self.emit(Instruction::LoadGlobal(reg, name_idx), 0);
                    return Ok(reg);
                }

                Err(LustError::CompileError(format!(
                    "Undefined variable or function: {}",
                    name
                )))
            }

            ExprKind::Binary { left, op, right } => match op {
                BinaryOp::And => self.compile_and_expr(expr.span, left, right),
                BinaryOp::Or => self.compile_or_expr(left, right),
                _ => {
                    let left_reg = self.compile_expr(left)?;
                    let saved_freereg = self.next_register;
                    let right_reg = self.compile_expr(right)?;
                    if self.next_register < saved_freereg {
                        self.next_register = saved_freereg;
                    }

                    let result_reg = self.allocate_register();
                    self.compile_binary_op(*op, result_reg, left_reg, right_reg)?;
                    Ok(result_reg)
                }
            },

            ExprKind::Unary { op, operand } => {
                let operand_reg = self.compile_expr(operand)?;
                let result_reg = self.allocate_register();
                match op {
                    UnaryOp::Neg => self.emit(Instruction::Neg(result_reg, operand_reg), 0),
                    UnaryOp::Not => self.emit(Instruction::Not(result_reg, operand_reg), 0),
                };
                self.free_register(operand_reg);
                Ok(result_reg)
            }

            ExprKind::Call { callee, args } => {
                if let ExprKind::FieldAccess { object, field } = &callee.kind {
                    if let ExprKind::Identifier(type_name) = &object.kind {
                        let is_module_alias = self
                            .current_module
                            .as_ref()
                            .and_then(|module| self.imports_by_module.get(module))
                            .map_or(false, |imports| {
                                imports.module_aliases.contains_key(type_name)
                            });
                        if (Self::looks_like_type_name(type_name) || is_module_alias)
                            && self.resolve_local(type_name).is_err()
                        {
                            let mut candidates = Vec::new();
                            let mut alias_candidate = format!("{}.{}", type_name, field);
                            if let Some(module) = &self.current_module {
                                if let Some(imports) = self.imports_by_module.get(module) {
                                    if let Some(real_mod) = imports.module_aliases.get(type_name) {
                                        alias_candidate = format!("{}.{}", real_mod, field);
                                    }
                                }
                            }

                            candidates.push(alias_candidate.clone());
                            let resolved_type = self.resolve_type_name(type_name);
                            let resolved_candidate = format!("{}.{}", resolved_type, field);
                            if resolved_candidate != alias_candidate {
                                candidates.push(resolved_candidate);
                            }

                            for static_method_name in candidates {
                                if let Some(&func_idx) =
                                    self.function_table.get(&static_method_name)
                                {
                                    let func_reg = self.allocate_register();
                                    let const_idx = self.add_constant(Value::Function(func_idx));
                                    self.emit(Instruction::LoadConst(func_reg, const_idx), 0);
                                    let first_arg_reg = if args.is_empty() {
                                        0
                                    } else {
                                        let arg_refs: Vec<&Expr> = args.iter().collect();
                                        self.place_exprs_consecutive(&arg_refs)?
                                    };
                                    let result_reg = self.allocate_register();
                                    self.emit(
                                        Instruction::Call(
                                            func_reg,
                                            first_arg_reg,
                                            args.len() as u8,
                                            result_reg,
                                        ),
                                        0,
                                    );
                                    return Ok(result_reg);
                                }
                            }

                            let enum_name_idx = self.add_string_constant(type_name);
                            let variant_idx = self.add_string_constant(field);
                            if args.is_empty() {
                                let result_reg = self.allocate_register();
                                self.emit(
                                    Instruction::NewEnumUnit(
                                        result_reg,
                                        enum_name_idx,
                                        variant_idx,
                                    ),
                                    0,
                                );
                                return Ok(result_reg);
                            } else {
                                let arg_refs: Vec<&Expr> = args.iter().collect();
                                let first_value_reg = self.place_exprs_consecutive(&arg_refs)?;
                                let result_reg = self.allocate_register();
                                self.emit(
                                    Instruction::NewEnumVariant(
                                        result_reg,
                                        enum_name_idx,
                                        variant_idx,
                                        first_value_reg,
                                        args.len() as u8,
                                    ),
                                    0,
                                );
                                return Ok(result_reg);
                            }
                        }
                    }
                }

                let first_arg_reg = if args.is_empty() {
                    0
                } else {
                    let arg_refs: Vec<&Expr> = args.iter().collect();
                    self.place_exprs_consecutive(&arg_refs)?
                };
                let min_next = first_arg_reg.wrapping_add(args.len() as u8);
                if self.next_register < min_next {
                    self.next_register = min_next;
                }

                let func_reg = self.compile_expr(callee)?;
                let result_reg = self.allocate_register();
                self.emit(
                    Instruction::Call(func_reg, first_arg_reg, args.len() as u8, result_reg),
                    0,
                );
                Ok(result_reg)
            }

            ExprKind::MethodCall {
                receiver,
                method,
                type_args,
                args,
            } => {
                match method.as_str() {
                    "map" if args.len() == 1 && type_args.is_none() => {
                        return self.compile_map_method(receiver, &args[0]);
                    }

                    "filter" if args.len() == 1 && type_args.is_none() => {
                        return self.compile_filter_method(receiver, &args[0]);
                    }

                    "reduce" if args.len() == 2 && type_args.is_none() => {
                        return self.compile_reduce_method(receiver, &args[0], &args[1]);
                    }

                    _ => {}
                }

                let _ = type_args;
                let obj_reg = self.compile_expr(receiver)?;
                let method_idx = self.add_string_constant(method);
                let result_reg = if args.is_empty() {
                    let result_reg = self.allocate_register();
                    self.emit(
                        Instruction::CallMethod(obj_reg, method_idx, 0, 0, result_reg),
                        0,
                    );
                    result_reg
                } else {
                    let arg_refs: Vec<&Expr> = args.iter().collect();
                    let first_arg_reg = self.place_exprs_consecutive(&arg_refs)?;
                    let result_reg = first_arg_reg;
                    self.emit(
                        Instruction::CallMethod(
                            obj_reg,
                            method_idx,
                            first_arg_reg,
                            args.len() as u8,
                            result_reg,
                        ),
                        0,
                    );
                    result_reg
                };
                Ok(result_reg)
            }

            ExprKind::FieldAccess { object, field } => {
                if let ExprKind::Identifier(enum_name) = &object.kind {
                    if Self::looks_like_type_name(enum_name)
                        && self.resolve_local(enum_name).is_err()
                    {
                        let enum_name_idx = self.add_string_constant(enum_name);
                        let variant_idx = self.add_string_constant(field);
                        let result_reg = self.allocate_register();
                        self.emit(
                            Instruction::NewEnumUnit(result_reg, enum_name_idx, variant_idx),
                            0,
                        );
                        return Ok(result_reg);
                    }
                }

                let obj_reg = self.compile_expr(object)?;
                let field_idx = self.add_string_constant(field);
                let result_reg = self.allocate_register();
                self.emit(Instruction::GetField(result_reg, obj_reg, field_idx), 0);
                self.free_register(obj_reg);
                Ok(result_reg)
            }

            ExprKind::Index { object, index } => {
                let obj_reg = self.compile_expr(object)?;
                let idx_reg = self.compile_expr(index)?;
                let result_reg = self.allocate_register();
                self.emit(Instruction::GetIndex(result_reg, obj_reg, idx_reg), 0);
                self.free_register(obj_reg);
                self.free_register(idx_reg);
                Ok(result_reg)
            }

            ExprKind::Array(elements) => {
                if elements.is_empty() {
                    let result_reg = self.allocate_register();
                    self.emit(Instruction::NewArray(result_reg, 0, 0), 0);
                    return Ok(result_reg);
                }

                let elem_refs: Vec<&Expr> = elements.iter().collect();
                let first_elem_reg = self.place_exprs_consecutive(&elem_refs)?;
                let result_reg = self.allocate_register();
                self.emit(
                    Instruction::NewArray(result_reg, first_elem_reg, elements.len() as u8),
                    0,
                );
                Ok(result_reg)
            }

            ExprKind::Map(entries) => {
                let result_reg = self.allocate_register();
                self.emit(Instruction::NewMap(result_reg), 0);
                for (key_expr, value_expr) in entries {
                    let key_reg = self.compile_expr(key_expr)?;
                    let value_reg = self.compile_expr(value_expr)?;
                    self.emit(Instruction::SetIndex(result_reg, key_reg, value_reg), 0);
                    self.free_register(key_reg);
                    self.free_register(value_reg);
                }

                Ok(result_reg)
            }

            ExprKind::StructLiteral { name, fields } => {
                let resolved_name = self.resolve_type_name(name);
                let name_idx = self.add_string_constant(&resolved_name);
                let first_field_name_idx = self.ensure_contiguous_field_name_constants(fields);
                let first_field_reg = if fields.is_empty() {
                    0
                } else {
                    let field_refs: Vec<&Expr> = fields.iter().map(|f| &f.value).collect();
                    self.place_exprs_consecutive(&field_refs)?
                };
                let result_reg = self.allocate_register();
                self.emit(
                    Instruction::NewStruct(
                        result_reg,
                        name_idx,
                        first_field_name_idx,
                        first_field_reg,
                        fields.len() as u8,
                    ),
                    0,
                );
                Ok(result_reg)
            }

            ExprKind::EnumConstructor {
                enum_name,
                variant,
                args,
            } => {
                let resolved_enum = self.resolve_type_name(enum_name);
                let enum_name_idx = self.add_string_constant(&resolved_enum);
                let variant_idx = self.add_string_constant(variant);
                if args.is_empty() {
                    let result_reg = self.allocate_register();
                    self.emit(
                        Instruction::NewEnumUnit(result_reg, enum_name_idx, variant_idx),
                        0,
                    );
                    Ok(result_reg)
                } else {
                    let arg_refs: Vec<&Expr> = args.iter().collect();
                    let first_value_reg = self.place_exprs_consecutive(&arg_refs)?;
                    let result_reg = self.allocate_register();
                    self.emit(
                        Instruction::NewEnumVariant(
                            result_reg,
                            enum_name_idx,
                            variant_idx,
                            first_value_reg,
                            args.len() as u8,
                        ),
                        0,
                    );
                    Ok(result_reg)
                }
            }

            ExprKind::Tuple(elements) => self.compile_exprs_to_tuple(elements),
            ExprKind::Paren(inner) => self.compile_expr(inner),
            ExprKind::Block(stmts) => {
                self.begin_scope();
                let mut result_reg = self.allocate_register();
                self.emit(Instruction::LoadNil(result_reg), 0);
                for stmt in stmts {
                    if let StmtKind::Expr(expr) = &stmt.kind {
                        self.free_register(result_reg);
                        result_reg = self.compile_expr(expr)?;
                    } else {
                        self.compile_stmt(stmt)?;
                    }
                }

                self.end_scope();
                Ok(result_reg)
            }

            ExprKind::Lambda {
                params,
                return_type,
                body,
            } => {
                let captured_vars = self.analyze_free_variables(body, &params)?;
                let lambda_func_idx = self.functions.len();
                let lambda_name = format!("<lambda@{}>", lambda_func_idx);
                let lambda_func = Function::new(&lambda_name, params.len() as u8, false);
                self.functions.push(lambda_func);
                self.try_set_lambda_signature(lambda_func_idx, &params, &return_type);
                let saved_func_idx = self.current_function;
                let saved_scopes = self.scopes.clone();
                let saved_next_reg = self.next_register;
                let saved_max_reg = self.max_register;
                self.current_function = lambda_func_idx;
                self.next_register = 0;
                self.max_register = 0;
                self.scopes.clear();
                let mut lambda_scope = Scope {
                    locals: HashMap::new(),
                    depth: 0,
                };
                for (param_name, _param_type) in params {
                    let reg = self.allocate_register();
                    lambda_scope.locals.insert(param_name.clone(), (reg, false));
                }

                self.scopes.push(lambda_scope);
                let mut upvalue_map = HashMap::new();
                for (upvalue_idx, var_name) in captured_vars.iter().enumerate() {
                    let upvalue_reg = self.allocate_register();
                    upvalue_map.insert(var_name.clone(), upvalue_reg);
                    self.emit(Instruction::LoadUpvalue(upvalue_reg, upvalue_idx as u8), 0);
                }

                if let Some(scope) = self.scopes.last_mut() {
                    for (var_name, reg) in &upvalue_map {
                        scope.locals.insert(var_name.clone(), (*reg, false));
                    }
                }

                let body_reg = self.compile_expr(body)?;
                self.emit(Instruction::Return(body_reg), 0);
                self.free_register(body_reg);
                self.functions[lambda_func_idx].set_register_count(self.max_register + 1);
                self.scopes = saved_scopes;
                self.current_function = saved_func_idx;
                self.next_register = saved_next_reg;
                self.max_register = saved_max_reg;
                for var_name in &captured_vars {
                    let var_reg = self.resolve_local(var_name)?;
                    self.functions[lambda_func_idx].add_upvalue(true, var_reg);
                }

                let result_reg = self.allocate_register();
                let first_upvalue_reg = if !captured_vars.is_empty() {
                    let first_reg = self.next_register;
                    for var_name in &captured_vars {
                        let var_reg = self.resolve_local(var_name)?;
                        let upvalue_src_reg = self.allocate_register();
                        if var_reg != upvalue_src_reg {
                            self.emit(Instruction::Move(upvalue_src_reg, var_reg), 0);
                        }
                    }

                    first_reg
                } else {
                    0
                };
                self.emit(
                    Instruction::Closure(
                        result_reg,
                        lambda_func_idx as u16,
                        first_upvalue_reg,
                        captured_vars.len() as u8,
                    ),
                    0,
                );
                for _ in 0..captured_vars.len() {
                    let last = self.next_register.saturating_sub(1);
                    self.free_register(last);
                }

                Ok(result_reg)
            }

            ExprKind::TypeCheck { expr, check_type } => {
                let value_reg = self.compile_expr(expr)?;
                let is_trait = match &check_type.kind {
                    crate::ast::TypeKind::Trait(_) => true,
                    crate::ast::TypeKind::Named(name) => self.trait_names.contains(name),
                    _ => false,
                };
                let is_likely_variant = if !is_trait {
                    if let crate::ast::TypeKind::Named(name) = &check_type.kind {
                        name.chars().next().map_or(false, |c| c.is_uppercase())
                    } else {
                        false
                    }
                } else {
                    false
                };
                if is_likely_variant {
                    if let crate::ast::TypeKind::Named(variant_name) = &check_type.kind {
                        let enum_name_idx = self.add_string_constant("");
                        let variant_idx = self.add_string_constant(variant_name);
                        let result_reg = self.allocate_register();
                        self.emit(
                            Instruction::IsEnumVariant(
                                result_reg,
                                value_reg,
                                enum_name_idx,
                                variant_idx,
                            ),
                            0,
                        );
                        self.free_register(value_reg);
                        return Ok(result_reg);
                    }
                }

                let type_string = match &check_type.kind {
                    crate::ast::TypeKind::Named(name) => self.resolve_type_name(name),
                    _ => Self::type_to_string(&check_type.kind),
                };
                let type_name_idx = self.add_string_constant(&type_string);
                let result_reg = self.allocate_register();
                self.emit(Instruction::TypeIs(result_reg, value_reg, type_name_idx), 0);
                self.free_register(value_reg);
                Ok(result_reg)
            }

            ExprKind::IsPattern { expr, pattern } => {
                let scrutinee_reg = self.compile_expr(expr)?;
                let result_reg = self.compile_is_pattern(scrutinee_reg, pattern)?;
                self.free_register(scrutinee_reg);
                Ok(result_reg)
            }

            ExprKind::Cast { expr, .. } => self.compile_expr(expr),
            ExprKind::Return(values) => {
                if values.is_empty() {
                    self.emit(Instruction::Return(255), 0);
                    Ok(255)
                } else if values.len() == 1 {
                    let reg = self.compile_expr(&values[0])?;
                    self.emit(Instruction::Return(reg), 0);
                    Ok(reg)
                } else {
                    let tuple_reg = self.compile_exprs_to_tuple(values)?;
                    self.emit(Instruction::Return(tuple_reg), 0);
                    Ok(tuple_reg)
                }
            }

            _ => Err(LustError::CompileError(format!(
                "Expression form '{}' is not yet supported by the bytecode compiler",
                Self::describe_expr_kind(&expr.kind)
            ))),
        }
    }

    fn compile_and_expr(&mut self, span: Span, left: &Expr, right: &Expr) -> Result<Register> {
        let left_bindings = self.extract_all_pattern_bindings(left);
        let right_bindings = self.extract_all_pattern_bindings(right);

        let left_reg = self.compile_expr(left)?;
        let skip_right = self.emit(Instruction::JumpIfNot(left_reg, 0), 0);
        let wrap_option = self.should_wrap_option(span);
        let option_indices = if wrap_option {
            let option_idx = self.add_string_constant("Option");
            let some_idx = self.add_string_constant("Some");
            let none_idx = self.add_string_constant("None");
            Some((option_idx, some_idx, none_idx))
        } else {
            None
        };

        for (scrutinee_expr, pattern) in &left_bindings {
            let enum_reg = self.compile_expr(scrutinee_expr)?;
            self.bind_pattern_variables(enum_reg, pattern)?;
            self.free_register(enum_reg);
        }

        let saved_next = self.next_register;
        let right_reg = self.compile_expr(right)?;
        if self.next_register < saved_next {
            self.next_register = saved_next;
        }

        let right_skip_bindings = if !right_bindings.is_empty() {
            Some(self.emit(Instruction::JumpIfNot(right_reg, 0), 0))
        } else {
            None
        };

        for (scrutinee_expr, pattern) in &right_bindings {
            let enum_reg = self.compile_expr(scrutinee_expr)?;
            self.bind_pattern_variables(enum_reg, pattern)?;
            self.free_register(enum_reg);
        }

        if let Some(skip_idx) = right_skip_bindings {
            let end_idx = self.current_chunk().instructions.len();
            self.current_chunk_mut().patch_jump(skip_idx, end_idx);
        }

        if let Some((option_idx, some_idx, _)) = option_indices {
            self.emit(
                Instruction::NewEnumVariant(left_reg, option_idx, some_idx, right_reg, 1),
                0,
            );
        } else {
            self.emit(Instruction::Move(left_reg, right_reg), 0);
        }
        self.free_register(right_reg);

        let jump_to_end = if wrap_option {
            Some(self.emit(Instruction::Jump(0), 0))
        } else {
            None
        };

        let end_idx = self.current_chunk().instructions.len();
        self.current_chunk_mut().patch_jump(skip_right, end_idx);

        if let Some((option_idx, _, none_idx)) = option_indices {
            self.emit(Instruction::NewEnumUnit(left_reg, option_idx, none_idx), 0);
        }

        if let Some(jump_idx) = jump_to_end {
            let exit_idx = self.current_chunk().instructions.len();
            self.current_chunk_mut().patch_jump(jump_idx, exit_idx);
        }
        Ok(left_reg)
    }

    fn compile_or_expr(&mut self, left: &Expr, right: &Expr) -> Result<Register> {
        let left_reg = self.compile_expr(left)?;
        let skip_right = self.emit(Instruction::JumpIf(left_reg, 0), 0);

        let saved_next = self.next_register;
        let right_reg = self.compile_expr(right)?;
        if self.next_register < saved_next {
            self.next_register = saved_next;
        }

        self.emit(Instruction::Move(left_reg, right_reg), 0);
        self.free_register(right_reg);

        let end_idx = self.current_chunk().instructions.len();
        self.current_chunk_mut().patch_jump(skip_right, end_idx);
        Ok(left_reg)
    }

    pub(super) fn compile_binary_op(
        &mut self,
        op: BinaryOp,
        dest: Register,
        left: Register,
        right: Register,
    ) -> Result<()> {
        let instr = match op {
            BinaryOp::Add => Instruction::Add(dest, left, right),
            BinaryOp::Sub => Instruction::Sub(dest, left, right),
            BinaryOp::Mul => Instruction::Mul(dest, left, right),
            BinaryOp::Div => Instruction::Div(dest, left, right),
            BinaryOp::Mod => Instruction::Mod(dest, left, right),
            BinaryOp::Eq => Instruction::Eq(dest, left, right),
            BinaryOp::Ne => Instruction::Ne(dest, left, right),
            BinaryOp::Lt => Instruction::Lt(dest, left, right),
            BinaryOp::Le => Instruction::Le(dest, left, right),
            BinaryOp::Gt => Instruction::Gt(dest, left, right),
            BinaryOp::Ge => Instruction::Ge(dest, left, right),
            BinaryOp::And => Instruction::And(dest, left, right),
            BinaryOp::Or => Instruction::Or(dest, left, right),
            BinaryOp::Concat => Instruction::Concat(dest, left, right),
            _ => {
                return Err(LustError::CompileError(format!(
                    "Binary operator '{}' is not supported by the bytecode compiler",
                    op
                )))
            }
        };
        self.emit(instr, 0);
        Ok(())
    }

    pub(super) fn place_exprs_consecutive(&mut self, exprs: &[&Expr]) -> Result<Register> {
        if exprs.is_empty() {
            return Ok(self.next_register);
        }

        let first_dest = self.next_register;
        for _ in 0..exprs.len() {
            self.allocate_register();
        }

        let reserved_next = self.next_register;
        for (i, expr) in exprs.iter().enumerate() {
            let result_reg = self.compile_expr(expr)?;
            let dest = first_dest + i as u8;
            if result_reg != dest {
                self.emit(Instruction::Move(dest, result_reg), 0);
                self.free_register(result_reg);
            }

            if self.next_register < reserved_next {
                self.next_register = reserved_next;
            }
        }

        Ok(first_dest)
    }

    pub(super) fn compile_exprs_to_tuple(&mut self, exprs: &[Expr]) -> Result<Register> {
        if exprs.len() == 1 {
            return self.compile_expr(&exprs[0]);
        }

        let expr_refs: Vec<&Expr> = exprs.iter().collect();
        let first_reg = self.place_exprs_consecutive(&expr_refs)?;
        let tuple_reg = self.allocate_register();
        self.emit(
            Instruction::TupleNew(tuple_reg, first_reg, exprs.len() as u8),
            0,
        );
        for offset in (0..exprs.len()).rev() {
            self.free_register(first_reg + offset as u8);
        }

        Ok(tuple_reg)
    }

    pub(super) fn assign_value_to_target(
        &mut self,
        target: &Expr,
        value_reg: Register,
    ) -> Result<()> {
        match &target.kind {
            ExprKind::Identifier(name) => {
                if let Ok(target_reg) = self.resolve_local(name) {
                    if value_reg != target_reg {
                        self.emit(Instruction::Move(target_reg, value_reg), 0);
                    }

                    if self.should_sync_module_local(name) {
                        self.emit_store_module_global(name, target_reg);
                    }
                } else if self.is_module_level_identifier(name) {
                    self.emit_store_module_global(name, value_reg);
                } else {
                    return Err(LustError::CompileError(format!(
                        "Undefined variable: {}",
                        name
                    )));
                }
            }

            ExprKind::FieldAccess { object, field } => {
                let object_reg = self.compile_expr(object)?;
                let field_idx = self.add_string_constant(field);
                self.emit(Instruction::SetField(object_reg, field_idx, value_reg), 0);
                self.free_register(object_reg);
            }

            ExprKind::Index { object, index } => {
                let object_reg = self.compile_expr(object)?;
                let index_reg = self.compile_expr(index)?;
                self.emit(Instruction::SetIndex(object_reg, index_reg, value_reg), 0);
                self.free_register(object_reg);
                self.free_register(index_reg);
            }

            _ => {
                return Err(LustError::CompileError(
                    "Invalid assignment target".to_string(),
                ))
            }
        }

        Ok(())
    }
}

impl Compiler {
    fn ensure_contiguous_field_name_constants(&mut self, fields: &[StructLiteralField]) -> u16 {
        if fields.is_empty() {
            return 0;
        }

        let mut indices = Vec::with_capacity(fields.len());
        for field in fields {
            indices.push(self.add_string_constant(&field.name));
        }

        let base = indices[0];
        let contiguous = indices
            .iter()
            .enumerate()
            .all(|(i, idx)| *idx == base + i as u16);
        if contiguous {
            return base;
        }

        let chunk = self.current_chunk_mut();
        let base = chunk.constants.len() as u16;
        for field in fields {
            chunk.constants.push(Value::string(field.name.clone()));
        }

        base
    }
}
