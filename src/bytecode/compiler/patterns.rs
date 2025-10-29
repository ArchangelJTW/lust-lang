use super::*;
impl Compiler {
    pub(super) fn compile_is_pattern(
        &mut self,
        scrutinee_reg: Register,
        pattern: &crate::ast::Pattern,
    ) -> Result<Register> {
        use crate::ast::Pattern;
        match pattern {
            Pattern::Wildcard => {
                let result_reg = self.allocate_register();
                let true_idx = self.add_constant(Value::Bool(true));
                self.emit(Instruction::LoadConst(result_reg, true_idx), 0);
                Ok(result_reg)
            }

            Pattern::Identifier(_) => {
                let result_reg = self.allocate_register();
                let true_idx = self.add_constant(Value::Bool(true));
                self.emit(Instruction::LoadConst(result_reg, true_idx), 0);
                Ok(result_reg)
            }

            Pattern::Literal(lit) => {
                let lit_reg = self.allocate_register();
                match lit {
                    crate::ast::Literal::Integer(i) => {
                        let const_idx = self.add_int_const(*i);
                        self.emit(Instruction::LoadConst(lit_reg, const_idx), 0);
                    }

                    crate::ast::Literal::Bool(b) => {
                        let const_idx = self.add_constant(Value::Bool(*b));
                        self.emit(Instruction::LoadConst(lit_reg, const_idx), 0);
                    }

                    crate::ast::Literal::Float(f) => {
                        let const_idx = self.add_constant(Value::Float(*f));
                        self.emit(Instruction::LoadConst(lit_reg, const_idx), 0);
                    }

                    crate::ast::Literal::String(s) => {
                        let const_idx = self.add_string_constant(s);
                        self.emit(Instruction::LoadConst(lit_reg, const_idx), 0);
                    }
                }

                let result_reg = self.allocate_register();
                self.emit(Instruction::Eq(result_reg, scrutinee_reg, lit_reg), 0);
                self.free_register(lit_reg);
                Ok(result_reg)
            }

            Pattern::Enum {
                enum_name,
                variant,
                bindings: _,
            } => {
                let actual_enum_name = if enum_name.is_empty() {
                    match variant.as_str() {
                        "Some" | "None" => "Option".to_string(),
                        "Ok" | "Err" => "Result".to_string(),
                        _ => enum_name.clone(),
                    }
                } else {
                    enum_name.clone()
                };
                let canonical_enum_name = if actual_enum_name.is_empty() {
                    actual_enum_name
                } else {
                    self.resolve_type_name(&actual_enum_name)
                };
                let enum_name_idx = self.add_string_constant(&canonical_enum_name);
                let variant_idx = self.add_string_constant(variant);
                let result_reg = self.allocate_register();
                self.emit(
                    Instruction::IsEnumVariant(
                        result_reg,
                        scrutinee_reg,
                        enum_name_idx,
                        variant_idx,
                    ),
                    0,
                );
                Ok(result_reg)
            }

            Pattern::TypeCheck(type_) => {
                let type_string = match &type_.kind {
                    crate::ast::TypeKind::Named(name) => self.resolve_type_name(name),
                    _ => Self::type_to_string(&type_.kind),
                };
                let type_name_idx = self.add_string_constant(&type_string);
                let result_reg = self.allocate_register();
                self.emit(
                    Instruction::TypeIs(result_reg, scrutinee_reg, type_name_idx),
                    0,
                );
                Ok(result_reg)
            }

            Pattern::Struct { .. } => Err(LustError::CompileError(
                "Struct patterns in `is` operator not yet implemented".to_string(),
            )),
        }
    }

    pub(super) fn extract_all_pattern_bindings<'a>(
        &self,
        expr: &'a Expr,
    ) -> Vec<(&'a Expr, &'a crate::ast::Pattern)> {
        use crate::ast::ExprKind;
        let mut bindings = Vec::new();
        match &expr.kind {
            ExprKind::IsPattern {
                expr: scrutinee,
                pattern,
            } => match pattern {
                crate::ast::Pattern::Enum {
                    bindings: pattern_bindings,
                    ..
                } if !pattern_bindings.is_empty() => {
                    bindings.push((scrutinee.as_ref(), pattern));
                }

                _ => {}
            },
            ExprKind::Binary { left, op, right } => {
                if matches!(op, crate::ast::BinaryOp::And) {
                    bindings.extend(self.extract_all_pattern_bindings(left));
                    bindings.extend(self.extract_all_pattern_bindings(right));
                }
            }

            _ => {}
        }

        bindings
    }

    pub(super) fn bind_pattern_variables(
        &mut self,
        enum_reg: Register,
        pattern: &crate::ast::Pattern,
    ) -> Result<()> {
        use crate::ast::Pattern;
        match pattern {
            Pattern::Enum { bindings, .. } => {
                for (index, binding) in bindings.iter().enumerate() {
                    match binding {
                        Pattern::Identifier(name) => {
                            let mut max_local_reg: i32 = -1;
                            for sc in &self.scopes {
                                for &(_reg, _) in sc.locals.values() {}
                            }

                            for sc in &self.scopes {
                                for &(reg, _) in sc.locals.values() {
                                    if (reg as i32) > max_local_reg {
                                        max_local_reg = reg as i32;
                                    }
                                }
                            }

                            let value_reg: Register = (max_local_reg + 1) as u8;
                            if self.next_register <= value_reg {
                                self.next_register = value_reg + 1;
                            }

                            if value_reg > self.max_register {
                                self.max_register = value_reg;
                            }

                            self.emit(
                                Instruction::GetEnumValue(value_reg, enum_reg, index as u8),
                                0,
                            );
                            if let Some(scope) = self.scopes.last_mut() {
                                scope.locals.insert(name.clone(), (value_reg, false));
                            }
                        }

                        Pattern::Wildcard => {
                            continue;
                        }

                        Pattern::Enum { .. } => {
                            let temp_value_reg = self.allocate_register();
                            self.emit(
                                Instruction::GetEnumValue(temp_value_reg, enum_reg, index as u8),
                                0,
                            );
                            self.bind_pattern_variables(temp_value_reg, binding)?;
                            self.free_register(temp_value_reg);
                        }

                        _ => {
                            return Err(LustError::CompileError(format!(
                                "Pattern binding '{}' is not supported by the bytecode compiler",
                                binding
                            )));
                        }
                    }
                }

                Ok(())
            }

            _ => Ok(()),
        }
    }
}
