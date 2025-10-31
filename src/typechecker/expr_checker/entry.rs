use super::*;
use alloc::{
    boxed::Box,
    format,
    string::ToString,
    vec::Vec,
};
use hashbrown::HashMap;
impl TypeChecker {
    pub fn check_expr(&mut self, expr: &Expr) -> Result<Type> {
        let mut ty = self.check_expr_with_hint(expr, None)?;
        if ty.span.start_line == 0 && expr.span.start_line > 0 {
            ty.span = expr.span;
        }

        if expr.span.start_line > 0 {
            if let Some(module) = &self.current_module {
                self.expr_types_by_module
                    .entry(module.clone())
                    .or_default()
                    .insert(expr.span, ty.clone());
            }
        }

        Ok(ty)
    }

    pub fn check_expr_with_hint(
        &mut self,
        expr: &Expr,
        expected_type: Option<&Type>,
    ) -> Result<Type> {
        match &expr.kind {
            ExprKind::Literal(lit) => self.check_literal(lit),
            ExprKind::Identifier(name) => self.env.lookup_variable(name).ok_or_else(|| {
                self.type_error_at(format!("Undefined variable '{}'", name), expr.span)
            }),
            ExprKind::Binary { left, op, right } => self.check_binary_expr(left, op, right),
            ExprKind::Unary { op, operand } => self.check_unary_expr(op, operand),
            ExprKind::Call { callee, args } => self.check_call_expr(expr.span, callee, args),
            ExprKind::MethodCall {
                receiver,
                method,
                type_args,
                args,
            } => {
                if method == "as" && type_args.is_some() {
                    let _receiver_type = self.check_expr(receiver)?;
                    if !args.is_empty() {
                        return Err(self.type_error(":as<T>() takes no arguments".to_string()));
                    }

                    let target_type = &type_args.as_ref().unwrap()[0];
                    let actual_target_type = if let TypeKind::Named(name) = &target_type.kind {
                        let resolved = self.resolve_type_key(name);
                        if self.env.lookup_trait(&resolved).is_some() {
                            Type::new(TypeKind::Trait(name.clone()), target_type.span)
                        } else {
                            target_type.clone()
                        }
                    } else {
                        target_type.clone()
                    };
                    let span = Self::dummy_span();
                    return Ok(Type::new(
                        TypeKind::Option(Box::new(actual_target_type)),
                        span,
                    ));
                }

                let _ = type_args;
                self.check_method_call(receiver, method, args)
            }

            ExprKind::FieldAccess { object, field } => {
                self.check_field_access_with_hint(expr.span, object, field, expected_type)
            }

            ExprKind::Index { object, index } => self.check_index_expr(object, index),
            ExprKind::Array(elements) => self.check_array_literal(elements, expected_type),
            ExprKind::Map(entries) => self.check_map_literal(entries),
            ExprKind::StructLiteral { name, fields } => {
                self.check_struct_literal(expr.span, name, fields)
            }

            ExprKind::Lambda {
                params,
                return_type,
                body,
            } => self.check_lambda(params, return_type.as_ref(), body),
            ExprKind::Cast { expr, target_type } => {
                let _expr_type = self.check_expr(expr)?;
                Ok(target_type.clone())
            }

            ExprKind::TypeCheck {
                expr,
                check_type: _,
            } => {
                let _expr_type = self.check_expr(expr)?;
                Ok(Type::new(TypeKind::Bool, Self::dummy_span()))
            }

            ExprKind::IsPattern { expr, pattern } => {
                let scrutinee_type = self.check_expr(expr)?;
                self.validate_is_pattern(pattern, &scrutinee_type)?;
                Ok(Type::new(TypeKind::Bool, Self::dummy_span()))
            }

            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => self.check_if_expr(condition, then_branch, else_branch),
            ExprKind::Block(stmts) => {
                self.env.push_scope();
                let mut result_type = Type::new(TypeKind::Unit, Self::dummy_span());
                for stmt in stmts {
                    match &stmt.kind {
                        StmtKind::Expr(expr) => {
                            result_type = self.check_expr(expr)?;
                        }

                        StmtKind::Return(values) => {
                            result_type = if values.is_empty() {
                                Type::new(TypeKind::Unit, Self::dummy_span())
                            } else if values.len() == 1 {
                                let expected = self.current_function_return_type.clone();
                                let mut raw =
                                    self.check_expr_with_hint(&values[0], expected.as_ref())?;
                                if raw.span.start_line == 0 && values[0].span.start_line > 0 {
                                    raw.span = values[0].span;
                                }

                                self.canonicalize_type(&raw)
                            } else {
                                let mut el_types = Vec::new();
                                for value in values {
                                    let raw = self.check_expr(value)?;
                                    el_types.push(self.canonicalize_type(&raw));
                                }

                                Type::new(TypeKind::Tuple(el_types), Self::dummy_span())
                            };
                            self.pending_generic_instances.take();
                            self.check_stmt(stmt)?;
                        }

                        _ => {
                            self.check_stmt(stmt)?;
                        }
                    }
                }

                self.env.pop_scope();
                if result_type.span.start_line == 0 {
                    result_type.span = expr.span;
                }

                Ok(result_type)
            }

            ExprKind::Range { .. } => Err(self.type_error_at(
                "Range expressions are not supported; use numeric for-loops".to_string(),
                expr.span,
            )),
            ExprKind::EnumConstructor {
                enum_name,
                variant,
                args,
            } => {
                let enum_def = self
                    .env
                    .lookup_enum(enum_name)
                    .ok_or_else(|| self.type_error(format!("Undefined enum '{}'", enum_name)))?
                    .clone();
                let variant_def = enum_def
                    .variants
                    .iter()
                    .find(|v| v.name == *variant)
                    .ok_or_else(|| {
                        self.type_error(format!(
                            "Enum '{}' has no variant '{}'",
                            enum_name, variant
                        ))
                    })?;
                if let Some(expected_fields) = &variant_def.fields {
                    if args.len() != expected_fields.len() {
                        return Err(self.type_error(format!(
                            "Variant '{}::{}' expects {} arguments, got {}",
                            enum_name,
                            variant,
                            expected_fields.len(),
                            args.len()
                        )));
                    }

                    let mut type_params = HashMap::new();
                    for (arg, expected_type) in args.iter().zip(expected_fields.iter()) {
                        let arg_type = self.check_expr(arg)?;
                        if let TypeKind::Generic(type_param) = &expected_type.kind {
                            type_params.insert(type_param.clone(), arg_type.clone());
                        } else {
                            self.unify(expected_type, &arg_type)?;
                        }
                    }

                    if !type_params.is_empty() {
                        self.pending_generic_instances = Some(type_params.clone());
                    }

                    if enum_name == "Option" {
                        if let Some(inner_type) = type_params.get("T") {
                            return Ok(Type::new(
                                TypeKind::Option(Box::new(inner_type.clone())),
                                Self::dummy_span(),
                            ));
                        }
                    } else if enum_name == "Result" {
                        if let (Some(ok_type), Some(err_type)) =
                            (type_params.get("T"), type_params.get("E"))
                        {
                            return Ok(Type::new(
                                TypeKind::Result(
                                    Box::new(ok_type.clone()),
                                    Box::new(err_type.clone()),
                                ),
                                Self::dummy_span(),
                            ));
                        }
                    }
                } else {
                    if !args.is_empty() {
                        return Err(self.type_error(format!(
                            "Variant '{}::{}' is a unit variant and takes no arguments",
                            enum_name, variant
                        )));
                    }
                }

                Ok(Type::new(
                    TypeKind::Named(enum_name.clone()),
                    Self::dummy_span(),
                ))
            }

            ExprKind::Tuple(elements) => {
                let expected_elements = expected_type.and_then(|ty| {
                    if let TypeKind::Tuple(elems) = &ty.kind {
                        Some(elems.clone())
                    } else {
                        None
                    }
                });
                let mut element_types = Vec::new();
                for (index, element) in elements.iter().enumerate() {
                    let hint = expected_elements
                        .as_ref()
                        .and_then(|elems| elems.get(index));
                    let mut raw_ty = if let Some(hint_ty) = hint {
                        self.check_expr_with_hint(element, Some(hint_ty))?
                    } else {
                        self.check_expr(element)?
                    };
                    if raw_ty.span.start_line == 0 && element.span.start_line > 0 {
                        raw_ty.span = element.span;
                    }

                    self.pending_generic_instances.take();
                    element_types.push(self.canonicalize_type(&raw_ty));
                }

                Ok(Type::new(TypeKind::Tuple(element_types), expr.span))
            }

            ExprKind::Return(exprs) => {
                let mut return_type = if exprs.is_empty() {
                    Type::new(TypeKind::Unit, Self::dummy_span())
                } else if exprs.len() == 1 {
                    let expected = self.current_function_return_type.clone();
                    let mut raw_ty = self.check_expr_with_hint(&exprs[0], expected.as_ref())?;
                    if raw_ty.span.start_line == 0 && exprs[0].span.start_line > 0 {
                        raw_ty.span = exprs[0].span;
                    }

                    self.pending_generic_instances.take();
                    raw_ty
                } else {
                    let mut types = Vec::new();
                    for value in exprs {
                        let raw_ty = self.check_expr(value)?;
                        let ty = self.canonicalize_type(&raw_ty);
                        self.pending_generic_instances.take();
                        types.push(ty);
                    }

                    Type::new(TypeKind::Tuple(types), Self::dummy_span())
                };
                if return_type.span.start_line == 0 {
                    if let Some(first) = exprs.first() {
                        return_type.span = first.span;
                    } else {
                        return_type.span = expr.span;
                    }
                }

                if let Some(expected_return) = &self.current_function_return_type {
                    self.unify(expected_return, &return_type)?;
                } else {
                    return Err(self.type_error("'return' outside of function".to_string()));
                }

                Ok(return_type)
            }

            ExprKind::Paren(inner) => self.check_expr(inner),
        }
    }
}
