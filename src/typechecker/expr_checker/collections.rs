use super::*;
impl TypeChecker {
    pub fn check_array_literal(
        &mut self,
        elements: &[Expr],
        expected_type: Option<&Type>,
    ) -> Result<Type> {
        if elements.is_empty() {
            if let Some(expected) = expected_type {
                return Ok(expected.clone());
            }

            let span = Self::dummy_span();
            return Ok(Type::new(
                TypeKind::Array(Box::new(Type::new(TypeKind::Unknown, span))),
                span,
            ));
        }

        let expected_elem_type = expected_type.and_then(|t| {
            if let TypeKind::Array(elem_type) = &t.kind {
                Some(elem_type.as_ref())
            } else {
                None
            }
        });
        if let Some(expected_elem) = expected_elem_type {
            if let TypeKind::Union(union_types) = &expected_elem.kind {
                for elem in elements {
                    let elem_type = self.check_expr(elem)?;
                    let mut matches = false;
                    for union_variant in union_types {
                        if self.types_equal(&elem_type, union_variant) {
                            matches = true;
                            break;
                        }
                    }

                    if !matches {
                        let union_desc = union_types
                            .iter()
                            .map(|t| t.to_string())
                            .collect::<Vec<_>>()
                            .join(" | ");
                        return Err(self.type_error(format!(
                            "Array element type '{}' does not match any type in union [{}]",
                            elem_type, union_desc
                        )));
                    }
                }

                return Ok(expected_type.unwrap().clone());
            }
        }

        if let Some(expected_elem) = expected_elem_type {
            if matches!(expected_elem.kind, TypeKind::Unknown) {
                for elem in elements {
                    self.check_expr(elem)?;
                }

                return Ok(expected_type.unwrap().clone());
            }

            if let TypeKind::Option(inner) = &expected_elem.kind {
                if matches!(inner.kind, TypeKind::Unknown) {
                    for elem in elements {
                        let elem_type = self.check_expr(elem)?;
                        let is_option = matches!(&elem_type.kind, TypeKind::Option(_))
                            || matches!(&elem_type.kind, TypeKind::Named(name) if name == "Option");
                        if !is_option {
                            return Err(self.type_error(format!(
                                "Expected Option type for Array<Option<unknown>>, got '{}'",
                                elem_type
                            )));
                        }
                    }

                    return Ok(expected_type.unwrap().clone());
                }
            }

            if let TypeKind::Result(ok_inner, err_inner) = &expected_elem.kind {
                if matches!(ok_inner.kind, TypeKind::Unknown)
                    || matches!(err_inner.kind, TypeKind::Unknown)
                {
                    for elem in elements {
                        let elem_type = self.check_expr(elem)?;
                        let is_result = matches!(&elem_type.kind, TypeKind::Result(_, _))
                            || matches!(&elem_type.kind, TypeKind::Named(name) if name == "Result");
                        if !is_result {
                            return Err(self.type_error(format!(
                                "Expected Result type for Array<Result<unknown, ...>>, got '{}'",
                                elem_type
                            )));
                        }
                    }

                    return Ok(expected_type.unwrap().clone());
                }
            }
        }

        let first_type = self.check_expr(&elements[0])?;
        for elem in &elements[1..] {
            let elem_type = self.check_expr(elem)?;
            self.unify(&first_type, &elem_type)?;
        }

        Ok(Type::new(
            TypeKind::Array(Box::new(first_type)),
            Self::dummy_span(),
        ))
    }

    pub fn check_map_literal(&mut self, entries: &[(Expr, Expr)]) -> Result<Type> {
        if entries.is_empty() {
            let span = Self::dummy_span();
            return Ok(Type::new(
                TypeKind::Map(
                    Box::new(Type::new(TypeKind::Unknown, span)),
                    Box::new(Type::new(TypeKind::Unknown, span)),
                ),
                span,
            ));
        }

        let (first_key, first_value) = &entries[0];
        let key_type = self.check_expr(first_key)?;
        let value_type = self.check_expr(first_value)?;
        if !self.env.type_implements_trait(&key_type, "Hashable") {
            return Err(self.type_error(format!(
                "Map key type '{}' must implement Hashable trait",
                key_type
            )));
        }

        for (key, value) in &entries[1..] {
            let k_type = self.check_expr(key)?;
            let v_type = self.check_expr(value)?;
            self.unify(&key_type, &k_type)?;
            self.unify(&value_type, &v_type)?;
        }

        Ok(Type::new(
            TypeKind::Map(Box::new(key_type), Box::new(value_type)),
            Self::dummy_span(),
        ))
    }

    pub fn check_struct_literal(
        &mut self,
        span: Span,
        name: &str,
        fields: &[StructLiteralField],
    ) -> Result<Type> {
        let key = self.resolve_type_key(name);
        let struct_def = self
            .env
            .lookup_struct(&key)
            .or_else(|| self.env.lookup_struct(name))
            .ok_or_else(|| self.type_error_at(format!("Undefined struct '{}'", name), span))?
            .clone();
        if fields.len() != struct_def.fields.len() {
            return Err(self.type_error_at(
                format!(
                    "Struct '{}' has {} fields, but {} were provided",
                    name,
                    struct_def.fields.len(),
                    fields.len()
                ),
                span,
            ));
        }

        for field in fields {
            let expected_type = struct_def
                .fields
                .iter()
                .find(|f| f.name == field.name)
                .map(|f| &f.ty)
                .ok_or_else(|| {
                    self.type_error_at(
                        format!("Struct '{}' has no field '{}'", name, field.name),
                        field.span,
                    )
                })?;
            let actual_type = self.check_expr(&field.value)?;
            match &expected_type.kind {
                TypeKind::Option(inner_expected) => {
                    if self.unify(inner_expected, &actual_type).is_err() {
                        self.unify(expected_type, &actual_type)?;
                    }
                }

                _ => {
                    self.unify(expected_type, &actual_type)?;
                }
            }
        }

        let ty_name = if self.env.lookup_struct(&key).is_some() {
            key
        } else {
            name.to_string()
        };
        Ok(Type::new(TypeKind::Named(ty_name), Self::dummy_span()))
    }

    pub fn check_lambda(
        &mut self,
        params: &[(String, Option<Type>)],
        return_type: Option<&Type>,
        body: &Expr,
    ) -> Result<Type> {
        self.env.push_scope();
        let expected_signature = self.expected_lambda_signature.take();
        let mut param_types = Vec::new();
        for (i, (param_name, param_type)) in params.iter().enumerate() {
            let ty = if let Some(explicit_type) = param_type {
                explicit_type.clone()
            } else if let Some((ref expected_params, _)) = expected_signature {
                if i < expected_params.len() {
                    expected_params[i].clone()
                } else {
                    Type::new(TypeKind::Infer, Self::dummy_span())
                }
            } else {
                Type::new(TypeKind::Infer, Self::dummy_span())
            };
            self.env.declare_variable(param_name.clone(), ty.clone())?;
            param_types.push(ty);
        }

        let saved_return_type = self.current_function_return_type.clone();
        let inferred_return_type = if let Some(explicit) = return_type {
            Some(explicit.clone())
        } else if let Some((_, expected_ret)) = expected_signature {
            expected_ret.or_else(|| Some(Type::new(TypeKind::Infer, Self::dummy_span())))
        } else {
            Some(Type::new(TypeKind::Infer, Self::dummy_span()))
        };
        self.current_function_return_type = inferred_return_type.clone();
        let body_type = self.check_expr(body)?;
        self.current_function_return_type = saved_return_type;
        let actual_return_type = if let Some(expected) = return_type {
            expected.clone()
        } else if let Some(inferred) = &inferred_return_type {
            if !matches!(inferred.kind, TypeKind::Infer) {
                inferred.clone()
            } else {
                body_type
            }
        } else {
            body_type
        };
        self.env.pop_scope();
        Ok(Type::new(
            TypeKind::Function {
                params: param_types,
                return_type: Box::new(actual_return_type),
            },
            Self::dummy_span(),
        ))
    }

    pub fn check_if_expr(
        &mut self,
        condition: &Expr,
        then_branch: &Expr,
        else_branch: &Option<Box<Expr>>,
    ) -> Result<Type> {
        let cond_type = self.check_expr(condition)?;
        self.unify(&Type::new(TypeKind::Bool, Self::dummy_span()), &cond_type)?;
        let then_type = self.check_expr(then_branch)?;
        if let Some(else_expr) = else_branch {
            let else_type = self.check_expr(else_expr)?;
            self.unify(&then_type, &else_type)?;
            Ok(then_type)
        } else {
            Ok(Type::new(TypeKind::Unit, Self::dummy_span()))
        }
    }
}
