use super::*;
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec::Vec,
};
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

    pub fn check_map_literal(
        &mut self,
        entries: &[(Expr, Expr)],
        expected_type: Option<&Type>,
    ) -> Result<Type> {
        let mut expected_key_ty: Option<&Type> = None;
        let mut expected_value_ty: Option<&Type> = None;
        let mut allow_mixed_keys = false;
        let mut allow_mixed_values = false;
        if let Some(expected) = expected_type {
            match &expected.kind {
                TypeKind::Map(key, value) => {
                    expected_key_ty = Some(key.as_ref());
                    expected_value_ty = Some(value.as_ref());
                    allow_mixed_keys = matches!(key.kind, TypeKind::Unknown | TypeKind::Infer);
                    allow_mixed_values = matches!(value.kind, TypeKind::Unknown | TypeKind::Infer);
                }

                TypeKind::Table => {
                    allow_mixed_keys = true;
                    allow_mixed_values = true;
                }

                _ => {}
            }
        }

        if entries.is_empty() {
            if let Some(expected) = expected_type {
                match &expected.kind {
                    TypeKind::Map(_, _) => {
                        return Ok(self.canonicalize_type(expected));
                    }

                    TypeKind::Table => {
                        let span = Self::dummy_span();
                        return Ok(Type::new(
                            TypeKind::Map(
                                Box::new(Type::new(TypeKind::Unknown, span)),
                                Box::new(Type::new(TypeKind::Unknown, span)),
                            ),
                            span,
                        ));
                    }

                    _ => {}
                }
            }

            let span = Self::dummy_span();
            return Ok(Type::new(
                TypeKind::Map(
                    Box::new(Type::new(TypeKind::Unknown, span)),
                    Box::new(Type::new(TypeKind::Unknown, span)),
                ),
                span,
            ));
        }

        let key_hint = expected_key_ty.and_then(|ty| {
            if matches!(ty.kind, TypeKind::Unknown | TypeKind::Infer) {
                None
            } else {
                Some(ty)
            }
        });
        let value_hint = expected_value_ty.and_then(|ty| {
            if matches!(ty.kind, TypeKind::Unknown | TypeKind::Infer) {
                None
            } else {
                Some(ty)
            }
        });

        let mut inferred_key_type: Option<Type> = None;
        let mut inferred_value_type: Option<Type> = None;
        for (key_expr, value_expr) in entries {
            let raw_key_type = if let Some(hint) = key_hint {
                self.check_expr_with_hint(key_expr, Some(hint))?
            } else {
                self.check_expr(key_expr)?
            };
            if !self.env.type_implements_trait(&raw_key_type, "Hashable") {
                return Err(self.type_error(format!(
                    "Map key type '{}' must implement Hashable trait",
                    raw_key_type
                )));
            }
            let canonical_key = self.canonicalize_type(&raw_key_type);

            let raw_value_type = if let Some(hint) = value_hint {
                self.check_expr_with_hint(value_expr, Some(hint))?
            } else {
                self.check_expr(value_expr)?
            };
            let canonical_value = self.canonicalize_type(&raw_value_type);

            if let Some(existing_key) = &inferred_key_type {
                if !allow_mixed_keys {
                    self.unify(existing_key, &canonical_key)?;
                }
            } else {
                inferred_key_type = Some(canonical_key.clone());
            }

            if let Some(existing_value) = &inferred_value_type {
                if !allow_mixed_values {
                    self.unify(existing_value, &canonical_value)?;
                }
            } else {
                inferred_value_type = Some(canonical_value.clone());
            }
        }

        let span = Self::dummy_span();
        let key_type = if allow_mixed_keys {
            expected_key_ty
                .and_then(|ty| Some(self.canonicalize_type(ty)))
                .unwrap_or_else(|| Type::new(TypeKind::Unknown, span))
        } else {
            inferred_key_type.unwrap_or_else(|| Type::new(TypeKind::Unknown, span))
        };
        let value_type = if allow_mixed_values {
            expected_value_ty
                .and_then(|ty| Some(self.canonicalize_type(ty)))
                .unwrap_or_else(|| Type::new(TypeKind::Unknown, span))
        } else {
            inferred_value_type.unwrap_or_else(|| Type::new(TypeKind::Unknown, span))
        };

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
