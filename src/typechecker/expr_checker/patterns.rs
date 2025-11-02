use super::*;
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
impl TypeChecker {
    pub fn validate_is_pattern(&mut self, pattern: &Pattern, scrutinee_type: &Type) -> Result<()> {
        match pattern {
            Pattern::Wildcard | Pattern::Literal(_) | Pattern::Identifier(_) => Ok(()),
            Pattern::TypeCheck(check_type) => {
                let _ = check_type;
                Ok(())
            }

            Pattern::Enum {
                enum_name: _,
                variant,
                bindings,
            } => {
                let (type_name, variant_types) = match &scrutinee_type.kind {
                    TypeKind::Named(name) => (name.clone(), None),
                    TypeKind::Option(inner) => {
                        ("Option".to_string(), Some(vec![(**inner).clone()]))
                    }

                    TypeKind::Result(ok, err) => (
                        "Result".to_string(),
                        Some(vec![(**ok).clone(), (**err).clone()]),
                    ),
                    TypeKind::Union(types) => {
                        for ty in types.iter() {
                            if let TypeKind::Named(name) = &ty.kind {
                                if let Some(_) = {
                                    let key = self.resolve_type_key(name);
                                    self.env
                                        .lookup_enum(&key)
                                        .or_else(|| self.env.lookup_enum(name))
                                } {
                                    return Ok(());
                                }
                            }

                            if matches!(ty.kind, TypeKind::Option(_) | TypeKind::Result(_, _)) {
                                return Ok(());
                            }
                        }

                        return Err(self.type_error(format!(
                            "Union type '{}' does not contain enum types compatible with variant '{}'",
                            scrutinee_type, variant
                        )));
                    }

                    _ => {
                        return Err(self.type_error(format!(
                            "Cannot use enum pattern on non-enum type '{}'",
                            scrutinee_type
                        )))
                    }
                };
                let enum_def = {
                    let key = self.resolve_type_key(&type_name);
                    self.env
                        .lookup_enum(&key)
                        .or_else(|| self.env.lookup_enum(&type_name))
                }
                .ok_or_else(|| self.type_error(format!("Undefined enum '{}'", type_name)))?
                .clone();
                let variant_def = enum_def
                    .variants
                    .iter()
                    .find(|v| &v.name == variant)
                    .ok_or_else(|| {
                        self.type_error(format!(
                            "Enum '{}' has no variant '{}'",
                            type_name, variant
                        ))
                    })?;
                if let Some(variant_fields) = &variant_def.fields {
                    if bindings.len() != variant_fields.len() {
                        return Err(self.type_error(format!(
                            "Variant '{}::{}' expects {} bindings, got {}",
                            type_name,
                            variant,
                            variant_fields.len(),
                            bindings.len()
                        )));
                    }

                    for (binding, field_type) in bindings.iter().zip(variant_fields.iter()) {
                        let bind_type = if let Some(ref types) = variant_types {
                            if let TypeKind::Generic(_) = &field_type.kind {
                                types.get(0).cloned().unwrap_or_else(|| field_type.clone())
                            } else {
                                field_type.clone()
                            }
                        } else {
                            field_type.clone()
                        };
                        self.validate_is_pattern(binding, &bind_type)?;
                    }
                } else {
                    if !bindings.is_empty() {
                        return Err(self.type_error(format!(
                            "Variant '{}::{}' is a unit variant and takes no bindings",
                            type_name, variant
                        )));
                    }
                }

                Ok(())
            }

            Pattern::Struct { .. } => Ok(()),
        }
    }

    pub fn extract_type_narrowings_from_expr(&mut self, expr: &Expr) -> Vec<(String, Type)> {
        let mut narrowings = Vec::new();
        match &expr.kind {
            ExprKind::TypeCheck {
                expr: scrutinee,
                check_type: target_type,
            } => {
                if let ExprKind::Identifier(var_name) = &scrutinee.kind {
                    if let Some(current_type) = self.env.lookup_variable(var_name) {
                        let narrowed_type = if let TypeKind::Named(name) = &target_type.kind {
                            let resolved = self.resolve_type_key(name);
                            if self.env.lookup_trait(&resolved).is_some() {
                                Type::new(TypeKind::Trait(name.clone()), target_type.span)
                            } else {
                                target_type.clone()
                            }
                        } else {
                            target_type.clone()
                        };
                        match &current_type.kind {
                            TypeKind::Unknown => {
                                narrowings.push((var_name.clone(), narrowed_type));
                            }

                            TypeKind::Union(types) => {
                                for ty in types {
                                    if self.types_equal(ty, target_type) {
                                        narrowings.push((var_name.clone(), target_type.clone()));
                                        break;
                                    }
                                }
                            }

                            _ => {}
                        }
                    }
                }
            }

            ExprKind::IsPattern {
                expr: scrutinee,
                pattern,
            } => {
                if let Pattern::TypeCheck(target_type) = pattern {
                    if let ExprKind::Identifier(var_name) = &scrutinee.kind {
                        if let Some(current_type) = self.env.lookup_variable(var_name) {
                            match &current_type.kind {
                                TypeKind::Unknown => {
                                    narrowings.push((var_name.clone(), target_type.clone()));
                                }

                                TypeKind::Union(types) => {
                                    for ty in types {
                                        if self.types_equal(ty, target_type) {
                                            narrowings
                                                .push((var_name.clone(), target_type.clone()));
                                            break;
                                        }
                                    }
                                }

                                _ => {}
                            }
                        }
                    }
                }
            }

            ExprKind::Binary { left, op, right } => {
                if matches!(op, BinaryOp::And) {
                    narrowings.extend(self.extract_type_narrowings_from_expr(left));
                    narrowings.extend(self.extract_type_narrowings_from_expr(right));
                }
            }

            ExprKind::Paren(inner) => {
                narrowings.extend(self.extract_type_narrowings_from_expr(inner));
            }

            _ => {}
        }

        narrowings
    }

    pub fn extract_all_pattern_bindings_from_expr<'a>(
        &self,
        expr: &'a Expr,
    ) -> Vec<(&'a Expr, Pattern)> {
        let mut bindings = Vec::new();
        match &expr.kind {
            ExprKind::IsPattern {
                expr: scrutinee,
                pattern,
            } => match pattern {
                Pattern::Enum {
                    bindings: pattern_bindings,
                    ..
                } if !pattern_bindings.is_empty() => {
                    bindings.push((scrutinee.as_ref(), pattern.clone()));
                }

                _ => {}
            },
            ExprKind::Binary { left, op, right } => {
                if matches!(op, BinaryOp::And) {
                    bindings.extend(self.extract_all_pattern_bindings_from_expr(left));
                    bindings.extend(self.extract_all_pattern_bindings_from_expr(right));
                }
            }

            ExprKind::Paren(inner) => {
                bindings.extend(self.extract_all_pattern_bindings_from_expr(inner));
            }

            _ => {}
        }

        bindings
    }

    pub fn bind_pattern(&mut self, pattern: &Pattern, scrutinee_type: &Type) -> Result<()> {
        match pattern {
            Pattern::Wildcard => Ok(()),
            Pattern::Identifier(name) => self
                .env
                .declare_variable(name.clone(), scrutinee_type.clone()),
            Pattern::Literal(_) => Ok(()),
            Pattern::Struct { name: _, fields: _ } => Ok(()),
            Pattern::Enum {
                enum_name: _,
                variant,
                bindings,
            } => {
                let (type_name, variant_types) = match &scrutinee_type.kind {
                    TypeKind::Named(name) => (name.clone(), None),
                    TypeKind::Option(inner) => {
                        ("Option".to_string(), Some(vec![(**inner).clone()]))
                    }

                    TypeKind::Result(ok, err) => (
                        "Result".to_string(),
                        Some(vec![(**ok).clone(), (**err).clone()]),
                    ),
                    _ => {
                        return Err(self
                            .type_error(format!("Expected enum type, got '{}'", scrutinee_type)))
                    }
                };
                let enum_def = {
                    let key = self.resolve_type_key(&type_name);
                    self.env
                        .lookup_enum(&key)
                        .or_else(|| self.env.lookup_enum(&type_name))
                }
                .ok_or_else(|| self.type_error(format!("Undefined enum '{}'", type_name)))?
                .clone();
                let variant_def = enum_def
                    .variants
                    .iter()
                    .find(|v| &v.name == variant)
                    .ok_or_else(|| {
                        self.type_error(format!(
                            "Enum '{}' has no variant '{}'",
                            type_name, variant
                        ))
                    })?;
                if let Some(variant_fields) = &variant_def.fields {
                    if bindings.len() != variant_fields.len() {
                        return Err(self.type_error(format!(
                            "Variant '{}::{}' expects {} bindings, got {}",
                            type_name,
                            variant,
                            variant_fields.len(),
                            bindings.len()
                        )));
                    }

                    for (i, (binding, field_type)) in
                        bindings.iter().zip(variant_fields.iter()).enumerate()
                    {
                        let concrete =
                            variant_types
                                .as_ref()
                                .and_then(|types| match type_name.as_str() {
                                    "Option" => {
                                        if variant == "Some" {
                                            types.get(0).cloned()
                                        } else {
                                            None
                                        }
                                    }

                                    "Result" => match variant.as_str() {
                                        "Ok" => types.get(0).cloned(),
                                        "Err" => types.get(1).cloned(),
                                        _ => types.get(i).cloned(),
                                    },
                                    _ => types.get(i).cloned(),
                                });
                        let bind_type = if let Some(concrete_type) = concrete {
                            concrete_type
                        } else if matches!(field_type.kind, TypeKind::Generic(_)) {
                            Type::new(TypeKind::Unknown, Self::dummy_span())
                        } else {
                            field_type.clone()
                        };
                        self.bind_pattern(binding, &bind_type)?;
                    }
                } else {
                    if !bindings.is_empty() {
                        return Err(self.type_error(format!(
                            "Variant '{}::{}' is a unit variant and has no bindings",
                            type_name, variant
                        )));
                    }
                }

                Ok(())
            }

            Pattern::TypeCheck(_) => Ok(()),
        }
    }
}
