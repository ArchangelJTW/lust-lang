use super::super::ShortCircuitInfo;
use super::*;
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use hashbrown::HashMap;
impl TypeChecker {
    pub fn check_literal(&self, lit: &Literal) -> Result<Type> {
        let span = Self::dummy_span();
        Ok(match lit {
            Literal::Integer(_) => Type::new(TypeKind::Int, span),
            Literal::Float(_) => Type::new(TypeKind::Float, span),
            Literal::String(_) => Type::new(TypeKind::String, span),
            Literal::Bool(_) => Type::new(TypeKind::Bool, span),
        })
    }

    pub fn check_binary_expr(
        &mut self,
        span: Span,
        left: &Expr,
        op: &BinaryOp,
        right: &Expr,
    ) -> Result<Type> {
        if matches!(op, BinaryOp::And) {
            return self.check_and_expr(span, left, right);
        }

        if matches!(op, BinaryOp::Or) {
            return self.check_or_expr(span, left, right);
        }

        let span = Self::dummy_span();
        let left_type = self.check_expr(left)?;
        let right_type = self.check_expr(right)?;
        match op {
            BinaryOp::Add
            | BinaryOp::Sub
            | BinaryOp::Mul
            | BinaryOp::Div
            | BinaryOp::Mod
            | BinaryOp::Pow => {
                if self.is_dynamic_numeric(&left_type) || self.is_dynamic_numeric(&right_type) {
                    return Ok(Type::new(TypeKind::Unknown, span));
                }
                if matches!(left_type.kind, TypeKind::Int | TypeKind::Float)
                    && matches!(right_type.kind, TypeKind::Int | TypeKind::Float)
                {
                    if matches!(left_type.kind, TypeKind::Float)
                        || matches!(right_type.kind, TypeKind::Float)
                    {
                        Ok(Type::new(TypeKind::Float, span))
                    } else {
                        Ok(Type::new(TypeKind::Int, span))
                    }
                } else {
                    Err(self.type_error_at(
                        format!(
                            "Arithmetic operator {} requires numeric types, got '{}' and '{}'",
                            op, left_type, right_type
                        ),
                        left.span,
                    ))
                }
            }

            BinaryOp::Eq
            | BinaryOp::Ne
            | BinaryOp::Lt
            | BinaryOp::Le
            | BinaryOp::Gt
            | BinaryOp::Ge => {
                if matches!(left_type.kind, TypeKind::Unknown)
                    || matches!(right_type.kind, TypeKind::Unknown)
                    || matches!(&left_type.kind, TypeKind::Named(name) if name == "LuaValue")
                    || matches!(&right_type.kind, TypeKind::Named(name) if name == "LuaValue")
                {
                    return Ok(Type::new(TypeKind::Bool, span));
                }

                if !self.types_equal(&left_type, &right_type) {
                    return Err(self.type_error(format!(
                        "Comparison requires compatible types, got '{}' and '{}'",
                        left_type, right_type
                    )));
                }

                Ok(Type::new(TypeKind::Bool, span))
            }

            BinaryOp::Concat => {
                if self.concat_operand_is_dynamic(&left_type)
                    || self.concat_operand_is_dynamic(&right_type)
                {
                    return Ok(Type::new(TypeKind::String, span));
                }

                if !self.concat_operand_implements_to_string(&left_type) {
                    return Err(self.type_error_at(
                        format!(
                            "Left operand of `..` must implement ToString trait, got '{}'",
                            left_type
                        ),
                        left.span,
                    ));
                }

                if !self.concat_operand_implements_to_string(&right_type) {
                    return Err(self.type_error_at(
                        format!(
                            "Right operand of `..` must implement ToString trait, got '{}'",
                            right_type
                        ),
                        right.span,
                    ));
                }

                Ok(Type::new(TypeKind::String, span))
            }

            BinaryOp::Range => {
                return Err(self.type_error(
                    "Range operator is not supported; use numeric for-loops".to_string(),
                ));
            }

            BinaryOp::And | BinaryOp::Or => {
                unreachable!("short-circuit operators handled earlier in check_binary_expr")
            }
        }
    }

    fn concat_operand_is_dynamic(&self, ty: &Type) -> bool {
        match &ty.kind {
            TypeKind::Unknown => true,
            TypeKind::Named(name) if name == "LuaValue" => true,
            TypeKind::Union(types) => types.iter().any(|t| self.concat_operand_is_dynamic(t)),
            _ => false,
        }
    }

    fn is_dynamic_numeric(&self, ty: &Type) -> bool {
        match &ty.kind {
            TypeKind::Unknown => true,
            TypeKind::Named(name) if name == "LuaValue" => true,
            TypeKind::Union(types) => types.iter().any(|t| self.is_dynamic_numeric(t)),
            _ => false,
        }
    }

    fn concat_operand_implements_to_string(&self, ty: &Type) -> bool {
        match &ty.kind {
            TypeKind::Union(types) => types
                .iter()
                .all(|t| self.concat_operand_implements_to_string(t)),
            _ => self.env.type_implements_trait(ty, "ToString"),
        }
    }

    fn check_and_expr(&mut self, span: Span, left: &Expr, right: &Expr) -> Result<Type> {
        let bool_type = Type::new(TypeKind::Bool, Self::dummy_span());
        let left_bindings = self.extract_all_pattern_bindings_from_expr(left);
        let left_narrowings = self.extract_type_narrowings_from_expr(left);
        let left_type = self.check_expr(left)?;
        if !left_bindings.is_empty() {
            self.unify(&bool_type, &left_type)?;
        }

        let left_info = self.short_circuit_profile(left, &left_type);

        self.env.push_scope();
        if !left_bindings.is_empty() {
            for (scrutinee, pattern) in left_bindings {
                if let Ok(scrutinee_type) = self.check_expr(scrutinee) {
                    let _ = self.bind_pattern(&pattern, &scrutinee_type);
                }
            }
        }

        for (var_name, narrowed_type) in left_narrowings {
            self.env.refine_variable_type(var_name, narrowed_type);
        }

        let right_type = self.check_expr(right)?;
        let right_bindings = self.extract_all_pattern_bindings_from_expr(right);
        if !right_bindings.is_empty() {
            self.unify(&bool_type, &right_type)?;
        }

        let right_narrowings = self.extract_type_narrowings_from_expr(right);
        for (var_name, narrowed_type) in right_narrowings {
            self.env.refine_variable_type(var_name, narrowed_type);
        }

        self.env.pop_scope();

        let right_info = self.short_circuit_profile(right, &right_type);
        let mut option_inner: Option<Type> = None;
        let should_optionize = self.should_optionize(&left_type, &right_type)
            || self.should_optionize_narrowed_value(left, right, &right_type);
        let (truthy, falsy, result_type) = if should_optionize {
            let inner = self.canonicalize_type(&right_type);
            option_inner = Some(inner.clone());
            let option_type = Type::new(TypeKind::Option(Box::new(inner)), span);
            (
                Some(option_type.clone()),
                Some(option_type.clone()),
                option_type,
            )
        } else {
            let truthy = if self.type_can_be_truthy(&left_type) {
                right_info
                    .truthy
                    .clone()
                    .or_else(|| Some(self.canonicalize_type(&right_type)))
            } else {
                None
            };

            let mut falsy_parts = Vec::new();
            if let Some(falsy) = left_info.falsy.clone() {
                falsy_parts.push(falsy);
            }

            if self.type_can_be_truthy(&left_type) {
                if let Some(falsy) = right_info.falsy.clone() {
                    falsy_parts.push(falsy);
                }
            }

            let falsy = self.merge_optional_types(falsy_parts);
            let result = self.combine_truthy_falsy(truthy.clone(), falsy.clone());
            (truthy, falsy, result)
        };

        self.record_short_circuit_info(
            span,
            &ShortCircuitInfo {
                truthy: truthy.clone(),
                falsy: falsy.clone(),
                option_inner: option_inner.clone(),
            },
        );
        Ok(result_type)
    }

    fn should_optionize_narrowed_value(
        &self,
        left: &Expr,
        right: &Expr,
        right_type: &Type,
    ) -> bool {
        if self.option_inner_type(right_type).is_some() {
            return false;
        }

        let scrutinee = match Self::extract_short_circuit_scrutinee(left) {
            Some(expr) => expr,
            None => return false,
        };

        let left_ident = Self::identifier_from_expr(scrutinee);
        let right_ident = Self::identifier_from_expr(right);
        match (left_ident, right_ident) {
            (Some(lhs), Some(rhs)) => lhs == rhs,
            _ => false,
        }
    }

    fn extract_short_circuit_scrutinee<'a>(expr: &'a Expr) -> Option<&'a Expr> {
        match &expr.kind {
            ExprKind::TypeCheck {
                expr: scrutinee, ..
            } => Some(scrutinee),
            ExprKind::IsPattern {
                expr: scrutinee, ..
            } => Some(scrutinee),
            ExprKind::Paren(inner) => Self::extract_short_circuit_scrutinee(inner),
            _ => None,
        }
    }

    fn identifier_from_expr<'a>(expr: &'a Expr) -> Option<&'a str> {
        match &expr.kind {
            ExprKind::Identifier(name) => Some(name.as_str()),
            ExprKind::Paren(inner) => Self::identifier_from_expr(inner),
            _ => None,
        }
    }

    fn check_or_expr(&mut self, span: Span, left: &Expr, right: &Expr) -> Result<Type> {
        let left_type = self.check_expr(left)?;
        let left_info = self.short_circuit_profile(left, &left_type);

        let right_type = self.check_expr(right)?;
        let right_info = self.short_circuit_profile(right, &right_type);

        let mut option_candidates: Vec<Type> = Vec::new();
        let mut option_spans: Vec<Span> = Vec::new();
        if let Some(inner) = left_info.option_inner.clone() {
            option_candidates.push(inner);
            option_spans.push(left.span);
        } else if let Some(inner) = self
            .option_inner_type(&left_type)
            .map(|ty| self.canonicalize_type(ty))
        {
            option_candidates.push(inner);
        }

        if let Some(inner) = right_info.option_inner.clone() {
            option_candidates.push(inner);
            option_spans.push(right.span);
        } else if let Some(inner) = self
            .option_inner_type(&right_type)
            .map(|ty| self.canonicalize_type(ty))
        {
            option_candidates.push(inner);
        }

        if option_candidates.len() >= 2 {
            let resolved_inner = option_candidates[0].clone();
            let mut all_compatible = true;
            for candidate in option_candidates.iter().skip(1) {
                if self
                    .unify(&resolved_inner, candidate)
                    .and_then(|_| self.unify(candidate, &resolved_inner))
                    .is_err()
                {
                    all_compatible = false;
                    break;
                }
            }

            let resolved_inner = if all_compatible {
                self.canonicalize_type(&resolved_inner)
            } else {
                self.canonicalize_type(&self.make_union_from_types(option_candidates))
            };
            let option_type = Type::new(TypeKind::Option(Box::new(resolved_inner.clone())), span);
            self.record_short_circuit_info(
                span,
                &ShortCircuitInfo {
                    truthy: Some(option_type.clone()),
                    falsy: Some(option_type.clone()),
                    option_inner: Some(resolved_inner),
                },
            );
            return Ok(option_type);
        }

        for span_to_clear in option_spans {
            self.clear_option_for_span(span_to_clear);
        }

        let mut truthy_parts: Vec<Type> = Vec::new();
        if self.type_can_be_truthy(&left_type) {
            if let Some(inner) = left_info.option_inner.clone() {
                truthy_parts.push(inner);
            } else if let Some(truthy) = left_info.truthy.clone() {
                truthy_parts.push(truthy);
            } else {
                truthy_parts.push(self.canonicalize_type(&left_type));
            }
        }

        if self.type_can_be_falsy(&left_type) && self.type_can_be_truthy(&right_type) {
            if let Some(inner) = right_info.option_inner.clone() {
                truthy_parts.push(inner);
            } else if let Some(truthy) = right_info.truthy.clone() {
                truthy_parts.push(truthy);
            } else {
                truthy_parts.push(self.canonicalize_type(&right_type));
            }
        }

        let truthy = self.merge_optional_types(truthy_parts);
        let falsy = if self.type_can_be_falsy(&left_type) {
            right_info
                .falsy
                .clone()
                .or_else(|| self.extract_falsy_type(&right_type))
        } else {
            None
        };

        let result = self.combine_truthy_falsy(truthy.clone(), falsy.clone());
        self.record_short_circuit_info(
            span,
            &ShortCircuitInfo {
                truthy,
                falsy,
                option_inner: None,
            },
        );
        Ok(result)
    }

    pub fn check_unary_expr(&mut self, op: &UnaryOp, operand: &Expr) -> Result<Type> {
        let operand_type = self.check_expr(operand)?;
        let span = Self::dummy_span();
        match op {
            UnaryOp::Neg => {
                if matches!(operand_type.kind, TypeKind::Int | TypeKind::Float) {
                    Ok(operand_type)
                } else {
                    Err(self.type_error(format!(
                        "Negation requires numeric type, got '{}'",
                        operand_type
                    )))
                }
            }

            UnaryOp::Not => {
                self.unify(&Type::new(TypeKind::Bool, span), &operand_type)?;
                Ok(Type::new(TypeKind::Bool, span))
            }
        }
    }

    pub fn check_call_expr(&mut self, span: Span, callee: &Expr, args: &[Expr]) -> Result<Type> {
        if let ExprKind::FieldAccess { object, field } = &callee.kind {
            if let ExprKind::Identifier(type_name) = &object.kind {
                let mut candidate_names: Vec<String> = Vec::new();
                if let Some(real_mod) = self.resolve_module_alias(type_name) {
                    candidate_names.push(format!("{}.{}", real_mod, field));
                }

                candidate_names.push(format!("{}.{}", type_name, field));
                let resolved_type = self.resolve_type_key(type_name);
                if resolved_type != *type_name {
                    candidate_names.push(format!("{}.{}", resolved_type, field));
                }

                let mut static_candidate: Option<(String, type_env::FunctionSignature)> = None;
                for name in candidate_names {
                    if let Some(sig) = self.env.lookup_function(&name) {
                        static_candidate = Some((name, sig.clone()));
                        break;
                    }
                }

                if let Some((resolved_name, sig)) = static_candidate {
                    let allow_varargs = sig.params.len() == 1
                        && matches!(sig.params[0].kind, TypeKind::Unknown)
                        && !args.is_empty();
                    let allow_optional = args.len() < sig.params.len()
                        && !args.is_empty()
                        && sig.params[args.len()..]
                            .iter()
                            .all(|param| matches!(param.kind, TypeKind::Unknown));
                    if args.len() > sig.params.len() && !allow_varargs
                        || (args.len() < sig.params.len() && !allow_optional && !allow_varargs)
                    {
                        return Err(self.type_error_at(
                            format!(
                                "Static method '{}' expects {} arguments, got {}",
                                resolved_name,
                                sig.params.len(),
                                args.len()
                            ),
                            span,
                        ));
                    }

                    for (i, (arg, expected_type)) in args
                        .iter()
                        .zip(sig.params.iter())
                        .take(sig.params.len().min(args.len()))
                        .enumerate()
                    {
                        let arg_type = self.check_expr(arg)?;
                        self.unify(expected_type, &arg_type).map_err(|_| {
                            self.type_error_at(
                                format!(
                                    "Argument {} to static method '{}': expected '{}', got '{}'",
                                    i + 1,
                                    resolved_name,
                                    expected_type,
                                    arg_type
                                ),
                                arg.span,
                            )
                        })?;
                    }

                    if allow_varargs || allow_optional {
                        return Ok(sig.return_type);
                    }

                    return Ok(sig.return_type);
                }

                let enum_lookup = {
                    let key = self.resolve_type_key(type_name);
                    self.env
                        .lookup_enum(&key)
                        .or_else(|| self.env.lookup_enum(type_name))
                };
                if let Some(enum_def) = enum_lookup {
                    let enum_def = enum_def.clone();
                    let variant = field;
                    let variant_def = enum_def
                        .variants
                        .iter()
                        .find(|v| &v.name == variant)
                        .ok_or_else(|| {
                            self.type_error_at(
                                format!("Enum '{}' has no variant '{}'", type_name, variant),
                                span,
                            )
                        })?;
                    if let Some(expected_fields) = &variant_def.fields {
                        if args.len() != expected_fields.len() {
                            return Err(self.type_error_at(
                                format!(
                                    "Variant '{}::{}' expects {} arguments, got {}",
                                    type_name,
                                    variant,
                                    expected_fields.len(),
                                    args.len()
                                ),
                                span,
                            ));
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

                        if type_name == "Option" {
                            if let Some(inner_type) = type_params.get("T") {
                                return Ok(Type::new(
                                    TypeKind::Option(Box::new(inner_type.clone())),
                                    Self::dummy_span(),
                                ));
                            }
                        } else if type_name == "Result" {
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

                        let enum_type_name = {
                            let key = self.resolve_type_key(type_name);
                            if self.env.lookup_enum(&key).is_some() {
                                key
                            } else {
                                type_name.clone()
                            }
                        };
                        return Ok(Type::new(
                            TypeKind::Named(enum_type_name),
                            Self::dummy_span(),
                        ));
                    } else {
                        if !args.is_empty() {
                            return Err(self.type_error(format!(
                                "Variant '{}::{}' is a unit variant and takes no arguments",
                                type_name, variant
                            )));
                        }

                        let enum_type_name = {
                            let key = self.resolve_type_key(type_name);
                            if self.env.lookup_enum(&key).is_some() {
                                key
                            } else {
                                type_name.clone()
                            }
                        };
                        return Ok(Type::new(
                            TypeKind::Named(enum_type_name),
                            Self::dummy_span(),
                        ));
                    }
                }
            }
        }

        if let ExprKind::Identifier(name) = &callee.kind {
            if let Some(var_type) = self.env.lookup_variable(name) {
                if let TypeKind::Function {
                    params: param_types,
                    return_type,
                } = &var_type.kind
                {
                    let mut expected_params = param_types.clone();
                    if args.len() != expected_params.len() {
                        if args.len() > expected_params.len() {
                            if let Some(last) = expected_params.last().cloned() {
                                let last_allows_varargs = matches!(last.kind, TypeKind::Unknown)
                                    || matches!(&last.kind, TypeKind::Named(name) if name == "LuaValue");
                                if last_allows_varargs {
                                    while expected_params.len() < args.len() {
                                        expected_params.push(last.clone());
                                    }
                                } else {
                                    return Err(self.type_error_at(
                                        format!(
                                            "Lambda '{}' expects {} arguments, got {}",
                                            name,
                                            param_types.len(),
                                            args.len()
                                        ),
                                        span,
                                    ));
                                }
                            }
                        } else {
                            let missing = &expected_params[args.len()..];
                            let missing_optional = missing.iter().all(|p| {
                                matches!(p.kind, TypeKind::Unknown)
                                    || matches!(&p.kind, TypeKind::Named(name) if name == "LuaValue")
                            });
                            if missing_optional {
                                expected_params.truncate(args.len());
                            } else {
                                return Err(self.type_error_at(
                                    format!(
                                        "Lambda '{}' expects {} arguments, got {}",
                                        name,
                                        param_types.len(),
                                        args.len()
                                    ),
                                    span,
                                ));
                            }
                        }
                    }

                    for (i, (arg, expected_type)) in args.iter().zip(expected_params.iter()).enumerate()
                    {
                        let arg_type = self.check_expr(arg)?;
                        self.unify(expected_type, &arg_type).map_err(|_| {
                            self.type_error_at(
                                format!(
                                    "Argument {} to lambda '{}': expected '{}', got '{}'",
                                    i + 1,
                                    name,
                                    expected_type,
                                    arg_type
                                ),
                                arg.span,
                            )
                        })?;
                    }

                    return Ok((**return_type).clone());
                }
            }

            let resolved = self.resolve_function_key(name);
            let sig_opt = self.env.lookup_function(&resolved).cloned();
            if sig_opt.is_none() {
                for arg in args {
                    self.check_expr(arg)?;
                }
                return Ok(Type::new(TypeKind::Unknown, span));
            }
            let sig = sig_opt.unwrap();
            let mut expected_params = sig.params.clone();
            if args.len() != expected_params.len() {
                if args.len() > expected_params.len() {
                    if let Some(last) = expected_params.last().cloned() {
                        let last_allows_varargs = matches!(last.kind, TypeKind::Unknown)
                            || matches!(&last.kind, TypeKind::Named(name) if name == "LuaValue");
                        if last_allows_varargs {
                            while expected_params.len() < args.len() {
                                expected_params.push(last.clone());
                            }
                        } else {
                            return Err(self.type_error_at(
                                format!(
                                    "Function '{}' expects {} arguments, got {}",
                                    name,
                                    sig.params.len(),
                                    args.len()
                                ),
                                callee.span,
                            ));
                        }
                    }
                } else {
                    let missing = &expected_params[args.len()..];
                    let missing_optional = missing.iter().all(|p| {
                        matches!(p.kind, TypeKind::Unknown)
                            || matches!(&p.kind, TypeKind::Named(name) if name == "LuaValue")
                    });
                    if missing_optional {
                        expected_params.truncate(args.len());
                    } else {
                        return Err(self.type_error_at(
                            format!(
                                "Function '{}' expects {} arguments, got {}",
                                name,
                                sig.params.len(),
                                args.len()
                            ),
                            callee.span,
                        ));
                    }
                }
            }

            for (i, (arg, expected_type)) in args.iter().zip(expected_params.iter()).enumerate() {
                let arg_type = self.check_expr(arg)?;
                self.unify_with_bounds(expected_type, &arg_type)
                    .map_err(|_| {
                        self.type_error_at(
                            format!(
                                "Argument {} to function '{}': expected '{}', got '{}'",
                                i + 1,
                                name,
                                expected_type,
                                arg_type
                            ),
                            arg.span,
                        )
                    })?;
            }

            Ok(sig.return_type)
        } else {
            let callee_type = self.check_expr(callee)?;
            match &callee_type.kind {
                TypeKind::Function { params, return_type } => {
                    let expected_params = params.clone();
                    if args.len() != expected_params.len() {
                        return Err(self.type_error_at(
                            format!(
                                "Function expects {} arguments, got {}",
                                expected_params.len(),
                                args.len()
                            ),
                            span,
                        ));
                    }

                    for (i, (arg, expected_type)) in args.iter().zip(expected_params.iter()).enumerate()
                    {
                        let arg_type = self.check_expr(arg)?;
                        self.unify_with_bounds(expected_type, &arg_type)
                            .map_err(|_| {
                                self.type_error_at(
                                    format!(
                                        "Argument {}: expected '{}', got '{}'",
                                        i + 1,
                                        expected_type,
                                        arg_type
                                    ),
                                    arg.span,
                                )
                            })?;
                    }

                    Ok((**return_type).clone())
                }

                TypeKind::Unknown => {
                    for arg in args {
                        self.check_expr(arg)?;
                    }
                    Ok(Type::new(TypeKind::Unknown, Self::dummy_span()))
                }

                TypeKind::Named(name) if name == "LuaValue" => {
                    for arg in args {
                        self.check_expr(arg)?;
                    }
                    Ok(Type::new(TypeKind::Unknown, Self::dummy_span()))
                }

                _ => Err(self.type_error_at(
                    format!("Cannot call expression of type '{}'", callee_type),
                    span,
                )),
            }
        }
    }

    pub fn check_method_call(
        &mut self,
        receiver: &Expr,
        method: &str,
        args: &[Expr],
    ) -> Result<Type> {
        let receiver_type = self.check_expr(receiver)?;
        let span = Self::dummy_span();
        if matches!(receiver_type.kind, TypeKind::Unknown)
            || matches!(&receiver_type.kind, TypeKind::Named(name) if name == "LuaValue")
        {
            for arg in args {
                self.check_expr(arg)?;
            }
            return Ok(Type::new(TypeKind::Unknown, span));
        }
        match &receiver_type.kind {
            TypeKind::String => match method {
                "len" => {
                    if !args.is_empty() {
                        return Err(self.type_error("len() takes no arguments".to_string()));
                    }

                    return Ok(Type::new(TypeKind::Int, span));
                }

                "substring" => {
                    if args.len() != 2 {
                        return Err(self.type_error("substring() requires 2 arguments".to_string()));
                    }

                    self.check_expr(&args[0])?;
                    self.check_expr(&args[1])?;
                    return Ok(Type::new(TypeKind::String, span));
                }

                "find" => {
                    if args.len() != 1 {
                        return Err(self.type_error("find() requires 1 argument".to_string()));
                    }

                    self.check_expr(&args[0])?;
                    return Ok(Type::new(TypeKind::Named("Option".to_string()), span));
                }

                "starts_with" | "ends_with" | "contains" => {
                    if args.len() != 1 {
                        return Err(self.type_error(format!("{}() requires 1 argument", method)));
                    }

                    self.check_expr(&args[0])?;
                    return Ok(Type::new(TypeKind::Bool, span));
                }

                "split" => {
                    if args.len() != 1 {
                        return Err(self.type_error("split() requires 1 argument".to_string()));
                    }

                    self.check_expr(&args[0])?;
                    return Ok(Type::new(
                        TypeKind::Array(Box::new(Type::new(TypeKind::String, span))),
                        span,
                    ));
                }

                "trim" | "trim_start" | "trim_end" | "to_upper" | "to_lower" => {
                    if !args.is_empty() {
                        return Err(self.type_error(format!("{}() takes no arguments", method)));
                    }

                    return Ok(Type::new(TypeKind::String, span));
                }

                "replace" => {
                    if args.len() != 2 {
                        return Err(self.type_error("replace() requires 2 arguments".to_string()));
                    }

                    self.check_expr(&args[0])?;
                    self.check_expr(&args[1])?;
                    return Ok(Type::new(TypeKind::String, span));
                }

                "is_empty" => {
                    if !args.is_empty() {
                        return Err(self.type_error("is_empty() takes no arguments".to_string()));
                    }

                    return Ok(Type::new(TypeKind::Bool, span));
                }

                "chars" | "lines" => {
                    if !args.is_empty() {
                        return Err(self.type_error(format!("{}() takes no arguments", method)));
                    }

                    return Ok(Type::new(
                        TypeKind::Array(Box::new(Type::new(TypeKind::String, span))),
                        span,
                    ));
                }

                _ => {}
            },
            TypeKind::Array(elem_type) => match method {
                "len" => {
                    if !args.is_empty() {
                        return Err(self.type_error("len() takes no arguments".to_string()));
                    }

                    return Ok(Type::new(TypeKind::Int, span));
                }

                "get" => {
                    if args.len() != 1 {
                        return Err(self.type_error("get() requires 1 argument".to_string()));
                    }

                    self.check_expr(&args[0])?;
                    return Ok(Type::new(TypeKind::Option(elem_type.clone()), span));
                }

                "first" | "last" => {
                    if !args.is_empty() {
                        return Err(self.type_error(format!("{}() takes no arguments", method)));
                    }

                    return Ok(Type::new(TypeKind::Option(elem_type.clone()), span));
                }

                "push" => {
                    if args.len() != 1 {
                        return Err(self.type_error("push() requires 1 argument".to_string()));
                    }

                    self.check_expr(&args[0])?;
                    return Ok(Type::new(TypeKind::Unit, span));
                }

                "pop" => {
                    if !args.is_empty() {
                        return Err(self.type_error("pop() takes no arguments".to_string()));
                    }

                    return Ok(Type::new(TypeKind::Option(elem_type.clone()), span));
                }

                "iter" => {
                    if !args.is_empty() {
                        return Err(self.type_error("iter() takes no arguments".to_string()));
                    }

                    return Ok(Type::new(TypeKind::Named("Iterator".to_string()), span));
                }

                "slice" => {
                    if args.len() != 2 {
                        return Err(self.type_error("slice() requires 2 arguments".to_string()));
                    }

                    self.check_expr(&args[0])?;
                    self.check_expr(&args[1])?;
                    return Ok(Type::new(TypeKind::Array(elem_type.clone()), span));
                }

                "clear" => {
                    if !args.is_empty() {
                        return Err(self.type_error("clear() takes no arguments".to_string()));
                    }

                    return Ok(Type::new(TypeKind::Unit, span));
                }

                "is_empty" => {
                    if !args.is_empty() {
                        return Err(self.type_error("is_empty() takes no arguments".to_string()));
                    }

                    return Ok(Type::new(TypeKind::Bool, span));
                }

                "map" => {
                    if args.len() != 1 {
                        return Err(
                            self.type_error("map() requires 1 argument (function)".to_string())
                        );
                    }

                    self.expected_lambda_signature = Some((vec![(**elem_type).clone()], None));
                    let func_type = self.check_expr(&args[0])?;
                    match &func_type.kind {
                        TypeKind::Function {
                            params: _,
                            return_type,
                        } => {
                            return Ok(Type::new(TypeKind::Array(return_type.clone()), span));
                        }

                        _ => {
                            return Err(
                                self.type_error("map() requires a function argument".to_string())
                            );
                        }
                    }
                }

                "filter" => {
                    if args.len() != 1 {
                        return Err(
                            self.type_error("filter() requires 1 argument (function)".to_string())
                        );
                    }

                    self.expected_lambda_signature = Some((
                        vec![(**elem_type).clone()],
                        Some(Type::new(TypeKind::Bool, span)),
                    ));
                    let func_type = self.check_expr(&args[0])?;
                    match &func_type.kind {
                        TypeKind::Function {
                            params: _,
                            return_type,
                        } => {
                            self.unify(&Type::new(TypeKind::Bool, span), return_type)?;
                            return Ok(Type::new(TypeKind::Array(elem_type.clone()), span));
                        }

                        _ => {
                            return Err(self
                                .type_error("filter() requires a function argument".to_string()));
                        }
                    }
                }

                "reduce" => {
                    if args.len() != 2 {
                        return Err(self.type_error(
                            "reduce() requires 2 arguments (initial value and function)"
                                .to_string(),
                        ));
                    }

                    let init_type = self.check_expr(&args[0])?;
                    self.expected_lambda_signature = Some((
                        vec![init_type.clone(), (**elem_type).clone()],
                        Some(init_type.clone()),
                    ));
                    let func_type = self.check_expr(&args[1])?;
                    match &func_type.kind {
                        TypeKind::Function {
                            params: _,
                            return_type,
                        } => {
                            self.unify(&init_type, return_type)?;
                            return Ok(init_type);
                        }

                        _ => {
                            return Err(self.type_error(
                                "reduce() requires a function as second argument".to_string(),
                            ));
                        }
                    }
                }

                _ => {}
            },
            TypeKind::Map(key_type, value_type) => match method {
                "iter" => {
                    if !args.is_empty() {
                        return Err(self.type_error("iter() takes no arguments".to_string()));
                    }

                    return Ok(Type::new(TypeKind::Named("Iterator".to_string()), span));
                }

                "len" => {
                    if !args.is_empty() {
                        return Err(self.type_error("len() takes no arguments".to_string()));
                    }

                    return Ok(Type::new(TypeKind::Int, span));
                }

                "get" => {
                    if args.len() != 1 {
                        return Err(self.type_error("get() requires 1 argument (key)".to_string()));
                    }

                    let arg_type = self.check_expr(&args[0])?;
                    self.unify(key_type, &arg_type)?;
                    return Ok(Type::new(TypeKind::Option(value_type.clone()), span));
                }

                "set" => {
                    if args.len() != 2 {
                        return Err(
                            self.type_error("set() requires 2 arguments (key, value)".to_string())
                        );
                    }

                    let key_arg_type = self.check_expr(&args[0])?;
                    let value_arg_type = self.check_expr(&args[1])?;
                    self.unify(key_type, &key_arg_type)?;
                    self.unify(value_type, &value_arg_type)?;
                    return Ok(Type::new(TypeKind::Unit, span));
                }

                "has" => {
                    if args.len() != 1 {
                        return Err(self.type_error("has() requires 1 argument (key)".to_string()));
                    }

                    let arg_type = self.check_expr(&args[0])?;
                    self.unify(key_type, &arg_type)?;
                    return Ok(Type::new(TypeKind::Bool, span));
                }

                "delete" => {
                    if args.len() != 1 {
                        return Err(
                            self.type_error("delete() requires 1 argument (key)".to_string())
                        );
                    }

                    let arg_type = self.check_expr(&args[0])?;
                    self.unify(key_type, &arg_type)?;
                    return Ok(Type::new(TypeKind::Option(value_type.clone()), span));
                }

                "keys" => {
                    if !args.is_empty() {
                        return Err(self.type_error("keys() takes no arguments".to_string()));
                    }

                    return Ok(Type::new(TypeKind::Array(key_type.clone()), span));
                }

                "values" => {
                    if !args.is_empty() {
                        return Err(self.type_error("values() takes no arguments".to_string()));
                    }

                    return Ok(Type::new(TypeKind::Array(value_type.clone()), span));
                }

                _ => {}
            },
            TypeKind::Named(type_name) if type_name == "Array" => match method {
                "len" => {
                    if !args.is_empty() {
                        return Err(self.type_error("len() takes no arguments".to_string()));
                    }

                    return Ok(Type::new(TypeKind::Int, span));
                }

                "get" => {
                    if args.len() != 1 {
                        return Err(self.type_error("get() requires 1 argument".to_string()));
                    }

                    self.check_expr(&args[0])?;
                    return Ok(Type::new(TypeKind::Named("Option".to_string()), span));
                }

                "first" | "last" => {
                    if !args.is_empty() {
                        return Err(self.type_error(format!("{}() takes no arguments", method)));
                    }

                    return Ok(Type::new(TypeKind::Named("Option".to_string()), span));
                }

                "push" => {
                    if args.len() != 1 {
                        return Err(self.type_error("push() requires 1 argument".to_string()));
                    }

                    self.check_expr(&args[0])?;
                    return Ok(Type::new(TypeKind::Unit, span));
                }

                "pop" => {
                    if !args.is_empty() {
                        return Err(self.type_error("pop() takes no arguments".to_string()));
                    }

                    return Ok(Type::new(TypeKind::Named("Option".to_string()), span));
                }

                "slice" => {
                    if args.len() != 2 {
                        return Err(self.type_error("slice() requires 2 arguments".to_string()));
                    }

                    self.check_expr(&args[0])?;
                    self.check_expr(&args[1])?;
                    return Ok(receiver_type.clone());
                }

                "clear" => {
                    if !args.is_empty() {
                        return Err(self.type_error("clear() takes no arguments".to_string()));
                    }

                    return Ok(Type::new(TypeKind::Unit, span));
                }

                "is_empty" => {
                    if !args.is_empty() {
                        return Err(self.type_error("is_empty() takes no arguments".to_string()));
                    }

                    return Ok(Type::new(TypeKind::Bool, span));
                }

                _ => {}
            },
            TypeKind::Option(inner_type) => match method {
                "is_some" | "is_none" => {
                    if !args.is_empty() {
                        return Err(self.type_error(format!("{}() takes no arguments", method)));
                    }

                    return Ok(Type::new(TypeKind::Bool, span));
                }

                "unwrap" => {
                    if !args.is_empty() {
                        return Err(self.type_error("unwrap() takes no arguments".to_string()));
                    }

                    return Ok((**inner_type).clone());
                }

                "unwrap_or" => {
                    if args.len() != 1 {
                        return Err(self.type_error("unwrap_or() requires 1 argument".to_string()));
                    }

                    let default_type = self.check_expr(&args[0])?;
                    return Ok(default_type);
                }

                _ => {}
            },
            TypeKind::Result(ok_type, _err_type) => match method {
                "is_ok" | "is_err" => {
                    if !args.is_empty() {
                        return Err(self.type_error(format!("{}() takes no arguments", method)));
                    }

                    return Ok(Type::new(TypeKind::Bool, span));
                }

                "unwrap" => {
                    if !args.is_empty() {
                        return Err(self.type_error("unwrap() takes no arguments".to_string()));
                    }

                    return Ok((**ok_type).clone());
                }

                "unwrap_or" => {
                    if args.len() != 1 {
                        return Err(self.type_error("unwrap_or() requires 1 argument".to_string()));
                    }

                    let default_type = self.check_expr(&args[0])?;
                    return Ok(default_type);
                }

                _ => {}
            },
            TypeKind::Named(type_name) if type_name == "Option" || type_name == "Result" => {
                match method {
                    "is_some" | "is_none" | "is_ok" | "is_err" => {
                        if !args.is_empty() {
                            return Err(self.type_error(format!("{}() takes no arguments", method)));
                        }

                        return Ok(Type::new(TypeKind::Bool, span));
                    }

                    "unwrap" => {
                        if !args.is_empty() {
                            return Err(self.type_error("unwrap() takes no arguments".to_string()));
                        }

                        if let ExprKind::Identifier(var_name) = &receiver.kind {
                            if let Some(concrete_type) =
                                self.env.lookup_generic_param(var_name, "T")
                            {
                                return Ok(concrete_type);
                            }
                        }

                        return Ok(Type::new(TypeKind::Unknown, span));
                    }

                    "unwrap_or" => {
                        if args.len() != 1 {
                            return Err(
                                self.type_error("unwrap_or() requires 1 argument".to_string())
                            );
                        }

                        let default_type = self.check_expr(&args[0])?;
                        return Ok(default_type);
                    }

                    _ => {}
                }
            }

            TypeKind::Float => match method {
                "to_int" => {
                    if !args.is_empty() {
                        return Err(self.type_error("to_int() takes no arguments".to_string()));
                    }

                    return Ok(Type::new(TypeKind::Int, span));
                }

                "floor" | "ceil" | "round" | "sqrt" | "abs" | "sin" | "cos" | "tan" | "asin"
                | "acos" | "atan" => {
                    if !args.is_empty() {
                        return Err(self.type_error(format!("{}() takes no arguments", method)));
                    }

                    return Ok(Type::new(TypeKind::Float, span));
                }

                "atan2" => {
                    if args.len() != 1 {
                        return Err(self.type_error("atan2() requires 1 argument".to_string()));
                    }

                    let other_type = self.check_expr(&args[0])?;
                    self.unify(&Type::new(TypeKind::Float, span), &other_type)?;
                    return Ok(Type::new(TypeKind::Float, span));
                }

                "min" | "max" => {
                    if args.len() != 1 {
                        return Err(self.type_error(format!("{}() requires 1 argument", method)));
                    }

                    self.check_expr(&args[0])?;
                    return Ok(Type::new(TypeKind::Float, span));
                }

                "clamp" => {
                    if args.len() != 2 {
                        return Err(
                            self.type_error("clamp() requires 2 arguments (min, max)".to_string())
                        );
                    }

                    let min_type = self.check_expr(&args[0])?;
                    let max_type = self.check_expr(&args[1])?;
                    self.unify(&Type::new(TypeKind::Float, span), &min_type)?;
                    self.unify(&Type::new(TypeKind::Float, span), &max_type)?;
                    return Ok(Type::new(TypeKind::Float, span));
                }

                _ => {}
            },
            TypeKind::Int => match method {
                "to_float" => {
                    if !args.is_empty() {
                        return Err(self.type_error("to_float() takes no arguments".to_string()));
                    }

                    return Ok(Type::new(TypeKind::Float, span));
                }

                "abs" => {
                    if !args.is_empty() {
                        return Err(self.type_error("abs() takes no arguments".to_string()));
                    }

                    return Ok(Type::new(TypeKind::Int, span));
                }

                "min" | "max" => {
                    if args.len() != 1 {
                        return Err(self.type_error(format!("{}() requires 1 argument", method)));
                    }

                    self.check_expr(&args[0])?;
                    return Ok(Type::new(TypeKind::Int, span));
                }

                "clamp" => {
                    if args.len() != 2 {
                        return Err(
                            self.type_error("clamp() requires 2 arguments (min, max)".to_string())
                        );
                    }

                    let min_type = self.check_expr(&args[0])?;
                    let max_type = self.check_expr(&args[1])?;
                    self.unify(&Type::new(TypeKind::Int, span), &min_type)?;
                    self.unify(&Type::new(TypeKind::Int, span), &max_type)?;
                    return Ok(Type::new(TypeKind::Int, span));
                }

                _ => {}
            },
            TypeKind::Named(type_name) if type_name == "LuaTable" => match method {
                "len" | "maxn" => {
                    if !args.is_empty() {
                        return Err(self.type_error(format!("{}() takes no arguments", method)));
                    }

                    return Ok(Type::new(TypeKind::Int, span));
                }
                "push" => {
                    if args.len() != 1 {
                        return Err(self.type_error("push() requires 1 argument".to_string()));
                    }

                    self.check_expr(&args[0])?;
                    return Ok(Type::new(TypeKind::Unit, span));
                }
                "insert" => {
                    if args.len() != 2 {
                        return Err(self.type_error("insert() requires 2 arguments".to_string()));
                    }

                    self.check_expr(&args[0])?;
                    self.check_expr(&args[1])?;
                    return Ok(Type::new(TypeKind::Unit, span));
                }
                "remove" => {
                    if args.len() != 1 {
                        return Err(self.type_error("remove() requires 1 argument".to_string()));
                    }

                    self.check_expr(&args[0])?;
                    return Ok(Type::new(TypeKind::Unknown, span));
                }
                "concat" => {
                    if args.len() != 3 {
                        return Err(self.type_error("concat() requires 3 arguments".to_string()));
                    }

                    self.check_expr(&args[0])?;
                    self.check_expr(&args[1])?;
                    self.check_expr(&args[2])?;
                    return Ok(Type::new(TypeKind::String, span));
                }
                "unpack" => {
                    if args.len() != 2 {
                        return Err(self.type_error("unpack() requires 2 arguments".to_string()));
                    }

                    self.check_expr(&args[0])?;
                    self.check_expr(&args[1])?;
                    return Ok(Type::new(TypeKind::Unknown, span));
                }
                "sort" => {
                    if args.len() != 1 {
                        return Err(self.type_error("sort() requires 1 argument".to_string()));
                    }

                    self.check_expr(&args[0])?;
                    return Ok(Type::new(TypeKind::Unit, span));
                }
                _ => {
                    return Err(self.type_error(format!(
                        "LuaTable has no method '{}'",
                        method
                    )));
                }
            },
            TypeKind::Named(type_name) => {
                if let Some(method_def) = self.env.lookup_method(type_name.as_str(), method) {
                    let method_def = method_def.clone();
                    let expected_args = method_def.params.len().saturating_sub(1);
                    if args.len() != expected_args {
                        return Err(self.type_error(format!(
                            "Method '{}' expects {} arguments, got {}",
                            method,
                            expected_args,
                            args.len()
                        )));
                    }

                    for (i, (arg, param)) in args
                        .iter()
                        .zip(method_def.params.iter().skip(1))
                        .enumerate()
                    {
                        let arg_type = self.check_expr(arg)?;
                        if !self.types_equal(&arg_type, &param.ty) {
                            return Err(self.type_error(format!(
                                "Argument {} to method '{}': expected '{}', got '{}'",
                                i + 1,
                                method,
                                param.ty,
                                arg_type
                            )));
                        }
                    }

                    return Ok(method_def
                        .return_type
                        .clone()
                        .unwrap_or(Type::new(TypeKind::Unit, span)));
                }
            }

            TypeKind::Generic(type_param) => {
                if let Some(trait_names) = self.current_trait_bounds.get(type_param.as_str()) {
                    for trait_name in trait_names {
                        if let Some(trait_def) = {
                            let key = self.resolve_type_key(trait_name.as_str());
                            self.env
                                .lookup_trait(&key)
                                .or_else(|| self.env.lookup_trait(trait_name.as_str()))
                        } {
                            if let Some(trait_method) =
                                trait_def.methods.iter().find(|m| m.name == method)
                            {
                                let expected_args =
                                    trait_method.params.iter().filter(|p| !p.is_self).count();
                                if args.len() != expected_args {
                                    return Err(self.type_error(format!(
                                        "Method '{}' expects {} arguments, got {}",
                                        method,
                                        expected_args,
                                        args.len()
                                    )));
                                }

                                return Ok(trait_method
                                    .return_type
                                    .clone()
                                    .unwrap_or(Type::new(TypeKind::Unit, span)));
                            }
                        }
                    }
                }
            }

            TypeKind::Trait(trait_name) => {
                if let Some(trait_def) = {
                    let key = self.resolve_type_key(trait_name);
                    self.env
                        .lookup_trait(&key)
                        .or_else(|| self.env.lookup_trait(trait_name.as_str()))
                } {
                    if let Some(trait_method) = trait_def.methods.iter().find(|m| m.name == method)
                    {
                        let expected_args =
                            trait_method.params.iter().filter(|p| !p.is_self).count();
                        if args.len() != expected_args {
                            return Err(self.type_error(format!(
                                "Method '{}' expects {} arguments, got {}",
                                method,
                                expected_args,
                                args.len()
                            )));
                        }

                        return Ok(trait_method
                            .return_type
                            .clone()
                            .unwrap_or(Type::new(TypeKind::Unit, span)));
                    }
                }
            }

            _ => {}
        }

        Err(self.type_error(format!(
            "Type '{}' has no method '{}'",
            receiver_type, method
        )))
    }

    pub fn check_field_access_with_hint(
        &mut self,
        span: Span,
        object: &Expr,
        field: &str,
        expected_type: Option<&Type>,
    ) -> Result<Type> {
        if let ExprKind::Identifier(enum_name) = &object.kind {
            if let Some(enum_def) = {
                let key = self.resolve_type_key(enum_name);
                self.env
                    .lookup_enum(&key)
                    .or_else(|| self.env.lookup_enum(enum_name))
            } {
                let enum_def = enum_def.clone();
                let variant_def = enum_def
                    .variants
                    .iter()
                    .find(|v| &v.name == field)
                    .ok_or_else(|| {
                        self.type_error_at(
                            format!("Enum '{}' has no variant '{}'", enum_name, field),
                            span,
                        )
                    })?;
                if variant_def.fields.is_some() {
                    return Err(self.type_error_at(
                        format!(
                            "Variant '{}::{}' has fields and must be called with arguments",
                            enum_name, field
                        ),
                        span,
                    ));
                }

                if let Some(expected) = expected_type {
                    match &expected.kind {
                        TypeKind::Option(_) if enum_name == "Option" => {
                            return Ok(expected.clone());
                        }

                        TypeKind::Result(_, _) if enum_name == "Result" => {
                            return Ok(expected.clone());
                        }

                        _ => {}
                    }
                }

                return Ok(Type::new(TypeKind::Named(enum_name.clone()), span));
            }
        }

        let object_type = self.check_expr(object)?;
        if matches!(object_type.kind, TypeKind::Unknown)
            || matches!(&object_type.kind, TypeKind::Named(name) if name == "LuaValue" || name == "LuaTable")
        {
            return Ok(Type::new(TypeKind::Unknown, span));
        }
        if let TypeKind::Union(types) = &object_type.kind {
            if types.iter().any(|t| {
                matches!(t.kind, TypeKind::Unknown)
                    || matches!(&t.kind, TypeKind::Named(name) if name == "LuaValue" || name == "LuaTable")
            }) {
                return Ok(Type::new(TypeKind::Unknown, span));
            }
        }
        if let TypeKind::Map(_, value_type) = &object_type.kind {
            return Ok(value_type.as_ref().clone());
        }
        let type_name = match &object_type.kind {
            TypeKind::Named(name) => name.clone(),
            _ => {
                return Err(self.type_error_at(
                    format!("Cannot access field on type '{}'", object_type),
                    object.span,
                ))
            }
        };
        self.env
            .lookup_struct_field(&type_name, field)
            .ok_or_else(|| {
                self.type_error_at(
                    format!("Type '{}' has no field '{}'", type_name, field),
                    span,
                )
            })
    }

    pub fn check_index_expr(&mut self, object: &Expr, index: &Expr) -> Result<Type> {
        let object_type = self.check_expr(object)?;
        let index_type = self.check_expr(index)?;
        if matches!(object_type.kind, TypeKind::Unknown)
            || matches!(&object_type.kind, TypeKind::Named(name) if name == "LuaValue" || name == "LuaTable")
        {
            return Ok(Type::new(TypeKind::Unknown, Self::dummy_span()));
        }
        if let TypeKind::Union(types) = &object_type.kind {
            if types.iter().any(|t| {
                matches!(t.kind, TypeKind::Unknown)
                    || matches!(&t.kind, TypeKind::Named(name) if name == "LuaValue" || name == "LuaTable")
            }) {
                return Ok(Type::new(TypeKind::Unknown, Self::dummy_span()));
            }
        }
        match &object_type.kind {
            TypeKind::Array(elem_type) => {
                self.unify(&Type::new(TypeKind::Int, Self::dummy_span()), &index_type)?;
                Ok(elem_type.as_ref().clone())
            }

            TypeKind::Map(key_type, value_type) => {
                self.unify(key_type.as_ref(), &index_type)?;
                Ok(value_type.as_ref().clone())
            }

            _ => Err(self.type_error(format!("Cannot index type '{}'", object_type))),
        }
    }
}
