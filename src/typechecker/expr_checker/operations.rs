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

    pub fn check_binary_expr(&mut self, left: &Expr, op: &BinaryOp, right: &Expr) -> Result<Type> {
        let span = Self::dummy_span();
        if matches!(op, BinaryOp::And) {
            let mut pattern_bindings = Vec::new();
            pattern_bindings.extend(self.extract_all_pattern_bindings_from_expr(left));
            pattern_bindings.extend(self.extract_all_pattern_bindings_from_expr(right));
            if !pattern_bindings.is_empty() {
                let left_type = self.check_expr(left)?;
                self.unify(&Type::new(TypeKind::Bool, span), &left_type)?;
                self.env.push_scope();
                for (scrutinee, pattern) in pattern_bindings {
                    if let Ok(scrutinee_type) = self.check_expr(scrutinee) {
                        let _ = self.bind_pattern(&pattern, &scrutinee_type);
                    }
                }

                let right_narrowings = self.extract_type_narrowings_from_expr(right);
                for (var_name, narrowed_type) in right_narrowings {
                    self.env.refine_variable_type(var_name, narrowed_type);
                }

                let right_type = self.check_expr(right)?;
                self.unify(&Type::new(TypeKind::Bool, span), &right_type)?;
                self.env.pop_scope();
                return Ok(Type::new(TypeKind::Bool, span));
            }
        }

        let left_type = self.check_expr(left)?;
        let right_type = self.check_expr(right)?;
        match op {
            BinaryOp::Add
            | BinaryOp::Sub
            | BinaryOp::Mul
            | BinaryOp::Div
            | BinaryOp::Mod
            | BinaryOp::Pow => {
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

            BinaryOp::And | BinaryOp::Or => {
                self.unify(&Type::new(TypeKind::Bool, span), &left_type)?;
                self.unify(&Type::new(TypeKind::Bool, span), &right_type)?;
                Ok(Type::new(TypeKind::Bool, span))
            }

            BinaryOp::Concat => {
                if !self.env.type_implements_trait(&left_type, "ToString") {
                    return Err(self.type_error_at(
                        format!(
                            "Left operand of `..` must implement ToString trait, got '{}'",
                            left_type
                        ),
                        left.span,
                    ));
                }

                if !self.env.type_implements_trait(&right_type, "ToString") {
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
        }
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
                    if args.len() != sig.params.len() {
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

                    for (i, (arg, expected_type)) in args.iter().zip(sig.params.iter()).enumerate()
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
                    if args.len() != param_types.len() {
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

                    for (i, (arg, expected_type)) in args.iter().zip(param_types.iter()).enumerate()
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
            let sig = self
                .env
                .lookup_function(&resolved)
                .ok_or_else(|| {
                    self.type_error_at(format!("Undefined function '{}'", name), callee.span)
                })?
                .clone();
            if args.len() != sig.params.len() {
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

            for (i, (arg, expected_type)) in args.iter().zip(sig.params.iter()).enumerate() {
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
            Err(self.type_error_at(
                "Only direct function/lambda calls are supported".to_string(),
                span,
            ))
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
            TypeKind::Table => match method {
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

                    self.check_expr(&args[0])?;
                    return Ok(Type::new(
                        TypeKind::Option(Box::new(Type::new(TypeKind::Unknown, span))),
                        span,
                    ));
                }

                "set" => {
                    if args.len() != 2 {
                        return Err(
                            self.type_error("set() requires 2 arguments (key, value)".to_string())
                        );
                    }

                    self.check_expr(&args[0])?;
                    self.check_expr(&args[1])?;
                    return Ok(Type::new(TypeKind::Unit, span));
                }

                "has" => {
                    if args.len() != 1 {
                        return Err(self.type_error("has() requires 1 argument (key)".to_string()));
                    }

                    self.check_expr(&args[0])?;
                    return Ok(Type::new(TypeKind::Bool, span));
                }

                "delete" => {
                    if args.len() != 1 {
                        return Err(
                            self.type_error("delete() requires 1 argument (key)".to_string())
                        );
                    }

                    self.check_expr(&args[0])?;
                    return Ok(Type::new(
                        TypeKind::Option(Box::new(Type::new(TypeKind::Unknown, span))),
                        span,
                    ));
                }

                "keys" => {
                    if !args.is_empty() {
                        return Err(self.type_error("keys() takes no arguments".to_string()));
                    }

                    return Ok(Type::new(
                        TypeKind::Array(Box::new(Type::new(TypeKind::Unknown, span))),
                        span,
                    ));
                }

                "values" => {
                    if !args.is_empty() {
                        return Err(self.type_error("values() takes no arguments".to_string()));
                    }

                    return Ok(Type::new(
                        TypeKind::Array(Box::new(Type::new(TypeKind::Unknown, span))),
                        span,
                    ));
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

                "floor" | "ceil" | "round" | "sqrt" | "abs" => {
                    if !args.is_empty() {
                        return Err(self.type_error(format!("{}() takes no arguments", method)));
                    }

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
        match &object_type.kind {
            TypeKind::Array(elem_type) => {
                self.unify(&Type::new(TypeKind::Int, Self::dummy_span()), &index_type)?;
                Ok(elem_type.as_ref().clone())
            }

            TypeKind::Map(key_type, value_type) => {
                self.unify(key_type.as_ref(), &index_type)?;
                Ok(Type::new(
                    TypeKind::Option(Box::new(value_type.as_ref().clone())),
                    Self::dummy_span(),
                ))
            }

            TypeKind::Table => {
                self.check_expr(index)?;
                Ok(Type::new(TypeKind::Unknown, Self::dummy_span()))
            }

            _ => Err(self.type_error(format!("Cannot index type '{}'", object_type))),
        }
    }
}
