use super::Parser;
use crate::{
    ast::{
        BinaryOp, Expr, ExprKind, Literal, Pattern, Span, StructLiteralField, Type, TypeKind,
        UnaryOp,
    },
    error::{LustError, Result},
    lexer::{Token, TokenKind},
    number::{parse_float_literal, parse_integer_literal},
};
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::{matches, unreachable};

fn is_placeholder(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::Identifier(name) => name == "_",
        ExprKind::Paren(inner) => is_placeholder(inner),
        _ => false,
    }
}

impl Parser {
    fn span_through_previous(&self, start: Span) -> Span {
        let end = &self.tokens[self.current - 1];
        Span::new(start.start_line, start.start_col, end.line, end.column)
    }

    #[inline(never)]
    pub(super) fn parse_expr(&mut self) -> Result<Expr> {
        #[cfg(feature = "esp32c6-logging")]
        log::info!(
            "        parse_expr: {:?} at line {}",
            self.peek_kind(),
            self.current_token().line
        );

        self.parse_assignment()
    }

    #[inline(never)]
    fn parse_assignment(&mut self) -> Result<Expr> {
        self.parse_pipe()
    }

    #[inline(never)]
    fn parse_pipe(&mut self) -> Result<Expr> {
        let mut expr = self.parse_logical_or()?;
        while self.match_token(&[TokenKind::PipeGreater]) {
            let right = self.parse_logical_or()?;
            expr = self.desugar_pipe_step(expr, right)?;
        }

        self.desugar_currying(expr)
    }

    fn desugar_pipe_step(&mut self, expr: Expr, right: Expr) -> Result<Expr> {
        let span = self.span_through_previous(expr.span);
        if is_placeholder(&expr) {
            return Err(self.error("Placeholder '_' cannot be used as pipe source expression"));
        }
        if is_placeholder(&right) {
            return Err(self.error("Placeholder '_' cannot be used as pipe target without a call"));
        }

        let expr = self.desugar_currying(expr)?;

        let is_unwrap = matches!(
            &right.kind,
            ExprKind::MethodCall {
                method,
                args,
                ..
            } if method == "try_unwrap" && args.is_empty()
        );

        if is_unwrap {
            let ExprKind::MethodCall { receiver, .. } = right.kind else {
                unreachable!()
            };
            let inner_target = *receiver;
            let piped_inner = self.pipe_into_target(expr, inner_target, span)?;
            Ok(Expr::new(
                ExprKind::MethodCall {
                    receiver: Box::new(piped_inner),
                    method: "try_unwrap".to_string(),
                    type_args: None,
                    args: vec![],
                },
                span,
            ))
        } else if let ExprKind::Paren(inner) = right.kind {
            let piped_inner = self.desugar_pipe_step(expr, *inner)?;
            Ok(Expr::new(ExprKind::Paren(Box::new(piped_inner)), span))
        } else {
            self.pipe_into_target(expr, right, span)
        }
    }

    fn pipe_into_target(&mut self, expr: Expr, target: Expr, span: Span) -> Result<Expr> {
        if is_placeholder(&target) {
            return Err(self.error("Placeholder '_' cannot be used as pipe target without a call"));
        }

        if matches!(
            &target.kind,
            ExprKind::MethodCall {
                method,
                args,
                ..
            } if method == "try_unwrap" && args.is_empty()
        ) {
            let ExprKind::MethodCall { receiver, .. } = target.kind else {
                unreachable!()
            };
            let piped_inner = self.pipe_into_target(expr, *receiver, span)?;
            return Ok(Expr::new(
                ExprKind::MethodCall {
                    receiver: Box::new(piped_inner),
                    method: "try_unwrap".to_string(),
                    type_args: None,
                    args: vec![],
                },
                span,
            ));
        }

        if let ExprKind::Paren(inner) = target.kind {
            let piped_inner = self.pipe_into_target(expr, *inner, span)?;
            return Ok(Expr::new(ExprKind::Paren(Box::new(piped_inner)), span));
        }

        match target.kind {
            ExprKind::Call {
                callee,
                type_args,
                mut args,
            } => {
                let placeholder_indices: Vec<usize> = args
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| is_placeholder(a))
                    .map(|(i, _)| i)
                    .collect();

                if placeholder_indices.is_empty() {
                    args.insert(0, expr);
                } else if placeholder_indices.len() == 1 {
                    args[placeholder_indices[0]] = expr;
                } else {
                    return Err(self.error("Multiple placeholders '_' not allowed in a single pipe expression"));
                }

                Ok(Expr::new(
                    ExprKind::Call {
                        callee,
                        type_args,
                        args,
                    },
                    span,
                ))
            }

            ExprKind::MethodCall {
                receiver,
                method,
                type_args,
                mut args,
            } => {
                let placeholder_indices: Vec<usize> = args
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| is_placeholder(a))
                    .map(|(i, _)| i)
                    .collect();

                if placeholder_indices.is_empty() {
                    args.insert(0, expr);
                } else if placeholder_indices.len() == 1 {
                    args[placeholder_indices[0]] = expr;
                } else {
                    return Err(self.error("Multiple placeholders '_' not allowed in a single pipe expression"));
                }

                Ok(Expr::new(
                    ExprKind::MethodCall {
                        receiver,
                        method,
                        type_args,
                        args,
                    },
                    span,
                ))
            }

            _ => Ok(Expr::new(
                ExprKind::Call {
                    callee: Box::new(target),
                    type_args: None,
                    args: vec![expr],
                },
                span,
            )),
        }
    }

    fn desugar_currying(&self, expr: Expr) -> Result<Expr> {
        let span = expr.span;
        match expr.kind {
            ExprKind::Identifier(ref name) if name == "_" => {
                Err(self.error("Placeholder '_' is only valid in call argument positions"))
            }

            ExprKind::Call {
                callee,
                type_args,
                args,
            } => {
                let callee = Box::new(self.desugar_currying(*callee)?);
                let desugared_args: Vec<Expr> = args
                    .into_iter()
                    .map(|a| self.desugar_currying_arg(a))
                    .collect::<Result<Vec<Expr>>>()?;

                let placeholder_indices: Vec<usize> = desugared_args
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| is_placeholder(a))
                    .map(|(i, _)| i)
                    .collect();

                if placeholder_indices.is_empty() {
                    Ok(Expr::new(
                        ExprKind::Call {
                            callee,
                            type_args,
                            args: desugared_args,
                        },
                        span,
                    ))
                } else {
                    let mut params = Vec::new();
                    let mut final_args = desugared_args;
                    for (param_idx, &arg_idx) in placeholder_indices.iter().enumerate() {
                        let param_name = format!("__curry_{}", param_idx);
                        let param_expr = Expr::new(
                            ExprKind::Identifier(param_name.clone()),
                            final_args[arg_idx].span,
                        );
                        final_args[arg_idx] = param_expr;
                        params.push((param_name, None));
                    }

                    let call_expr = Expr::new(
                        ExprKind::Call {
                            callee,
                            type_args,
                            args: final_args,
                        },
                        span,
                    );

                    Ok(Expr::new(
                        ExprKind::Lambda {
                            params,
                            return_type: None,
                            body: Box::new(call_expr),
                        },
                        span,
                    ))
                }
            }

            ExprKind::MethodCall {
                receiver,
                method,
                type_args,
                args,
            } => {
                let receiver = Box::new(self.desugar_currying(*receiver)?);
                let desugared_args: Vec<Expr> = args
                    .into_iter()
                    .map(|a| self.desugar_currying_arg(a))
                    .collect::<Result<Vec<Expr>>>()?;

                let placeholder_indices: Vec<usize> = desugared_args
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| is_placeholder(a))
                    .map(|(i, _)| i)
                    .collect();

                if placeholder_indices.is_empty() {
                    Ok(Expr::new(
                        ExprKind::MethodCall {
                            receiver,
                            method,
                            type_args,
                            args: desugared_args,
                        },
                        span,
                    ))
                } else {
                    let mut params = Vec::new();
                    let mut final_args = desugared_args;
                    for (param_idx, &arg_idx) in placeholder_indices.iter().enumerate() {
                        let param_name = format!("__curry_{}", param_idx);
                        let param_expr = Expr::new(
                            ExprKind::Identifier(param_name.clone()),
                            final_args[arg_idx].span,
                        );
                        final_args[arg_idx] = param_expr;
                        params.push((param_name, None));
                    }

                    let method_expr = Expr::new(
                        ExprKind::MethodCall {
                            receiver,
                            method,
                            type_args,
                            args: final_args,
                        },
                        span,
                    );

                    Ok(Expr::new(
                        ExprKind::Lambda {
                            params,
                            return_type: None,
                            body: Box::new(method_expr),
                        },
                        span,
                    ))
                }
            }

            ExprKind::Binary { left, op, right } => Ok(Expr::new(
                ExprKind::Binary {
                    left: Box::new(self.desugar_currying(*left)?),
                    op,
                    right: Box::new(self.desugar_currying(*right)?),
                },
                span,
            )),

            ExprKind::Unary { op, operand } => Ok(Expr::new(
                ExprKind::Unary {
                    op,
                    operand: Box::new(self.desugar_currying(*operand)?),
                },
                span,
            )),

            ExprKind::FieldAccess { object, field } => Ok(Expr::new(
                ExprKind::FieldAccess {
                    object: Box::new(self.desugar_currying(*object)?),
                    field,
                },
                span,
            )),

            ExprKind::Index { object, index } => Ok(Expr::new(
                ExprKind::Index {
                    object: Box::new(self.desugar_currying(*object)?),
                    index: Box::new(self.desugar_currying(*index)?),
                },
                span,
            )),

            ExprKind::Array(elements) => {
                let elements = elements
                    .into_iter()
                    .map(|e| self.desugar_currying(e))
                    .collect::<Result<Vec<Expr>>>()?;
                Ok(Expr::new(ExprKind::Array(elements), span))
            }

            ExprKind::Map(pairs) => {
                let mut desugared_pairs = Vec::new();
                for (k, v) in pairs {
                    let k = self.desugar_currying(k)?;
                    let v = self.desugar_currying(v)?;
                    desugared_pairs.push((k, v));
                }
                Ok(Expr::new(ExprKind::Map(desugared_pairs), span))
            }

            ExprKind::Tuple(elements) => {
                let elements = elements
                    .into_iter()
                    .map(|e| self.desugar_currying(e))
                    .collect::<Result<Vec<Expr>>>()?;
                Ok(Expr::new(ExprKind::Tuple(elements), span))
            }

            ExprKind::StructLiteral { name, fields } => {
                let mut desugared_fields = Vec::new();
                for mut f in fields {
                    f.value = self.desugar_currying(f.value)?;
                    desugared_fields.push(f);
                }
                Ok(Expr::new(
                    ExprKind::StructLiteral {
                        name,
                        fields: desugared_fields,
                    },
                    span,
                ))
            }

            ExprKind::EnumConstructor {
                enum_name,
                variant,
                args,
            } => {
                let desugared_args: Vec<Expr> = args
                    .into_iter()
                    .map(|a| self.desugar_currying_arg(a))
                    .collect::<Result<Vec<Expr>>>()?;

                let placeholder_indices: Vec<usize> = desugared_args
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| is_placeholder(a))
                    .map(|(i, _)| i)
                    .collect();

                if placeholder_indices.is_empty() {
                    Ok(Expr::new(
                        ExprKind::EnumConstructor {
                            enum_name,
                            variant,
                            args: desugared_args,
                        },
                        span,
                    ))
                } else {
                    let mut params = Vec::new();
                    let mut final_args = desugared_args;
                    for (param_idx, &arg_idx) in placeholder_indices.iter().enumerate() {
                        let param_name = format!("__curry_{}", param_idx);
                        let param_expr = Expr::new(
                            ExprKind::Identifier(param_name.clone()),
                            final_args[arg_idx].span,
                        );
                        final_args[arg_idx] = param_expr;
                        params.push((param_name, None));
                    }

                    let enum_expr = Expr::new(
                        ExprKind::EnumConstructor {
                            enum_name,
                            variant,
                            args: final_args,
                        },
                        span,
                    );

                    Ok(Expr::new(
                        ExprKind::Lambda {
                            params,
                            return_type: None,
                            body: Box::new(enum_expr),
                        },
                        span,
                    ))
                }
            }

            ExprKind::Lambda {
                params,
                return_type,
                body,
            } => Ok(Expr::new(
                ExprKind::Lambda {
                    params,
                    return_type,
                    body: Box::new(self.desugar_currying(*body)?),
                },
                span,
            )),

            ExprKind::Paren(inner) => {
                let inner = self.desugar_currying(*inner)?;
                Ok(Expr::new(ExprKind::Paren(Box::new(inner)), span))
            }

            ExprKind::Cast { expr, target_type } => Ok(Expr::new(
                ExprKind::Cast {
                    expr: Box::new(self.desugar_currying(*expr)?),
                    target_type,
                },
                span,
            )),

            ExprKind::TypeCheck { expr, check_type } => Ok(Expr::new(
                ExprKind::TypeCheck {
                    expr: Box::new(self.desugar_currying(*expr)?),
                    check_type,
                },
                span,
            )),

            ExprKind::IsPattern { expr, pattern } => Ok(Expr::new(
                ExprKind::IsPattern {
                    expr: Box::new(self.desugar_currying(*expr)?),
                    pattern,
                },
                span,
            )),

            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let condition = Box::new(self.desugar_currying(*condition)?);
                let then_branch = Box::new(self.desugar_currying(*then_branch)?);
                let else_branch = if let Some(e) = else_branch {
                    Some(Box::new(self.desugar_currying(*e)?))
                } else {
                    None
                };
                Ok(Expr::new(
                    ExprKind::If {
                        condition,
                        then_branch,
                        else_branch,
                    },
                    span,
                ))
            }

            ExprKind::Block(stmts) => Ok(Expr::new(ExprKind::Block(stmts), span)),

            ExprKind::EffectBlock(stmts) => Ok(Expr::new(ExprKind::EffectBlock(stmts), span)),

            ExprKind::Return(exprs) => {
                let exprs = exprs
                    .into_iter()
                    .map(|e| self.desugar_currying(e))
                    .collect::<Result<Vec<Expr>>>()?;
                Ok(Expr::new(ExprKind::Return(exprs), span))
            }

            ExprKind::Range {
                start,
                end,
                inclusive,
            } => Ok(Expr::new(
                ExprKind::Range {
                    start: Box::new(self.desugar_currying(*start)?),
                    end: Box::new(self.desugar_currying(*end)?),
                    inclusive,
                },
                span,
            )),

            ExprKind::Literal(_) | ExprKind::Identifier(_) | ExprKind::RegexLiteral { .. } => {
                Ok(Expr::new(expr.kind, span))
            }
        }
    }

    fn desugar_currying_arg(&self, arg: Expr) -> Result<Expr> {
        if is_placeholder(&arg) {
            Ok(arg)
        } else {
            self.desugar_currying(arg)
        }
    }

    #[inline(never)]
    fn parse_logical_or(&mut self) -> Result<Expr> {
        let mut expr = self.parse_logical_and()?;
        while self.match_token(&[TokenKind::Or]) {
            let op = BinaryOp::Or;
            let right = self.parse_logical_and()?;
            let span = self.span_through_previous(expr.span);
            expr = Expr::new(
                ExprKind::Binary {
                    left: Box::new(expr),
                    op,
                    right: Box::new(right),
                },
                span,
            );
        }

        Ok(expr)
    }

    #[inline(never)]
    fn parse_logical_and(&mut self) -> Result<Expr> {
        let mut expr = self.parse_comparison()?;
        while self.match_token(&[TokenKind::And]) {
            let op = BinaryOp::And;
            let right = self.parse_comparison()?;
            let span = self.span_through_previous(expr.span);
            expr = Expr::new(
                ExprKind::Binary {
                    left: Box::new(expr),
                    op,
                    right: Box::new(right),
                },
                span,
            );
        }

        Ok(expr)
    }

    #[inline(never)]
    fn parse_comparison(&mut self) -> Result<Expr> {
        let mut expr = self.parse_range()?;
        while self.match_token(&[
            TokenKind::DoubleEqual,
            TokenKind::NotEqual,
            TokenKind::Less,
            TokenKind::LessEqual,
            TokenKind::Greater,
            TokenKind::GreaterEqual,
        ]) {
            let op = match self.tokens[self.current - 1].kind {
                TokenKind::DoubleEqual => BinaryOp::Eq,
                TokenKind::NotEqual => BinaryOp::Ne,
                TokenKind::Less => BinaryOp::Lt,
                TokenKind::LessEqual => BinaryOp::Le,
                TokenKind::Greater => BinaryOp::Gt,
                TokenKind::GreaterEqual => BinaryOp::Ge,
                _ => unreachable!(),
            };
            let right = self.parse_range()?;
            let span = self.span_through_previous(expr.span);
            expr = Expr::new(
                ExprKind::Binary {
                    left: Box::new(expr),
                    op,
                    right: Box::new(right),
                },
                span,
            );
        }

        Ok(expr)
    }

    fn parse_range(&mut self) -> Result<Expr> {
        let expr = self.parse_bit_or()?;
        if self.match_token(&[TokenKind::DoubleDot]) {
            let end = self.parse_bit_or()?;
            let span = self.span_through_previous(expr.span);
            return Ok(Expr::new(
                ExprKind::Range {
                    start: Box::new(expr),
                    end: Box::new(end),
                    inclusive: false,
                },
                span,
            ));
        }

        Ok(expr)
    }

    fn parse_bit_or(&mut self) -> Result<Expr> {
        let mut expr = self.parse_bit_xor()?;
        while self.match_token(&[TokenKind::Pipe]) {
            let right = self.parse_bit_xor()?;
            let span = self.span_through_previous(expr.span);
            expr = Expr::new(
                ExprKind::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::BitOr,
                    right: Box::new(right),
                },
                span,
            );
        }

        Ok(expr)
    }

    fn parse_bit_xor(&mut self) -> Result<Expr> {
        let mut expr = self.parse_bit_and()?;
        while self.match_token(&[TokenKind::DoubleCaret, TokenKind::Tilde]) {
            let right = self.parse_bit_and()?;
            let span = self.span_through_previous(expr.span);
            expr = Expr::new(
                ExprKind::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::BitXor,
                    right: Box::new(right),
                },
                span,
            );
        }

        Ok(expr)
    }

    fn parse_bit_and(&mut self) -> Result<Expr> {
        let mut expr = self.parse_shift()?;
        while self.match_token(&[TokenKind::Ampersand]) {
            let right = self.parse_shift()?;
            let span = self.span_through_previous(expr.span);
            expr = Expr::new(
                ExprKind::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::BitAnd,
                    right: Box::new(right),
                },
                span,
            );
        }

        Ok(expr)
    }

    fn parse_shift(&mut self) -> Result<Expr> {
        let mut expr = self.parse_concat()?;
        while self.match_token(&[TokenKind::DoubleLess, TokenKind::DoubleGreater]) {
            let op = match self.tokens[self.current - 1].kind {
                TokenKind::DoubleLess => BinaryOp::Shl,
                TokenKind::DoubleGreater => BinaryOp::Shr,
                _ => unreachable!(),
            };
            let right = self.parse_concat()?;
            let span = self.span_through_previous(expr.span);
            expr = Expr::new(
                ExprKind::Binary {
                    left: Box::new(expr),
                    op,
                    right: Box::new(right),
                },
                span,
            );
        }

        Ok(expr)
    }

    fn is_string_like_expr(expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Literal(Literal::String(_)) => true,
            ExprKind::Binary {
                op: BinaryOp::Concat,
                ..
            } => true,
            _ => false,
        }
    }

    fn parse_concat(&mut self) -> Result<Expr> {
        let expr = self.parse_term()?;
        if self.check(TokenKind::DoubleDot) {
            if !Self::is_string_like_expr(&expr) {
                let next_is_string = self
                    .peek_ahead(1)
                    .map(|next| matches!(next.kind, TokenKind::String))
                    .unwrap_or(false);
                if !next_is_string {
                    return Ok(expr);
                }
            }

            self.advance();
            let right = self.parse_concat()?;
            let span = self.span_through_previous(expr.span);
            return Ok(Expr::new(
                ExprKind::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::Concat,
                    right: Box::new(right),
                },
                span,
            ));
        }

        Ok(expr)
    }

    fn parse_term(&mut self) -> Result<Expr> {
        let mut expr = self.parse_factor()?;
        while self.match_token(&[TokenKind::Plus, TokenKind::Minus]) {
            let op = match self.tokens[self.current - 1].kind {
                TokenKind::Plus => BinaryOp::Add,
                TokenKind::Minus => BinaryOp::Sub,
                _ => unreachable!(),
            };
            let right = self.parse_factor()?;
            let span = self.span_through_previous(expr.span);
            expr = Expr::new(
                ExprKind::Binary {
                    left: Box::new(expr),
                    op,
                    right: Box::new(right),
                },
                span,
            );
        }

        Ok(expr)
    }

    fn parse_factor(&mut self) -> Result<Expr> {
        let mut expr = self.parse_unary()?;
        while self.match_token(&[
            TokenKind::Star,
            TokenKind::Slash,
            TokenKind::DoubleSlash,
            TokenKind::Percent,
        ]) {
            let op = match self.tokens[self.current - 1].kind {
                TokenKind::Star => BinaryOp::Mul,
                TokenKind::Slash => BinaryOp::Div,
                TokenKind::DoubleSlash => BinaryOp::FloorDiv,
                TokenKind::Percent => BinaryOp::Mod,
                _ => unreachable!(),
            };
            let right = self.parse_unary()?;
            let span = self.span_through_previous(expr.span);
            expr = Expr::new(
                ExprKind::Binary {
                    left: Box::new(expr),
                    op,
                    right: Box::new(right),
                },
                span,
            );
        }

        Ok(expr)
    }

    fn parse_unary(&mut self) -> Result<Expr> {
        if self.match_token(&[TokenKind::Minus, TokenKind::Not, TokenKind::Tilde]) {
            let start_token = self.tokens[self.current - 1].clone();
            let op = match start_token.kind {
                TokenKind::Minus => UnaryOp::Neg,
                TokenKind::Not => UnaryOp::Not,
                TokenKind::Tilde => UnaryOp::BitNot,
                _ => unreachable!(),
            };

            if op == UnaryOp::Neg && self.check(TokenKind::Integer) {
                let next_token = self.current_token();
                let clean: String = next_token.lexeme.chars().filter(|&c| c != '_').collect();
                #[cfg(feature = "std")]
                let is_min = clean == "9223372036854775808";
                #[cfg(not(feature = "std"))]
                let is_min = clean == "2147483648";

                if is_min {
                    let end_token = self.advance().clone();
                    let span = self.make_span(&start_token, &end_token);
                    #[cfg(feature = "std")]
                    let min_val: i64 = i64::MIN;
                    #[cfg(not(feature = "std"))]
                    let min_val: i32 = i32::MIN;
                    return Ok(Expr::new(
                        ExprKind::Literal(Literal::Integer(min_val)),
                        span,
                    ));
                }
            }

            let operand = self.parse_unary()?;
            let end_token = self.tokens[self.current - 1].clone();
            return Ok(Expr::new(
                ExprKind::Unary {
                    op,
                    operand: Box::new(operand),
                },
                self.make_span(&start_token, &end_token),
            ));
        }

        self.parse_power()
    }

    fn parse_power(&mut self) -> Result<Expr> {
        let expr = self.parse_postfix()?;
        if self.match_token(&[TokenKind::Caret]) {
            let right = self.parse_unary()?;
            let span = self.span_through_previous(expr.span);
            return Ok(Expr::new(
                ExprKind::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::Pow,
                    right: Box::new(right),
                },
                span,
            ));
        }

        Ok(expr)
    }

    fn parse_call_arg(&mut self) -> Result<Expr> {
        let mut lookahead = self.current;
        let mut paren_count = 0;
        while self.tokens.get(lookahead).is_some_and(|t| t.kind == TokenKind::LeftParen) {
            paren_count += 1;
            lookahead += 1;
        }
        if self
            .tokens
            .get(lookahead)
            .is_some_and(|t| t.kind == TokenKind::Identifier && t.lexeme == "_")
        {
            let mut close_lookahead = lookahead + 1;
            let mut close_count = 0;
            while self
                .tokens
                .get(close_lookahead)
                .is_some_and(|t| t.kind == TokenKind::RightParen)
                && close_count < paren_count
            {
                close_count += 1;
                close_lookahead += 1;
            }
            if close_count == paren_count {
                let mut lparens = Vec::with_capacity(paren_count);
                for _ in 0..paren_count {
                    lparens.push(self.advance().clone());
                }
                let token = self.advance().clone();
                let inner_span = Span::new(
                    token.line,
                    token.column,
                    token.line,
                    token.column.saturating_add(token.lexeme.chars().count()),
                );
                let mut expr = Expr::new(ExprKind::Identifier(token.lexeme), inner_span);
                for i in (0..paren_count).rev() {
                    let lp = &lparens[i];
                    let rp = self.advance().clone();
                    let span = Span::new(
                        lp.line,
                        lp.column,
                        rp.line,
                        rp.column.saturating_add(rp.lexeme.chars().count()),
                    );
                    expr = Expr::new(ExprKind::Paren(Box::new(expr)), span);
                }
                return Ok(expr);
            }
        }
        self.parse_expr()
    }

    #[inline(never)]
    fn parse_postfix(&mut self) -> Result<Expr> {
        let mut expr = self.parse_primary()?;
        loop {
            match self.peek_kind() {
                TokenKind::LeftParen => {
                    self.advance();
                    let mut args = Vec::new();
                    if !self.check(TokenKind::RightParen) {
                        args.push(self.parse_call_arg()?);
                        while self.match_token(&[TokenKind::Comma]) {
                            args.push(self.parse_call_arg()?);
                        }
                    }

                    self.consume(TokenKind::RightParen, "Expected ')' after arguments")?;
                    let span = expr.span;
                    expr = Expr::new(
                        ExprKind::Call {
                            callee: Box::new(expr),
                            type_args: None,
                            args,
                        },
                        span,
                    );
                }

                TokenKind::Less if self.looks_like_generic_call(&expr) => {
                    let type_args = self.parse_type_arguments()?;
                    self.consume(TokenKind::LeftParen, "Expected '(' after type arguments")?;
                    let mut args = Vec::new();
                    if !self.check(TokenKind::RightParen) {
                        args.push(self.parse_call_arg()?);
                        while self.match_token(&[TokenKind::Comma]) {
                            args.push(self.parse_call_arg()?);
                        }
                    }
                    self.consume(TokenKind::RightParen, "Expected ')' after arguments")?;
                    let span = expr.span;
                    expr = Expr::new(
                        ExprKind::Call {
                            callee: Box::new(expr),
                            type_args: Some(type_args),
                            args,
                        },
                        span,
                    );
                }

                TokenKind::Colon => {
                    self.advance();
                    let method = if self.check(TokenKind::Identifier) {
                        self.expect_identifier()?
                    } else {
                        let token = self.current_token().clone();
                        let method_name = token.lexeme.clone();
                        self.advance();
                        method_name
                    };
                    let type_args = if self.check(TokenKind::Less) {
                        self.advance();
                        let mut types = vec![self.parse_type()?];
                        while self.match_token(&[TokenKind::Comma]) {
                            types.push(self.parse_type()?);
                        }

                        self.consume_closing_angle_bracket("Expected '>' after type arguments")?;
                        Some(types)
                    } else {
                        None
                    };
                    self.consume(TokenKind::LeftParen, "Expected '(' after method name")?;
                    let mut args = Vec::new();
                    if !self.check(TokenKind::RightParen) {
                        args.push(self.parse_call_arg()?);
                        while self.match_token(&[TokenKind::Comma]) {
                            args.push(self.parse_call_arg()?);
                        }
                    }

                    let right_paren =
                        self.consume(TokenKind::RightParen, "Expected ')' after arguments")?;
                    let (start_line, start_col) = if expr.span.start_line > 0 {
                        (expr.span.start_line, expr.span.start_col)
                    } else {
                        (right_paren.line, right_paren.column)
                    };
                    let span =
                        Span::new(start_line, start_col, right_paren.line, right_paren.column);
                    expr = Expr::new(
                        ExprKind::MethodCall {
                            receiver: Box::new(expr),
                            method,
                            type_args,
                            args,
                        },
                        span,
                    );
                }

                TokenKind::Dot => {
                    self.advance();
                    let field_token =
                        self.consume(TokenKind::Identifier, "Expected field name after '.'")?;
                    let field = field_token.lexeme.clone();
                    let end_col = field_token
                        .column
                        .saturating_add(field.len().saturating_sub(1));
                    let (start_line, start_col) = if expr.span.start_line > 0 {
                        (expr.span.start_line, expr.span.start_col)
                    } else {
                        (field_token.line, field_token.column)
                    };
                    let span = Span::new(start_line, start_col, field_token.line, end_col);
                    expr = Expr::new(
                        ExprKind::FieldAccess {
                            object: Box::new(expr),
                            field,
                        },
                        span,
                    );
                }

                TokenKind::LeftBrace => {
                    if let Some(path) = Parser::expr_to_path(&expr) {
                        expr = self.parse_struct_literal(path)?;
                        continue;
                    } else {
                        break;
                    }
                }

                TokenKind::LeftBracket => {
                    self.advance();
                    let index = self.parse_expr()?;
                    let right_bracket =
                        self.consume(TokenKind::RightBracket, "Expected ']' after index")?;
                    let (start_line, start_col) = if expr.span.start_line > 0 {
                        (expr.span.start_line, expr.span.start_col)
                    } else {
                        (right_bracket.line, right_bracket.column)
                    };
                    let span = Span::new(
                        start_line,
                        start_col,
                        right_bracket.line,
                        right_bracket.column,
                    );
                    expr = Expr::new(
                        ExprKind::Index {
                            object: Box::new(expr),
                            index: Box::new(index),
                        },
                        span,
                    );
                }

                TokenKind::As => {
                    self.advance();
                    let target_type = self.parse_type()?;
                    let span = self.span_through_previous(expr.span);
                    expr = Expr::new(
                        ExprKind::Cast {
                            expr: Box::new(expr),
                            target_type,
                        },
                        span,
                    );
                }

                TokenKind::Is => {
                    self.advance();

                    #[cfg(feature = "esp32c6-logging")]
                    log::info!("        parsing 'is' expression");

                    let is_pattern = if self.check(TokenKind::Identifier) {
                        self.peek_ahead(1)
                            .map(|t| t.kind == TokenKind::LeftParen)
                            .unwrap_or(false)
                    } else if matches!(
                        self.peek_kind(),
                        TokenKind::Integer
                            | TokenKind::Float
                            | TokenKind::String
                            | TokenKind::True
                            | TokenKind::False
                            | TokenKind::Minus
                    ) {
                        true
                    } else {
                        false
                    };
                    if is_pattern {
                        #[cfg(feature = "esp32c6-logging")]
                        log::info!("        parsing is pattern");

                        let pattern = self.parse_pattern()?;
                        let span = self.span_through_previous(expr.span);
                        expr = Expr::new(
                            ExprKind::IsPattern {
                                expr: Box::new(expr),
                                pattern,
                            },
                            span,
                        );
                    } else {
                        let check_type = self.parse_type()?;
                        let span = self.span_through_previous(expr.span);
                        expr = Expr::new(
                            ExprKind::TypeCheck {
                                expr: Box::new(expr),
                                check_type,
                            },
                            span,
                        );
                    }
                }

                TokenKind::Question => {
                    self.advance();
                    let span = self.span_through_previous(expr.span);
                    expr = Expr::new(
                        ExprKind::MethodCall {
                            receiver: Box::new(expr),
                            method: "try_unwrap".to_string(),
                            type_args: None,
                            args: vec![],
                        },
                        span,
                    );
                }

                _ => break,
            }
        }

        Ok(expr)
    }

    fn looks_like_generic_call(&self, callee: &Expr) -> bool {
        if !matches!(
            callee.kind,
            ExprKind::Identifier(_) | ExprKind::FieldAccess { .. }
        ) {
            return false;
        }
        let Some(less) = self.tokens.get(self.current) else {
            return false;
        };
        if let Some(previous) = self
            .current
            .checked_sub(1)
            .and_then(|index| self.tokens.get(index))
            && (less.line != previous.line
                || less.column
                    != previous
                        .column
                        .saturating_add(previous.lexeme.chars().count()))
        {
            return false;
        }
        let mut depth = 0usize;
        let mut index = self.current;
        while let Some(token) = self.tokens.get(index) {
            match token.kind {
                TokenKind::Less => depth += 1,
                TokenKind::Greater => {
                    if depth == 0 {
                        return false;
                    }
                    depth -= 1;
                    if depth == 0 {
                        return self
                            .tokens
                            .get(index + 1)
                            .is_some_and(|next| next.kind == TokenKind::LeftParen);
                    }
                }
                TokenKind::DoubleGreater => {
                    if depth < 2 {
                        return false;
                    }
                    depth -= 2;
                    if depth == 0 {
                        return self
                            .tokens
                            .get(index + 1)
                            .is_some_and(|next| next.kind == TokenKind::LeftParen);
                    }
                }
                TokenKind::Eof => return false,
                _ => {}
            }
            index += 1;
        }
        false
    }

    fn expr_to_path(expr: &Expr) -> Option<String> {
        match &expr.kind {
            ExprKind::Identifier(name) => Some(name.clone()),
            ExprKind::FieldAccess { object, field } => {
                let mut prefix = Self::expr_to_path(object)?;
                prefix.push('.');
                prefix.push_str(field);
                Some(prefix)
            }

            _ => None,
        }
    }

    #[inline(never)]
    fn parse_primary(&mut self) -> Result<Expr> {
        let start_token = self.current_token().clone();
        let kind = match self.peek_kind() {
            TokenKind::Integer => {
                let token = self.advance();
                let value = parse_integer_literal(&token.lexeme)
                    .map_err(|e| self.error(&format!("{}", e)))?;
                ExprKind::Literal(Literal::Integer(value))
            }

            TokenKind::Float => {
                let token = self.advance();
                let value = parse_float_literal(&token.lexeme)
                    .map_err(|e| self.error(&format!("{}", e)))?;
                ExprKind::Literal(Literal::Float(value))
            }

            TokenKind::String => {
                let token = self.advance().clone();
                let string_value = self.unescape_string_literal(&token)?;
                ExprKind::Literal(Literal::String(string_value))
            }

            TokenKind::True => {
                self.advance();
                ExprKind::Literal(Literal::Bool(true))
            }

            TokenKind::False => {
                self.advance();
                ExprKind::Literal(Literal::Bool(false))
            }

            TokenKind::EffectKw => {
                self.advance();
                let mut statements = Vec::new();
                if self.match_token(&[TokenKind::LeftBrace]) {
                    while !self.check(TokenKind::RightBrace) && !self.is_at_end() {
                        statements.push(self.parse_stmt()?);
                    }
                    self.consume(TokenKind::RightBrace, "Expected '}' after effect block")?;
                } else if self.match_token(&[TokenKind::Do]) {
                    while !self.check(TokenKind::End) && !self.is_at_end() {
                        statements.push(self.parse_stmt()?);
                    }
                    self.consume(TokenKind::End, "Expected 'end' after effect block")?;
                } else {
                    return Err(self.error("Expected 'do' or '{' after 'effect'"));
                }
                let end_token = self.tokens[self.current - 1].clone();
                return Ok(Expr::new(
                    ExprKind::EffectBlock(statements),
                    self.make_span(&start_token, &end_token),
                ));
            }

            TokenKind::Identifier => {
                if (self.current_token().lexeme == "regex"
                    || self.regex_aliases.contains(&self.current_token().lexeme))
                    && self.peek_ahead(1).is_some_and(|t| t.kind == TokenKind::Bang)
                    && self.peek_ahead(2).is_some_and(|t| t.kind == TokenKind::String)
                {
                    self.advance();
                    self.advance();
                    let str_token = self.advance().clone();
                    let pattern = self.unescape_string_literal(&str_token)?;
                    let end_token = self.tokens[self.current - 1].clone();
                    let span = self.make_span(&start_token, &end_token);
                    return Ok(Expr::new(ExprKind::RegexLiteral { pattern, span }, span));
                }

                if self.current_token().lexeme == "effect"
                    && (self.peek_ahead(1).is_some_and(|t| t.kind == TokenKind::Do || t.kind == TokenKind::LeftBrace))
                {
                    self.advance();
                    let mut statements = Vec::new();
                    if self.match_token(&[TokenKind::LeftBrace]) {
                        while !self.check(TokenKind::RightBrace) && !self.is_at_end() {
                            statements.push(self.parse_stmt()?);
                        }
                        self.consume(TokenKind::RightBrace, "Expected '}' after effect block")?;
                    } else if self.match_token(&[TokenKind::Do]) {
                        while !self.check(TokenKind::End) && !self.is_at_end() {
                            statements.push(self.parse_stmt()?);
                        }
                        self.consume(TokenKind::End, "Expected 'end' after effect block")?;
                    }
                    let end_token = self.tokens[self.current - 1].clone();
                    return Ok(Expr::new(
                        ExprKind::EffectBlock(statements),
                        self.make_span(&start_token, &end_token),
                    ));
                }

                let name = self.expect_identifier()?;
                if self.check(TokenKind::LeftBrace) {
                    return self.parse_struct_literal(name);
                }

                if self.check(TokenKind::LeftParen) {
                    return Ok(Expr::new(
                        ExprKind::Identifier(name),
                        self.make_span(&start_token, &self.tokens[self.current - 1]),
                    ));
                }

                ExprKind::Identifier(name)
            }

            TokenKind::LeftParen => {
                self.advance();
                let mut expressions = Vec::new();
                expressions.push(self.parse_expr()?);
                while self.match_token(&[TokenKind::Comma]) {
                    expressions.push(self.parse_expr()?);
                }

                let end_token = self.current_token().clone();
                self.consume(TokenKind::RightParen, "Expected ')' after expression")?;
                if expressions.len() == 1 {
                    let span = self.make_span(&start_token, &end_token);
                    let inner = expressions.pop().expect("one expression present");
                    return Ok(Expr::new(
                        ExprKind::Paren(Box::new(inner)),
                        span,
                    ));
                } else {
                    let span = self.make_span(&start_token, &end_token);
                    return Ok(Expr::new(ExprKind::Tuple(expressions), span));
                }
            }

            TokenKind::LeftBracket => {
                return self.parse_array_literal();
            }

            TokenKind::LeftBrace => {
                return self.parse_map_or_block();
            }

            TokenKind::If => {
                return self.parse_if_expr();
            }

            TokenKind::Return => {
                self.advance();
                let mut values = Vec::new();
                if !(self.check(TokenKind::End)
                    || self.check(TokenKind::Else)
                    || self.check(TokenKind::Elseif)
                    || self.check(TokenKind::Newline)
                    || self.is_at_end())
                {
                    values.push(self.parse_expr()?);
                    while self.match_token(&[TokenKind::Comma]) {
                        values.push(self.parse_expr()?);
                    }
                }

                ExprKind::Return(values)
            }

            TokenKind::Function => {
                if self
                    .peek_ahead(1)
                    .is_some_and(|t| t.kind == TokenKind::LeftParen)
                {
                    return self.parse_lambda_function();
                } else {
                    return Err(self.error("Unexpected 'function' keyword in expression context"));
                }
            }

            _ => {
                return Err(self.error(&format!(
                    "Unexpected token in expression: {:?}",
                    self.peek_kind()
                )));
            }
        };
        let end_token = self.tokens[self.current - 1].clone();
        Ok(Expr::new(kind, self.make_span(&start_token, &end_token)))
    }

    fn parse_struct_literal(&mut self, name: String) -> Result<Expr> {
        let start_token = self.tokens[self.current - 1].clone();
        self.consume(TokenKind::LeftBrace, "Expected '{'")?;
        let mut fields = Vec::new();
        if !self.check(TokenKind::RightBrace) {
            loop {
                let name_token = self.current_token().clone();
                self.consume(TokenKind::Identifier, "Expected field name")?;
                let field_name = name_token.lexeme.clone();
                self.consume(TokenKind::Equal, "Expected '=' after field name")?;
                let value = self.parse_expr()?;
                let field_span = self.make_span(&name_token, &name_token);
                fields.push(StructLiteralField {
                    name: field_name,
                    value,
                    span: field_span,
                });
                if !self.match_token(&[TokenKind::Comma]) {
                    break;
                }
            }
        }

        self.consume(TokenKind::RightBrace, "Expected '}' after struct fields")?;
        let end_token = self.tokens[self.current - 1].clone();
        Ok(Expr::new(
            ExprKind::StructLiteral { name, fields },
            self.make_span(&start_token, &end_token),
        ))
    }

    fn parse_array_literal(&mut self) -> Result<Expr> {
        let start_token = self.current_token().clone();
        self.advance();
        let mut elements = Vec::new();
        if !self.check(TokenKind::RightBracket) {
            elements.push(self.parse_expr()?);
            while self.match_token(&[TokenKind::Comma]) {
                if self.check(TokenKind::RightBracket) {
                    break;
                }

                elements.push(self.parse_expr()?);
            }
        }

        self.consume(TokenKind::RightBracket, "Expected ']' after array elements")?;
        let end_token = self.tokens[self.current - 1].clone();
        Ok(Expr::new(
            ExprKind::Array(elements),
            self.make_span(&start_token, &end_token),
        ))
    }

    fn parse_map_or_block(&mut self) -> Result<Expr> {
        let start_token = self.current_token().clone();
        self.advance();
        if self.check(TokenKind::RightBrace) {
            self.advance();
            let end_token = self.tokens[self.current - 1].clone();
            return Ok(Expr::new(
                ExprKind::Map(vec![]),
                self.make_span(&start_token, &end_token),
            ));
        }

        if self.check(TokenKind::LeftBracket) || self.check(TokenKind::Identifier) {
            let mut pairs = Vec::new();
            loop {
                if self.check(TokenKind::LeftBracket) {
                    self.advance();
                    let key = self.parse_expr()?;
                    self.consume(TokenKind::RightBracket, "Expected ']' after map key")?;
                    self.consume(TokenKind::Equal, "Expected '=' after map key")?;
                    let value = self.parse_expr()?;
                    pairs.push((key, value));
                } else if self.check(TokenKind::Identifier) {
                    let key_token = self.advance().clone();
                    let span = self.make_span(&key_token, &key_token);
                    if !self.match_token(&[TokenKind::Equal, TokenKind::Colon]) {
                        return Err(self.error("Expected '=' or ':' after map key"));
                    }
                    let key_expr = Expr::new(
                        ExprKind::Literal(Literal::String(key_token.lexeme.clone())),
                        span,
                    );
                    let value = self.parse_expr()?;
                    pairs.push((key_expr, value));
                } else {
                    return Err(self.error(
                        "Expected '[' expression ']' or identifier before '=' in map literal",
                    ));
                }

                if !self.match_token(&[TokenKind::Comma]) {
                    break;
                }

                if self.check(TokenKind::RightBrace) {
                    break;
                }
            }

            self.consume(TokenKind::RightBrace, "Expected '}' after map entries")?;
            let end_token = self.tokens[self.current - 1].clone();
            return Ok(Expr::new(
                ExprKind::Map(pairs),
                self.make_span(&start_token, &end_token),
            ));
        }

        Err(self.error("Block expressions not yet implemented"))
    }

    fn parse_if_expr(&mut self) -> Result<Expr> {
        let start_token = self.current_token().clone();
        self.advance();
        let condition = Box::new(self.parse_expr()?);
        self.consume(TokenKind::Then, "Expected 'then' after if condition")?;
        let then_branch = Box::new(self.parse_expr()?);
        let else_branch = if self.match_token(&[TokenKind::Else]) {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };
        let end_token = self.tokens[self.current - 1].clone();
        Ok(Expr::new(
            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            },
            self.make_span(&start_token, &end_token),
        ))
    }

    #[inline(never)]
    pub(super) fn parse_pattern(&mut self) -> Result<Pattern> {
        match self.peek_kind() {
            TokenKind::Identifier => {
                let name = self.expect_identifier()?;
                if self.check(TokenKind::LeftParen) {
                    self.advance();

                    #[cfg(feature = "esp32c6-logging")]
                    log::info!("      parsing pattern Some({})", name);

                    // Pre-allocate for typical pattern binding count (1-4 bindings)
                    let mut bindings = Vec::with_capacity(4);
                    if !self.check(TokenKind::RightParen) {
                        bindings.push(self.parse_pattern()?);
                        while self.match_token(&[TokenKind::Comma]) {
                            bindings.push(self.parse_pattern()?);
                        }
                    }

                    self.consume(TokenKind::RightParen, "Expected ')' after pattern")?;

                    // Shrink to fit
                    bindings.shrink_to_fit();

                    Ok(Pattern::Enum {
                        enum_name: String::new(),
                        variant: name,
                        bindings,
                    })
                } else {
                    if matches!(
                        name.as_str(),
                        "string" | "int" | "float" | "bool" | "unknown"
                    ) {
                        use crate::ast::{Span, Type, TypeKind};
                        let type_kind = match name.as_str() {
                            "string" => TypeKind::String,
                            "int" => TypeKind::Int,
                            "float" => TypeKind::Float,
                            "bool" => TypeKind::Bool,
                            "unknown" => TypeKind::Unknown,
                            _ => unreachable!(),
                        };
                        Ok(Pattern::TypeCheck(Type::new(type_kind, Span::dummy())))
                    } else {
                        Ok(Pattern::Identifier(name))
                    }
                }
            }

            TokenKind::Minus => {
                self.advance();
                match self.peek_kind() {
                    TokenKind::Integer => {
                        let token = self.advance().clone();
                        let clean: String = token.lexeme.chars().filter(|&c| c != '_').collect();
                        #[cfg(feature = "std")]
                        let is_min = clean == "9223372036854775808";
                        #[cfg(not(feature = "std"))]
                        let is_min = clean == "2147483648";

                        let lit = if is_min {
                            #[cfg(feature = "std")]
                            let min_val: i64 = i64::MIN;
                            #[cfg(not(feature = "std"))]
                            let min_val: i32 = i32::MIN;
                            Literal::Integer(min_val)
                        } else {
                            let value = parse_integer_literal(&token.lexeme)
                                .map_err(|e| self.error(&format!("{}", e)))?;
                            Literal::Integer(
                                value.checked_neg().ok_or_else(|| {
                                    self.error("Integer overflow in pattern negation")
                                })?,
                            )
                        };
                        Ok(Pattern::Literal(lit))
                    }
                    TokenKind::Float => {
                        let token = self.advance().clone();
                        let value = parse_float_literal(&token.lexeme)
                            .map_err(|e| self.error(&format!("{}", e)))?;
                        Ok(Pattern::Literal(Literal::Float(-value)))
                    }
                    _ => Err(self.error("Expected integer or float literal after '-' in pattern")),
                }
            }

            TokenKind::Integer
            | TokenKind::Float
            | TokenKind::String
            | TokenKind::True
            | TokenKind::False => {
                let token = self.advance().clone();
                let lit = match token.kind {
                    TokenKind::Integer => {
                        let value = parse_integer_literal(&token.lexeme)
                            .map_err(|e| self.error(&format!("{}", e)))?;
                        Literal::Integer(value)
                    }
                    TokenKind::Float => {
                        let value = parse_float_literal(&token.lexeme)
                            .map_err(|e| self.error(&format!("{}", e)))?;
                        Literal::Float(value)
                    }
                    TokenKind::String => {
                        let s = self.unescape_string_literal(&token)?;
                        Literal::String(s)
                    }

                    TokenKind::True => Literal::Bool(true),
                    TokenKind::False => Literal::Bool(false),
                    _ => unreachable!(),
                };
                Ok(Pattern::Literal(lit))
            }

            TokenKind::As => {
                self.advance();
                let ty = self.parse_type()?;
                Ok(Pattern::TypeCheck(ty))
            }

            _ => Ok(Pattern::Wildcard),
        }
    }

    pub(super) fn unescape_string_literal(&self, token: &Token) -> Result<String> {
        let raw = &token.lexeme;
        if raw.len() < 2 {
            return Err(LustError::ParserError {
                line: token.line,
                column: token.column,
                message: "Invalid string literal".to_string(),
                module: None,
            });
        }

        let mut result = String::with_capacity(raw.len());
        let mut chars = raw[1..raw.len() - 1].chars();
        while let Some(ch) = chars.next() {
            if ch != '\\' {
                result.push(ch);
                continue;
            }

            let escape = chars.next().ok_or_else(|| LustError::ParserError {
                line: token.line,
                column: token.column,
                message: "Incomplete escape sequence in string literal".to_string(),
                module: None,
            })?;
            match escape {
                'n' => result.push('\n'),
                'r' => result.push('\r'),
                't' => result.push('\t'),
                '\\' => result.push('\\'),
                '"' => result.push('"'),
                '\'' => result.push('\''),
                '0' => result.push('\0'),
                _ => {
                    return Err(LustError::ParserError {
                        line: token.line,
                        column: token.column,
                        message: format!("Unsupported escape sequence: \\{}", escape),
                        module: None,
                    });
                }
            }
        }

        Ok(result)
    }

    fn parse_lambda_function(&mut self) -> Result<Expr> {
        let start_token = self.current_token().clone();
        self.advance();
        self.consume(TokenKind::LeftParen, "Expected '(' after 'function'")?;
        let mut params = Vec::new();
        if !self.check(TokenKind::RightParen) {
            loop {
                let param_name = self.expect_identifier()?;
                let param_type = if self.match_token(&[TokenKind::Colon]) {
                    Some(self.parse_type()?)
                } else {
                    None
                };
                params.push((param_name, param_type));
                if !self.match_token(&[TokenKind::Comma]) {
                    break;
                }
            }
        }

        self.consume(TokenKind::RightParen, "Expected ')' after parameters")?;
        let return_type = if self.match_token(&[TokenKind::Colon, TokenKind::Arrow]) {
            Some(self.parse_type()?)
        } else {
            None
        };
        let (error_type, requirements) = self.parse_optional_effects_spec()?;
        let return_type = if error_type.is_some() || !requirements.is_empty() {
            let span = return_type.as_ref().map(|t| t.span).unwrap_or_else(Span::dummy);
            let success = Box::new(return_type.unwrap_or_else(|| Type::new(TypeKind::Unit, span)));
            let error = Box::new(
                error_type.unwrap_or_else(|| Type::new(TypeKind::Never, Span::dummy())),
            );
            Some(Type::new(
                TypeKind::Effect {
                    success,
                    error,
                    requirements,
                },
                span,
            ))
        } else {
            return_type
        };
        let mut statements = Vec::new();
        while !self.check(TokenKind::End) && !self.is_at_end() {
            if self.match_token(&[TokenKind::Newline]) {
                continue;
            }

            statements.push(self.parse_stmt()?);
        }

        self.consume(TokenKind::End, "Expected 'end' after lambda body")?;
        let end_token = self.tokens[self.current - 1].clone();
        let body = Box::new(Expr::new(
            ExprKind::Block(statements),
            self.make_span(&start_token, &end_token),
        ));
        Ok(Expr::new(
            ExprKind::Lambda {
                params,
                return_type,
                body,
            },
            self.make_span(&start_token, &end_token),
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::{BinaryOp, ExprKind, Literal, UnaryOp};
    use crate::intern::Interner;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn parse_expression(src: &str) -> crate::ast::Expr {
        let mut interner = Interner::new();
        let mut lexer = Lexer::new(src, &mut interner);
        let tokens = lexer.tokenize().expect("tokenize");
        let mut parser = Parser::new(tokens);
        parser.parse_expr().expect("parse_expr")
    }

    #[test]
    fn test_power_precedence_and_associativity() {
        // -2 ^ 2 => -(2 ^ 2)
        let expr = parse_expression("-2 ^ 2");
        match &expr.kind {
            ExprKind::Unary {
                op: UnaryOp::Neg,
                operand,
            } => match &operand.kind {
                ExprKind::Binary {
                    left,
                    op: BinaryOp::Pow,
                    right,
                } => {
                    assert!(matches!(left.kind, ExprKind::Literal(Literal::Integer(2))));
                    assert!(matches!(right.kind, ExprKind::Literal(Literal::Integer(2))));
                }
                other => panic!("expected Binary Pow, got {:?}", other),
            },
            other => panic!("expected Unary Neg, got {:?}", other),
        }

        // 2 ^ 3 ^ 2 => 2 ^ (3 ^ 2)
        let expr = parse_expression("2 ^ 3 ^ 2");
        match &expr.kind {
            ExprKind::Binary {
                left,
                op: BinaryOp::Pow,
                right,
            } => {
                assert!(matches!(left.kind, ExprKind::Literal(Literal::Integer(2))));
                match &right.kind {
                    ExprKind::Binary {
                        left: l2,
                        op: BinaryOp::Pow,
                        right: r2,
                    } => {
                        assert!(matches!(l2.kind, ExprKind::Literal(Literal::Integer(3))));
                        assert!(matches!(r2.kind, ExprKind::Literal(Literal::Integer(2))));
                    }
                    other => panic!("expected nested Binary Pow, got {:?}", other),
                }
            }
            other => panic!("expected Binary Pow, got {:?}", other),
        }

        // 2 ^ -2 => 2 ^ (-2)
        let expr = parse_expression("2 ^ -2");
        match &expr.kind {
            ExprKind::Binary {
                left,
                op: BinaryOp::Pow,
                right,
            } => {
                assert!(matches!(left.kind, ExprKind::Literal(Literal::Integer(2))));
                match &right.kind {
                    ExprKind::Unary {
                        op: UnaryOp::Neg,
                        operand,
                    } => {
                        assert!(matches!(operand.kind, ExprKind::Literal(Literal::Integer(2))));
                    }
                    other => panic!("expected Unary Neg on right operand, got {:?}", other),
                }
            }
            other => panic!("expected Binary Pow, got {:?}", other),
        }
    }

    #[test]
    fn test_bitwise_and_floor_div_parsing() {
        // // floor div
        let expr = parse_expression("10 // 3");
        assert!(matches!(
            expr.kind,
            ExprKind::Binary {
                op: BinaryOp::FloorDiv,
                ..
            }
        ));

        // & bit and
        let expr = parse_expression("a & b");
        assert!(matches!(
            expr.kind,
            ExprKind::Binary {
                op: BinaryOp::BitAnd,
                ..
            }
        ));

        // | bit or
        let expr = parse_expression("a | b");
        assert!(matches!(
            expr.kind,
            ExprKind::Binary {
                op: BinaryOp::BitOr,
                ..
            }
        ));

        // ^^ bit xor
        let expr = parse_expression("a ^^ b");
        assert!(matches!(
            expr.kind,
            ExprKind::Binary {
                op: BinaryOp::BitXor,
                ..
            }
        ));

        // ~ bit xor (binary)
        let expr = parse_expression("a ~ b");
        assert!(matches!(
            expr.kind,
            ExprKind::Binary {
                op: BinaryOp::BitXor,
                ..
            }
        ));

        // ~ bit not (unary)
        let expr = parse_expression("~a");
        assert!(matches!(
            expr.kind,
            ExprKind::Unary {
                op: UnaryOp::BitNot,
                ..
            }
        ));

        // << and >> shift
        let expr = parse_expression("1 << 4");
        assert!(matches!(
            expr.kind,
            ExprKind::Binary {
                op: BinaryOp::Shl,
                ..
            }
        ));

        let expr = parse_expression("16 >> 2");
        assert!(matches!(
            expr.kind,
            ExprKind::Binary {
                op: BinaryOp::Shr,
                ..
            }
        ));
    }

    #[test]
    fn test_precedence_hierarchy() {
        // a + b << c => (a + b) << c
        let expr = parse_expression("a + b << c");
        match &expr.kind {
            ExprKind::Binary {
                left,
                op: BinaryOp::Shl,
                ..
            } => {
                assert!(matches!(left.kind, ExprKind::Binary { op: BinaryOp::Add, .. }));
            }
            other => panic!("expected Shift with Add left, got {:?}", other),
        }

        // x & y | z => (x & y) | z
        let expr = parse_expression("x & y | z");
        match &expr.kind {
            ExprKind::Binary {
                left,
                op: BinaryOp::BitOr,
                ..
            } => {
                assert!(matches!(left.kind, ExprKind::Binary { op: BinaryOp::BitAnd, .. }));
            }
            other => panic!("expected BitOr with BitAnd left, got {:?}", other),
        }

        // x << y == z => (x << y) == z
        let expr = parse_expression("x << y == z");
        match &expr.kind {
            ExprKind::Binary {
                left,
                op: BinaryOp::Eq,
                ..
            } => {
                assert!(matches!(left.kind, ExprKind::Binary { op: BinaryOp::Shl, .. }));
            }
            other => panic!("expected Eq with Shl left, got {:?}", other),
        }
    }

    #[test]
    fn test_string_concat_vs_range() {
        let expr = parse_expression("\"str\" .. 10");
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Concat,
                ..
            } => {}
            other => panic!("expected Concat for \"str\" .. 10, got {:?}", other),
        }

        let expr = parse_expression("1..10");
        match &expr.kind {
            ExprKind::Range { .. } => {}
            other => panic!("expected Range for 1..10, got {:?}", other),
        }
    }

    #[test]
    fn test_nested_generics_in_statements() {
        let mut interner = Interner::new();
        let mut lexer = Lexer::new("local x: Array<Array<int>> = [[1]]", &mut interner);
        let tokens = lexer.tokenize().expect("tokenize");
        let mut parser = Parser::new(tokens);
        let items = parser.parse().expect("parse items");
        assert_eq!(items.len(), 1);
    }
}
