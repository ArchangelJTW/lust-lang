use super::Parser;
use crate::{
    ast::{BinaryOp, Expr, ExprKind, Literal, Pattern, Span, StructLiteralField, UnaryOp},
    error::{LustError, Result},
    lexer::{Token, TokenKind},
};
impl Parser {
    pub(super) fn parse_expr(&mut self) -> Result<Expr> {
        self.parse_assignment()
    }

    fn parse_assignment(&mut self) -> Result<Expr> {
        self.parse_logical_or()
    }

    fn parse_logical_or(&mut self) -> Result<Expr> {
        let mut expr = self.parse_logical_and()?;
        while self.match_token(&[TokenKind::Or]) {
            let op = BinaryOp::Or;
            let right = self.parse_logical_and()?;
            let span = expr.span;
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

    fn parse_logical_and(&mut self) -> Result<Expr> {
        let mut expr = self.parse_comparison()?;
        while self.match_token(&[TokenKind::And]) {
            let op = BinaryOp::And;
            let right = self.parse_comparison()?;
            let span = expr.span;
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
            let span = expr.span;
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
        let expr = self.parse_concat()?;
        if self.match_token(&[TokenKind::DoubleDot]) {
            let end = self.parse_concat()?;
            let span = expr.span;
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

    fn parse_concat(&mut self) -> Result<Expr> {
        let mut expr = self.parse_term()?;
        while self.check(TokenKind::DoubleDot) {
            if let Some(next) = self.peek_ahead(1) {
                if matches!(next.kind, TokenKind::Integer | TokenKind::Float) {
                    break;
                }
            }

            self.advance();
            let right = self.parse_term()?;
            let span = expr.span;
            expr = Expr::new(
                ExprKind::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::Concat,
                    right: Box::new(right),
                },
                span,
            );
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
            let span = expr.span;
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
        let mut expr = self.parse_power()?;
        while self.match_token(&[TokenKind::Star, TokenKind::Slash, TokenKind::Percent]) {
            let op = match self.tokens[self.current - 1].kind {
                TokenKind::Star => BinaryOp::Mul,
                TokenKind::Slash => BinaryOp::Div,
                TokenKind::Percent => BinaryOp::Mod,
                _ => unreachable!(),
            };
            let right = self.parse_power()?;
            let span = expr.span;
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

    fn parse_power(&mut self) -> Result<Expr> {
        let expr = self.parse_unary()?;
        if self.match_token(&[TokenKind::Caret]) {
            let right = self.parse_power()?;
            let span = expr.span;
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

    fn parse_unary(&mut self) -> Result<Expr> {
        if self.match_token(&[TokenKind::Minus, TokenKind::Not]) {
            let start_token = self.tokens[self.current - 1].clone();
            let op = match start_token.kind {
                TokenKind::Minus => UnaryOp::Neg,
                TokenKind::Not => UnaryOp::Not,
                _ => unreachable!(),
            };
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

        self.parse_postfix()
    }

    fn parse_postfix(&mut self) -> Result<Expr> {
        let mut expr = self.parse_primary()?;
        loop {
            match self.peek_kind() {
                TokenKind::LeftParen => {
                    self.advance();
                    let mut args = Vec::new();
                    if !self.check(TokenKind::RightParen) {
                        args.push(self.parse_expr()?);
                        while self.match_token(&[TokenKind::Comma]) {
                            args.push(self.parse_expr()?);
                        }
                    }

                    self.consume(TokenKind::RightParen, "Expected ')' after arguments")?;
                    let span = expr.span;
                    expr = Expr::new(
                        ExprKind::Call {
                            callee: Box::new(expr),
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

                        self.consume(TokenKind::Greater, "Expected '>' after type arguments")?;
                        Some(types)
                    } else {
                        None
                    };
                    self.consume(TokenKind::LeftParen, "Expected '(' after method name")?;
                    let mut args = Vec::new();
                    if !self.check(TokenKind::RightParen) {
                        args.push(self.parse_expr()?);
                        while self.match_token(&[TokenKind::Comma]) {
                            args.push(self.parse_expr()?);
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
                    let span = expr.span;
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
                    let span = expr.span;
                    let is_pattern = if self.check(TokenKind::Identifier) {
                        self.peek_ahead(1)
                            .map(|t| t.kind == TokenKind::LeftParen)
                            .unwrap_or(false)
                    } else {
                        false
                    };
                    if is_pattern {
                        let pattern = self.parse_pattern()?;
                        expr = Expr::new(
                            ExprKind::IsPattern {
                                expr: Box::new(expr),
                                pattern,
                            },
                            span,
                        );
                    } else {
                        let check_type = self.parse_type()?;
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
                    let span = expr.span;
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

    fn parse_primary(&mut self) -> Result<Expr> {
        let start_token = self.current_token().clone();
        let kind = match self.peek_kind() {
            TokenKind::Integer => {
                let token = self.advance();
                let value = token
                    .lexeme
                    .parse::<i64>()
                    .map_err(|_| self.error("Invalid integer literal"))?;
                ExprKind::Literal(Literal::Integer(value))
            }

            TokenKind::Float => {
                let token = self.advance();
                let value = token
                    .lexeme
                    .parse::<f64>()
                    .map_err(|_| self.error("Invalid float literal"))?;
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

            TokenKind::Identifier => {
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
                    return Ok(Expr::new(
                        ExprKind::Paren(Box::new(expressions.into_iter().next().unwrap())),
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
                    .map_or(false, |t| t.kind == TokenKind::LeftParen)
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
                    self.consume(TokenKind::Equal, "Expected '=' after map key")?;
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

    pub(super) fn parse_pattern(&mut self) -> Result<Pattern> {
        match self.peek_kind() {
            TokenKind::Identifier => {
                let name = self.expect_identifier()?;
                if self.check(TokenKind::LeftParen) {
                    self.advance();
                    let mut bindings = Vec::new();
                    if !self.check(TokenKind::RightParen) {
                        bindings.push(self.parse_pattern()?);
                        while self.match_token(&[TokenKind::Comma]) {
                            bindings.push(self.parse_pattern()?);
                        }
                    }

                    self.consume(TokenKind::RightParen, "Expected ')' after pattern")?;
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

            TokenKind::Integer
            | TokenKind::Float
            | TokenKind::String
            | TokenKind::True
            | TokenKind::False => {
                let token = self.advance().clone();
                let lit = match token.kind {
                    TokenKind::Integer => Literal::Integer(token.lexeme.parse().unwrap()),
                    TokenKind::Float => Literal::Float(token.lexeme.parse().unwrap()),
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
                    })
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
