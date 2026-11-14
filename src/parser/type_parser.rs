use super::Parser;
use crate::{
    ast::{Span, Type, TypeKind},
    error::Result,
    lexer::TokenKind,
};
use alloc::{boxed::Box, format, vec, vec::Vec};
impl Parser {
    pub(super) fn parse_type(&mut self) -> Result<Type> {
        let start_token = self.current_token().clone();
        let first_type = self.parse_primary_type()?;
        if self.check(TokenKind::Pipe) {
            let mut types = vec![first_type];
            while self.match_token(&[TokenKind::Pipe]) {
                types.push(self.parse_primary_type()?);
            }

            let end_token = self.tokens[self.current - 1].clone();
            return Ok(Type::new(
                TypeKind::Union(types),
                self.make_span(&start_token, &end_token),
            ));
        }

        Ok(first_type)
    }

    fn parse_primary_type(&mut self) -> Result<Type> {
        let start_token = self.current_token().clone();
        let kind = match self.peek_kind() {
            TokenKind::Identifier => {
                let first_segment = self.expect_identifier()?;
                let mut segments = vec![first_segment.clone()];
                while self.match_token(&[TokenKind::Dot]) {
                    let segment = self.expect_identifier()?;
                    segments.push(segment);
                }

                let qualified_name = segments.join(".");
                let is_single_segment = segments.len() == 1;
                let base = &segments[0];
                if is_single_segment {
                    match base.as_str() {
                        "int" => TypeKind::Int,
                        "float" => TypeKind::Float,
                        "string" => TypeKind::String,
                        "bool" => TypeKind::Bool,
                        "unknown" => TypeKind::Unknown,
                        "never" => TypeKind::Never,
                        "typeof" => {
                            let first_seg = self.expect_identifier()?;
                            let mut path = first_seg;
                            while self.match_token(&[TokenKind::Dot]) {
                                let seg = self.expect_identifier()?;
                                path.push('.');
                                path.push_str(&seg);
                            }
                            TypeKind::Typeof(path)
                        }
                        "Effect" if self.check(TokenKind::Less) => {
                            return self.parse_effect_type_args(&start_token);
                        }
                        _ => {
                            if self.check(TokenKind::Less) {
                                let type_args = self.parse_type_arguments()?;
                                match base.as_str() {
                                    "Array" if type_args.len() == 1 => TypeKind::Array(Box::new(
                                        type_args.into_iter().next().unwrap(),
                                    )),
                                    "Map" if type_args.len() == 2 => {
                                        let mut iter = type_args.into_iter();
                                        TypeKind::Map(
                                            Box::new(iter.next().unwrap()),
                                            Box::new(iter.next().unwrap()),
                                        )
                                    }

                                    "Option" if type_args.len() == 1 => TypeKind::Option(Box::new(
                                        type_args.into_iter().next().unwrap(),
                                    )),
                                    "Result" if type_args.len() == 2 => {
                                        let mut iter = type_args.into_iter();
                                        TypeKind::Result(
                                            Box::new(iter.next().unwrap()),
                                            Box::new(iter.next().unwrap()),
                                        )
                                    }

                                    "Infer" if type_args.len() == 1 => TypeKind::InferType(Box::new(
                                        type_args.into_iter().next().unwrap(),
                                    )),
                                    "InferInput" if type_args.len() == 1 => TypeKind::InferInput(Box::new(
                                        type_args.into_iter().next().unwrap(),
                                    )),
                                    "InferOutput" if type_args.len() == 1 => TypeKind::InferOutput(Box::new(
                                        type_args.into_iter().next().unwrap(),
                                    )),

                                    _ => TypeKind::GenericInstance {
                                        name: qualified_name,
                                        type_args,
                                    },
                                }
                            } else {
                                TypeKind::Named(qualified_name)
                            }
                        }
                    }
                } else if self.check(TokenKind::Less) {
                    let type_args = self.parse_type_arguments()?;
                    TypeKind::GenericInstance {
                        name: qualified_name,
                        type_args,
                    }
                } else {
                    TypeKind::Named(qualified_name)
                }
            }

            TokenKind::Ampersand => {
                self.advance();
                TypeKind::Ref(Box::new(self.parse_type()?))
            }

            TokenKind::Star => {
                self.advance();
                TypeKind::Pointer {
                    mutable: true,
                    pointee: Box::new(self.parse_type()?),
                }
            }

            TokenKind::Function => {
                self.advance();
                self.consume(TokenKind::LeftParen, "Expected '(' after 'function'")?;
                let mut params = Vec::new();
                if !self.check(TokenKind::RightParen) {
                    params.push(self.parse_type()?);
                    while self.match_token(&[TokenKind::Comma]) {
                        params.push(self.parse_type()?);
                    }
                }

                self.consume(
                    TokenKind::RightParen,
                    "Expected ')' after function parameters",
                )?;
                let return_type = if self.match_token(&[TokenKind::Colon]) {
                    Box::new(self.parse_type()?)
                } else {
                    Box::new(Type::new(TypeKind::Unit, Span::dummy()))
                };
                let (error_type, requirements) = self.parse_optional_effects_spec()?;
                let return_type = if error_type.is_some() || !requirements.is_empty() {
                    let span = return_type.span;
                    Box::new(Type::new(
                        TypeKind::Effect {
                            success: return_type,
                            error: Box::new(
                                error_type.unwrap_or_else(|| Type::new(TypeKind::Never, Span::dummy())),
                            ),
                            requirements,
                        },
                        span,
                    ))
                } else {
                    return_type
                };
                TypeKind::Function {
                    params,
                    return_type,
                }
            }

            TokenKind::LeftParen => {
                self.advance();
                if self.check(TokenKind::RightParen) {
                    self.consume(TokenKind::RightParen, "Expected ')' after type")?;
                    return Ok(Type::new(TypeKind::Unit, Span::dummy()));
                }

                let first_type = self.parse_type()?;
                let mut types = vec![first_type];
                while self.match_token(&[TokenKind::Comma]) {
                    types.push(self.parse_type()?);
                }

                let end_token = self.current_token().clone();
                self.consume(TokenKind::RightParen, "Expected ')' after type")?;
                if types.len() == 1 {
                    let mut result_type = types.into_iter().next().unwrap();
                    while self.match_token(&[TokenKind::Question]) {
                        let q_end = self.tokens[self.current - 1].clone();
                        let span = self.make_span(&start_token, &q_end);
                        result_type = Type::new(TypeKind::Option(Box::new(result_type)), span);
                    }
                    return Ok(result_type);
                }

                return Ok(Type::new(
                    TypeKind::Tuple(types),
                    self.make_span(&start_token, &end_token),
                ));
            }

            _ => {
                return Err(self.error(&format!("Expected type, got {:?}", self.peek_kind())));
            }
        };
        let end_token = self.tokens[self.current - 1].clone();
        let mut result_type = Type::new(kind, self.make_span(&start_token, &end_token));
        while self.match_token(&[TokenKind::Question]) {
            let q_end = self.tokens[self.current - 1].clone();
            let span = self.make_span(&start_token, &q_end);
            result_type = Type::new(TypeKind::Option(Box::new(result_type)), span);
        }
        Ok(result_type)
    }

    pub(super) fn parse_type_arguments(&mut self) -> Result<Vec<Type>> {
        self.consume(TokenKind::Less, "Expected '<' after type name")?;
        let mut type_args = vec![self.parse_type()?];
        while self.match_token(&[TokenKind::Comma]) {
            if self.check(TokenKind::Greater) || self.check(TokenKind::DoubleGreater) {
                break;
            }
            type_args.push(self.parse_type()?);
        }

        self.consume_closing_angle_bracket("Expected '>' after type arguments")?;
        Ok(type_args)
    }

    pub(super) fn consume_closing_angle_bracket(&mut self, message: &str) -> Result<()> {
        if self.check(TokenKind::Greater) {
            self.advance();
            Ok(())
        } else if self.check(TokenKind::DoubleGreater) {
            let token = &mut self.tokens[self.current];
            token.kind = TokenKind::Greater;
            token.column += 1;
            Ok(())
        } else {
            let token = self.current_token();
            Err(crate::error::LustError::ParserError {
                line: token.line,
                column: token.column,
                message: format!("{} (got {:?}, expected '>')", message, token.kind),
                module: None,
            })
        }
    }

    fn parse_effect_type_args(&mut self, start_token: &crate::lexer::Token) -> Result<Type> {
        self.consume(TokenKind::Less, "Expected '<' after 'Effect'")?;
        let success = Box::new(self.parse_type()?);
        let (error, requirements) = if self.match_token(&[TokenKind::Comma]) {
            let err = Box::new(self.parse_type()?);
            let reqs = if self.match_token(&[TokenKind::Comma]) {
                let mut r = Vec::new();
                if !self.check(TokenKind::Greater) && !self.check(TokenKind::DoubleGreater) {
                    if self.match_token(&[TokenKind::LeftParen]) {
                        if !self.check(TokenKind::RightParen) {
                            r.push(self.parse_type()?);
                            while self.match_token(&[TokenKind::Plus]) {
                                r.push(self.parse_type()?);
                            }
                        }
                        self.consume(TokenKind::RightParen, "Expected ')' after requirements")?;
                    } else {
                        r.push(self.parse_type()?);
                        while self.match_token(&[TokenKind::Plus]) {
                            r.push(self.parse_type()?);
                        }
                    }
                }
                r
            } else {
                Vec::new()
            };
            (err, reqs)
        } else {
            (
                Box::new(Type::new(TypeKind::Never, Span::dummy())),
                Vec::new(),
            )
        };
        self.consume_closing_angle_bracket("Expected '>' after type arguments")?;
        let end_token = self.tokens[self.current - 1].clone();
        Ok(Type::new(
            TypeKind::Effect {
                success,
                error,
                requirements,
            },
            self.make_span(start_token, &end_token),
        ))
    }

    pub(super) fn parse_requirements_row(&mut self) -> Result<Vec<Type>> {
        let mut reqs = Vec::new();
        if self.match_token(&[TokenKind::LeftParen]) {
            if !self.check(TokenKind::RightParen) {
                reqs.push(self.parse_type()?);
                while self.match_token(&[TokenKind::Plus]) {
                    reqs.push(self.parse_type()?);
                }
            }
            self.consume(TokenKind::RightParen, "Expected ')' after requirements list")?;
        } else {
            reqs.push(self.parse_type()?);
            while self.match_token(&[TokenKind::Plus]) {
                reqs.push(self.parse_type()?);
            }
        }
        Ok(reqs)
    }

    pub(super) fn parse_optional_effects_spec(&mut self) -> Result<(Option<Type>, Vec<Type>)> {
        let mut error_type = None;
        let mut requirements = Vec::new();
        if self.match_token(&[TokenKind::Bang]) {
            error_type = Some(self.parse_type()?);
        }
        if self.match_token(&[TokenKind::Tilde]) {
            requirements = self.parse_requirements_row()?;
        }
        Ok((error_type, requirements))
    }
}
