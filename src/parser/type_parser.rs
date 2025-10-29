use super::Parser;
use crate::{
    ast::{Span, Type, TypeKind},
    error::Result,
    lexer::TokenKind,
};
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
                        "Table" => TypeKind::Table,
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

                                    _ => TypeKind::GenericInstance {
                                        name: qualified_name,
                                        type_args,
                                    },
                                }
                            } else if base.len() == 1 && base.chars().next().unwrap().is_uppercase()
                            {
                                TypeKind::Generic(base.clone())
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
                if self.match_token(&[TokenKind::Mut]) {
                    TypeKind::MutRef(Box::new(self.parse_type()?))
                } else {
                    TypeKind::Ref(Box::new(self.parse_type()?))
                }
            }

            TokenKind::Star => {
                self.advance();
                let mutable = self.match_token(&[TokenKind::Mut]);
                TypeKind::Pointer {
                    mutable,
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
                    return Ok(types.into_iter().next().unwrap());
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
        Ok(Type::new(kind, self.make_span(&start_token, &end_token)))
    }

    fn parse_type_arguments(&mut self) -> Result<Vec<Type>> {
        self.consume(TokenKind::Less, "Expected '<' after type name")?;
        let mut type_args = vec![self.parse_type()?];
        while self.match_token(&[TokenKind::Comma]) {
            type_args.push(self.parse_type()?);
        }

        self.consume(TokenKind::Greater, "Expected '>' after type arguments")?;
        Ok(type_args)
    }
}
