use super::Parser;
use crate::{
    ast::{
        EnumDef, EnumVariant, ExternItem, FieldOwnership, FunctionDef, FunctionParam, ImplBlock,
        Item, ItemKind, StructDef, StructField, TraitBound, TraitDef, TraitMethod, Type, TypeKind,
        Visibility,
    },
    error::Result,
    lexer::TokenKind,
};
impl Parser {
    pub(super) fn parse_item(&mut self) -> Result<Item> {
        let start_token = self.current_token().clone();
        let visibility = if self.match_token(&[TokenKind::Pub]) {
            Visibility::Public
        } else {
            Visibility::Private
        };
        let kind = match self.peek_kind() {
            TokenKind::Function => {
                let func_def = self.parse_function(visibility)?;
                ItemKind::Function(func_def)
            }

            TokenKind::Struct => {
                let struct_def = self.parse_struct(visibility)?;
                ItemKind::Struct(struct_def)
            }

            TokenKind::Enum => {
                let enum_def = self.parse_enum(visibility)?;
                ItemKind::Enum(enum_def)
            }

            TokenKind::Trait => {
                let trait_def = self.parse_trait(visibility)?;
                ItemKind::Trait(trait_def)
            }

            TokenKind::Impl => {
                let impl_block = self.parse_impl()?;
                ItemKind::Impl(impl_block)
            }

            TokenKind::Type => {
                self.advance();
                let name = self.expect_identifier()?;
                let type_params = if self.match_token(&[TokenKind::Less]) {
                    let mut params = vec![self.expect_identifier()?];
                    while self.match_token(&[TokenKind::Comma]) {
                        params.push(self.expect_identifier()?);
                    }

                    self.consume(TokenKind::Greater, "Expected '>' after type parameters")?;
                    params
                } else {
                    vec![]
                };
                self.consume(TokenKind::Equal, "Expected '=' after type alias name")?;
                let target = self.parse_type()?;
                ItemKind::TypeAlias {
                    name,
                    type_params,
                    target,
                }
            }

            TokenKind::Const => {
                self.advance();
                let name = self.expect_identifier()?;
                self.consume(TokenKind::Colon, "Expected ':' after const name")?;
                let ty = self.parse_type()?;
                self.consume(TokenKind::Equal, "Expected '=' after const type")?;
                let value = self.parse_expr()?;
                ItemKind::Const { name, ty, value }
            }

            TokenKind::Static => {
                self.advance();
                let mutable = self.match_token(&[TokenKind::Mut]);
                let name = self.expect_identifier()?;
                self.consume(TokenKind::Colon, "Expected ':' after static name")?;
                let ty = self.parse_type()?;
                self.consume(TokenKind::Equal, "Expected '=' after static type")?;
                let value = self.parse_expr()?;
                ItemKind::Static {
                    name,
                    mutable,
                    ty,
                    value,
                }
            }

            TokenKind::Use => {
                self.advance();
                let mut path = vec![self.expect_identifier()?];
                while self.match_token(&[TokenKind::Dot]) {
                    if self.match_token(&[TokenKind::LeftBrace]) {
                        let mut items = Vec::new();
                        loop {
                            if self.match_token(&[TokenKind::Star]) {
                                self.consume(TokenKind::RightBrace, "Expected '}' after glob")?;
                                return Ok(Item::new(
                                    ItemKind::Use {
                                        public: visibility == Visibility::Public,
                                        tree: crate::ast::UseTree::Glob { prefix: path },
                                    },
                                    self.make_span(&start_token, &self.tokens[self.current - 1]),
                                ));
                            }

                            let mut item_path = vec![self.expect_identifier()?];
                            while self.match_token(&[TokenKind::Dot]) {
                                item_path.push(self.expect_identifier()?);
                            }

                            let alias = if self.match_token(&[TokenKind::As]) {
                                Some(self.expect_identifier()?)
                            } else {
                                None
                            };
                            items.push(crate::ast::UseTreeItem {
                                path: item_path,
                                alias,
                            });
                            if !self.match_token(&[TokenKind::Comma]) {
                                break;
                            }
                        }

                        self.consume(TokenKind::RightBrace, "Expected '}' after use group")?;
                        let end_token = self.tokens[self.current - 1].clone();
                        return Ok(Item::new(
                            ItemKind::Use {
                                public: visibility == Visibility::Public,
                                tree: crate::ast::UseTree::Group {
                                    prefix: path,
                                    items,
                                },
                            },
                            self.make_span(&start_token, &end_token),
                        ));
                    }

                    if self.match_token(&[TokenKind::Star]) {
                        let end_token = self.tokens[self.current - 1].clone();
                        return Ok(Item::new(
                            ItemKind::Use {
                                public: visibility == Visibility::Public,
                                tree: crate::ast::UseTree::Glob { prefix: path },
                            },
                            self.make_span(&start_token, &end_token),
                        ));
                    }

                    path.push(self.expect_identifier()?);
                }

                let alias = if self.match_token(&[TokenKind::As]) {
                    Some(self.expect_identifier()?)
                } else {
                    None
                };
                ItemKind::Use {
                    public: visibility == Visibility::Public,
                    tree: crate::ast::UseTree::Path {
                        path,
                        alias,
                        import_module: false,
                    },
                }
            }

            TokenKind::Module => {
                self.advance();
                let name = self.expect_identifier()?;
                ItemKind::Module {
                    name,
                    items: vec![],
                }
            }

            TokenKind::Extern => {
                self.advance();
                let abi = if self.check(TokenKind::String) {
                    let token = self.advance().clone();
                    self.unescape_string_literal(&token)?
                } else {
                    "C".to_string()
                };
                self.consume(TokenKind::LeftBrace, "Expected '{' after extern")?;
                let mut items = Vec::new();
                while !self.check(TokenKind::RightBrace) && !self.is_at_end() {
                    self.consume(TokenKind::Function, "Expected 'function' in extern block")?;
                    let name = self.expect_identifier()?;
                    self.consume(TokenKind::LeftParen, "Expected '(' after function name")?;
                    let mut params = Vec::new();
                    if !self.check(TokenKind::RightParen) {
                        params.push(self.parse_type()?);
                        while self.match_token(&[TokenKind::Comma]) {
                            params.push(self.parse_type()?);
                        }
                    }

                    self.consume(TokenKind::RightParen, "Expected ')' after parameters")?;
                    let return_type = if self.match_token(&[TokenKind::Colon]) {
                        Some(self.parse_type()?)
                    } else {
                        None
                    };
                    items.push(ExternItem::Function {
                        name,
                        params,
                        return_type,
                    });
                }

                self.consume(TokenKind::RightBrace, "Expected '}' after extern block")?;
                ItemKind::Extern { abi, items }
            }

            _ => {
                return Err(self.error(&format!("Expected item, got {:?}", self.peek_kind())));
            }
        };
        let end_token = self.tokens[self.current - 1].clone();
        Ok(Item::new(kind, self.make_span(&start_token, &end_token)))
    }

    fn parse_function(&mut self, visibility: Visibility) -> Result<FunctionDef> {
        self.consume(TokenKind::Function, "Expected 'function'")?;
        let first_name = self.expect_identifier()?;
        let (name, is_method) = if self.match_token(&[TokenKind::Colon]) {
            let method_name = self.expect_identifier()?;
            (format!("{}:{}", first_name, method_name), true)
        } else if self.match_token(&[TokenKind::Dot]) {
            let func_name = self.expect_identifier()?;
            (format!("{}.{}", first_name, func_name), false)
        } else {
            (first_name, false)
        };
        let (type_params, trait_bounds) = self.parse_type_params_with_bounds()?;
        self.consume(TokenKind::LeftParen, "Expected '(' after function name")?;
        let mut params = Vec::new();
        if !self.check(TokenKind::RightParen) {
            loop {
                let is_self =
                    self.check(TokenKind::Identifier) && self.current_token().lexeme == "self";
                let param_name = self.expect_identifier()?;
                let ty = if is_self && self.peek_kind() != TokenKind::Colon {
                    crate::ast::Type::new(crate::ast::TypeKind::Infer, crate::ast::Span::dummy())
                } else {
                    self.consume(TokenKind::Colon, "Expected ':' after parameter name")?;
                    self.parse_type()?
                };
                params.push(FunctionParam {
                    name: param_name,
                    ty,
                    is_self,
                });
                if !self.match_token(&[TokenKind::Comma]) {
                    break;
                }
            }
        }

        self.consume(TokenKind::RightParen, "Expected ')' after parameters")?;
        let return_type = if self.match_token(&[TokenKind::Colon]) {
            Some(self.parse_type()?)
        } else {
            None
        };
        let mut body = Vec::new();
        while !self.check(TokenKind::End) && !self.is_at_end() {
            body.push(self.parse_stmt()?);
        }

        self.consume(TokenKind::End, "Expected 'end' after function body")?;
        Ok(FunctionDef {
            name,
            type_params,
            trait_bounds,
            params,
            return_type,
            body,
            is_method,
            visibility,
        })
    }

    fn parse_struct(&mut self, visibility: Visibility) -> Result<StructDef> {
        self.consume(TokenKind::Struct, "Expected 'struct'")?;
        let name = self.expect_identifier()?;
        let (type_params, trait_bounds) = self.parse_type_params_with_bounds()?;
        let mut fields = Vec::new();
        while !self.check(TokenKind::End) && !self.is_at_end() {
            let field_vis = if self.match_token(&[TokenKind::Pub]) {
                Visibility::Public
            } else {
                Visibility::Private
            };
            let field_name = self.expect_identifier()?;
            self.consume(TokenKind::Colon, "Expected ':' after field name")?;
            let mut ownership = FieldOwnership::Strong;
            if self.check(TokenKind::Identifier) && self.current_token().lexeme.as_str() == "ref" {
                self.advance();
                ownership = FieldOwnership::Weak;
            }

            let mut field_type = self.parse_type()?;
            let mut weak_target = None;
            if let FieldOwnership::Weak = ownership {
                weak_target = Some(field_type.clone());
                let span = field_type.span;
                field_type = Type::new(TypeKind::Option(Box::new(field_type)), span);
            }

            fields.push(StructField {
                name: field_name,
                ty: field_type,
                visibility: field_vis,
                ownership,
                weak_target,
            });
            self.match_token(&[TokenKind::Comma]);
        }

        self.consume(TokenKind::End, "Expected 'end' after struct fields")?;
        Ok(StructDef {
            name,
            type_params,
            trait_bounds,
            fields,
            visibility,
        })
    }

    fn parse_enum(&mut self, visibility: Visibility) -> Result<EnumDef> {
        self.consume(TokenKind::Enum, "Expected 'enum'")?;
        let name = self.expect_identifier()?;
        let (type_params, trait_bounds) = self.parse_type_params_with_bounds()?;
        let mut variants = Vec::new();
        while !self.check(TokenKind::End) && !self.is_at_end() {
            let variant_name = self.expect_identifier()?;
            let fields = if self.match_token(&[TokenKind::LeftParen]) {
                let mut types = Vec::new();
                if !self.check(TokenKind::RightParen) {
                    types.push(self.parse_type()?);
                    while self.match_token(&[TokenKind::Comma]) {
                        types.push(self.parse_type()?);
                    }
                }

                self.consume(TokenKind::RightParen, "Expected ')' after variant fields")?;
                Some(types)
            } else {
                None
            };
            variants.push(EnumVariant {
                name: variant_name,
                fields,
            });
            self.match_token(&[TokenKind::Comma]);
        }

        self.consume(TokenKind::End, "Expected 'end' after enum variants")?;
        Ok(EnumDef {
            name,
            type_params,
            trait_bounds,
            variants,
            visibility,
        })
    }

    fn parse_trait(&mut self, visibility: Visibility) -> Result<TraitDef> {
        self.consume(TokenKind::Trait, "Expected 'trait'")?;
        let name = self.expect_identifier()?;
        let type_params = if self.match_token(&[TokenKind::Less]) {
            let mut params = vec![self.expect_identifier()?];
            while self.match_token(&[TokenKind::Comma]) {
                params.push(self.expect_identifier()?);
            }

            self.consume(TokenKind::Greater, "Expected '>' after type parameters")?;
            params
        } else {
            vec![]
        };
        let mut methods = Vec::new();
        while !self.check(TokenKind::End) && !self.is_at_end() {
            self.consume(TokenKind::Function, "Expected 'function' in trait")?;
            let method_name = self.expect_identifier()?;
            let method_type_params = if self.match_token(&[TokenKind::Less]) {
                let mut params = vec![self.expect_identifier()?];
                while self.match_token(&[TokenKind::Comma]) {
                    params.push(self.expect_identifier()?);
                }

                self.consume(TokenKind::Greater, "Expected '>' after type parameters")?;
                params
            } else {
                vec![]
            };
            self.consume(TokenKind::LeftParen, "Expected '(' after method name")?;
            let mut params = Vec::new();
            if !self.check(TokenKind::RightParen) {
                loop {
                    let is_self =
                        self.check(TokenKind::Identifier) && self.current_token().lexeme == "self";
                    let param_name = self.expect_identifier()?;
                    let ty = if is_self && self.peek_kind() != TokenKind::Colon {
                        crate::ast::Type::new(
                            crate::ast::TypeKind::Unknown,
                            crate::ast::Span::dummy(),
                        )
                    } else {
                        self.consume(TokenKind::Colon, "Expected ':' after parameter name")?;
                        self.parse_type()?
                    };
                    params.push(FunctionParam {
                        name: param_name,
                        ty,
                        is_self,
                    });
                    if !self.match_token(&[TokenKind::Comma]) {
                        break;
                    }
                }
            }

            self.consume(TokenKind::RightParen, "Expected ')' after parameters")?;
            let return_type = if self.match_token(&[TokenKind::Colon]) {
                Some(self.parse_type()?)
            } else {
                None
            };
            let default_impl = if !self.check(TokenKind::Function) && !self.check(TokenKind::End) {
                let mut body = Vec::new();
                while !self.check(TokenKind::End) && !self.is_at_end() {
                    body.push(self.parse_stmt()?);
                }

                self.consume(TokenKind::End, "Expected 'end' after method body")?;
                Some(body)
            } else {
                None
            };
            methods.push(TraitMethod {
                name: method_name,
                type_params: method_type_params,
                params,
                return_type,
                default_impl,
            });
        }

        self.consume(TokenKind::End, "Expected 'end' after trait methods")?;
        Ok(TraitDef {
            name,
            type_params,
            methods,
            visibility,
        })
    }

    fn parse_impl(&mut self) -> Result<ImplBlock> {
        self.consume(TokenKind::Impl, "Expected 'impl'")?;
        let type_params = if self.match_token(&[TokenKind::Less]) {
            let mut params = vec![self.expect_identifier()?];
            while self.match_token(&[TokenKind::Comma]) {
                params.push(self.expect_identifier()?);
            }

            self.consume(TokenKind::Greater, "Expected '>' after type parameters")?;
            params
        } else {
            vec![]
        };
        let first_name = self.expect_identifier()?;
        let (trait_name, target_type) = if self.match_token(&[TokenKind::For]) {
            (Some(first_name), self.parse_type()?)
        } else {
            (
                None,
                crate::ast::Type::new(
                    crate::ast::TypeKind::Named(first_name),
                    crate::ast::Span::dummy(),
                ),
            )
        };
        let where_clause = Vec::new();
        let mut methods = Vec::new();
        while !self.check(TokenKind::End) && !self.is_at_end() {
            let func_def = self.parse_function(Visibility::Public)?;
            methods.push(func_def);
        }

        self.consume(TokenKind::End, "Expected 'end' after impl methods")?;
        Ok(ImplBlock {
            type_params,
            trait_name,
            target_type,
            methods,
            where_clause,
        })
    }

    fn parse_type_params_with_bounds(&mut self) -> Result<(Vec<String>, Vec<TraitBound>)> {
        if !self.match_token(&[TokenKind::Less]) {
            return Ok((vec![], vec![]));
        }

        let mut type_params = Vec::new();
        let mut trait_bounds = Vec::new();
        loop {
            let param_name = self.expect_identifier()?;
            type_params.push(param_name.clone());
            if self.match_token(&[TokenKind::Colon]) {
                let mut traits = vec![self.expect_identifier()?];
                while self.match_token(&[TokenKind::Plus]) {
                    traits.push(self.expect_identifier()?);
                }

                trait_bounds.push(TraitBound {
                    type_param: param_name,
                    traits,
                });
            }

            if !self.match_token(&[TokenKind::Comma]) {
                break;
            }
        }

        self.consume(TokenKind::Greater, "Expected '>' after type parameters")?;
        Ok((type_params, trait_bounds))
    }
}
