use super::Parser;
use crate::{
    ast::{
        EnumDef, EnumVariant, ExternItem, ExternParam, FieldOwnership, FunctionDef, FunctionParam,
        ImplBlock, Item, ItemKind, StructDef, StructField, TraitBound, TraitDef, TraitMethod, Type,
        TypeKind, Visibility,
    },
    error::Result,
    lexer::TokenKind,
};
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
impl Parser {
    #[inline(never)]
    pub(super) fn parse_item(&mut self) -> Result<Item> {
        self.collect_pending_docs();
        let doc = self.take_pending_docs();
        let start_token = self.current_token().clone();
        let visibility = if self.match_token(&[TokenKind::Local]) {
            Visibility::Private
        } else {
            Visibility::Public
        };

        #[cfg(feature = "esp32c6-logging")]
        log::info!(
            "parse_item: parsing {:?} at line {}",
            self.peek_kind(),
            start_token.line
        );

        let kind = match self.peek_kind() {
            TokenKind::Function => {
                #[cfg(feature = "esp32c6-logging")]
                log::info!("  parsing function");

                let func_def = self.parse_function(visibility, doc)?;
                ItemKind::Function(func_def)
            }

            TokenKind::Struct => {
                let struct_def = self.parse_struct(visibility, doc)?;
                ItemKind::Struct(struct_def)
            }

            TokenKind::Enum => {
                let enum_def = self.parse_enum(visibility, doc)?;
                ItemKind::Enum(enum_def)
            }

            TokenKind::Trait => {
                let trait_def = self.parse_trait(visibility, doc)?;
                ItemKind::Trait(trait_def)
            }

            TokenKind::Impl => {
                let impl_block = self.parse_impl()?;
                ItemKind::Impl(impl_block)
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
                                        public: false,
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
                                public: false,
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
                                public: false,
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
                    public: false,
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
                let mut items = Vec::new();
                let mut block_doc = doc;
                let uses_braces = if self.check(TokenKind::LeftBrace) {
                    self.advance();
                    true
                } else {
                    false
                };
                let terminator = if uses_braces {
                    TokenKind::RightBrace
                } else {
                    TokenKind::End
                };
                while !self.check(terminator) && !self.is_at_end() {
                    self.collect_pending_docs();
                    let mut item_doc = self.take_pending_docs();
                    if item_doc.is_none() {
                        item_doc = block_doc.take();
                    }
                    match self.peek_kind() {
                        TokenKind::Function => {
                            self.advance();
                            let mut name = self.expect_identifier()?;
                            loop {
                                if self.check(TokenKind::Colon) {
                                    self.advance();
                                    let part = self.expect_identifier()?;
                                    name.push(':');
                                    name.push_str(&part);
                                } else if self.check(TokenKind::Dot) {
                                    self.advance();
                                    let part = self.expect_identifier()?;
                                    name.push('.');
                                    name.push_str(&part);
                                } else {
                                    break;
                                }
                            }
                            self.consume(TokenKind::LeftParen, "Expected '(' after function name")?;
                            let mut params = Vec::new();
                            if !self.check(TokenKind::RightParen) {
                                params.push(self.parse_extern_param()?);
                                while self.match_token(&[TokenKind::Comma]) {
                                    params.push(self.parse_extern_param()?);
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
                                doc: item_doc,
                            });
                        }

                        TokenKind::Identifier if self.current_token().lexeme == "const" => {
                            self.advance();
                            let mut name = self.expect_identifier()?;
                            while self.check(TokenKind::Dot) {
                                self.advance();
                                let part = self.expect_identifier()?;
                                name.push('.');
                                name.push_str(&part);
                            }
                            self.consume(TokenKind::Colon, "Expected ':' after const name")?;
                            let ty = self.parse_type()?;
                            items.push(ExternItem::Const {
                                name,
                                ty,
                                doc: item_doc,
                            });
                        }

                        TokenKind::Struct => {
                            let struct_def = self.parse_struct(Visibility::Public, item_doc)?;
                            items.push(ExternItem::Struct(struct_def));
                        }

                        TokenKind::Enum => {
                            let enum_def = self.parse_enum(Visibility::Public, item_doc)?;
                            items.push(ExternItem::Enum(enum_def));
                        }

                        _ => {
                            return Err(self.error(
                                "Expected 'function', 'const', 'struct', or 'enum' in extern block",
                            ));
                        }
                    }
                }

                if uses_braces {
                    self.consume(TokenKind::RightBrace, "Expected '}' after extern block")?;
                    // allow optional trailing 'end' for compatibility
                    if self.match_token(&[TokenKind::End]) {
                        // consumed optional end
                    }
                } else {
                    self.consume(TokenKind::End, "Expected 'end' after extern block")?;
                }
                ItemKind::Extern { abi, items }
            }

            _ => {
                return Err(self.error(&format!("Expected item, got {:?}", self.peek_kind())));
            }
        };
        let end_token = self.tokens[self.current - 1].clone();
        Ok(Item::new(kind, self.make_span(&start_token, &end_token)))
    }

    fn parse_function(
        &mut self,
        visibility: Visibility,
        doc: Option<String>,
    ) -> Result<FunctionDef> {
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

        #[cfg(feature = "esp32c6-logging")]
        log::info!("    parsing body for function '{}'", name);

        // Estimate statement count based on remaining tokens
        let remaining_tokens = self.tokens.len() - self.current;
        let estimated_stmts = (remaining_tokens / 5).max(4); // Rough estimate: 5 tokens per statement
        let mut body = Vec::with_capacity(estimated_stmts);

        #[cfg(feature = "esp32c6-logging")]
        let mut stmt_count = 0;
        while !self.check(TokenKind::End) && !self.is_at_end() {
            #[cfg(feature = "esp32c6-logging")]
            {
                stmt_count += 1;
                if stmt_count % 10 == 0 {
                    log::info!("    parsed {} statements", stmt_count);
                }
            }

            body.push(self.parse_stmt()?);
        }

        #[cfg(feature = "esp32c6-logging")]
        log::info!("    function '{}' has {} statements", name, stmt_count);

        self.consume(TokenKind::End, "Expected 'end' after function body")?;

        // Shrink the body vector to actual size to save memory
        body.shrink_to_fit();

        Ok(FunctionDef {
            name,
            type_params,
            trait_bounds,
            params,
            return_type,
            body,
            is_method,
            visibility,
            doc,
        })
    }

    fn parse_struct(&mut self, visibility: Visibility, doc: Option<String>) -> Result<StructDef> {
        self.consume(TokenKind::Struct, "Expected 'struct'")?;
        let name = self.expect_identifier()?;
        let (type_params, trait_bounds) = self.parse_type_params_with_bounds()?;
        let mut fields = Vec::new();
        while !self.check(TokenKind::End) && !self.is_at_end() {
            while self.check(TokenKind::DocComment) {
                self.advance();
            }
            if self.check(TokenKind::End) {
                break;
            }
            let field_vis = if self.match_token(&[TokenKind::Local]) {
                Visibility::Private
            } else {
                Visibility::Public
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
            doc,
        })
    }

    fn parse_enum(&mut self, visibility: Visibility, doc: Option<String>) -> Result<EnumDef> {
        self.consume(TokenKind::Enum, "Expected 'enum'")?;
        let name = self.expect_identifier()?;
        let (type_params, trait_bounds) = self.parse_type_params_with_bounds()?;
        let mut variants = Vec::new();
        while !self.check(TokenKind::End) && !self.is_at_end() {
            while self.check(TokenKind::DocComment) {
                self.advance();
            }
            if self.check(TokenKind::End) {
                break;
            }
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
            doc,
        })
    }

    fn parse_trait(&mut self, visibility: Visibility, doc: Option<String>) -> Result<TraitDef> {
        self.consume(TokenKind::Trait, "Expected 'trait'")?;
        let name = self.expect_identifier()?;
        let type_params = if self.match_token(&[TokenKind::Less]) {
            let mut params = vec![self.expect_identifier()?];
            while self.match_token(&[TokenKind::Comma]) {
                if self.check(TokenKind::Greater) {
                    break;
                }
                params.push(self.expect_identifier()?);
            }
            self.consume(TokenKind::Greater, "Expected '>' after type parameters")?;
            params
        } else {
            vec![]
        };
        let mut methods = Vec::new();
        while !self.check(TokenKind::End) && !self.is_at_end() {
            self.collect_pending_docs();
            self.consume(TokenKind::Function, "Expected 'function' in trait")?;
            let method_name = self.expect_identifier()?;
            let method_type_params = if self.match_token(&[TokenKind::Less]) {
                let mut params = vec![self.expect_identifier()?];
                while self.match_token(&[TokenKind::Comma]) {
                    if self.check(TokenKind::Greater) {
                        break;
                    }
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
            let method_doc = self.take_pending_docs();
            methods.push(TraitMethod {
                name: method_name,
                type_params: method_type_params,
                params,
                return_type,
                default_impl,
                doc: method_doc,
            });
        }

        self.consume(TokenKind::End, "Expected 'end' after trait methods")?;
        Ok(TraitDef {
            name,
            type_params,
            methods,
            visibility,
            doc,
        })
    }

    fn parse_impl(&mut self) -> Result<ImplBlock> {
        self.consume(TokenKind::Impl, "Expected 'impl'")?;
        let (type_params, where_clause) = self.parse_type_params_with_bounds()?;
        let first_name = self.expect_identifier()?;
        let (trait_name, target_type) = if self.match_token(&[TokenKind::For]) {
            (Some(first_name), self.parse_type()?)
        } else {
            let target_type = if self.check(TokenKind::Less) {
                crate::ast::Type::new(
                    crate::ast::TypeKind::GenericInstance {
                        name: first_name,
                        type_args: self.parse_type_arguments()?,
                    },
                    crate::ast::Span::dummy(),
                )
            } else {
                crate::ast::Type::new(
                    crate::ast::TypeKind::Named(first_name),
                    crate::ast::Span::dummy(),
                )
            };
            (None, target_type)
        };
        let mut methods = Vec::new();
        while !self.check(TokenKind::End) && !self.is_at_end() {
            let visibility = if self.match_token(&[TokenKind::Local]) {
                Visibility::Private
            } else {
                Visibility::Public
            };
            self.collect_pending_docs();
            let method_doc = self.take_pending_docs();
            let func_def = self.parse_function(visibility, method_doc)?;
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

    fn parse_extern_param(&mut self) -> Result<ExternParam> {
        let name = if self.check(TokenKind::Identifier)
            && self
                .peek_ahead(1)
                .is_some_and(|token| token.kind == TokenKind::Colon)
        {
            let param_name = self.expect_identifier()?;
            self.consume(TokenKind::Colon, "Expected ':' after parameter name")?;
            Some(param_name)
        } else {
            None
        };
        let ty = self.parse_type()?;
        Ok(ExternParam { name, ty })
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
            if self.check(TokenKind::Greater) {
                break;
            }
        }

        self.consume(TokenKind::Greater, "Expected '>' after type parameters")?;
        Ok((type_params, trait_bounds))
    }
}
