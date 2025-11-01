mod expr_parser;
mod item_parser;
mod stmt_parser;
mod type_parser;
use crate::{
    ast::{Item, ItemKind, Span},
    error::{LustError, Result},
    lexer::{Token, TokenKind},
};
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, current: 0 }
    }

    pub fn parse(&mut self) -> Result<Vec<Item>> {
        let mut items = Vec::new();
        while !self.is_at_end() {
            if self.is_item_start() {
                items.push(self.parse_item()?);
            } else {
                let start_token = self.current_token().clone();
                let mut stmts = Vec::new();
                while !self.is_at_end() && !self.is_item_start() {
                    stmts.push(self.parse_stmt()?);
                }

                if !stmts.is_empty() {
                    let end_token = if self.current > 0 {
                        self.tokens[self.current - 1].clone()
                    } else {
                        self.current_token().clone()
                    };
                    items.push(Item::new(
                        ItemKind::Script(stmts),
                        self.make_span(&start_token, &end_token),
                    ));
                } else {
                    break;
                }
            }
        }

        Ok(items)
    }

    fn is_item_start(&self) -> bool {
        match self.peek_kind() {
            TokenKind::Function
            | TokenKind::Struct
            | TokenKind::Enum
            | TokenKind::Trait
            | TokenKind::Impl
            | TokenKind::Type
            | TokenKind::Const
            | TokenKind::Static
            | TokenKind::Use
            | TokenKind::Module
            | TokenKind::Extern => true,
            TokenKind::Pub => self.peek_ahead(1).map_or(false, |t| {
                matches!(
                    t.kind,
                    TokenKind::Function
                        | TokenKind::Struct
                        | TokenKind::Enum
                        | TokenKind::Trait
                        | TokenKind::Impl
                        | TokenKind::Type
                        | TokenKind::Const
                        | TokenKind::Static
                        | TokenKind::Use
                        | TokenKind::Module
                        | TokenKind::Extern
                )
            }),
            _ => false,
        }
    }

    fn current_token(&self) -> &Token {
        &self.tokens[self.current]
    }

    fn peek_kind(&self) -> TokenKind {
        self.current_token().kind.clone()
    }

    fn peek_ahead(&self, n: usize) -> Option<&Token> {
        self.tokens.get(self.current + n)
    }

    fn advance(&mut self) -> &Token {
        if !self.is_at_end() {
            self.current += 1;
        }

        &self.tokens[self.current - 1]
    }

    fn is_at_end(&self) -> bool {
        self.peek_kind() == TokenKind::Eof
    }

    fn check(&self, kind: TokenKind) -> bool {
        if self.is_at_end() {
            return false;
        }

        self.peek_kind() == kind
    }

    fn match_token(&mut self, kinds: &[TokenKind]) -> bool {
        for kind in kinds {
            if self.check(kind.clone()) {
                self.advance();
                return true;
            }
        }

        false
    }

    fn consume(&mut self, kind: TokenKind, message: &str) -> Result<&Token> {
        if self.check(kind) {
            Ok(self.advance())
        } else {
            let token = self.current_token();
            Err(LustError::ParserError {
                line: token.line,
                column: token.column,
                message: format!("{} (got {:?}, expected {:?})", message, token.kind, kind),
                module: None,
            })
        }
    }

    fn expect_identifier(&mut self) -> Result<String> {
        let token = self.consume(TokenKind::Identifier, "Expected identifier")?;
        Ok(token.lexeme.clone())
    }

    fn make_span(&self, start_token: &Token, end_token: &Token) -> Span {
        Span::new(
            start_token.line,
            start_token.column,
            end_token.line,
            end_token.column,
        )
    }

    fn error(&self, message: &str) -> LustError {
        let token = self.current_token();
        LustError::ParserError {
            line: token.line,
            column: token.column,
            message: message.to_string(),
            module: None,
        }
    }

    #[allow(dead_code)]
    fn synchronize(&mut self) {
        self.advance();
        while !self.is_at_end() {
            match self.peek_kind() {
                TokenKind::Function
                | TokenKind::Local
                | TokenKind::If
                | TokenKind::While
                | TokenKind::For
                | TokenKind::Return
                | TokenKind::Struct
                | TokenKind::Enum
                | TokenKind::Trait
                | TokenKind::Impl => return,
                _ => {}
            }

            self.advance();
        }
    }
}
