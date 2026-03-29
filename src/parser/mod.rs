mod expr_parser;
mod item_parser;
mod stmt_parser;
mod type_parser;
use crate::{
    ast::{Item, ItemKind, Span},
    error::{LustError, Result},
    lexer::{Lexer, Token, TokenKind},
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

    /// Create a parser from a lexer using streaming tokenization.
    /// More memory-efficient for embedded targets.
    pub fn from_lexer(lexer: &mut Lexer<'_>) -> Result<Self> {
        // Pre-allocate based on source size to avoid repeated reallocations
        let estimated_tokens = (lexer.source_len() / 6).max(16);

        #[cfg(feature = "esp32c6-logging")]
        log::info!("Parser::from_lexer: pre-allocating for ~{} tokens", estimated_tokens);

        let mut tokens = Vec::with_capacity(estimated_tokens);

        #[cfg(feature = "esp32c6-logging")]
        log::info!("Parser::from_lexer: collecting tokens...");

        for token_result in lexer.tokenize_iter() {
            tokens.push(token_result?);
        }

        #[cfg(feature = "esp32c6-logging")]
        {
            let with_lexeme = tokens.iter().filter(|t| !t.lexeme.is_empty()).count();
            log::info!("Parser::from_lexer: collected {} tokens ({} with lexemes)", tokens.len(), with_lexeme);
        }

        // Shrink to actual size to save memory
        tokens.shrink_to_fit();

        Ok(Self { tokens, current: 0 })
    }

    /// Returns the number of tokens (for debugging)
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }

    pub fn parse(&mut self) -> Result<Vec<Item>> {
        // Estimate items: roughly 1 item per 20-50 tokens for typical code
        let estimated_items = (self.tokens.len() / 30).max(8);
        let mut items = Vec::with_capacity(estimated_items);

        #[cfg(feature = "esp32c6-logging")]
        log::info!("Parser::parse: starting with capacity for ~{} items", estimated_items);

        while !self.is_at_end() {
            if self.is_item_start() {
                items.push(self.parse_item()?);
            } else {
                let start_line = self.current_token().line;
                let start_column = self.current_token().column;

                let mut stmts = Vec::new();
                while !self.is_at_end() && !self.is_item_start() {
                    stmts.push(self.parse_stmt()?);
                }

                if !stmts.is_empty() {
                    // Get end position without cloning
                    let (end_line, end_column) = if self.current > 0 {
                        let prev_token = &self.tokens[self.current - 1];
                        (prev_token.line, prev_token.column)
                    } else {
                        (start_line, start_column)
                    };

                    items.push(Item::new(
                        ItemKind::Script(stmts),
                        Span::new(start_line, start_column, end_line, end_column),
                    ));
                } else {
                    break;
                }
            }
        }

        #[cfg(feature = "esp32c6-logging")]
        log::info!("Parser::parse: parsed {} items", items.len());

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
        if self.check(TokenKind::Identifier) || self.check(TokenKind::Type) {
            // Avoid cloning the token - just take the lexeme directly
            let lexeme = self.current_token().lexeme.clone();
            self.advance();
            Ok(lexeme)
        } else {
            let token = self.current_token();
            Err(LustError::ParserError {
                line: token.line,
                column: token.column,
                message: format!(
                    "Expected identifier (got {:?}, expected Identifier)",
                    token.kind
                ),
                module: None,
            })
        }
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
