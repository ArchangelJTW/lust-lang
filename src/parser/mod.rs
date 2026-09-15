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
    /// Doc comments collected from the current position, waiting to be
    /// attached to the next declaration.
    pending_docs: Vec<String>,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            current: 0,
            pending_docs: Vec::new(),
        }
    }

    /// Create a parser from a lexer using streaming tokenization.
    /// More memory-efficient for embedded targets.
    pub fn from_lexer(lexer: &mut Lexer<'_>) -> Result<Self> {
        // Pre-allocate based on source size to avoid repeated reallocations
        let estimated_tokens = (lexer.source_len() / 6).max(16);

        #[cfg(feature = "esp32c6-logging")]
        log::info!(
            "Parser::from_lexer: pre-allocating for ~{} tokens",
            estimated_tokens
        );

        let mut tokens = Vec::with_capacity(estimated_tokens);

        #[cfg(feature = "esp32c6-logging")]
        log::info!("Parser::from_lexer: collecting tokens...");

        for token_result in lexer.tokenize_iter() {
            tokens.push(token_result?);
        }

        #[cfg(feature = "esp32c6-logging")]
        {
            let with_lexeme = tokens.iter().filter(|t| !t.lexeme.is_empty()).count();
            log::info!(
                "Parser::from_lexer: collected {} tokens ({} with lexemes)",
                tokens.len(),
                with_lexeme
            );
        }

        // Shrink to actual size to save memory
        tokens.shrink_to_fit();

        Ok(Self {
            tokens,
            current: 0,
            pending_docs: Vec::new(),
        })
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
        log::info!(
            "Parser::parse: starting with capacity for ~{} items",
            estimated_items
        );

        while !self.is_at_end() {
            self.collect_pending_docs();
            if self.is_item_start() {
                items.push(self.parse_item()?);
            } else {
                let start_line = self.current_token().line;
                let start_column = self.current_token().column;

                let mut stmts = Vec::new();
                while !self.is_at_end() && !self.is_item_start() {
                    self.collect_pending_docs();
                    if self.is_item_start() {
                        break; // collected docs belong to the next item
                    }
                    self.pending_docs.clear(); // docs on plain statements are dropped
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
            | TokenKind::Use
            | TokenKind::Module
            | TokenKind::Extern => true,
            TokenKind::Local => self.peek_ahead(1).is_some_and(|t| {
                matches!(
                    t.kind,
                    TokenKind::Function | TokenKind::Struct | TokenKind::Enum | TokenKind::Trait
                )
            }),
            _ => false,
        }
    }

    fn current_token(&self) -> &Token {
        &self.tokens[self.current]
    }

    /// Consumes any doc comment tokens at the current position, collecting
    /// their text for the next declaration.
    pub(super) fn collect_pending_docs(&mut self) {
        while self.check(TokenKind::DocComment) {
            let lexeme = self.current_token().lexeme.clone();
            self.advance();
            self.pending_docs.push(lexeme);
        }
    }

    /// Returns the collected doc comments (joined with newlines) and resets
    /// the pending buffer.
    pub(super) fn take_pending_docs(&mut self) -> Option<String> {
        if self.pending_docs.is_empty() {
            return None;
        }
        let doc = self.pending_docs.join("\n");
        self.pending_docs.clear();
        Some(doc)
    }

    fn peek_kind(&self) -> TokenKind {
        self.current_token().kind
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
            if self.check(*kind) {
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
        if self.check(TokenKind::Identifier) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::ExternItem;
    use crate::intern::Interner;
    use crate::lexer::Lexer;

    fn parse_source(source: &str) -> Vec<Item> {
        let mut interner = Interner::new();
        let mut lexer = Lexer::new(source, &mut interner);
        let tokens = lexer.tokenize().expect("tokenize");
        Parser::new(tokens)
            .parse()
            .expect("parse")
    }

    #[test]
    fn doc_comments_attach_to_declarations() {
        let items = parse_source(
            "--- Adds one.\nfunction add_one(x: int): int\n    return x + 1\nend\n",
        );
        assert_eq!(items.len(), 1);
        match &items[0].kind {
            ItemKind::Function(func) => {
                assert_eq!(func.doc.as_deref(), Some("Adds one."));
            }
            other => panic!("expected function item, got {:?}", other),
        }
    }

    #[test]
    fn multiple_doc_lines_join() {
        let items = parse_source(
            "--- Line one.\n--- Line two.\nstruct Widget\n    x: int\nend\n",
        );
        match &items[0].kind {
            ItemKind::Struct(def) => {
                assert_eq!(def.doc.as_deref(), Some("Line one.\nLine two."));
            }
            other => panic!("expected struct item, got {:?}", other),
        }
    }

    #[test]
    fn four_dashes_is_a_regular_comment() {
        let items = parse_source(
            "---- not a doc\nfunction add_one(x: int): int\n    return x + 1\nend\n",
        );
        match &items[0].kind {
            ItemKind::Function(func) => assert!(func.doc.is_none()),
            other => panic!("expected function item, got {:?}", other),
        }
    }

    #[test]
    fn three_dashes_without_space_is_a_regular_comment() {
        let items = parse_source(
            "---not a doc\nfunction add_one(x: int): int\n    return x + 1\nend\n",
        );
        match &items[0].kind {
            ItemKind::Function(func) => assert!(func.doc.is_none()),
            other => panic!("expected function item, got {:?}", other),
        }
    }

    #[test]
    fn double_dash_is_a_regular_comment() {
        let items = parse_source(
            "-- not a doc\nfunction add_one(x: int): int\n    return x + 1\nend\n",
        );
        match &items[0].kind {
            ItemKind::Function(func) => assert!(func.doc.is_none()),
            other => panic!("expected function item, got {:?}", other),
        }
    }

    #[test]
    fn doc_comments_attach_to_impl_methods() {
        let items = parse_source(
            "struct Point\n    x: int\nend\nimpl Point\n    --- Constructs a point.\n    function new(x: int): Point\n        return Point { x = x }\n    end\nend\n",
        );
        let mut method_doc = None;
        for item in &items {
            if let ItemKind::Impl(impl_block) = &item.kind {
                method_doc = impl_block.methods[0].doc.clone();
            }
        }
        assert_eq!(method_doc.as_deref(), Some("Constructs a point."));
    }

    #[test]
    fn doc_comments_attach_to_extern_functions() {
        let items = parse_source(
            "extern\n    --- Host callback.\n    function on_event(int)\nend\n",
        );
        match &items[0].kind {
            ItemKind::Extern { items: extern_items, .. } => match &extern_items[0] {
                ExternItem::Function { doc, .. } => {
                    assert_eq!(doc.as_deref(), Some("Host callback."));
                }
                other => panic!("expected extern function, got {:?}", other),
            },
            other => panic!("expected extern item, got {:?}", other),
        }
    }

    #[test]
    fn doc_comment_before_extern_block_attaches_to_first_item() {
        let items = parse_source(
            "--- Host callback.\nextern\n    function on_event(int)\n    function other(int)\nend\n",
        );
        match &items[0].kind {
            ItemKind::Extern { items: extern_items, .. } => {
                match &extern_items[0] {
                    ExternItem::Function { doc, .. } => {
                        assert_eq!(doc.as_deref(), Some("Host callback."));
                    }
                    other => panic!("expected extern function, got {:?}", other),
                }
                match &extern_items[1] {
                    ExternItem::Function { doc, .. } => {
                        assert!(doc.is_none(), "second extern item should not inherit doc");
                    }
                    other => panic!("expected extern function, got {:?}", other),
                }
            }
            other => panic!("expected extern item, got {:?}", other),
        }
    }

    #[test]
    fn doc_comments_do_not_leak_between_declarations() {
        let items = parse_source(
            "--- Doc for first.\nfunction first(): int\n    return 1\nend\nfunction second(): int\n    return 2\nend\n",
        );
        let docs: Vec<Option<String>> = items
            .iter()
            .map(|item| match &item.kind {
                ItemKind::Function(func) => func.doc.clone(),
                _ => None,
            })
            .collect();
        assert_eq!(docs[0].as_deref(), Some("Doc for first."));
        assert!(docs[1].is_none());
    }
}
