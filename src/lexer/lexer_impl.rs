use super::token::{Token, TokenKind};
use crate::error::{LustError, Result};
use crate::intern::Interner;
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};

pub struct Lexer<'a> {
    input: &'a str,
    position: usize,
    line: usize,
    column: usize,
    /// String interner for deduplicating identifiers and literals
    interner: &'a mut Interner,
}

/// Streaming token iterator for memory-constrained environments
pub struct TokenIterator<'lexer, 'a> {
    lexer: &'lexer mut Lexer<'a>,
    finished: bool,
}

impl<'a> Lexer<'a> {
    /// Create a new lexer with string interning for memory efficiency
    pub fn new(input: &'a str, interner: &'a mut Interner) -> Self {
        Self {
            input,
            position: 0,
            line: 1,
            column: 1,
            interner,
        }
    }

    /// Returns the length of the source in bytes
    pub fn source_len(&self) -> usize {
        self.input.len()
    }

    pub fn tokenize(&mut self) -> Result<Vec<Token>> {
        #[cfg(feature = "esp32c6-logging")]
        {
            log::info!("Lexer::tokenize: starting for {} bytes", self.input.len());
        }

        let mut tokens = Vec::new();
        let mut iterations = 0;
        let max_iterations = self.input.len() * 2; // Safety limit

        while !self.is_at_end() {
            iterations += 1;
            if iterations > max_iterations {
                return Err(LustError::LexerError {
                    line: self.line,
                    column: self.column,
                    message: "Tokenization exceeded maximum iterations (possible infinite loop)"
                        .to_string(),
                    module: None,
                });
            }

            // Log progress every 100 iterations
            #[cfg(feature = "esp32c6-logging")]
            {
                if iterations % 100 == 0 {
                    log::info!(
                        "Lexer::tokenize: iteration {} at line {}",
                        iterations,
                        self.line
                    );
                }
            }

            self.skip_whitespace_and_comments()?;
            if self.is_at_end() {
                break;
            }

            if let Some(token) = self.take_doc_comment() {
                tokens.push(token);
                continue;
            }

            let token = self.next_token()?;
            tokens.push(token);
        }

        tokens.push(Token::new(
            TokenKind::Eof,
            String::new(),
            self.line,
            self.column,
        ));

        #[cfg(feature = "esp32c6-logging")]
        log::info!(
            "Lexer::tokenize: complete, {} tokens in {} iterations",
            tokens.len(),
            iterations
        );

        Ok(tokens)
    }

    /// Returns an iterator over tokens without allocating all tokens upfront.
    /// More memory-efficient for embedded targets with limited heap.
    pub fn tokenize_iter(&mut self) -> TokenIterator<'_, 'a> {
        TokenIterator {
            lexer: self,
            finished: false,
        }
    }

    /// Advances to the next token (for streaming tokenization)
    pub fn next_token_streaming(&mut self) -> Result<Option<Token>> {
        self.skip_whitespace_and_comments()?;
        if self.is_at_end() {
            return Ok(None);
        }
        if let Some(token) = self.take_doc_comment() {
            return Ok(Some(token));
        }
        let token = self.next_token()?;
        Ok(Some(token))
    }

    fn next_token(&mut self) -> Result<Token> {
        let start_line = self.line;
        let start_column = self.column;
        let ch = self.current_char();

        // Fixed tokens - no string allocation needed
        match ch {
            '(' => {
                self.advance();
                return Ok(Token::simple(
                    TokenKind::LeftParen,
                    start_line,
                    start_column,
                ));
            }
            ')' => {
                self.advance();
                return Ok(Token::simple(
                    TokenKind::RightParen,
                    start_line,
                    start_column,
                ));
            }
            '{' => {
                self.advance();
                return Ok(Token::simple(
                    TokenKind::LeftBrace,
                    start_line,
                    start_column,
                ));
            }
            '}' => {
                self.advance();
                return Ok(Token::simple(
                    TokenKind::RightBrace,
                    start_line,
                    start_column,
                ));
            }
            '[' => {
                self.advance();
                return Ok(Token::simple(
                    TokenKind::LeftBracket,
                    start_line,
                    start_column,
                ));
            }
            ']' => {
                self.advance();
                return Ok(Token::simple(
                    TokenKind::RightBracket,
                    start_line,
                    start_column,
                ));
            }
            ',' => {
                self.advance();
                return Ok(Token::simple(TokenKind::Comma, start_line, start_column));
            }
            ';' => {
                self.advance();
                return Ok(Token::simple(
                    TokenKind::Semicolon,
                    start_line,
                    start_column,
                ));
            }
            '%' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    return Ok(Token::simple(
                        TokenKind::PercentEqual,
                        start_line,
                        start_column,
                    ));
                } else {
                    return Ok(Token::simple(TokenKind::Percent, start_line, start_column));
                }
            }
            '^' => {
                self.advance();
                if self.current_char() == '^' {
                    self.advance();
                    if self.current_char() == '=' {
                        self.advance();
                        return Ok(Token::simple(
                            TokenKind::DoubleCaretEqual,
                            start_line,
                            start_column,
                        ));
                    }
                    return Ok(Token::simple(
                        TokenKind::DoubleCaret,
                        start_line,
                        start_column,
                    ));
                } else if self.current_char() == '=' {
                    self.advance();
                    return Ok(Token::simple(
                        TokenKind::CaretEqual,
                        start_line,
                        start_column,
                    ));
                } else {
                    return Ok(Token::simple(TokenKind::Caret, start_line, start_column));
                }
            }
            '?' => {
                self.advance();
                return Ok(Token::simple(TokenKind::Question, start_line, start_column));
            }
            '&' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    return Ok(Token::simple(
                        TokenKind::AmpersandEqual,
                        start_line,
                        start_column,
                    ));
                } else {
                    return Ok(Token::simple(
                        TokenKind::Ampersand,
                        start_line,
                        start_column,
                    ));
                }
            }
            '|' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    return Ok(Token::simple(
                        TokenKind::PipeEqual,
                        start_line,
                        start_column,
                    ));
                } else if self.current_char() == '>' {
                    self.advance();
                    return Ok(Token::simple(
                        TokenKind::PipeGreater,
                        start_line,
                        start_column,
                    ));
                } else {
                    return Ok(Token::simple(TokenKind::Pipe, start_line, start_column));
                }
            }
            '+' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    return Ok(Token::simple(
                        TokenKind::PlusEqual,
                        start_line,
                        start_column,
                    ));
                } else {
                    return Ok(Token::simple(TokenKind::Plus, start_line, start_column));
                }
            }
            '-' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    return Ok(Token::simple(
                        TokenKind::MinusEqual,
                        start_line,
                        start_column,
                    ));
                } else if self.current_char() == '>' {
                    self.advance();
                    return Ok(Token::simple(TokenKind::Arrow, start_line, start_column));
                } else {
                    return Ok(Token::simple(TokenKind::Minus, start_line, start_column));
                }
            }
            '*' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    return Ok(Token::simple(
                        TokenKind::StarEqual,
                        start_line,
                        start_column,
                    ));
                } else {
                    return Ok(Token::simple(TokenKind::Star, start_line, start_column));
                }
            }
            '/' => {
                self.advance();
                if self.current_char() == '/' {
                    self.advance();
                    if self.current_char() == '=' {
                        self.advance();
                        return Ok(Token::simple(
                            TokenKind::DoubleSlashEqual,
                            start_line,
                            start_column,
                        ));
                    }
                    return Ok(Token::simple(
                        TokenKind::DoubleSlash,
                        start_line,
                        start_column,
                    ));
                } else if self.current_char() == '=' {
                    self.advance();
                    return Ok(Token::simple(
                        TokenKind::SlashEqual,
                        start_line,
                        start_column,
                    ));
                } else {
                    return Ok(Token::simple(TokenKind::Slash, start_line, start_column));
                }
            }
            '=' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    return Ok(Token::simple(
                        TokenKind::DoubleEqual,
                        start_line,
                        start_column,
                    ));
                } else if self.current_char() == '>' {
                    self.advance();
                    return Ok(Token::simple(TokenKind::FatArrow, start_line, start_column));
                } else {
                    return Ok(Token::simple(TokenKind::Equal, start_line, start_column));
                }
            }
            '~' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    return Ok(Token::simple(TokenKind::NotEqual, start_line, start_column));
                } else {
                    return Ok(Token::simple(TokenKind::Tilde, start_line, start_column));
                }
            }
            '!' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    return Ok(Token::simple(TokenKind::NotEqual, start_line, start_column));
                } else {
                    return Ok(Token::simple(TokenKind::Bang, start_line, start_column));
                }
            }
            '<' => {
                self.advance();
                if self.current_char() == '<' {
                    self.advance();
                    if self.current_char() == '=' {
                        self.advance();
                        return Ok(Token::simple(
                            TokenKind::DoubleLessEqual,
                            start_line,
                            start_column,
                        ));
                    }
                    return Ok(Token::simple(
                        TokenKind::DoubleLess,
                        start_line,
                        start_column,
                    ));
                } else if self.current_char() == '=' {
                    self.advance();
                    return Ok(Token::simple(
                        TokenKind::LessEqual,
                        start_line,
                        start_column,
                    ));
                } else {
                    return Ok(Token::simple(TokenKind::Less, start_line, start_column));
                }
            }
            '>' => {
                self.advance();
                if self.current_char() == '>' {
                    self.advance();
                    if self.current_char() == '=' {
                        self.advance();
                        return Ok(Token::simple(
                            TokenKind::DoubleGreaterEqual,
                            start_line,
                            start_column,
                        ));
                    }
                    return Ok(Token::simple(
                        TokenKind::DoubleGreater,
                        start_line,
                        start_column,
                    ));
                } else if self.current_char() == '=' {
                    self.advance();
                    return Ok(Token::simple(
                        TokenKind::GreaterEqual,
                        start_line,
                        start_column,
                    ));
                } else {
                    return Ok(Token::simple(TokenKind::Greater, start_line, start_column));
                }
            }
            ':' => {
                self.advance();
                if self.current_char() == ':' {
                    self.advance();
                    return Ok(Token::simple(
                        TokenKind::DoubleColon,
                        start_line,
                        start_column,
                    ));
                } else {
                    return Ok(Token::simple(TokenKind::Colon, start_line, start_column));
                }
            }
            '.' => {
                self.advance();
                if self.current_char() == '.' {
                    self.advance();
                    return Ok(Token::simple(
                        TokenKind::DoubleDot,
                        start_line,
                        start_column,
                    ));
                } else {
                    return Ok(Token::simple(TokenKind::Dot, start_line, start_column));
                }
            }
            _ => {}
        }

        // Variable tokens - zero-copy slice scanning
        match ch {
            '"' | '\'' => self.scan_string(start_line, start_column),
            '0'..='9' => self.scan_number(start_line, start_column),
            'a'..='z' | 'A'..='Z' | '_' => self.scan_identifier(start_line, start_column),
            _ => Err(LustError::LexerError {
                line: start_line,
                column: start_column,
                message: format!("Unexpected character: {}", ch),
                module: None,
            }),
        }
    }

    fn scan_string(&mut self, start_line: usize, start_column: usize) -> Result<Token> {
        let quote = self.current_char();
        let start_pos = self.position;
        self.advance();
        while !self.is_at_end() && self.current_char() != quote {
            if self.current_char() == '\\' {
                self.advance();
                if !self.is_at_end() {
                    self.advance();
                }
            } else {
                self.advance();
            }
        }

        if self.is_at_end() {
            return Err(LustError::LexerError {
                line: start_line,
                column: start_column,
                message: "Unterminated string".to_string(),
                module: None,
            });
        }

        self.advance();

        let slice = &self.input[start_pos..self.position];
        let symbol = self.interner.intern(slice);
        let interned = self.interner.get(symbol).to_string();
        Ok(Token::new(
            TokenKind::String,
            interned,
            start_line,
            start_column,
        ))
    }

    fn scan_radix_number(
        &mut self,
        prefix: char,
        start_pos: usize,
        start_line: usize,
        start_column: usize,
    ) -> Result<Token> {
        let radix = match prefix {
            'x' | 'X' => 16,
            'b' | 'B' => 2,
            'o' | 'O' => 8,
            _ => unreachable!(),
        };

        // Advance past '0' and prefix
        self.advance();
        self.advance();

        if self.is_at_end() {
            return Err(LustError::LexerError {
                line: start_line,
                column: start_column,
                message: "Expected at least one valid digit after radix prefix".to_string(),
                module: None,
            });
        }

        let first = self.current_char();
        if first == '_' {
            return Err(LustError::LexerError {
                line: start_line,
                column: start_column,
                message: "Radix prefix cannot be immediately followed by an underscore".to_string(),
                module: None,
            });
        }

        if !first.is_ascii_alphanumeric() {
            return Err(LustError::LexerError {
                line: start_line,
                column: start_column,
                message: "Expected at least one valid digit after radix prefix".to_string(),
                module: None,
            });
        }

        if !first.is_digit(radix) {
            // Consume invalid alphanumeric run to prevent token splitting
            while !self.is_at_end() && self.current_char().is_ascii_alphanumeric() {
                self.advance();
            }
            return Err(LustError::LexerError {
                line: start_line,
                column: start_column,
                message: "Invalid digit for radix literal".to_string(),
                module: None,
            });
        }

        let mut last_char = first;
        self.advance();

        while !self.is_at_end() {
            let c = self.current_char();
            if c == '_' {
                if last_char == '_' {
                    return Err(LustError::LexerError {
                        line: start_line,
                        column: start_column,
                        message: "Consecutive underscores in number literal are not allowed".to_string(),
                        module: None,
                    });
                }
                match self.peek(1) {
                    Some(next) if next.is_ascii_alphanumeric() => {
                        if !next.is_digit(radix) {
                            self.advance(); // consume '_'
                            while !self.is_at_end() && self.current_char().is_ascii_alphanumeric() {
                                self.advance();
                            }
                            return Err(LustError::LexerError {
                                line: start_line,
                                column: start_column,
                                message: "Invalid digit for radix literal".to_string(),
                                module: None,
                            });
                        }
                    }
                    _ => {
                        return Err(LustError::LexerError {
                            line: start_line,
                            column: start_column,
                            message: "Trailing underscore in number literal is not allowed".to_string(),
                            module: None,
                        });
                    }
                }
                last_char = '_';
                self.advance();
            } else if c.is_digit(radix) {
                last_char = c;
                self.advance();
            } else if c.is_ascii_alphanumeric() {
                while !self.is_at_end() && self.current_char().is_ascii_alphanumeric() {
                    self.advance();
                }
                return Err(LustError::LexerError {
                    line: start_line,
                    column: start_column,
                    message: "Invalid digit for radix literal".to_string(),
                    module: None,
                });
            } else {
                break;
            }
        }

        let slice = &self.input[start_pos..self.position];
        let symbol = self.interner.intern(slice);
        let interned = self.interner.get(symbol).to_string();
        Ok(Token::new(TokenKind::Integer, interned, start_line, start_column))
    }

    fn scan_number(&mut self, start_line: usize, start_column: usize) -> Result<Token> {
        let start_pos = self.position;
        // Check for radix prefix (0x, 0b, 0o)
        if self.current_char() == '0' {
            if let Some(prefix) = self.peek(1) {
                if matches!(prefix, 'x' | 'X' | 'b' | 'B' | 'o' | 'O') {
                    return self.scan_radix_number(prefix, start_pos, start_line, start_column);
                }
            }
        }

        let mut is_float = false;
        let mut last_char = '0';
        while !self.is_at_end() {
            let b = self.input.as_bytes()[self.position];
            if b.is_ascii_digit() {
                last_char = b as char;
                self.advance();
            } else if b == b'_' {
                if last_char == '_' {
                    return Err(LustError::LexerError {
                        line: start_line,
                        column: start_column,
                        message: "Consecutive underscores in number literal are not allowed".to_string(),
                        module: None,
                    });
                }
                match self.peek(1) {
                    Some('.') => {
                        return Err(LustError::LexerError {
                            line: start_line,
                            column: start_column,
                            message: "Underscore immediately preceding decimal point is not allowed".to_string(),
                            module: None,
                        });
                    }
                    Some('e' | 'E') => {
                        return Err(LustError::LexerError {
                            line: start_line,
                            column: start_column,
                            message: "Underscore immediately preceding exponent is not allowed".to_string(),
                            module: None,
                        });
                    }
                    Some(c) if c.is_ascii_digit() => {
                        last_char = '_';
                        self.advance();
                    }
                    _ => {
                        return Err(LustError::LexerError {
                            line: start_line,
                            column: start_column,
                            message: "Trailing underscore in number literal is not allowed".to_string(),
                            module: None,
                        });
                    }
                }
            } else {
                break;
            }
        }

        // Float / range disambiguation
        if !self.is_at_end() && self.current_char() == '.' {
            if self.peek(1) == Some('.') {
                // Range operator (.. or ..=): do NOT consume dot, stop here!
            } else if self.peek(1) == Some('_') {
                return Err(LustError::LexerError {
                    line: start_line,
                    column: start_column,
                    message: "Underscore immediately following decimal point is not allowed".to_string(),
                    module: None,
                });
            } else if self.peek(1).is_some_and(|c| c.is_ascii_digit()) {
                is_float = true;
                self.advance(); // consume '.'
                last_char = '.';
                while !self.is_at_end() {
                    let b = self.input.as_bytes()[self.position];
                    if b.is_ascii_digit() {
                        last_char = b as char;
                        self.advance();
                    } else if b == b'_' {
                        if last_char == '_' {
                            return Err(LustError::LexerError {
                                line: start_line,
                                column: start_column,
                                message: "Consecutive underscores in number literal are not allowed".to_string(),
                                module: None,
                            });
                        }
                        match self.peek(1) {
                            Some('e' | 'E') => {
                                return Err(LustError::LexerError {
                                    line: start_line,
                                    column: start_column,
                                    message: "Underscore immediately preceding exponent is not allowed".to_string(),
                                    module: None,
                                });
                            }
                            Some(c) if c.is_ascii_digit() => {
                                last_char = '_';
                                self.advance();
                            }
                            _ => {
                                return Err(LustError::LexerError {
                                    line: start_line,
                                    column: start_column,
                                    message: "Trailing underscore in float literal is not allowed".to_string(),
                                    module: None,
                                });
                            }
                        }
                    } else {
                        break;
                    }
                }
            }
            // else: single dot followed by non-digit (e.g. 1.method or 1. ): do NOT consume dot!
        }

        // Exponent check (e / E)
        if !self.is_at_end() && (self.current_char() == 'e' || self.current_char() == 'E') {
            is_float = true;
            self.advance(); // consume e/E
            if !self.is_at_end() && (self.current_char() == '+' || self.current_char() == '-') {
                self.advance();
                if !self.is_at_end() && self.current_char() == '_' {
                    return Err(LustError::LexerError {
                        line: start_line,
                        column: start_column,
                        message: "Underscore immediately following exponent indicator or sign is not allowed".to_string(),
                        module: None,
                    });
                }
            } else if !self.is_at_end() && self.current_char() == '_' {
                return Err(LustError::LexerError {
                    line: start_line,
                    column: start_column,
                    message: "Underscore immediately following exponent indicator or sign is not allowed".to_string(),
                    module: None,
                });
            }

            if self.is_at_end() || !self.input.as_bytes()[self.position].is_ascii_digit() {
                while !self.is_at_end() && self.current_char().is_ascii_alphanumeric() {
                    self.advance();
                }
                return Err(LustError::LexerError {
                    line: start_line,
                    column: start_column,
                    message: "Exponent indicator must be followed by digits".to_string(),
                    module: None,
                });
            }

            last_char = '0';
            while !self.is_at_end() {
                let b = self.input.as_bytes()[self.position];
                if b.is_ascii_digit() {
                    last_char = b as char;
                    self.advance();
                } else if b == b'_' {
                    if last_char == '_' {
                        return Err(LustError::LexerError {
                            line: start_line,
                            column: start_column,
                            message: "Consecutive underscores in number literal are not allowed".to_string(),
                            module: None,
                        });
                    }
                    if self.peek(1).is_some_and(|c| c.is_ascii_digit()) {
                        last_char = '_';
                        self.advance();
                    } else {
                        return Err(LustError::LexerError {
                            line: start_line,
                            column: start_column,
                            message: "Trailing underscore in number literal is not allowed".to_string(),
                            module: None,
                        });
                    }
                } else {
                    break;
                }
            }
        }

        // Check for trailing invalid alphanumeric suffix (e.g. 100abc)
        if !self.is_at_end() && (self.current_char().is_ascii_alphabetic() || self.current_char() == '_') {
            while !self.is_at_end() && (self.current_char().is_ascii_alphanumeric() || self.current_char() == '_') {
                self.advance();
            }
            return Err(LustError::LexerError {
                line: start_line,
                column: start_column,
                message: "Invalid trailing characters in number literal".to_string(),
                module: None,
            });
        }

        let kind = if is_float {
            TokenKind::Float
        } else {
            TokenKind::Integer
        };

        let slice = &self.input[start_pos..self.position];
        let symbol = self.interner.intern(slice);
        let interned = self.interner.get(symbol).to_string();
        Ok(Token::new(kind, interned, start_line, start_column))
    }

    fn scan_identifier(&mut self, start_line: usize, start_column: usize) -> Result<Token> {
        let start_pos = self.position;
        while !self.is_at_end() {
            let b = self.input.as_bytes()[self.position];
            if b.is_ascii_alphanumeric() || b == b'_' {
                self.advance();
            } else if !b.is_ascii() {
                let ch = self.current_char();
                if ch.is_alphanumeric() || ch == '_' {
                    self.advance();
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        let slice = &self.input[start_pos..self.position];
        if let Some(kind) = TokenKind::keyword(slice) {
            return Ok(Token::simple(kind, start_line, start_column));
        }

        let symbol = self.interner.intern(slice);
        let interned = self.interner.get(symbol).to_string();
        Ok(Token::new(
            TokenKind::Identifier,
            interned,
            start_line,
            start_column,
        ))
    }

    fn skip_whitespace_and_comments(&mut self) -> Result<()> {
        while !self.is_at_end() {
            match self.current_char() {
                ' ' | '\t' | '\r' => {
                    self.advance();
                }

                '\n' => {
                    self.advance();
                    self.line += 1;
                    self.column = 1;
                }

                '-' => {
                    if self.peek(1) == Some('-') {
                        if self.peek(2) == Some('[') && self.peek(3) == Some('[') {
                            self.advance();
                            self.advance();
                            self.advance();
                            self.advance();
                            self.skip_block_comment()?;
                            continue;
                        }

                        // Exactly three leading dashes mark a doc comment;
                        // a fourth dash keeps the line an ordinary comment.
                        if self.peek(2) == Some('-') && self.peek(3) != Some('-') {
                            break;
                        }

                        self.advance();
                        self.advance();
                        while !self.is_at_end() && self.current_char() != '\n' {
                            self.advance();
                        }
                    } else {
                        break;
                    }
                }

                '#' => {
                    self.advance();
                    while !self.is_at_end() && self.current_char() != '\n' {
                        self.advance();
                    }
                }

                _ => break,
            }
        }

        Ok(())
    }

    /// Scans a doc comment beginning with exactly three dashes. Leading
    /// horizontal whitespace after the marker is ignored; a fourth dash makes
    /// it an ordinary comment.
    fn take_doc_comment(&mut self) -> Option<Token> {
        if self.current_char() != '-'
            || self.peek(1) != Some('-')
            || self.peek(2) != Some('-')
            || self.peek(3) == Some('-')
        {
            return None;
        }

        let line = self.line;
        let column = self.column;
        self.advance();
        self.advance();
        self.advance();
        while matches!(self.current_char(), ' ' | '\t' | '\r') {
            self.advance();
        }

        let content_start = self.position;
        while !self.is_at_end() && self.current_char() != '\n' {
            self.advance();
        }
        let content_end = self.position;
        let text = self.input[content_start..content_end]
            .trim_end_matches([' ', '\t', '\r'])
            .to_string();

        Some(Token::new(TokenKind::DocComment, text, line, column))
    }

    fn skip_block_comment(&mut self) -> Result<()> {
        while !self.is_at_end() {
            if self.current_char() == ']' && self.peek(1) == Some(']') {
                self.advance();
                self.advance();
                return Ok(());
            }

            if self.current_char() == '\n' {
                self.advance();
                self.line += 1;
                self.column = 1;
            } else {
                self.advance();
            }
        }

        Err(LustError::LexerError {
            line: self.line,
            column: self.column,
            message: "Unterminated block comment".to_string(),
            module: None,
        })
    }

    fn current_char(&self) -> char {
        self.input[self.position..].chars().next().unwrap_or('\0')
    }

    fn peek(&self, offset: usize) -> Option<char> {
        // Optimized: use byte indexing for ASCII (common case)
        // Fall back to char iteration only for multibyte sequences
        let bytes = &self.input.as_bytes()[self.position..];

        if offset == 0 {
            if bytes.is_empty() {
                return None;
            }
            // Fast path for ASCII
            if bytes[0] < 128 {
                return Some(bytes[0] as char);
            }
        }

        // For offset > 0 or multibyte, iterate
        let mut iter = self.input[self.position..].chars();
        for _ in 0..offset {
            iter.next();
        }
        iter.next()
    }

    fn advance(&mut self) {
        if let Some(ch) = self.input[self.position..].chars().next() {
            self.position += ch.len_utf8();
            self.column += 1;
        }
    }

    fn is_at_end(&self) -> bool {
        self.position >= self.input.len()
    }
}

impl<'lexer, 'a> core::iter::Iterator for TokenIterator<'lexer, 'a> {
    type Item = Result<Token>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        match self.lexer.next_token_streaming() {
            Ok(Some(token)) => Some(Ok(token)),
            Ok(None) => {
                self.finished = true;
                // Emit EOF token
                Some(Ok(Token::new(
                    TokenKind::Eof,
                    String::new(),
                    self.lexer.line,
                    self.lexer.column,
                )))
            }
            Err(e) => {
                self.finished = true;
                Some(Err(e))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_math_and_bitwise_tokens() {
        let mut interner = Interner::new();
        let input = "// //= % %= ^ ^= ^^ ^^= & &= | |= << <<= >> >>= ~ ~= / /=";
        let mut lexer = Lexer::new(input, &mut interner);
        let tokens = lexer.tokenize().unwrap();
        let kinds: Vec<TokenKind> = tokens.into_iter().map(|t| t.kind).collect();
        assert_eq!(
            kinds,
            vec![
                TokenKind::DoubleSlash,
                TokenKind::DoubleSlashEqual,
                TokenKind::Percent,
                TokenKind::PercentEqual,
                TokenKind::Caret,
                TokenKind::CaretEqual,
                TokenKind::DoubleCaret,
                TokenKind::DoubleCaretEqual,
                TokenKind::Ampersand,
                TokenKind::AmpersandEqual,
                TokenKind::Pipe,
                TokenKind::PipeEqual,
                TokenKind::DoubleLess,
                TokenKind::DoubleLessEqual,
                TokenKind::DoubleGreater,
                TokenKind::DoubleGreaterEqual,
                TokenKind::Tilde,
                TokenKind::NotEqual,
                TokenKind::Slash,
                TokenKind::SlashEqual,
                TokenKind::Eof,
            ]
        );
    }
}
