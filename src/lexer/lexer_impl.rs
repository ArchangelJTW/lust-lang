use super::token::{Token, TokenKind};
use crate::error::{LustError, Result};
use crate::intern::{Interner, Symbol};
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
                    message: "Tokenization exceeded maximum iterations (possible infinite loop)".to_string(),
                    module: None,
                });
            }

            // Log progress every 100 iterations
            #[cfg(feature = "esp32c6-logging")]
            {
                if iterations % 100 == 0 {
                    log::info!("Lexer::tokenize: iteration {} at line {}", iterations, self.line);
                }
            }

            self.skip_whitespace_and_comments()?;
            if self.is_at_end() {
                break;
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
        log::info!("Lexer::tokenize: complete, {} tokens in {} iterations", tokens.len(), iterations);

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
                return Ok(Token::simple(TokenKind::LeftParen, start_line, start_column));
            }
            ')' => {
                self.advance();
                return Ok(Token::simple(TokenKind::RightParen, start_line, start_column));
            }
            '{' => {
                self.advance();
                return Ok(Token::simple(TokenKind::LeftBrace, start_line, start_column));
            }
            '}' => {
                self.advance();
                return Ok(Token::simple(TokenKind::RightBrace, start_line, start_column));
            }
            '[' => {
                self.advance();
                return Ok(Token::simple(TokenKind::LeftBracket, start_line, start_column));
            }
            ']' => {
                self.advance();
                return Ok(Token::simple(TokenKind::RightBracket, start_line, start_column));
            }
            ',' => {
                self.advance();
                return Ok(Token::simple(TokenKind::Comma, start_line, start_column));
            }
            ';' => {
                self.advance();
                return Ok(Token::simple(TokenKind::Semicolon, start_line, start_column));
            }
            '%' => {
                self.advance();
                return Ok(Token::simple(TokenKind::Percent, start_line, start_column));
            }
            '^' => {
                self.advance();
                return Ok(Token::simple(TokenKind::Caret, start_line, start_column));
            }
            '?' => {
                self.advance();
                return Ok(Token::simple(TokenKind::Question, start_line, start_column));
            }
            '&' => {
                self.advance();
                return Ok(Token::simple(TokenKind::Ampersand, start_line, start_column));
            }
            '|' => {
                self.advance();
                return Ok(Token::simple(TokenKind::Pipe, start_line, start_column));
            }
            '+' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    return Ok(Token::simple(TokenKind::PlusEqual, start_line, start_column));
                } else {
                    return Ok(Token::simple(TokenKind::Plus, start_line, start_column));
                }
            }
            '-' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    return Ok(Token::simple(TokenKind::MinusEqual, start_line, start_column));
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
                    return Ok(Token::simple(TokenKind::StarEqual, start_line, start_column));
                } else {
                    return Ok(Token::simple(TokenKind::Star, start_line, start_column));
                }
            }
            '/' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    return Ok(Token::simple(TokenKind::SlashEqual, start_line, start_column));
                } else {
                    return Ok(Token::simple(TokenKind::Slash, start_line, start_column));
                }
            }
            '=' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    return Ok(Token::simple(TokenKind::DoubleEqual, start_line, start_column));
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
                    return Err(LustError::LexerError {
                        line: start_line,
                        column: start_column,
                        message: format!("Unexpected character: {}", ch),
                        module: None,
                    });
                }
            }
            '!' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    return Ok(Token::simple(TokenKind::NotEqual, start_line, start_column));
                } else {
                    return Err(LustError::LexerError {
                        line: start_line,
                        column: start_column,
                        message: format!("Unexpected character: {}", ch),
                        module: None,
                    });
                }
            }
            '<' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    return Ok(Token::simple(TokenKind::LessEqual, start_line, start_column));
                } else {
                    return Ok(Token::simple(TokenKind::Less, start_line, start_column));
                }
            }
            '>' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    return Ok(Token::simple(TokenKind::GreaterEqual, start_line, start_column));
                } else {
                    return Ok(Token::simple(TokenKind::Greater, start_line, start_column));
                }
            }
            ':' => {
                self.advance();
                if self.current_char() == ':' {
                    self.advance();
                    return Ok(Token::simple(TokenKind::DoubleColon, start_line, start_column));
                } else {
                    return Ok(Token::simple(TokenKind::Colon, start_line, start_column));
                }
            }
            '.' => {
                self.advance();
                if self.current_char() == '.' {
                    self.advance();
                    return Ok(Token::simple(TokenKind::DoubleDot, start_line, start_column));
                } else {
                    return Ok(Token::simple(TokenKind::Dot, start_line, start_column));
                }
            }
            _ => {}
        }

        // Variable tokens - need string allocation
        let (kind, lexeme) = match ch {
            '"' | '\'' => self.scan_string()?,
            '0'..='9' => self.scan_number()?,
            'a'..='z' | 'A'..='Z' | '_' => self.scan_identifier()?,
            _ => {
                return Err(LustError::LexerError {
                    line: start_line,
                    column: start_column,
                    message: format!("Unexpected character: {}", ch),
                    module: None,
                });
            }
        };
        Ok(Token::new(kind, lexeme, start_line, start_column))
    }

    fn scan_string(&mut self) -> Result<(TokenKind, String)> {
        let quote = self.current_char();
        let start_line = self.line;
        let start_column = self.column;
        self.advance();
        let mut value = String::new();
        value.push(quote);
        while !self.is_at_end() && self.current_char() != quote {
            if self.current_char() == '\\' {
                value.push(self.current_char());
                self.advance();
                if !self.is_at_end() {
                    value.push(self.current_char());
                    self.advance();
                }
            } else {
                value.push(self.current_char());
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

        value.push(self.current_char());
        self.advance();

        // Intern the string literal to save memory
        let symbol = self.interner.intern(&value);
        let interned = self.interner.get(symbol).to_string();
        Ok((TokenKind::String, interned))
    }

    fn scan_number(&mut self) -> Result<(TokenKind, String)> {
        let mut value = String::new();
        let mut is_float = false;
        while !self.is_at_end() && self.current_char().is_ascii_digit() {
            value.push(self.current_char());
            self.advance();
        }

        if !self.is_at_end() && self.current_char() == '.' {
            if self.peek(1) != Some('.') && self.peek(1).map_or(false, |c| c.is_ascii_digit()) {
                is_float = true;
                value.push(self.current_char());
                self.advance();
                while !self.is_at_end() && self.current_char().is_ascii_digit() {
                    value.push(self.current_char());
                    self.advance();
                }
            }
        }

        if !self.is_at_end() && (self.current_char() == 'e' || self.current_char() == 'E') {
            is_float = true;
            value.push(self.current_char());
            self.advance();
            if !self.is_at_end() && (self.current_char() == '+' || self.current_char() == '-') {
                value.push(self.current_char());
                self.advance();
            }

            while !self.is_at_end() && self.current_char().is_ascii_digit() {
                value.push(self.current_char());
                self.advance();
            }
        }

        let kind = if is_float {
            TokenKind::Float
        } else {
            TokenKind::Integer
        };

        // Intern numbers to save memory (many repeated literals like 0, 1, 2)
        let symbol = self.interner.intern(&value);
        let interned = self.interner.get(symbol).to_string();
        Ok((kind, interned))
    }

    fn scan_identifier(&mut self) -> Result<(TokenKind, String)> {
        let mut value = String::new();
        while !self.is_at_end()
            && (self.current_char().is_alphanumeric() || self.current_char() == '_')
        {
            value.push(self.current_char());
            self.advance();
        }

        let kind = TokenKind::keyword(&value).unwrap_or(TokenKind::Identifier);

        // Intern identifiers - this is the big win!
        // Identifiers like 'local', 'int', 'function' appear many times
        let symbol = self.interner.intern(&value);
        let interned = self.interner.get(symbol).to_string();
        Ok((kind, interned))
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
        self.input[self.position..]
            .chars()
            .next()
            .unwrap_or('\0')
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
