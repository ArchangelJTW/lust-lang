use super::token::{Token, TokenKind};
use crate::error::{LustError, Result};
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
pub struct Lexer {
    input: Vec<char>,
    position: usize,
    line: usize,
    column: usize,
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        Self {
            input: input.chars().collect(),
            position: 0,
            line: 1,
            column: 1,
        }
    }

    pub fn tokenize(&mut self) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();
        while !self.is_at_end() {
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
        Ok(tokens)
    }

    fn next_token(&mut self) -> Result<Token> {
        let start_line = self.line;
        let start_column = self.column;
        let ch = self.current_char();
        let (kind, lexeme) = match ch {
            '(' => {
                self.advance();
                (TokenKind::LeftParen, "(".to_string())
            }

            ')' => {
                self.advance();
                (TokenKind::RightParen, ")".to_string())
            }

            '{' => {
                self.advance();
                (TokenKind::LeftBrace, "{".to_string())
            }

            '}' => {
                self.advance();
                (TokenKind::RightBrace, "}".to_string())
            }

            '[' => {
                self.advance();
                (TokenKind::LeftBracket, "[".to_string())
            }

            ']' => {
                self.advance();
                (TokenKind::RightBracket, "]".to_string())
            }

            ',' => {
                self.advance();
                (TokenKind::Comma, ",".to_string())
            }

            ';' => {
                self.advance();
                (TokenKind::Semicolon, ";".to_string())
            }

            '%' => {
                self.advance();
                (TokenKind::Percent, "%".to_string())
            }

            '^' => {
                self.advance();
                (TokenKind::Caret, "^".to_string())
            }

            '?' => {
                self.advance();
                (TokenKind::Question, "?".to_string())
            }

            '&' => {
                self.advance();
                (TokenKind::Ampersand, "&".to_string())
            }

            '|' => {
                self.advance();
                (TokenKind::Pipe, "|".to_string())
            }

            '+' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    (TokenKind::PlusEqual, "+=".to_string())
                } else {
                    (TokenKind::Plus, "+".to_string())
                }
            }

            '-' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    (TokenKind::MinusEqual, "-=".to_string())
                } else if self.current_char() == '>' {
                    self.advance();
                    (TokenKind::Arrow, "->".to_string())
                } else {
                    (TokenKind::Minus, "-".to_string())
                }
            }

            '*' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    (TokenKind::StarEqual, "*=".to_string())
                } else {
                    (TokenKind::Star, "*".to_string())
                }
            }

            '/' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    (TokenKind::SlashEqual, "/=".to_string())
                } else {
                    (TokenKind::Slash, "/".to_string())
                }
            }

            '=' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    (TokenKind::DoubleEqual, "==".to_string())
                } else if self.current_char() == '>' {
                    self.advance();
                    (TokenKind::FatArrow, "=>".to_string())
                } else {
                    (TokenKind::Equal, "=".to_string())
                }
            }

            '~' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    (TokenKind::NotEqual, "~=".to_string())
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
                    (TokenKind::NotEqual, "!=".to_string())
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
                    (TokenKind::LessEqual, "<=".to_string())
                } else {
                    (TokenKind::Less, "<".to_string())
                }
            }

            '>' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    (TokenKind::GreaterEqual, ">=".to_string())
                } else {
                    (TokenKind::Greater, ">".to_string())
                }
            }

            ':' => {
                self.advance();
                if self.current_char() == ':' {
                    self.advance();
                    (TokenKind::DoubleColon, "::".to_string())
                } else {
                    (TokenKind::Colon, ":".to_string())
                }
            }

            '.' => {
                self.advance();
                if self.current_char() == '.' {
                    self.advance();
                    (TokenKind::DoubleDot, "..".to_string())
                } else {
                    (TokenKind::Dot, ".".to_string())
                }
            }

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
        Ok((TokenKind::String, value))
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
        Ok((kind, value))
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
        Ok((kind, value))
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
        if self.is_at_end() {
            '\0'
        } else {
            self.input[self.position]
        }
    }

    fn peek(&self, offset: usize) -> Option<char> {
        let pos = self.position + offset;
        if pos < self.input.len() {
            Some(self.input[pos])
        } else {
            None
        }
    }

    fn advance(&mut self) {
        if !self.is_at_end() {
            self.position += 1;
            self.column += 1;
        }
    }

    fn is_at_end(&self) -> bool {
        self.position >= self.input.len()
    }
}
