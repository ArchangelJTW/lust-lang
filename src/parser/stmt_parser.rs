use super::Parser;
use crate::{
    ast::{Expr, LocalBinding, Stmt, StmtKind},
    error::Result,
    lexer::TokenKind,
};
use alloc::{vec, vec::Vec};
impl Parser {
    pub(super) fn parse_stmt(&mut self) -> Result<Stmt> {
        let start_token = self.current_token().clone();
        let kind = match self.peek_kind() {
            TokenKind::Local => {
                self.advance();
                let mutable = self.match_token(&[TokenKind::Mut]);
                let mut bindings = Vec::new();
                loop {
                    let name_token = self.current_token().clone();
                    self.consume(TokenKind::Identifier, "Expected identifier")?;
                    let name = name_token.lexeme.clone();
                    let type_annotation = if self.match_token(&[TokenKind::Colon]) {
                        Some(self.parse_type()?)
                    } else {
                        None
                    };
                    bindings.push(LocalBinding {
                        name,
                        type_annotation,
                        span: self.make_span(&name_token, &name_token),
                    });
                    if !self.match_token(&[TokenKind::Comma]) {
                        break;
                    }
                }

                let initializer = if self.match_token(&[TokenKind::Equal]) {
                    Some(self.parse_expr_list()?)
                } else {
                    None
                };
                StmtKind::Local {
                    mutable,
                    bindings,
                    initializer,
                }
            }

            TokenKind::If => {
                self.advance();
                let condition = self.parse_expr()?;
                self.consume(TokenKind::Then, "Expected 'then' after if condition")?;
                let mut then_block = Vec::new();
                while !self.check(TokenKind::Elseif)
                    && !self.check(TokenKind::Else)
                    && !self.check(TokenKind::End)
                    && !self.is_at_end()
                {
                    then_block.push(self.parse_stmt()?);
                }

                let mut elseif_branches = Vec::new();
                while self.match_token(&[TokenKind::Elseif]) {
                    let elseif_condition = self.parse_expr()?;
                    self.consume(TokenKind::Then, "Expected 'then' after elseif condition")?;
                    let mut elseif_block = Vec::new();
                    while !self.check(TokenKind::Elseif)
                        && !self.check(TokenKind::Else)
                        && !self.check(TokenKind::End)
                        && !self.is_at_end()
                    {
                        elseif_block.push(self.parse_stmt()?);
                    }

                    elseif_branches.push((elseif_condition, elseif_block));
                }

                let else_block = if self.match_token(&[TokenKind::Else]) {
                    let mut block = Vec::new();
                    while !self.check(TokenKind::End) && !self.is_at_end() {
                        block.push(self.parse_stmt()?);
                    }

                    Some(block)
                } else {
                    None
                };
                self.consume(TokenKind::End, "Expected 'end' after if statement")?;
                StmtKind::If {
                    condition,
                    then_block,
                    elseif_branches,
                    else_block,
                }
            }

            TokenKind::While => {
                self.advance();
                let condition = self.parse_expr()?;
                self.consume(TokenKind::Do, "Expected 'do' after while condition")?;
                let mut body = Vec::new();
                while !self.check(TokenKind::End) && !self.is_at_end() {
                    body.push(self.parse_stmt()?);
                }

                self.consume(TokenKind::End, "Expected 'end' after while body")?;
                StmtKind::While { condition, body }
            }

            TokenKind::For => {
                self.advance();
                let first_var = self.expect_identifier()?;
                if self.match_token(&[TokenKind::Equal]) {
                    let start = self.parse_expr()?;
                    self.consume(TokenKind::Comma, "Expected ',' after for loop start value")?;
                    let end = self.parse_expr()?;
                    let step = if self.match_token(&[TokenKind::Comma]) {
                        Some(self.parse_expr()?)
                    } else {
                        None
                    };
                    self.consume(TokenKind::Do, "Expected 'do' after for loop range")?;
                    let mut body = Vec::new();
                    while !self.check(TokenKind::End) && !self.is_at_end() {
                        body.push(self.parse_stmt()?);
                    }

                    self.consume(TokenKind::End, "Expected 'end' after for body")?;
                    StmtKind::ForNumeric {
                        variable: first_var,
                        start,
                        end,
                        step,
                        body,
                    }
                } else {
                    let mut variables = vec![first_var];
                    while self.match_token(&[TokenKind::Comma]) {
                        variables.push(self.expect_identifier()?);
                    }

                    self.consume(TokenKind::In, "Expected 'in' after for variable(s)")?;
                    let iterator = self.parse_expr()?;
                    self.consume(TokenKind::Do, "Expected 'do' after for iterator")?;
                    let mut body = Vec::new();
                    while !self.check(TokenKind::End) && !self.is_at_end() {
                        body.push(self.parse_stmt()?);
                    }

                    self.consume(TokenKind::End, "Expected 'end' after for body")?;
                    StmtKind::ForIn {
                        variables,
                        iterator,
                        body,
                    }
                }
            }

            TokenKind::Return => {
                self.advance();
                let values = if self.check(TokenKind::End)
                    || self.check(TokenKind::Elseif)
                    || self.check(TokenKind::Else)
                    || self.is_at_end()
                {
                    Vec::new()
                } else {
                    self.parse_expr_list()?
                };
                StmtKind::Return(values)
            }

            TokenKind::Break => {
                self.advance();
                StmtKind::Break
            }

            TokenKind::Continue => {
                self.advance();
                StmtKind::Continue
            }

            TokenKind::Do => {
                self.advance();
                let mut statements = Vec::new();
                while !self.check(TokenKind::End) && !self.is_at_end() {
                    statements.push(self.parse_stmt()?);
                }

                self.consume(TokenKind::End, "Expected 'end' after do block")?;
                StmtKind::Block(statements)
            }

            _ => {
                if let Some(local_stmt) = self.try_parse_implicit_global_decl()? {
                    local_stmt
                } else {
                    let expr = self.parse_expr()?;
                    let mut targets = vec![expr];
                    while self.match_token(&[TokenKind::Comma]) {
                        targets.push(self.parse_expr()?);
                    }

                    if self.match_token(&[TokenKind::Equal]) {
                        let values = self.parse_expr_list()?;
                        StmtKind::Assign { targets, values }
                    } else if self.match_token(&[
                        TokenKind::PlusEqual,
                        TokenKind::MinusEqual,
                        TokenKind::StarEqual,
                        TokenKind::SlashEqual,
                    ]) && targets.len() == 1
                    {
                        let op = match self.tokens[self.current - 1].kind {
                            TokenKind::PlusEqual => crate::ast::BinaryOp::Add,
                            TokenKind::MinusEqual => crate::ast::BinaryOp::Sub,
                            TokenKind::StarEqual => crate::ast::BinaryOp::Mul,
                            TokenKind::SlashEqual => crate::ast::BinaryOp::Div,
                            _ => unreachable!(),
                        };
                        let value = self.parse_expr()?;
                        StmtKind::CompoundAssign {
                            target: targets.remove(0),
                            op,
                            value,
                        }
                    } else if targets.len() > 1 {
                        return Err(self.error("Expected '=' after assignment targets"));
                    } else {
                        StmtKind::Expr(targets.remove(0))
                    }
                }
            }
        };
        let end_token = self.tokens[self.current - 1].clone();
        Ok(Stmt::new(kind, self.make_span(&start_token, &end_token)))
    }

    fn try_parse_implicit_global_decl(&mut self) -> Result<Option<StmtKind>> {
        let start_index = self.current;
        if !self.check(TokenKind::Identifier) {
            return Ok(None);
        }

        match self.peek_ahead(1) {
            Some(token) if token.kind == TokenKind::Colon => {}
            _ => return Ok(None),
        }

        let mut bindings = Vec::new();
        loop {
            let name_token = self.current_token().clone();
            self.advance();
            if !self.match_token(&[TokenKind::Colon]) {
                self.current = start_index;
                return Ok(None);
            }

            let type_annotation = match self.parse_type() {
                Ok(ty) => ty,
                Err(err) => {
                    self.current = start_index;
                    return Err(err);
                }
            };
            let type_end_token = self.tokens[self.current - 1].clone();
            bindings.push(LocalBinding {
                name: name_token.lexeme.clone(),
                type_annotation: Some(type_annotation),
                span: self.make_span(&name_token, &type_end_token),
            });
            if self.check(TokenKind::LeftParen) {
                self.current = start_index;
                return Ok(None);
            }

            if self.match_token(&[TokenKind::Comma]) {
                if !self.check(TokenKind::Identifier) {
                    self.current = start_index;
                    return Ok(None);
                }

                match self.peek_ahead(1) {
                    Some(token) if token.kind == TokenKind::Colon => {}
                    _ => {
                        self.current = start_index;
                        return Ok(None);
                    }
                }

                continue;
            }

            break;
        }

        if !self.match_token(&[TokenKind::Equal]) {
            self.current = start_index;
            return Ok(None);
        }

        let initializer = self.parse_expr_list()?;
        Ok(Some(StmtKind::Local {
            bindings,
            mutable: false,
            initializer: Some(initializer),
        }))
    }

    fn parse_expr_list(&mut self) -> Result<Vec<Expr>> {
        let mut exprs = Vec::new();
        exprs.push(self.parse_expr()?);
        while self.match_token(&[TokenKind::Comma]) {
            exprs.push(self.parse_expr()?);
        }

        Ok(exprs)
    }
}
