use super::*;
impl Compiler {
    pub(super) fn analyze_free_variables(
        &self,
        expr: &Expr,
        params: &[(String, Option<crate::ast::Type>)],
    ) -> Result<Vec<String>> {
        use std::collections::HashSet;
        let mut free_vars = HashSet::new();
        let mut bound_vars = HashSet::new();
        for (param_name, _) in params {
            bound_vars.insert(param_name.clone());
        }

        self.find_free_vars_in_expr(expr, &mut free_vars, &mut bound_vars);
        let mut captured_vars = Vec::new();
        for var in free_vars {
            if self.resolve_local(&var).is_ok() {
                captured_vars.push(var);
            }
        }

        Ok(captured_vars)
    }

    pub(super) fn find_free_vars_in_expr(
        &self,
        expr: &Expr,
        free_vars: &mut std::collections::HashSet<String>,
        bound_vars: &std::collections::HashSet<String>,
    ) {
        match &expr.kind {
            ExprKind::Identifier(name) => {
                if !bound_vars.contains(name) && !self.is_stdlib_symbol(name) {
                    free_vars.insert(name.clone());
                }
            }

            ExprKind::Binary { left, right, .. } => {
                self.find_free_vars_in_expr(left, free_vars, bound_vars);
                self.find_free_vars_in_expr(right, free_vars, bound_vars);
            }

            ExprKind::Unary { operand, .. } => {
                self.find_free_vars_in_expr(operand, free_vars, bound_vars);
            }

            ExprKind::Call { callee, args } => {
                self.find_free_vars_in_expr(callee, free_vars, bound_vars);
                for arg in args {
                    self.find_free_vars_in_expr(arg, free_vars, bound_vars);
                }
            }

            ExprKind::MethodCall { receiver, args, .. } => {
                self.find_free_vars_in_expr(receiver, free_vars, bound_vars);
                for arg in args {
                    self.find_free_vars_in_expr(arg, free_vars, bound_vars);
                }
            }

            ExprKind::FieldAccess { object, .. } => {
                self.find_free_vars_in_expr(object, free_vars, bound_vars);
            }

            ExprKind::Index { object, index } => {
                self.find_free_vars_in_expr(object, free_vars, bound_vars);
                self.find_free_vars_in_expr(index, free_vars, bound_vars);
            }

            ExprKind::Array(elements) => {
                for elem in elements {
                    self.find_free_vars_in_expr(elem, free_vars, bound_vars);
                }
            }

            ExprKind::Map(entries) => {
                for (key, value) in entries {
                    self.find_free_vars_in_expr(key, free_vars, bound_vars);
                    self.find_free_vars_in_expr(value, free_vars, bound_vars);
                }
            }

            ExprKind::Tuple(elements) => {
                for element in elements {
                    self.find_free_vars_in_expr(element, free_vars, bound_vars);
                }
            }

            ExprKind::StructLiteral { fields, .. } => {
                for field in fields {
                    self.find_free_vars_in_expr(&field.value, free_vars, bound_vars);
                }
            }

            ExprKind::EnumConstructor { args, .. } => {
                for arg in args {
                    self.find_free_vars_in_expr(arg, free_vars, bound_vars);
                }
            }

            ExprKind::Block(stmts) => {
                let mut block_bound = bound_vars.clone();
                for stmt in stmts {
                    self.find_free_vars_in_stmt(stmt, free_vars, &mut block_bound);
                }
            }

            ExprKind::Lambda { params, body, .. } => {
                let mut lambda_bound = bound_vars.clone();
                for (param_name, _) in params {
                    lambda_bound.insert(param_name.clone());
                }

                self.find_free_vars_in_expr(body, free_vars, &lambda_bound);
            }

            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.find_free_vars_in_expr(condition, free_vars, bound_vars);
                self.find_free_vars_in_expr(then_branch, free_vars, bound_vars);
                if let Some(else_expr) = else_branch {
                    self.find_free_vars_in_expr(else_expr, free_vars, bound_vars);
                }
            }

            ExprKind::Cast { expr, .. } => {
                self.find_free_vars_in_expr(expr, free_vars, bound_vars);
            }

            ExprKind::TypeCheck { expr, .. } => {
                self.find_free_vars_in_expr(expr, free_vars, bound_vars);
            }

            ExprKind::IsPattern { expr, pattern } => {
                self.find_free_vars_in_expr(expr, free_vars, bound_vars);
                self.find_free_vars_in_pattern(pattern, free_vars, bound_vars);
            }

            ExprKind::Range { start, end, .. } => {
                self.find_free_vars_in_expr(start, free_vars, bound_vars);
                self.find_free_vars_in_expr(end, free_vars, bound_vars);
            }

            ExprKind::Return(values) => {
                for expr in values {
                    self.find_free_vars_in_expr(expr, free_vars, bound_vars);
                }
            }

            ExprKind::Paren(inner) => {
                self.find_free_vars_in_expr(inner, free_vars, bound_vars);
            }

            ExprKind::Literal(_) => {}
        }
    }

    pub(super) fn find_free_vars_in_stmt(
        &self,
        stmt: &Stmt,
        free_vars: &mut std::collections::HashSet<String>,
        bound_vars: &mut std::collections::HashSet<String>,
    ) {
        match &stmt.kind {
            StmtKind::Local {
                bindings,
                initializer,
                ..
            } => {
                if let Some(exprs) = initializer {
                    for expr in exprs {
                        self.find_free_vars_in_expr(expr, free_vars, bound_vars);
                    }
                }

                for binding in bindings {
                    bound_vars.insert(binding.name.clone());
                }
            }

            StmtKind::Assign { targets, values } => {
                for target in targets {
                    self.find_free_vars_in_expr(target, free_vars, bound_vars);
                }

                for value in values {
                    self.find_free_vars_in_expr(value, free_vars, bound_vars);
                }
            }

            StmtKind::CompoundAssign { target, value, .. } => {
                self.find_free_vars_in_expr(target, free_vars, bound_vars);
                self.find_free_vars_in_expr(value, free_vars, bound_vars);
            }

            StmtKind::Expr(expr) => {
                self.find_free_vars_in_expr(expr, free_vars, bound_vars);
            }

            StmtKind::If {
                condition,
                then_block,
                elseif_branches,
                else_block,
            } => {
                self.find_free_vars_in_expr(condition, free_vars, bound_vars);
                for stmt in then_block {
                    self.find_free_vars_in_stmt(stmt, free_vars, bound_vars);
                }

                for (cond, block) in elseif_branches {
                    self.find_free_vars_in_expr(cond, free_vars, bound_vars);
                    for stmt in block {
                        self.find_free_vars_in_stmt(stmt, free_vars, bound_vars);
                    }
                }

                if let Some(block) = else_block {
                    for stmt in block {
                        self.find_free_vars_in_stmt(stmt, free_vars, bound_vars);
                    }
                }
            }

            StmtKind::While { condition, body } => {
                self.find_free_vars_in_expr(condition, free_vars, bound_vars);
                for stmt in body {
                    self.find_free_vars_in_stmt(stmt, free_vars, bound_vars);
                }
            }

            StmtKind::ForNumeric {
                variable,
                start,
                end,
                step,
                body,
            } => {
                self.find_free_vars_in_expr(start, free_vars, bound_vars);
                self.find_free_vars_in_expr(end, free_vars, bound_vars);
                if let Some(step_expr) = step {
                    self.find_free_vars_in_expr(step_expr, free_vars, bound_vars);
                }

                let mut loop_bound = bound_vars.clone();
                loop_bound.insert(variable.clone());
                for stmt in body {
                    self.find_free_vars_in_stmt(stmt, free_vars, &mut loop_bound);
                }
            }

            StmtKind::ForIn {
                variables,
                iterator,
                body,
            } => {
                self.find_free_vars_in_expr(iterator, free_vars, bound_vars);
                let mut loop_bound = bound_vars.clone();
                for var in variables {
                    loop_bound.insert(var.clone());
                }

                for stmt in body {
                    self.find_free_vars_in_stmt(stmt, free_vars, &mut loop_bound);
                }
            }

            StmtKind::Return(values) => {
                for expr in values {
                    self.find_free_vars_in_expr(expr, free_vars, bound_vars);
                }
            }

            StmtKind::Block(stmts) => {
                for stmt in stmts {
                    self.find_free_vars_in_stmt(stmt, free_vars, bound_vars);
                }
            }

            StmtKind::Break | StmtKind::Continue => {}
        }
    }

    pub(super) fn find_free_vars_in_pattern(
        &self,
        pattern: &crate::ast::Pattern,
        free_vars: &mut std::collections::HashSet<String>,
        bound_vars: &std::collections::HashSet<String>,
    ) {
        use crate::ast::Pattern;
        match pattern {
            Pattern::Wildcard | Pattern::Literal(_) | Pattern::TypeCheck(_) => {}
            Pattern::Identifier(_) => {}
            Pattern::Enum { bindings, .. } => {
                for binding in bindings {
                    self.find_free_vars_in_pattern(binding, free_vars, bound_vars);
                }
            }

            Pattern::Struct { fields, .. } => {
                for (_, pat) in fields {
                    self.find_free_vars_in_pattern(pat, free_vars, bound_vars);
                }
            }
        }
    }
}
