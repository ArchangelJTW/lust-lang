use super::TypeChecker;
use crate::{ast::*, error::Result};
impl TypeChecker {
    pub(super) fn check_stmt(&mut self, stmt: &Stmt) -> Result<()> {
        match &stmt.kind {
            StmtKind::Local {
                bindings,
                mutable,
                initializer,
            } => self.check_local_stmt(
                bindings,
                *mutable,
                initializer.as_ref().map(|values| values.as_slice()),
            ),
            StmtKind::Assign { targets, values } => self.check_assign_stmt(targets, values),
            StmtKind::CompoundAssign { target, op, value } => {
                self.check_compound_assign(target, op, value)
            }

            StmtKind::Expr(expr) => {
                self.check_expr(expr)?;
                Ok(())
            }

            StmtKind::If {
                condition,
                then_block,
                elseif_branches,
                else_block,
            } => self.check_if_stmt(condition, then_block, elseif_branches, else_block.as_ref()),
            StmtKind::While { condition, body } => self.check_while_stmt(condition, body),
            StmtKind::ForNumeric {
                variable,
                start,
                end,
                step,
                body,
            } => self.check_for_numeric_stmt(variable, start, end, step.as_ref(), body),
            StmtKind::ForIn {
                variables,
                iterator,
                body,
            } => self.check_for_in_stmt(variables, iterator, body),
            StmtKind::Return(values) => self.check_return_stmt(values),
            StmtKind::Break => {
                if !self.in_loop {
                    return Err(self.type_error("'break' outside of loop".to_string()));
                }

                Ok(())
            }

            StmtKind::Continue => {
                if !self.in_loop {
                    return Err(self.type_error("'continue' outside of loop".to_string()));
                }

                Ok(())
            }

            StmtKind::Block(stmts) => {
                self.env.push_scope();
                for stmt in stmts {
                    self.check_stmt(stmt)?;
                }

                self.env.pop_scope();
                Ok(())
            }
        }
    }

    pub(super) fn check_local_stmt(
        &mut self,
        bindings: &[LocalBinding],
        _mutable: bool,
        initializer: Option<&[Expr]>,
    ) -> Result<()> {
        if bindings.is_empty() {
            return Ok(());
        }

        let binding_count = bindings.len();
        let mut expr_types: Vec<Type> = Vec::new();
        let mut expr_generics: Vec<Option<std::collections::HashMap<String, Type>>> = Vec::new();
        let annotation_hint = if bindings.len() == 1 {
            bindings[0]
                .type_annotation
                .as_ref()
                .map(|ty| self.canonicalize_type(ty))
        } else {
            None
        };
        if let Some(exprs) = initializer {
            for expr in exprs {
                let mut raw_type = if let Some(hint) = annotation_hint.as_ref() {
                    self.check_expr_with_hint(expr, Some(hint))?
                } else {
                    self.check_expr(expr)?
                };
                if raw_type.span.start_line == 0 && expr.span.start_line > 0 {
                    raw_type.span = expr.span;
                }

                let expr_type = self.canonicalize_type(&raw_type);
                let generics = self.pending_generic_instances.take();
                expr_types.push(expr_type);
                expr_generics.push(generics);
            }
        }

        let mut value_types: Vec<Type> = Vec::new();
        let mut value_generics: Vec<Option<std::collections::HashMap<String, Type>>> = Vec::new();
        if let Some(exprs) = initializer {
            if binding_count == 1 && exprs.len() == 1 {
                value_types.push(expr_types[0].clone());
                value_generics.push(expr_generics[0].clone());
            } else {
                for (ty, generics) in expr_types.iter().zip(expr_generics.iter()) {
                    if let TypeKind::Tuple(elements) = &ty.kind {
                        if elements.is_empty() {
                            continue;
                        }

                        for element in elements {
                            value_types.push(self.canonicalize_type(element));
                            value_generics.push(None);
                        }
                    } else {
                        value_types.push(ty.clone());
                        value_generics.push(generics.clone());
                    }
                }

                if value_types.len() != binding_count {
                    return Err(self.type_error(format!(
                        "Initializer provides {} value(s) but {} binding(s) declared",
                        value_types.len(),
                        binding_count
                    )));
                }
            }
        } else {
            for binding in bindings {
                if binding.type_annotation.is_none() {
                    return Err(self.type_error(format!(
                        "Variable '{}' must have either a type annotation or an initializer",
                        binding.name
                    )));
                }
            }
        }

        for (index, binding) in bindings.iter().enumerate() {
            let annotation = binding
                .type_annotation
                .as_ref()
                .map(|ty| self.canonicalize_type(ty));
            let inferred = if initializer.is_some() {
                if binding_count == 1 && expr_types.len() == 1 {
                    Some(expr_types[0].clone())
                } else {
                    Some(value_types[index].clone())
                }
            } else {
                None
            };
            let var_type = match (annotation, inferred) {
                (Some(ann), Some(inf)) => {
                    self.unify(&ann, &inf)?;
                    ann
                }

                (Some(ann), None) => ann,
                (None, Some(inf)) => inf,
                (None, None) => unreachable!(),
            };
            let generics_map = if initializer.is_some() {
                if binding_count == 1 && expr_types.len() == 1 {
                    expr_generics[0].clone()
                } else {
                    value_generics.get(index).cloned().flatten()
                }
            } else {
                None
            };
            if let Some(type_params) = generics_map {
                for (type_param, concrete_type) in type_params {
                    self.env.record_generic_instance(
                        binding.name.clone(),
                        type_param,
                        concrete_type,
                    );
                }
            }

            if binding.span.start_line > 0 {
                if let Some(module) = &self.current_module {
                    self.variable_types_by_module
                        .entry(module.clone())
                        .or_default()
                        .insert(binding.span, var_type.clone());
                }
            }

            self.env.declare_variable(binding.name.clone(), var_type)?;
        }

        Ok(())
    }

    fn check_assign_stmt(&mut self, targets: &[Expr], values: &[Expr]) -> Result<()> {
        if targets.is_empty() {
            return Ok(());
        }

        if values.is_empty() {
            return Err(self.type_error("Assignment requires a value".to_string()));
        }

        let mut expr_types: Vec<Type> = Vec::new();
        for value in values {
            let raw_type = self.check_expr(value)?;
            let val_type = self.canonicalize_type(&raw_type);
            self.pending_generic_instances.take();
            expr_types.push(val_type);
        }

        let mut expanded_types: Vec<Type> = Vec::new();
        if targets.len() == 1 && values.len() == 1 {
            expanded_types.push(expr_types[0].clone());
        } else {
            for ty in &expr_types {
                if let TypeKind::Tuple(elements) = &ty.kind {
                    if elements.is_empty() {
                        continue;
                    }

                    for element in elements {
                        expanded_types.push(self.canonicalize_type(element));
                    }
                } else {
                    expanded_types.push(ty.clone());
                }
            }

            if expanded_types.len() != targets.len() {
                return Err(self.type_error(format!(
                    "Assignment provides {} value(s) but {} target(s) declared",
                    expanded_types.len(),
                    targets.len()
                )));
            }
        }

        for (index, target) in targets.iter().enumerate() {
            let raw_target_type = self.check_expr(target)?;
            let target_type = self.canonicalize_type(&raw_target_type);
            let value_type = if targets.len() == 1 && values.len() == 1 {
                expr_types[0].clone()
            } else {
                expanded_types[index].clone()
            };
            if let ExprKind::FieldAccess { object, field } = &target.kind {
                if let Some(inner_type) = self.weak_field_target_type(object, field)? {
                    match self.unify(&target_type, &value_type) {
                        Ok(_) => continue,
                        Err(err) => {
                            if self.types_equal(&inner_type, &value_type) {
                                continue;
                            } else {
                                return Err(err);
                            }
                        }
                    }
                }
            }

            self.unify(&target_type, &value_type)?;
        }

        Ok(())
    }

    fn check_compound_assign(&mut self, target: &Expr, op: &BinaryOp, value: &Expr) -> Result<()> {
        let target_type = self.check_expr(target)?;
        let value_type = self.check_expr(value)?;
        let result_type = match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
                if matches!(target_type.kind, TypeKind::Int | TypeKind::Float)
                    && matches!(value_type.kind, TypeKind::Int | TypeKind::Float)
                {
                    target_type.clone()
                } else {
                    return Err(self.type_error(format!(
                        "Compound assignment {}= requires numeric types",
                        op
                    )));
                }
            }

            _ => {
                return Err(self.type_error(format!(
                    "Compound assignment not supported for operator {}",
                    op
                )));
            }
        };
        self.unify(&target_type, &result_type)?;
        Ok(())
    }

    fn check_if_stmt(
        &mut self,
        condition: &Expr,
        then_block: &[Stmt],
        elseif_branches: &[(Expr, Vec<Stmt>)],
        else_block: Option<&Vec<Stmt>>,
    ) -> Result<()> {
        let pattern_bindings = self.extract_all_pattern_bindings(condition);
        let cond_type = self.check_expr(condition)?;
        self.unify(
            &Type::new(TypeKind::Bool, TypeChecker::dummy_span()),
            &cond_type,
        )?;
        self.env.push_scope();
        for (scrutinee_expr, pattern) in pattern_bindings {
            if let Ok(scrutinee_type) = self.check_expr(scrutinee_expr) {
                self.bind_pattern(&pattern, &scrutinee_type)?;
            }
        }

        let type_narrowings = self.extract_type_narrowings(condition);
        for (var_name, narrowed_type) in type_narrowings {
            self.env.refine_variable_type(var_name, narrowed_type);
        }

        for stmt in then_block {
            self.check_stmt(stmt)?;
        }

        self.env.pop_scope();
        for (elseif_cond, elseif_body) in elseif_branches {
            let elseif_pattern_bindings = self.extract_all_pattern_bindings(elseif_cond);
            let elseif_cond_type = self.check_expr(elseif_cond)?;
            self.unify(
                &Type::new(TypeKind::Bool, TypeChecker::dummy_span()),
                &elseif_cond_type,
            )?;
            self.env.push_scope();
            for (scrutinee_expr, pattern) in elseif_pattern_bindings {
                if let Ok(scrutinee_type) = self.check_expr(scrutinee_expr) {
                    self.bind_pattern(&pattern, &scrutinee_type)?;
                }
            }

            let elseif_type_narrowings = self.extract_type_narrowings(elseif_cond);
            for (var_name, narrowed_type) in elseif_type_narrowings {
                self.env.refine_variable_type(var_name, narrowed_type);
            }

            for stmt in elseif_body {
                self.check_stmt(stmt)?;
            }

            self.env.pop_scope();
        }

        if let Some(else_stmts) = else_block {
            self.env.push_scope();
            for stmt in else_stmts {
                self.check_stmt(stmt)?;
            }

            self.env.pop_scope();
        }

        Ok(())
    }

    fn extract_all_pattern_bindings<'a>(&self, expr: &'a Expr) -> Vec<(&'a Expr, Pattern)> {
        let mut bindings = Vec::new();
        match &expr.kind {
            ExprKind::IsPattern {
                expr: scrutinee,
                pattern,
            } => match pattern {
                Pattern::Enum {
                    bindings: pattern_bindings,
                    ..
                } if !pattern_bindings.is_empty() => {
                    bindings.push((scrutinee.as_ref(), pattern.clone()));
                }

                _ => {}
            },
            ExprKind::Binary { left, op, right } => {
                if matches!(op, BinaryOp::And) {
                    bindings.extend(self.extract_all_pattern_bindings(left));
                    bindings.extend(self.extract_all_pattern_bindings(right));
                }
            }

            _ => {}
        }

        bindings
    }

    fn extract_type_narrowings(&mut self, expr: &Expr) -> Vec<(String, Type)> {
        let mut narrowings = Vec::new();
        match &expr.kind {
            ExprKind::TypeCheck {
                expr: scrutinee,
                check_type: target_type,
            } => {
                if let ExprKind::Identifier(var_name) = &scrutinee.kind {
                    if let Some(current_type) = self.env.lookup_variable(var_name) {
                        let narrowed_type = if let TypeKind::Named(name) = &target_type.kind {
                            let resolved_trait = self.resolve_type_key(name);
                            if self.env.lookup_trait(&resolved_trait).is_some() {
                                Type::new(TypeKind::Trait(name.clone()), target_type.span)
                            } else {
                                target_type.clone()
                            }
                        } else {
                            target_type.clone()
                        };
                        match &current_type.kind {
                            TypeKind::Unknown => {
                                narrowings.push((var_name.clone(), narrowed_type));
                            }

                            TypeKind::Union(types) => {
                                for ty in types {
                                    if self.types_equal(ty, target_type) {
                                        narrowings.push((var_name.clone(), target_type.clone()));
                                        break;
                                    }
                                }
                            }

                            _ => {}
                        }
                    }
                }
            }

            ExprKind::IsPattern {
                expr: scrutinee,
                pattern,
            } => {
                if let Pattern::TypeCheck(target_type) = pattern {
                    if let ExprKind::Identifier(var_name) = &scrutinee.kind {
                        if let Some(current_type) = self.env.lookup_variable(var_name) {
                            let narrowed_type = if let TypeKind::Named(name) = &target_type.kind {
                                let resolved_trait = self.resolve_type_key(name);
                                if self.env.lookup_trait(&resolved_trait).is_some() {
                                    Type::new(TypeKind::Trait(name.clone()), target_type.span)
                                } else {
                                    target_type.clone()
                                }
                            } else {
                                target_type.clone()
                            };
                            match &current_type.kind {
                                TypeKind::Unknown => {
                                    narrowings.push((var_name.clone(), narrowed_type));
                                }

                                TypeKind::Union(types) => {
                                    for ty in types {
                                        if self.types_equal(ty, target_type) {
                                            narrowings
                                                .push((var_name.clone(), target_type.clone()));
                                            break;
                                        }
                                    }
                                }

                                _ => {}
                            }
                        }
                    }
                }
            }

            ExprKind::Binary { left, op, right } => {
                if matches!(op, BinaryOp::And) {
                    narrowings.extend(self.extract_type_narrowings(left));
                    narrowings.extend(self.extract_type_narrowings(right));
                }
            }

            _ => {}
        }

        narrowings
    }

    fn check_while_stmt(&mut self, condition: &Expr, body: &[Stmt]) -> Result<()> {
        let cond_type = self.check_expr(condition)?;
        self.unify(
            &Type::new(TypeKind::Bool, TypeChecker::dummy_span()),
            &cond_type,
        )?;
        let prev_in_loop = self.in_loop;
        self.in_loop = true;
        self.env.push_scope();
        for stmt in body {
            self.check_stmt(stmt)?;
        }

        self.env.pop_scope();
        self.in_loop = prev_in_loop;
        Ok(())
    }

    fn check_for_numeric_stmt(
        &mut self,
        variable: &str,
        start: &Expr,
        end: &Expr,
        step: Option<&Expr>,
        body: &[Stmt],
    ) -> Result<()> {
        let start_type = self.check_expr(start)?;
        let end_type = self.check_expr(end)?;
        if !matches!(start_type.kind, TypeKind::Int | TypeKind::Float) {
            return Err(self.type_error(format!(
                "For loop start value must be numeric, got {}",
                start_type
            )));
        }

        if !matches!(end_type.kind, TypeKind::Int | TypeKind::Float) {
            return Err(self.type_error(format!(
                "For loop end value must be numeric, got {}",
                end_type
            )));
        }

        if let Some(step_expr) = step {
            let step_type = self.check_expr(step_expr)?;
            if !matches!(step_type.kind, TypeKind::Int | TypeKind::Float) {
                return Err(self.type_error(format!(
                    "For loop step value must be numeric, got {}",
                    step_type
                )));
            }
        }

        let loop_var_type = start_type.clone();
        let prev_in_loop = self.in_loop;
        self.in_loop = true;
        self.env.push_scope();
        self.env
            .declare_variable(variable.to_string(), loop_var_type)?;
        for stmt in body {
            self.check_stmt(stmt)?;
        }

        self.env.pop_scope();
        self.in_loop = prev_in_loop;
        Ok(())
    }

    fn check_for_in_stmt(
        &mut self,
        variables: &[String],
        iterator: &Expr,
        body: &[Stmt],
    ) -> Result<()> {
        let iterator_type = self.check_expr(iterator)?;
        if let crate::ast::ExprKind::MethodCall {
            receiver, method, ..
        } = &iterator.kind
        {
            let recv_ty = self.check_expr(receiver)?;
            match (&recv_ty.kind, method.as_str()) {
                (TypeKind::Map(key_ty, val_ty), "iter") => {
                    if variables.len() != 2 {
                        return Err(self.type_error(format!(
                            "Map iteration yields 2 values (key, value), but {} variables were specified",
                            variables.len()
                        )));
                    }

                    let prev_in_loop = self.in_loop;
                    self.in_loop = true;
                    self.env.push_scope();
                    self.env
                        .declare_variable(variables[0].clone(), (**key_ty).clone())?;
                    self.env
                        .declare_variable(variables[1].clone(), (**val_ty).clone())?;
                    for stmt in body {
                        self.check_stmt(stmt)?;
                    }

                    self.env.pop_scope();
                    self.in_loop = prev_in_loop;
                    return Ok(());
                }

                (TypeKind::Table, "iter") => {
                    if variables.len() != 2 {
                        return Err(self.type_error(format!(
                            "Table iteration yields 2 values (key, value), but {} variables were specified",
                            variables.len()
                        )));
                    }

                    let prev_in_loop = self.in_loop;
                    self.in_loop = true;
                    self.env.push_scope();
                    let unknown = Type::new(TypeKind::Unknown, TypeChecker::dummy_span());
                    self.env
                        .declare_variable(variables[0].clone(), unknown.clone())?;
                    self.env.declare_variable(variables[1].clone(), unknown)?;
                    for stmt in body {
                        self.check_stmt(stmt)?;
                    }

                    self.env.pop_scope();
                    self.in_loop = prev_in_loop;
                    return Ok(());
                }

                (TypeKind::Array(elem_ty), "iter") => {
                    if variables.len() != 1 {
                        return Err(self.type_error(format!(
                            "Array iteration yields 1 value, but {} variables were specified",
                            variables.len()
                        )));
                    }

                    let prev_in_loop = self.in_loop;
                    self.in_loop = true;
                    self.env.push_scope();
                    self.env
                        .declare_variable(variables[0].clone(), (**elem_ty).clone())?;
                    for stmt in body {
                        self.check_stmt(stmt)?;
                    }

                    self.env.pop_scope();
                    self.in_loop = prev_in_loop;
                    return Ok(());
                }

                _ => {}
            }
        }

        match &iterator_type.kind {
            TypeKind::Array(elem_type) => {
                if variables.len() != 1 {
                    return Err(self.type_error(format!(
                        "Array iteration yields 1 value, but {} variables were specified",
                        variables.len()
                    )));
                }

                let prev_in_loop = self.in_loop;
                self.in_loop = true;
                self.env.push_scope();
                self.env
                    .declare_variable(variables[0].clone(), (**elem_type).clone())?;
                for stmt in body {
                    self.check_stmt(stmt)?;
                }

                self.env.pop_scope();
                self.in_loop = prev_in_loop;
                Ok(())
            }

            _ => Err(self.type_error(format!("Cannot iterate over type {}", iterator_type))),
        }
    }

    fn check_return_stmt(&mut self, values: &[Expr]) -> Result<()> {
        let return_type = if values.is_empty() {
            Type::new(TypeKind::Unit, TypeChecker::dummy_span())
        } else if values.len() == 1 {
            let raw_ty = self.check_expr(&values[0])?;
            let ty = self.canonicalize_type(&raw_ty);
            self.pending_generic_instances.take();
            ty
        } else {
            let mut element_types = Vec::new();
            for value in values {
                let raw_ty = self.check_expr(value)?;
                let ty = self.canonicalize_type(&raw_ty);
                self.pending_generic_instances.take();
                element_types.push(ty);
            }

            Type::new(TypeKind::Tuple(element_types), TypeChecker::dummy_span())
        };
        if let Some(expected_return) = &self.current_function_return_type {
            self.unify(expected_return, &return_type)?;
        } else {
            return Err(self.type_error("'return' outside of function".to_string()));
        }

        Ok(())
    }

    fn weak_field_target_type(&mut self, object: &Expr, field_name: &str) -> Result<Option<Type>> {
        let object_type = self.check_expr(object)?;
        let canonical_object = self.canonicalize_type(&object_type);
        let struct_name = match &canonical_object.kind {
            TypeKind::Named(name) => name.clone(),
            TypeKind::GenericInstance { name, .. } => name.clone(),
            _ => return Ok(None),
        };
        let resolved = self.resolve_type_key(&struct_name);
        let struct_def = self
            .env
            .lookup_struct(&resolved)
            .or_else(|| self.env.lookup_struct(&struct_name));
        let struct_def = match struct_def {
            Some(def) => def,
            None => return Ok(None),
        };
        if let Some(field) = struct_def.fields.iter().find(|f| f.name == *field_name) {
            if matches!(field.ownership, FieldOwnership::Weak) {
                if let Some(inner) = &field.weak_target {
                    return Ok(Some(self.canonicalize_type(inner)));
                }
            }
        }

        Ok(None)
    }
}
