use crate::ast::{
    BinaryOp, Expr, ExprKind, FunctionDef, Item, ItemKind, Literal, Stmt, StmtKind, StructDef,
    Type, TypeKind, UnaryOp,
};
use crate::data::layout::{CanonicalTypeTag, LogicalLayoutV1};
use crate::data::materialize::struct_def_to_layout;
use crate::mir::ir::*;
use alloc::{boxed::Box, format, string::String, vec, vec::Vec};
use hashbrown::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MirLowerError {
    #[error("Binary operator '{0}' is not supported by MIR lowering")]
    UnsupportedBinaryOp(BinaryOp),
    #[error("Unary operator '{0}' is not supported by MIR lowering")]
    UnsupportedUnaryOp(UnaryOp),
    #[error("Lvalue expression is not supported by MIR lowering: {0}")]
    UnsupportedLvalue(String),
    #[error("Failed or unlowerable expression cannot be lowered to MIR: {0}")]
    UnlowerableExpression(String),
}

#[derive(Debug, Clone)]
pub(super) enum MirCleanupKind {
    Defer,
    ErrDefer,
}

#[derive(Debug, Clone)]
pub(super) struct MirCleanup {
    pub(super) kind: MirCleanupKind,
    pub(super) stmt: Box<Stmt>,
}

#[derive(Debug, Clone)]
pub(super) struct LocalThunk {
    pub(super) body: Vec<Stmt>,
    pub(super) captured_bindings: Vec<(String, LocalId, MirType)>,
    pub(super) result_type: MirType,
    #[allow(dead_code)]
    pub(super) span: crate::ast::Span,
}

#[derive(Debug, Clone)]
pub(super) enum ScopeBinding {
    Local(LocalId),
    Thunk(LocalThunk),
}

pub struct AstLowering<'a> {
    current_fn: &'a mut MirFunction,
    current_block: BlockId,
    scopes: Vec<HashMap<String, ScopeBinding>>,
    cleanups: Vec<Vec<MirCleanup>>,
    loop_stack: Vec<(BlockId, BlockId)>, // (header/continue_block, exit_block)
    struct_fields: HashMap<String, Vec<String>>,
    struct_field_types: HashMap<String, Vec<MirType>>,
    struct_field_substructs: HashMap<String, Vec<Option<String>>>,
    struct_layouts: HashMap<String, LogicalLayoutV1>,
    struct_ops: Vec<MirStructOp>,
    local_field_names: HashMap<LocalId, Vec<String>>,
    known_typed_locals: HashSet<LocalId>,
    thunk_return_target: Option<(LocalId, BlockId)>,
}

impl<'a> AstLowering<'a> {
    pub fn new(func: &'a mut MirFunction) -> Self {
        let entry = func.entry;
        let mut scope = HashMap::new();
        let mut known_typed_locals = HashSet::new();
        // Register parameters in the root scope
        for (idx, p_id) in func.params.iter().enumerate() {
            known_typed_locals.insert(*p_id);
            let p_name = func.locals[p_id.0]
                .name
                .clone()
                .unwrap_or_else(|| format!("arg{idx}"));
            scope.insert(p_name, ScopeBinding::Local(*p_id));
        }

        Self {
            current_fn: func,
            current_block: entry,
            scopes: vec![scope],
            cleanups: vec![Vec::new()],
            loop_stack: Vec::new(),
            struct_fields: HashMap::new(),
            struct_field_types: HashMap::new(),
            struct_field_substructs: HashMap::new(),
            struct_layouts: HashMap::new(),
            struct_ops: Vec::new(),
            local_field_names: HashMap::new(),
            known_typed_locals,
            thunk_return_target: None,
        }
    }

    fn operand_definite_mir_type(&self, op: &Operand) -> MirType {
        match op {
            Operand::Constant(c) => match c {
                MirConstant::Int(_) => MirType::Int,
                MirConstant::Float(_) => MirType::Float,
                MirConstant::Bool(_) => MirType::Bool,
                MirConstant::String(_) => MirType::String,
                MirConstant::Unit => MirType::Unit,
            },
            Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => {
                let ty = &self.current_fn.locals[loc.0].ty;
                if *ty == MirType::Int && !self.known_typed_locals.contains(loc) {
                    MirType::Unknown
                } else {
                    ty.clone()
                }
            }
            _ => MirType::Unknown,
        }
    }

    pub fn register_struct_def(&mut self, def: &StructDef) {
        let layout = struct_def_to_layout(def);
        let field_mir_types: Vec<MirType> = def
            .fields
            .iter()
            .map(|f| mir_type_from_ast(&f.ty))
            .collect();
        self.struct_field_types
            .insert(def.name.clone(), field_mir_types);
        let substructs = def
            .fields
            .iter()
            .map(|f| match &f.ty.kind {
                TypeKind::Named(sub)
                    if !matches!(
                        sub.as_str(),
                        "ObjectId" | "DateTimeUtc" | "Decimal128" | "Binary" | "Bytes"
                    ) =>
                {
                    Some(sub.clone())
                }
                _ => None,
            })
            .collect();
        self.struct_field_substructs
            .insert(def.name.clone(), substructs);
        self.register_struct_layout(layout);
    }

    pub fn register_struct_layout(&mut self, layout: LogicalLayoutV1) {
        let names = layout.fields.iter().map(|f| f.name.clone()).collect();
        self.struct_fields.insert(layout.type_name.clone(), names);
        self.struct_layouts.insert(layout.type_name.clone(), layout);
    }

    pub fn struct_ops(&self) -> &[MirStructOp] {
        &self.struct_ops
    }

    pub fn lower_struct_decode_into(
        &mut self,
        struct_name: &str,
        src_bytes: Operand,
        span: crate::ast::Span,
    ) -> Result<LocalId, MirLowerError> {
        let layout = self.struct_layouts.get(struct_name).cloned().ok_or_else(|| {
            MirLowerError::UnlowerableExpression(format!(
                "Unknown struct layout '{}' for StructDecodeInto",
                struct_name
            ))
        })?;
        let dst = self
            .current_fn
            .add_local(None, MirType::Struct(struct_name.to_string()), true, span);
        let succ_bb = self.current_fn.new_block();
        self.emit_stmt(Statement::StorageLive(dst, span));
        self.struct_ops.push(MirStructOp::StructDecodeInto {
            struct_name: struct_name.to_string(),
            layout: MirStructLayout::from(&layout),
            dst,
            src_bytes: src_bytes.clone(),
        });
        self.terminate_block(Terminator::Call {
            callee: mir_struct_decode_callee(struct_name),
            args: vec![src_bytes],
            destination: Place::Local(dst),
            success: succ_bb,
            error: None,
            span,
        });
        self.current_block = succ_bb;
        Ok(dst)
    }

    pub fn lower_module_items(
        module_name: impl Into<String>,
        items: &[Item],
    ) -> Result<MirModule, MirLowerError> {
        let mut module = MirModule::new(module_name.into());
        let mut defs = Vec::new();
        for item in items {
            if let ItemKind::Struct(def) = &item.kind {
                let layout = struct_def_to_layout(def);
                module.register_struct_layout(MirStructLayout::from(&layout));
                defs.push(def);
            }
        }
        for item in items {
            if let ItemKind::Function(func_def) = &item.kind {
                let (mir_fn, ops) = Self::lower_function_with_structs(func_def, &defs)?;
                module.add_function(mir_fn);
                for op in ops {
                    module.record_struct_op(op);
                }
            }
        }
        Ok(module)
    }

    pub fn lower_function_with_structs(
        func_def: &FunctionDef,
        struct_defs: &[&StructDef],
    ) -> Result<(MirFunction, Vec<MirStructOp>), MirLowerError> {
        Self::lower_function_with_structs_and_owner(func_def, None, struct_defs)
    }

    pub fn lower_function_with_structs_and_owner(
        func_def: &FunctionDef,
        owner_struct: Option<&str>,
        struct_defs: &[&StructDef],
    ) -> Result<(MirFunction, Vec<MirStructOp>), MirLowerError> {
        let sig = FunctionSignature {
            name: func_def.name.clone(),
            param_types: func_def
                .params
                .iter()
                .map(|p| {
                    if let Some(owner) = owner_struct {
                        if (p.is_self || p.name == "self")
                            && matches!(p.ty.kind, TypeKind::Infer | TypeKind::Unknown)
                        {
                            return MirType::Struct(owner.to_string());
                        }
                        if matches!(&p.ty.kind, TypeKind::Named(n) if n == "Self") {
                            return MirType::Struct(owner.to_string());
                        }
                    }
                    mir_type_from_ast(&p.ty)
                })
                .collect(),
            return_type: func_def
                .return_type
                .as_ref()
                .map(|rt| {
                    if let Some(owner) = owner_struct
                        && matches!(&rt.kind, TypeKind::Named(n) if n == "Self")
                    {
                        MirType::Struct(owner.to_string())
                    } else {
                        mir_type_from_ast(rt)
                    }
                })
                .unwrap_or(MirType::Unit),
            is_public: true,
            effects: func_def.effects.clone(),
        };

        let mut mir_fn = MirFunction::new(sig, crate::ast::Span::dummy());
        for (i, p) in func_def.params.iter().enumerate() {
            let p_id = mir_fn.params[i];
            mir_fn.locals[p_id.0].name = Some(p.name.clone());
        }

        let ops = {
            let mut lowering = AstLowering::new(&mut mir_fn);
            for def in struct_defs {
                lowering.register_struct_def(def);
            }
            for stmt in &func_def.body {
                lowering.lower_statement(stmt)?;
            }

            let cur_block = lowering.current_block;
            if lowering.current_fn.blocks[cur_block.0].terminator.is_none() {
                if let Some(root_cleanups) = lowering.cleanups.last().cloned() {
                    for c in root_cleanups.iter().rev() {
                        if matches!(c.kind, MirCleanupKind::Defer) {
                            lowering.lower_statement(&c.stmt)?;
                        }
                    }
                }
                let final_block = lowering.current_block;
                if lowering.current_fn.blocks[final_block.0].terminator.is_none() {
                    lowering.current_fn.blocks[final_block.0].terminator =
                        Some(Terminator::Return(None, crate::ast::Span::dummy()));
                }
            }
            lowering.struct_ops
        };

        Ok((mir_fn, ops))
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
        self.cleanups.push(Vec::new());
    }

    fn pop_scope(&mut self) -> Result<(), MirLowerError> {
        self.scopes.pop();
        if let Some(cleanups) = self.cleanups.pop() {
            for c in cleanups.into_iter().rev() {
                if matches!(c.kind, MirCleanupKind::Defer) {
                    self.lower_statement(&c.stmt)?;
                }
            }
        }
        Ok(())
    }

    fn insert_var(&mut self, name: String, local_id: LocalId) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, ScopeBinding::Local(local_id));
        }
    }

    fn insert_thunk(&mut self, name: String, thunk: LocalThunk) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, ScopeBinding::Thunk(thunk));
        }
    }

    fn lookup_var(&self, name: &str) -> Option<LocalId> {
        for scope in self.scopes.iter().rev() {
            if let Some(binding) = scope.get(name) {
                match binding {
                    ScopeBinding::Local(id) => return Some(*id),
                    ScopeBinding::Thunk(_) => return None,
                }
            }
        }
        None
    }

    fn lookup_thunk(&self, name: &str) -> Option<&LocalThunk> {
        for scope in self.scopes.iter().rev() {
            if let Some(binding) = scope.get(name) {
                match binding {
                    ScopeBinding::Thunk(thunk) => return Some(thunk),
                    ScopeBinding::Local(_) => return None,
                }
            }
        }
        None
    }

    fn get_place_struct_name(&self, place: &Place) -> Option<String> {
        match place {
            Place::Local(loc) => {
                if let MirType::Struct(name) = &self.current_fn.locals[loc.0].ty {
                    Some(name.clone())
                } else {
                    None
                }
            }
            Place::Field(base, field_idx) => {
                let base_struct = self.get_place_struct_name(base)?;
                let subs = self.struct_field_substructs.get(&base_struct)?;
                subs.get(*field_idx).cloned().flatten()
            }
            Place::Index(..) => None,
        }
    }

    fn get_place_mir_type(&self, place: &Place) -> Option<MirType> {
        match place {
            Place::Local(loc) => self.current_fn.locals.get(loc.0).map(|l| l.ty.clone()),
            Place::Index(base, _) => match self.get_place_mir_type(base)? {
                MirType::Array(inner) => Some(*inner),
                MirType::Tuple(elems) => elems.first().cloned(),
                _ => None,
            },
            Place::Field(base, field_idx) => {
                if let Some(base_struct) = self.get_place_struct_name(base) {
                    if let Some(fty) = self
                        .struct_field_types
                        .get(&base_struct)
                        .and_then(|v| v.get(*field_idx))
                    {
                        return Some(fty.clone());
                    }
                    if let Some(layout) = self.struct_layouts.get(&base_struct)
                        && let Some(f_desc) = layout.fields.get(*field_idx)
                    {
                        return Some(match f_desc.tag {
                            CanonicalTypeTag::Bool => MirType::Bool,
                            CanonicalTypeTag::Int64
                            | CanonicalTypeTag::UInt64
                            | CanonicalTypeTag::DateTimeUtc => MirType::Int,
                            CanonicalTypeTag::Float64 => MirType::Float,
                            CanonicalTypeTag::StringSlice => MirType::String,
                            _ => MirType::Int,
                        });
                    }
                }
                match self.get_place_mir_type(base)? {
                    MirType::Tuple(elems) => elems.get(*field_idx).cloned(),
                    _ => None,
                }
            }
        }
    }

    fn peek_place_expected_type(&self, expr: &Expr) -> Option<MirType> {
        match &expr.kind {
            ExprKind::Paren(inner) => self.peek_place_expected_type(inner),
            ExprKind::Identifier(name) => {
                let loc = self.lookup_var(name)?;
                self.current_fn.locals.get(loc.0).map(|l| l.ty.clone())
            }
            ExprKind::FieldAccess { object, field } if field != "len" => {
                let base_ty = self.peek_place_expected_type(object)?;
                if let MirType::Struct(struct_name) = base_ty
                    && let Some(names) = self.struct_fields.get(&struct_name)
                    && let Some(pos) = names.iter().position(|n| n == field)
                    && let Some(ftys) = self.struct_field_types.get(&struct_name)
                {
                    return ftys.get(pos).cloned();
                }
                None
            }
            _ => None,
        }
    }

    fn resolve_field_idx(&self, base_place: &Place, field: &str) -> Option<usize> {
        // 1. If the exact struct name of `base_place` is known and its schema is registered,
        //    resolve strictly against that schema (never fall through to unrelated heuristics).
        if let Some(struct_name) = self.get_place_struct_name(base_place) {
            let mut has_schema = false;
            if let Some(names) = self.struct_fields.get(&struct_name) {
                has_schema = true;
                if let Some(pos) = names.iter().position(|n| n == field) {
                    return Some(pos);
                }
            }
            if let Some(layout) = self.struct_layouts.get(&struct_name) {
                has_schema = true;
                if let Some(f_desc) = layout.field_by_name(field) {
                    return Some(f_desc.ordinal as usize);
                }
            }
            if has_schema {
                return None;
            }
        }

        // 2. For direct locals only (never unwrap nested Field/Index back to a parent struct),
        //    check per-local field names and declared local type.
        let base_ty: Option<&MirType> = match base_place {
            Place::Local(loc) => {
                if let Some(names) = self.local_field_names.get(loc) {
                    return names.iter().position(|n| n == field);
                }
                self.current_fn.locals.get(loc.0).map(|l| &l.ty)
            }
            Place::Index(inner, _) => {
                if let Place::Local(loc) = &**inner {
                    match self.current_fn.locals.get(loc.0).map(|l| &l.ty) {
                        Some(MirType::Array(elem)) => Some(elem.as_ref()),
                        _ => None,
                    }
                } else {
                    None
                }
            }
            Place::Field(_, _) => None,
        };

        // 3. Tuple numeric/underscore field access: only valid on Tuple (or Unknown) base.
        if matches!(base_ty, Some(MirType::Tuple(_) | MirType::Unknown)) {
            if let Ok(idx) = field.parse::<usize>() {
                if let Some(MirType::Tuple(elems)) = base_ty {
                    return (idx < elems.len()).then_some(idx);
                }
                return Some(idx);
            }
            if let Some(stripped) = field.strip_prefix('_') {
                if let Ok(idx) = stripped.parse::<usize>() {
                    if let Some(MirType::Tuple(elems)) = base_ty {
                        return (idx < elems.len()).then_some(idx);
                    }
                    return Some(idx);
                }
            }
        }

        // 4. Legacy coordinate fallback ONLY for unregistered Struct locals (never scalars/Enums/constants).
        if matches!(base_ty, Some(MirType::Struct(_))) {
            return match field {
                "x" => Some(0),
                "y" => Some(1),
                "z" => Some(2),
                "w" => Some(3),
                _ => None,
            };
        }

        None
    }

    pub fn lower_function(func_def: &FunctionDef) -> Result<MirFunction, MirLowerError> {
        let sig = FunctionSignature {
            name: func_def.name.clone(),
            param_types: func_def
                .params
                .iter()
                .map(|p| mir_type_from_ast(&p.ty))
                .collect(),
            return_type: func_def
                .return_type
                .as_ref()
                .map(mir_type_from_ast)
                .unwrap_or(MirType::Unit),
            is_public: true,
            effects: func_def.effects.clone(),
        };

        // Create MirFunction with default entry block
        let mut mir_fn = MirFunction::new(sig, crate::ast::Span::dummy());

        // Update parameter names if defined in func_def
        for (i, p) in func_def.params.iter().enumerate() {
            let p_id = mir_fn.params[i];
            mir_fn.locals[p_id.0].name = Some(p.name.clone());
        }

        {
            let mut lowering = AstLowering::new(&mut mir_fn);
            for stmt in &func_def.body {
                lowering.lower_statement(stmt)?;
            }

            // Ensure current block has a terminator
            let cur_block = lowering.current_block;
            if lowering.current_fn.blocks[cur_block.0].terminator.is_none() {
                if let Some(root_cleanups) = lowering.cleanups.last().cloned() {
                    for c in root_cleanups.iter().rev() {
                        if matches!(c.kind, MirCleanupKind::Defer) {
                            lowering.lower_statement(&c.stmt)?;
                        }
                    }
                }
                let final_block = lowering.current_block;
                if lowering.current_fn.blocks[final_block.0].terminator.is_none() {
                    lowering.current_fn.blocks[final_block.0].terminator =
                        Some(Terminator::Return(None, crate::ast::Span::dummy()));
                }
            }
        }

        Ok(mir_fn)
    }

    pub fn lower_script(name: &str, stmts: &[Stmt]) -> Result<MirFunction, MirLowerError> {
        let sig = FunctionSignature {
            name: name.into(),
            param_types: vec![],
            return_type: MirType::Unit,
            is_public: true,
            effects: None,
        };

        let mut mir_fn = MirFunction::new(sig, crate::ast::Span::dummy());
        {
            let mut lowering = AstLowering::new(&mut mir_fn);
            for stmt in stmts {
                lowering.lower_statement(stmt)?;
            }

            let cur_block = lowering.current_block;
            if lowering.current_fn.blocks[cur_block.0].terminator.is_none() {
                if let Some(root_cleanups) = lowering.cleanups.last().cloned() {
                    for c in root_cleanups.iter().rev() {
                        if matches!(c.kind, MirCleanupKind::Defer) {
                            lowering.lower_statement(&c.stmt)?;
                        }
                    }
                }
                let final_block = lowering.current_block;
                if lowering.current_fn.blocks[final_block.0].terminator.is_none() {
                    lowering.current_fn.blocks[final_block.0].terminator =
                        Some(Terminator::Return(None, crate::ast::Span::dummy()));
                }
            }
        }

        Ok(mir_fn)
    }

    pub fn lower_statement(&mut self, stmt: &Stmt) -> Result<(), MirLowerError> {
        if self.current_fn.blocks[self.current_block.0].terminator.is_some() {
            let dead_bb = self.current_fn.new_block();
            self.current_fn.blocks[dead_bb.0].terminator = Some(Terminator::Unreachable(stmt.span));
            self.current_block = dead_bb;
        }

        match &stmt.kind {
            StmtKind::Local { bindings, initializer } => {
                if let Some(inits) = initializer {
                    let has_effect = inits.iter().any(|e| matches!(&e.kind, ExprKind::EffectBlock(_)));
                    if has_effect {
                        if inits.len() != 1 || bindings.len() != 1 {
                            return Err(MirLowerError::UnlowerableExpression(
                                "EffectBlock initializer must be a single binding to a local variable".into(),
                            ));
                        }
                        if let ExprKind::EffectBlock(stmts) = &inits[0].kind {
                            return self.lower_effect_block(&bindings[0].name, stmts, inits[0].span, bindings[0].span);
                        }
                    }
                }

                let init_operands = if let Some(inits) = initializer {
                    let mut ops = Vec::with_capacity(inits.len());
                    for (idx, expr) in inits.iter().enumerate() {
                        let expected_ty = bindings
                            .get(idx)
                            .and_then(|b| b.type_annotation.as_ref())
                            .map(mir_type_from_ast);
                        ops.push(self.lower_expr_with_expected(expr, expected_ty.as_ref())?);
                    }
                    ops
                } else {
                    Vec::new()
                };

                for (idx, b) in bindings.iter().enumerate() {
                    let has_init = initializer.as_ref().is_some_and(|inits| idx < inits.len());
                    let init_op = init_operands.get(idx).cloned().unwrap_or(Operand::Constant(MirConstant::Int(0)));
                    let inferred_ty = if has_init {
                        self.operand_definite_mir_type(&init_op)
                    } else {
                        MirType::Int
                    };
                    let ty = b
                        .type_annotation
                        .as_ref()
                        .map(mir_type_from_ast)
                        .unwrap_or(inferred_ty);
                    // Part A (`sieve.lust`): when an explicitly annotated `Array<T>` local is initialized
                    // with an empty array literal `[]`, specialize the temporary local and its empty
                    // aggregate from `Array(Unknown)` directly to `Array(T)`.
                    if let MirType::Array(elem_ty) = &ty
                        && **elem_ty != MirType::Unknown
                        && let Operand::Move(Place::Local(tmp)) | Operand::Copy(Place::Local(tmp)) = &init_op
                        && self.current_fn.locals[tmp.0].ty == MirType::Array(Box::new(MirType::Unknown))
                    {
                        self.current_fn.locals[tmp.0].ty = ty.clone();
                        for prev_stmt in self.current_fn.blocks[self.current_block.0].statements.iter_mut().rev() {
                            if let Statement::Assign(
                                Place::Local(dst),
                                Rvalue::Aggregate(agg_ty, elems),
                                _,
                            ) = prev_stmt
                                && *dst == *tmp
                                && elems.is_empty()
                            {
                                *agg_ty = ty.clone();
                                break;
                            }
                        }
                    }
                    let is_known = b.type_annotation.is_some()
                        || (ty == MirType::Int
                            && (!has_init || matches!(init_op, Operand::Constant(MirConstant::Int(_)))));
                    let local_id = self.current_fn.add_local(
                        Some(b.name.clone()),
                        ty,
                        true,
                        b.span,
                    );
                    if is_known {
                        self.known_typed_locals.insert(local_id);
                    }
                    self.insert_var(b.name.clone(), local_id);

                    if let Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src)) = &init_op {
                        if let Some(names) = self.local_field_names.get(src).cloned() {
                            self.local_field_names.insert(local_id, names);
                        }
                    }

                    self.emit_stmt(Statement::StorageLive(local_id, b.span));
                    self.emit_stmt(Statement::Assign(Place::Local(local_id), Rvalue::Use(init_op), b.span));
                }
            }
            StmtKind::Assign { targets, values } => {
                for (target, val) in targets.iter().zip(values.iter()) {
                    if let ExprKind::Identifier(name) = &target.kind {
                        if self.lookup_thunk(name).is_some() {
                            return Err(MirLowerError::UnlowerableExpression(format!(
                                "Effect thunk '{name}' cannot be reassigned or mutated"
                            )));
                        }
                    }
                    if let ExprKind::Identifier(name) = &val.kind {
                        if self.lookup_thunk(name).is_some() {
                            return Err(MirLowerError::UnlowerableExpression(format!(
                                "Effect thunk '{name}' cannot be aliased or copied"
                            )));
                        }
                    }
                    let expected_ty = self.peek_place_expected_type(target);
                    if targets.len() == 1
                        && let ExprKind::Identifier(name) = &target.kind
                        && let Some(dst_local) = self.lookup_var(name)
                    {
                        let mut inner_val = val;
                        while let ExprKind::Paren(p) = &inner_val.kind {
                            inner_val = p;
                        }
                        if let ExprKind::Binary {
                            op: BinaryOp::Concat,
                            left,
                            right,
                        } = &inner_val.kind
                        {
                            let lhs_op = self.lower_expr(left)?;
                            let rhs_op = self.lower_expr(right)?;
                            self.emit_stmt(Statement::Assign(
                                Place::Local(dst_local),
                                Rvalue::BinaryOp(MirBinaryOp::Concat, lhs_op, rhs_op),
                                stmt.span,
                            ));
                            continue;
                        }
                    }
                    let rval_op = self.lower_expr_with_expected(val, expected_ty.as_ref())?;
                    let place = self.lower_place(target)?
                        .ok_or_else(|| MirLowerError::UnsupportedLvalue(format!("{:?}", target.kind)))?;
                    self.emit_stmt(Statement::Assign(place, Rvalue::Use(rval_op), stmt.span));
                }
            }
            StmtKind::CompoundAssign { target, op, value } => {
                if let ExprKind::Identifier(name) = &target.kind {
                    if self.lookup_thunk(name).is_some() {
                        return Err(MirLowerError::UnlowerableExpression(format!(
                            "Effect thunk '{name}' cannot be reassigned or mutated"
                        )));
                    }
                }
                let mir_op = mir_binop_from_ast(*op)?;
                let place = self.lower_place(target)?
                    .ok_or_else(|| MirLowerError::UnsupportedLvalue(format!("{:?}", target.kind)))?;
                let lhs_op = Operand::Copy(place.clone());
                let rhs_op = self.lower_expr(value)?;
                self.emit_stmt(Statement::Assign(
                    place,
                    Rvalue::BinaryOp(mir_op, lhs_op, rhs_op),
                    stmt.span,
                ));
            }
            StmtKind::Expr(expr) => {
                self.lower_expr(expr)?;
            }
            StmtKind::If {
                condition,
                then_block,
                elseif_branches,
                else_block,
            } => {
                self.lower_if(condition, then_block, elseif_branches, else_block.as_deref(), stmt.span)?;
            }
            StmtKind::While { condition, body } => {
                let header = self.current_fn.new_block();
                let body_block = self.current_fn.new_block();
                let exit_block = self.current_fn.new_block();

                // Branch from current block to header
                self.terminate_block(Terminator::Goto(header, stmt.span));

                // Header: evaluate condition
                self.current_block = header;
                self.emit_stmt(Statement::ResourceCheck(ResourceKind::Gas, 1, stmt.span));
                let (cond_op, body_bindings) = self.lower_condition_with_bindings(condition, stmt.span)?;
                self.terminate_block(Terminator::Cond {
                    cond: cond_op,
                    then_block: body_block,
                    else_block: exit_block,
                    span: stmt.span,
                });

                // Body block
                self.current_block = body_block;
                self.loop_stack.push((header, exit_block));
                self.push_scope();
                for (name, ty, place, unwrap_option) in body_bindings {
                    let binding_local = self.current_fn.add_local(Some(name.clone()), ty.clone(), false, stmt.span);
                    if ty != MirType::Unknown {
                        self.known_typed_locals.insert(binding_local);
                    }
                    self.emit_stmt(Statement::StorageLive(binding_local, stmt.span));
                    let rval = if unwrap_option {
                        Rvalue::Cast(Operand::Copy(place), ty)
                    } else {
                        Rvalue::Use(Operand::Copy(place))
                    };
                    self.emit_stmt(Statement::Assign(
                        Place::Local(binding_local),
                        rval,
                        stmt.span,
                    ));
                    self.insert_var(name, binding_local);
                }
                for s in body {
                    self.lower_statement(s)?;
                }
                self.pop_scope()?;
                self.loop_stack.pop();

                // Loop back to header
                if self.current_fn.blocks[self.current_block.0].terminator.is_none() {
                    self.terminate_block(Terminator::Goto(header, stmt.span));
                }

                self.current_block = exit_block;
            }
            StmtKind::Return(exprs) => {
                if let Some((res_local, cont_bb)) = self.thunk_return_target {
                    if exprs.len() != 1 {
                        return Err(MirLowerError::UnlowerableExpression(
                            "Effect thunk return must return exactly one value".into(),
                        ));
                    }
                    let op = self.lower_expr(&exprs[0])?;
                    self.emit_stmt(Statement::Assign(
                        Place::Local(res_local),
                        Rvalue::Use(op),
                        stmt.span,
                    ));
                    self.terminate_block(Terminator::Goto(cont_bb, stmt.span));
                } else {
                    let ret_op = if let Some(first) = exprs.first() {
                        let expected_ret = self.current_fn.signature.return_type.clone();
                        let op = self.lower_expr_with_expected(first, Some(&expected_ret))?;
                        let ret_ty = if matches!(
                            self.current_fn.signature.return_type,
                            MirType::Unit | MirType::Unknown
                        ) {
                            MirType::Int
                        } else {
                            self.current_fn.signature.return_type.clone()
                        };
                        let temp = self.current_fn.add_local(None, ret_ty, true, stmt.span);
                        self.emit_stmt(Statement::StorageLive(temp, stmt.span));
                        self.emit_stmt(Statement::Assign(Place::Local(temp), Rvalue::Use(op), stmt.span));
                        Some(Operand::Copy(Place::Local(temp)))
                    } else {
                        None
                    };

                    for scope_cleanups in self.cleanups.clone().iter().rev() {
                        for c in scope_cleanups.iter().rev() {
                            if matches!(c.kind, MirCleanupKind::Defer) {
                                self.lower_statement(&c.stmt)?;
                            }
                        }
                    }

                    self.terminate_block(Terminator::Return(ret_op, stmt.span));
                }
            }
            StmtKind::ForNumeric {
                variable,
                start,
                end,
                step,
                body,
            } => {
                let start_op = self.lower_expr(start)?;
                let end_op = self.lower_expr(end)?;
                let step_op = if let Some(step_expr) = step {
                    self.lower_expr(step_expr)?
                } else {
                    Operand::Constant(MirConstant::Int(1))
                };

                let static_step = match step {
                    Some(s) => const_int_from_expr(s),
                    None => Some(1),
                };

                let end_local = self.current_fn.add_local(None, MirType::Int, false, stmt.span);
                self.emit_stmt(Statement::StorageLive(end_local, stmt.span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(end_local),
                    Rvalue::Use(end_op),
                    stmt.span,
                ));

                let step_local = self.current_fn.add_local(None, MirType::Int, false, stmt.span);
                self.emit_stmt(Statement::StorageLive(step_local, stmt.span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(step_local),
                    Rvalue::Use(step_op),
                    stmt.span,
                ));

                let var_local = self.current_fn.add_local(
                    Some(variable.clone()),
                    MirType::Int,
                    true,
                    stmt.span,
                );
                self.known_typed_locals.insert(var_local);
                self.emit_stmt(Statement::StorageLive(var_local, stmt.span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(var_local),
                    Rvalue::Use(start_op),
                    stmt.span,
                ));

                let header_bb = self.current_fn.new_block();
                let body_bb = self.current_fn.new_block();
                let step_bb = self.current_fn.new_block();
                let exit_bb = self.current_fn.new_block();

                self.terminate_block(Terminator::Goto(header_bb, stmt.span));

                // Header block
                self.current_block = header_bb;
                self.emit_stmt(Statement::ResourceCheck(ResourceKind::Gas, 1, stmt.span));
                let cond_temp = self.current_fn.add_local(None, MirType::Bool, true, stmt.span);
                self.emit_stmt(Statement::StorageLive(cond_temp, stmt.span));

                let cond_rval = match static_step {
                    Some(val) if val < 0 => Rvalue::BinaryOp(
                        MirBinaryOp::GtEq,
                        Operand::Copy(Place::Local(var_local)),
                        Operand::Copy(Place::Local(end_local)),
                    ),
                    Some(_) => Rvalue::BinaryOp(
                        MirBinaryOp::LtEq,
                        Operand::Copy(Place::Local(var_local)),
                        Operand::Copy(Place::Local(end_local)),
                    ),
                    None => {
                        let is_neg_temp = self.current_fn.add_local(None, MirType::Bool, true, stmt.span);
                        self.emit_stmt(Statement::StorageLive(is_neg_temp, stmt.span));
                        self.emit_stmt(Statement::Assign(
                            Place::Local(is_neg_temp),
                            Rvalue::BinaryOp(
                                MirBinaryOp::Lt,
                                Operand::Copy(Place::Local(step_local)),
                                Operand::Constant(MirConstant::Int(0)),
                            ),
                            stmt.span,
                        ));
                        let ge_temp = self.current_fn.add_local(None, MirType::Bool, true, stmt.span);
                        self.emit_stmt(Statement::StorageLive(ge_temp, stmt.span));
                        self.emit_stmt(Statement::Assign(
                            Place::Local(ge_temp),
                            Rvalue::BinaryOp(
                                MirBinaryOp::GtEq,
                                Operand::Copy(Place::Local(var_local)),
                                Operand::Copy(Place::Local(end_local)),
                            ),
                            stmt.span,
                        ));
                        let le_temp = self.current_fn.add_local(None, MirType::Bool, true, stmt.span);
                        self.emit_stmt(Statement::StorageLive(le_temp, stmt.span));
                        self.emit_stmt(Statement::Assign(
                            Place::Local(le_temp),
                            Rvalue::BinaryOp(
                                MirBinaryOp::LtEq,
                                Operand::Copy(Place::Local(var_local)),
                                Operand::Copy(Place::Local(end_local)),
                            ),
                            stmt.span,
                        ));
                        let neg_and_ge = self.current_fn.add_local(None, MirType::Bool, true, stmt.span);
                        self.emit_stmt(Statement::StorageLive(neg_and_ge, stmt.span));
                        self.emit_stmt(Statement::Assign(
                            Place::Local(neg_and_ge),
                            Rvalue::BinaryOp(
                                MirBinaryOp::And,
                                Operand::Copy(Place::Local(is_neg_temp)),
                                Operand::Copy(Place::Local(ge_temp)),
                            ),
                            stmt.span,
                        ));
                        let not_neg = self.current_fn.add_local(None, MirType::Bool, true, stmt.span);
                        self.emit_stmt(Statement::StorageLive(not_neg, stmt.span));
                        self.emit_stmt(Statement::Assign(
                            Place::Local(not_neg),
                            Rvalue::UnaryOp(MirUnaryOp::Not, Operand::Copy(Place::Local(is_neg_temp))),
                            stmt.span,
                        ));
                        let pos_and_le = self.current_fn.add_local(None, MirType::Bool, true, stmt.span);
                        self.emit_stmt(Statement::StorageLive(pos_and_le, stmt.span));
                        self.emit_stmt(Statement::Assign(
                            Place::Local(pos_and_le),
                            Rvalue::BinaryOp(
                                MirBinaryOp::And,
                                Operand::Copy(Place::Local(not_neg)),
                                Operand::Copy(Place::Local(le_temp)),
                            ),
                            stmt.span,
                        ));
                        Rvalue::BinaryOp(
                            MirBinaryOp::Or,
                            Operand::Copy(Place::Local(neg_and_ge)),
                            Operand::Copy(Place::Local(pos_and_le)),
                        )
                    }
                };

                self.emit_stmt(Statement::Assign(Place::Local(cond_temp), cond_rval, stmt.span));
                self.terminate_block(Terminator::Cond {
                    cond: Operand::Move(Place::Local(cond_temp)),
                    then_block: body_bb,
                    else_block: exit_bb,
                    span: stmt.span,
                });

                // Body block
                self.current_block = body_bb;
                self.push_scope();
                self.insert_var(variable.clone(), var_local);
                self.loop_stack.push((step_bb, exit_bb));
                for s in body {
                    self.lower_statement(s)?;
                }
                self.loop_stack.pop();
                self.pop_scope()?;
                if self.current_fn.blocks[self.current_block.0].terminator.is_none() {
                    self.terminate_block(Terminator::Goto(step_bb, stmt.span));
                }

                // Step block
                self.current_block = step_bb;
                let next_var_temp = self.current_fn.add_local(None, MirType::Int, true, stmt.span);
                self.emit_stmt(Statement::StorageLive(next_var_temp, stmt.span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(next_var_temp),
                    Rvalue::BinaryOp(
                        MirBinaryOp::Add,
                        Operand::Copy(Place::Local(var_local)),
                        Operand::Copy(Place::Local(step_local)),
                    ),
                    stmt.span,
                ));
                self.emit_stmt(Statement::Assign(
                    Place::Local(var_local),
                    Rvalue::Use(Operand::Move(Place::Local(next_var_temp))),
                    stmt.span,
                ));
                self.terminate_block(Terminator::Goto(header_bb, stmt.span));

                self.current_block = exit_bb;
            }
            StmtKind::ForIn { variables, iterator, body } => {
                let is_ipairs = matches!(
                    &iterator.kind,
                    ExprKind::Call { callee, args, .. }
                        if matches!(&callee.kind, ExprKind::Identifier(name) if name == "ipairs")
                            && args.len() == 1
                            && variables.len() >= 2
                );

                if is_ipairs {
                    let args = match &iterator.kind {
                        ExprKind::Call { args, .. } => args,
                        _ => unreachable!(),
                    };
                    let arr_op = self.lower_expr(&args[0])?;
                    let arr_local = self.current_fn.add_local(
                        None,
                        MirType::Array(Box::new(MirType::Int)),
                        true,
                        stmt.span,
                    );
                    self.emit_stmt(Statement::StorageLive(arr_local, stmt.span));
                    self.emit_stmt(Statement::Assign(Place::Local(arr_local), Rvalue::Use(arr_op), stmt.span));

                    let len_local = self.current_fn.add_local(None, MirType::Int, true, stmt.span);
                    self.emit_stmt(Statement::StorageLive(len_local, stmt.span));
                    self.emit_stmt(Statement::Assign(
                        Place::Local(len_local),
                        Rvalue::UnaryOp(MirUnaryOp::Len, Operand::Copy(Place::Local(arr_local))),
                        stmt.span,
                    ));

                    let idx_local = self.current_fn.add_local(None, MirType::Int, true, stmt.span);
                    self.emit_stmt(Statement::StorageLive(idx_local, stmt.span));
                    self.emit_stmt(Statement::Assign(
                        Place::Local(idx_local),
                        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
                        stmt.span,
                    ));

                    let header_bb = self.current_fn.new_block();
                    let body_bb = self.current_fn.new_block();
                    let step_bb = self.current_fn.new_block();
                    let exit_bb = self.current_fn.new_block();

                    self.terminate_block(Terminator::Goto(header_bb, stmt.span));

                    // header_bb
                    self.current_block = header_bb;
                    self.emit_stmt(Statement::ResourceCheck(ResourceKind::Gas, 1, stmt.span));
                    let cond_temp = self.current_fn.add_local(None, MirType::Bool, true, stmt.span);
                    self.emit_stmt(Statement::StorageLive(cond_temp, stmt.span));
                    self.emit_stmt(Statement::Assign(
                        Place::Local(cond_temp),
                        Rvalue::BinaryOp(
                            MirBinaryOp::Lt,
                            Operand::Copy(Place::Local(idx_local)),
                            Operand::Copy(Place::Local(len_local)),
                        ),
                        stmt.span,
                    ));
                    self.terminate_block(Terminator::Cond {
                        cond: Operand::Move(Place::Local(cond_temp)),
                        then_block: body_bb,
                        else_block: exit_bb,
                        span: stmt.span,
                    });

                    // body_bb
                    self.current_block = body_bb;
                    self.push_scope();

                    let var0_local = self.current_fn.add_local(Some(variables[0].clone()), MirType::Int, true, stmt.span);
                    self.emit_stmt(Statement::StorageLive(var0_local, stmt.span));
                    self.emit_stmt(Statement::Assign(
                        Place::Local(var0_local),
                        Rvalue::Use(Operand::Copy(Place::Local(idx_local))),
                        stmt.span,
                    ));
                    self.insert_var(variables[0].clone(), var0_local);

                    let var1_local = self.current_fn.add_local(Some(variables[1].clone()), MirType::Int, true, stmt.span);
                    self.emit_stmt(Statement::StorageLive(var1_local, stmt.span));
                    self.emit_stmt(Statement::Assign(
                        Place::Local(var1_local),
                        Rvalue::Use(Operand::Copy(Place::Index(Box::new(Place::Local(arr_local)), idx_local))),
                        stmt.span,
                    ));
                    self.insert_var(variables[1].clone(), var1_local);

                    self.loop_stack.push((step_bb, exit_bb));
                    for s in body {
                        self.lower_statement(s)?;
                    }
                    self.loop_stack.pop();
                    self.pop_scope()?;
                    if self.current_fn.blocks[self.current_block.0].terminator.is_none() {
                        self.terminate_block(Terminator::Goto(step_bb, stmt.span));
                    }

                    // step_bb
                    self.current_block = step_bb;
                    let next_idx = self.current_fn.add_local(None, MirType::Int, true, stmt.span);
                    self.emit_stmt(Statement::StorageLive(next_idx, stmt.span));
                    self.emit_stmt(Statement::Assign(
                        Place::Local(next_idx),
                        Rvalue::BinaryOp(
                            MirBinaryOp::Add,
                            Operand::Copy(Place::Local(idx_local)),
                            Operand::Constant(MirConstant::Int(1)),
                        ),
                        stmt.span,
                    ));
                    self.emit_stmt(Statement::Assign(
                        Place::Local(idx_local),
                        Rvalue::Use(Operand::Move(Place::Local(next_idx))),
                        stmt.span,
                    ));
                    self.terminate_block(Terminator::Goto(header_bb, stmt.span));

                    self.current_block = exit_bb;
                } else if let ExprKind::Range { start, end, inclusive } = &iterator.kind {
                    let start_op = self.lower_expr(start)?;
                    let end_op = self.lower_expr(end)?;

                    let end_local = self.current_fn.add_local(None, MirType::Int, false, stmt.span);
                    self.emit_stmt(Statement::StorageLive(end_local, stmt.span));
                    self.emit_stmt(Statement::Assign(Place::Local(end_local), Rvalue::Use(end_op), stmt.span));

                    let var_name = variables.first().cloned().unwrap_or_else(|| "i".into());
                    let var_local = self.current_fn.add_local(Some(var_name.clone()), MirType::Int, true, stmt.span);
                    self.emit_stmt(Statement::StorageLive(var_local, stmt.span));
                    self.emit_stmt(Statement::Assign(Place::Local(var_local), Rvalue::Use(start_op), stmt.span));

                    let header_bb = self.current_fn.new_block();
                    let body_bb = self.current_fn.new_block();
                    let step_bb = self.current_fn.new_block();
                    let exit_bb = self.current_fn.new_block();

                    self.terminate_block(Terminator::Goto(header_bb, stmt.span));

                    // header_bb
                    self.current_block = header_bb;
                    self.emit_stmt(Statement::ResourceCheck(ResourceKind::Gas, 1, stmt.span));
                    let cond_temp = self.current_fn.add_local(None, MirType::Bool, true, stmt.span);
                    self.emit_stmt(Statement::StorageLive(cond_temp, stmt.span));
                    let cmp_op = if *inclusive { MirBinaryOp::LtEq } else { MirBinaryOp::Lt };
                    self.emit_stmt(Statement::Assign(
                        Place::Local(cond_temp),
                        Rvalue::BinaryOp(
                            cmp_op,
                            Operand::Copy(Place::Local(var_local)),
                            Operand::Copy(Place::Local(end_local)),
                        ),
                        stmt.span,
                    ));
                    self.terminate_block(Terminator::Cond {
                        cond: Operand::Move(Place::Local(cond_temp)),
                        then_block: body_bb,
                        else_block: exit_bb,
                        span: stmt.span,
                    });

                    // body_bb
                    self.current_block = body_bb;
                    self.push_scope();
                    self.insert_var(var_name, var_local);
                    self.loop_stack.push((step_bb, exit_bb));
                    for s in body {
                        self.lower_statement(s)?;
                    }
                    self.loop_stack.pop();
                    self.pop_scope()?;
                    if self.current_fn.blocks[self.current_block.0].terminator.is_none() {
                        self.terminate_block(Terminator::Goto(step_bb, stmt.span));
                    }

                    // step_bb
                    self.current_block = step_bb;
                    let next_var = self.current_fn.add_local(None, MirType::Int, true, stmt.span);
                    self.emit_stmt(Statement::StorageLive(next_var, stmt.span));
                    self.emit_stmt(Statement::Assign(
                        Place::Local(next_var),
                        Rvalue::BinaryOp(
                            MirBinaryOp::Add,
                            Operand::Copy(Place::Local(var_local)),
                            Operand::Constant(MirConstant::Int(1)),
                        ),
                        stmt.span,
                    ));
                    self.emit_stmt(Statement::Assign(
                        Place::Local(var_local),
                        Rvalue::Use(Operand::Move(Place::Local(next_var))),
                        stmt.span,
                    ));
                    self.terminate_block(Terminator::Goto(header_bb, stmt.span));

                    self.current_block = exit_bb;
                } else {
                    let arr_op = self.lower_expr(iterator)?;
                    let arr_local = self.current_fn.add_local(
                        None,
                        MirType::Array(Box::new(MirType::Int)),
                        true,
                        stmt.span,
                    );
                    self.emit_stmt(Statement::StorageLive(arr_local, stmt.span));
                    self.emit_stmt(Statement::Assign(Place::Local(arr_local), Rvalue::Use(arr_op), stmt.span));

                    let len_local = self.current_fn.add_local(None, MirType::Int, true, stmt.span);
                    self.emit_stmt(Statement::StorageLive(len_local, stmt.span));
                    self.emit_stmt(Statement::Assign(
                        Place::Local(len_local),
                        Rvalue::UnaryOp(MirUnaryOp::Len, Operand::Copy(Place::Local(arr_local))),
                        stmt.span,
                    ));

                    let idx_local = self.current_fn.add_local(None, MirType::Int, true, stmt.span);
                    self.emit_stmt(Statement::StorageLive(idx_local, stmt.span));
                    self.emit_stmt(Statement::Assign(
                        Place::Local(idx_local),
                        Rvalue::Use(Operand::Constant(MirConstant::Int(0))),
                        stmt.span,
                    ));

                    let header_bb = self.current_fn.new_block();
                    let body_bb = self.current_fn.new_block();
                    let step_bb = self.current_fn.new_block();
                    let exit_bb = self.current_fn.new_block();

                    self.terminate_block(Terminator::Goto(header_bb, stmt.span));

                    // header_bb
                    self.current_block = header_bb;
                    self.emit_stmt(Statement::ResourceCheck(ResourceKind::Gas, 1, stmt.span));
                    let cond_temp = self.current_fn.add_local(None, MirType::Bool, true, stmt.span);
                    self.emit_stmt(Statement::StorageLive(cond_temp, stmt.span));
                    self.emit_stmt(Statement::Assign(
                        Place::Local(cond_temp),
                        Rvalue::BinaryOp(
                            MirBinaryOp::Lt,
                            Operand::Copy(Place::Local(idx_local)),
                            Operand::Copy(Place::Local(len_local)),
                        ),
                        stmt.span,
                    ));
                    self.terminate_block(Terminator::Cond {
                        cond: Operand::Move(Place::Local(cond_temp)),
                        then_block: body_bb,
                        else_block: exit_bb,
                        span: stmt.span,
                    });

                    // body_bb
                    self.current_block = body_bb;
                    self.push_scope();

                    if variables.len() == 1 {
                        let item_var = &variables[0];
                        let item_local = self.current_fn.add_local(Some(item_var.clone()), MirType::Int, true, stmt.span);
                        self.emit_stmt(Statement::StorageLive(item_local, stmt.span));
                        self.emit_stmt(Statement::Assign(
                            Place::Local(item_local),
                            Rvalue::Use(Operand::Copy(Place::Index(Box::new(Place::Local(arr_local)), idx_local))),
                            stmt.span,
                        ));
                        self.insert_var(item_var.clone(), item_local);
                    } else if variables.len() >= 2 {
                        let var0_local = self.current_fn.add_local(Some(variables[0].clone()), MirType::Int, true, stmt.span);
                        self.emit_stmt(Statement::StorageLive(var0_local, stmt.span));
                        self.emit_stmt(Statement::Assign(
                            Place::Local(var0_local),
                            Rvalue::Use(Operand::Copy(Place::Local(idx_local))),
                            stmt.span,
                        ));
                        self.insert_var(variables[0].clone(), var0_local);

                        let var1_local = self.current_fn.add_local(Some(variables[1].clone()), MirType::Int, true, stmt.span);
                        self.emit_stmt(Statement::StorageLive(var1_local, stmt.span));
                        self.emit_stmt(Statement::Assign(
                            Place::Local(var1_local),
                            Rvalue::Use(Operand::Copy(Place::Index(Box::new(Place::Local(arr_local)), idx_local))),
                            stmt.span,
                        ));
                        self.insert_var(variables[1].clone(), var1_local);
                    }

                    self.loop_stack.push((step_bb, exit_bb));
                    for s in body {
                        self.lower_statement(s)?;
                    }
                    self.loop_stack.pop();
                    self.pop_scope()?;
                    if self.current_fn.blocks[self.current_block.0].terminator.is_none() {
                        self.terminate_block(Terminator::Goto(step_bb, stmt.span));
                    }

                    // step_bb
                    self.current_block = step_bb;
                    let next_idx = self.current_fn.add_local(None, MirType::Int, true, stmt.span);
                    self.emit_stmt(Statement::StorageLive(next_idx, stmt.span));
                    self.emit_stmt(Statement::Assign(
                        Place::Local(next_idx),
                        Rvalue::BinaryOp(
                            MirBinaryOp::Add,
                            Operand::Copy(Place::Local(idx_local)),
                            Operand::Constant(MirConstant::Int(1)),
                        ),
                        stmt.span,
                    ));
                    self.emit_stmt(Statement::Assign(
                        Place::Local(idx_local),
                        Rvalue::Use(Operand::Move(Place::Local(next_idx))),
                        stmt.span,
                    ));
                    self.terminate_block(Terminator::Goto(header_bb, stmt.span));

                    self.current_block = exit_bb;
                }
            }
            StmtKind::Break => {
                if let Some((_, exit_bb)) = self.loop_stack.last().cloned() {
                    self.terminate_block(Terminator::Goto(exit_bb, stmt.span));
                }
            }
            StmtKind::Continue => {
                if let Some((cont_bb, _)) = self.loop_stack.last().cloned() {
                    self.terminate_block(Terminator::Goto(cont_bb, stmt.span));
                }
            }
            StmtKind::Block(stmts) => {
                self.push_scope();
                for s in stmts {
                    self.lower_statement(s)?;
                }
                self.pop_scope()?;
            }
            StmtKind::Defer(s) => {
                if let Some(cleanups) = self.cleanups.last_mut() {
                    cleanups.push(MirCleanup {
                        kind: MirCleanupKind::Defer,
                        stmt: s.clone(),
                    });
                }
            }
            StmtKind::ErrDefer(s) => {
                if let Some(cleanups) = self.cleanups.last_mut() {
                    cleanups.push(MirCleanup {
                        kind: MirCleanupKind::ErrDefer,
                        stmt: s.clone(),
                    });
                }
            }
            StmtKind::Scoped(stmts) => {
                self.push_scope();
                for s in stmts {
                    self.lower_statement(s)?;
                }
                self.pop_scope()?;
            }
        }
        Ok(())
    }

    fn lower_condition_with_bindings(
        &mut self,
        condition: &Expr,
        span: crate::ast::Span,
    ) -> Result<(Operand, Vec<(String, MirType, Place, bool)>), MirLowerError> {
        match &condition.kind {
            ExprKind::Paren(inner) => self.lower_condition_with_bindings(inner, span),
            ExprKind::IsPattern { expr: scrutinee, pattern } => {
                match pattern {
                    crate::ast::Pattern::Enum { variant, bindings, .. }
                        if (variant == "Ok" || variant == "Some") =>
                    {
                        if let ExprKind::Index { object, index } = &scrutinee.kind {
                            let base_place = match self.lower_place(object)? {
                                Some(p) => p,
                                None => {
                                    let obj_op = self.lower_expr(object)?;
                                    let temp = self.current_fn.add_local(
                                        None,
                                        MirType::Array(Box::new(MirType::Int)),
                                        true,
                                        span,
                                    );
                                    self.emit_stmt(Statement::StorageLive(temp, span));
                                    self.emit_stmt(Statement::Assign(
                                        Place::Local(temp),
                                        Rvalue::Use(obj_op),
                                        span,
                                    ));
                                    Place::Local(temp)
                                }
                            };

                            let elem_ty = match &base_place {
                                Place::Local(loc) => match &self.current_fn.locals[loc.0].ty {
                                    MirType::Array(inner) => (**inner).clone(),
                                    _ => MirType::Int,
                                },
                                _ => MirType::Int,
                            };

                            let idx_op = self.lower_expr(index)?;
                            let idx_local = match &idx_op {
                                Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => *loc,
                                _ => {
                                    let temp = self.current_fn.add_local(None, MirType::Int, true, span);
                                    self.emit_stmt(Statement::StorageLive(temp, span));
                                    self.emit_stmt(Statement::Assign(
                                        Place::Local(temp),
                                        Rvalue::Use(idx_op),
                                        span,
                                    ));
                                    temp
                                }
                            };

                            let len_temp = self.current_fn.add_local(None, MirType::Int, true, span);
                            self.emit_stmt(Statement::StorageLive(len_temp, span));
                            self.emit_stmt(Statement::Assign(
                                Place::Local(len_temp),
                                Rvalue::UnaryOp(MirUnaryOp::Len, Operand::Copy(base_place.clone())),
                                span,
                            ));

                            let ge_zero_temp = self.current_fn.add_local(None, MirType::Bool, true, span);
                            self.emit_stmt(Statement::StorageLive(ge_zero_temp, span));
                            self.emit_stmt(Statement::Assign(
                                Place::Local(ge_zero_temp),
                                Rvalue::BinaryOp(
                                    MirBinaryOp::GtEq,
                                    Operand::Copy(Place::Local(idx_local)),
                                    Operand::Constant(MirConstant::Int(0)),
                                ),
                                span,
                            ));

                            let lt_len_temp = self.current_fn.add_local(None, MirType::Bool, true, span);
                            self.emit_stmt(Statement::StorageLive(lt_len_temp, span));
                            self.emit_stmt(Statement::Assign(
                                Place::Local(lt_len_temp),
                                Rvalue::BinaryOp(
                                    MirBinaryOp::Lt,
                                    Operand::Copy(Place::Local(idx_local)),
                                    Operand::Copy(Place::Local(len_temp)),
                                ),
                                span,
                            ));

                            let in_bounds_temp = self.current_fn.add_local(None, MirType::Bool, true, span);
                            self.emit_stmt(Statement::StorageLive(in_bounds_temp, span));
                            self.emit_stmt(Statement::Assign(
                                Place::Local(in_bounds_temp),
                                Rvalue::BinaryOp(
                                    MirBinaryOp::And,
                                    Operand::Copy(Place::Local(ge_zero_temp)),
                                    Operand::Copy(Place::Local(lt_len_temp)),
                                ),
                                span,
                            ));

                            let elem_place = Place::Index(Box::new(base_place), idx_local);
                            let mut out_bindings = Vec::new();
                            if let Some(crate::ast::Pattern::Identifier(binding_name)) = bindings.first() {
                                out_bindings.push((binding_name.clone(), elem_ty, elem_place, false));
                            }

                            return Ok((Operand::Move(Place::Local(in_bounds_temp)), out_bindings));
                        }

                        // PERF-T2-5: Lower `if place is Some(binding)` on struct fields or locals of
                        // type `Option<Struct(S)>` into a zero-boxing null-pointer check (`val != 0`).
                        if variant == "Some" && bindings.len() <= 1 {
                            let valid_binding = bindings.first().is_none_or(|p| {
                                matches!(
                                    p,
                                    crate::ast::Pattern::Identifier(_) | crate::ast::Pattern::Wildcard
                                )
                            });
                            if valid_binding {
                                let (scrutinee_op, scrutinee_ty) = if let Some(p) = self.lower_place(scrutinee)? {
                                    let ty = self.get_place_mir_type(&p);
                                    (Operand::Copy(p), ty)
                                } else {
                                    let op = self.lower_expr(scrutinee)?;
                                    let ty = Some(self.operand_definite_mir_type(&op));
                                    (op, ty)
                                };
                                if let Some(MirType::Option(inner)) = scrutinee_ty
                                    && let MirType::Struct(struct_name) = *inner
                                    && !struct_name.is_empty()
                                {
                                    let opt_ty = MirType::Option(Box::new(MirType::Struct(struct_name.clone())));
                                    let opt_tmp = self.current_fn.add_local(None, opt_ty, true, span);
                                    self.known_typed_locals.insert(opt_tmp);
                                    self.emit_stmt(Statement::StorageLive(opt_tmp, span));
                                    self.emit_stmt(Statement::Assign(
                                        Place::Local(opt_tmp),
                                        Rvalue::Use(scrutinee_op),
                                        span,
                                    ));

                                    let cond_tmp = self.current_fn.add_local(None, MirType::Bool, true, span);
                                    self.emit_stmt(Statement::StorageLive(cond_tmp, span));
                                    self.emit_stmt(Statement::Assign(
                                        Place::Local(cond_tmp),
                                        Rvalue::BinaryOp(
                                            MirBinaryOp::NotEq,
                                            Operand::Copy(Place::Local(opt_tmp)),
                                            Operand::Constant(MirConstant::Int(0)),
                                        ),
                                        span,
                                    ));

                                    let mut out_bindings = Vec::new();
                                    if let Some(crate::ast::Pattern::Identifier(binding_name)) = bindings.first() {
                                        out_bindings.push((
                                            binding_name.clone(),
                                            MirType::Struct(struct_name),
                                            Place::Local(opt_tmp),
                                            true,
                                        ));
                                    }
                                    return Ok((Operand::Move(Place::Local(cond_tmp)), out_bindings));
                                }
                            }
                        }

                        Err(MirLowerError::UnlowerableExpression(format!(
                            "non-indexed is-pattern '{variant}' is not supported in MIR lowering"
                        )))
                    }
                    crate::ast::Pattern::Enum { variant, bindings, .. }
                        if variant == "None" && bindings.is_empty() =>
                    {
                        let (scrutinee_op, scrutinee_ty) = if let Some(p) = self.lower_place(scrutinee)? {
                            let ty = self.get_place_mir_type(&p);
                            (Operand::Copy(p), ty)
                        } else {
                            let op = self.lower_expr(scrutinee)?;
                            let ty = Some(self.operand_definite_mir_type(&op));
                            (op, ty)
                        };
                        if let Some(MirType::Option(inner)) = scrutinee_ty
                            && let MirType::Struct(struct_name) = *inner
                            && !struct_name.is_empty()
                        {
                            let opt_ty = MirType::Option(Box::new(MirType::Struct(struct_name)));
                            let opt_tmp = self.current_fn.add_local(None, opt_ty, true, span);
                            self.known_typed_locals.insert(opt_tmp);
                            self.emit_stmt(Statement::StorageLive(opt_tmp, span));
                            self.emit_stmt(Statement::Assign(
                                Place::Local(opt_tmp),
                                Rvalue::Use(scrutinee_op),
                                span,
                            ));

                            let cond_tmp = self.current_fn.add_local(None, MirType::Bool, true, span);
                            self.emit_stmt(Statement::StorageLive(cond_tmp, span));
                            self.emit_stmt(Statement::Assign(
                                Place::Local(cond_tmp),
                                Rvalue::BinaryOp(
                                    MirBinaryOp::Eq,
                                    Operand::Copy(Place::Local(opt_tmp)),
                                    Operand::Constant(MirConstant::Int(0)),
                                ),
                                span,
                            ));
                            return Ok((Operand::Move(Place::Local(cond_tmp)), Vec::new()));
                        }

                        Err(MirLowerError::UnlowerableExpression(
                            "non-indexed is-pattern 'None' is not supported in MIR lowering".to_string(),
                        ))
                    }
                    crate::ast::Pattern::Wildcard => {
                        Ok((Operand::Constant(MirConstant::Bool(true)), Vec::new()))
                    }
                    crate::ast::Pattern::Identifier(name) => {
                        let (place, ty) = if let Some(p) = self.lower_place(scrutinee)? {
                            let ty = match &p {
                                Place::Local(loc) => self.current_fn.locals[loc.0].ty.clone(),
                                _ => MirType::Int,
                            };
                            (p, ty)
                        } else {
                            let op = self.lower_expr(scrutinee)?;
                            let temp = self.current_fn.add_local(None, MirType::Int, true, span);
                            self.emit_stmt(Statement::StorageLive(temp, span));
                            self.emit_stmt(Statement::Assign(Place::Local(temp), Rvalue::Use(op), span));
                            (Place::Local(temp), MirType::Int)
                        };
                        Ok((Operand::Constant(MirConstant::Bool(true)), vec![(name.clone(), ty, place, false)]))
                    }
                    crate::ast::Pattern::Literal(lit) => {
                        let scrutinee_op = self.lower_expr(scrutinee)?;
                        let lit_op = match lit {
                            Literal::Integer(i) => Operand::Constant(MirConstant::Int(*i)),
                            Literal::Float(f) => Operand::Constant(MirConstant::Float(*f)),
                            Literal::Bool(b) => Operand::Constant(MirConstant::Bool(*b)),
                            Literal::String(s) => Operand::Constant(MirConstant::String(s.clone())),
                        };
                        let temp = self.current_fn.add_local(None, MirType::Bool, true, span);
                        self.emit_stmt(Statement::StorageLive(temp, span));
                        self.emit_stmt(Statement::Assign(
                            Place::Local(temp),
                            Rvalue::BinaryOp(MirBinaryOp::Eq, scrutinee_op, lit_op),
                            span,
                        ));
                        Ok((Operand::Move(Place::Local(temp)), Vec::new()))
                    }
                    _ => Err(MirLowerError::UnlowerableExpression(
                        "unsupported is-pattern in MIR lowering".to_string(),
                    )),
                }
            }
            ExprKind::TypeCheck { expr: scrutinee, check_type }
                if matches!(&check_type.kind, TypeKind::Named(n) if n == "None") =>
            {
                let (scrutinee_op, scrutinee_ty) = if let Some(p) = self.lower_place(scrutinee)? {
                    let ty = self.get_place_mir_type(&p);
                    (Operand::Copy(p), ty)
                } else {
                    let op = self.lower_expr(scrutinee)?;
                    let ty = Some(self.operand_definite_mir_type(&op));
                    (op, ty)
                };
                if let Some(MirType::Option(inner)) = scrutinee_ty
                    && let MirType::Struct(struct_name) = *inner
                    && !struct_name.is_empty()
                {
                    let opt_ty = MirType::Option(Box::new(MirType::Struct(struct_name)));
                    let opt_tmp = self.current_fn.add_local(None, opt_ty, true, span);
                    self.known_typed_locals.insert(opt_tmp);
                    self.emit_stmt(Statement::StorageLive(opt_tmp, span));
                    self.emit_stmt(Statement::Assign(
                        Place::Local(opt_tmp),
                        Rvalue::Use(scrutinee_op),
                        span,
                    ));

                    let cond_tmp = self.current_fn.add_local(None, MirType::Bool, true, span);
                    self.emit_stmt(Statement::StorageLive(cond_tmp, span));
                    self.emit_stmt(Statement::Assign(
                        Place::Local(cond_tmp),
                        Rvalue::BinaryOp(
                            MirBinaryOp::Eq,
                            Operand::Copy(Place::Local(opt_tmp)),
                            Operand::Constant(MirConstant::Int(0)),
                        ),
                        span,
                    ));
                    return Ok((Operand::Move(Place::Local(cond_tmp)), Vec::new()));
                }
                Err(MirLowerError::UnlowerableExpression(
                    "TypeCheck 'is None' on non-Option<Struct> is not supported in MIR lowering"
                        .to_string(),
                ))
            }
            _ => Ok((self.lower_expr(condition)?, Vec::new())),
        }
    }

    fn lower_if(
        &mut self,
        condition: &Expr,
        then_block: &[Stmt],
        elseif_branches: &[(Expr, Vec<Stmt>)],
        else_block: Option<&[Stmt]>,
        span: crate::ast::Span,
    ) -> Result<(), MirLowerError> {
        let (cond_op, then_bindings) = self.lower_condition_with_bindings(condition, span)?;
        let then_bb = self.current_fn.new_block();
        let merge_bb = self.current_fn.new_block();

        let next_cond_bb = if !elseif_branches.is_empty() || else_block.is_some() {
            self.current_fn.new_block()
        } else {
            merge_bb
        };

        self.terminate_block(Terminator::Cond {
            cond: cond_op,
            then_block: then_bb,
            else_block: next_cond_bb,
            span,
        });

        // Lower Then Block
        self.current_block = then_bb;
        self.push_scope();
        for (name, ty, place, unwrap_option) in then_bindings {
            let binding_local = self.current_fn.add_local(Some(name.clone()), ty.clone(), false, span);
            if ty != MirType::Unknown {
                self.known_typed_locals.insert(binding_local);
            }
            self.emit_stmt(Statement::StorageLive(binding_local, span));
            let rval = if unwrap_option {
                Rvalue::Cast(Operand::Copy(place), ty)
            } else {
                Rvalue::Use(Operand::Copy(place))
            };
            self.emit_stmt(Statement::Assign(
                Place::Local(binding_local),
                rval,
                span,
            ));
            self.insert_var(name, binding_local);
        }
        for s in then_block {
            self.lower_statement(s)?;
        }
        self.pop_scope()?;
        if self.current_fn.blocks[self.current_block.0].terminator.is_none() {
            self.terminate_block(Terminator::Goto(merge_bb, span));
        }

        // Lower ElseIf branches if present
        let mut cur_else_bb = next_cond_bb;
        for (i, (elif_cond, elif_body)) in elseif_branches.iter().enumerate() {
            if cur_else_bb == merge_bb {
                break;
            }
            self.current_block = cur_else_bb;
            let (elif_op, elif_bindings) = self.lower_condition_with_bindings(elif_cond, span)?;
            let elif_then = self.current_fn.new_block();
            let next_elif = if i + 1 < elseif_branches.len() || else_block.is_some() {
                self.current_fn.new_block()
            } else {
                merge_bb
            };

            self.terminate_block(Terminator::Cond {
                cond: elif_op,
                then_block: elif_then,
                else_block: next_elif,
                span,
            });

            self.current_block = elif_then;
            self.push_scope();
            for (name, ty, place, unwrap_option) in elif_bindings {
                let binding_local = self.current_fn.add_local(Some(name.clone()), ty.clone(), false, span);
                if ty != MirType::Unknown {
                    self.known_typed_locals.insert(binding_local);
                }
                self.emit_stmt(Statement::StorageLive(binding_local, span));
                let rval = if unwrap_option {
                    Rvalue::Cast(Operand::Copy(place), ty)
                } else {
                    Rvalue::Use(Operand::Copy(place))
                };
                self.emit_stmt(Statement::Assign(
                    Place::Local(binding_local),
                    rval,
                    span,
                ));
                self.insert_var(name, binding_local);
            }
            for s in elif_body {
                self.lower_statement(s)?;
            }
            self.pop_scope()?;
            if self.current_fn.blocks[self.current_block.0].terminator.is_none() {
                self.terminate_block(Terminator::Goto(merge_bb, span));
            }

            cur_else_bb = next_elif;
        }

        // Lower Else block if present
        if let Some(eb) = else_block {
            if cur_else_bb != merge_bb {
                self.current_block = cur_else_bb;
                self.push_scope();
                for s in eb {
                    self.lower_statement(s)?;
                }
                self.pop_scope()?;
                if self.current_fn.blocks[self.current_block.0].terminator.is_none() {
                    self.terminate_block(Terminator::Goto(merge_bb, span));
                }
            }
        }

        self.current_block = merge_bb;
        Ok(())
    }

    pub fn lower_expr(&mut self, expr: &Expr) -> Result<Operand, MirLowerError> {
        self.lower_expr_with_expected(expr, None)
    }

    fn lower_option_struct_none(
        &mut self,
        expected_ty: Option<&MirType>,
        span: crate::ast::Span,
    ) -> Option<Operand> {
        let Some(MirType::Option(inner)) = expected_ty else {
            return None;
        };
        if !matches!(&**inner, MirType::Struct(name) if !name.is_empty()) {
            return None;
        }
        let opt_ty = MirType::Option(inner.clone());
        let temp = self.current_fn.add_local(None, opt_ty.clone(), true, span);
        self.known_typed_locals.insert(temp);
        self.emit_stmt(Statement::StorageLive(temp, span));
        self.emit_stmt(Statement::Assign(
            Place::Local(temp),
            Rvalue::Cast(Operand::Constant(MirConstant::Int(0)), opt_ty),
            span,
        ));
        Some(Operand::Move(Place::Local(temp)))
    }

    fn lower_option_struct_some(
        &mut self,
        arg_expr: &Expr,
        expected_ty: Option<&MirType>,
        span: crate::ast::Span,
    ) -> Result<Operand, MirLowerError> {
        let inner_expected = match expected_ty {
            Some(MirType::Option(inner)) => Some(inner.as_ref()),
            _ => None,
        };
        let arg_op = self.lower_expr_with_expected(arg_expr, inner_expected)?;
        let arg_ty = self.operand_definite_mir_type(&arg_op);
        let opt_ty = match arg_ty {
            MirType::Struct(struct_name) if !struct_name.is_empty() => {
                Some(MirType::Option(Box::new(MirType::Struct(struct_name))))
            }
            _ => match expected_ty {
                Some(MirType::Option(inner))
                    if matches!(&**inner, MirType::Struct(name) if !name.is_empty()) =>
                {
                    Some(MirType::Option(inner.clone()))
                }
                _ => None,
            },
        };
        if let Some(opt_ty) = opt_ty {
            let temp = self.current_fn.add_local(None, opt_ty.clone(), true, span);
            self.known_typed_locals.insert(temp);
            self.emit_stmt(Statement::StorageLive(temp, span));
            self.emit_stmt(Statement::Assign(
                Place::Local(temp),
                Rvalue::Cast(arg_op, opt_ty),
                span,
            ));
            return Ok(Operand::Move(Place::Local(temp)));
        }
        Ok(arg_op)
    }

    pub fn lower_expr_with_expected(
        &mut self,
        expr: &Expr,
        expected_ty: Option<&MirType>,
    ) -> Result<Operand, MirLowerError> {
        match &expr.kind {
            ExprKind::Literal(lit) => Ok(match lit {
                Literal::Integer(i) => Operand::Constant(MirConstant::Int(*i)),
                Literal::Float(f) => Operand::Constant(MirConstant::Float(*f)),
                Literal::Bool(b) => Operand::Constant(MirConstant::Bool(*b)),
                Literal::String(s) => Operand::Constant(MirConstant::String(s.clone())),
            }),
            ExprKind::RegexLiteral { pattern, .. } => {
                Ok(Operand::Constant(MirConstant::String(pattern.clone())))
            }
            ExprKind::Identifier(name) => {
                if let Some(loc_id) = self.lookup_var(name) {
                    Ok(Operand::Copy(Place::Local(loc_id)))
                } else if self.lookup_thunk(name).is_some() {
                    Err(MirLowerError::UnlowerableExpression(format!(
                        "Effect thunk '{name}' cannot escape or be used as a value"
                    )))
                } else if name == "None" {
                    if let Some(none_op) = self.lower_option_struct_none(expected_ty, expr.span) {
                        Ok(none_op)
                    } else {
                        Ok(Operand::Constant(MirConstant::Int(0)))
                    }
                } else {
                    Ok(Operand::Constant(MirConstant::Int(0)))
                }
            }
            ExprKind::EffectBlock(_) => {
                Err(MirLowerError::UnlowerableExpression(
                    "Bare EffectBlock in unsupported value context; only local bindings are supported".into(),
                ))
            }
            ExprKind::Paren(inner) => self.lower_expr_with_expected(inner, expected_ty),
            ExprKind::Binary { left, op, right } => {
                if *op == BinaryOp::And {
                    let res_local = self.current_fn.add_local(None, MirType::Bool, true, expr.span);
                    self.emit_stmt(Statement::StorageLive(res_local, expr.span));
                    let lhs_op = self.lower_expr(left)?;
                    self.emit_stmt(Statement::Assign(
                        Place::Local(res_local),
                        Rvalue::Use(lhs_op.clone()),
                        expr.span,
                    ));

                    let rhs_bb = self.current_fn.new_block();
                    let merge_bb = self.current_fn.new_block();

                    self.terminate_block(Terminator::Cond {
                        cond: lhs_op,
                        then_block: rhs_bb,
                        else_block: merge_bb,
                        span: expr.span,
                    });

                    self.current_block = rhs_bb;
                    let rhs_op = self.lower_expr(right)?;
                    self.emit_stmt(Statement::Assign(
                        Place::Local(res_local),
                        Rvalue::Use(rhs_op),
                        expr.span,
                    ));
                    if self.current_fn.blocks[self.current_block.0].terminator.is_none() {
                        self.terminate_block(Terminator::Goto(merge_bb, expr.span));
                    }

                    self.current_block = merge_bb;
                    return Ok(Operand::Move(Place::Local(res_local)));
                }

                if *op == BinaryOp::Or {
                    let res_local = self.current_fn.add_local(None, MirType::Bool, true, expr.span);
                    self.emit_stmt(Statement::StorageLive(res_local, expr.span));
                    let lhs_op = self.lower_expr(left)?;
                    self.emit_stmt(Statement::Assign(
                        Place::Local(res_local),
                        Rvalue::Use(lhs_op.clone()),
                        expr.span,
                    ));

                    let rhs_bb = self.current_fn.new_block();
                    let merge_bb = self.current_fn.new_block();

                    self.terminate_block(Terminator::Cond {
                        cond: lhs_op,
                        then_block: merge_bb,
                        else_block: rhs_bb,
                        span: expr.span,
                    });

                    self.current_block = rhs_bb;
                    let rhs_op = self.lower_expr(right)?;
                    self.emit_stmt(Statement::Assign(
                        Place::Local(res_local),
                        Rvalue::Use(rhs_op),
                        expr.span,
                    ));
                    if self.current_fn.blocks[self.current_block.0].terminator.is_none() {
                        self.terminate_block(Terminator::Goto(merge_bb, expr.span));
                    }

                    self.current_block = merge_bb;
                    return Ok(Operand::Move(Place::Local(res_local)));
                }

                let lhs = self.lower_expr(left)?;
                let rhs = self.lower_expr(right)?;
                let mir_op = mir_binop_from_ast(*op)?;
                let res_ty = match mir_op {
                    MirBinaryOp::Eq | MirBinaryOp::NotEq | MirBinaryOp::Lt | MirBinaryOp::LtEq | MirBinaryOp::Gt | MirBinaryOp::GtEq => MirType::Bool,
                    MirBinaryOp::Concat => MirType::String,
                    _ => MirType::Int,
                };
                let temp = self.current_fn.add_local(None, res_ty, true, expr.span);
                self.emit_stmt(Statement::StorageLive(temp, expr.span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(temp),
                    Rvalue::BinaryOp(mir_op, lhs, rhs),
                    expr.span,
                ));
                Ok(Operand::Move(Place::Local(temp)))
            }
            ExprKind::Unary { op, operand } => {
                let inner = self.lower_expr(operand)?;
                let mir_op = match op {
                    UnaryOp::Neg => MirUnaryOp::Neg,
                    UnaryOp::Not => MirUnaryOp::Not,
                    UnaryOp::BitNot => MirUnaryOp::BitNot,
                };
                let res_ty = match mir_op {
                    MirUnaryOp::Not => MirType::Bool,
                    _ => MirType::Int,
                };
                let temp = self.current_fn.add_local(None, res_ty, true, expr.span);
                self.emit_stmt(Statement::StorageLive(temp, expr.span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(temp),
                    Rvalue::UnaryOp(mir_op, inner),
                    expr.span,
                ));
                Ok(Operand::Move(Place::Local(temp)))
            }
            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let (cond_op, bindings) = self.lower_condition_with_bindings(condition, expr.span)?;
                let res_local = self.current_fn.add_local(None, MirType::Int, true, expr.span);
                self.emit_stmt(Statement::StorageLive(res_local, expr.span));

                let then_bb = self.current_fn.new_block();
                let else_bb = self.current_fn.new_block();
                let merge_bb = self.current_fn.new_block();

                self.terminate_block(Terminator::Cond {
                    cond: cond_op,
                    then_block: then_bb,
                    else_block: else_bb,
                    span: expr.span,
                });

                self.current_block = then_bb;
                self.push_scope();
                for (name, ty, place, unwrap_option) in bindings {
                    let binding_local = self.current_fn.add_local(Some(name.clone()), ty.clone(), false, expr.span);
                    if ty != MirType::Unknown {
                        self.known_typed_locals.insert(binding_local);
                    }
                    self.emit_stmt(Statement::StorageLive(binding_local, expr.span));
                    let rval = if unwrap_option {
                        Rvalue::Cast(Operand::Copy(place), ty)
                    } else {
                        Rvalue::Use(Operand::Copy(place))
                    };
                    self.emit_stmt(Statement::Assign(
                        Place::Local(binding_local),
                        rval,
                        expr.span,
                    ));
                    self.insert_var(name, binding_local);
                }
                let then_op = self.lower_expr_with_expected(then_branch, expected_ty)?;
                self.emit_stmt(Statement::Assign(
                    Place::Local(res_local),
                    Rvalue::Use(then_op),
                    expr.span,
                ));
                self.pop_scope()?;
                if self.current_fn.blocks[self.current_block.0].terminator.is_none() {
                    self.terminate_block(Terminator::Goto(merge_bb, expr.span));
                }

                self.current_block = else_bb;
                if let Some(eb) = else_branch {
                    let else_op = self.lower_expr_with_expected(eb, expected_ty)?;
                    self.emit_stmt(Statement::Assign(
                        Place::Local(res_local),
                        Rvalue::Use(else_op),
                        expr.span,
                    ));
                }
                if self.current_fn.blocks[self.current_block.0].terminator.is_none() {
                    self.terminate_block(Terminator::Goto(merge_bb, expr.span));
                }

                self.current_block = merge_bb;
                Ok(Operand::Move(Place::Local(res_local)))
            }
            ExprKind::IsPattern { .. } => {
                let (cond_op, _) = self.lower_condition_with_bindings(expr, expr.span)?;
                Ok(cond_op)
            }
            ExprKind::TypeCheck { check_type, .. }
                if matches!(&check_type.kind, TypeKind::Named(n) if n == "None") =>
            {
                let (cond_op, _) = self.lower_condition_with_bindings(expr, expr.span)?;
                Ok(cond_op)
            }
            ExprKind::Array(elements) => {
                let mut lowered_elems = Vec::with_capacity(elements.len());
                for e in elements {
                    lowered_elems.push(self.lower_expr(e)?);
                }
                let elem_ty = if lowered_elems.is_empty() {
                    MirType::Unknown
                } else {
                    let mut unified: Option<MirType> = None;
                    let mut unresolved_or_conflict = false;
                    for op in &lowered_elems {
                        let ty = self.operand_definite_mir_type(op);
                        if ty == MirType::Unknown {
                            unresolved_or_conflict = true;
                            break;
                        }
                        match &unified {
                            None => unified = Some(ty),
                            Some(prev) if *prev == ty => {}
                            Some(_) => {
                                unresolved_or_conflict = true;
                                break;
                            }
                        }
                    }
                    if unresolved_or_conflict {
                        MirType::Unknown
                    } else {
                        unified.unwrap_or(MirType::Unknown)
                    }
                };
                let arr_ty = MirType::Array(Box::new(elem_ty));
                let temp = self.current_fn.add_local(None, arr_ty.clone(), true, expr.span);
                self.emit_stmt(Statement::StorageLive(temp, expr.span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(temp),
                    Rvalue::Aggregate(arr_ty, lowered_elems),
                    expr.span,
                ));
                Ok(Operand::Move(Place::Local(temp)))
            }
            ExprKind::Index { object, index } => {
                let obj_op = self.lower_expr(object)?;
                let idx_op = self.lower_expr(index)?;

                let base_place = match obj_op {
                    Operand::Copy(p) | Operand::Move(p) => p,
                    Operand::Constant(_) => {
                        let temp = self.current_fn.add_local(None, MirType::Array(Box::new(MirType::Int)), true, expr.span);
                        self.emit_stmt(Statement::StorageLive(temp, expr.span));
                        self.emit_stmt(Statement::Assign(Place::Local(temp), Rvalue::Use(obj_op), expr.span));
                        Place::Local(temp)
                    }
                };
                let idx_local = match idx_op {
                    Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => loc,
                    _ => {
                        let temp = self.current_fn.add_local(None, MirType::Int, true, expr.span);
                        self.emit_stmt(Statement::StorageLive(temp, expr.span));
                        self.emit_stmt(Statement::Assign(Place::Local(temp), Rvalue::Use(idx_op), expr.span));
                        temp
                    }
                };
                let elem_ty = match &base_place {
                    Place::Local(loc) => match &self.current_fn.locals[loc.0].ty {
                        MirType::Array(inner) => (**inner).clone(),
                        _ => MirType::Int,
                    },
                    _ => MirType::Int,
                };
                let elem_place = Place::Index(Box::new(base_place), idx_local);
                let val_temp = self.current_fn.add_local(None, elem_ty, true, expr.span);
                self.emit_stmt(Statement::StorageLive(val_temp, expr.span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(val_temp),
                    Rvalue::Use(Operand::Copy(elem_place)),
                    expr.span,
                ));
                Ok(Operand::Move(Place::Local(val_temp)))
            }
            ExprKind::MethodCall { receiver, method, args, .. } if method == "len" && args.is_empty() => {
                let rec_op = self.lower_expr(receiver)?;
                if let MirType::Struct(struct_name) = self.operand_definite_mir_type(&rec_op)
                    && !struct_name.is_empty()
                {
                    return self.lower_struct_method_call(
                        struct_name,
                        method,
                        rec_op,
                        args,
                        expr.span,
                    );
                }
                let temp = self.current_fn.add_local(None, MirType::Int, true, expr.span);
                self.emit_stmt(Statement::StorageLive(temp, expr.span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(temp),
                    Rvalue::UnaryOp(MirUnaryOp::Len, rec_op),
                    expr.span,
                ));
                Ok(Operand::Move(Place::Local(temp)))
            }
            ExprKind::MethodCall { receiver, method, args, .. } if method == "unwrap" && args.is_empty() => {
                if Self::is_index_expr(receiver) {
                    self.lower_expr(receiver)
                } else {
                    let rec_op = self.lower_expr(receiver)?;
                    if let MirType::Struct(struct_name) = self.operand_definite_mir_type(&rec_op)
                        && !struct_name.is_empty()
                    {
                        self.lower_struct_method_call(
                            struct_name,
                            method,
                            rec_op,
                            args,
                            expr.span,
                        )
                    } else {
                        Err(MirLowerError::UnlowerableExpression(format!(
                            "method call ':{method}' on non-index non-struct receiver is not supported in MIR lowering"
                        )))
                    }
                }
            }
            ExprKind::MethodCall { receiver, method, args, .. } if method == "run_sync" => {
                if !args.is_empty() {
                    return Err(MirLowerError::UnlowerableExpression(
                        "run_sync() takes no arguments".into(),
                    ));
                }
                let thunk_name = match &receiver.kind {
                    ExprKind::Identifier(name) => name,
                    _ => {
                        return Err(MirLowerError::UnlowerableExpression(
                            "run_sync receiver must be a local effect thunk identifier".into(),
                        ));
                    }
                };
                let thunk = self.lookup_thunk(thunk_name).cloned().ok_or_else(|| {
                    MirLowerError::UnlowerableExpression(format!(
                        "'{thunk_name}' is not an active local effect thunk in the current scope"
                    ))
                })?;
                self.lower_effect_run_sync(&thunk, expr.span)
            }
            ExprKind::Call { callee, args, .. }
                if args.is_empty()
                    && matches!(&callee.kind, ExprKind::FieldAccess { field, .. } if field == "len") =>
            {
                if let ExprKind::FieldAccess { object, .. } = &callee.kind {
                    let rec_op = self.lower_expr(object)?;
                    let temp = self.current_fn.add_local(None, MirType::Int, true, expr.span);
                    self.emit_stmt(Statement::StorageLive(temp, expr.span));
                    self.emit_stmt(Statement::Assign(
                        Place::Local(temp),
                        Rvalue::UnaryOp(MirUnaryOp::Len, rec_op),
                        expr.span,
                    ));
                    Ok(Operand::Move(Place::Local(temp)))
                } else {
                    Ok(Operand::Constant(MirConstant::Int(0)))
                }
            }
            ExprKind::Call { callee, args, .. }
                if args.len() == 1
                    && matches!(&callee.kind, ExprKind::Identifier(name) if name == "len") =>
            {
                let rec_op = self.lower_expr(&args[0])?;
                let temp = self.current_fn.add_local(None, MirType::Int, true, expr.span);
                self.emit_stmt(Statement::StorageLive(temp, expr.span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(temp),
                    Rvalue::UnaryOp(MirUnaryOp::Len, rec_op),
                    expr.span,
                ));
                Ok(Operand::Move(Place::Local(temp)))
            }
            ExprKind::FieldAccess { object, field } if field == "len" => {
                let rec_op = self.lower_expr(object)?;
                let temp = self.current_fn.add_local(None, MirType::Int, true, expr.span);
                self.emit_stmt(Statement::StorageLive(temp, expr.span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(temp),
                    Rvalue::UnaryOp(MirUnaryOp::Len, rec_op),
                    expr.span,
                ));
                Ok(Operand::Move(Place::Local(temp)))
            }
            ExprKind::Tuple(elements) => {
                let mut lowered = Vec::with_capacity(elements.len());
                let mut elem_types = Vec::with_capacity(elements.len());
                let mut field_names = Vec::with_capacity(elements.len());
                for (i, elem) in elements.iter().enumerate() {
                    let op = self.lower_expr(elem)?;
                    elem_types.push(self.operand_definite_mir_type(&op));
                    lowered.push(op);
                    field_names.push(format!("{i}"));
                }
                let tuple_ty = MirType::Tuple(elem_types);
                let temp = self.current_fn.add_local(None, tuple_ty.clone(), true, expr.span);
                self.local_field_names.insert(temp, field_names);
                self.emit_stmt(Statement::StorageLive(temp, expr.span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(temp),
                    Rvalue::Aggregate(tuple_ty, lowered),
                    expr.span,
                ));
                Ok(Operand::Move(Place::Local(temp)))
            }
            ExprKind::StructLiteral { name, fields } => {
                let mut lowered = Vec::with_capacity(fields.len());
                let mut field_names = Vec::with_capacity(fields.len());
                let mut field_substructs = Vec::with_capacity(fields.len());
                let declared_names_opt = self.struct_fields.get(name).cloned();
                let declared_types_opt = self.struct_field_types.get(name).cloned();
                for f in fields {
                    let sub = match &f.value.kind {
                        ExprKind::StructLiteral { name: sub_name, .. } => Some(sub_name.clone()),
                        _ => None,
                    };
                    field_substructs.push(sub);
                    let expected_field_ty = declared_names_opt
                        .as_ref()
                        .and_then(|names| names.iter().position(|n| n == &f.name))
                        .and_then(|pos| declared_types_opt.as_ref().and_then(|tys| tys.get(pos)));
                    lowered.push(self.lower_expr_with_expected(&f.value, expected_field_ty)?);
                    field_names.push(f.name.clone());
                }
                let (ordered_lowered, ordered_names) =
                    if let Some(declared_fields) = declared_names_opt
                        && declared_fields.len() == fields.len()
                        && declared_fields.iter().all(|df| field_names.contains(df))
                    {
                        let mut by_name: HashMap<String, Operand> = field_names
                            .into_iter()
                            .zip(lowered.into_iter())
                            .collect();
                        let ordered: Vec<Operand> = declared_fields
                            .iter()
                            .map(|df| by_name.remove(df).unwrap())
                            .collect();
                        (ordered, declared_fields)
                    } else {
                        self.struct_fields.insert(name.clone(), field_names.clone());
                        self.struct_field_substructs
                            .insert(name.clone(), field_substructs);
                        (lowered, field_names)
                    };
                let struct_ty = MirType::Struct(name.clone());
                let temp = self.current_fn.add_local(None, struct_ty.clone(), true, expr.span);
                self.local_field_names.insert(temp, ordered_names);
                self.emit_stmt(Statement::StorageLive(temp, expr.span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(temp),
                    Rvalue::Aggregate(struct_ty, ordered_lowered),
                    expr.span,
                ));
                Ok(Operand::Move(Place::Local(temp)))
            }
            ExprKind::EnumConstructor { enum_name, variant, args } => {
                if enum_name == "Option" && variant == "None" && args.is_empty() {
                    if let Some(none_op) = self.lower_option_struct_none(expected_ty, expr.span) {
                        return Ok(none_op);
                    }
                } else if enum_name == "Option" && variant == "Some" && args.len() == 1 {
                    return self.lower_option_struct_some(&args[0], expected_ty, expr.span);
                }
                Err(MirLowerError::UnlowerableExpression(format!(
                    "enum constructor '{enum_name}.{variant}' is not supported in MIR lowering"
                )))
            }
            ExprKind::FieldAccess { object, field } if field != "len" => {
                if let ExprKind::Identifier(name) = &object.kind
                    && self.lookup_var(name).is_none()
                {
                    if name == "Option" && field == "None" {
                        if let Some(none_op) = self.lower_option_struct_none(expected_ty, expr.span) {
                            return Ok(none_op);
                        }
                    }
                    return Err(MirLowerError::UnlowerableExpression(format!(
                        "field access on non-local '{name}.{field}' is not supported in MIR lowering"
                    )));
                }
                let base_place = if let Some(p) = self.lower_place(object)? {
                    p
                } else {
                    let obj_op = self.lower_expr(object)?;
                    match obj_op {
                        Operand::Copy(p) | Operand::Move(p) => p,
                        Operand::Constant(_) => {
                            return Err(MirLowerError::UnlowerableExpression(format!(
                                "field access '.{field}' on constant value is not supported in MIR lowering"
                            )));
                        }
                    }
                };
                let field_idx = self.resolve_field_idx(&base_place, field).ok_or_else(|| {
                    MirLowerError::UnlowerableExpression(format!(
                        "cannot resolve field '{field}' in MIR lowering"
                    ))
                })?;
                let field_place = Place::Field(Box::new(base_place.clone()), field_idx);
                let mut field_mir_ty = MirType::Int;
                let mut recorded_load: Option<(String, u16, usize, CanonicalTypeTag)> = None;
                if let Some(struct_name) = self.get_place_struct_name(&base_place) {
                    let explicit_field_ty = self
                        .struct_field_types
                        .get(&struct_name)
                        .and_then(|v| v.get(field_idx))
                        .cloned();
                    if let Some(fty) = explicit_field_ty.as_ref() {
                        field_mir_ty = fty.clone();
                    }
                    if let Some(layout) = self.struct_layouts.get(&struct_name) {
                        if let Some(f_desc) = layout.field_by_name(field) {
                            if explicit_field_ty.is_none() {
                                field_mir_ty = match f_desc.tag {
                                    CanonicalTypeTag::Bool => MirType::Bool,
                                    CanonicalTypeTag::Int64
                                    | CanonicalTypeTag::UInt64
                                    | CanonicalTypeTag::DateTimeUtc => MirType::Int,
                                    CanonicalTypeTag::Float64 => MirType::Float,
                                    CanonicalTypeTag::StringSlice => MirType::String,
                                    _ => MirType::Int,
                                };
                            }
                            recorded_load = Some((
                                struct_name,
                                f_desc.ordinal,
                                f_desc.byte_offset,
                                f_desc.tag,
                            ));
                        }
                    }
                }
                let temp = self.current_fn.add_local(None, field_mir_ty, true, expr.span);
                if let Some((struct_name, ordinal, byte_offset, tag)) = recorded_load {
                    self.struct_ops.push(MirStructOp::StructLoadField {
                        struct_name,
                        base: base_place,
                        ordinal,
                        byte_offset,
                        tag,
                        dst: temp,
                    });
                }
                self.emit_stmt(Statement::StorageLive(temp, expr.span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(temp),
                    Rvalue::Use(Operand::Copy(field_place)),
                    expr.span,
                ));
                Ok(Operand::Move(Place::Local(temp)))
            }
            ExprKind::MethodCall { receiver, method, args, .. } if method == "try_unwrap" && args.is_empty() => {
                if let ExprKind::Index { object, index } = &receiver.kind {
                    let base_place = match self.lower_place(object)? {
                        Some(p) => p,
                        None => {
                            let obj_op = self.lower_expr(object)?;
                            let temp = self.current_fn.add_local(None, MirType::Array(Box::new(MirType::Int)), true, expr.span);
                            self.emit_stmt(Statement::StorageLive(temp, expr.span));
                            self.emit_stmt(Statement::Assign(Place::Local(temp), Rvalue::Use(obj_op), expr.span));
                            Place::Local(temp)
                        }
                    };
                    let idx_op = self.lower_expr(index)?;
                    let idx_local = match idx_op {
                        Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => loc,
                        _ => {
                            let temp = self.current_fn.add_local(None, MirType::Int, true, expr.span);
                            self.emit_stmt(Statement::StorageLive(temp, expr.span));
                            self.emit_stmt(Statement::Assign(Place::Local(temp), Rvalue::Use(idx_op), expr.span));
                            temp
                        }
                    };

                    let len_temp = self.current_fn.add_local(None, MirType::Int, true, expr.span);
                    self.emit_stmt(Statement::StorageLive(len_temp, expr.span));
                    self.emit_stmt(Statement::Assign(
                        Place::Local(len_temp),
                        Rvalue::UnaryOp(MirUnaryOp::Len, Operand::Copy(base_place.clone())),
                        expr.span,
                    ));

                    let ge_zero_temp = self.current_fn.add_local(None, MirType::Bool, true, expr.span);
                    self.emit_stmt(Statement::StorageLive(ge_zero_temp, expr.span));
                    self.emit_stmt(Statement::Assign(
                        Place::Local(ge_zero_temp),
                        Rvalue::BinaryOp(
                            MirBinaryOp::GtEq,
                            Operand::Copy(Place::Local(idx_local)),
                            Operand::Constant(MirConstant::Int(0)),
                        ),
                        expr.span,
                    ));

                    let lt_len_temp = self.current_fn.add_local(None, MirType::Bool, true, expr.span);
                    self.emit_stmt(Statement::StorageLive(lt_len_temp, expr.span));
                    self.emit_stmt(Statement::Assign(
                        Place::Local(lt_len_temp),
                        Rvalue::BinaryOp(
                            MirBinaryOp::Lt,
                            Operand::Copy(Place::Local(idx_local)),
                            Operand::Copy(Place::Local(len_temp)),
                        ),
                        expr.span,
                    ));

                    let in_bounds_temp = self.current_fn.add_local(None, MirType::Bool, true, expr.span);
                    self.emit_stmt(Statement::StorageLive(in_bounds_temp, expr.span));
                    self.emit_stmt(Statement::Assign(
                        Place::Local(in_bounds_temp),
                        Rvalue::BinaryOp(
                            MirBinaryOp::And,
                            Operand::Copy(Place::Local(ge_zero_temp)),
                            Operand::Copy(Place::Local(lt_len_temp)),
                        ),
                        expr.span,
                    ));

                    let ok_bb = self.current_fn.new_block();
                    let err_bb = self.current_fn.new_block();

                    self.terminate_block(Terminator::Cond {
                        cond: Operand::Move(Place::Local(in_bounds_temp)),
                        then_block: ok_bb,
                        else_block: err_bb,
                        span: expr.span,
                    });

                    // err_bb
                    self.current_block = err_bb;
                    self.terminate_block(Terminator::Return(Some(Operand::Constant(MirConstant::Int(-1))), expr.span));

                    // ok_bb
                    self.current_block = ok_bb;
                    let elem_place = Place::Index(Box::new(base_place), idx_local);
                    let res_local = self.current_fn.add_local(None, MirType::Int, true, expr.span);
                    self.emit_stmt(Statement::StorageLive(res_local, expr.span));
                    self.emit_stmt(Statement::Assign(
                        Place::Local(res_local),
                        Rvalue::Use(Operand::Copy(elem_place)),
                        expr.span,
                    ));
                    Ok(Operand::Move(Place::Local(res_local)))
                } else {
                    self.lower_expr(receiver)
                }
            }
            ExprKind::Call { callee, args, .. } => {
                let callee_name = match &callee.kind {
                    ExprKind::Identifier(name) => name.clone(),
                    ExprKind::FieldAccess { object, field } => {
                        if let ExprKind::Identifier(obj_name) = &object.kind {
                            format!("{obj_name}.{field}")
                        } else {
                            "unknown_callee".into()
                        }
                    }
                    _ => "unknown_callee".into(),
                };
                match callee_name.as_str() {
                    "Some" | "Option.Some" => {
                        if let Some(arg) = args.first() {
                            return self.lower_option_struct_some(arg, expected_ty, expr.span);
                        }
                    }
                    "Ok" | "Result.Ok" => {
                        if let Some(arg) = args.first() {
                            return self.lower_expr(arg);
                        }
                    }
                    "Err" | "Result.Err" => {
                        if let Some(arg) = args.first() {
                            return self.lower_expr(arg);
                        }
                    }
                    "None" | "Option.None" => {
                        if let Some(none_op) = self.lower_option_struct_none(expected_ty, expr.span) {
                            return Ok(none_op);
                        }
                        return Ok(Operand::Constant(MirConstant::Int(0)));
                    }
                    _ => {}
                }
                if let Some(struct_name) = callee_name.strip_suffix(".decode") {
                    if self.struct_layouts.contains_key(struct_name) && args.len() == 1 {
                        let src_op = self.lower_expr(&args[0])?;
                        let dst = self.lower_struct_decode_into(struct_name, src_op, expr.span)?;
                        return Ok(Operand::Move(Place::Local(dst)));
                    }
                }
                let mut lowered_args = Vec::with_capacity(args.len());
                for a in args {
                    lowered_args.push(self.lower_expr(a)?);
                }
                let call_ret_ty = if callee_name == "tostring" {
                    MirType::String
                } else {
                    MirType::Int
                };
                let temp = self.current_fn.add_local(None, call_ret_ty, true, expr.span);
                let succ_bb = self.current_fn.new_block();

                self.emit_stmt(Statement::StorageLive(temp, expr.span));
                self.terminate_block(Terminator::Call {
                    callee: callee_name,
                    args: lowered_args,
                    destination: Place::Local(temp),
                    success: succ_bb,
                    error: None,
                    span: expr.span,
                });

                self.current_block = succ_bb;
                Ok(Operand::Move(Place::Local(temp)))
            }
            ExprKind::MethodCall { receiver, method, args, .. } => {
                let rec_op = self.lower_expr(receiver)?;
                if let MirType::Struct(struct_name) = self.operand_definite_mir_type(&rec_op)
                    && !struct_name.is_empty()
                {
                    self.lower_struct_method_call(struct_name, method, rec_op, args, expr.span)
                } else {
                    Err(MirLowerError::UnlowerableExpression(format!(
                        "method call ':{method}' on non-struct receiver is not supported in MIR lowering"
                    )))
                }
            }
            ExprKind::Cast { expr: inner, target_type } => {
                let inner_op = self.lower_expr(inner)?;
                let mir_target_ty = mir_type_from_ast(target_type);
                let temp = self.current_fn.add_local(None, mir_target_ty.clone(), true, expr.span);
                self.emit_stmt(Statement::StorageLive(temp, expr.span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(temp),
                    Rvalue::Cast(inner_op, mir_target_ty),
                    expr.span,
                ));
                Ok(Operand::Move(Place::Local(temp)))
            }
            _ => Err(MirLowerError::UnlowerableExpression(format!("{:?}", expr.kind))),
        }
    }

    fn lower_place(&mut self, expr: &Expr) -> Result<Option<Place>, MirLowerError> {
        match &expr.kind {
            ExprKind::Identifier(name) => {
                if self.lookup_thunk(name).is_some() {
                    return Err(MirLowerError::UnlowerableExpression(format!(
                        "Effect thunk '{name}' cannot be reassigned or used as an lvalue"
                    )));
                }
                Ok(self.lookup_var(name).map(Place::Local))
            }
            ExprKind::Paren(inner) => self.lower_place(inner),
            ExprKind::Index { object, index } => {
                let base_place = match self.lower_place(object)? {
                    Some(p) => p,
                    None => return Err(MirLowerError::UnsupportedLvalue(format!("{:?}", object.kind))),
                };
                let idx_op = self.lower_expr(index)?;
                let idx_local = match idx_op {
                    Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => loc,
                    _ => {
                        let temp = self.current_fn.add_local(None, MirType::Int, true, expr.span);
                        self.emit_stmt(Statement::StorageLive(temp, expr.span));
                        self.emit_stmt(Statement::Assign(Place::Local(temp), Rvalue::Use(idx_op), expr.span));
                        temp
                    }
                };
                Ok(Some(Place::Index(Box::new(base_place), idx_local)))
            }
            ExprKind::FieldAccess { object, field } if field != "len" => {
                let base_place = match self.lower_place(object)? {
                    Some(p) => p,
                    None => return Err(MirLowerError::UnsupportedLvalue(format!("{:?}", object.kind))),
                };
                if let Some(field_idx) = self.resolve_field_idx(&base_place, field) {
                    Ok(Some(Place::Field(Box::new(base_place), field_idx)))
                } else {
                    Err(MirLowerError::UnsupportedLvalue(format!("{:?}", expr.kind)))
                }
            }
            ExprKind::FieldAccess { .. } => Err(MirLowerError::UnsupportedLvalue(format!("{:?}", expr.kind))),
            _ => Ok(None),
        }
    }

    fn is_index_expr(expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Index { .. } => true,
            ExprKind::Paren(inner) => Self::is_index_expr(inner),
            _ => false,
        }
    }

    fn lower_struct_method_call(
        &mut self,
        struct_name: String,
        method: &str,
        rec_op: Operand,
        args: &[Expr],
        span: crate::ast::Span,
    ) -> Result<Operand, MirLowerError> {
        let mut lowered_args = Vec::with_capacity(1 + args.len());
        lowered_args.push(rec_op);
        for a in args {
            lowered_args.push(self.lower_expr(a)?);
        }
        let callee_name = format!("{struct_name}:{method}");
        let temp = self.current_fn.add_local(None, MirType::Int, true, span);
        let succ_bb = self.current_fn.new_block();

        self.emit_stmt(Statement::StorageLive(temp, span));
        self.terminate_block(Terminator::Call {
            callee: callee_name,
            args: lowered_args,
            destination: Place::Local(temp),
            success: succ_bb,
            error: None,
            span,
        });

        self.current_block = succ_bb;
        Ok(Operand::Move(Place::Local(temp)))
    }

    fn emit_stmt(&mut self, stmt: Statement) {
        self.current_fn.blocks[self.current_block.0]
            .statements
            .push(stmt);
    }

    fn terminate_block(&mut self, term: Terminator) {
        self.current_fn.blocks[self.current_block.0].terminator = Some(term);
    }

    fn lower_effect_block(
        &mut self,
        name: &str,
        stmts: &[Stmt],
        _expr_span: crate::ast::Span,
        binding_span: crate::ast::Span,
    ) -> Result<(), MirLowerError> {
        let mut locally_bound = HashSet::new();
        let mut free_vars = HashSet::new();
        validate_and_find_captures_in_stmts(stmts, &mut locally_bound, &mut free_vars)?;

        let mut sorted_vars: Vec<String> = free_vars.into_iter().collect();
        sorted_vars.sort();

        let mut captured_bindings = Vec::new();
        for var_name in sorted_vars {
            if self.lookup_thunk(&var_name).is_some() {
                return Err(MirLowerError::UnlowerableExpression(format!(
                    "Effect thunk '{var_name}' cannot be captured in nested effect"
                )));
            }
            if let Some(loc_id) = self.lookup_var(&var_name) {
                let loc_ty = &self.current_fn.locals[loc_id.0].ty;
                if *loc_ty != MirType::Int {
                    return Err(MirLowerError::UnlowerableExpression(format!(
                        "Unsupported capture type for '{var_name}': only read-only integers are supported, found {loc_ty:?}"
                    )));
                }
                let snapshot_local = self.current_fn.add_local(
                    Some(format!("__thunk_cap_{}_{}", name, var_name)),
                    MirType::Int,
                    false,
                    binding_span,
                );
                self.emit_stmt(Statement::StorageLive(snapshot_local, binding_span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(snapshot_local),
                    Rvalue::Use(Operand::Copy(Place::Local(loc_id))),
                    binding_span,
                ));
                captured_bindings.push((var_name, snapshot_local, MirType::Int));
            }
        }

        let thunk = LocalThunk {
            body: stmts.to_vec(),
            captured_bindings,
            result_type: MirType::Int,
            span: binding_span,
        };
        self.insert_thunk(name.to_string(), thunk);
        Ok(())
    }

    fn lower_effect_run_sync(
        &mut self,
        thunk: &LocalThunk,
        span: crate::ast::Span,
    ) -> Result<Operand, MirLowerError> {
        let result_local = self.current_fn.add_local(
            Some("__thunk_res".into()),
            thunk.result_type.clone(),
            true,
            span,
        );
        self.emit_stmt(Statement::StorageLive(result_local, span));

        let continuation_bb = self.current_fn.new_block();

        let saved_loop_stack = core::mem::take(&mut self.loop_stack);
        let saved_thunk_return_target = self.thunk_return_target;
        let saved_scopes = core::mem::take(&mut self.scopes);
        let saved_cleanups = core::mem::take(&mut self.cleanups);

        self.thunk_return_target = Some((result_local, continuation_bb));
        self.cleanups = vec![Vec::new()];

        let mut thunk_root_scope = HashMap::new();
        for (var_name, snapshot_local, _ty) in &thunk.captured_bindings {
            thunk_root_scope.insert(var_name.clone(), ScopeBinding::Local(*snapshot_local));
        }
        self.scopes = vec![thunk_root_scope];

        let inline_res = (|| {
            for stmt in &thunk.body {
                self.lower_statement(stmt)?;
            }
            if self.current_fn.blocks[self.current_block.0].terminator.is_none() {
                return Err(MirLowerError::UnlowerableExpression(
                    "Effect thunk must explicitly return on all paths; fallthrough is not supported".into(),
                ));
            }
            Ok(())
        })();

        self.loop_stack = saved_loop_stack;
        self.thunk_return_target = saved_thunk_return_target;
        self.scopes = saved_scopes;
        self.cleanups = saved_cleanups;

        inline_res?;

        self.current_block = continuation_bb;
        Ok(Operand::Move(Place::Local(result_local)))
    }
}

fn validate_and_find_captures_in_expr(
    expr: &Expr,
    locally_bound: &HashSet<String>,
    free_vars: &mut HashSet<String>,
) -> Result<(), MirLowerError> {
    match &expr.kind {
        ExprKind::Literal(_) | ExprKind::RegexLiteral { .. } => Ok(()),
        ExprKind::Identifier(name) => {
            if !locally_bound.contains(name) {
                free_vars.insert(name.clone());
            }
            Ok(())
        }
        ExprKind::Binary { left, right, .. } => {
            validate_and_find_captures_in_expr(left, locally_bound, free_vars)?;
            validate_and_find_captures_in_expr(right, locally_bound, free_vars)?;
            Ok(())
        }
        ExprKind::Unary { operand, .. } => {
            validate_and_find_captures_in_expr(operand, locally_bound, free_vars)
        }
        ExprKind::Call { callee, args, .. } => {
            validate_and_find_captures_in_expr(callee, locally_bound, free_vars)?;
            for a in args {
                validate_and_find_captures_in_expr(a, locally_bound, free_vars)?;
            }
            Ok(())
        }
        ExprKind::MethodCall { receiver, args, .. } => {
            validate_and_find_captures_in_expr(receiver, locally_bound, free_vars)?;
            for a in args {
                validate_and_find_captures_in_expr(a, locally_bound, free_vars)?;
            }
            Ok(())
        }
        ExprKind::FieldAccess { object, .. } => {
            validate_and_find_captures_in_expr(object, locally_bound, free_vars)
        }
        ExprKind::Index { object, index } => {
            validate_and_find_captures_in_expr(object, locally_bound, free_vars)?;
            validate_and_find_captures_in_expr(index, locally_bound, free_vars)?;
            Ok(())
        }
        ExprKind::Array(elems) | ExprKind::Tuple(elems) => {
            for e in elems {
                validate_and_find_captures_in_expr(e, locally_bound, free_vars)?;
            }
            Ok(())
        }
        ExprKind::Map(pairs) => {
            for (k, v) in pairs {
                validate_and_find_captures_in_expr(k, locally_bound, free_vars)?;
                validate_and_find_captures_in_expr(v, locally_bound, free_vars)?;
            }
            Ok(())
        }
        ExprKind::StructLiteral { fields, .. } => {
            for f in fields {
                validate_and_find_captures_in_expr(&f.value, locally_bound, free_vars)?;
            }
            Ok(())
        }
        ExprKind::EnumConstructor { args, .. } => {
            for a in args {
                validate_and_find_captures_in_expr(a, locally_bound, free_vars)?;
            }
            Ok(())
        }
        ExprKind::Lambda { .. } => Err(MirLowerError::UnlowerableExpression(
            "Lambdas are not supported inside effect thunk".into(),
        )),
        ExprKind::Paren(inner)
        | ExprKind::Cast { expr: inner, .. }
        | ExprKind::TypeCheck { expr: inner, .. } => {
            validate_and_find_captures_in_expr(inner, locally_bound, free_vars)
        }
        ExprKind::IsPattern { expr: inner, .. } => {
            validate_and_find_captures_in_expr(inner, locally_bound, free_vars)
        }
        ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            validate_and_find_captures_in_expr(condition, locally_bound, free_vars)?;
            validate_and_find_captures_in_expr(then_branch, locally_bound, free_vars)?;
            if let Some(eb) = else_branch {
                validate_and_find_captures_in_expr(eb, locally_bound, free_vars)?;
            }
            Ok(())
        }
        ExprKind::Block(stmts) => {
            let mut inner_bound = locally_bound.clone();
            validate_and_find_captures_in_stmts(stmts, &mut inner_bound, free_vars)
        }
        ExprKind::EffectBlock(_) => Err(MirLowerError::UnlowerableExpression(
            "Nested effect blocks are not supported".into(),
        )),
        ExprKind::Return(exprs) => {
            for e in exprs {
                validate_and_find_captures_in_expr(e, locally_bound, free_vars)?;
            }
            Ok(())
        }
        ExprKind::Range { start, end, .. } => {
            validate_and_find_captures_in_expr(start, locally_bound, free_vars)?;
            validate_and_find_captures_in_expr(end, locally_bound, free_vars)?;
            Ok(())
        }
    }
}

fn validate_and_find_captures_in_stmts(
    stmts: &[Stmt],
    locally_bound: &mut HashSet<String>,
    free_vars: &mut HashSet<String>,
) -> Result<(), MirLowerError> {
    for stmt in stmts {
        match &stmt.kind {
            StmtKind::Local {
                bindings,
                initializer,
            } => {
                if let Some(inits) = initializer {
                    for i in inits {
                        validate_and_find_captures_in_expr(i, locally_bound, free_vars)?;
                    }
                }
                for b in bindings {
                    locally_bound.insert(b.name.clone());
                }
            }
            StmtKind::Assign { targets, values } => {
                for v in values {
                    validate_and_find_captures_in_expr(v, locally_bound, free_vars)?;
                }
                for t in targets {
                    if let ExprKind::Identifier(name) = &t.kind {
                        if !locally_bound.contains(name) {
                            return Err(MirLowerError::UnlowerableExpression(format!(
                                "Write to captured variable '{name}' is not supported in effect thunk"
                            )));
                        }
                    } else {
                        validate_and_find_captures_in_expr(t, locally_bound, free_vars)?;
                    }
                }
            }
            StmtKind::CompoundAssign { target, op: _, value } => {
                validate_and_find_captures_in_expr(value, locally_bound, free_vars)?;
                if let ExprKind::Identifier(name) = &target.kind {
                    if !locally_bound.contains(name) {
                        return Err(MirLowerError::UnlowerableExpression(format!(
                            "Write to captured variable '{name}' is not supported in effect thunk"
                        )));
                    }
                } else {
                    validate_and_find_captures_in_expr(target, locally_bound, free_vars)?;
                }
            }
            StmtKind::Expr(e) => {
                validate_and_find_captures_in_expr(e, locally_bound, free_vars)?;
            }
            StmtKind::If {
                condition,
                then_block,
                elseif_branches,
                else_block,
            } => {
                validate_and_find_captures_in_expr(condition, locally_bound, free_vars)?;
                let mut then_bound = locally_bound.clone();
                validate_and_find_captures_in_stmts(then_block, &mut then_bound, free_vars)?;
                for (c, body) in elseif_branches {
                    validate_and_find_captures_in_expr(c, locally_bound, free_vars)?;
                    let mut elif_bound = locally_bound.clone();
                    validate_and_find_captures_in_stmts(body, &mut elif_bound, free_vars)?;
                }
                if let Some(eb) = else_block {
                    let mut else_bound = locally_bound.clone();
                    validate_and_find_captures_in_stmts(eb, &mut else_bound, free_vars)?;
                }
            }
            StmtKind::While { condition, body } => {
                validate_and_find_captures_in_expr(condition, locally_bound, free_vars)?;
                let mut body_bound = locally_bound.clone();
                validate_and_find_captures_in_stmts(body, &mut body_bound, free_vars)?;
            }
            StmtKind::ForNumeric {
                variable,
                start,
                end,
                step,
                body,
            } => {
                validate_and_find_captures_in_expr(start, locally_bound, free_vars)?;
                validate_and_find_captures_in_expr(end, locally_bound, free_vars)?;
                if let Some(s) = step {
                    validate_and_find_captures_in_expr(s, locally_bound, free_vars)?;
                }
                let mut body_bound = locally_bound.clone();
                body_bound.insert(variable.clone());
                validate_and_find_captures_in_stmts(body, &mut body_bound, free_vars)?;
            }
            StmtKind::ForIn {
                variables,
                iterator,
                body,
            } => {
                validate_and_find_captures_in_expr(iterator, locally_bound, free_vars)?;
                let mut body_bound = locally_bound.clone();
                for v in variables {
                    body_bound.insert(v.clone());
                }
                validate_and_find_captures_in_stmts(body, &mut body_bound, free_vars)?;
            }
            StmtKind::Return(exprs) => {
                if exprs.is_empty() {
                    return Err(MirLowerError::UnlowerableExpression(
                        "Effect thunk return must return an integer value".into(),
                    ));
                }
                for e in exprs {
                    validate_and_find_captures_in_expr(e, locally_bound, free_vars)?;
                }
            }
            StmtKind::Break | StmtKind::Continue => {}
            StmtKind::Block(inner) | StmtKind::Scoped(inner) => {
                let mut inner_bound = locally_bound.clone();
                validate_and_find_captures_in_stmts(inner, &mut inner_bound, free_vars)?;
            }
            StmtKind::Defer(_) | StmtKind::ErrDefer(_) => {
                return Err(MirLowerError::UnlowerableExpression(
                    "Cleanups (defer/errdefer) are not supported inside effect thunk".into(),
                ));
            }
        }
    }
    Ok(())
}

fn const_int_from_expr(expr: &Expr) -> Option<i64> {
    match &expr.kind {
        ExprKind::Literal(Literal::Integer(val)) => Some(*val),
        ExprKind::Paren(inner) => const_int_from_expr(inner),
        ExprKind::Unary {
            op: UnaryOp::Neg,
            operand,
        } => const_int_from_expr(operand).map(|v| -v),
        _ => None,
    }
}

pub fn mir_type_from_ast(ty: &Type) -> MirType {
    match &ty.kind {
        TypeKind::Int => MirType::Int,
        TypeKind::Float => MirType::Float,
        TypeKind::Bool => MirType::Bool,
        TypeKind::String => MirType::String,
        TypeKind::Unit => MirType::Unit,
        TypeKind::Named(s) => MirType::Struct(s.clone()),
        TypeKind::Tuple(types) => MirType::Tuple(types.iter().map(mir_type_from_ast).collect()),
        TypeKind::Array(inner) => MirType::Array(Box::new(mir_type_from_ast(inner))),
        TypeKind::Map(k, v) => {
            MirType::Map(Box::new(mir_type_from_ast(k)), Box::new(mir_type_from_ast(v)))
        }
        TypeKind::Option(inner) => match mir_type_from_ast(inner) {
            MirType::Struct(name) if !name.is_empty() => {
                MirType::Option(Box::new(MirType::Struct(name)))
            }
            _ => MirType::Enum("Option".to_string()),
        },
        TypeKind::Result(_, _) => MirType::Enum("Result".to_string()),
        TypeKind::Effect { success, error, .. } => MirType::Effect {
            return_type: Box::new(mir_type_from_ast(success)),
            error_type: Box::new(mir_type_from_ast(error)),
        },
        TypeKind::Typeof(s) => MirType::Struct(s.clone()),
        TypeKind::InferType(inner) | TypeKind::InferInput(inner) | TypeKind::InferOutput(inner) => {
            mir_type_from_ast(inner)
        }
        TypeKind::Schema { output, .. } => mir_type_from_ast(output),
        _ => MirType::Int,
    }
}

pub fn mir_binop_from_ast(op: BinaryOp) -> Result<MirBinaryOp, MirLowerError> {
    match op {
        BinaryOp::Add => Ok(MirBinaryOp::Add),
        BinaryOp::Sub => Ok(MirBinaryOp::Sub),
        BinaryOp::Mul => Ok(MirBinaryOp::Mul),
        BinaryOp::Div => Ok(MirBinaryOp::Div),
        BinaryOp::Mod => Ok(MirBinaryOp::Mod),
        BinaryOp::FloorDiv => Ok(MirBinaryOp::FloorDiv),
        BinaryOp::Pow => Ok(MirBinaryOp::Pow),
        BinaryOp::BitAnd => Ok(MirBinaryOp::BitAnd),
        BinaryOp::BitOr => Ok(MirBinaryOp::BitOr),
        BinaryOp::BitXor => Ok(MirBinaryOp::BitXor),
        BinaryOp::Shl => Ok(MirBinaryOp::Shl),
        BinaryOp::Shr => Ok(MirBinaryOp::Shr),
        BinaryOp::Eq => Ok(MirBinaryOp::Eq),
        BinaryOp::Ne => Ok(MirBinaryOp::NotEq),
        BinaryOp::Lt => Ok(MirBinaryOp::Lt),
        BinaryOp::Le => Ok(MirBinaryOp::LtEq),
        BinaryOp::Gt => Ok(MirBinaryOp::Gt),
        BinaryOp::Ge => Ok(MirBinaryOp::GtEq),
        BinaryOp::And => Ok(MirBinaryOp::And),
        BinaryOp::Or => Ok(MirBinaryOp::Or),
        BinaryOp::Concat => Ok(MirBinaryOp::Concat),
        _ => Err(MirLowerError::UnsupportedBinaryOp(op)),
    }
}
