use crate::ast::{
    BinaryOp, EnumDef, Expr, ExprKind, FunctionDef, Item, ItemKind, Literal, Stmt, StmtKind,
    StructDef, Type, TypeKind, UnaryOp,
};
use crate::data::layout::{CanonicalTypeTag, LogicalFieldDesc, LogicalLayoutV1};
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

#[derive(Debug, Clone)]
struct PendingMapGetCall {
    call_bb: BlockId,
    tmp_local: LocalId,
    assign_bb: BlockId,
    map_local: Option<LocalId>,
    key_local: Option<LocalId>,
    val_ty: MirType,
}

#[derive(Debug, Clone)]
struct PendingMapGetLocal {
    call_bb: BlockId,
    tmp_local: LocalId,
    assign_bb: BlockId,
    map_local: Option<LocalId>,
    key_local: Option<LocalId>,
    val_ty: MirType,
    var_name: String,
    exclusive_in_current_stmt: bool,
}

#[derive(Debug, Clone)]
enum ConditionBinding {
    Place {
        name: String,
        ty: MirType,
        place: Place,
        unwrap_option: bool,
    },
    CallUnwrap {
        name: String,
        ty: MirType,
        callee: String,
        args: Vec<Operand>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeQueueMethod {
    Offer,
    Take,
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
    enum_defs: HashMap<String, Vec<(String, Vec<MirType>)>>,
    scalar_enum_layouts: HashMap<String, (String, Vec<MirType>)>,
    struct_ops: Vec<MirStructOp>,
    local_field_names: HashMap<LocalId, Vec<String>>,
    known_typed_locals: HashSet<LocalId>,
    last_map_get_call: Option<PendingMapGetCall>,
    last_map_get_defs: HashMap<LocalId, PendingMapGetLocal>,
    field_call_counter: usize,
    thunk_return_target: Option<(LocalId, BlockId)>,
    lifted_functions: Option<&'a mut Vec<MirFunction>>,
    lambda_counter: Option<&'a mut usize>,
    lower_native_queue_intrinsics: bool,
    native_queue_locals: HashSet<LocalId>,
    native_queue_methods: HashMap<String, (LocalId, NativeQueueMethod)>,
    string_byte_int_only_locals: HashSet<String>,
}

impl<'a> AstLowering<'a> {
    pub fn new(func: &'a mut MirFunction) -> Self {
        Self::new_with_module_sink(func, None, None)
    }

    fn new_with_module_sink(
        func: &'a mut MirFunction,
        lifted_functions: Option<&'a mut Vec<MirFunction>>,
        lambda_counter: Option<&'a mut usize>,
    ) -> Self {
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
            enum_defs: HashMap::new(),
            scalar_enum_layouts: HashMap::new(),
            struct_ops: Vec::new(),
            local_field_names: HashMap::new(),
            known_typed_locals,
            last_map_get_call: None,
            last_map_get_defs: HashMap::new(),
            field_call_counter: 0,
            thunk_return_target: None,
            lifted_functions,
            lambda_counter,
            lower_native_queue_intrinsics: false,
            native_queue_locals: HashSet::new(),
            native_queue_methods: HashMap::new(),
            string_byte_int_only_locals: HashSet::new(),
        }
    }

    fn mir_type_for_ast_with_enums(
        ty: &Type,
        enum_defs: &HashMap<String, Vec<(String, Vec<MirType>)>>,
        scalar_enum_layouts: &HashMap<String, (String, Vec<MirType>)>,
    ) -> MirType {
        match &ty.kind {
            TypeKind::Int => MirType::Int,
            TypeKind::Float => MirType::Float,
            TypeKind::Bool => MirType::Bool,
            TypeKind::String => MirType::String,
            TypeKind::Unit => MirType::Unit,
            TypeKind::Unknown => MirType::Unknown,
            TypeKind::Named(s) => {
                if let Some((synth_name, _)) = scalar_enum_layouts.get(s) {
                    MirType::Struct(synth_name.clone())
                } else if enum_defs.contains_key(s) {
                    MirType::Enum(s.clone())
                } else {
                    MirType::Struct(s.clone())
                }
            }
            TypeKind::Tuple(types) => MirType::Tuple(
                types
                    .iter()
                    .map(|t| Self::mir_type_for_ast_with_enums(t, enum_defs, scalar_enum_layouts))
                    .collect(),
            ),
            TypeKind::Array(inner) => MirType::Array(Box::new(Self::mir_type_for_ast_with_enums(
                inner,
                enum_defs,
                scalar_enum_layouts,
            ))),
            TypeKind::Map(k, v) => MirType::Map(
                Box::new(Self::mir_type_for_ast_with_enums(
                    k,
                    enum_defs,
                    scalar_enum_layouts,
                )),
                Box::new(Self::mir_type_for_ast_with_enums(
                    v,
                    enum_defs,
                    scalar_enum_layouts,
                )),
            ),
            TypeKind::Function {
                params,
                return_type,
            } => MirType::Function {
                params: params
                    .iter()
                    .map(|t| Self::mir_type_for_ast_with_enums(t, enum_defs, scalar_enum_layouts))
                    .collect(),
                return_type: Box::new(Self::mir_type_for_ast_with_enums(
                    return_type,
                    enum_defs,
                    scalar_enum_layouts,
                )),
            },
            TypeKind::Option(inner) => MirType::Option(Box::new(
                Self::mir_type_for_ast_with_enums(inner, enum_defs, scalar_enum_layouts),
            )),
            TypeKind::Result(_, _) => MirType::Enum("Result".to_string()),
            TypeKind::Effect { success, error, .. } => MirType::Effect {
                return_type: Box::new(Self::mir_type_for_ast_with_enums(
                    success,
                    enum_defs,
                    scalar_enum_layouts,
                )),
                error_type: Box::new(Self::mir_type_for_ast_with_enums(
                    error,
                    enum_defs,
                    scalar_enum_layouts,
                )),
            },
            TypeKind::Typeof(s) => MirType::Struct(s.clone()),
            TypeKind::InferType(inner)
            | TypeKind::InferInput(inner)
            | TypeKind::InferOutput(inner) => {
                Self::mir_type_for_ast_with_enums(inner, enum_defs, scalar_enum_layouts)
            }
            TypeKind::Schema { output, .. } => {
                Self::mir_type_for_ast_with_enums(output, enum_defs, scalar_enum_layouts)
            }
            _ => MirType::Int,
        }
    }

    fn mir_type_for_ast(&self, ty: &Type) -> MirType {
        Self::mir_type_for_ast_with_enums(ty, &self.enum_defs, &self.scalar_enum_layouts)
    }

    fn try_build_scalar_enum_layout(
        enum_name: &str,
        variants: &[(String, Vec<MirType>)],
        type_id: u64,
    ) -> Option<(LogicalLayoutV1, Vec<MirType>)> {
        if variants.is_empty() {
            return None;
        }
        let max_arity = variants.iter().map(|(_, fs)| fs.len()).max().unwrap_or(0);
        if max_arity > 4 {
            return None;
        }
        let mut slot_types: Vec<Option<MirType>> = vec![None; max_arity];
        for (_, fields) in variants {
            for (idx, fty) in fields.iter().enumerate() {
                if !matches!(fty, MirType::Int | MirType::Bool | MirType::Float) {
                    return None;
                }
                match &slot_types[idx] {
                    None => slot_types[idx] = Some(fty.clone()),
                    Some(prev) if prev == fty => {}
                    Some(_) => return None,
                }
            }
        }
        let resolved_slots: Vec<MirType> = slot_types
            .into_iter()
            .map(|o| o.unwrap_or(MirType::Int))
            .collect();
        let synth_name = format!("__Enum_{enum_name}");
        let mut fields = Vec::with_capacity(1 + resolved_slots.len());
        fields.push(LogicalFieldDesc::new(0, "__tag", CanonicalTypeTag::Int64));
        for (i, st) in resolved_slots.iter().enumerate() {
            let tag = match st {
                MirType::Int => CanonicalTypeTag::Int64,
                MirType::Float => CanonicalTypeTag::Float64,
                MirType::Bool => CanonicalTypeTag::Bool,
                _ => return None,
            };
            fields.push(LogicalFieldDesc::new((i + 1) as u16, format!("__f{i}"), tag));
        }
        let layout = LogicalLayoutV1::new(type_id, synth_name, fields);
        Some((layout, resolved_slots))
    }

    fn scalar_enum_default_const(ty: &MirType) -> MirConstant {
        match ty {
            MirType::Int => MirConstant::Int(0),
            MirType::Float => MirConstant::Float(0.0),
            MirType::Bool => MirConstant::Bool(false),
            _ => MirConstant::Int(0),
        }
    }

    pub fn register_enum_def(&mut self, enum_def: &EnumDef) {
        let variants: Vec<(String, Vec<MirType>)> = enum_def
            .variants
            .iter()
            .map(|v| {
                let field_tys = v
                    .fields
                    .as_ref()
                    .map(|fs| fs.iter().map(mir_type_from_ast).collect())
                    .unwrap_or_default();
                (v.name.clone(), field_tys)
            })
            .collect();
        if let Some((layout, slot_types)) = Self::try_build_scalar_enum_layout(
            &enum_def.name,
            &variants,
            0xE000u64.wrapping_add(self.enum_defs.len() as u64),
        ) {
            let synth_name = layout.type_name.clone();
            let mut full_types = Vec::with_capacity(1 + slot_types.len());
            full_types.push(MirType::Int);
            full_types.extend(slot_types.iter().cloned());
            self.struct_field_types.insert(synth_name.clone(), full_types);
            self.struct_field_substructs
                .insert(synth_name.clone(), vec![None; 1 + slot_types.len()]);
            self.register_struct_layout(layout);
            self.scalar_enum_layouts
                .insert(enum_def.name.clone(), (synth_name, slot_types));
        }
        self.enum_defs.insert(enum_def.name.clone(), variants);
    }

    fn operand_is_explicit_unknown(&self, op: &Operand) -> bool {
        match op {
            Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => {
                self.current_fn.locals[loc.0].ty == MirType::Unknown
            }
            _ => false,
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
        Self::lower_module_items_inner(module_name, items, false)
    }

    pub fn lower_module_items_with_native_queues(
        module_name: impl Into<String>,
        items: &[Item],
    ) -> Result<MirModule, MirLowerError> {
        Self::lower_module_items_inner(module_name, items, true)
    }

    fn lower_module_items_inner(
        module_name: impl Into<String>,
        items: &[Item],
        lower_native_queue_intrinsics: bool,
    ) -> Result<MirModule, MirLowerError> {
        let mut module = MirModule::new(module_name.into());
        let mut defs = Vec::new();
        let mut enum_defs_ast: Vec<&EnumDef> = Vec::new();
        let mut next_enum_type_id = 0xE000u64;
        for item in items {
            match &item.kind {
                ItemKind::Struct(def) => {
                    let layout = struct_def_to_layout(def);
                    module.register_struct_layout(MirStructLayout::from(&layout));
                    defs.push(def);
                }
                ItemKind::Enum(enum_def) => {
                    let variants: Vec<(String, Vec<MirType>)> = enum_def
                        .variants
                        .iter()
                        .map(|v| {
                            let field_tys = v
                                .fields
                                .as_ref()
                                .map(|fs| fs.iter().map(mir_type_from_ast).collect())
                                .unwrap_or_default();
                            (v.name.clone(), field_tys)
                        })
                        .collect();
                    if let Some((layout, _)) = Self::try_build_scalar_enum_layout(
                        &enum_def.name,
                        &variants,
                        next_enum_type_id,
                    ) {
                        next_enum_type_id = next_enum_type_id.wrapping_add(1);
                        module.register_struct_layout(MirStructLayout::from(&layout));
                    }
                    module.register_enum_def(MirEnumDef {
                        name: enum_def.name.clone(),
                        variants,
                    });
                    enum_defs_ast.push(enum_def);
                }
                _ => {}
            }
        }
        let mut lifted_functions = Vec::new();
        let mut lambda_counter = 0usize;
        for item in items {
            match &item.kind {
                ItemKind::Function(func_def) => {
                    let (mir_fn, ops) = Self::lower_function_inner(
                        func_def,
                        None,
                        &defs,
                        &enum_defs_ast,
                        Some(&mut lifted_functions),
                        Some(&mut lambda_counter),
                        lower_native_queue_intrinsics,
                    )?;
                    module.add_function(mir_fn);
                    for op in ops {
                        module.record_struct_op(op);
                    }
                }
                ItemKind::Impl(impl_def) => {
                    let target_name = match &impl_def.target_type.kind {
                        TypeKind::Named(name) | TypeKind::GenericInstance { name, .. } => {
                            name.clone()
                        }
                        _ => format!("{}", impl_def.target_type),
                    };
                    for method_fn in &impl_def.methods {
                        let (mut mir_fn, ops) = Self::lower_function_inner(
                            method_fn,
                            Some(&target_name),
                            &defs,
                            &enum_defs_ast,
                            Some(&mut lifted_functions),
                            Some(&mut lambda_counter),
                            lower_native_queue_intrinsics,
                        )?;
                        if !mir_fn.signature.name.contains(':') {
                            mir_fn.signature.name = format!("{}:{}", target_name, method_fn.name);
                        }
                        module.add_function(mir_fn);
                        for op in ops {
                            module.record_struct_op(op);
                        }
                    }
                }
                _ => {}
            }
        }
        for lifted_fn in lifted_functions {
            module.add_function(lifted_fn);
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
        Self::lower_function_inner(func_def, owner_struct, struct_defs, &[], None, None, false)
    }

    pub fn lower_function_with_structs_enums_and_owner(
        func_def: &FunctionDef,
        owner_struct: Option<&str>,
        struct_defs: &[&StructDef],
        enum_defs: &[&EnumDef],
        lifted_functions: Option<&mut Vec<MirFunction>>,
        lambda_counter: Option<&mut usize>,
    ) -> Result<(MirFunction, Vec<MirStructOp>), MirLowerError> {
        Self::lower_function_inner(
            func_def,
            owner_struct,
            struct_defs,
            enum_defs,
            lifted_functions,
            lambda_counter,
            true,
        )
    }

    fn lower_function_inner(
        func_def: &FunctionDef,
        owner_struct: Option<&str>,
        struct_defs: &[&StructDef],
        enum_defs_ast: &[&EnumDef],
        lifted_functions: Option<&mut Vec<MirFunction>>,
        lambda_counter: Option<&mut usize>,
        lower_native_queue_intrinsics: bool,
    ) -> Result<(MirFunction, Vec<MirStructOp>), MirLowerError> {
        let mut enum_defs_map: HashMap<String, Vec<(String, Vec<MirType>)>> = HashMap::new();
        let mut scalar_enum_map: HashMap<String, (String, Vec<MirType>)> = HashMap::new();
        for (idx, edef) in enum_defs_ast.iter().enumerate() {
            let variants: Vec<(String, Vec<MirType>)> = edef
                .variants
                .iter()
                .map(|v| {
                    let field_tys = v
                        .fields
                        .as_ref()
                        .map(|fs| fs.iter().map(mir_type_from_ast).collect())
                        .unwrap_or_default();
                    (v.name.clone(), field_tys)
                })
                .collect();
            if let Some((layout, slot_types)) = Self::try_build_scalar_enum_layout(
                &edef.name,
                &variants,
                0xE000u64.wrapping_add(idx as u64),
            ) {
                scalar_enum_map.insert(edef.name.clone(), (layout.type_name, slot_types));
            }
            enum_defs_map.insert(edef.name.clone(), variants);
        }

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
                    Self::mir_type_for_ast_with_enums(&p.ty, &enum_defs_map, &scalar_enum_map)
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
                        Self::mir_type_for_ast_with_enums(rt, &enum_defs_map, &scalar_enum_map)
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

        let (ops, collected_layouts) = {
            let mut lowering =
                AstLowering::new_with_module_sink(&mut mir_fn, lifted_functions, lambda_counter);
            lowering.lower_native_queue_intrinsics = lower_native_queue_intrinsics;
            for def in struct_defs {
                lowering.register_struct_def(def);
            }
            for edef in enum_defs_ast {
                lowering.register_enum_def(edef);
            }
            lowering.lower_statements(&func_def.body)?;

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
            let mut layouts: Vec<MirStructLayout> = lowering
                .struct_layouts
                .values()
                .map(MirStructLayout::from)
                .collect();
            layouts.sort_by(|a, b| a.type_name.cmp(&b.type_name));
            (lowering.struct_ops, layouts)
        };
        mir_fn.struct_layouts = collected_layouts;

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
                let loc_ty = self.current_fn.locals.get(loc.0).map(|l| &l.ty);
                if let Some(names) = self.local_field_names.get(loc) {
                    if let Some(pos) = names.iter().position(|n| n == field) {
                        return Some(pos);
                    }
                    // Allow tuple locals (e.g. `t._0`) to fall through to Step 3 tuple resolution.
                    if !matches!(loc_ty, Some(MirType::Tuple(_))) {
                        return None;
                    }
                }
                loc_ty
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
            lowering.lower_statements(&func_def.body)?;

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
            lowering.lower_statements(stmts)?;

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

    fn lower_statements(&mut self, stmts: &[Stmt]) -> Result<(), MirLowerError> {
        for (idx, stmt) in stmts.iter().enumerate() {
            let remaining = &stmts[idx + 1..];
            if !self.last_map_get_defs.is_empty() {
                for pending in self.last_map_get_defs.values_mut() {
                    let count_in_cur = count_ident_in_stmt(stmt, &pending.var_name);
                    let used_later = stmts_mention_ident(remaining, &pending.var_name);
                    pending.exclusive_in_current_stmt = count_in_cur == 1 && !used_later;
                }
            }
            if let StmtKind::Local {
                bindings,
                initializer: Some(inits),
            } = &stmt.kind
                && bindings.len() == 1
                && inits.len() == 1
                && let Some(ref ann) = bindings[0].type_annotation
                && is_lua_value_array_type(ann)
                && is_string_byte_call(&inits[0])
                && array_local_only_matched_as_int(remaining, &bindings[0].name)
            {
                self.string_byte_int_only_locals
                    .insert(bindings[0].name.clone());
            }
            self.lower_statement(stmt)?;
        }
        Ok(())
    }

    fn emit_condition_bindings(
        &mut self,
        bindings: Vec<ConditionBinding>,
        span: crate::ast::Span,
    ) {
        for binding in bindings {
            match binding {
                ConditionBinding::Place {
                    name,
                    ty,
                    place,
                    unwrap_option,
                } => {
                    let binding_local =
                        self.current_fn
                            .add_local(Some(name.clone()), ty.clone(), false, span);
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
                ConditionBinding::CallUnwrap {
                    name,
                    ty,
                    callee,
                    args,
                } => {
                    let binding_local =
                        self.current_fn
                            .add_local(Some(name.clone()), ty.clone(), false, span);
                    if ty != MirType::Unknown {
                        self.known_typed_locals.insert(binding_local);
                    }
                    let succ_bb = self.current_fn.new_block();
                    self.emit_stmt(Statement::StorageLive(binding_local, span));
                    self.terminate_block(Terminator::Call {
                        callee,
                        args,
                        destination: Place::Local(binding_local),
                        success: succ_bb,
                        error: None,
                        span,
                    });
                    self.current_block = succ_bb;
                    self.insert_var(name, binding_local);
                }
            }
        }
    }

    pub fn lower_statement(&mut self, stmt: &Stmt) -> Result<(), MirLowerError> {
        if self.current_fn.blocks[self.current_block.0].terminator.is_some() {
            let dead_bb = self.current_fn.new_block();
            self.current_fn.blocks[dead_bb.0].terminator = Some(Terminator::Unreachable(stmt.span));
            self.current_block = dead_bb;
        }
        // Keep only pending map.get definitions that belong to the current basic block.
        let cur_bb = self.current_block;
        self.last_map_get_defs
            .retain(|_, p| p.assign_bb == cur_bb);

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

                let mut init_map_gets: Vec<Option<PendingMapGetCall>> = Vec::new();
                let mut init_queue_methods: Vec<Option<(LocalId, NativeQueueMethod)>> = Vec::new();
                let mut int_only_byte_flags: Vec<bool> = Vec::with_capacity(bindings.len());
                for b in bindings {
                    int_only_byte_flags.push(self.string_byte_int_only_locals.remove(&b.name));
                }
                let init_operands = if let Some(inits) = initializer {
                    let mut ops = Vec::with_capacity(inits.len());
                    for (idx, expr) in inits.iter().enumerate() {
                        if self.lower_native_queue_intrinsics
                            && let ExprKind::FieldAccess { object, field } = &expr.kind
                            && let ExprKind::Identifier(q_name) = &object.kind
                            && let Some(q_loc) = self.lookup_var(q_name)
                            && self.native_queue_locals.contains(&q_loc)
                            && let Some(kind) = match field.as_str() {
                                "offer" | "try_offer" => Some(NativeQueueMethod::Offer),
                                "take" | "try_take" => Some(NativeQueueMethod::Take),
                                _ => None,
                            }
                        {
                            self.last_map_get_call = None;
                            ops.push(Operand::Copy(Place::Local(q_loc)));
                            init_map_gets.push(None);
                            init_queue_methods.push(Some((q_loc, kind)));
                            continue;
                        }
                        let expected_ty = if int_only_byte_flags.get(idx).copied().unwrap_or(false) {
                            Some(MirType::Array(Box::new(MirType::Int)))
                        } else {
                            bindings
                                .get(idx)
                                .and_then(|b| b.type_annotation.as_ref())
                                .map(|t| self.mir_type_for_ast(t))
                        };
                        self.last_map_get_call = None;
                        let op = self.lower_expr_with_expected(expr, expected_ty.as_ref())?;
                        let pending_mg = self.last_map_get_call.take();
                        ops.push(op);
                        init_map_gets.push(pending_mg);
                        init_queue_methods.push(None);
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
                    let ty = if int_only_byte_flags.get(idx).copied().unwrap_or(false) {
                        MirType::Array(Box::new(MirType::Int))
                    } else if init_queue_methods.get(idx).and_then(|x| *x).is_some() {
                        MirType::Int
                    } else {
                        b.type_annotation
                            .as_ref()
                            .map(|t| self.mir_type_for_ast(t))
                            .unwrap_or(inferred_ty)
                    };
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
                        || int_only_byte_flags.get(idx).copied().unwrap_or(false)
                        || init_queue_methods.get(idx).and_then(|x| *x).is_some()
                        || (ty == MirType::Int
                            && (!has_init || matches!(init_op, Operand::Constant(MirConstant::Int(_)))));
                    let local_id = self.current_fn.add_local(
                        Some(b.name.clone()),
                        ty.clone(),
                        true,
                        b.span,
                    );
                    if is_known {
                        self.known_typed_locals.insert(local_id);
                    }
                    self.insert_var(b.name.clone(), local_id);
                    if let Some(Some((q_loc, kind))) = init_queue_methods.get(idx) {
                        self.native_queue_methods.insert(b.name.clone(), (*q_loc, *kind));
                    } else {
                        self.native_queue_methods.remove(&b.name);
                    }

                    if let Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src)) = &init_op {
                        if self.native_queue_locals.contains(src)
                            && init_queue_methods.get(idx).and_then(|x| *x).is_none()
                        {
                            self.native_queue_locals.insert(local_id);
                        }
                        if let Some(names) = self.local_field_names.get(src).cloned() {
                            self.local_field_names.insert(local_id, names);
                        }
                        if let Some(Some(mg)) = init_map_gets.get(idx)
                            && mg.tmp_local == *src
                            && mg.assign_bb == self.current_block
                        {
                            let val_ty = match (&mg.val_ty, &ty) {
                                (MirType::Unknown, MirType::Option(inner)) => (**inner).clone(),
                                (v, _) => v.clone(),
                            };
                            self.last_map_get_defs.insert(
                                local_id,
                                PendingMapGetLocal {
                                    call_bb: mg.call_bb,
                                    tmp_local: mg.tmp_local,
                                    assign_bb: mg.assign_bb,
                                    map_local: mg.map_local,
                                    key_local: mg.key_local,
                                    val_ty,
                                    var_name: b.name.clone(),
                                    exclusive_in_current_stmt: false,
                                },
                            );
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
                    if let Place::Local(mutated_loc) = &place {
                        self.last_map_get_defs.retain(|k, p| {
                            *k != *mutated_loc
                                && p.map_local != Some(*mutated_loc)
                                && p.key_local != Some(*mutated_loc)
                        });
                        if let ExprKind::Identifier(name) = &target.kind {
                            self.native_queue_methods.remove(name);
                        }
                        if self.native_queue_locals.remove(mutated_loc) {
                            self.native_queue_methods
                                .retain(|_, (ql, _)| *ql != *mutated_loc);
                        }
                    }
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
                if let Place::Local(mutated_loc) = &place {
                    self.last_map_get_defs.retain(|k, p| {
                        *k != *mutated_loc
                            && p.map_local != Some(*mutated_loc)
                            && p.key_local != Some(*mutated_loc)
                    });
                }
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
                let (cond_op, body_bindings, proven_bounds) =
                    self.lower_condition_with_bindings(condition, stmt.span)?;
                if let Some((base_local, idx_local)) = proven_bounds {
                    self.current_fn
                        .proven_in_bounds
                        .entry(body_block.0)
                        .or_default()
                        .push((base_local, idx_local));
                }
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
                self.emit_condition_bindings(body_bindings, stmt.span);
                self.lower_statements(body)?;
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
                self.lower_statements(body)?;
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
                self.lower_statements(stmts)?;
                self.pop_scope()?;
            }
            StmtKind::Defer(s) => {
                if let Some(cleanups) = self.cleanups.last_mut() {
                    cleanups.push(MirCleanup {
                        kind: MirCleanupKind::Defer,
                        stmt: s.clone(),
                    });
                }
                self.lower_cleanup_block(MirCleanupKind::Defer, s)?;
            }
            StmtKind::ErrDefer(s) => {
                if let Some(cleanups) = self.cleanups.last_mut() {
                    cleanups.push(MirCleanup {
                        kind: MirCleanupKind::ErrDefer,
                        stmt: s.clone(),
                    });
                }
                self.lower_cleanup_block(MirCleanupKind::ErrDefer, s)?;
            }
            StmtKind::Scoped(stmts) => {
                self.push_scope();
                self.lower_statements(stmts)?;
                self.pop_scope()?;
            }
        }
        Ok(())
    }

    fn lower_cleanup_block(
        &mut self,
        kind: MirCleanupKind,
        cleanup_stmt: &Stmt,
    ) -> Result<(), MirLowerError> {
        let saved_block = self.current_block;
        let saved_loop_stack = core::mem::take(&mut self.loop_stack);
        let cleanup_bb = self.current_fn.new_block();
        let start_block_idx = cleanup_bb.0;
        self.current_fn.blocks[cleanup_bb.0].is_cleanup = true;
        self.current_block = cleanup_bb;

        self.push_scope();
        let stmt_res = self.lower_statement(cleanup_stmt);
        let pop_res = self.pop_scope();
        let end_bb = self.current_block;
        if self.current_fn.blocks[end_bb.0].terminator.is_none() {
            self.current_fn.blocks[end_bb.0].terminator =
                Some(Terminator::Return(None, cleanup_stmt.span));
        }
        for block in &mut self.current_fn.blocks[start_block_idx..] {
            block.is_cleanup = true;
        }

        self.loop_stack = saved_loop_stack;
        self.current_block = saved_block;

        stmt_res?;
        pop_res?;

        self.current_fn.cleanup_blocks.push(MirCleanupEntry {
            kind,
            block: cleanup_bb,
        });
        Ok(())
    }

    /// Emits the canonical array bounds-check sequence for `0 <= idx_local < len(base_place)`
    /// using single-use `Move` temporaries so Cranelift (`NG-W14-8`) lowers it to a single
    /// unsigned comparison `(idx as u64) < (len as u64)` (`IntCC::UnsignedLessThan`), while
    /// preserving full MIR/VM/AOT structural compatibility.
    fn emit_array_in_bounds_check(
        &mut self,
        base_place: &Place,
        idx_local: LocalId,
        span: crate::ast::Span,
    ) -> LocalId {
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
                Operand::Move(Place::Local(len_temp)),
            ),
            span,
        ));

        let in_bounds_temp = self.current_fn.add_local(None, MirType::Bool, true, span);
        self.emit_stmt(Statement::StorageLive(in_bounds_temp, span));
        self.emit_stmt(Statement::Assign(
            Place::Local(in_bounds_temp),
            Rvalue::BinaryOp(
                MirBinaryOp::And,
                Operand::Move(Place::Local(ge_zero_temp)),
                Operand::Move(Place::Local(lt_len_temp)),
            ),
            span,
        ));
        in_bounds_temp
    }

    fn resolve_enum_variant_for_type(
        &self,
        scrutinee_ty: Option<&MirType>,
        explicit_enum_name: &str,
        variant: &str,
    ) -> Option<(String, usize, Vec<MirType>, bool)> {
        let candidate_enum = if !explicit_enum_name.is_empty() {
            Some(explicit_enum_name.to_string())
        } else {
            match scrutinee_ty {
                Some(MirType::Struct(sname)) if sname.starts_with("__Enum_") => {
                    Some(sname.trim_start_matches("__Enum_").to_string())
                }
                Some(MirType::Enum(ename)) => Some(ename.clone()),
                _ => None,
            }
        }?;
        let variants = self.enum_defs.get(&candidate_enum)?;
        let (vidx, (_, field_tys)) = variants
            .iter()
            .enumerate()
            .find(|(_, (vname, _))| vname == variant)?;
        let is_scalar = self.scalar_enum_layouts.contains_key(&candidate_enum);
        Some((candidate_enum, vidx, field_tys.clone(), is_scalar))
    }

    fn lower_enum_variant_condition(
        &mut self,
        scrutinee: &Expr,
        enum_name: &str,
        vidx: usize,
        field_tys: &[MirType],
        is_scalar: bool,
        bindings: &[crate::ast::Pattern],
        span: crate::ast::Span,
    ) -> Result<(Operand, Vec<ConditionBinding>, Option<(LocalId, LocalId)>), MirLowerError> {
        let base_place = if let Some(p) = self.lower_place(scrutinee)? {
            p
        } else {
            let op = self.lower_expr(scrutinee)?;
            let ty = if is_scalar {
                MirType::Struct(format!("__Enum_{enum_name}"))
            } else {
                MirType::Enum(enum_name.to_string())
            };
            let tmp = self.current_fn.add_local(None, ty, true, span);
            self.known_typed_locals.insert(tmp);
            self.emit_stmt(Statement::StorageLive(tmp, span));
            self.emit_stmt(Statement::Assign(Place::Local(tmp), Rvalue::Use(op), span));
            Place::Local(tmp)
        };

        if is_scalar {
            let tag_tmp = self.current_fn.add_local(None, MirType::Int, true, span);
            self.known_typed_locals.insert(tag_tmp);
            self.emit_stmt(Statement::StorageLive(tag_tmp, span));
            self.emit_stmt(Statement::Assign(
                Place::Local(tag_tmp),
                Rvalue::Use(Operand::Copy(Place::Field(Box::new(base_place.clone()), 0))),
                span,
            ));

            let cond_tmp = self.current_fn.add_local(None, MirType::Bool, true, span);
            self.emit_stmt(Statement::StorageLive(cond_tmp, span));
            self.emit_stmt(Statement::Assign(
                Place::Local(cond_tmp),
                Rvalue::BinaryOp(
                    MirBinaryOp::Eq,
                    Operand::Move(Place::Local(tag_tmp)),
                    Operand::Constant(MirConstant::Int(vidx as i64)),
                ),
                span,
            ));

            let mut out_bindings = Vec::new();
            for (idx, pat) in bindings.iter().enumerate() {
                if let crate::ast::Pattern::Identifier(bname) = pat {
                    let fty = field_tys.get(idx).cloned().unwrap_or(MirType::Int);
                    out_bindings.push(ConditionBinding::Place {
                        name: bname.clone(),
                        ty: fty,
                        place: Place::Field(Box::new(base_place.clone()), idx + 1),
                        unwrap_option: false,
                    });
                }
            }
            Ok((Operand::Move(Place::Local(cond_tmp)), out_bindings, None))
        } else {
            let tag_tmp = self.current_fn.add_local(None, MirType::Int, true, span);
            self.known_typed_locals.insert(tag_tmp);
            let succ_bb = self.current_fn.new_block();
            self.emit_stmt(Statement::StorageLive(tag_tmp, span));
            self.terminate_block(Terminator::Call {
                callee: "enum.tag".into(),
                args: vec![Operand::Copy(base_place.clone())],
                destination: Place::Local(tag_tmp),
                success: succ_bb,
                error: None,
                span,
            });
            self.current_block = succ_bb;

            let cond_tmp = self.current_fn.add_local(None, MirType::Bool, true, span);
            self.emit_stmt(Statement::StorageLive(cond_tmp, span));
            self.emit_stmt(Statement::Assign(
                Place::Local(cond_tmp),
                Rvalue::BinaryOp(
                    MirBinaryOp::Eq,
                    Operand::Move(Place::Local(tag_tmp)),
                    Operand::Constant(MirConstant::Int(vidx as i64)),
                ),
                span,
            ));

            let mut out_bindings = Vec::new();
            for (idx, pat) in bindings.iter().enumerate() {
                if let crate::ast::Pattern::Identifier(bname) = pat {
                    let fty = field_tys.get(idx).cloned().unwrap_or(MirType::Unknown);
                    out_bindings.push(ConditionBinding::CallUnwrap {
                        name: bname.clone(),
                        ty: fty,
                        callee: "enum.payload".into(),
                        args: vec![
                            Operand::Copy(base_place.clone()),
                            Operand::Constant(MirConstant::Int(idx as i64)),
                        ],
                    });
                }
            }
            Ok((Operand::Move(Place::Local(cond_tmp)), out_bindings, None))
        }
    }

    fn lower_condition_with_bindings(
        &mut self,
        condition: &Expr,
        span: crate::ast::Span,
    ) -> Result<
        (
            Operand,
            Vec<ConditionBinding>,
            Option<(LocalId, LocalId)>,
        ),
        MirLowerError,
    > {
        match &condition.kind {
            ExprKind::Paren(inner) => self.lower_condition_with_bindings(inner, span),
            ExprKind::IsPattern { expr: scrutinee, pattern } => {
                match pattern {
                    crate::ast::Pattern::Enum { enum_name, variant, bindings }
                        if (variant == "Ok" || variant == "Some") && enum_name.is_empty() =>
                    {
                        let indexed_parts = Self::extract_index_operands(scrutinee);
                        if let Some((object, index)) = indexed_parts {
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

                            let in_bounds_temp =
                                self.emit_array_in_bounds_check(&base_place, idx_local, span);

                            let proven_bounds = match &base_place {
                                Place::Local(base_local) => Some((*base_local, idx_local)),
                                _ => None,
                            };

                            let elem_place = Place::Index(Box::new(base_place), idx_local);
                            let mut out_bindings = Vec::new();
                            if let Some(crate::ast::Pattern::Identifier(binding_name)) = bindings.first() {
                                out_bindings.push(ConditionBinding::Place {
                                    name: binding_name.clone(),
                                    ty: elem_ty,
                                    place: elem_place,
                                    unwrap_option: false,
                                });
                            }

                            return Ok((
                                Operand::Move(Place::Local(in_bounds_temp)),
                                out_bindings,
                                proven_bounds,
                            ));
                        }

                        // Lower `if place is Some(binding)` on `Option<T>` (zero-boxing for Option<Struct>,
                        // general Option check for Option<Int|Float|Bool|String|...>).
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
                                if let Some(MirType::Option(inner)) = scrutinee_ty {
                                    let opt_ty = MirType::Option(inner.clone());
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
                                        out_bindings.push(ConditionBinding::Place {
                                            name: binding_name.clone(),
                                            ty: *inner,
                                            place: Place::Local(opt_tmp),
                                            unwrap_option: true,
                                        });
                                    }
                                    return Ok((Operand::Move(Place::Local(cond_tmp)), out_bindings, None));
                                }
                            }
                        }

                        Err(MirLowerError::UnlowerableExpression(format!(
                            "non-indexed is-pattern '{variant}' is not supported in MIR lowering"
                        )))
                    }
                    crate::ast::Pattern::Enum { enum_name, variant, bindings }
                        if variant == "None" && bindings.is_empty() && enum_name.is_empty() =>
                    {
                        let (scrutinee_op, scrutinee_ty) = if let Some(p) = self.lower_place(scrutinee)? {
                            let ty = self.get_place_mir_type(&p);
                            (Operand::Copy(p), ty)
                        } else {
                            let op = self.lower_expr(scrutinee)?;
                            let ty = Some(self.operand_definite_mir_type(&op));
                            (op, ty)
                        };
                        if let Some(MirType::Option(inner)) = scrutinee_ty {
                            let opt_ty = MirType::Option(inner);
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
                            return Ok((Operand::Move(Place::Local(cond_tmp)), Vec::new(), None));
                        }

                        Err(MirLowerError::UnlowerableExpression(
                            "non-indexed is-pattern 'None' is not supported in MIR lowering".to_string(),
                        ))
                    }
                    crate::ast::Pattern::Enum { enum_name, variant, bindings } => {
                        let scrutinee_ty = if let Some(p) = self.lower_place(scrutinee)? {
                            self.get_place_mir_type(&p)
                        } else {
                            self.peek_place_expected_type(scrutinee)
                        };

                        // 1. User enum pattern match (`if op is Lit(v)` / `if op is Add(a, b)` / `if op is Reset`)
                        if let Some((resolved_enum, vidx, field_tys, is_scalar)) =
                            self.resolve_enum_variant_for_type(scrutinee_ty.as_ref(), enum_name, variant)
                        {
                            return self.lower_enum_variant_condition(
                                scrutinee,
                                &resolved_enum,
                                vidx,
                                &field_tys,
                                is_scalar,
                                bindings,
                                span,
                            );
                        }

                        // 2a. Specialized scalar `MirType::Int` scrutinee (e.g. `string.byte`
                        //     specialized from `Array<LuaValue>` to `Array<int>` in `BENCH-NG-09`).
                        if enum_name.is_empty()
                            && bindings.len() <= 1
                            && scrutinee_ty == Some(MirType::Int)
                        {
                            match variant.as_str() {
                                "Int" => {
                                    let base_place = if let Some(p) = self.lower_place(scrutinee)? {
                                        p
                                    } else {
                                        let op = self.lower_expr(scrutinee)?;
                                        let tmp = self.current_fn.add_local(
                                            None,
                                            MirType::Int,
                                            true,
                                            span,
                                        );
                                        self.known_typed_locals.insert(tmp);
                                        self.emit_stmt(Statement::StorageLive(tmp, span));
                                        self.emit_stmt(Statement::Assign(
                                            Place::Local(tmp),
                                            Rvalue::Use(op),
                                            span,
                                        ));
                                        Place::Local(tmp)
                                    };
                                    let mut out_bindings = Vec::new();
                                    if let Some(crate::ast::Pattern::Identifier(bname)) =
                                        bindings.first()
                                    {
                                        out_bindings.push(ConditionBinding::Place {
                                            name: bname.clone(),
                                            ty: MirType::Int,
                                            place: base_place,
                                            unwrap_option: false,
                                        });
                                    }
                                    return Ok((
                                        Operand::Constant(MirConstant::Bool(true)),
                                        out_bindings,
                                        None,
                                    ));
                                }
                                "Float" | "String" | "Bool" | "Nil" => {
                                    return Ok((
                                        Operand::Constant(MirConstant::Bool(false)),
                                        Vec::new(),
                                        None,
                                    ));
                                }
                                _ => {}
                            }
                        }

                        // 2b. Primitive runtime type-tag pattern on `LuaValue` / `unknown`:
                        //     `if lv is Int(v)` / `Float(v)` / `String(v)` / `Bool(v)` (BENCH-NG-09)
                        let is_dynamic_scrutinee = matches!(
                            scrutinee_ty.as_ref(),
                            None | Some(MirType::Unknown)
                        ) || matches!(
                            scrutinee_ty.as_ref(),
                            Some(MirType::Struct(s)) if s == "LuaValue"
                        );
                        if is_dynamic_scrutinee && bindings.len() <= 1 {
                            let tag_info = match variant.as_str() {
                                "Int" => Some(("__mir_unknown_is_int", "__mir_unknown_unwrap_int", MirType::Int)),
                                "Float" => Some(("__mir_unknown_is_float", "__mir_unknown_unwrap_float", MirType::Float)),
                                "String" => Some(("__mir_unknown_is_string", "__mir_unknown_unwrap_string", MirType::String)),
                                "Bool" => Some(("__mir_unknown_is_bool", "__mir_unknown_unwrap_bool", MirType::Bool)),
                                _ => None,
                            };
                            if let Some((is_callee, unwrap_callee, bind_ty)) = tag_info {
                                let base_place = if let Some(p) = self.lower_place(scrutinee)? {
                                    p
                                } else {
                                    let op = self.lower_expr(scrutinee)?;
                                    let tmp = self.current_fn.add_local(None, MirType::Unknown, true, span);
                                    self.known_typed_locals.insert(tmp);
                                    self.emit_stmt(Statement::StorageLive(tmp, span));
                                    self.emit_stmt(Statement::Assign(Place::Local(tmp), Rvalue::Use(op), span));
                                    Place::Local(tmp)
                                };
                                let cond_tmp = self.current_fn.add_local(None, MirType::Bool, true, span);
                                self.known_typed_locals.insert(cond_tmp);
                                let succ_bb = self.current_fn.new_block();
                                self.emit_stmt(Statement::StorageLive(cond_tmp, span));
                                self.terminate_block(Terminator::Call {
                                    callee: is_callee.into(),
                                    args: vec![Operand::Copy(base_place.clone())],
                                    destination: Place::Local(cond_tmp),
                                    success: succ_bb,
                                    error: None,
                                    span,
                                });
                                self.current_block = succ_bb;

                                let mut out_bindings = Vec::new();
                                if let Some(crate::ast::Pattern::Identifier(bname)) = bindings.first() {
                                    out_bindings.push(ConditionBinding::CallUnwrap {
                                        name: bname.clone(),
                                        ty: bind_ty,
                                        callee: unwrap_callee.into(),
                                        args: vec![Operand::Copy(base_place)],
                                    });
                                }
                                return Ok((Operand::Move(Place::Local(cond_tmp)), out_bindings, None));
                            }
                        }

                        Err(MirLowerError::UnlowerableExpression(format!(
                            "unsupported is-pattern '{variant}' in MIR lowering"
                        )))
                    }
                    crate::ast::Pattern::Wildcard => {
                        Ok((Operand::Constant(MirConstant::Bool(true)), Vec::new(), None))
                    }
                    crate::ast::Pattern::Identifier(name) => {
                        let scrutinee_ty = if let Some(p) = self.lower_place(scrutinee)? {
                            self.get_place_mir_type(&p)
                        } else {
                            self.peek_place_expected_type(scrutinee)
                        };
                        if let Some((resolved_enum, vidx, field_tys, is_scalar)) =
                            self.resolve_enum_variant_for_type(scrutinee_ty.as_ref(), "", name)
                            && field_tys.is_empty()
                        {
                            return self.lower_enum_variant_condition(
                                scrutinee,
                                &resolved_enum,
                                vidx,
                                &field_tys,
                                is_scalar,
                                &[],
                                span,
                            );
                        }
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
                        Ok((
                            Operand::Constant(MirConstant::Bool(true)),
                            vec![ConditionBinding::Place {
                                name: name.clone(),
                                ty,
                                place,
                                unwrap_option: false,
                            }],
                            None,
                        ))
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
                        Ok((Operand::Move(Place::Local(temp)), Vec::new(), None))
                    }
                    _ => Err(MirLowerError::UnlowerableExpression(
                        "unsupported is-pattern in MIR lowering".to_string(),
                    )),
                }
            }
            ExprKind::TypeCheck { expr: scrutinee, check_type }
                if matches!(&check_type.kind, TypeKind::Named(_)) =>
            {
                let TypeKind::Named(type_name) = &check_type.kind else {
                    unreachable!();
                };
                if type_name == "None" {
                    let (scrutinee_op, scrutinee_ty) = if let Some(p) = self.lower_place(scrutinee)? {
                        let ty = self.get_place_mir_type(&p);
                        (Operand::Copy(p), ty)
                    } else {
                        let op = self.lower_expr(scrutinee)?;
                        let ty = Some(self.operand_definite_mir_type(&op));
                        (op, ty)
                    };
                    if let Some(MirType::Option(inner)) = scrutinee_ty {
                        let opt_ty = MirType::Option(inner);
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
                        return Ok((Operand::Move(Place::Local(cond_tmp)), Vec::new(), None));
                    }
                    return Err(MirLowerError::UnlowerableExpression(
                        "TypeCheck 'is None' on non-Option is not supported in MIR lowering"
                            .to_string(),
                    ));
                }

                // Unit enum variant check parsed as `TypeCheck` (e.g. `if op is Reset`)
                let scrutinee_ty = if let Some(p) = self.lower_place(scrutinee)? {
                    self.get_place_mir_type(&p)
                } else {
                    self.peek_place_expected_type(scrutinee)
                };
                if let Some((resolved_enum, vidx, field_tys, is_scalar)) =
                    self.resolve_enum_variant_for_type(scrutinee_ty.as_ref(), "", type_name)
                    && field_tys.is_empty()
                {
                    return self.lower_enum_variant_condition(
                        scrutinee,
                        &resolved_enum,
                        vidx,
                        &field_tys,
                        is_scalar,
                        &[],
                        span,
                    );
                }

                Err(MirLowerError::UnlowerableExpression(format!(
                    "TypeCheck 'is {type_name}' is not supported in MIR lowering"
                )))
            }
            _ => Ok((self.lower_expr(condition)?, Vec::new(), None)),
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
        let (cond_op, then_bindings, then_proven_bounds) =
            self.lower_condition_with_bindings(condition, span)?;
        let then_bb = self.current_fn.new_block();
        let merge_bb = self.current_fn.new_block();

        let next_cond_bb = if !elseif_branches.is_empty() || else_block.is_some() {
            self.current_fn.new_block()
        } else {
            merge_bb
        };

        if let Some((base_local, idx_local)) = then_proven_bounds {
            self.current_fn
                .proven_in_bounds
                .entry(then_bb.0)
                .or_default()
                .push((base_local, idx_local));
        }

        self.terminate_block(Terminator::Cond {
            cond: cond_op,
            then_block: then_bb,
            else_block: next_cond_bb,
            span,
        });

        // Lower Then Block
        self.current_block = then_bb;
        self.push_scope();
        self.emit_condition_bindings(then_bindings, span);
        self.lower_statements(then_block)?;
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
            let (elif_op, elif_bindings, elif_proven_bounds) =
                self.lower_condition_with_bindings(elif_cond, span)?;
            let elif_then = self.current_fn.new_block();
            let next_elif = if i + 1 < elseif_branches.len() || else_block.is_some() {
                self.current_fn.new_block()
            } else {
                merge_bb
            };

            if let Some((base_local, idx_local)) = elif_proven_bounds {
                self.current_fn
                    .proven_in_bounds
                    .entry(elif_then.0)
                    .or_default()
                    .push((base_local, idx_local));
            }

            self.terminate_block(Terminator::Cond {
                cond: elif_op,
                then_block: elif_then,
                else_block: next_elif,
                span,
            });

            self.current_block = elif_then;
            self.push_scope();
            self.emit_condition_bindings(elif_bindings, span);
            self.lower_statements(elif_body)?;
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
                self.lower_statements(eb)?;
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
                let any_unknown =
                    self.operand_is_explicit_unknown(&lhs) || self.operand_is_explicit_unknown(&rhs);
                let res_ty = match mir_op {
                    MirBinaryOp::Eq
                    | MirBinaryOp::NotEq
                    | MirBinaryOp::Lt
                    | MirBinaryOp::LtEq
                    | MirBinaryOp::Gt
                    | MirBinaryOp::GtEq => MirType::Bool,
                    MirBinaryOp::Concat => {
                        if any_unknown {
                            MirType::Unknown
                        } else {
                            MirType::String
                        }
                    }
                    _ if any_unknown => MirType::Unknown,
                    _ => MirType::Int,
                };
                let temp = self.current_fn.add_local(None, res_ty.clone(), true, expr.span);
                if res_ty == MirType::Unknown {
                    self.known_typed_locals.insert(temp);
                }
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
                let (cond_op, bindings, proven_bounds) =
                    self.lower_condition_with_bindings(condition, expr.span)?;
                let res_local = self.current_fn.add_local(None, MirType::Int, true, expr.span);
                self.emit_stmt(Statement::StorageLive(res_local, expr.span));

                let then_bb = self.current_fn.new_block();
                let else_bb = self.current_fn.new_block();
                let merge_bb = self.current_fn.new_block();

                if let Some((base_local, idx_local)) = proven_bounds {
                    self.current_fn
                        .proven_in_bounds
                        .entry(then_bb.0)
                        .or_default()
                        .push((base_local, idx_local));
                }

                self.terminate_block(Terminator::Cond {
                    cond: cond_op,
                    then_block: then_bb,
                    else_block: else_bb,
                    span: expr.span,
                });

                self.current_block = then_bb;
                self.push_scope();
                self.emit_condition_bindings(bindings, expr.span);
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
                let (cond_op, _, _) = self.lower_condition_with_bindings(expr, expr.span)?;
                Ok(cond_op)
            }
            ExprKind::TypeCheck { check_type, .. }
                if matches!(&check_type.kind, TypeKind::Named(n) if n == "None") =>
            {
                let (cond_op, _, _) = self.lower_condition_with_bindings(expr, expr.span)?;
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
                let mut unwrapped_recv = receiver.as_ref();
                while let ExprKind::Paren(inner) = &unwrapped_recv.kind {
                    unwrapped_recv = inner;
                }
                if let ExprKind::Call { callee, args: call_args, .. } = &unwrapped_recv.kind
                    && call_args.len() == 2
                {
                    match Self::call_callee_name(callee).as_deref() {
                        Some("array.get" | "table.get") => {
                            return self.lower_array_index_read(
                                &call_args[0],
                                &call_args[1],
                                expr.span,
                            );
                        }
                        Some("map.get") => {
                            return self.lower_expr(unwrapped_recv);
                        }
                        _ => {}
                    }
                }
                if Self::is_index_expr(receiver) {
                    return self.lower_expr(receiver);
                }
                if let ExprKind::Identifier(var_name) = &unwrapped_recv.kind
                    && let Some(loc_id) = self.lookup_var(var_name)
                    && let Some(fused_op) = self.try_fuse_pending_map_get_unwrap(loc_id, expected_ty)
                {
                    return Ok(fused_op);
                }
                let rec_op = self.lower_expr(receiver)?;
                let rec_ty = self.operand_definite_mir_type(&rec_op);
                match rec_ty {
                    MirType::Option(inner) => {
                        self.lower_option_unwrap(rec_op, *inner, expr.span)
                    }
                    MirType::Enum(ref ename) if ename == "Option" || ename == "Result" => {
                        let out_ty = expected_ty.cloned().unwrap_or(MirType::Unknown);
                        self.lower_enum_option_unwrap(rec_op, out_ty, expr.span)
                    }
                    MirType::Struct(struct_name)
                        if !struct_name.is_empty() && struct_name != "LuaValue" =>
                    {
                        self.lower_struct_method_call(
                            struct_name,
                            method,
                            rec_op,
                            args,
                            expr.span,
                        )
                    }
                    MirType::Unknown | MirType::Struct(_) => {
                        self.lower_unknown_method_call(rec_op, method, args, expr.span)
                    }
                    _ => Err(MirLowerError::UnlowerableExpression(format!(
                        "method call ':{method}' on non-index non-struct receiver is not supported in MIR lowering"
                    ))),
                }
            }
            ExprKind::MethodCall { receiver, method, args, .. } if method == "unwrap_or" && args.len() == 1 => {
                let mut unwrapped_recv = receiver.as_ref();
                while let ExprKind::Paren(inner) = &unwrapped_recv.kind {
                    unwrapped_recv = inner;
                }
                if let ExprKind::Call { callee, args: call_args, .. } = &unwrapped_recv.kind
                    && call_args.len() == 2
                    && Self::call_callee_name(callee).as_deref() == Some("map.get")
                {
                    let m_op = self.lower_expr(&call_args[0])?;
                    let k_op = self.lower_expr(&call_args[1])?;
                    let d_op = self.lower_expr_with_expected(&args[0], expected_ty)?;
                    let res_ty = match self.operand_definite_mir_type(&m_op) {
                        MirType::Map(_, v_ty) if *v_ty != MirType::Unknown => *v_ty,
                        _ => expected_ty
                            .cloned()
                            .unwrap_or_else(|| self.operand_definite_mir_type(&d_op)),
                    };
                    let temp = self.current_fn.add_local(None, res_ty.clone(), true, expr.span);
                    if res_ty != MirType::Unknown {
                        self.known_typed_locals.insert(temp);
                    }
                    let succ_bb = self.current_fn.new_block();
                    self.emit_stmt(Statement::StorageLive(temp, expr.span));
                    self.terminate_block(Terminator::Call {
                        callee: "map.get_unwrap_or".into(),
                        args: vec![m_op, k_op, d_op],
                        destination: Place::Local(temp),
                        success: succ_bb,
                        error: None,
                        span: expr.span,
                    });
                    self.current_block = succ_bb;
                    return Ok(Operand::Move(Place::Local(temp)));
                }
                let rec_op = self.lower_expr(receiver)?;
                let rec_ty = self.operand_definite_mir_type(&rec_op);
                match rec_ty {
                    MirType::Option(inner) => {
                        self.lower_option_unwrap_or(rec_op, *inner, &args[0], expr.span)
                    }
                    MirType::Enum(ref ename) if ename == "Option" || ename == "Result" => {
                        let out_ty = expected_ty.cloned().unwrap_or(MirType::Unknown);
                        self.lower_enum_option_unwrap_or(rec_op, out_ty, &args[0], expr.span)
                    }
                    MirType::Struct(struct_name)
                        if !struct_name.is_empty() && struct_name != "LuaValue" =>
                    {
                        self.lower_struct_method_call(
                            struct_name,
                            method,
                            rec_op,
                            args,
                            expr.span,
                        )
                    }
                    MirType::Unknown | MirType::Struct(_) => {
                        self.lower_unknown_method_call(rec_op, method, args, expr.span)
                    }
                    _ => Err(MirLowerError::UnlowerableExpression(format!(
                        "method call ':{method}' on non-option receiver is not supported in MIR lowering"
                    ))),
                }
            }
            ExprKind::MethodCall { receiver, method, args, .. }
                if matches!(method.as_str(), "is_some" | "is_none" | "is_ok" | "is_err")
                    && args.is_empty() =>
            {
                let rec_op = self.lower_expr(receiver)?;
                let rec_ty = self.operand_definite_mir_type(&rec_op);
                let positive = matches!(method.as_str(), "is_some" | "is_ok");
                match rec_ty {
                    MirType::Option(_) => {
                        let op = if positive {
                            MirBinaryOp::NotEq
                        } else {
                            MirBinaryOp::Eq
                        };
                        let temp = self.current_fn.add_local(None, MirType::Bool, true, expr.span);
                        self.known_typed_locals.insert(temp);
                        self.emit_stmt(Statement::StorageLive(temp, expr.span));
                        self.emit_stmt(Statement::Assign(
                            Place::Local(temp),
                            Rvalue::BinaryOp(op, rec_op, Operand::Constant(MirConstant::Int(0))),
                            expr.span,
                        ));
                        Ok(Operand::Move(Place::Local(temp)))
                    }
                    MirType::Enum(ref ename) if ename == "Option" || ename == "Result" => {
                        let tag_tmp = self.current_fn.add_local(None, MirType::Int, true, expr.span);
                        self.known_typed_locals.insert(tag_tmp);
                        let succ_bb = self.current_fn.new_block();
                        self.emit_stmt(Statement::StorageLive(tag_tmp, expr.span));
                        self.terminate_block(Terminator::Call {
                            callee: "enum.tag".into(),
                            args: vec![rec_op],
                            destination: Place::Local(tag_tmp),
                            success: succ_bb,
                            error: None,
                            span: expr.span,
                        });
                        self.current_block = succ_bb;
                        let cmp_op = if positive {
                            MirBinaryOp::Eq
                        } else {
                            MirBinaryOp::NotEq
                        };
                        let cond_tmp = self.current_fn.add_local(None, MirType::Bool, true, expr.span);
                        self.known_typed_locals.insert(cond_tmp);
                        self.emit_stmt(Statement::StorageLive(cond_tmp, expr.span));
                        self.emit_stmt(Statement::Assign(
                            Place::Local(cond_tmp),
                            Rvalue::BinaryOp(
                                cmp_op,
                                Operand::Move(Place::Local(tag_tmp)),
                                Operand::Constant(MirConstant::Int(0)),
                            ),
                            expr.span,
                        ));
                        Ok(Operand::Move(Place::Local(cond_tmp)))
                    }
                    MirType::Struct(struct_name)
                        if !struct_name.is_empty() && struct_name != "LuaValue" =>
                    {
                        self.lower_struct_method_call(
                            struct_name,
                            method,
                            rec_op,
                            args,
                            expr.span,
                        )
                    }
                    MirType::Unknown | MirType::Struct(_) => {
                        self.lower_unknown_method_call(rec_op, method, args, expr.span)
                    }
                    _ => Err(MirLowerError::UnlowerableExpression(format!(
                        "method call ':{method}' on non-option receiver is not supported in MIR lowering"
                    ))),
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
                if let Some(op) = self.try_lower_user_enum_constructor(enum_name, variant, args, expr.span)? {
                    return Ok(op);
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
                    if let Some(op) = self.try_lower_user_enum_constructor(name, field, &[], expr.span)? {
                        return Ok(op);
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
                let Some(field_idx) = self.resolve_field_idx(&base_place, field) else {
                    let base_ty = self.get_place_mir_type(&base_place);
                    let is_dynamic_base = matches!(
                        base_ty.as_ref(),
                        None | Some(MirType::Unknown)
                    ) || matches!(base_ty.as_ref(), Some(MirType::Struct(s)) if s == "LuaValue");
                    if is_dynamic_base {
                        let temp = self.current_fn.add_local(None, MirType::Unknown, true, expr.span);
                        self.known_typed_locals.insert(temp);
                        let succ_bb = self.current_fn.new_block();
                        self.emit_stmt(Statement::StorageLive(temp, expr.span));
                        self.terminate_block(Terminator::Call {
                            callee: "__mir_unknown_get_field".into(),
                            args: vec![
                                Operand::Copy(base_place),
                                Operand::Constant(MirConstant::String(field.clone())),
                            ],
                            destination: Place::Local(temp),
                            success: succ_bb,
                            error: None,
                            span: expr.span,
                        });
                        self.current_block = succ_bb;
                        return Ok(Operand::Move(Place::Local(temp)));
                    }
                    return Err(MirLowerError::UnlowerableExpression(format!(
                        "cannot resolve field '{field}' in MIR lowering"
                    )));
                };
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

                    let in_bounds_temp =
                        self.emit_array_in_bounds_check(&base_place, idx_local, expr.span);

                    let ok_bb = self.current_fn.new_block();
                    let err_bb = self.current_fn.new_block();

                    if let Place::Local(base_local) = &base_place {
                        self.current_fn
                            .proven_in_bounds
                            .entry(ok_bb.0)
                            .or_default()
                            .push((*base_local, idx_local));
                    }

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
                // Native integer queue lowering (`BENCH-NG-12` / `NG-W16-5`):
                // Lower `offer(val)` / `take()` (or `q.offer(val)` / `q.take()`) directly to
                // `queue.offer_int(q, val)` / `queue.take_int(q)` without `MirType::Unknown`.
                if self.lower_native_queue_intrinsics {
                    let native_q_call = match &callee.kind {
                        ExprKind::Identifier(name) => self.native_queue_methods.get(name).copied(),
                        ExprKind::FieldAccess { object, field } => {
                            if let ExprKind::Identifier(q_name) = &object.kind
                                && let Some(q_loc) = self.lookup_var(q_name)
                                && self.native_queue_locals.contains(&q_loc)
                            {
                                match field.as_str() {
                                    "offer" | "try_offer" => {
                                        Some((q_loc, NativeQueueMethod::Offer))
                                    }
                                    "take" | "try_take" => Some((q_loc, NativeQueueMethod::Take)),
                                    _ => None,
                                }
                            } else {
                                None
                            }
                        }
                        _ => None,
                    };
                    if let Some((q_loc, method)) = native_q_call {
                        let (intrinsic_callee, mut call_args) = match method {
                            NativeQueueMethod::Offer => (
                                "queue.offer_int",
                                vec![Operand::Copy(Place::Local(q_loc))],
                            ),
                            NativeQueueMethod::Take => (
                                "queue.take_int",
                                vec![Operand::Copy(Place::Local(q_loc))],
                            ),
                        };
                        for a in args {
                            call_args.push(self.lower_expr(a)?);
                        }
                        self.last_map_get_defs.clear();
                        let temp = self.current_fn.add_local(None, MirType::Int, true, expr.span);
                        self.known_typed_locals.insert(temp);
                        let succ_bb = self.current_fn.new_block();
                        self.emit_stmt(Statement::StorageLive(temp, expr.span));
                        self.terminate_block(Terminator::Call {
                            callee: intrinsic_callee.into(),
                            args: call_args,
                            destination: Place::Local(temp),
                            success: succ_bb,
                            error: None,
                            span: expr.span,
                        });
                        self.current_block = succ_bb;
                        return Ok(Operand::Move(Place::Local(temp)));
                    }
                }
                // Direct field-call on a local variable (e.g. `q.offer(x)` or `s.cb(x)`):
                // resolve field access into a named local and invoke it as a dynamic/closure call.
                if let ExprKind::FieldAccess { object, field } = &callee.kind {
                    let is_local_obj = match &object.kind {
                        ExprKind::Identifier(obj_name) => self.lookup_var(obj_name).is_some(),
                        _ => true,
                    };
                    if is_local_obj {
                        let fn_op = self.lower_expr(callee)?;
                        let fn_ty = self.operand_definite_mir_type(&fn_op);
                        let call_var_name = format!("__field_call_{}", self.field_call_counter);
                        self.field_call_counter += 1;
                        let fn_loc = self.current_fn.add_local(
                            Some(call_var_name.clone()),
                            fn_ty.clone(),
                            true,
                            expr.span,
                        );
                        self.known_typed_locals.insert(fn_loc);
                        self.emit_stmt(Statement::StorageLive(fn_loc, expr.span));
                        self.emit_stmt(Statement::Assign(
                            Place::Local(fn_loc),
                            Rvalue::Use(fn_op),
                            expr.span,
                        ));
                        let mut lowered_args = Vec::with_capacity(args.len());
                        for a in args {
                            lowered_args.push(self.lower_expr(a)?);
                        }
                        let ret_ty = match &fn_ty {
                            MirType::Function { return_type, .. } => (**return_type).clone(),
                            _ => expected_ty.cloned().unwrap_or(MirType::Unknown),
                        };
                        let temp = self.current_fn.add_local(None, ret_ty.clone(), true, expr.span);
                        if ret_ty != MirType::Unknown {
                            self.known_typed_locals.insert(temp);
                        }
                        let succ_bb = self.current_fn.new_block();
                        self.emit_stmt(Statement::StorageLive(temp, expr.span));
                        self.terminate_block(Terminator::Call {
                            callee: call_var_name,
                            args: lowered_args,
                            destination: Place::Local(temp),
                            success: succ_bb,
                            error: None,
                            span: expr.span,
                        });
                        self.current_block = succ_bb;
                        return Ok(Operand::Move(Place::Local(temp)));
                    }
                    if let ExprKind::Identifier(enum_name) = &object.kind
                        && let Some(op) =
                            self.try_lower_user_enum_constructor(enum_name, field, args, expr.span)?
                    {
                        return Ok(op);
                    }
                }
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
                if args.len() == 3 && matches!(callee_name.as_str(), "array.set" | "table.set") {
                    let arr_op = self.lower_expr(&args[0])?;
                    let is_map_target = callee_name == "table.set"
                        && matches!(self.operand_definite_mir_type(&arr_op), MirType::Map(_, _));
                    if !is_map_target {
                        let a_loc = match arr_op {
                            Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => {
                                loc
                            }
                            other => {
                                let arr_ty = match self.operand_definite_mir_type(&other) {
                                    MirType::Unknown => MirType::Array(Box::new(MirType::Unknown)),
                                    ty => ty,
                                };
                                let temp =
                                    self.current_fn.add_local(None, arr_ty, true, expr.span);
                                self.emit_stmt(Statement::StorageLive(temp, expr.span));
                                self.emit_stmt(Statement::Assign(
                                    Place::Local(temp),
                                    Rvalue::Use(other),
                                    expr.span,
                                ));
                                temp
                            }
                        };
                        let idx_op = self.lower_expr(&args[1])?;
                        let i_loc = match idx_op {
                            Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => {
                                loc
                            }
                            other => {
                                let temp = self.current_fn.add_local(
                                    None,
                                    MirType::Int,
                                    true,
                                    expr.span,
                                );
                                self.emit_stmt(Statement::StorageLive(temp, expr.span));
                                self.emit_stmt(Statement::Assign(
                                    Place::Local(temp),
                                    Rvalue::Use(other),
                                    expr.span,
                                ));
                                temp
                            }
                        };
                        let v_op = self.lower_expr(&args[2])?;
                        self.emit_stmt(Statement::Assign(
                            Place::Index(Box::new(Place::Local(a_loc)), i_loc),
                            Rvalue::Use(v_op),
                            expr.span,
                        ));
                        return Ok(Operand::Constant(MirConstant::Unit));
                    }
                    let idx_op = self.lower_expr(&args[1])?;
                    let v_op = self.lower_expr(&args[2])?;
                    let temp = self.current_fn.add_local(None, MirType::Unit, true, expr.span);
                    let succ_bb = self.current_fn.new_block();
                    self.emit_stmt(Statement::StorageLive(temp, expr.span));
                    self.terminate_block(Terminator::Call {
                        callee: callee_name,
                        args: vec![arr_op, idx_op, v_op],
                        destination: Place::Local(temp),
                        success: succ_bb,
                        error: None,
                        span: expr.span,
                    });
                    self.current_block = succ_bb;
                    return Ok(Operand::Move(Place::Local(temp)));
                }
                let mut lowered_args = Vec::with_capacity(args.len());
                for a in args {
                    lowered_args.push(self.lower_expr(a)?);
                }
                if self.lower_native_queue_intrinsics
                    && matches!(
                        callee_name.as_str(),
                        "Queue" | "sync.Queue" | "@std/sync.Queue"
                    )
                {
                    let cap_op = lowered_args
                        .into_iter()
                        .next()
                        .unwrap_or(Operand::Constant(MirConstant::Int(64)));
                    self.last_map_get_defs.clear();
                    let temp = self.current_fn.add_local(None, MirType::Int, true, expr.span);
                    self.known_typed_locals.insert(temp);
                    self.native_queue_locals.insert(temp);
                    let succ_bb = self.current_fn.new_block();
                    self.emit_stmt(Statement::StorageLive(temp, expr.span));
                    self.terminate_block(Terminator::Call {
                        callee: "queue.new".into(),
                        args: vec![cap_op],
                        destination: Place::Local(temp),
                        success: succ_bb,
                        error: None,
                        span: expr.span,
                    });
                    self.current_block = succ_bb;
                    return Ok(Operand::Move(Place::Local(temp)));
                }
                let is_map_get = callee_name == "map.get" && lowered_args.len() == 2;
                let map_get_val_ty = if is_map_get {
                    match self.operand_definite_mir_type(&lowered_args[0]) {
                        MirType::Map(_, v_ty) => (*v_ty).clone(),
                        _ => MirType::Unknown,
                    }
                } else {
                    MirType::Unknown
                };
                let local_callee_ty = if !callee_name.contains('.') && !callee_name.contains(':') {
                    self.lookup_var(&callee_name)
                        .map(|loc| self.current_fn.locals[loc.0].ty.clone())
                } else {
                    None
                };
                let (call_ret_ty, mark_known) = if callee_name == "tostring" {
                    (MirType::String, true)
                } else if callee_name == "string.byte" {
                    if let Some(MirType::Array(inner)) = expected_ty {
                        (MirType::Array(inner.clone()), true)
                    } else if matches!(expected_ty, Some(MirType::Int)) {
                        (MirType::Int, true)
                    } else {
                        (MirType::Int, false)
                    }
                } else if matches!(
                    callee_name.as_str(),
                    "Queue" | "sync.Queue" | "@std/sync.Queue"
                ) {
                    (MirType::Unknown, true)
                } else if let Some(MirType::Function { return_type, .. }) = local_callee_ty {
                    ((*return_type).clone(), true)
                } else if matches!(local_callee_ty, Some(MirType::Unknown)) {
                    (MirType::Unknown, true)
                } else if is_map_get {
                    if let Some(MirType::Option(inner)) = expected_ty {
                        (MirType::Option(inner.clone()), true)
                    } else {
                        (MirType::Int, false)
                    }
                } else {
                    (MirType::Int, false)
                };
                let temp = self.current_fn.add_local(None, call_ret_ty, true, expr.span);
                if mark_known {
                    self.known_typed_locals.insert(temp);
                }
                let call_bb = self.current_block;
                let succ_bb = self.current_fn.new_block();

                let map_get_meta = if is_map_get {
                    let map_local = match &lowered_args[0] {
                        Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l)) => Some(*l),
                        _ => None,
                    };
                    let key_local = match &lowered_args[1] {
                        Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l)) => Some(*l),
                        _ => None,
                    };
                    Some(PendingMapGetCall {
                        call_bb,
                        tmp_local: temp,
                        assign_bb: succ_bb,
                        map_local,
                        key_local,
                        val_ty: map_get_val_ty,
                    })
                } else {
                    None
                };

                // Any non-read call may mutate maps, so invalidate pending map.get fusions.
                if callee_name != "map.get" && callee_name != "map.has" && callee_name != "map.len" {
                    self.last_map_get_defs.clear();
                }

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
                self.last_map_get_call = map_get_meta;
                Ok(Operand::Move(Place::Local(temp)))
            }
            ExprKind::MethodCall { receiver, method, args, .. } => {
                let rec_op = self.lower_expr(receiver)?;
                match self.operand_definite_mir_type(&rec_op) {
                    MirType::Struct(struct_name)
                        if !struct_name.is_empty() && struct_name != "LuaValue" =>
                    {
                        self.lower_struct_method_call(struct_name, method, rec_op, args, expr.span)
                    }
                    MirType::Unknown | MirType::Struct(_) => {
                        self.lower_unknown_method_call(rec_op, method, args, expr.span)
                    }
                    _ => Err(MirLowerError::UnlowerableExpression(format!(
                        "method call ':{method}' on non-struct receiver is not supported in MIR lowering"
                    ))),
                }
            }
            ExprKind::Cast { expr: inner, target_type } => {
                let inner_op = self.lower_expr(inner)?;
                let mir_target_ty = self.mir_type_for_ast(target_type);
                let temp = self.current_fn.add_local(None, mir_target_ty.clone(), true, expr.span);
                self.emit_stmt(Statement::StorageLive(temp, expr.span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(temp),
                    Rvalue::Cast(inner_op, mir_target_ty),
                    expr.span,
                ));
                Ok(Operand::Move(Place::Local(temp)))
            }
            ExprKind::Map(entries) => {
                let map_ty = match expected_ty {
                    Some(MirType::Map(k, v)) => MirType::Map(k.clone(), v.clone()),
                    _ => MirType::Map(Box::new(MirType::Unknown), Box::new(MirType::Unknown)),
                };
                let map_local = self.current_fn.add_local(None, map_ty, true, expr.span);
                self.known_typed_locals.insert(map_local);
                let succ_bb = self.current_fn.new_block();
                self.emit_stmt(Statement::StorageLive(map_local, expr.span));
                self.terminate_block(Terminator::Call {
                    callee: "map.new".into(),
                    args: Vec::new(),
                    destination: Place::Local(map_local),
                    success: succ_bb,
                    error: None,
                    span: expr.span,
                });
                self.current_block = succ_bb;

                for (k_expr, v_expr) in entries {
                    let k_op = self.lower_expr(k_expr)?;
                    let v_op = self.lower_expr(v_expr)?;
                    let set_tmp = self.current_fn.add_local(None, MirType::Unit, true, expr.span);
                    let set_succ = self.current_fn.new_block();
                    self.emit_stmt(Statement::StorageLive(set_tmp, expr.span));
                    self.terminate_block(Terminator::Call {
                        callee: "map.set".into(),
                        args: vec![Operand::Copy(Place::Local(map_local)), k_op, v_op],
                        destination: Place::Local(set_tmp),
                        success: set_succ,
                        error: None,
                        span: expr.span,
                    });
                    self.current_block = set_succ;
                }
                Ok(Operand::Move(Place::Local(map_local)))
            }
            ExprKind::Lambda {
                params,
                return_type,
                body,
            } => self.lower_lambda(params, return_type.as_ref(), body, expr.span),
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

    fn call_callee_name(callee: &Expr) -> Option<String> {
        match &callee.kind {
            ExprKind::Identifier(name) => Some(name.clone()),
            ExprKind::FieldAccess { object, field } => {
                if let ExprKind::Identifier(obj_name) = &object.kind {
                    Some(format!("{obj_name}.{field}"))
                } else {
                    None
                }
            }
            ExprKind::Paren(inner) => Self::call_callee_name(inner),
            _ => None,
        }
    }

    fn extract_index_operands(expr: &Expr) -> Option<(&Expr, &Expr)> {
        match &expr.kind {
            ExprKind::Index { object, index } => Some((object.as_ref(), index.as_ref())),
            ExprKind::Call { callee, args, .. }
                if args.len() == 2
                    && matches!(
                        Self::call_callee_name(callee).as_deref(),
                        Some("array.get" | "table.get")
                    ) =>
            {
                Some((&args[0], &args[1]))
            }
            ExprKind::Paren(inner) => Self::extract_index_operands(inner),
            _ => None,
        }
    }

    fn is_index_expr(expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Index { .. } => true,
            ExprKind::Call { callee, args, .. }
                if args.len() == 2
                    && matches!(
                        Self::call_callee_name(callee).as_deref(),
                        Some("array.get" | "table.get")
                    ) =>
            {
                true
            }
            ExprKind::Paren(inner) => Self::is_index_expr(inner),
            _ => false,
        }
    }

    fn lower_array_index_read(
        &mut self,
        arr_expr: &Expr,
        idx_expr: &Expr,
        span: crate::ast::Span,
    ) -> Result<Operand, MirLowerError> {
        let arr_op = self.lower_expr(arr_expr)?;
        let a_loc = match arr_op {
            Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => loc,
            other => {
                let arr_ty = match self.operand_definite_mir_type(&other) {
                    MirType::Unknown => MirType::Array(Box::new(MirType::Unknown)),
                    ty => ty,
                };
                let temp = self.current_fn.add_local(None, arr_ty, true, span);
                self.emit_stmt(Statement::StorageLive(temp, span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(temp),
                    Rvalue::Use(other),
                    span,
                ));
                temp
            }
        };
        let idx_op = self.lower_expr(idx_expr)?;
        let i_loc = match idx_op {
            Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => loc,
            other => {
                let temp = self.current_fn.add_local(None, MirType::Int, true, span);
                self.emit_stmt(Statement::StorageLive(temp, span));
                self.emit_stmt(Statement::Assign(
                    Place::Local(temp),
                    Rvalue::Use(other),
                    span,
                ));
                temp
            }
        };
        let elem_ty = match &self.current_fn.locals[a_loc.0].ty {
            MirType::Array(inner) => (**inner).clone(),
            _ => MirType::Int,
        };
        let elem_place = Place::Index(Box::new(Place::Local(a_loc)), i_loc);
        let val_temp = self.current_fn.add_local(None, elem_ty, true, span);
        self.emit_stmt(Statement::StorageLive(val_temp, span));
        self.emit_stmt(Statement::Assign(
            Place::Local(val_temp),
            Rvalue::Use(Operand::Copy(elem_place)),
            span,
        ));
        Ok(Operand::Move(Place::Local(val_temp)))
    }

    fn lower_lambda(
        &mut self,
        params: &[(String, Option<Type>)],
        return_type: Option<&Type>,
        body: &Expr,
        span: crate::ast::Span,
    ) -> Result<Operand, MirLowerError> {
        let (Some(_), Some(_)) = (&self.lifted_functions, &self.lambda_counter) else {
            return Err(MirLowerError::UnlowerableExpression(
                "lambda lifting requires module-level MIR lowering".into(),
            ));
        };

        let mut locally_bound = HashSet::new();
        for (p_name, _) in params {
            locally_bound.insert(p_name.clone());
        }
        let mut free_vars = HashSet::new();
        collect_lambda_free_vars_in_expr(body, &locally_bound, &mut free_vars);

        let mut sorted_free: Vec<String> = free_vars.into_iter().collect();
        sorted_free.sort();

        let mut captures: Vec<(String, Operand, MirType)> = Vec::new();
        let mut lifted_param_names: Vec<String> = Vec::new();
        let mut lifted_param_types: Vec<MirType> = Vec::new();
        for var_name in sorted_free {
            if let Some(loc_id) = self.lookup_var(&var_name) {
                let cap_ty = self.current_fn.locals[loc_id.0].ty.clone();
                captures.push((
                    var_name.clone(),
                    Operand::Copy(Place::Local(loc_id)),
                    cap_ty.clone(),
                ));
                lifted_param_names.push(var_name);
                lifted_param_types.push(cap_ty);
            }
        }

        let mut lambda_param_types: Vec<MirType> = Vec::with_capacity(params.len());
        for (p_name, p_ty) in params {
            let mir_ty = p_ty
                .as_ref()
                .map(|t| self.mir_type_for_ast(t))
                .unwrap_or(MirType::Int);
            lambda_param_types.push(mir_ty.clone());
            lifted_param_names.push(p_name.clone());
            lifted_param_types.push(mir_ty);
        }

        let lambda_ret_ty = return_type
            .map(|t| self.mir_type_for_ast(t))
            .unwrap_or(MirType::Unknown);

        let lambda_id = {
            let counter = self.lambda_counter.as_deref_mut().unwrap();
            let id = *counter;
            *counter += 1;
            id
        };
        let func_name = format!("__lust_lambda_{lambda_id}");

        let sig = FunctionSignature {
            name: func_name.clone(),
            param_types: lifted_param_types,
            return_type: lambda_ret_ty.clone(),
            is_public: false,
            effects: None,
        };
        let mut lambda_fn = MirFunction::new(sig, span);
        for (i, p_name) in lifted_param_names.iter().enumerate() {
            let p_id = lambda_fn.params[i];
            lambda_fn.locals[p_id.0].name = Some(p_name.clone());
        }

        let struct_fields = self.struct_fields.clone();
        let struct_field_types = self.struct_field_types.clone();
        let struct_field_substructs = self.struct_field_substructs.clone();
        let struct_layouts = self.struct_layouts.clone();
        let enum_defs = self.enum_defs.clone();
        let scalar_enum_layouts = self.scalar_enum_layouts.clone();

        let nested_native_q = self.lower_native_queue_intrinsics;
        let (nested_ops, nested_layouts) = {
            let mut nested = AstLowering::new_with_module_sink(
                &mut lambda_fn,
                self.lifted_functions.as_deref_mut(),
                self.lambda_counter.as_deref_mut(),
            );
            nested.lower_native_queue_intrinsics = nested_native_q;
            nested.struct_fields = struct_fields;
            nested.struct_field_types = struct_field_types;
            nested.struct_field_substructs = struct_field_substructs;
            nested.struct_layouts = struct_layouts;
            nested.enum_defs = enum_defs;
            nested.scalar_enum_layouts = scalar_enum_layouts;

            match &body.kind {
                ExprKind::Block(stmts) => {
                    let has_value_ret = !matches!(lambda_ret_ty, MirType::Unit);
                    let len = stmts.len();
                    for (idx, stmt) in stmts.iter().enumerate() {
                        let is_last = idx + 1 == len;
                        if is_last
                            && has_value_ret
                            && let StmtKind::Expr(tail_expr) = &stmt.kind
                        {
                            let val_op = nested.lower_expr_with_expected(
                                tail_expr,
                                Some(&lambda_ret_ty),
                            )?;
                            if nested.current_fn.blocks[nested.current_block.0]
                                .terminator
                                .is_none()
                            {
                                nested.terminate_block(Terminator::Return(
                                    Some(val_op),
                                    tail_expr.span,
                                ));
                            }
                        } else {
                            nested.lower_statement(stmt)?;
                        }
                    }
                    let cur_block = nested.current_block;
                    if nested.current_fn.blocks[cur_block.0].terminator.is_none() {
                        if let Some(root_cleanups) = nested.cleanups.last().cloned() {
                            for c in root_cleanups.iter().rev() {
                                if matches!(c.kind, MirCleanupKind::Defer) {
                                    nested.lower_statement(&c.stmt)?;
                                }
                            }
                        }
                        let final_block = nested.current_block;
                        if nested.current_fn.blocks[final_block.0].terminator.is_none() {
                            nested.current_fn.blocks[final_block.0].terminator =
                                Some(Terminator::Return(None, span));
                        }
                    }
                }
                _ => {
                    let val_op =
                        nested.lower_expr_with_expected(body, Some(&lambda_ret_ty))?;
                    let cur_block = nested.current_block;
                    if nested.current_fn.blocks[cur_block.0].terminator.is_none() {
                        nested.terminate_block(Terminator::Return(Some(val_op), body.span));
                    }
                }
            }
            let mut layouts: Vec<MirStructLayout> = nested
                .struct_layouts
                .values()
                .map(MirStructLayout::from)
                .collect();
            layouts.sort_by(|a, b| a.type_name.cmp(&b.type_name));
            (nested.struct_ops, layouts)
        };
        lambda_fn.struct_layouts = nested_layouts;

        self.struct_ops.extend(nested_ops);
        if let Some(lifted) = self.lifted_functions.as_deref_mut() {
            lifted.push(lambda_fn);
        }

        let closure_ty = MirType::Function {
            params: lambda_param_types,
            return_type: Box::new(lambda_ret_ty),
        };
        let temp = self.current_fn.add_local(None, closure_ty, true, span);
        self.known_typed_locals.insert(temp);
        self.emit_stmt(Statement::StorageLive(temp, span));
        self.emit_stmt(Statement::Assign(
            Place::Local(temp),
            Rvalue::Closure {
                func_name,
                captures,
            },
            span,
        ));
        Ok(Operand::Move(Place::Local(temp)))
    }

    fn try_fuse_pending_map_get_unwrap(
        &mut self,
        loc_id: LocalId,
        expected_ty: Option<&MirType>,
    ) -> Option<Operand> {
        let pending = self.last_map_get_defs.get(&loc_id)?;
        if pending.assign_bb != self.current_block || !pending.exclusive_in_current_stmt {
            return None;
        }
        let pending = self.last_map_get_defs.remove(&loc_id)?;
        let resolved_val_ty = if pending.val_ty != MirType::Unknown {
            pending.val_ty
        } else if let Some(et) = expected_ty
            && !matches!(et, MirType::Option(_) | MirType::Unknown)
        {
            et.clone()
        } else {
            MirType::Int
        };
        if let Some(Terminator::Call { callee, .. }) =
            &mut self.current_fn.blocks[pending.call_bb.0].terminator
            && callee == "map.get"
        {
            *callee = "map.get_unwrap".into();
            self.current_fn.locals[pending.tmp_local.0].ty = resolved_val_ty.clone();
            self.current_fn.locals[loc_id.0].ty = resolved_val_ty.clone();
            if resolved_val_ty != MirType::Unknown {
                self.known_typed_locals.insert(pending.tmp_local);
                self.known_typed_locals.insert(loc_id);
            }
            return Some(Operand::Copy(Place::Local(loc_id)));
        }
        None
    }

    fn lower_option_unwrap(
        &mut self,
        rec_op: Operand,
        inner_ty: MirType,
        span: crate::ast::Span,
    ) -> Result<Operand, MirLowerError> {
        let opt_ty = MirType::Option(Box::new(inner_ty.clone()));
        let opt_tmp = self.current_fn.add_local(None, opt_ty, true, span);
        self.known_typed_locals.insert(opt_tmp);
        self.emit_stmt(Statement::StorageLive(opt_tmp, span));
        self.emit_stmt(Statement::Assign(
            Place::Local(opt_tmp),
            Rvalue::Use(rec_op),
            span,
        ));

        let is_some_tmp = self.current_fn.add_local(None, MirType::Bool, true, span);
        self.known_typed_locals.insert(is_some_tmp);
        self.emit_stmt(Statement::StorageLive(is_some_tmp, span));
        self.emit_stmt(Statement::Assign(
            Place::Local(is_some_tmp),
            Rvalue::BinaryOp(
                MirBinaryOp::NotEq,
                Operand::Copy(Place::Local(opt_tmp)),
                Operand::Constant(MirConstant::Int(0)),
            ),
            span,
        ));

        let ok_bb = self.current_fn.new_block();
        let err_bb = self.current_fn.new_block();
        let err_done_bb = self.current_fn.new_block();
        self.terminate_block(Terminator::Cond {
            cond: Operand::Move(Place::Local(is_some_tmp)),
            then_block: ok_bb,
            else_block: err_bb,
            span,
        });

        self.current_block = err_bb;
        let err_dst = self.current_fn.add_local(None, MirType::Unit, true, span);
        self.emit_stmt(Statement::StorageLive(err_dst, span));
        self.terminate_block(Terminator::Call {
            callee: "error".into(),
            args: vec![Operand::Constant(MirConstant::String(
                "called unwrap() on None".into(),
            ))],
            destination: Place::Local(err_dst),
            success: err_done_bb,
            error: None,
            span,
        });
        self.current_block = err_done_bb;
        self.terminate_block(Terminator::Unreachable(span));

        self.current_block = ok_bb;
        let res_local = self.current_fn.add_local(None, inner_ty.clone(), true, span);
        if inner_ty != MirType::Unknown {
            self.known_typed_locals.insert(res_local);
        }
        self.emit_stmt(Statement::StorageLive(res_local, span));
        self.emit_stmt(Statement::Assign(
            Place::Local(res_local),
            Rvalue::Cast(Operand::Copy(Place::Local(opt_tmp)), inner_ty),
            span,
        ));
        Ok(Operand::Move(Place::Local(res_local)))
    }

    fn lower_enum_option_unwrap(
        &mut self,
        rec_op: Operand,
        out_ty: MirType,
        span: crate::ast::Span,
    ) -> Result<Operand, MirLowerError> {
        let opt_ty = self.operand_definite_mir_type(&rec_op);
        let opt_tmp = self.current_fn.add_local(None, opt_ty, true, span);
        self.known_typed_locals.insert(opt_tmp);
        self.emit_stmt(Statement::StorageLive(opt_tmp, span));
        self.emit_stmt(Statement::Assign(
            Place::Local(opt_tmp),
            Rvalue::Use(rec_op),
            span,
        ));

        let tag_tmp = self.current_fn.add_local(None, MirType::Int, true, span);
        self.known_typed_locals.insert(tag_tmp);
        let after_tag_bb = self.current_fn.new_block();
        self.emit_stmt(Statement::StorageLive(tag_tmp, span));
        self.terminate_block(Terminator::Call {
            callee: "enum.tag".into(),
            args: vec![Operand::Copy(Place::Local(opt_tmp))],
            destination: Place::Local(tag_tmp),
            success: after_tag_bb,
            error: None,
            span,
        });
        self.current_block = after_tag_bb;

        let is_ok_tmp = self.current_fn.add_local(None, MirType::Bool, true, span);
        self.known_typed_locals.insert(is_ok_tmp);
        self.emit_stmt(Statement::StorageLive(is_ok_tmp, span));
        self.emit_stmt(Statement::Assign(
            Place::Local(is_ok_tmp),
            Rvalue::BinaryOp(
                MirBinaryOp::Eq,
                Operand::Move(Place::Local(tag_tmp)),
                Operand::Constant(MirConstant::Int(0)),
            ),
            span,
        ));

        let ok_bb = self.current_fn.new_block();
        let err_bb = self.current_fn.new_block();
        let err_done_bb = self.current_fn.new_block();
        self.terminate_block(Terminator::Cond {
            cond: Operand::Move(Place::Local(is_ok_tmp)),
            then_block: ok_bb,
            else_block: err_bb,
            span,
        });

        self.current_block = err_bb;
        let err_dst = self.current_fn.add_local(None, MirType::Unit, true, span);
        self.emit_stmt(Statement::StorageLive(err_dst, span));
        self.terminate_block(Terminator::Call {
            callee: "error".into(),
            args: vec![Operand::Constant(MirConstant::String(
                "called unwrap() on None/Err".into(),
            ))],
            destination: Place::Local(err_dst),
            success: err_done_bb,
            error: None,
            span,
        });
        self.current_block = err_done_bb;
        self.terminate_block(Terminator::Unreachable(span));

        self.current_block = ok_bb;
        let res_local = self.current_fn.add_local(None, out_ty.clone(), true, span);
        if out_ty != MirType::Unknown {
            self.known_typed_locals.insert(res_local);
        }
        let succ_bb = self.current_fn.new_block();
        self.emit_stmt(Statement::StorageLive(res_local, span));
        self.terminate_block(Terminator::Call {
            callee: "enum.payload".into(),
            args: vec![
                Operand::Copy(Place::Local(opt_tmp)),
                Operand::Constant(MirConstant::Int(0)),
            ],
            destination: Place::Local(res_local),
            success: succ_bb,
            error: None,
            span,
        });
        self.current_block = succ_bb;
        Ok(Operand::Move(Place::Local(res_local)))
    }

    fn lower_option_unwrap_or(
        &mut self,
        rec_op: Operand,
        inner_ty: MirType,
        default_expr: &Expr,
        span: crate::ast::Span,
    ) -> Result<Operand, MirLowerError> {
        let opt_ty = MirType::Option(Box::new(inner_ty.clone()));
        let opt_tmp = self.current_fn.add_local(None, opt_ty, true, span);
        self.known_typed_locals.insert(opt_tmp);
        self.emit_stmt(Statement::StorageLive(opt_tmp, span));
        self.emit_stmt(Statement::Assign(
            Place::Local(opt_tmp),
            Rvalue::Use(rec_op),
            span,
        ));

        let is_some_tmp = self.current_fn.add_local(None, MirType::Bool, true, span);
        self.known_typed_locals.insert(is_some_tmp);
        self.emit_stmt(Statement::StorageLive(is_some_tmp, span));
        self.emit_stmt(Statement::Assign(
            Place::Local(is_some_tmp),
            Rvalue::BinaryOp(
                MirBinaryOp::NotEq,
                Operand::Copy(Place::Local(opt_tmp)),
                Operand::Constant(MirConstant::Int(0)),
            ),
            span,
        ));

        let res_local = self.current_fn.add_local(None, inner_ty.clone(), true, span);
        if inner_ty != MirType::Unknown {
            self.known_typed_locals.insert(res_local);
        }
        self.emit_stmt(Statement::StorageLive(res_local, span));

        let then_bb = self.current_fn.new_block();
        let else_bb = self.current_fn.new_block();
        let merge_bb = self.current_fn.new_block();
        self.terminate_block(Terminator::Cond {
            cond: Operand::Move(Place::Local(is_some_tmp)),
            then_block: then_bb,
            else_block: else_bb,
            span,
        });

        self.current_block = then_bb;
        self.emit_stmt(Statement::Assign(
            Place::Local(res_local),
            Rvalue::Cast(Operand::Copy(Place::Local(opt_tmp)), inner_ty.clone()),
            span,
        ));
        self.terminate_block(Terminator::Goto(merge_bb, span));

        self.current_block = else_bb;
        let def_op = self.lower_expr_with_expected(default_expr, Some(&inner_ty))?;
        self.emit_stmt(Statement::Assign(
            Place::Local(res_local),
            Rvalue::Use(def_op),
            span,
        ));
        if self.current_fn.blocks[self.current_block.0].terminator.is_none() {
            self.terminate_block(Terminator::Goto(merge_bb, span));
        }

        self.current_block = merge_bb;
        Ok(Operand::Move(Place::Local(res_local)))
    }

    fn lower_enum_option_unwrap_or(
        &mut self,
        rec_op: Operand,
        out_ty: MirType,
        default_expr: &Expr,
        span: crate::ast::Span,
    ) -> Result<Operand, MirLowerError> {
        let opt_ty = self.operand_definite_mir_type(&rec_op);
        let opt_tmp = self.current_fn.add_local(None, opt_ty, true, span);
        self.known_typed_locals.insert(opt_tmp);
        self.emit_stmt(Statement::StorageLive(opt_tmp, span));
        self.emit_stmt(Statement::Assign(
            Place::Local(opt_tmp),
            Rvalue::Use(rec_op),
            span,
        ));

        let tag_tmp = self.current_fn.add_local(None, MirType::Int, true, span);
        self.known_typed_locals.insert(tag_tmp);
        let after_tag_bb = self.current_fn.new_block();
        self.emit_stmt(Statement::StorageLive(tag_tmp, span));
        self.terminate_block(Terminator::Call {
            callee: "enum.tag".into(),
            args: vec![Operand::Copy(Place::Local(opt_tmp))],
            destination: Place::Local(tag_tmp),
            success: after_tag_bb,
            error: None,
            span,
        });
        self.current_block = after_tag_bb;

        let is_ok_tmp = self.current_fn.add_local(None, MirType::Bool, true, span);
        self.known_typed_locals.insert(is_ok_tmp);
        self.emit_stmt(Statement::StorageLive(is_ok_tmp, span));
        self.emit_stmt(Statement::Assign(
            Place::Local(is_ok_tmp),
            Rvalue::BinaryOp(
                MirBinaryOp::Eq,
                Operand::Move(Place::Local(tag_tmp)),
                Operand::Constant(MirConstant::Int(0)),
            ),
            span,
        ));

        let res_local = self.current_fn.add_local(None, out_ty.clone(), true, span);
        if out_ty != MirType::Unknown {
            self.known_typed_locals.insert(res_local);
        }
        self.emit_stmt(Statement::StorageLive(res_local, span));

        let then_bb = self.current_fn.new_block();
        let else_bb = self.current_fn.new_block();
        let merge_bb = self.current_fn.new_block();
        self.terminate_block(Terminator::Cond {
            cond: Operand::Move(Place::Local(is_ok_tmp)),
            then_block: then_bb,
            else_block: else_bb,
            span,
        });

        self.current_block = then_bb;
        self.terminate_block(Terminator::Call {
            callee: "enum.payload".into(),
            args: vec![
                Operand::Copy(Place::Local(opt_tmp)),
                Operand::Constant(MirConstant::Int(0)),
            ],
            destination: Place::Local(res_local),
            success: merge_bb,
            error: None,
            span,
        });

        self.current_block = else_bb;
        let def_op = self.lower_expr_with_expected(default_expr, Some(&out_ty))?;
        self.emit_stmt(Statement::Assign(
            Place::Local(res_local),
            Rvalue::Use(def_op),
            span,
        ));
        if self.current_fn.blocks[self.current_block.0].terminator.is_none() {
            self.terminate_block(Terminator::Goto(merge_bb, span));
        }

        self.current_block = merge_bb;
        Ok(Operand::Move(Place::Local(res_local)))
    }

    fn try_lower_user_enum_constructor(
        &mut self,
        enum_name: &str,
        variant: &str,
        args: &[Expr],
        span: crate::ast::Span,
    ) -> Result<Option<Operand>, MirLowerError> {
        let Some(variants) = self.enum_defs.get(enum_name).cloned() else {
            return Ok(None);
        };
        let Some((vidx, (_, field_tys))) = variants
            .iter()
            .enumerate()
            .find(|(_, (vname, _))| vname == variant)
        else {
            return Ok(None);
        };

        let mut lowered_args = Vec::with_capacity(args.len());
        for (i, a) in args.iter().enumerate() {
            let expected = field_tys.get(i);
            lowered_args.push(self.lower_expr_with_expected(a, expected)?);
        }

        if let Some((synth_name, slot_types)) = self.scalar_enum_layouts.get(enum_name).cloned() {
            let mut elems = Vec::with_capacity(1 + slot_types.len());
            elems.push(Operand::Constant(MirConstant::Int(vidx as i64)));
            for (i, st) in slot_types.iter().enumerate() {
                if let Some(arg_op) = lowered_args.get(i) {
                    elems.push(arg_op.clone());
                } else {
                    elems.push(Operand::Constant(Self::scalar_enum_default_const(st)));
                }
            }
            let struct_ty = MirType::Struct(synth_name);
            let temp = self.current_fn.add_local(None, struct_ty.clone(), true, span);
            self.known_typed_locals.insert(temp);
            self.emit_stmt(Statement::StorageLive(temp, span));
            self.emit_stmt(Statement::Assign(
                Place::Local(temp),
                Rvalue::Aggregate(struct_ty, elems),
                span,
            ));
            return Ok(Some(Operand::Move(Place::Local(temp))));
        }

        let enum_ty = MirType::Enum(enum_name.to_string());
        let temp = self.current_fn.add_local(None, enum_ty, true, span);
        self.known_typed_locals.insert(temp);
        let succ_bb = self.current_fn.new_block();
        self.emit_stmt(Statement::StorageLive(temp, span));
        self.terminate_block(Terminator::Call {
            callee: format!("{enum_name}.{variant}"),
            args: lowered_args,
            destination: Place::Local(temp),
            success: succ_bb,
            error: None,
            span,
        });
        self.current_block = succ_bb;
        Ok(Some(Operand::Move(Place::Local(temp))))
    }

    fn lower_unknown_method_call(
        &mut self,
        rec_op: Operand,
        method: &str,
        args: &[Expr],
        span: crate::ast::Span,
    ) -> Result<Operand, MirLowerError> {
        let mut all_args = Vec::with_capacity(2 + args.len());
        all_args.push(rec_op);
        all_args.push(Operand::Constant(MirConstant::String(method.to_string())));
        for a in args {
            all_args.push(self.lower_expr(a)?);
        }
        let temp = self.current_fn.add_local(None, MirType::Unknown, true, span);
        self.known_typed_locals.insert(temp);
        let succ_bb = self.current_fn.new_block();
        self.emit_stmt(Statement::StorageLive(temp, span));
        self.terminate_block(Terminator::Call {
            callee: "__mir_unknown_call_method".into(),
            args: all_args,
            destination: Place::Local(temp),
            success: succ_bb,
            error: None,
            span,
        });
        self.current_block = succ_bb;
        Ok(Operand::Move(Place::Local(temp)))
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

fn collect_pattern_bindings(pattern: &crate::ast::Pattern, bound: &mut HashSet<String>) {
    match pattern {
        crate::ast::Pattern::Identifier(name) => {
            bound.insert(name.clone());
        }
        crate::ast::Pattern::Enum { bindings, .. } => {
            for b in bindings {
                collect_pattern_bindings(b, bound);
            }
        }
        crate::ast::Pattern::Struct { fields, .. } => {
            for (_, p) in fields {
                collect_pattern_bindings(p, bound);
            }
        }
        _ => {}
    }
}

fn collect_lambda_free_vars_in_expr(
    expr: &Expr,
    locally_bound: &HashSet<String>,
    free_vars: &mut HashSet<String>,
) {
    match &expr.kind {
        ExprKind::Literal(_) | ExprKind::RegexLiteral { .. } => {}
        ExprKind::Identifier(name) => {
            if !locally_bound.contains(name) {
                free_vars.insert(name.clone());
            }
        }
        ExprKind::Binary { left, right, .. } => {
            collect_lambda_free_vars_in_expr(left, locally_bound, free_vars);
            collect_lambda_free_vars_in_expr(right, locally_bound, free_vars);
        }
        ExprKind::Unary { operand, .. } => {
            collect_lambda_free_vars_in_expr(operand, locally_bound, free_vars);
        }
        ExprKind::Call { callee, args, .. } => {
            collect_lambda_free_vars_in_expr(callee, locally_bound, free_vars);
            for a in args {
                collect_lambda_free_vars_in_expr(a, locally_bound, free_vars);
            }
        }
        ExprKind::MethodCall { receiver, args, .. } => {
            collect_lambda_free_vars_in_expr(receiver, locally_bound, free_vars);
            for a in args {
                collect_lambda_free_vars_in_expr(a, locally_bound, free_vars);
            }
        }
        ExprKind::FieldAccess { object, .. } => {
            collect_lambda_free_vars_in_expr(object, locally_bound, free_vars);
        }
        ExprKind::Index { object, index } => {
            collect_lambda_free_vars_in_expr(object, locally_bound, free_vars);
            collect_lambda_free_vars_in_expr(index, locally_bound, free_vars);
        }
        ExprKind::Array(elems) | ExprKind::Tuple(elems) => {
            for e in elems {
                collect_lambda_free_vars_in_expr(e, locally_bound, free_vars);
            }
        }
        ExprKind::Map(pairs) => {
            for (k, v) in pairs {
                collect_lambda_free_vars_in_expr(k, locally_bound, free_vars);
                collect_lambda_free_vars_in_expr(v, locally_bound, free_vars);
            }
        }
        ExprKind::StructLiteral { fields, .. } => {
            for f in fields {
                collect_lambda_free_vars_in_expr(&f.value, locally_bound, free_vars);
            }
        }
        ExprKind::EnumConstructor { args, .. } => {
            for a in args {
                collect_lambda_free_vars_in_expr(a, locally_bound, free_vars);
            }
        }
        ExprKind::Lambda { params, body, .. } => {
            let mut inner_bound = locally_bound.clone();
            for (p_name, _) in params {
                inner_bound.insert(p_name.clone());
            }
            collect_lambda_free_vars_in_expr(body, &inner_bound, free_vars);
        }
        ExprKind::Paren(inner)
        | ExprKind::Cast { expr: inner, .. }
        | ExprKind::TypeCheck { expr: inner, .. }
        | ExprKind::IsPattern { expr: inner, .. } => {
            collect_lambda_free_vars_in_expr(inner, locally_bound, free_vars);
        }
        ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_lambda_free_vars_in_expr(condition, locally_bound, free_vars);
            let mut then_bound = locally_bound.clone();
            if let ExprKind::IsPattern { pattern, .. } = &condition.kind {
                collect_pattern_bindings(pattern, &mut then_bound);
            }
            collect_lambda_free_vars_in_expr(then_branch, &then_bound, free_vars);
            if let Some(eb) = else_branch {
                collect_lambda_free_vars_in_expr(eb, locally_bound, free_vars);
            }
        }
        ExprKind::Block(stmts) | ExprKind::EffectBlock(stmts) => {
            let mut inner_bound = locally_bound.clone();
            collect_lambda_free_vars_in_stmts(stmts, &mut inner_bound, free_vars);
        }
        ExprKind::Return(exprs) => {
            for e in exprs {
                collect_lambda_free_vars_in_expr(e, locally_bound, free_vars);
            }
        }
        ExprKind::Range { start, end, .. } => {
            collect_lambda_free_vars_in_expr(start, locally_bound, free_vars);
            collect_lambda_free_vars_in_expr(end, locally_bound, free_vars);
        }
    }
}

fn collect_lambda_free_vars_in_stmts(
    stmts: &[Stmt],
    locally_bound: &mut HashSet<String>,
    free_vars: &mut HashSet<String>,
) {
    for stmt in stmts {
        match &stmt.kind {
            StmtKind::Local {
                bindings,
                initializer,
            } => {
                if let Some(inits) = initializer {
                    for i in inits {
                        collect_lambda_free_vars_in_expr(i, locally_bound, free_vars);
                    }
                }
                for b in bindings {
                    locally_bound.insert(b.name.clone());
                }
            }
            StmtKind::Assign { targets, values } => {
                for v in values {
                    collect_lambda_free_vars_in_expr(v, locally_bound, free_vars);
                }
                for t in targets {
                    collect_lambda_free_vars_in_expr(t, locally_bound, free_vars);
                }
            }
            StmtKind::CompoundAssign { target, value, .. } => {
                collect_lambda_free_vars_in_expr(value, locally_bound, free_vars);
                collect_lambda_free_vars_in_expr(target, locally_bound, free_vars);
            }
            StmtKind::Expr(e) => {
                collect_lambda_free_vars_in_expr(e, locally_bound, free_vars);
            }
            StmtKind::If {
                condition,
                then_block,
                elseif_branches,
                else_block,
            } => {
                collect_lambda_free_vars_in_expr(condition, locally_bound, free_vars);
                let mut then_bound = locally_bound.clone();
                if let ExprKind::IsPattern { pattern, .. } = &condition.kind {
                    collect_pattern_bindings(pattern, &mut then_bound);
                }
                collect_lambda_free_vars_in_stmts(then_block, &mut then_bound, free_vars);
                for (c, body) in elseif_branches {
                    collect_lambda_free_vars_in_expr(c, locally_bound, free_vars);
                    let mut elif_bound = locally_bound.clone();
                    if let ExprKind::IsPattern { pattern, .. } = &c.kind {
                        collect_pattern_bindings(pattern, &mut elif_bound);
                    }
                    collect_lambda_free_vars_in_stmts(body, &mut elif_bound, free_vars);
                }
                if let Some(eb) = else_block {
                    let mut else_bound = locally_bound.clone();
                    collect_lambda_free_vars_in_stmts(eb, &mut else_bound, free_vars);
                }
            }
            StmtKind::While { condition, body } => {
                collect_lambda_free_vars_in_expr(condition, locally_bound, free_vars);
                let mut body_bound = locally_bound.clone();
                if let ExprKind::IsPattern { pattern, .. } = &condition.kind {
                    collect_pattern_bindings(pattern, &mut body_bound);
                }
                collect_lambda_free_vars_in_stmts(body, &mut body_bound, free_vars);
            }
            StmtKind::ForNumeric {
                variable,
                start,
                end,
                step,
                body,
            } => {
                collect_lambda_free_vars_in_expr(start, locally_bound, free_vars);
                collect_lambda_free_vars_in_expr(end, locally_bound, free_vars);
                if let Some(s) = step {
                    collect_lambda_free_vars_in_expr(s, locally_bound, free_vars);
                }
                let mut body_bound = locally_bound.clone();
                body_bound.insert(variable.clone());
                collect_lambda_free_vars_in_stmts(body, &mut body_bound, free_vars);
            }
            StmtKind::ForIn {
                variables,
                iterator,
                body,
            } => {
                collect_lambda_free_vars_in_expr(iterator, locally_bound, free_vars);
                let mut body_bound = locally_bound.clone();
                for v in variables {
                    body_bound.insert(v.clone());
                }
                collect_lambda_free_vars_in_stmts(body, &mut body_bound, free_vars);
            }
            StmtKind::Return(exprs) => {
                for e in exprs {
                    collect_lambda_free_vars_in_expr(e, locally_bound, free_vars);
                }
            }
            StmtKind::Break | StmtKind::Continue => {}
            StmtKind::Block(inner) | StmtKind::Scoped(inner) => {
                let mut inner_bound = locally_bound.clone();
                collect_lambda_free_vars_in_stmts(inner, &mut inner_bound, free_vars);
            }
            StmtKind::Defer(s) | StmtKind::ErrDefer(s) => {
                let mut inner_bound = locally_bound.clone();
                collect_lambda_free_vars_in_stmts(core::slice::from_ref(s.as_ref()), &mut inner_bound, free_vars);
            }
        }
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

fn count_ident_in_expr(expr: &Expr, target: &str) -> usize {
    match &expr.kind {
        ExprKind::Literal(_) | ExprKind::RegexLiteral { .. } => 0,
        ExprKind::Identifier(name) => usize::from(name == target),
        ExprKind::Binary { left, right, .. } => {
            count_ident_in_expr(left, target) + count_ident_in_expr(right, target)
        }
        ExprKind::Unary { operand, .. }
        | ExprKind::FieldAccess { object: operand, .. }
        | ExprKind::Paren(operand)
        | ExprKind::Cast { expr: operand, .. }
        | ExprKind::TypeCheck { expr: operand, .. }
        | ExprKind::IsPattern { expr: operand, .. } => count_ident_in_expr(operand, target),
        ExprKind::Call { callee, args, .. } => {
            count_ident_in_expr(callee, target)
                + args.iter().map(|a| count_ident_in_expr(a, target)).sum::<usize>()
        }
        ExprKind::MethodCall { receiver, args, .. } => {
            count_ident_in_expr(receiver, target)
                + args.iter().map(|a| count_ident_in_expr(a, target)).sum::<usize>()
        }
        ExprKind::Index { object, index } => {
            count_ident_in_expr(object, target) + count_ident_in_expr(index, target)
        }
        ExprKind::Array(elems) | ExprKind::Tuple(elems) | ExprKind::Return(elems) => {
            elems.iter().map(|e| count_ident_in_expr(e, target)).sum()
        }
        ExprKind::Map(pairs) => pairs
            .iter()
            .map(|(k, v)| count_ident_in_expr(k, target) + count_ident_in_expr(v, target))
            .sum(),
        ExprKind::StructLiteral { fields, .. } => fields
            .iter()
            .map(|f| count_ident_in_expr(&f.value, target))
            .sum(),
        ExprKind::EnumConstructor { args, .. } => {
            args.iter().map(|a| count_ident_in_expr(a, target)).sum()
        }
        ExprKind::Lambda { body, .. } => count_ident_in_expr(body, target),
        ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            count_ident_in_expr(condition, target)
                + count_ident_in_expr(then_branch, target)
                + else_branch
                    .as_ref()
                    .map_or(0, |e| count_ident_in_expr(e, target))
        }
        ExprKind::Block(stmts) | ExprKind::EffectBlock(stmts) => {
            stmts.iter().map(|s| count_ident_in_stmt(s, target)).sum()
        }
        ExprKind::Range { start, end, .. } => {
            count_ident_in_expr(start, target) + count_ident_in_expr(end, target)
        }
    }
}

fn count_ident_in_stmt(stmt: &Stmt, target: &str) -> usize {
    match &stmt.kind {
        StmtKind::Local { initializer, .. } => initializer.as_ref().map_or(0, |inits| {
            inits.iter().map(|e| count_ident_in_expr(e, target)).sum()
        }),
        StmtKind::Assign { targets, values } => {
            targets.iter().map(|e| count_ident_in_expr(e, target)).sum::<usize>()
                + values.iter().map(|e| count_ident_in_expr(e, target)).sum::<usize>()
        }
        StmtKind::CompoundAssign { target: t, value, .. } => {
            count_ident_in_expr(t, target) + count_ident_in_expr(value, target)
        }
        StmtKind::Expr(e) => count_ident_in_expr(e, target),
        StmtKind::If {
            condition,
            then_block,
            elseif_branches,
            else_block,
        } => {
            count_ident_in_expr(condition, target)
                + then_block.iter().map(|s| count_ident_in_stmt(s, target)).sum::<usize>()
                + elseif_branches
                    .iter()
                    .map(|(c, b)| {
                        count_ident_in_expr(c, target)
                            + b.iter().map(|s| count_ident_in_stmt(s, target)).sum::<usize>()
                    })
                    .sum::<usize>()
                + else_block
                    .as_ref()
                    .map_or(0, |b| b.iter().map(|s| count_ident_in_stmt(s, target)).sum())
        }
        StmtKind::While { condition, body } => {
            count_ident_in_expr(condition, target)
                + body.iter().map(|s| count_ident_in_stmt(s, target)).sum::<usize>()
        }
        StmtKind::ForNumeric {
            start,
            end,
            step,
            body,
            ..
        } => {
            count_ident_in_expr(start, target)
                + count_ident_in_expr(end, target)
                + step.as_ref().map_or(0, |s| count_ident_in_expr(s, target))
                + body.iter().map(|s| count_ident_in_stmt(s, target)).sum::<usize>()
        }
        StmtKind::ForIn { iterator, body, .. } => {
            count_ident_in_expr(iterator, target)
                + body.iter().map(|s| count_ident_in_stmt(s, target)).sum::<usize>()
        }
        StmtKind::Return(exprs) => exprs.iter().map(|e| count_ident_in_expr(e, target)).sum(),
        StmtKind::Break | StmtKind::Continue => 0,
        StmtKind::Block(stmts) | StmtKind::Scoped(stmts) => {
            stmts.iter().map(|s| count_ident_in_stmt(s, target)).sum()
        }
        StmtKind::Defer(s) | StmtKind::ErrDefer(s) => count_ident_in_stmt(s, target),
    }
}

fn stmts_mention_ident(stmts: &[Stmt], target: &str) -> bool {
    stmts.iter().any(|s| count_ident_in_stmt(s, target) > 0)
}

fn is_lua_value_array_type(ty: &Type) -> bool {
    matches!(
        &ty.kind,
        TypeKind::Array(inner) if matches!(&inner.kind, TypeKind::Named(name) if name == "LuaValue")
    )
}

fn is_string_byte_call(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::Paren(inner) => is_string_byte_call(inner),
        ExprKind::Call { callee, .. } => matches!(
            AstLowering::call_callee_name(callee).as_deref(),
            Some("string.byte")
        ),
        _ => false,
    }
}

fn var_only_matched_as_int_in_stmts(stmts: &[Stmt], var_name: &str) -> bool {
    let mut saw_match = false;
    for stmt in stmts {
        if count_ident_in_stmt(stmt, var_name) == 0 {
            continue;
        }
        match &stmt.kind {
            StmtKind::If {
                condition,
                then_block,
                elseif_branches,
                else_block,
            } => {
                let mut cond = condition;
                while let ExprKind::Paren(inner) = &cond.kind {
                    cond = inner;
                }
                let ok_cond = matches!(
                    &cond.kind,
                    ExprKind::IsPattern {
                        expr: scrutinee,
                        pattern: crate::ast::Pattern::Enum { enum_name, variant, bindings },
                    } if enum_name.is_empty()
                        && variant == "Int"
                        && bindings.len() <= 1
                        && matches!(&scrutinee.kind, ExprKind::Identifier(n) if n == var_name)
                );
                if !ok_cond {
                    return false;
                }
                if stmts_mention_ident(then_block, var_name) {
                    return false;
                }
                for (c, b) in elseif_branches {
                    if count_ident_in_expr(c, var_name) > 0 || stmts_mention_ident(b, var_name) {
                        return false;
                    }
                }
                if let Some(eb) = else_block
                    && stmts_mention_ident(eb, var_name)
                {
                    return false;
                }
                saw_match = true;
            }
            _ => return false,
        }
    }
    saw_match
}

fn array_local_only_matched_as_int(stmts: &[Stmt], arr_name: &str) -> bool {
    let mut saw_use = false;
    for stmt in stmts {
        if count_ident_in_stmt(stmt, arr_name) == 0 {
            continue;
        }
        match &stmt.kind {
            StmtKind::If {
                condition,
                then_block,
                elseif_branches,
                else_block,
            } => {
                let mut cond = condition;
                while let ExprKind::Paren(inner) = &cond.kind {
                    cond = inner;
                }
                let ExprKind::IsPattern {
                    expr: scrutinee,
                    pattern:
                        crate::ast::Pattern::Enum {
                            enum_name,
                            variant,
                            bindings,
                        },
                } = &cond.kind
                else {
                    return false;
                };
                if !enum_name.is_empty()
                    || (variant != "Ok" && variant != "Some")
                    || bindings.len() != 1
                {
                    return false;
                }
                let Some((obj, idx_expr)) = AstLowering::extract_index_operands(scrutinee) else {
                    return false;
                };
                if !matches!(&obj.kind, ExprKind::Identifier(n) if n == arr_name)
                    || count_ident_in_expr(idx_expr, arr_name) > 0
                {
                    return false;
                }
                let crate::ast::Pattern::Identifier(elem_var) = &bindings[0] else {
                    return false;
                };
                if stmts_mention_ident(then_block, arr_name)
                    || !var_only_matched_as_int_in_stmts(then_block, elem_var)
                {
                    return false;
                }
                for (c, b) in elseif_branches {
                    if count_ident_in_expr(c, arr_name) > 0 || stmts_mention_ident(b, arr_name) {
                        return false;
                    }
                }
                if let Some(eb) = else_block
                    && stmts_mention_ident(eb, arr_name)
                {
                    return false;
                }
                saw_use = true;
            }
            _ => return false,
        }
    }
    saw_use
}

pub fn mir_type_from_ast(ty: &Type) -> MirType {
    match &ty.kind {
        TypeKind::Int => MirType::Int,
        TypeKind::Float => MirType::Float,
        TypeKind::Bool => MirType::Bool,
        TypeKind::String => MirType::String,
        TypeKind::Unit => MirType::Unit,
        TypeKind::Unknown => MirType::Unknown,
        TypeKind::Named(s) => MirType::Struct(s.clone()),
        TypeKind::Tuple(types) => MirType::Tuple(types.iter().map(mir_type_from_ast).collect()),
        TypeKind::Array(inner) => MirType::Array(Box::new(mir_type_from_ast(inner))),
        TypeKind::Map(k, v) => {
            MirType::Map(Box::new(mir_type_from_ast(k)), Box::new(mir_type_from_ast(v)))
        }
        TypeKind::Function {
            params,
            return_type,
        } => MirType::Function {
            params: params.iter().map(mir_type_from_ast).collect(),
            return_type: Box::new(mir_type_from_ast(return_type)),
        },
        TypeKind::Option(inner) => MirType::Option(Box::new(mir_type_from_ast(inner))),
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
