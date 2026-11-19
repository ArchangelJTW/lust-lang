//! Cranelift JIT dispatch, eligibility checks, and compound value marshaling for the Lust VM.

use crate::ast::types::TypeKind;
use crate::bytecode::{LustMap, StructLayout, StructObject, Value};
use crate::error::{LustError, Result};
#[cfg(feature = "std")]
use crate::mir::cranelift_backend::{
    DeoptSiteKind, MIR_STATUS_DEOPT, MIR_STATUS_ERR_DIV_BY_ZERO, MIR_STATUS_ERR_NEG_EXPONENT,
    MIR_STATUS_ERR_OOM, MIR_STATUS_ERR_OUT_OF_BOUNDS, MIR_STATUS_ERR_OUT_OF_GAS, MIR_STATUS_OK,
    apply_module_callee_return_types, find_loop_headers, get_registered_struct_layout_by_name,
    is_effect_or_handler_callee, is_vm_effect_intrinsic_callee, lust_mir_alloc_array_typed,
    lust_mir_alloc_struct_named, lust_mir_alloc_tuple_typed, mir_array_elem_tag,
};
#[cfg(feature = "test-deopt")]
use crate::vm::DeoptPayloadSource;
#[cfg(feature = "std")]
use crate::vm::CompiledDeoptRecipe;
use crate::mir::ir::{
    MirConstant, MirFunction, MirType, MirUnaryOp, Operand, ResourceKind, Rvalue, Statement,
    Terminator,
};
use crate::vm::VM;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;

#[derive(Debug, Clone, Copy)]
pub enum RawArg {
    Int(i64),
    Float(f64),
}

#[derive(Debug, Clone)]
pub struct MarshalledArray {
    pub rc: Rc<RefCell<Vec<Value>>>,
    pub ptr: *mut i64,
    pub elem_ty: MirType,
    pub param_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct MarshalledStruct {
    pub rc: Rc<StructObject>,
    pub ptr: *mut i64,
    pub field_count: usize,
    pub param_indices: Vec<usize>,
}

enum CraneliftOutcome {
    Fallback,
    Return(Value),
}

#[derive(Debug)]
pub struct DecodedArray {
    pub matched_input_rc: Option<Rc<RefCell<Vec<Value>>>>,
    pub _ptr: *mut i64,
    pub elements: Vec<Value>,
}

fn validate_owned_header(
    mir_ctx: &crate::mir::runtime_helpers::MirContext,
    raw_ptr_val: i64,
    kind_label: &str,
    check_kind: impl FnOnce(&crate::mir::runtime_helpers::MirAllocationKind) -> Result<()>,
) -> Result<(*mut i64, usize)> {
    if raw_ptr_val == 0 {
        return Err(LustError::RuntimeError {
            message: format!(
                "Invalid {kind_label} pointer provenance: null pointer (0x0) is not an owned allocation base"
            ),
        });
    }

    let raw_addr = raw_ptr_val as usize;
    // 1. Provenance: exact base in owned allocation records of the active invocation scope
    let alloc_rec = mir_ctx
        .get_active_allocation_record(raw_addr)
        .ok_or_else(|| LustError::RuntimeError {
            message: format!(
                "Invalid {kind_label} pointer provenance: pointer {:#x} is not an owned allocation base of current invocation",
                raw_addr
            ),
        })?;

    // 2. Kind & element/type validation BEFORE reading header or payload words
    check_kind(&alloc_rec.kind)?;

    // 3. Extent & header validation (verify capacity >= 2 BEFORE dereferencing header words)
    if alloc_rec.word_capacity < 2 {
        return Err(LustError::RuntimeError {
            message: format!(
                "Malformed {kind_label} allocation: word_capacity ({}) < 2",
                alloc_rec.word_capacity
            ),
        });
    }
    let base_ptr = alloc_rec.base_ptr;
    let len_raw = unsafe { *base_ptr };
    let cap_raw = unsafe { *base_ptr.add(1) };
    if len_raw < 0 || cap_raw < 0 {
        return Err(LustError::RuntimeError {
            message: format!(
                "Malformed {kind_label} header: negative len ({len_raw}) or cap ({cap_raw})"
            ),
        });
    }
    let len = len_raw as usize;
    let cap = cap_raw as usize;
    if len > cap {
        return Err(LustError::RuntimeError {
            message: format!("Malformed {kind_label} header: len ({len}) > cap ({cap})"),
        });
    }
    if cap
        .checked_add(2)
        .is_none_or(|needed| needed > alloc_rec.word_capacity)
    {
        return Err(LustError::RuntimeError {
            message: format!(
                "Malformed {kind_label} header: extent ({} words) exceeds owned buffer capacity ({} words)",
                cap + 2,
                alloc_rec.word_capacity
            ),
        });
    }
    Ok((base_ptr, len))
}

pub fn validate_and_decode_array(
    mir_ctx: &crate::mir::runtime_helpers::MirContext,
    marshalled_arrays: &[MarshalledArray],
    raw_ptr_val: i64,
    expected_elem_ty: &MirType,
) -> Result<DecodedArray> {
    if !VM::is_cranelift_supported_array_elem_type(expected_elem_ty) {
        return Err(LustError::RuntimeError {
            message: format!(
                "Unsupported non-primitive array element type {:?}",
                expected_elem_ty
            ),
        });
    }

    let (base_ptr, len) = validate_owned_header(mir_ctx, raw_ptr_val, "array", |kind| {
        match kind {
            crate::mir::runtime_helpers::MirAllocationKind::Array(existing_elem_ty) => {
                if existing_elem_ty != expected_elem_ty {
                    return Err(LustError::RuntimeError {
                        message: format!(
                            "Array element type mismatch: expected {:?}, found {:?}",
                            expected_elem_ty, existing_elem_ty
                        ),
                    });
                }
                Ok(())
            }
            other => Err(LustError::RuntimeError {
                message: format!(
                    "Type/kind mismatch: expected Array({:?}) but allocation is {:?}",
                    expected_elem_ty, other
                ),
            }),
        }
    })?;

    // Element decoding
    let mut elements = Vec::with_capacity(len);
    for i in 0..len {
        let raw = unsafe { *base_ptr.add(2 + i) };
        let elem = match expected_elem_ty {
            MirType::Int => Value::Int(raw),
            MirType::Bool => Value::Bool(raw != 0),
            MirType::Float => Value::Float(f64::from_bits(raw as u64)),
            MirType::String => mir_ctx.get_handle(raw).unwrap_or_else(|| Value::string("")),
            _ => unreachable!("validated primitive array element type"),
        };
        elements.push(elem);
    }

    // Match against input arrays to preserve Rc identity
    let matched_input_rc = marshalled_arrays
        .iter()
        .find(|m| m.ptr == base_ptr)
        .map(|m| m.rc.clone());

    Ok(DecodedArray {
        matched_input_rc,
        _ptr: base_ptr,
        elements,
    })
}

pub fn validate_and_decode_tuple(
    mir_ctx: &crate::mir::runtime_helpers::MirContext,
    raw_ptr_val: i64,
    expected_elem_types: &[MirType],
) -> Result<Vec<Value>> {
    for elem_ty in expected_elem_types {
        if !VM::is_cranelift_supported_array_elem_type(elem_ty) {
            return Err(LustError::RuntimeError {
                message: format!(
                    "Unsupported non-primitive tuple element type {:?}",
                    elem_ty
                ),
            });
        }
    }

    let mut found_tuple_elems: Option<Vec<MirType>> = None;
    let (base_ptr, len) = validate_owned_header(mir_ctx, raw_ptr_val, "tuple", |kind| {
        match kind {
            crate::mir::runtime_helpers::MirAllocationKind::Tuple(existing_elems) => {
                found_tuple_elems = Some(existing_elems.clone());
                Ok(())
            }
            other => Err(LustError::RuntimeError {
                message: format!(
                    "Type/kind mismatch: expected Tuple({:?}) but allocation is {:?}",
                    expected_elem_types, other
                ),
            }),
        }
    })?;

    if let Some(existing_elems) = found_tuple_elems {
        if existing_elems.as_slice() != expected_elem_types {
            return Err(LustError::RuntimeError {
                message: format!(
                    "Tuple element type mismatch: expected {:?}, found {:?}",
                    expected_elem_types, existing_elems
                ),
            });
        }
    }

    if len != expected_elem_types.len() {
        return Err(LustError::RuntimeError {
            message: format!(
                "Malformed tuple header: len ({len}) != expected arity ({})",
                expected_elem_types.len()
            ),
        });
    }

    let mut elems = Vec::with_capacity(len);
    for (i, elem_ty) in expected_elem_types.iter().enumerate() {
        let raw = unsafe { *base_ptr.add(2 + i) };
        let elem = match elem_ty {
            MirType::Int => Value::Int(raw),
            MirType::Bool => Value::Bool(raw != 0),
            MirType::Float => Value::Float(f64::from_bits(raw as u64)),
            MirType::String => mir_ctx.get_handle(raw).unwrap_or_else(|| Value::string("")),
            _ => unreachable!("validated primitive tuple element type"),
        };
        elems.push(elem);
    }
    Ok(elems)
}

pub fn validate_and_decode_struct(
    mir_ctx: &crate::mir::runtime_helpers::MirContext,
    marshalled_structs: &[MarshalledStruct],
    raw_ptr_val: i64,
    expected_name: &str,
    layout_opt: Option<&Rc<StructLayout>>,
) -> Result<Value> {
    let (base_ptr, len) = validate_owned_header(mir_ctx, raw_ptr_val, "struct", |kind| {
        match kind {
            crate::mir::runtime_helpers::MirAllocationKind::Struct(existing_name) => {
                if existing_name != expected_name {
                    return Err(LustError::RuntimeError {
                        message: format!(
                            "Struct type mismatch: expected '{}', found '{}'",
                            expected_name, existing_name
                        ),
                    });
                }
                Ok(())
            }
            other => Err(LustError::RuntimeError {
                message: format!(
                    "Type/kind mismatch: expected Struct('{}') but allocation is {:?}",
                    expected_name, other
                ),
            }),
        }
    })?;

    if let Some(layout) = layout_opt {
        let expected_fields = layout.field_names().len();
        if len != expected_fields {
            return Err(LustError::RuntimeError {
                message: format!(
                    "Malformed struct header for '{}': len ({len}) != expected field count ({expected_fields})",
                    expected_name
                ),
            });
        }
    }

    if let Some(matched) = marshalled_structs.iter().find(|m| m.ptr == base_ptr) {
        return Ok(Value::Struct(matched.rc.clone()));
    }

    let mut fields = Vec::with_capacity(len);
    for i in 0..len {
        let raw = unsafe { *base_ptr.add(2 + i) };
        let fld_val = if let Some(layout) = layout_opt {
            if i < layout.field_names().len() {
                match &layout.field_type(i).kind {
                    TypeKind::Int => Value::Int(raw),
                    TypeKind::Float => Value::Float(f64::from_bits(raw as u64)),
                    TypeKind::Bool => Value::Bool(raw != 0),
                    TypeKind::String => {
                        mir_ctx.get_handle(raw).unwrap_or_else(|| Value::string(""))
                    }
                    TypeKind::Map(_, _) => {
                        mir_ctx.get_handle(raw).unwrap_or_else(|| Value::map(LustMap::default()))
                    }
                    TypeKind::Option(_) | TypeKind::Result(_, _) => {
                        mir_ctx.get_handle(raw).unwrap_or(Value::Nil)
                    }
                    other => {
                        return Err(LustError::RuntimeError {
                            message: format!(
                                "Unsupported struct field type '{}' at '{}.{}' for Cranelift decoding",
                                other, expected_name, i
                            ),
                        });
                    }
                }
            } else {
                return Err(LustError::RuntimeError {
                    message: format!(
                        "Struct field index {i} out of bounds for '{}'",
                        expected_name
                    ),
                });
            }
        } else if let Some(h_val) = mir_ctx.get_handle(raw) {
            h_val
        } else if mir_ctx.get_allocation_record(raw as usize).is_some() {
            return Err(LustError::RuntimeError {
                message: format!(
                    "Unsupported aggregate pointer field at '{}.{}' without struct layout metadata",
                    expected_name, i
                ),
            });
        } else {
            Value::Int(raw)
        };
        fields.push(fld_val);
    }
    let layout = layout_opt.cloned().unwrap_or_else(|| {
        Rc::new(StructLayout::new(
            expected_name.to_string(),
            vec![],
            vec![],
            vec![],
            vec![],
        ))
    });
    Ok(Value::Struct(StructObject::new(
        expected_name.to_string(),
        layout,
        fields,
    )))
}

pub fn stage_input_writebacks(
    mir_ctx: &crate::mir::runtime_helpers::MirContext,
    marshalled_arrays: &[MarshalledArray],
) -> Result<Vec<(Rc<RefCell<Vec<Value>>>, Vec<Value>)>> {
    let mut staged = Vec::with_capacity(marshalled_arrays.len());
    for m in marshalled_arrays {
        let decoded = validate_and_decode_array(mir_ctx, marshalled_arrays, m.ptr as i64, &m.elem_ty)?;
        staged.push((m.rc.clone(), decoded.elements));
    }
    Ok(staged)
}

pub fn stage_struct_writebacks(
    mir_ctx: &crate::mir::runtime_helpers::MirContext,
    marshalled_structs: &[MarshalledStruct],
) -> Result<Vec<(Rc<StructObject>, Vec<Value>)>> {
    let mut staged = Vec::with_capacity(marshalled_structs.len());
    for m in marshalled_structs {
        let (base_ptr, len) = validate_owned_header(mir_ctx, m.ptr as i64, "struct", |kind| {
            match kind {
                crate::mir::runtime_helpers::MirAllocationKind::Struct(alloc_name) => {
                    if alloc_name != m.rc.name.as_str() {
                        return Err(LustError::RuntimeError {
                            message: format!(
                                "Struct type mismatch: expected '{}', found '{}'",
                                m.rc.name, alloc_name
                            ),
                        });
                    }
                    Ok(())
                }
                other => Err(LustError::RuntimeError {
                    message: format!(
                        "Type/kind mismatch: expected Struct allocation, found {:?}",
                        other
                    ),
                }),
            }
        })?;
        if len != m.field_count {
            return Err(LustError::RuntimeError {
                message: format!(
                    "Malformed struct header: len ({len}) != field_count ({})",
                    m.field_count
                ),
            });
        }
        let current_fields = m.rc.fields.try_borrow().map_err(|_| LustError::RuntimeError {
            message: "Failed to acquire borrow for struct writeback staging: mutable borrow active".into(),
        })?;
        let mut staged_fields = current_fields.clone();
        for (i, field) in staged_fields.iter_mut().take(len).enumerate() {
            let raw = unsafe { *base_ptr.add(2 + i) };
            match field {
                Value::Int(n) => *n = raw,
                Value::Bool(b) => *b = raw != 0,
                Value::Float(f) => *f = f64::from_bits(raw as u64),
                Value::String(_) | Value::Map(_) | Value::Enum(_) => {
                    if let Some(val) = mir_ctx.get_handle(raw) {
                        *field = val;
                    }
                }
                _ => {}
            }
        }
        staged.push((m.rc.clone(), staged_fields));
    }
    Ok(staged)
}

pub fn commit_staged_writebacks_and_structs(
    staged_arrays: Vec<(Rc<RefCell<Vec<Value>>>, Vec<Value>)>,
    staged_structs: Vec<(Rc<StructObject>, Vec<Value>)>,
) -> Result<()> {
    // 1. Acquire all mutable borrows for arrays and structs before publishing any changes
    let array_rcs: Vec<Rc<RefCell<Vec<Value>>>> =
        staged_arrays.iter().map(|(rc, _)| rc.clone()).collect();
    let mut array_guards = Vec::with_capacity(array_rcs.len());
    for rc in &array_rcs {
        match rc.try_borrow_mut() {
            Ok(g) => array_guards.push(g),
            Err(_) => {
                return Err(LustError::RuntimeError {
                    message: "Failed to acquire mutable borrow for array writeback: aliased borrow active".into(),
                });
            }
        }
    }

    let struct_rcs: Vec<Rc<StructObject>> =
        staged_structs.iter().map(|(rc, _)| rc.clone()).collect();
    let mut struct_guards = Vec::with_capacity(struct_rcs.len());
    for rc in &struct_rcs {
        match rc.fields.try_borrow_mut() {
            Ok(g) => struct_guards.push(g),
            Err(_) => {
                return Err(LustError::RuntimeError {
                    message: "Failed to acquire mutable borrow for struct writeback: aliased borrow active".into(),
                });
            }
        }
    }

    // 2. Publish all staged array and struct mutations atomically
    for ((_, new_elems), mut guard) in staged_arrays.into_iter().zip(array_guards) {
        *guard = new_elems;
    }
    for ((_, new_fields), mut guard) in staged_structs.into_iter().zip(struct_guards) {
        *guard = new_fields;
    }

    // 3. Invalidate specialized caches for updated arrays
    for rc in &array_rcs {
        crate::bytecode::value::invalidate_specialized_array_cache(rc);
    }

    Ok(())
}

pub fn commit_staged_writebacks(
    staged: Vec<(Rc<RefCell<Vec<Value>>>, Vec<Value>)>,
) -> Result<()> {
    commit_staged_writebacks_and_structs(staged, Vec::new())
}

fn stage_and_commit_all_writebacks(
    mir_ctx: &crate::mir::runtime_helpers::MirContext,
    marshalled_arrays: &[MarshalledArray],
    marshalled_structs: &[MarshalledStruct],
) -> Result<()> {
    let staged_arrays = stage_input_writebacks(mir_ctx, marshalled_arrays)?;
    let staged_structs = stage_struct_writebacks(mir_ctx, marshalled_structs)?;
    commit_staged_writebacks_and_structs(staged_arrays, staged_structs)
}

impl VM {
    #[cfg(feature = "std")]
    pub(crate) fn is_cranelift_supported_array_elem_type(ty: &MirType) -> bool {
        matches!(
            ty,
            MirType::Int | MirType::Bool | MirType::Float | MirType::String
        )
    }

    #[cfg(feature = "std")]
    pub(crate) fn type_contains_unknown(ty: &MirType) -> bool {
        match ty {
            MirType::Unknown => true,
            MirType::Array(inner) => Self::type_contains_unknown(inner),
            MirType::Tuple(elems) => elems.iter().any(Self::type_contains_unknown),
            MirType::Map(k, v) => {
                Self::type_contains_unknown(k) || Self::type_contains_unknown(v)
            }
            MirType::Function { params, return_type } => {
                params.iter().any(Self::type_contains_unknown)
                    || Self::type_contains_unknown(return_type)
            }
            MirType::Int
            | MirType::Bool
            | MirType::Unit
            | MirType::Float
            | MirType::String
            | MirType::Struct(_)
            | MirType::Enum(_)
            | MirType::Effect { .. } => false,
        }
    }

    #[cfg(feature = "std")]
    pub(crate) fn is_cranelift_supported_type(ty: &MirType) -> bool {
        match ty {
            MirType::Int | MirType::Bool | MirType::Unit | MirType::Float | MirType::String | MirType::Unknown => true,
            MirType::Array(inner) => {
                Self::is_cranelift_supported_array_elem_type(inner)
                    || matches!(&**inner, MirType::Unknown)
            }
            MirType::Tuple(elems) => elems
                .iter()
                .all(|e| Self::is_cranelift_supported_array_elem_type(e) || matches!(e, MirType::Unknown)),
            MirType::Struct(_) => true,
            MirType::Map(k, v) => Self::is_cranelift_supported_type(k) && Self::is_cranelift_supported_type(v),
            MirType::Enum(_) => true,
            MirType::Function { .. } | MirType::Effect { .. } => false,
        }
    }

    #[cfg(feature = "std")]
    pub(crate) fn is_cranelift_eligible_operand(op: &Operand) -> bool {
        match op {
            Operand::Copy(_) | Operand::Move(_) => true,
            Operand::Constant(c) => matches!(
                c,
                MirConstant::Int(_)
                    | MirConstant::Float(_)
                    | MirConstant::Bool(_)
                    | MirConstant::String(_)
            ),
        }
    }

    #[cfg(feature = "std")]
    pub(crate) fn is_cranelift_eligible_rvalue(rval: &Rvalue) -> bool {
        match rval {
            Rvalue::Use(op) | Rvalue::UnaryOp(_, op) => Self::is_cranelift_eligible_operand(op),
            Rvalue::BinaryOp(_, l, r) => {
                Self::is_cranelift_eligible_operand(l) && Self::is_cranelift_eligible_operand(r)
            }
            Rvalue::Aggregate(ty, elems) => {
                Self::is_cranelift_supported_type(ty)
                    && elems.iter().all(Self::is_cranelift_eligible_operand)
            }
            Rvalue::Cast(op, target_ty) => {
                Self::is_cranelift_eligible_operand(op)
                    && Self::is_cranelift_supported_type(target_ty)
            }
        }
    }

    #[cfg(feature = "std")]
    pub(crate) fn is_cranelift_supported_struct(&self, name: &str) -> bool {
        if name.is_empty() || get_registered_struct_layout_by_name(name).is_some() {
            return false;
        }
        let Some(info) = self.struct_metadata.get(name) else {
            return false;
        };
        (0..info.layout.field_names().len()).all(|i| {
            matches!(
                info.layout.field_type(i).kind,
                TypeKind::Int
                    | TypeKind::Bool
                    | TypeKind::Float
                    | TypeKind::String
                    | TypeKind::Map(_, _)
                    | TypeKind::Option(_)
                    | TypeKind::Result(_, _)
            )
        })
    }

    #[cfg(feature = "std")]
    fn is_cranelift_supported_type_in_vm(&self, ty: &MirType) -> bool {
        match ty {
            MirType::Struct(name) => self.is_cranelift_supported_struct(name),
            MirType::Map(k, v) => {
                self.is_cranelift_supported_type_in_vm(k) && self.is_cranelift_supported_type_in_vm(v)
            }
            other => Self::is_cranelift_supported_type(other),
        }
    }

    #[cfg(feature = "std")]
    fn mir_place_type_in_vm(
        place: &crate::mir::ir::Place,
        locals: &[crate::mir::ir::LocalDecl],
        struct_meta: Option<&hashbrown::HashMap<String, crate::vm::RuntimeStructInfo>>,
    ) -> Option<MirType> {
        use crate::mir::ir::Place;
        match place {
            Place::Local(loc) => locals.get(loc.0).map(|l| l.ty.clone()),
            Place::Index(base, _) => match Self::mir_place_type_in_vm(base, locals, struct_meta)? {
                MirType::Array(inner) => Some(*inner),
                MirType::Tuple(elems) => elems.first().cloned(),
                _ => None,
            },
            Place::Field(base, idx) => match Self::mir_place_type_in_vm(base, locals, struct_meta)? {
                MirType::Tuple(elems) => elems.get(*idx).cloned(),
                MirType::Struct(name) => struct_meta
                    .and_then(|m| m.get(&name))
                    .and_then(|info| {
                        if *idx < info.layout.field_names().len() {
                            Some(crate::mir::lowering::mir_type_from_ast(
                                info.layout.field_type(*idx),
                            ))
                        } else {
                            None
                        }
                    }),
                _ => None,
            },
        }
    }

    #[cfg(feature = "std")]
    fn mir_operand_type_in_vm(
        op: &Operand,
        locals: &[crate::mir::ir::LocalDecl],
        struct_meta: Option<&hashbrown::HashMap<String, crate::vm::RuntimeStructInfo>>,
    ) -> Option<MirType> {
        match op {
            Operand::Copy(p) | Operand::Move(p) => {
                Self::mir_place_type_in_vm(p, locals, struct_meta)
            }
            Operand::Constant(c) => Some(match c {
                MirConstant::Int(_) => MirType::Int,
                MirConstant::Float(_) => MirType::Float,
                MirConstant::Bool(_) => MirType::Bool,
                MirConstant::String(_) => MirType::String,
                MirConstant::Unit => MirType::Unit,
            }),
        }
    }

    #[cfg(feature = "std")]
    fn check_refined_cranelift_eligibility(
        refined: &MirFunction,
        struct_meta: Option<&hashbrown::HashMap<String, crate::vm::RuntimeStructInfo>>,
        check_struct: &dyn Fn(&str) -> bool,
    ) -> bool {
        let check_type = |ty: &MirType| -> bool {
            if Self::type_contains_unknown(ty) || !Self::is_cranelift_supported_type(ty) {
                return false;
            }
            match ty {
                MirType::Struct(name) => check_struct(name),
                _ => true,
            }
        };
        if !check_type(&refined.signature.return_type)
            || !refined.params.iter().all(|p| check_type(&refined.locals[p.0].ty))
            || !refined.locals.iter().all(|l| check_type(&l.ty))
        {
            return false;
        }
        for block in &refined.blocks {
            for stmt in &block.statements {
                if let Statement::Assign(place, rval, _) = stmt {
                    if let crate::mir::ir::Place::Field(base, _) = place {
                        if let Some(MirType::Struct(name)) =
                            Self::mir_place_type_in_vm(base, &refined.locals, struct_meta)
                        {
                            if !check_struct(&name) {
                                return false;
                            }
                        }
                    }
                    match rval {
                        Rvalue::Aggregate(agg_ty, elems) => {
                            if Self::type_contains_unknown(agg_ty) || !check_type(agg_ty) {
                                return false;
                            }
                            match agg_ty {
                                MirType::Array(elem_ty) => {
                                    if !Self::is_cranelift_supported_array_elem_type(elem_ty) {
                                        return false;
                                    }
                                    for elem in elems {
                                        if Self::mir_operand_type_in_vm(
                                            elem,
                                            &refined.locals,
                                            struct_meta,
                                        )
                                        .as_ref()
                                            != Some(&**elem_ty)
                                        {
                                            return false;
                                        }
                                    }
                                }
                                MirType::Tuple(elem_types) => {
                                    if elem_types.len() != elems.len() {
                                        return false;
                                    }
                                    for (elem, expected_ty) in elems.iter().zip(elem_types.iter()) {
                                        if !Self::is_cranelift_supported_array_elem_type(expected_ty)
                                            || Self::mir_operand_type_in_vm(
                                                elem,
                                                &refined.locals,
                                                struct_meta,
                                            )
                                            .as_ref()
                                                != Some(expected_ty)
                                        {
                                            return false;
                                        }
                                    }
                                }
                                MirType::Struct(name) => {
                                    if !check_struct(name) {
                                        return false;
                                    }
                                }
                                _ => {}
                            }
                        }
                        Rvalue::UnaryOp(MirUnaryOp::Len, op) => {
                            let op_ty =
                                Self::mir_operand_type_in_vm(op, &refined.locals, struct_meta);
                            if !matches!(op_ty, Some(MirType::Array(_) | MirType::String)) {
                                return false;
                            }
                        }
                        Rvalue::Cast(_, target_ty) => {
                            if !check_type(target_ty) {
                                return false;
                            }
                        }
                        _ => {}
                    }
                }
            }
            if let Some(Terminator::Call { callee, args, .. }) = &block.terminator {
                if matches!(callee.as_str(), "table.len" | "array.len") {
                    if args.len() != 1 {
                        return false;
                    }
                    let arg_ty =
                        Self::mir_operand_type_in_vm(&args[0], &refined.locals, struct_meta);
                    if !matches!(arg_ty, Some(MirType::Array(_))) {
                        return false;
                    }
                } else if callee == "string.len" {
                    if args.len() != 1 {
                        return false;
                    }
                    let arg_ty =
                        Self::mir_operand_type_in_vm(&args[0], &refined.locals, struct_meta);
                    if arg_ty != Some(MirType::String) {
                        return false;
                    }
                }
            }
        }
        true
    }

    #[cfg(feature = "std")]
    fn is_cranelift_eligible_precheck(mir_fn: &MirFunction) -> bool {
        if mir_fn.params.len() > 6
            || !Self::is_cranelift_supported_type(&mir_fn.signature.return_type)
        {
            return false;
        }
        if !mir_fn
            .params
            .iter()
            .all(|pid| Self::is_cranelift_supported_type(&mir_fn.locals[pid.0].ty))
        {
            return false;
        }
        if !mir_fn
            .locals
            .iter()
            .all(|loc| Self::is_cranelift_supported_type(&loc.ty))
        {
            return false;
        }
        for block in &mir_fn.blocks {
            for stmt in &block.statements {
                match stmt {
                    Statement::Assign(_, rval, _) => {
                        if !Self::is_cranelift_eligible_rvalue(rval) {
                            return false;
                        }
                    }
                    Statement::ResourceCheck(ResourceKind::Memory, _, _) => {}
                    Statement::StorageLive(..)
                    | Statement::StorageDead(..)
                    | Statement::ResourceCheck(ResourceKind::Gas, _, _) => {}
                }
            }
            if let Some(term) = &block.terminator {
                match term {
                    Terminator::Goto(..) | Terminator::Unreachable(_) => {}
                    Terminator::Cond { cond, .. } => {
                        if !Self::is_cranelift_eligible_operand(cond) {
                            return false;
                        }
                    }
                    Terminator::Switch { discr, .. } => {
                        if !Self::is_cranelift_eligible_operand(discr) {
                            return false;
                        }
                    }
                    Terminator::Call { args, .. } => {
                        if !args.iter().all(Self::is_cranelift_eligible_operand) {
                            return false;
                        }
                    }
                    Terminator::Return(opt_op, _) => {
                        if let Some(op) = opt_op
                            && !Self::is_cranelift_eligible_operand(op)
                        {
                            return false;
                        }
                    }
                    Terminator::Raise(..) => return false,
                }
            }
        }
        true
    }

    #[cfg(feature = "std")]
    #[allow(dead_code)]
    pub(crate) fn is_cranelift_eligible_signature(mir_fn: &MirFunction) -> bool {
        if !Self::is_cranelift_eligible_precheck(mir_fn) {
            return false;
        }
        let refined = crate::mir::cranelift_backend::jit_engine::refine_mir_function(mir_fn);
        Self::check_refined_cranelift_eligibility(&refined, None, &|_| true)
    }

    #[cfg(feature = "std")]
    fn refine_mir_function_in_vm(&self, mir_fn: &MirFunction) -> MirFunction {
        let mut f = mir_fn.clone();
        let callee_rets: hashbrown::HashMap<String, MirType> = self
            .mir_functions
            .iter()
            .map(|(k, v)| (k.clone(), v.signature.return_type.clone()))
            .collect();
        apply_module_callee_return_types(&mut f, &callee_rets);
        crate::mir::cranelift_backend::jit_engine::refine_mir_function(&f)
    }

    #[cfg(feature = "std")]
    pub(crate) fn is_cranelift_eligible_in_vm(&self, mir_fn: &MirFunction) -> bool {
        if !Self::is_cranelift_eligible_precheck(mir_fn) {
            return false;
        }
        if !self.is_cranelift_supported_type_in_vm(&mir_fn.signature.return_type)
            || !mir_fn
                .params
                .iter()
                .all(|pid| self.is_cranelift_supported_type_in_vm(&mir_fn.locals[pid.0].ty))
            || !mir_fn
                .locals
                .iter()
                .all(|loc| self.is_cranelift_supported_type_in_vm(&loc.ty))
        {
            return false;
        }
        let refined = self.refine_mir_function_in_vm(mir_fn);
        Self::check_refined_cranelift_eligibility(
            &refined,
            Some(&self.struct_metadata),
            &|name| self.is_cranelift_supported_struct(name),
        )
    }

    #[cfg(feature = "std")]
    fn is_cranelift_builtin_callee(callee: &str) -> bool {
        matches!(
            callee,
            "print"
                | "println"
                | "math.abs"
                | "abs"
                | "math.min"
                | "min"
                | "math.max"
                | "max"
                | "math.clamp"
                | "clamp"
                | "table.len"
                | "array.len"
                | "string.len"
                | "map.len"
                | "map.get"
                | "map.set"
                | "map.new"
                | "enum.tag"
                | "enum.payload"
                | "enum.new"
                | "enum.alloc"
        )
    }

    #[cfg(feature = "std")]
    fn is_supported_deopt_recipe_type(ty: &MirType) -> bool {
        matches!(
            ty,
            MirType::Int | MirType::Bool | MirType::Float | MirType::Unit | MirType::String
        ) || matches!(
            ty,
            MirType::Array(inner)
                if matches!(&**inner, MirType::Int | MirType::Bool | MirType::Float)
        )
    }

    #[cfg(feature = "std")]
    fn build_function_deopt_recipes(
        &self,
        func_name: &str,
        mir_fn: &MirFunction,
        compilation_generation: u64,
        hook_authorized_for_fn: bool,
    ) -> core::result::Result<(Vec<CompiledDeoptRecipe>, bool), String> {
        let Some(bc_fn) = self.functions.iter().find(|f| f.name == func_name) else {
            return Err(format!("Missing bytecode function for '{}'", func_name));
        };
        let refined = self.refine_mir_function_in_vm(mir_fn);
        let mut recipes = Vec::new();
        let mut has_effect_site = false;

        for (b_idx, block) in refined.blocks.iter().enumerate() {
            if let Some(Terminator::Call { callee, .. }) = &block.terminator
                && is_effect_or_handler_callee(callee)
            {
                if !is_vm_effect_intrinsic_callee(callee) && !hook_authorized_for_fn {
                    return Err(format!(
                        "Callee '{}' in '{}' requires test-deopt hook authorization",
                        callee, func_name
                    ));
                }
                has_effect_site = true;
                let proven_indices = refined.derive_proven_site_metadata(b_idx).map_err(|err| {
                    format!(
                        "Cannot derive authoritative deopt metadata for site {}: {}",
                        b_idx, err
                    )
                })?;
                let precise_pc = bc_fn
                    .terminator_pcs
                    .get(&b_idx)
                    .copied()
                    .ok_or_else(|| {
                        format!(
                            "Missing terminator bytecode PC mapping for block {} in '{}'",
                            b_idx, func_name
                        )
                    })?;
                let mut proven_locals = Vec::with_capacity(proven_indices.len());
                for idx in proven_indices {
                    let ty = refined
                        .locals
                        .get(idx)
                        .map(|l| l.ty.clone())
                        .ok_or_else(|| format!("Local index {} out of bounds in '{}'", idx, func_name))?;
                    if Self::type_contains_unknown(&ty) || !Self::is_supported_deopt_recipe_type(&ty) {
                        return Err(format!(
                            "Unsupported deopt local {} type {:?} at site {} in '{}'",
                            idx, ty, b_idx, func_name
                        ));
                    }
                    proven_locals.push((idx, ty));
                }
                recipes.push(CompiledDeoptRecipe {
                    func_name: func_name.to_string(),
                    compilation_generation,
                    block_id: b_idx,
                    site_kind: DeoptSiteKind::PreTerminator,
                    precise_pc,
                    proven_locals,
                });
            }
        }

        let loop_headers = find_loop_headers(&refined.blocks);
        for b_idx in loop_headers {
            let proven_indices = refined
                .derive_proven_site_metadata_at_entry(b_idx)
                .map_err(|err| {
                    format!(
                        "Cannot derive authoritative deopt metadata for loop header safepoint {}: {}",
                        b_idx, err
                    )
                })?;
            let mut proven_locals = Vec::with_capacity(proven_indices.len());
            for idx in proven_indices {
                let ty = refined
                    .locals
                    .get(idx)
                    .map(|l| l.ty.clone())
                    .ok_or_else(|| format!("Local index {} out of bounds in '{}'", idx, func_name))?;
                if Self::type_contains_unknown(&ty) || !Self::is_supported_deopt_recipe_type(&ty) {
                    return Err(format!(
                        "Unsupported deopt local {} type {:?} at loop header {} in '{}'",
                        idx, ty, b_idx, func_name
                    ));
                }
                proven_locals.push((idx, ty));
            }
            if !bc_fn.block_entry_pcs.is_empty() {
                let precise_pc = bc_fn
                    .block_entry_pcs
                    .get(&b_idx)
                    .copied()
                    .ok_or_else(|| {
                        format!(
                            "Missing block entry bytecode PC mapping for block {} in '{}'",
                            b_idx, func_name
                        )
                    })?;
                recipes.push(CompiledDeoptRecipe {
                    func_name: func_name.to_string(),
                    compilation_generation,
                    block_id: b_idx,
                    site_kind: DeoptSiteKind::BlockEntry,
                    precise_pc,
                    proven_locals,
                });
            } else if has_effect_site {
                return Err(format!(
                    "Missing block entry bytecode PC mapping for block {} in '{}'",
                    b_idx, func_name
                ));
            }
        }

        Ok((recipes, has_effect_site))
    }

    /// Compiles all uncompiled eligible registered MIR functions using CraneliftJitCompiler.
    #[cfg(feature = "std")]
    pub fn compile_cranelift(&mut self) -> Result<()> {
        let _guard = self.mir_context.enter();
        let mut eligible: hashbrown::HashSet<String> = self
            .mir_functions
            .iter()
            .filter(|(_, f)| self.is_cranelift_eligible_in_vm(f))
            .map(|(name, _)| name.clone())
            .collect();

        #[cfg(feature = "test-deopt")]
        let hook_target: Option<String> = self
            .test_deopt_spike_hook
            .as_ref()
            .map(|h| h.target_function.clone());
        #[cfg(not(feature = "test-deopt"))]
        let hook_target: Option<String> = None;

        let next_gen = if self.cranelift_compiled_generation == self.cranelift_generation {
            self.cranelift_generation.wrapping_add(1).max(1)
        } else {
            self.cranelift_generation.max(1)
        };
        let mut recipes_by_fn: hashbrown::HashMap<String, Vec<CompiledDeoptRecipe>> =
            hashbrown::HashMap::new();
        let mut may_deopt_effect: hashbrown::HashSet<String> = hashbrown::HashSet::new();

        let candidate_names: Vec<String> = eligible.iter().cloned().collect();
        for name in candidate_names {
            let Some(mir_fn) = self.mir_functions.get(&name) else {
                eligible.remove(&name);
                continue;
            };
            let hook_auth = hook_target.as_deref() == Some(name.as_str());
            match self.build_function_deopt_recipes(&name, mir_fn, next_gen, hook_auth) {
                Ok((recipes, has_effect)) => {
                    if has_effect {
                        may_deopt_effect.insert(name.clone());
                    }
                    recipes_by_fn.insert(name, recipes);
                }
                Err(err_msg) => {
                    if hook_auth {
                        return Err(LustError::RuntimeError { message: err_msg });
                    }
                    eligible.remove(&name);
                }
            }
        }

        loop {
            let to_remove: Vec<String> = eligible
                .iter()
                .filter(|name| {
                    let Some(mir_fn) = self.mir_functions.get(*name) else {
                        return true;
                    };
                    mir_fn.blocks.iter().any(|block| {
                        let Some(Terminator::Call { callee, args, .. }) = &block.terminator else {
                            return false;
                        };
                        if Self::is_cranelift_builtin_callee(callee) {
                            return false;
                        }
                        if is_effect_or_handler_callee(callee) {
                            return false;
                        }
                        if !eligible.contains(callee) || may_deopt_effect.contains(callee) {
                            return true;
                        }
                        self.mir_functions
                            .get(callee)
                            .is_none_or(|target_fn| target_fn.params.len() != args.len())
                    })
                })
                .cloned()
                .collect();
            if to_remove.is_empty() {
                break;
            }
            for name in to_remove {
                eligible.remove(&name);
            }
        }

        if eligible.is_empty() {
            return Ok(());
        }

        let all_compiled = self.cranelift_jit.is_some()
            && eligible
                .iter()
                .all(|name| self.cranelift_compiled.contains_key(name));
        if all_compiled {
            return Ok(());
        }

        let mut pending_recipes = hashbrown::HashMap::new();
        let mut sorted_names: Vec<String> = eligible.into_iter().collect();
        sorted_names.sort();
        for name in &sorted_names {
            if let Some(fn_recipes) = recipes_by_fn.remove(name) {
                for recipe in fn_recipes {
                    pending_recipes.insert(
                        (recipe.func_name.clone(), recipe.block_id, recipe.site_kind),
                        recipe,
                    );
                }
            }
        }

        for (name, info) in &self.struct_metadata {
            let field_types: Vec<MirType> = (0..info.layout.field_names().len())
                .map(|i| crate::mir::lowering::mir_type_from_ast(info.layout.field_type(i)))
                .collect();
            self.mir_context.register_struct_field_types(name, field_types);
        }

        let mut module = crate::mir::ir::MirModule::new("vm_cranelift_module".into());
        for name in &sorted_names {
            if let Some(f) = self.mir_functions.get(name) {
                module.add_function(f.clone());
            }
        }

        let mut compiler = if self.cranelift_jit.is_some() && self.cranelift_compiled.is_empty() {
            self.cranelift_jit.take().unwrap()
        } else {
            crate::mir::cranelift_backend::jit_engine::CraneliftJitCompiler::new().map_err(|e| {
                LustError::RuntimeError {
                    message: format!("Failed to initialize Cranelift JIT compiler: {}", e),
                }
            })?
        };

        let compiled_map = match compiler.compile_module(&module) {
            Ok(map) => map,
            Err(e) => {
                return Err(LustError::RuntimeError {
                    message: format!("Cranelift JIT module compilation failed: {:?}", e),
                });
            }
        };

        // Transactional publish ONLY after compile_module returns Ok
        self.cranelift_jit = Some(compiler);
        self.cranelift_compiled.clear();
        self.cranelift_recipes.clear();
        for (name, ptr) in compiled_map {
            self.cranelift_compiled.insert(name, ptr);
            self.cranelift_compilation_count += 1;
        }
        for (key, recipe) in pending_recipes {
            self.cranelift_recipes.insert(key, recipe);
        }
        self.cranelift_generation = next_gen;
        self.cranelift_compiled_generation = next_gen;
        Ok(())
    }

    #[cfg(not(feature = "std"))]
    pub fn compile_cranelift(&mut self) -> Result<()> {
        Ok(())
    }

    /// Attempts to execute the top call frame natively using Cranelift JIT.
    #[cfg(feature = "std")]
    pub(crate) fn try_run_cranelift_top_frame(&mut self) -> Result<Option<Option<Value>>> {
        let (func_idx, param_count, has_upvalues) = match self.call_stack.last() {
            Some(f) => {
                let p_cnt = self
                    .functions
                    .get(f.function_idx)
                    .map(|func| func.param_count as usize)
                    .unwrap_or(0);
                (
                    f.function_idx,
                    p_cnt,
                    !f.upvalues.is_empty() || !f.shared_upvalues.is_empty(),
                )
            }
            None => return Ok(Some(Some(Value::Nil))),
        };
        if has_upvalues || param_count > 6 {
            return Ok(None);
        }
        let func_name = match self.functions.get(func_idx) {
            Some(f) => f.name.clone(),
            None => {
                return Err(LustError::RuntimeError {
                    message: format!("Function index {} not found in VM", func_idx),
                });
            }
        };

        let Some((param_types, ret_type, local_types)) = self.mir_functions.get(&func_name).and_then(|f| {
            if !self.is_cranelift_eligible_in_vm(f) {
                return None;
            }
            let refined = self.refine_mir_function_in_vm(f);
            let param_types: Vec<MirType> = f
                .params
                .iter()
                .map(|pid| {
                    let sig_ty = &f.locals[pid.0].ty;
                    if Self::type_contains_unknown(sig_ty) {
                        refined.locals[pid.0].ty.clone()
                    } else {
                        sig_ty.clone()
                    }
                })
                .collect();
            let local_types: Vec<MirType> = f
                .locals
                .iter()
                .enumerate()
                .map(|(idx, l)| {
                    if Self::type_contains_unknown(&l.ty) {
                        refined.locals[idx].ty.clone()
                    } else {
                        l.ty.clone()
                    }
                })
                .collect();
            Some((param_types, f.signature.return_type.clone(), local_types))
        }) else {
            return Ok(None);
        };
        if param_types.len() != param_count
            || Self::type_contains_unknown(&ret_type)
            || !self.is_cranelift_supported_type_in_vm(&ret_type)
        {
            return Ok(None);
        }
        if self
            .call_stack
            .last()
            .is_none_or(|frame| frame.registers.len() < param_count)
        {
            return Ok(None);
        }

        // Compile before entering MirContext invocation scope and packing arguments
        if !self.cranelift_compiled.contains_key(&func_name) {
            self.compile_cranelift()?;
        }
        let Some(code_ptr) = self.cranelift_compiled.get(&func_name).copied() else {
            return Ok(None);
        };

        let baseline = self.mir_context.capture_baseline();
        let mut native_entered = false;
        let outcome = self.try_run_cranelift_top_frame_scoped(
            func_idx,
            &func_name,
            &param_types,
            &ret_type,
            &local_types,
            code_ptr,
            &mut native_entered,
        );
        self.mir_context
            .clear_to_baselines(baseline.handles, baseline.buffers);

        match outcome {
            Ok(CraneliftOutcome::Fallback) => Ok(None),
            Ok(CraneliftOutcome::Return(result_val)) => {
                let finished = self.finish_return(result_val)?;
                Ok(Some(finished))
            }
            Err(err) => {
                crate::mir::cranelift_backend::take_current_sticky_error();
                crate::mir::cranelift_backend::take_deopt_payload();
                if native_entered {
                    if let Some(frame) = self.call_stack.pop() {
                        self.recycle_frame(frame);
                    }
                }
                Err(err)
            }
        }
    }

    #[cfg(feature = "std")]
    #[allow(clippy::too_many_arguments)]
    fn try_run_cranelift_top_frame_scoped(
        &mut self,
        _func_idx: usize,
        func_name: &str,
        param_types: &[MirType],
        ret_type: &MirType,
        local_types: &[MirType],
        code_ptr: *const u8,
        native_entered: &mut bool,
    ) -> Result<CraneliftOutcome> {
        let param_count = param_types.len();

        // --- INVOCATION CONTEXT ACTIVATION & RESOURCE INITIALIZATION ---
        // Activate invocation context before any argument packing allocations.
        // Keep host marshaling exempt from guest memory limit until native entry.
        let _mir_guard = self.mir_context.enter();

        let vm_gas_budget = self.gas_remaining();
        let vm_has_mem_budget = self.memory_budget_bytes().is_some();
        let mem_limit = match self.memory_remaining_bytes() {
            Some(rem_mem) => i64::try_from(rem_mem).unwrap_or(i64::MAX),
            None => i64::MAX,
        };

        self.mir_context.set_status(MIR_STATUS_OK);
        self.mir_context.set_memory_used(0);
        self.mir_context.set_max_memory_limit(i64::MAX);
        if let Some(rem_gas) = vm_gas_budget {
            self.mir_context
                .set_gas(i64::try_from(rem_gas).unwrap_or(i64::MAX));
        }

        for (name, info) in &self.struct_metadata {
            let field_types: Vec<MirType> = (0..info.layout.field_names().len())
                .map(|i| crate::mir::lowering::mir_type_from_ast(info.layout.field_type(i)))
                .collect();
            self.mir_context.register_struct_field_types(name, field_types);
        }

        // Convert parameters to RawArg with alias-preserving tracking
        let mut raw_args = Vec::with_capacity(param_count);
        let mut marshalled_arrays: Vec<MarshalledArray> = Vec::new();
        let mut marshalled_array_by_ptr: hashbrown::HashMap<*const RefCell<Vec<Value>>, usize> =
            hashbrown::HashMap::new();
        let mut marshalled_structs: Vec<MarshalledStruct> = Vec::new();
        let mut marshalled_struct_by_ptr: hashbrown::HashMap<*const StructObject, usize> =
            hashbrown::HashMap::new();

        {
            let frame = self.call_stack.last().unwrap();
            if frame.registers.len() < param_count {
                return Ok(CraneliftOutcome::Fallback);
            }
            for (i, expected_ty) in param_types.iter().enumerate() {
                let val = &frame.registers[i];
                let arg = match (val, expected_ty) {
                    (Value::Int(n), MirType::Int | MirType::Unknown) => RawArg::Int(*n),
                    (Value::Bool(b), MirType::Bool | MirType::Unknown) => RawArg::Int(i64::from(*b)),
                    (Value::Float(f), MirType::Float | MirType::Unknown) => RawArg::Float(*f),
                    (Value::String(_), MirType::String | MirType::Unknown) => {
                        let h = self.mir_context.register_handle(val.clone());
                        RawArg::Int(h)
                    }
                    (Value::Array(rc), MirType::Array(elem_ty)) => {
                        if !Self::is_cranelift_supported_array_elem_type(elem_ty) {
                            return Ok(CraneliftOutcome::Fallback);
                        }
                        crate::bytecode::value::try_sync_specialized_array(rc).map_err(|msg| {
                            LustError::RuntimeError { message: msg }
                        })?;
                        let rc_key = Rc::as_ptr(rc);
                        if let Some(&existing_idx) = marshalled_array_by_ptr.get(&rc_key) {
                            let existing = &mut marshalled_arrays[existing_idx];
                            if existing.elem_ty != **elem_ty {
                                return Ok(CraneliftOutcome::Fallback);
                            }
                            existing.param_indices.push(i);
                            raw_args.push(RawArg::Int(existing.ptr as i64));
                            continue;
                        }

                        let elements = rc.try_borrow().map_err(|_| LustError::RuntimeError {
                            message: "Failed to acquire borrow for array parameter packing: mutable borrow active".into(),
                        })?;
                        let all_elems_match = elements.iter().all(|elem| {
                            matches!(
                                (elem, &**elem_ty),
                                (Value::Int(_), MirType::Int)
                                    | (Value::Bool(_), MirType::Bool)
                                    | (Value::Float(_), MirType::Float)
                                    | (Value::String(_), MirType::String)
                            )
                        });
                        if !all_elems_match {
                            return Ok(CraneliftOutcome::Fallback);
                        }

                        let len = elements.len();
                        let elem_tag = mir_array_elem_tag(elem_ty).ok_or_else(|| LustError::RuntimeError {
                            message: format!("Unsupported non-primitive array element type {:?}", elem_ty),
                        })?;
                        let ptr = lust_mir_alloc_array_typed(len as i64, elem_tag);
                        if ptr.is_null() {
                            return Err(LustError::RuntimeError {
                                message: "Out of memory allocating array buffer for Cranelift JIT".into(),
                            });
                        }
                        for (idx, elem) in elements.iter().enumerate() {
                            let raw_elem = match (elem, &**elem_ty) {
                                (Value::Int(n), MirType::Int) => *n,
                                (Value::Bool(b), MirType::Bool) => i64::from(*b),
                                (Value::Float(f), MirType::Float) => f.to_bits() as i64,
                                (Value::String(_), MirType::String) => {
                                    self.mir_context.register_handle(elem.clone())
                                }
                                _ => unreachable!("array elements pre-verified"),
                            };
                            unsafe {
                                *ptr.add(2 + idx) = raw_elem;
                            }
                        }
                        let new_idx = marshalled_arrays.len();
                        marshalled_arrays.push(MarshalledArray {
                            rc: rc.clone(),
                            ptr,
                            elem_ty: (**elem_ty).clone(),
                            param_indices: vec![i],
                        });
                        marshalled_array_by_ptr.insert(rc_key, new_idx);
                        RawArg::Int(ptr as i64)
                    }
                    (Value::Tuple(rc), MirType::Tuple(elem_types)) => {
                        if rc.len() != elem_types.len() {
                            return Ok(CraneliftOutcome::Fallback);
                        }
                        let all_elems_match = rc.iter().zip(elem_types.iter()).all(|(elem, ty)| {
                            matches!(
                                (elem, ty),
                                (Value::Int(_), MirType::Int)
                                    | (Value::Bool(_), MirType::Bool)
                                    | (Value::Float(_), MirType::Float)
                                    | (Value::String(_), MirType::String)
                            )
                        });
                        if !all_elems_match {
                            return Ok(CraneliftOutcome::Fallback);
                        }

                        let len = rc.len();
                        let mut tags = Vec::with_capacity(len);
                        for elem_ty in elem_types {
                            let Some(tag) = mir_array_elem_tag(elem_ty) else {
                                return Ok(CraneliftOutcome::Fallback);
                            };
                            tags.push(tag);
                        }
                        let ptr = lust_mir_alloc_tuple_typed(len as i64, tags.as_ptr());
                        if ptr.is_null() {
                            return Err(LustError::RuntimeError {
                                message: "Out of memory allocating tuple buffer for Cranelift JIT".into(),
                            });
                        }
                        for (idx, (elem, elem_ty)) in rc.iter().zip(elem_types.iter()).enumerate() {
                            let raw_elem = match (elem, elem_ty) {
                                (Value::Int(n), MirType::Int) => *n,
                                (Value::Bool(b), MirType::Bool) => i64::from(*b),
                                (Value::Float(f), MirType::Float) => f.to_bits() as i64,
                                (Value::String(_), MirType::String) => {
                                    self.mir_context.register_handle(elem.clone())
                                }
                                _ => unreachable!("tuple elements pre-verified"),
                            };
                            unsafe {
                                *ptr.add(2 + idx) = raw_elem;
                            }
                        }
                        RawArg::Int(ptr as i64)
                    }
                    (Value::Struct(rc), MirType::Struct(expected_name)) => {
                        if rc.name.as_str() != expected_name.as_str()
                            || !self.is_cranelift_supported_struct(expected_name)
                        {
                            return Ok(CraneliftOutcome::Fallback);
                        }
                        let Some(expected_info) = self.struct_metadata.get(expected_name) else {
                            return Ok(CraneliftOutcome::Fallback);
                        };
                        let rc_key = Rc::as_ptr(rc);
                        if let Some(&existing_idx) = marshalled_struct_by_ptr.get(&rc_key) {
                            let existing = &mut marshalled_structs[existing_idx];
                            existing.param_indices.push(i);
                            raw_args.push(RawArg::Int(existing.ptr as i64));
                            continue;
                        }

                        let fields = rc.fields.try_borrow().map_err(|_| LustError::RuntimeError {
                            message: "Failed to acquire borrow for struct parameter packing: mutable borrow active".into(),
                        })?;
                        let expected_field_count = expected_info.layout.field_names().len();
                        if rc.layout.field_names() != expected_info.layout.field_names()
                            || fields.len() != expected_field_count
                        {
                            return Ok(CraneliftOutcome::Fallback);
                        }
                        let all_fields_match = fields.iter().enumerate().all(|(idx, fld)| {
                            matches!(
                                (fld, &expected_info.layout.field_type(idx).kind),
                                (Value::Int(_), TypeKind::Int)
                                    | (Value::Bool(_), TypeKind::Bool)
                                    | (Value::Float(_), TypeKind::Float)
                                    | (Value::String(_), TypeKind::String)
                                    | (Value::Map(_), TypeKind::Map(_, _))
                                    | (Value::Enum(_), TypeKind::Option(_) | TypeKind::Result(_, _))
                            )
                        });
                        if !all_fields_match {
                            return Ok(CraneliftOutcome::Fallback);
                        }

                        let len = fields.len();
                        let ptr = lust_mir_alloc_struct_named(
                            len as i64,
                            expected_name.as_ptr(),
                            expected_name.len() as i64,
                        );
                        if ptr.is_null() {
                            return Err(LustError::RuntimeError {
                                message: "Out of memory allocating struct buffer for Cranelift JIT".into(),
                            });
                        }
                        for (idx, fld) in fields.iter().enumerate() {
                            let raw_fld = match fld {
                                Value::Int(n) => *n,
                                Value::Bool(b) => i64::from(*b),
                                Value::Float(f) => f.to_bits() as i64,
                                Value::String(_) | Value::Map(_) | Value::Enum(_) => {
                                    self.mir_context.register_handle(fld.clone())
                                }
                                _ => unreachable!("struct fields pre-verified"),
                            };
                            unsafe {
                                *ptr.add(2 + idx) = raw_fld;
                            }
                        }
                        let new_idx = marshalled_structs.len();
                        marshalled_structs.push(MarshalledStruct {
                            rc: rc.clone(),
                            ptr,
                            field_count: len,
                            param_indices: vec![i],
                        });
                        marshalled_struct_by_ptr.insert(rc_key, new_idx);
                        RawArg::Int(ptr as i64)
                    }
                    (Value::Map(_), MirType::Map(_, _) | MirType::Unknown) => {
                        let h = self.mir_context.register_handle(val.clone());
                        RawArg::Int(h)
                    }
                    (Value::Enum(_), MirType::Enum(_) | MirType::Unknown) => {
                        let h = self.mir_context.register_handle(val.clone());
                        RawArg::Int(h)
                    }
                    _ => {
                        return Ok(CraneliftOutcome::Fallback);
                    }
                };
                raw_args.push(arg);
            }
        }

        // Apply guest memory budget immediately before native entry
        self.mir_context.set_memory_used(0);
        self.mir_context.set_max_memory_limit(mem_limit);

        let gas_entry_snapshot = self.mir_context.gas_remaining();

        self.cranelift_entry_count += 1;
        *native_entered = true;
        let invocation_id = self.cranelift_entry_count;
        let frame_depth = self.call_stack.len();
        let _inv_guard = crate::mir::cranelift_backend::push_native_invocation_with_generation(
            func_name.to_string(),
            frame_depth,
            invocation_id,
            self.cranelift_compiled_generation,
        );
        let _jit_guard = self.enter_jit_scope();

        // Charge base gas for entry
        crate::mir::cranelift_backend::jit_engine::lust_mir_charge_gas(1);

        // Execute JIT compiled code matching argument types and return type
        let (raw_int, raw_float): (i64, f64) = match (raw_args.as_slice(), &ret_type) {
            // 0 args
            (&[], MirType::Float) => {
                let func: extern "C" fn() -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func())
            }
            (&[], MirType::Unit) => {
                let func: extern "C" fn() = unsafe { core::mem::transmute(code_ptr) };
                func();
                (0, 0.0)
            }
            (&[], _) => {
                let func: extern "C" fn() -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(), 0.0)
            }

            // 1 arg
            (&[RawArg::Float(a)], MirType::Float) => {
                let func: extern "C" fn(f64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a))
            }
            (&[RawArg::Float(a)], MirType::Unit) => {
                let func: extern "C" fn(f64) = unsafe { core::mem::transmute(code_ptr) };
                func(a);
                (0, 0.0)
            }
            (&[RawArg::Float(a)], _) => {
                let func: extern "C" fn(f64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a), 0.0)
            }
            (&[RawArg::Int(a)], MirType::Float) => {
                let func: extern "C" fn(i64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a))
            }
            (&[RawArg::Int(a)], MirType::Unit) => {
                let func: extern "C" fn(i64) = unsafe { core::mem::transmute(code_ptr) };
                func(a);
                (0, 0.0)
            }
            (&[RawArg::Int(a)], _) => {
                let func: extern "C" fn(i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a), 0.0)
            }

            // 2 args
            (&[RawArg::Float(a), RawArg::Float(b)], MirType::Float) => {
                let func: extern "C" fn(f64, f64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b))
            }
            (&[RawArg::Float(a), RawArg::Float(b)], MirType::Unit) => {
                let func: extern "C" fn(f64, f64) = unsafe { core::mem::transmute(code_ptr) };
                func(a, b);
                (0, 0.0)
            }
            (&[RawArg::Float(a), RawArg::Float(b)], _) => {
                let func: extern "C" fn(f64, f64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b), 0.0)
            }
            (&[RawArg::Int(a), RawArg::Float(b)], MirType::Float) => {
                let func: extern "C" fn(i64, f64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b))
            }
            (&[RawArg::Int(a), RawArg::Float(b)], MirType::Unit) => {
                let func: extern "C" fn(i64, f64) = unsafe { core::mem::transmute(code_ptr) };
                func(a, b);
                (0, 0.0)
            }
            (&[RawArg::Int(a), RawArg::Float(b)], _) => {
                let func: extern "C" fn(i64, f64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b), 0.0)
            }
            (&[RawArg::Float(a), RawArg::Int(b)], MirType::Float) => {
                let func: extern "C" fn(f64, i64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b))
            }
            (&[RawArg::Float(a), RawArg::Int(b)], MirType::Unit) => {
                let func: extern "C" fn(f64, i64) = unsafe { core::mem::transmute(code_ptr) };
                func(a, b);
                (0, 0.0)
            }
            (&[RawArg::Float(a), RawArg::Int(b)], _) => {
                let func: extern "C" fn(f64, i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b), 0.0)
            }
            (&[RawArg::Int(a), RawArg::Int(b)], MirType::Float) => {
                let func: extern "C" fn(i64, i64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b))
            }
            (&[RawArg::Int(a), RawArg::Int(b)], MirType::Unit) => {
                let func: extern "C" fn(i64, i64) = unsafe { core::mem::transmute(code_ptr) };
                func(a, b);
                (0, 0.0)
            }
            (&[RawArg::Int(a), RawArg::Int(b)], _) => {
                let func: extern "C" fn(i64, i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b), 0.0)
            }

            // 3 args
            (&[RawArg::Float(a), RawArg::Float(b), RawArg::Float(c)], MirType::Float) => {
                let func: extern "C" fn(f64, f64, f64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b, c))
            }
            (&[RawArg::Float(a), RawArg::Float(b), RawArg::Float(c)], _) => {
                let func: extern "C" fn(f64, f64, f64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b, c), 0.0)
            }
            (&[RawArg::Int(a), RawArg::Int(b), RawArg::Int(c)], MirType::Float) => {
                let func: extern "C" fn(i64, i64, i64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b, c))
            }
            (&[RawArg::Int(a), RawArg::Int(b), RawArg::Int(c)], _) => {
                let func: extern "C" fn(i64, i64, i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b, c), 0.0)
            }
            (&[RawArg::Int(a), RawArg::Int(b), RawArg::Float(c)], MirType::Float) => {
                let func: extern "C" fn(i64, i64, f64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b, c))
            }
            (&[RawArg::Int(a), RawArg::Int(b), RawArg::Float(c)], _) => {
                let func: extern "C" fn(i64, i64, f64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b, c), 0.0)
            }
            (&[RawArg::Float(a), RawArg::Float(b), RawArg::Int(c)], MirType::Float) => {
                let func: extern "C" fn(f64, f64, i64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b, c))
            }
            (&[RawArg::Float(a), RawArg::Float(b), RawArg::Int(c)], _) => {
                let func: extern "C" fn(f64, f64, i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b, c), 0.0)
            }
            (&[RawArg::Int(a), RawArg::Float(b), RawArg::Int(c)], MirType::Float) => {
                let func: extern "C" fn(i64, f64, i64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b, c))
            }
            (&[RawArg::Int(a), RawArg::Float(b), RawArg::Int(c)], _) => {
                let func: extern "C" fn(i64, f64, i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b, c), 0.0)
            }
            (&[RawArg::Float(a), RawArg::Int(b), RawArg::Float(c)], MirType::Float) => {
                let func: extern "C" fn(f64, i64, f64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b, c))
            }
            (&[RawArg::Float(a), RawArg::Int(b), RawArg::Float(c)], _) => {
                let func: extern "C" fn(f64, i64, f64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b, c), 0.0)
            }
            (&[RawArg::Int(a), RawArg::Float(b), RawArg::Float(c)], MirType::Float) => {
                let func: extern "C" fn(i64, f64, f64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b, c))
            }
            (&[RawArg::Int(a), RawArg::Float(b), RawArg::Float(c)], _) => {
                let func: extern "C" fn(i64, f64, f64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b, c), 0.0)
            }
            (&[RawArg::Float(a), RawArg::Int(b), RawArg::Int(c)], MirType::Float) => {
                let func: extern "C" fn(f64, i64, i64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b, c))
            }
            (&[RawArg::Float(a), RawArg::Int(b), RawArg::Int(c)], _) => {
                let func: extern "C" fn(f64, i64, i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b, c), 0.0)
            }

            // 4 args (all Int/Float combinations)
            (&[ref a0, ref a1, ref a2, ref a3], ret_ty) => {
                macro_rules! call4 {
                    ($t0:ty, $v0:expr, $t1:ty, $v1:expr, $t2:ty, $v2:expr, $t3:ty, $v3:expr) => {
                        match ret_ty {
                            MirType::Float => {
                                let func: extern "C" fn($t0, $t1, $t2, $t3) -> f64 =
                                    unsafe { core::mem::transmute(code_ptr) };
                                (0, func($v0, $v1, $v2, $v3))
                            }
                            MirType::Unit => {
                                let func: extern "C" fn($t0, $t1, $t2, $t3) =
                                    unsafe { core::mem::transmute(code_ptr) };
                                func($v0, $v1, $v2, $v3);
                                (0, 0.0)
                            }
                            _ => {
                                let func: extern "C" fn($t0, $t1, $t2, $t3) -> i64 =
                                    unsafe { core::mem::transmute(code_ptr) };
                                (func($v0, $v1, $v2, $v3), 0.0)
                            }
                        }
                    };
                }
                match (a0, a1, a2, a3) {
                    (RawArg::Int(a), RawArg::Int(b), RawArg::Int(c), RawArg::Int(d)) => call4!(i64, *a, i64, *b, i64, *c, i64, *d),
                    (RawArg::Int(a), RawArg::Int(b), RawArg::Int(c), RawArg::Float(d)) => call4!(i64, *a, i64, *b, i64, *c, f64, *d),
                    (RawArg::Int(a), RawArg::Int(b), RawArg::Float(c), RawArg::Int(d)) => call4!(i64, *a, i64, *b, f64, *c, i64, *d),
                    (RawArg::Int(a), RawArg::Int(b), RawArg::Float(c), RawArg::Float(d)) => call4!(i64, *a, i64, *b, f64, *c, f64, *d),
                    (RawArg::Int(a), RawArg::Float(b), RawArg::Int(c), RawArg::Int(d)) => call4!(i64, *a, f64, *b, i64, *c, i64, *d),
                    (RawArg::Int(a), RawArg::Float(b), RawArg::Int(c), RawArg::Float(d)) => call4!(i64, *a, f64, *b, i64, *c, f64, *d),
                    (RawArg::Int(a), RawArg::Float(b), RawArg::Float(c), RawArg::Int(d)) => call4!(i64, *a, f64, *b, f64, *c, i64, *d),
                    (RawArg::Int(a), RawArg::Float(b), RawArg::Float(c), RawArg::Float(d)) => call4!(i64, *a, f64, *b, f64, *c, f64, *d),
                    (RawArg::Float(a), RawArg::Int(b), RawArg::Int(c), RawArg::Int(d)) => call4!(f64, *a, i64, *b, i64, *c, i64, *d),
                    (RawArg::Float(a), RawArg::Int(b), RawArg::Int(c), RawArg::Float(d)) => call4!(f64, *a, i64, *b, i64, *c, f64, *d),
                    (RawArg::Float(a), RawArg::Int(b), RawArg::Float(c), RawArg::Int(d)) => call4!(f64, *a, i64, *b, f64, *c, i64, *d),
                    (RawArg::Float(a), RawArg::Int(b), RawArg::Float(c), RawArg::Float(d)) => call4!(f64, *a, i64, *b, f64, *c, f64, *d),
                    (RawArg::Float(a), RawArg::Float(b), RawArg::Int(c), RawArg::Int(d)) => call4!(f64, *a, f64, *b, i64, *c, i64, *d),
                    (RawArg::Float(a), RawArg::Float(b), RawArg::Int(c), RawArg::Float(d)) => call4!(f64, *a, f64, *b, i64, *c, f64, *d),
                    (RawArg::Float(a), RawArg::Float(b), RawArg::Float(c), RawArg::Int(d)) => call4!(f64, *a, f64, *b, f64, *c, i64, *d),
                    (RawArg::Float(a), RawArg::Float(b), RawArg::Float(c), RawArg::Float(d)) => call4!(f64, *a, f64, *b, f64, *c, f64, *d),
                }
            }

            // 5 args
            (&[RawArg::Float(a), RawArg::Float(b), RawArg::Float(c), RawArg::Float(d), RawArg::Float(e)], MirType::Float) => {
                let func: extern "C" fn(f64, f64, f64, f64, f64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b, c, d, e))
            }
            (&[RawArg::Float(a), RawArg::Float(b), RawArg::Float(c), RawArg::Float(d), RawArg::Float(e)], _) => {
                let func: extern "C" fn(f64, f64, f64, f64, f64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b, c, d, e), 0.0)
            }
            (&[RawArg::Int(a), RawArg::Int(b), RawArg::Int(c), RawArg::Int(d), RawArg::Int(e)], MirType::Float) => {
                let func: extern "C" fn(i64, i64, i64, i64, i64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b, c, d, e))
            }
            (&[RawArg::Int(a), RawArg::Int(b), RawArg::Int(c), RawArg::Int(d), RawArg::Int(e)], _) => {
                let func: extern "C" fn(i64, i64, i64, i64, i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b, c, d, e), 0.0)
            }

            // 6 args
            (&[RawArg::Float(a), RawArg::Float(b), RawArg::Float(c), RawArg::Float(d), RawArg::Float(e), RawArg::Float(f)], MirType::Float) => {
                let func: extern "C" fn(f64, f64, f64, f64, f64, f64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b, c, d, e, f))
            }
            (&[RawArg::Float(a), RawArg::Float(b), RawArg::Float(c), RawArg::Float(d), RawArg::Float(e), RawArg::Float(f)], _) => {
                let func: extern "C" fn(f64, f64, f64, f64, f64, f64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b, c, d, e, f), 0.0)
            }
            (&[RawArg::Int(a), RawArg::Int(b), RawArg::Int(c), RawArg::Int(d), RawArg::Int(e), RawArg::Int(f)], MirType::Float) => {
                let func: extern "C" fn(i64, i64, i64, i64, i64, i64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b, c, d, e, f))
            }
            (&[RawArg::Int(a), RawArg::Int(b), RawArg::Int(c), RawArg::Int(d), RawArg::Int(e), RawArg::Int(f)], _) => {
                let func: extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b, c, d, e, f), 0.0)
            }

            _ => {
                *native_entered = false;
                self.cranelift_entry_count = self.cranelift_entry_count.saturating_sub(1);
                return Ok(CraneliftOutcome::Fallback);
            }
        };

        let status = crate::mir::cranelift_backend::jit_engine::lust_mir_get_status();
        drop(_jit_guard);
        drop(_inv_guard);

        let gas_after = self.mir_context.gas_remaining();
        let gas_spent = (gas_entry_snapshot.saturating_sub(gas_after)).max(0) as u64;
        let gas_charge_res = if vm_gas_budget.is_some() && gas_spent > 0 {
            self.budgets.charge_gas(gas_spent)
        } else {
            Ok(())
        };

        let mem_spent = usize::try_from(self.mir_context.memory_used().max(0)).unwrap_or(usize::MAX);
        let mem_charge_res = if vm_has_mem_budget && mem_spent > 0 {
            self.budgets.charge_mem_bytes(mem_spent)
        } else {
            Ok(())
        };

        if let Some(sticky_err) = crate::mir::cranelift_backend::take_current_sticky_error() {
            crate::mir::cranelift_backend::take_deopt_payload();
            return Err(LustError::RuntimeError {
                message: format!("Native deopt helper error: {}", sticky_err),
            });
        }

        if status != MIR_STATUS_OK {
            if status == MIR_STATUS_DEOPT {
                if let Err(err) = gas_charge_res {
                    crate::mir::cranelift_backend::take_deopt_payload();
                    if let Err(wb_err) = stage_and_commit_all_writebacks(
                        &self.mir_context,
                        &marshalled_arrays,
                        &marshalled_structs,
                    ) {
                        return Err(LustError::RuntimeError {
                            message: format!("{}; writeback failed: {}", err, wb_err),
                        });
                    }
                    return Err(err);
                }
                if let Err(err) = mem_charge_res {
                    crate::mir::cranelift_backend::take_deopt_payload();
                    if let Err(wb_err) = stage_and_commit_all_writebacks(
                        &self.mir_context,
                        &marshalled_arrays,
                        &marshalled_structs,
                    ) {
                        return Err(LustError::RuntimeError {
                            message: format!("{}; writeback failed: {}", err, wb_err),
                        });
                    }
                    return Err(err);
                }
                return self.restore_deopt_top_frame(
                    func_name,
                    frame_depth,
                    invocation_id,
                    local_types,
                    &marshalled_arrays,
                    &marshalled_structs,
                );
            }

            crate::mir::cranelift_backend::take_deopt_payload();
            let primary_err = if status == MIR_STATUS_ERR_OUT_OF_GAS {
                match gas_charge_res {
                    Err(e) => e,
                    Ok(()) => LustError::RuntimeError {
                        message: "Out of gas during Cranelift JIT execution".into(),
                    },
                }
            } else if status == MIR_STATUS_ERR_OOM {
                match mem_charge_res {
                    Err(e) => e,
                    Ok(()) => LustError::RuntimeError {
                        message: "Out of memory during Cranelift JIT execution".into(),
                    },
                }
            } else {
                let message = match status {
                    MIR_STATUS_ERR_DIV_BY_ZERO => {
                        "Division or modulo by zero during Cranelift JIT execution".into()
                    }
                    MIR_STATUS_ERR_NEG_EXPONENT => {
                        "Negative exponent in integer power during Cranelift JIT execution".into()
                    }
                    MIR_STATUS_ERR_OUT_OF_BOUNDS => {
                        "Array index or pointer out of bounds during Cranelift JIT execution".into()
                    }
                    _ => format!("Cranelift native execution failed with status {}", status),
                };
                LustError::RuntimeError { message }
            };

            if status == MIR_STATUS_ERR_DIV_BY_ZERO
                || status == MIR_STATUS_ERR_OUT_OF_GAS
                || status == MIR_STATUS_ERR_NEG_EXPONENT
                || status == MIR_STATUS_ERR_OOM
            {
                if let Err(wb_err) = stage_and_commit_all_writebacks(
                    &self.mir_context,
                    &marshalled_arrays,
                    &marshalled_structs,
                ) {
                    return Err(LustError::RuntimeError {
                        message: format!("{}; writeback failed: {}", primary_err, wb_err),
                    });
                }
            }
            return Err(primary_err);
        }

        if let Err(err) = gas_charge_res {
            if let Err(wb_err) = stage_and_commit_all_writebacks(
                &self.mir_context,
                &marshalled_arrays,
                &marshalled_structs,
            ) {
                return Err(LustError::RuntimeError {
                    message: format!("{}; writeback failed: {}", err, wb_err),
                });
            }
            return Err(err);
        }
        if let Err(err) = mem_charge_res {
            if let Err(wb_err) = stage_and_commit_all_writebacks(
                &self.mir_context,
                &marshalled_arrays,
                &marshalled_structs,
            ) {
                return Err(LustError::RuntimeError {
                    message: format!("{}; writeback failed: {}", err, wb_err),
                });
            }
            return Err(err);
        }

        // Stage writebacks for all marshalled input arrays and structs BEFORE mutating any caller state
        let staged_arrays = stage_input_writebacks(&self.mir_context, &marshalled_arrays)?;
        let staged_structs = stage_struct_writebacks(&self.mir_context, &marshalled_structs)?;

        // Reconstruct return value
        let result_val = match ret_type {
            MirType::Float => Value::Float(raw_float),
            MirType::Int => Value::Int(raw_int),
            MirType::Unknown => {
                if let Some(val) = self.mir_context.get_handle(raw_int) {
                    val
                } else {
                    Value::Int(raw_int)
                }
            }
            MirType::Bool => Value::Bool((raw_int as u8) != 0),
            MirType::Unit => Value::Nil,
            MirType::String => {
                self.mir_context.get_handle(raw_int).unwrap_or_else(|| Value::string(""))
            }
            MirType::Array(elem_ty) => {
                let decoded = validate_and_decode_array(
                    &self.mir_context,
                    &marshalled_arrays,
                    raw_int,
                    elem_ty,
                )?;
                if let Some(orig_rc) = decoded.matched_input_rc {
                    Value::Array(orig_rc)
                } else {
                    Value::Array(Rc::new(RefCell::new(decoded.elements)))
                }
            }
            MirType::Tuple(elem_types) => {
                let elems = validate_and_decode_tuple(&self.mir_context, raw_int, elem_types)?;
                Value::Tuple(Rc::new(elems))
            }
            MirType::Struct(name) => {
                let layout = self
                    .struct_metadata
                    .get(name)
                    .map(|info| &info.layout)
                    .ok_or_else(|| LustError::RuntimeError {
                        message: format!(
                            "Missing struct layout metadata for '{name}' during Cranelift return decoding"
                        ),
                    })?;
                validate_and_decode_struct(
                    &self.mir_context,
                    &marshalled_structs,
                    raw_int,
                    name,
                    Some(layout),
                )?
            }
            MirType::Map(_, _) => {
                self.mir_context.get_handle(raw_int).unwrap_or_else(|| Value::map(LustMap::default()))
            }
            MirType::Enum(name) => {
                if name == "Option" {
                    if raw_int == 0 {
                        Value::none()
                    } else if let Some(val) = self.mir_context.get_handle(raw_int) {
                        if matches!(&val, Value::Enum(_)) {
                            val
                        } else {
                            Value::some(val)
                        }
                    } else {
                        Value::some(Value::Int(raw_int))
                    }
                } else if name == "Result" {
                    if let Some(val) = self.mir_context.get_handle(raw_int) {
                        if matches!(&val, Value::Enum(_)) {
                            val
                        } else {
                            Value::ok(val)
                        }
                    } else {
                        Value::ok(Value::Int(raw_int))
                    }
                } else {
                    self.mir_context.get_handle(raw_int).unwrap_or(Value::Nil)
                }
            }
            _ => unreachable!("validated supported return type"),
        };

        // Commit all staged array and struct writes atomically
        commit_staged_writebacks_and_structs(staged_arrays, staged_structs)?;
        crate::mir::cranelift_backend::take_deopt_payload();

        Ok(CraneliftOutcome::Return(result_val))
    }

    #[cfg(not(feature = "std"))]
    pub(crate) fn try_run_cranelift_top_frame(&mut self) -> Result<Option<Option<Value>>> {
        Ok(None)
    }

    #[cfg(feature = "std")]
    #[allow(clippy::too_many_arguments)]
    fn restore_deopt_top_frame(
        &mut self,
        func_name: &str,
        frame_depth: usize,
        invocation_id: u64,
        local_types: &[MirType],
        marshalled_arrays: &[MarshalledArray],
        marshalled_structs: &[MarshalledStruct],
    ) -> Result<CraneliftOutcome> {
        #[cfg(feature = "test-deopt")]
        let (expected_block, payload, source_label) = {
            let spike_auth = match self.test_deopt_spike_hook {
                Some(ref h)
                    if h.target_function == func_name && h.triggered_count < h.max_triggers =>
                {
                    Some(h.deopt_block)
                }
                _ => None,
            };
            if let Some(expected_block) = spike_auth {
                if let Some(ref mut h) = self.test_deopt_spike_hook {
                    h.triggered_count += 1;
                }
                let (payload, payload_source) = if let Some(injected) = self
                    .test_deopt_spike_hook
                    .as_mut()
                    .and_then(|h| h.injected_payload.take())
                {
                    crate::mir::cranelift_backend::take_deopt_payload();
                    (injected, DeoptPayloadSource::TestInjectedSeam)
                } else {
                    let p = crate::mir::cranelift_backend::take_deopt_payload().ok_or_else(
                        || LustError::RuntimeError {
                            message: "Missing deopt payload during authorized spike deopt".into(),
                        },
                    )?;
                    (p, DeoptPayloadSource::NativeExecution)
                };
                self.last_deopt_source = Some(payload_source);
                let source_label = match payload_source {
                    DeoptPayloadSource::TestInjectedSeam => "test-injected seam",
                    DeoptPayloadSource::NativeExecution => "native-produced",
                };
                (Some(expected_block), payload, source_label)
            } else {
                let p =
                    crate::mir::cranelift_backend::take_deopt_payload().ok_or_else(|| {
                        LustError::RuntimeError {
                            message: "Missing deopt payload during native deopt".into(),
                        }
                    })?;
                self.last_deopt_source = Some(DeoptPayloadSource::NativeExecution);
                (None, p, "native-produced")
            }
        };

        #[cfg(not(feature = "test-deopt"))]
        let (expected_block, payload, source_label) = {
            let p = crate::mir::cranelift_backend::take_deopt_payload().ok_or_else(|| {
                LustError::RuntimeError {
                    message: "Missing deopt payload during native deopt".into(),
                }
            })?;
            (None::<usize>, p, "native-produced")
        };

        let compiled_gen = self.cranelift_compiled_generation;
        let p_ident = payload.frame_identity.as_ref().ok_or_else(|| {
            LustError::RuntimeError {
                message: format!(
                    "Deopt payload ({source_label}) missing required strict frame identity: func_name, frame_depth, or invocation_id"
                ),
            }
        })?;
        if p_ident.func_name != func_name
            || p_ident.frame_depth != frame_depth
            || p_ident.invocation_id != invocation_id
            || p_ident.compilation_generation != compiled_gen
        {
            return Err(LustError::RuntimeError {
                message: format!(
                    "Deopt payload ({source_label}) frame identity / compilation generation mismatch: expected ({}, depth {}, inv {}, gen {}), got ({}, depth {}, inv {}, gen {})",
                    func_name,
                    frame_depth,
                    invocation_id,
                    compiled_gen,
                    p_ident.func_name,
                    p_ident.frame_depth,
                    p_ident.invocation_id,
                    p_ident.compilation_generation
                ),
            });
        }

        let target_block_id = payload.target_pc;
        if let Some(exp_block) = expected_block {
            if target_block_id != exp_block {
                return Err(LustError::RuntimeError {
                    message: format!(
                        "Deopt site mismatch: expected block {}, got {} ({source_label})",
                        exp_block, target_block_id
                    ),
                });
            }
        }

        let recipe = self
            .cranelift_recipes
            .get(&(func_name.to_string(), target_block_id, payload.site_kind))
            .cloned()
            .ok_or_else(|| LustError::RuntimeError {
                message: format!(
                    "Missing compiled deopt recipe for ({}, block {}, {:?}) ({source_label})",
                    func_name, target_block_id, payload.site_kind
                ),
            })?;
        if recipe.compilation_generation != compiled_gen {
            return Err(LustError::RuntimeError {
                message: format!(
                    "Stale compiled deopt recipe generation {} (expected {}) for ({}, block {})",
                    recipe.compilation_generation, compiled_gen, func_name, target_block_id
                ),
            });
        }

        let top_frame_reg_count = self
            .call_stack
            .last()
            .map(|frame| frame.registers.len())
            .ok_or_else(|| LustError::RuntimeError {
                message: "Call stack empty during deopt reconstruction".into(),
            })?;

        let mut seen_indices = hashbrown::HashSet::with_capacity(payload.live_locals.len());
        for &(loc_idx, _) in &payload.live_locals {
            if !seen_indices.insert(loc_idx) {
                return Err(LustError::RuntimeError {
                    message: format!(
                        "Deopt payload ({source_label}) contains duplicate local index {}",
                        loc_idx
                    ),
                });
            }
        }

        for &(req_idx, _) in &recipe.proven_locals {
            if !seen_indices.contains(&req_idx) {
                return Err(LustError::RuntimeError {
                    message: format!(
                        "Deopt payload ({source_label}) missing required local {} at deopt site",
                        req_idx
                    ),
                });
            }
        }

        for &loc_idx in &seen_indices {
            if !recipe.proven_locals.iter().any(|(idx, _)| *idx == loc_idx) {
                return Err(LustError::RuntimeError {
                    message: format!(
                        "Deopt payload ({source_label}) contains unexpected local {} not in authoritative site metadata",
                        loc_idx
                    ),
                });
            }
        }

        let mut materialized_fresh_arrays: hashbrown::HashMap<usize, Rc<RefCell<Vec<Value>>>> =
            hashbrown::HashMap::new();
        let mut validated_restorations = Vec::with_capacity(payload.live_locals.len());
        for &(loc_idx, raw_val) in &payload.live_locals {
            if loc_idx >= local_types.len() || loc_idx >= top_frame_reg_count {
                return Err(LustError::RuntimeError {
                    message: format!(
                        "Deopt payload ({source_label}) local index {} out of bounds",
                        loc_idx
                    ),
                });
            }
            let mir_ty = &local_types[loc_idx];
            let restored_val = match mir_ty {
                MirType::Int => Value::Int(raw_val),
                MirType::Bool => Value::Bool(raw_val != 0),
                MirType::Float => Value::Float(f64::from_bits(raw_val as u64)),
                MirType::Unit => Value::Nil,
                MirType::String => match self.mir_context.get_handle(raw_val) {
                    Some(Value::String(s_rc)) => Value::String(Rc::clone(&s_rc)),
                    Some(other) => {
                        return Err(LustError::RuntimeError {
                            message: format!(
                                "Deopt payload ({source_label}) invalid handle {} for String local {}: expected String, got {:?}",
                                raw_val, loc_idx, other
                            ),
                        });
                    }
                    None => {
                        return Err(LustError::RuntimeError {
                            message: format!(
                                "Deopt payload ({source_label}) missing String handle {} for local {}",
                                raw_val, loc_idx
                            ),
                        });
                    }
                },
                MirType::Array(inner)
                    if matches!(&**inner, MirType::Int | MirType::Bool | MirType::Float) =>
                {
                    let decoded = validate_and_decode_array(
                        &self.mir_context,
                        marshalled_arrays,
                        raw_val,
                        inner,
                    )?;
                    if let Some(orig_rc) = decoded.matched_input_rc {
                        Value::Array(orig_rc)
                    } else {
                        let rc = materialized_fresh_arrays
                            .entry(decoded._ptr as usize)
                            .or_insert_with(|| Rc::new(RefCell::new(decoded.elements)))
                            .clone();
                        Value::Array(rc)
                    }
                }
                _ => {
                    return Err(LustError::RuntimeError {
                        message: format!(
                            "Unsupported non-scalar type {:?} for deopt local {} in ({source_label})",
                            mir_ty, loc_idx
                        ),
                    });
                }
            };
            validated_restorations.push((loc_idx, restored_val));
        }

        stage_and_commit_all_writebacks(&self.mir_context, marshalled_arrays, marshalled_structs)?;

        let top_frame = self.call_stack.last_mut().expect("validated non-empty");
        top_frame.ip = recipe.precise_pc;
        for (loc_idx, val) in validated_restorations {
            top_frame.registers[loc_idx] = val;
        }

        Ok(CraneliftOutcome::Fallback)
    }

    /// Executes the active call frame using the native Cranelift JIT compiler backend.
    #[cfg(feature = "std")]
    pub fn run_cranelift(&mut self) -> Result<Value> {
        if let Some(finished) = self.try_run_cranelift_top_frame()? {
            if let Some(val) = finished {
                return Ok(val);
            }
            if let Some(target_depth) = self.call_until_depth
                && self.call_stack.len() == target_depth
                && let Some(return_value) = self.pending_return_value.take()
            {
                self.call_until_depth = None;
                return Ok(return_value);
            }
        }
        let result = if self.effective_jit_enabled() {
            self.run_inner::<true>()
        } else {
            self.run_inner::<false>()
        };
        crate::bytecode::value::sync_all_specialized_arrays();
        result
    }

    /// In no_std mode, fall back to interpreter dispatch.
    #[cfg(not(feature = "std"))]
    pub fn run_cranelift(&mut self) -> Result<Value> {
        self.run_inner::<false>()
    }
}
