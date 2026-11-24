//! Cranelift JIT dispatch, eligibility checks, and compound value marshaling for the Lust VM.

use crate::ast::types::TypeKind;
use crate::bytecode::value::UnboxedArrayData;
use crate::bytecode::{
    FlatArrayBuffer, FlatArrayElemTag, LustArray, LustMap, StructLayout, StructObject, Value,
};
use crate::error::{LustError, Result};
#[cfg(feature = "std")]
use crate::mir::cranelift_backend::{
    DeoptSiteKind, MIR_STATUS_DEOPT, MIR_STATUS_ERR_DIV_BY_ZERO, MIR_STATUS_ERR_NEG_EXPONENT,
    MIR_STATUS_ERR_OOM, MIR_STATUS_ERR_OUT_OF_BOUNDS, MIR_STATUS_ERR_OUT_OF_GAS, MIR_STATUS_OK,
    apply_module_callee_return_types, compute_default_module_roots,
    compute_reachable_native_functions, find_loop_headers, get_registered_struct_layout_by_name,
    is_effect_or_handler_callee, is_vm_effect_intrinsic_callee, lust_mir_alloc_array_typed,
    lust_mir_alloc_tuple_typed, mir_array_elem_tag,
};
#[cfg(all(feature = "std", test))]
use crate::mir::cranelift_backend::lust_mir_alloc_struct_named;
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

#[derive(Debug, Clone, Copy)]
pub enum RawArg {
    Int(i64),
    Float(f64),
}

#[derive(Debug, Clone)]
pub struct MarshalledArray {
    pub rc: Rc<LustArray>,
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

#[derive(Debug)]
enum CraneliftOutcome {
    Fallback,
    Return(Value),
}

#[cfg(feature = "std")]
struct PreparedCraneliftModule {
    module: crate::mir::ir::MirModule,
    eligible: hashbrown::HashSet<String>,
    func_idx_by_name: hashbrown::HashMap<String, usize>,
    eager_continuations: hashbrown::HashMap<String, crate::bytecode::Function>,
    deferred_continuations: hashbrown::HashSet<String>,
    deferred_dead_helpers: hashbrown::HashSet<String>,
    effective_roots: hashbrown::HashSet<String>,
    skipped_dead_helper_count: usize,
}

#[derive(Debug)]
pub struct DecodedArray {
    pub matched_input_rc: Option<Rc<LustArray>>,
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

    if raw_ptr_val == 0 {
        return Err(LustError::RuntimeError {
            message: "Invalid array pointer provenance: null pointer (0x0) is not an owned allocation base".into(),
        });
    }
    if raw_ptr_val < 0 {
        return Err(LustError::RuntimeError {
            message: format!(
                "Invalid array pointer provenance: negative pointer {}",
                raw_ptr_val
            ),
        });
    }

    let raw_addr = raw_ptr_val as usize;
    let alloc_rec = mir_ctx
        .get_active_allocation_record(raw_addr)
        .ok_or_else(|| LustError::RuntimeError {
            message: format!(
                "Invalid array pointer provenance: pointer {:#x} is not an owned allocation base of current invocation",
                raw_addr
            ),
        })?;

    match &alloc_rec.kind {
        crate::mir::runtime_helpers::MirAllocationKind::Array(existing_elem_ty) => {
            if existing_elem_ty != expected_elem_ty {
                return Err(LustError::RuntimeError {
                    message: format!(
                        "Array element type mismatch: expected {:?}, found {:?}",
                        expected_elem_ty, existing_elem_ty
                    ),
                });
            }
        }
        other => {
            return Err(LustError::RuntimeError {
                message: format!(
                    "Type/kind mismatch: expected Array({:?}) but allocation is {:?}",
                    expected_elem_ty, other
                ),
            });
        }
    }

    if alloc_rec.word_capacity < 2 {
        return Err(LustError::RuntimeError {
            message: format!(
                "Malformed array allocation: word_capacity ({}) < 2",
                alloc_rec.word_capacity
            ),
        });
    }

    let base_ptr = alloc_rec.base_ptr;
    let len_raw = unsafe { *base_ptr };
    let cap_raw = unsafe { *base_ptr.add(1) };
    let is_bool = matches!(expected_elem_ty, MirType::Bool);
    let (len, data_ptr) = if cap_raw < 0 {
        let (len, _cap, data_ptr) = mir_ctx
            .resolve_validated_array_data(base_ptr)
            .ok_or_else(|| LustError::RuntimeError {
                message: format!(
                    "Malformed array header: negative cap ({cap_raw}) or invalid redirected backing buffer at {:#x} (word_capacity={})",
                    raw_addr, alloc_rec.word_capacity
                ),
            })?;
        (len, data_ptr)
    } else {
        if len_raw < 0 {
            return Err(LustError::RuntimeError {
                message: format!(
                    "Malformed array header: negative len ({len_raw}) or cap ({cap_raw})"
                ),
            });
        }
        let len = len_raw as usize;
        let cap = cap_raw as usize;
        if len > cap {
            return Err(LustError::RuntimeError {
                message: format!("Malformed array header: len ({len}) > cap ({cap})"),
            });
        }
        let needed_payload_words = if is_bool {
            cap.checked_add(7).map(|n| n / 8)
        } else {
            Some(cap)
        };
        if needed_payload_words
            .and_then(|w| w.checked_add(2))
            .is_none_or(|needed| needed > alloc_rec.word_capacity)
        {
            return Err(LustError::RuntimeError {
                message: format!(
                    "Malformed array header: extent ({} words) exceeds owned buffer capacity ({} words)",
                    needed_payload_words.unwrap_or(cap).saturating_add(2),
                    alloc_rec.word_capacity
                ),
            });
        }
        (len, unsafe { base_ptr.add(2) as *const i64 })
    };

    // Element decoding (supports both flat inline and cap < 0 redirected backing buffers;
    // Bool arrays use 1-byte u8 element stride per NG-W10-5).
    let mut elements = Vec::with_capacity(len);
    if is_bool {
        let byte_ptr = data_ptr as *const u8;
        for i in 0..len {
            let raw = unsafe { *byte_ptr.add(i) };
            elements.push(Value::Bool(raw != 0));
        }
    } else {
        for i in 0..len {
            let raw = unsafe { *data_ptr.add(i) };
            let elem = match expected_elem_ty {
                MirType::Int => Value::Int(raw),
                MirType::Float => Value::Float(f64::from_bits(raw as u64)),
                MirType::String => mir_ctx.get_handle(raw).unwrap_or_else(|| Value::string("")),
                MirType::Struct(name) => {
                    if let Some(val @ Value::Struct(_)) = mir_ctx.get_handle(raw) {
                        val
                    } else {
                        validate_and_decode_struct(mir_ctx, &[], raw, name, None)?
                    }
                }
                _ => unreachable!("validated array element type"),
            };
            elements.push(elem);
        }
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
        if !VM::is_cranelift_supported_primitive_elem_type(elem_ty) {
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

pub const MAX_STRUCT_DECODE_DEPTH: usize = 256;

fn resolve_option_struct_target_name<'a>(
    owner_struct_name: &str,
    inner_name: &'a str,
    struct_meta: Option<&hashbrown::HashMap<String, crate::vm::RuntimeStructInfo>>,
) -> alloc::borrow::Cow<'a, str> {
    if !inner_name.contains('.')
        && let Some((mod_prefix, _)) = owner_struct_name.rsplit_once('.')
    {
        let qualified = format!("{mod_prefix}.{inner_name}");
        if struct_meta.is_some_and(|m| m.contains_key(&qualified)) {
            return alloc::borrow::Cow::Owned(qualified);
        }
    }
    alloc::borrow::Cow::Borrowed(inner_name)
}

fn validate_and_decode_struct_inner(
    mir_ctx: &crate::mir::runtime_helpers::MirContext,
    marshalled_structs: &[MarshalledStruct],
    raw_ptr_val: i64,
    expected_name: &str,
    layout_opt: Option<&Rc<StructLayout>>,
    struct_meta: Option<&hashbrown::HashMap<String, crate::vm::RuntimeStructInfo>>,
    visited_ptrs: &mut hashbrown::HashSet<usize>,
    materialized_fresh_structs: &mut hashbrown::HashMap<usize, Rc<StructObject>>,
    depth: usize,
) -> Result<Value> {
    if depth > MAX_STRUCT_DECODE_DEPTH {
        return Err(LustError::RuntimeError {
            message: format!(
                "Struct decoding depth limit ({MAX_STRUCT_DECODE_DEPTH}) exceeded for '{expected_name}'"
            ),
        });
    }

    let (base_ptr, len) = validate_owned_header(mir_ctx, raw_ptr_val, "struct", |kind| {
        match kind {
            crate::mir::runtime_helpers::MirAllocationKind::Struct(existing_name) => {
                let existing_name: &str = existing_name.as_ref();
                let layout_matches = layout_opt.is_some_and(|l| existing_name == l.name());
                let simple_expected = expected_name
                    .rsplit_once('.')
                    .map(|(_, s)| s)
                    .unwrap_or(expected_name);
                let simple_existing = existing_name
                    .rsplit_once('.')
                    .map(|(_, s)| s)
                    .unwrap_or(existing_name);
                let qualified_same = layout_matches
                    || (existing_name != expected_name
                        && existing_name == simple_expected
                        && struct_meta.is_some_and(|m| {
                            m.get(expected_name)
                                .zip(m.get(existing_name))
                                .is_some_and(|(a, b)| Rc::ptr_eq(&a.layout, &b.layout))
                        }))
                    || (existing_name != expected_name
                        && expected_name == simple_existing
                        && struct_meta.is_some_and(|m| {
                            m.get(expected_name)
                                .zip(m.get(existing_name))
                                .is_some_and(|(a, b)| Rc::ptr_eq(&a.layout, &b.layout))
                        }));
                if existing_name != expected_name && !qualified_same {
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

    let base_addr = base_ptr as usize;
    if !visited_ptrs.insert(base_addr) {
        return Err(LustError::RuntimeError {
            message: format!(
                "Cyclic struct pointer graph detected at {:#x} while decoding '{expected_name}'",
                base_addr
            ),
        });
    }

    if let Some(matched) = marshalled_structs.iter().find(|m| m.ptr == base_ptr) {
        visited_ptrs.remove(&base_addr);
        return Ok(Value::Struct(matched.rc.clone()));
    }

    if let Some(cached_rc) = materialized_fresh_structs.get(&base_addr) {
        visited_ptrs.remove(&base_addr);
        return Ok(Value::Struct(cached_rc.clone()));
    }

    let decode_fields_res: Result<Vec<Value>> = (|| {
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
                        TypeKind::Option(inner) => match &inner.kind {
                            TypeKind::Named(sub) => {
                                if raw == 0 {
                                    Value::none()
                                } else {
                                    let sub_name = resolve_option_struct_target_name(
                                        expected_name,
                                        sub,
                                        struct_meta,
                                    );
                                    let sub_layout: Option<&Rc<StructLayout>> =
                                        if depth > 0
                                            && sub_name.as_ref() == expected_name
                                            && layout_opt.is_some()
                                        {
                                            layout_opt
                                        } else if let Some(meta) = struct_meta {
                                            meta.get(sub_name.as_ref()).map(|info| &info.layout)
                                        } else if layout.name() == sub.as_str()
                                            || expected_name == sub.as_str()
                                        {
                                            Some(layout)
                                        } else {
                                            None
                                        };
                                    let decoded_sub = validate_and_decode_struct_inner(
                                        mir_ctx,
                                        marshalled_structs,
                                        raw,
                                        sub_name.as_ref(),
                                        sub_layout,
                                        struct_meta,
                                        visited_ptrs,
                                        materialized_fresh_structs,
                                        depth + 1,
                                    )?;
                                    Value::some(decoded_sub)
                                }
                            }
                            _ => mir_ctx.get_handle(raw).unwrap_or(Value::Nil),
                        },
                        TypeKind::Result(_, _) => {
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
        Ok(fields)
    })();

    visited_ptrs.remove(&base_addr);
    let fields = decode_fields_res?;

    let layout = layout_opt.cloned().unwrap_or_else(|| {
        Rc::new(StructLayout::new(
            expected_name.to_string(),
            vec![],
            vec![],
            vec![],
            vec![],
        ))
    });
    let struct_name = if layout.name().is_empty() {
        expected_name.to_string()
    } else {
        layout.name().to_string()
    };
    let rc = StructObject::new(struct_name, layout, fields);
    materialized_fresh_structs.insert(base_addr, rc.clone());
    Ok(Value::Struct(rc))
}

pub(super) fn validate_and_decode_struct_with_metadata(
    mir_ctx: &crate::mir::runtime_helpers::MirContext,
    marshalled_structs: &[MarshalledStruct],
    raw_ptr_val: i64,
    expected_name: &str,
    layout_opt: Option<&Rc<StructLayout>>,
    struct_meta: Option<&hashbrown::HashMap<String, crate::vm::RuntimeStructInfo>>,
) -> Result<Value> {
    let mut visited_ptrs = hashbrown::HashSet::new();
    let mut materialized_fresh_structs = hashbrown::HashMap::new();
    validate_and_decode_struct_inner(
        mir_ctx,
        marshalled_structs,
        raw_ptr_val,
        expected_name,
        layout_opt,
        struct_meta,
        &mut visited_ptrs,
        &mut materialized_fresh_structs,
        0,
    )
}

pub fn validate_and_decode_struct(
    mir_ctx: &crate::mir::runtime_helpers::MirContext,
    marshalled_structs: &[MarshalledStruct],
    raw_ptr_val: i64,
    expected_name: &str,
    layout_opt: Option<&Rc<StructLayout>>,
) -> Result<Value> {
    validate_and_decode_struct_with_metadata(
        mir_ctx,
        marshalled_structs,
        raw_ptr_val,
        expected_name,
        layout_opt,
        None,
    )
}

#[inline]
fn param_indices_need_writeback(param_indices: &[usize], readonly_params: &[bool]) -> bool {
    if param_indices.is_empty() {
        return true;
    }
    param_indices
        .iter()
        .any(|&p| !readonly_params.get(p).copied().unwrap_or(false))
}

pub fn stage_input_writebacks_with_readonly(
    mir_ctx: &crate::mir::runtime_helpers::MirContext,
    marshalled_arrays: &[MarshalledArray],
    readonly_params: &[bool],
) -> Result<Vec<(Rc<LustArray>, Vec<Value>)>> {
    let mut staged = Vec::new();
    for m in marshalled_arrays {
        if !param_indices_need_writeback(&m.param_indices, readonly_params) {
            continue;
        }
        let decoded =
            validate_and_decode_array(mir_ctx, marshalled_arrays, m.ptr as i64, &m.elem_ty)?;
        staged.push((m.rc.clone(), decoded.elements));
    }
    Ok(staged)
}

pub fn stage_input_writebacks(
    mir_ctx: &crate::mir::runtime_helpers::MirContext,
    marshalled_arrays: &[MarshalledArray],
) -> Result<Vec<(Rc<LustArray>, Vec<Value>)>> {
    stage_input_writebacks_with_readonly(mir_ctx, marshalled_arrays, &[])
}

fn stage_struct_writebacks_with_readonly_and_meta(
    mir_ctx: &crate::mir::runtime_helpers::MirContext,
    marshalled_structs: &[MarshalledStruct],
    readonly_params: &[bool],
    struct_meta: Option<&hashbrown::HashMap<String, crate::vm::RuntimeStructInfo>>,
) -> Result<Vec<(Rc<StructObject>, Vec<Value>)>> {
    let mut staged = Vec::new();
    let mut materialized_fresh_structs = hashbrown::HashMap::new();
    for m in marshalled_structs {
        if !param_indices_need_writeback(&m.param_indices, readonly_params) {
            continue;
        }
        let (base_ptr, len) = validate_owned_header(mir_ctx, m.ptr as i64, "struct", |kind| {
            match kind {
                crate::mir::runtime_helpers::MirAllocationKind::Struct(alloc_name) => {
                    let alloc_name: &str = alloc_name.as_ref();
                    let simple_name = m
                        .rc
                        .name
                        .rsplit_once('.')
                        .map(|(_, s)| s)
                        .unwrap_or(m.rc.name.as_str());
                    if alloc_name != m.rc.name.as_str() && alloc_name != simple_name {
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
            let opt_struct_sub = (i < m.rc.layout.field_names().len())
                .then(|| &m.rc.layout.field_type(i).kind)
                .and_then(|k| match k {
                    TypeKind::Option(inner) => match &inner.kind {
                        TypeKind::Named(sub) => Some(sub.as_str()),
                        _ => None,
                    },
                    _ => None,
                });
            if let Some(sub) = opt_struct_sub {
                if raw == 0 {
                    *field = Value::none();
                } else {
                    let sub_name = resolve_option_struct_target_name(
                        m.rc.name.as_str(),
                        sub,
                        struct_meta,
                    );
                    let sub_layout: Option<&Rc<StructLayout>> = if let Some(meta) = struct_meta {
                        meta.get(sub_name.as_ref()).map(|info| &info.layout)
                    } else if m.rc.layout.name() == sub || m.rc.name.as_str() == sub {
                        Some(&m.rc.layout)
                    } else {
                        None
                    };
                    let mut visited_ptrs = hashbrown::HashSet::new();
                    visited_ptrs.insert(base_ptr as usize);
                    let decoded_sub = validate_and_decode_struct_inner(
                        mir_ctx,
                        marshalled_structs,
                        raw,
                        sub_name.as_ref(),
                        sub_layout,
                        struct_meta,
                        &mut visited_ptrs,
                        &mut materialized_fresh_structs,
                        1,
                    )?;
                    *field = Value::some(decoded_sub);
                }
                continue;
            }
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

pub fn stage_struct_writebacks_with_readonly(
    mir_ctx: &crate::mir::runtime_helpers::MirContext,
    marshalled_structs: &[MarshalledStruct],
    readonly_params: &[bool],
) -> Result<Vec<(Rc<StructObject>, Vec<Value>)>> {
    stage_struct_writebacks_with_readonly_and_meta(mir_ctx, marshalled_structs, readonly_params, None)
}

pub fn stage_struct_writebacks(
    mir_ctx: &crate::mir::runtime_helpers::MirContext,
    marshalled_structs: &[MarshalledStruct],
) -> Result<Vec<(Rc<StructObject>, Vec<Value>)>> {
    stage_struct_writebacks_with_readonly(mir_ctx, marshalled_structs, &[])
}

#[cfg(feature = "std")]
thread_local! {
    static CRANELIFT_WRITEBACK_COMMIT_COUNT: core::cell::Cell<u64> =
        const { core::cell::Cell::new(0) };
}

#[cfg(feature = "std")]
pub fn cranelift_writeback_commit_count() -> u64 {
    CRANELIFT_WRITEBACK_COMMIT_COUNT.with(|c| c.get())
}

#[cfg(feature = "std")]
pub fn reset_cranelift_writeback_commit_count() {
    CRANELIFT_WRITEBACK_COMMIT_COUNT.with(|c| c.set(0));
}

#[cfg(feature = "test-deopt")]
thread_local! {
    static LAST_DEOPT_SROA_MATERIALIZATIONS: core::cell::RefCell<Vec<(usize, Value)>> =
        const { core::cell::RefCell::new(Vec::new()) };
}

#[cfg(feature = "test-deopt")]
pub fn take_last_deopt_sroa_materializations() -> Vec<(usize, Value)> {
    LAST_DEOPT_SROA_MATERIALIZATIONS.with(|slot| core::mem::take(&mut *slot.borrow_mut()))
}

pub fn commit_staged_writebacks_and_structs(
    staged_arrays: Vec<(Rc<LustArray>, Vec<Value>)>,
    staged_structs: Vec<(Rc<StructObject>, Vec<Value>)>,
) -> Result<()> {
    if staged_arrays.is_empty() && staged_structs.is_empty() {
        return Ok(());
    }
    // 1. Acquire all mutable borrows for arrays and structs before publishing any changes
    let array_rcs: Vec<Rc<LustArray>> =
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

    #[cfg(feature = "std")]
    CRANELIFT_WRITEBACK_COMMIT_COUNT.with(|c| c.set(c.get().wrapping_add(1)));

    Ok(())
}

pub fn commit_staged_writebacks(
    staged: Vec<(Rc<LustArray>, Vec<Value>)>,
) -> Result<()> {
    commit_staged_writebacks_and_structs(staged, Vec::new())
}

fn stage_and_commit_all_writebacks_with_readonly(
    mir_ctx: &crate::mir::runtime_helpers::MirContext,
    marshalled_arrays: &[MarshalledArray],
    marshalled_structs: &[MarshalledStruct],
    readonly_params: &[bool],
    struct_meta: Option<&hashbrown::HashMap<String, crate::vm::RuntimeStructInfo>>,
) -> Result<()> {
    let staged_arrays =
        stage_input_writebacks_with_readonly(mir_ctx, marshalled_arrays, readonly_params)?;
    let staged_structs = stage_struct_writebacks_with_readonly_and_meta(
        mir_ctx,
        marshalled_structs,
        readonly_params,
        struct_meta,
    )?;
    commit_staged_writebacks_and_structs(staged_arrays, staged_structs)
}

#[cfg(feature = "std")]
#[inline]
fn is_proven_immutable_scalar(ty: &MirType) -> bool {
    matches!(
        ty,
        MirType::Int | MirType::Float | MirType::Bool | MirType::Unit | MirType::String
    )
}

#[cfg(feature = "std")]
#[inline]
fn is_readonly_cranelift_builtin(callee: &str) -> bool {
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
            | "string.sub"
            | "string.byte"
            | "string.byte_at"
            | "string.gsub"
            | "string.split"
            | "string.find"
            | "tostring"
            | "map.len"
            | "map.has"
            | "enum.tag"
            | "enum.payload"
            | "enum.payload_at"
    )
}

#[cfg(feature = "std")]
fn operand_alias_source_local(
    op: &Operand,
    locals: &[crate::mir::ir::LocalDecl],
    struct_meta: Option<&hashbrown::HashMap<String, crate::vm::RuntimeStructInfo>>,
) -> Option<usize> {
    use crate::mir::ir::Place;
    match op {
        Operand::Copy(place) | Operand::Move(place) => match place {
            Place::Local(loc) => Some(loc.0),
            Place::Field(..) | Place::Index(..) => {
                let projected_ty = VM::mir_place_type_in_vm(place, locals, struct_meta)
                    .unwrap_or(MirType::Unknown);
                if is_proven_immutable_scalar(&projected_ty) {
                    None
                } else {
                    Some(place.root_local().0)
                }
            }
        },
        Operand::Constant(_) => None,
    }
}

#[cfg(feature = "std")]
#[inline]
fn union_bool_row(dst: &mut [bool], src: &[bool]) -> bool {
    let mut changed = false;
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        if s && !*d {
            *d = true;
            changed = true;
        }
    }
    changed
}

/// Computes per-function proven read-only parameter masks over the final post-inline
/// compiled `MirModule` via monotone fixed-point alias and mutation propagation.
#[cfg(feature = "std")]
pub fn compute_readonly_params(
    module: &crate::mir::ir::MirModule,
) -> hashbrown::HashMap<String, Vec<bool>> {
    compute_readonly_params_with_structs(module, None)
}

#[cfg(feature = "std")]
fn compute_readonly_params_with_structs(
    module: &crate::mir::ir::MirModule,
    struct_meta: Option<&hashbrown::HashMap<String, crate::vm::RuntimeStructInfo>>,
) -> hashbrown::HashMap<String, Vec<bool>> {
    use crate::mir::ir::Place;

    let fn_count = module.functions.len();
    let mut fn_index_by_name: hashbrown::HashMap<&str, Option<usize>> =
        hashbrown::HashMap::with_capacity(fn_count);
    for (fi, f) in module.functions.iter().enumerate() {
        fn_index_by_name
            .entry(f.signature.name.as_str())
            .and_modify(|slot| *slot = None)
            .or_insert(Some(fi));
    }

    let mut alias_params: Vec<Vec<Vec<bool>>> = Vec::with_capacity(fn_count);
    let mut mutated: Vec<Vec<bool>> = Vec::with_capacity(fn_count);
    let mut returns_param: Vec<Vec<bool>> = Vec::with_capacity(fn_count);

    for f in &module.functions {
        let p_len = f.params.len();
        let l_len = f.locals.len();
        let mut fn_aliases = vec![vec![false; p_len]; l_len];
        let mut fn_mutated = vec![false; p_len];
        for (pi, param_loc) in f.params.iter().enumerate() {
            if param_loc.0 < l_len {
                fn_aliases[param_loc.0][pi] = true;
            } else {
                // Malformed MIR parameter local index: fail conservative.
                fn_mutated[pi] = true;
            }
        }
        alias_params.push(fn_aliases);
        mutated.push(fn_mutated);
        returns_param.push(vec![false; p_len]);
    }

    loop {
        let mut changed = false;
        for (fi, f) in module.functions.iter().enumerate() {
            let l_len = f.locals.len();
            let p_len = f.params.len();
            if p_len == 0 {
                continue;
            }

            for block in &f.blocks {
                for stmt in &block.statements {
                    let Statement::Assign(lhs, rval, _) = stmt else {
                        continue;
                    };
                    match lhs {
                        Place::Local(dst) => match rval {
                            Rvalue::Use(op) | Rvalue::Cast(op, _) => {
                                if dst.0 < l_len {
                                    let maybe_src = match op {
                                        Operand::Copy(Place::Local(src))
                                        | Operand::Move(Place::Local(src)) => Some(src.0),
                                        Operand::Copy(place @ (Place::Field(..) | Place::Index(..)))
                                        | Operand::Move(place @ (Place::Field(..) | Place::Index(..))) => {
                                            let dst_scalar =
                                                is_proven_immutable_scalar(&f.locals[dst.0].ty);
                                            let proj_scalar = VM::mir_place_type_in_vm(
                                                place,
                                                &f.locals,
                                                struct_meta,
                                            )
                                            .is_some_and(|t| is_proven_immutable_scalar(&t));
                                            if dst_scalar || proj_scalar {
                                                None
                                            } else {
                                                Some(place.root_local().0)
                                            }
                                        }
                                        Operand::Constant(_) => None,
                                    };
                                    if let Some(src) = maybe_src
                                        && src < l_len
                                        && src != dst.0
                                    {
                                        let src_bits = alias_params[fi][src].clone();
                                        if union_bool_row(&mut alias_params[fi][dst.0], &src_bits) {
                                            changed = true;
                                        }
                                    }
                                }
                            }
                            Rvalue::Aggregate(_, elems) => {
                                // Condition (c): storing a parameter alias into another aggregate escapes it.
                                for elem in elems {
                                    if let Some(src) =
                                        operand_alias_source_local(elem, &f.locals, struct_meta)
                                        && src < l_len
                                    {
                                        if union_bool_row(&mut mutated[fi], &alias_params[fi][src])
                                        {
                                            changed = true;
                                        }
                                    }
                                }
                            }
                            Rvalue::Closure { captures, .. } => {
                                for (_, op, _) in captures {
                                    if let Some(src) =
                                        operand_alias_source_local(op, &f.locals, struct_meta)
                                        && src < l_len
                                    {
                                        if union_bool_row(&mut mutated[fi], &alias_params[fi][src])
                                        {
                                            changed = true;
                                        }
                                    }
                                }
                            }
                            Rvalue::BinaryOp(..) | Rvalue::UnaryOp(..) => {}
                        },
                        Place::Field(..) | Place::Index(..) => {
                            // Condition (a): projected write mutates root base local.
                            let root = lhs.root_local().0;
                            if root < l_len
                                && union_bool_row(&mut mutated[fi], &alias_params[fi][root])
                            {
                                changed = true;
                            }
                            // Condition (c): storing a parameter alias into a projected field/index escapes it.
                            match rval {
                                Rvalue::Use(op) | Rvalue::Cast(op, _) => {
                                    if let Some(src) =
                                        operand_alias_source_local(op, &f.locals, struct_meta)
                                        && src < l_len
                                    {
                                        if union_bool_row(&mut mutated[fi], &alias_params[fi][src])
                                        {
                                            changed = true;
                                        }
                                    }
                                }
                                Rvalue::Aggregate(_, elems) => {
                                    for elem in elems {
                                        if let Some(src) =
                                            operand_alias_source_local(elem, &f.locals, struct_meta)
                                            && src < l_len
                                        {
                                            if union_bool_row(
                                                &mut mutated[fi],
                                                &alias_params[fi][src],
                                            ) {
                                                changed = true;
                                            }
                                        }
                                    }
                                }
                                Rvalue::Closure { captures, .. } => {
                                    for (_, op, _) in captures {
                                        if let Some(src) =
                                            operand_alias_source_local(op, &f.locals, struct_meta)
                                            && src < l_len
                                        {
                                            if union_bool_row(
                                                &mut mutated[fi],
                                                &alias_params[fi][src],
                                            ) {
                                                changed = true;
                                            }
                                        }
                                    }
                                }
                                Rvalue::BinaryOp(..) | Rvalue::UnaryOp(..) => {}
                            }
                        }
                    }
                }

                match &block.terminator {
                    Some(Terminator::Return(Some(op), _)) => {
                        if let Some(src) = operand_alias_source_local(op, &f.locals, struct_meta)
                            && src < l_len
                        {
                            if union_bool_row(&mut returns_param[fi], &alias_params[fi][src]) {
                                changed = true;
                            }
                        }
                    }
                    Some(Terminator::Raise(op, _)) => {
                        if let Some(src) = operand_alias_source_local(op, &f.locals, struct_meta)
                            && src < l_len
                        {
                            if union_bool_row(&mut mutated[fi], &alias_params[fi][src]) {
                                changed = true;
                            }
                        }
                    }
                    Some(Terminator::Call {
                        callee,
                        args,
                        destination,
                        ..
                    }) => {
                        // Condition (a): projected destination of a Call mutates its root base local.
                        if matches!(destination, Place::Field(..) | Place::Index(..)) {
                            let root = destination.root_local().0;
                            if root < l_len
                                && union_bool_row(&mut mutated[fi], &alias_params[fi][root])
                            {
                                changed = true;
                            }
                        }

                        if is_effect_or_handler_callee(callee) {
                            // Condition (e): effect/handler callees are conservatively mutating.
                            for arg in args {
                                if let Some(src) =
                                    operand_alias_source_local(arg, &f.locals, struct_meta)
                                    && src < l_len
                                {
                                    if union_bool_row(&mut mutated[fi], &alias_params[fi][src]) {
                                        changed = true;
                                    }
                                }
                            }
                        } else if is_readonly_cranelift_builtin(callee) {
                            // Proven pure read-only scalar-returning builtin.
                        } else if let Some(Some(gi)) =
                            fn_index_by_name.get(callee.as_str()).copied()
                        {
                            let g_param_len = module.functions[gi].params.len();
                            if args.len() != g_param_len {
                                // Arity mismatch: fail conservative.
                                for arg in args {
                                    if let Some(src) =
                                        operand_alias_source_local(arg, &f.locals, struct_meta)
                                        && src < l_len
                                    {
                                        if union_bool_row(&mut mutated[fi], &alias_params[fi][src])
                                        {
                                            changed = true;
                                        }
                                    }
                                }
                            } else {
                                for (k, arg) in args.iter().enumerate() {
                                    if let Some(src) =
                                        operand_alias_source_local(arg, &f.locals, struct_meta)
                                        && src < l_len
                                    {
                                        // Condition (d): callee mutates argument k.
                                        if mutated[gi][k]
                                            && union_bool_row(
                                                &mut mutated[fi],
                                                &alias_params[fi][src],
                                            )
                                        {
                                            changed = true;
                                        }
                                        // Callee returns an alias of argument k.
                                        if returns_param[gi][k] {
                                            match destination {
                                                Place::Local(dst) if dst.0 < l_len => {
                                                    if src != dst.0 {
                                                        let src_bits = alias_params[fi][src].clone();
                                                        if union_bool_row(
                                                            &mut alias_params[fi][dst.0],
                                                            &src_bits,
                                                        ) {
                                                            changed = true;
                                                        }
                                                    }
                                                }
                                                Place::Field(..) | Place::Index(..) => {
                                                    if union_bool_row(
                                                        &mut mutated[fi],
                                                        &alias_params[fi][src],
                                                    ) {
                                                        changed = true;
                                                    }
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            // Conditions (b) & (e): mutating builtins (array.push, table.push,
                            // map.set, etc.) and unknown/external callees are conservatively mutating.
                            for arg in args {
                                if let Some(src) =
                                    operand_alias_source_local(arg, &f.locals, struct_meta)
                                    && src < l_len
                                {
                                    if union_bool_row(&mut mutated[fi], &alias_params[fi][src]) {
                                        changed = true;
                                    }
                                }
                            }
                        }
                    }
                    Some(
                        Terminator::Goto(..)
                        | Terminator::Cond { .. }
                        | Terminator::Switch { .. }
                        | Terminator::Return(None, _)
                        | Terminator::Unreachable(_),
                    )
                    | None => {}
                }
            }
        }
        if !changed {
            break;
        }
    }

    let mut out = hashbrown::HashMap::with_capacity(fn_count);
    for (fi, f) in module.functions.iter().enumerate() {
        let ro: Vec<bool> = mutated[fi].iter().map(|&m| !m).collect();
        out.insert(f.signature.name.clone(), ro);
    }
    out
}

/// Pure Cranelift runtime builtins guaranteed to NEVER invoke host/user Rust callbacks (`PERF-T4-3`).
/// Conservatively excludes `print`/`println` (console sink) and all effect/handler/unknown callees.
#[cfg(feature = "std")]
#[inline]
fn is_host_callback_free_cranelift_builtin(callee: &str) -> bool {
    matches!(
        callee,
        "math.abs"
            | "abs"
            | "math.min"
            | "min"
            | "math.max"
            | "max"
            | "math.clamp"
            | "clamp"
            | "table.len"
            | "array.len"
            | "array.push"
            | "table.push"
            | "string.len"
            | "string.sub"
            | "string.byte"
            | "string.byte_at"
            | "string.gsub"
            | "string.split"
            | "string.find"
            | "tostring"
            | "map.len"
            | "map.has"
            | "map.get"
            | "map.get_unwrap"
            | "map.get_unwrap_or"
            | "map.set"
            | "map.new"
            | "enum.tag"
            | "enum.payload"
            | "enum.payload_at"
            | "enum.new"
            | "enum.alloc"
            | "enum.new2"
            | "enum.alloc2"
    )
}

/// Infers the exact `MirType` of a call argument operand in `func` for fail-closed
/// signature compatibility checking in `compute_has_external_or_host_calls` (`PERF-T4-3`).
#[cfg(feature = "std")]
fn call_arg_mir_type(func: &MirFunction, op: &Operand) -> MirType {
    use crate::mir::ir::Place;
    match op {
        Operand::Constant(MirConstant::Int(_)) => MirType::Int,
        Operand::Constant(MirConstant::Float(_)) => MirType::Float,
        Operand::Constant(MirConstant::Bool(_)) => MirType::Bool,
        Operand::Constant(MirConstant::String(_)) => MirType::String,
        Operand::Constant(MirConstant::Unit) => MirType::Unit,
        Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => func
            .locals
            .get(loc.0)
            .map(|l| l.ty.clone())
            .unwrap_or(MirType::Unknown),
        Operand::Copy(Place::Index(base, _)) | Operand::Move(Place::Index(base, _)) => {
            if let Place::Local(loc) = base.as_ref()
                && let Some(MirType::Array(elem)) = func.locals.get(loc.0).map(|l| &l.ty)
            {
                elem.as_ref().clone()
            } else {
                MirType::Unknown
            }
        }
        Operand::Copy(Place::Field(..)) | Operand::Move(Place::Field(..)) => MirType::Unknown,
    }
}

/// Computes transitive `has_external_or_host_calls` per function in `module` (`PERF-T4-3`).
///
/// A function is marked `false` ONLY when every `Terminator::Call` in its reachable module
/// call graph targets either another compiled module function (with matching arity, matching
/// known parameter/argument types, and recursively `false`) or a pure callback-free Cranelift
/// builtin, and zero `Terminator::Raise` or effect/handler/external/unknown calls are present.
#[cfg(feature = "std")]
pub fn compute_has_external_or_host_calls(
    module: &crate::mir::ir::MirModule,
) -> hashbrown::HashMap<String, bool> {
    let fn_count = module.functions.len();
    let mut fn_index_by_name: hashbrown::HashMap<&str, Option<usize>> =
        hashbrown::HashMap::with_capacity(fn_count);
    for (fi, f) in module.functions.iter().enumerate() {
        fn_index_by_name
            .entry(f.signature.name.as_str())
            .and_modify(|slot| *slot = None)
            .or_insert(Some(fi));
    }

    let mut has_ext = vec![false; fn_count];
    let mut edges: Vec<Vec<usize>> = vec![Vec::new(); fn_count];

    for (fi, f) in module.functions.iter().enumerate() {
        for block in &f.blocks {
            match &block.terminator {
                Some(Terminator::Call { callee, args, .. }) => {
                    if is_host_callback_free_cranelift_builtin(callee) {
                        continue;
                    }
                    if is_effect_or_handler_callee(callee) {
                        has_ext[fi] = true;
                        continue;
                    }
                    if let Some(Some(gi)) = fn_index_by_name.get(callee.as_str()).copied() {
                        let callee_fn = &module.functions[gi];
                        let arity_and_types_match = callee_fn.params.len() == args.len()
                            && callee_fn.params.iter().zip(args.iter()).all(|(&param_loc, arg)| {
                                let Some(param_decl) = callee_fn.locals.get(param_loc.0) else {
                                    return false;
                                };
                                let arg_ty = call_arg_mir_type(f, arg);
                                param_decl.ty != MirType::Unknown && arg_ty == param_decl.ty
                            });
                        if arity_and_types_match {
                            edges[fi].push(gi);
                        } else {
                            has_ext[fi] = true;
                        }
                    } else {
                        has_ext[fi] = true;
                    }
                }
                Some(Terminator::Raise(..)) => {
                    has_ext[fi] = true;
                }
                _ => {}
            }
        }
    }

    loop {
        let mut changed = false;
        for fi in 0..fn_count {
            if has_ext[fi] {
                continue;
            }
            if edges[fi].iter().any(|&gi| has_ext[gi]) {
                has_ext[fi] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut out = hashbrown::HashMap::with_capacity(fn_count);
    for (fi, f) in module.functions.iter().enumerate() {
        out.insert(f.signature.name.clone(), has_ext[fi]);
    }
    out
}

#[cfg(feature = "std")]
#[inline]
fn flat_array_elem_tag(elem_ty: &MirType) -> Option<FlatArrayElemTag> {
    match elem_ty {
        MirType::Int => Some(FlatArrayElemTag::Int),
        MirType::Float => Some(FlatArrayElemTag::Float),
        MirType::Bool => Some(FlatArrayElemTag::Bool),
        _ => None,
    }
}

/// Returns a cached read-only `FlatArrayBuffer` on hit, or builds and publishes one via the
/// exact 4-step atomic OOM-rollback protocol (`PERF-T4-3`):
///   1. Checked `req_bytes = (len + 2) * 8` (overflow -> OOM).
///   2. Reserve `req_bytes` against `mir_ctx.max_memory_limit` BEFORE allocating or touching
///      `rc.flat_cache` (breach -> leave `flat_cache` unmodified, set `MIR_STATUS_ERR_OOM`, return `Err`).
///   3. Allocate and populate a local `Vec<i64>` while validating every element matches `tag`
///      (mismatch -> refund `req_bytes`, leave `flat_cache` untouched, return `Ok(None)` for fallback).
///   4. Publish `Rc<FlatArrayBuffer>` into `rc.flat_cache` ONLY after full population succeeds.
///
/// Return values:
/// - `Ok(Some((buf, true)))`: cache hit (zero bytes allocated/charged).
/// - `Ok(Some((buf, false)))`: cache miss built and published atomically.
/// - `Ok(None)`: homogeneous element type mismatch (reserved bytes refunded, cache untouched).
/// - `Err(err)`: OOM or checked arithmetic overflow (status set to `MIR_STATUS_ERR_OOM`, cache untouched).
#[cfg(feature = "std")]
pub fn build_or_get_cached_flat_array(
    mir_ctx: &crate::mir::runtime_helpers::MirContext,
    rc: &Rc<LustArray>,
    elements: &[Value],
    tag: FlatArrayElemTag,
) -> Result<Option<(Rc<FlatArrayBuffer>, bool)>> {
    if let Some(buf) = rc.cached_flat(tag) {
        return Ok(Some((buf, true)));
    }

    // Fast-path for unboxed typed primitive arrays (NG-W10-1 + NG-W10-5):
    // Copy directly via single `ptr::copy_nonoverlapping` without materializing `Vec<Value>`.
    let unboxed_res: Option<Result<Option<Rc<FlatArrayBuffer>>>> = rc.with_unboxed_data(|unb| {
        let len = match (unb, tag) {
            (UnboxedArrayData::None, _) => return None,
            (UnboxedArrayData::Int(v), FlatArrayElemTag::Int) => v.len(),
            (UnboxedArrayData::Float(v), FlatArrayElemTag::Float) => v.len(),
            (UnboxedArrayData::Bool(v), FlatArrayElemTag::Bool) => v.len(),
            (UnboxedArrayData::Int(v), _) if v.is_empty() => 0,
            (UnboxedArrayData::Float(v), _) if v.is_empty() => 0,
            (UnboxedArrayData::Bool(v), _) if v.is_empty() => 0,
            _ => return Some(Ok(None)),
        };

        let payload_words = if tag == FlatArrayElemTag::Bool {
            let Some(w) = len.checked_add(7).map(|n| n / 8) else {
                mir_ctx.set_status(MIR_STATUS_ERR_OOM);
                return Some(Err(LustError::RuntimeError {
                    message: "Out of memory: flat bool array cache word length overflow".into(),
                }));
            };
            w
        } else {
            len
        };
        let Some(word_len) = payload_words.checked_add(2) else {
            mir_ctx.set_status(MIR_STATUS_ERR_OOM);
            return Some(Err(LustError::RuntimeError {
                message: "Out of memory: flat array cache word length overflow".into(),
            }));
        };
        let Some(req_bytes_u) = word_len.checked_mul(core::mem::size_of::<i64>()) else {
            mir_ctx.set_status(MIR_STATUS_ERR_OOM);
            return Some(Err(LustError::RuntimeError {
                message: "Out of memory: flat array cache byte size overflow".into(),
            }));
        };
        let Ok(req_bytes) = i64::try_from(req_bytes_u) else {
            mir_ctx.set_status(MIR_STATUS_ERR_OOM);
            return Some(Err(LustError::RuntimeError {
                message: "Out of memory: flat array cache byte size exceeds i64::MAX".into(),
            }));
        };

        let limit = mir_ctx
            .max_memory_limit
            .load(core::sync::atomic::Ordering::SeqCst);
        let prev = mir_ctx
            .memory_used
            .fetch_add(req_bytes, core::sync::atomic::Ordering::SeqCst);
        let mem_rollback = core::mem::DropGuard::new(req_bytes, |bytes| {
            mir_ctx
                .memory_used
                .fetch_sub(bytes, core::sync::atomic::Ordering::SeqCst);
        });
        if prev.saturating_add(req_bytes) > limit {
            mir_ctx.set_status(MIR_STATUS_ERR_OOM);
            return Some(Err(LustError::RuntimeError {
                message: "Out of memory allocating cached flat array buffer for Cranelift JIT"
                    .into(),
            }));
        }

        let mut words = Vec::<i64>::new();
        if words.try_reserve_exact(word_len).is_err() {
            mir_ctx.set_status(MIR_STATUS_ERR_OOM);
            return Some(Err(LustError::RuntimeError {
                message: "Out of memory reserving host buffer for cached flat array".into(),
            }));
        }
        words.resize(word_len, 0);
        words[0] = len as i64;
        words[1] = len as i64;
        if len > 0 {
            unsafe {
                let dst = words.as_mut_ptr().add(2);
                match (unb, tag) {
                    (UnboxedArrayData::Int(v), FlatArrayElemTag::Int) => {
                        core::ptr::copy_nonoverlapping(v.as_ptr(), dst, len);
                    }
                    (UnboxedArrayData::Float(v), FlatArrayElemTag::Float) => {
                        core::ptr::copy_nonoverlapping(v.as_ptr() as *const i64, dst, len);
                    }
                    (UnboxedArrayData::Bool(v), FlatArrayElemTag::Bool) => {
                        core::ptr::copy_nonoverlapping(v.as_ptr(), dst as *mut u8, len);
                    }
                    _ => unreachable!("unboxed array tag pre-verified"),
                }
            }
        }
        let buf = Rc::new(FlatArrayBuffer::new(words));
        rc.set_cached_flat(tag, Rc::clone(&buf));
        core::mem::DropGuard::dismiss(mem_rollback);
        Some(Ok(Some(buf)))
    });
    if let Some(res) = unboxed_res {
        return Ok(res?.map(|buf| (buf, false)));
    }

    // Step 1: Checked byte calculation (`(payload_words + 2) * 8`; 1-byte stride for Bool)
    let len = elements.len();
    let payload_words = if tag == FlatArrayElemTag::Bool {
        let Some(w) = len.checked_add(7).map(|n| n / 8) else {
            mir_ctx.set_status(MIR_STATUS_ERR_OOM);
            return Err(LustError::RuntimeError {
                message: "Out of memory: flat bool array cache word length overflow".into(),
            });
        };
        w
    } else {
        len
    };
    let Some(word_len) = payload_words.checked_add(2) else {
        mir_ctx.set_status(MIR_STATUS_ERR_OOM);
        return Err(LustError::RuntimeError {
            message: "Out of memory: flat array cache word length overflow".into(),
        });
    };
    let Some(req_bytes_u) = word_len.checked_mul(core::mem::size_of::<i64>()) else {
        mir_ctx.set_status(MIR_STATUS_ERR_OOM);
        return Err(LustError::RuntimeError {
            message: "Out of memory: flat array cache byte size overflow".into(),
        });
    };
    let Ok(req_bytes) = i64::try_from(req_bytes_u) else {
        mir_ctx.set_status(MIR_STATUS_ERR_OOM);
        return Err(LustError::RuntimeError {
            message: "Out of memory: flat array cache byte size exceeds i64::MAX".into(),
        });
    };

    // Step 2: Reserve `req_bytes` against `mir_ctx.max_memory_limit` BEFORE allocation or cache mutation
    let limit = mir_ctx
        .max_memory_limit
        .load(core::sync::atomic::Ordering::SeqCst);
    let prev = mir_ctx
        .memory_used
        .fetch_add(req_bytes, core::sync::atomic::Ordering::SeqCst);
    let mem_rollback = core::mem::DropGuard::new(req_bytes, |bytes| {
        mir_ctx
            .memory_used
            .fetch_sub(bytes, core::sync::atomic::Ordering::SeqCst);
    });
    if prev.saturating_add(req_bytes) > limit {
        mir_ctx.set_status(MIR_STATUS_ERR_OOM);
        return Err(LustError::RuntimeError {
            message: "Out of memory allocating cached flat array buffer for Cranelift JIT".into(),
        });
    }

    // Step 3: Allocate and populate local flat buffer, validating every element
    let mut words = Vec::<i64>::new();
    if words.try_reserve_exact(word_len).is_err() {
        mir_ctx.set_status(MIR_STATUS_ERR_OOM);
        return Err(LustError::RuntimeError {
            message: "Out of memory reserving host buffer for cached flat array".into(),
        });
    }
    if tag == FlatArrayElemTag::Bool {
        words.resize(word_len, 0);
        words[0] = len as i64;
        words[1] = len as i64;
        let dst_bytes = unsafe { words.as_mut_ptr().add(2) as *mut u8 };
        for (idx, elem) in elements.iter().enumerate() {
            match elem {
                Value::Bool(b) => unsafe {
                    *dst_bytes.add(idx) = u8::from(*b);
                },
                _ => {
                    // Element type mismatch: `mem_rollback` refunds reserved bytes, leave flat_cache untouched.
                    return Ok(None);
                }
            }
        }
    } else {
        words.push(len as i64);
        words.push(len as i64);
        for elem in elements {
            let raw = match (elem, tag) {
                (Value::Int(n), FlatArrayElemTag::Int) => *n,
                (Value::Float(f), FlatArrayElemTag::Float) => f.to_bits() as i64,
                _ => {
                    // Element type mismatch: `mem_rollback` refunds reserved bytes, leave flat_cache untouched.
                    return Ok(None);
                }
            };
            words.push(raw);
        }
    }

    // Step 4: Publish into `LustArray.flat_cache` ONLY after full population succeeds
    let buf = Rc::new(FlatArrayBuffer::new(words));
    rc.set_cached_flat(tag, Rc::clone(&buf));
    core::mem::DropGuard::dismiss(mem_rollback);
    Ok(Some((buf, false)))
}

impl VM {
    #[cfg(feature = "std")]
    pub(crate) fn is_cranelift_supported_primitive_elem_type(ty: &MirType) -> bool {
        matches!(
            ty,
            MirType::Int | MirType::Bool | MirType::Float | MirType::String
        )
    }

    #[cfg(feature = "std")]
    pub(crate) fn is_cranelift_supported_array_elem_type(ty: &MirType) -> bool {
        Self::is_cranelift_supported_primitive_elem_type(ty)
            || matches!(ty, MirType::Struct(name) if !name.is_empty())
    }

    #[cfg(feature = "std")]
    pub(crate) fn type_contains_unknown(ty: &MirType) -> bool {
        match ty {
            MirType::Unknown => true,
            MirType::Array(inner) => Self::type_contains_unknown(inner),
            MirType::Option(inner) => Self::type_contains_unknown(inner),
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
            MirType::Option(inner) => {
                matches!(
                    &**inner,
                    MirType::Int
                        | MirType::Bool
                        | MirType::Float
                        | MirType::String
                        | MirType::Unknown
                ) || matches!(&**inner, MirType::Struct(name) if !name.is_empty())
            }
            MirType::Tuple(elems) => elems.iter().all(|e| {
                Self::is_cranelift_supported_primitive_elem_type(e)
                    || matches!(e, MirType::Unknown)
            }),
            MirType::Struct(_) => true,
            MirType::Map(k, v) => Self::is_cranelift_supported_type(k) && Self::is_cranelift_supported_type(v),
            MirType::Enum(name) => !name.is_empty(),
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
            Rvalue::Closure { .. } => false,
        }
    }

    #[cfg(feature = "std")]
    fn resolve_struct_field_mir_type_in_map(
        owner_struct_name: &str,
        field_ty: &crate::ast::Type,
        struct_meta: &hashbrown::HashMap<String, crate::vm::RuntimeStructInfo>,
    ) -> MirType {
        let mut ty = crate::mir::lowering::mir_type_from_ast(field_ty);
        if let MirType::Option(inner) = &mut ty
            && let MirType::Struct(inner_name) = inner.as_mut()
        {
            *inner_name = resolve_option_struct_target_name(
                owner_struct_name,
                inner_name,
                Some(struct_meta),
            )
            .into_owned();
        }
        ty
    }

    #[cfg(feature = "std")]
    pub(crate) fn is_cranelift_supported_struct(&self, name: &str) -> bool {
        let mut visiting = hashbrown::HashSet::new();
        self.is_cranelift_supported_struct_inner(name, &mut visiting)
    }

    #[cfg(feature = "std")]
    fn is_cranelift_supported_struct_inner(
        &self,
        name: &str,
        visiting: &mut hashbrown::HashSet<String>,
    ) -> bool {
        if name.is_empty() || get_registered_struct_layout_by_name(name).is_some() {
            return false;
        }
        let Some(info) = self.struct_metadata.get(name) else {
            return false;
        };
        let canonical = info.layout.name().to_string();
        if !visiting.insert(canonical.clone()) {
            return true;
        }
        let ok = (0..info.layout.field_names().len()).all(|i| match &info.layout.field_type(i).kind {
            TypeKind::Int
            | TypeKind::Bool
            | TypeKind::Float
            | TypeKind::String
            | TypeKind::Map(_, _)
            | TypeKind::Result(_, _) => true,
            TypeKind::Option(inner) => match &inner.kind {
                TypeKind::Named(sub) => {
                    let resolved = resolve_option_struct_target_name(
                        name,
                        sub,
                        Some(self.struct_metadata.as_map()),
                    );
                    if self.struct_metadata.contains_key(resolved.as_ref()) {
                        self.is_cranelift_supported_struct_inner(resolved.as_ref(), visiting)
                    } else {
                        false
                    }
                }
                _ => true,
            },
            _ => false,
        });
        visiting.remove(&canonical);
        ok
    }

    #[cfg(feature = "std")]
    fn is_cranelift_supported_type_in_vm(&self, ty: &MirType) -> bool {
        match ty {
            MirType::Struct(name) => self.is_cranelift_supported_struct(name),
            MirType::Array(inner) => match &**inner {
                MirType::Struct(name) => self.is_cranelift_supported_struct(name),
                other => {
                    Self::is_cranelift_supported_primitive_elem_type(other)
                        || matches!(other, MirType::Unknown)
                }
            },
            MirType::Option(inner) => match &**inner {
                MirType::Int
                | MirType::Bool
                | MirType::Float
                | MirType::String
                | MirType::Unknown => true,
                MirType::Struct(name) => self.is_cranelift_supported_struct(name),
                _ => false,
            },
            MirType::Map(k, v) => {
                self.is_cranelift_supported_type_in_vm(k) && self.is_cranelift_supported_type_in_vm(v)
            }
            MirType::Enum(name) => !name.is_empty(),
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
                    .and_then(|m| {
                        let info = m.get(&name)?;
                        if *idx < info.layout.field_names().len() {
                            Some(Self::resolve_struct_field_mir_type_in_map(
                                &name,
                                info.layout.field_type(*idx),
                                m,
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
    fn check_refined_type(ty: &MirType, check_struct: &dyn Fn(&str) -> bool) -> bool {
        if Self::type_contains_unknown(ty) || !Self::is_cranelift_supported_type(ty) {
            return false;
        }
        match ty {
            MirType::Struct(name) => check_struct(name),
            MirType::Array(inner) => match &**inner {
                MirType::Struct(name) => check_struct(name),
                _ => true,
            },
            MirType::Option(inner) => match &**inner {
                MirType::Int | MirType::Bool | MirType::Float | MirType::String => true,
                MirType::Struct(name) => check_struct(name),
                _ => false,
            },
            MirType::Map(k, v) => {
                Self::check_refined_type(k, check_struct)
                    && Self::check_refined_type(v, check_struct)
            }
            MirType::Enum(name) => !name.is_empty(),
            _ => true,
        }
    }

    #[cfg(feature = "std")]
    fn check_refined_place(
        place: &crate::mir::ir::Place,
        locals: &[crate::mir::ir::LocalDecl],
        struct_meta: Option<&hashbrown::HashMap<String, crate::vm::RuntimeStructInfo>>,
        check_struct: &dyn Fn(&str) -> bool,
    ) -> bool {
        use crate::mir::ir::Place;
        match place {
            Place::Local(loc) => locals
                .get(loc.0)
                .map(|l| Self::check_refined_type(&l.ty, check_struct))
                .unwrap_or(false),
            Place::Index(base, idx_loc) => {
                if !Self::check_refined_place(base, locals, struct_meta, check_struct) {
                    return false;
                }
                if !locals
                    .get(idx_loc.0)
                    .map(|l| l.ty == MirType::Int)
                    .unwrap_or(false)
                {
                    return false;
                }
                match Self::mir_place_type_in_vm(base, locals, struct_meta) {
                    Some(MirType::Array(elem)) => match &*elem {
                        MirType::Struct(name) => check_struct(name),
                        other => Self::is_cranelift_supported_primitive_elem_type(other),
                    },
                    Some(MirType::Tuple(elems)) => {
                        !elems.is_empty()
                            && elems
                                .iter()
                                .all(Self::is_cranelift_supported_primitive_elem_type)
                    }
                    _ => false,
                }
            }
            Place::Field(base, idx) => {
                if !Self::check_refined_place(base, locals, struct_meta, check_struct) {
                    return false;
                }
                match Self::mir_place_type_in_vm(base, locals, struct_meta) {
                    Some(MirType::Struct(name)) => {
                        if !check_struct(&name) {
                            return false;
                        }
                        if let Some(meta) = struct_meta {
                            let Some(info) = meta.get(&name) else {
                                return false;
                            };
                            if *idx >= info.layout.field_names().len() {
                                return false;
                            }
                            let field_ty = Self::resolve_struct_field_mir_type_in_map(
                                &name,
                                info.layout.field_type(*idx),
                                meta,
                            );
                            Self::check_refined_type(&field_ty, check_struct)
                                && !matches!(
                                    field_ty,
                                    MirType::Array(_)
                                        | MirType::Tuple(_)
                                        | MirType::Struct(_)
                                        | MirType::Function { .. }
                                        | MirType::Unknown
                                )
                        } else {
                            true
                        }
                    }
                    Some(MirType::Tuple(elems)) => {
                        let Some(elem_ty) = elems.get(*idx) else {
                            return false;
                        };
                        Self::check_refined_type(elem_ty, check_struct)
                            && !matches!(
                                elem_ty,
                                MirType::Array(_)
                                    | MirType::Tuple(_)
                                    | MirType::Struct(_)
                                    | MirType::Function { .. }
                                    | MirType::Unknown
                            )
                    }
                    _ => false,
                }
            }
        }
    }

    #[cfg(feature = "std")]
    fn check_refined_operand(
        op: &Operand,
        locals: &[crate::mir::ir::LocalDecl],
        struct_meta: Option<&hashbrown::HashMap<String, crate::vm::RuntimeStructInfo>>,
        check_struct: &dyn Fn(&str) -> bool,
    ) -> bool {
        match op {
            Operand::Copy(p) | Operand::Move(p) => {
                Self::check_refined_place(p, locals, struct_meta, check_struct)
            }
            Operand::Constant(_) => Self::is_cranelift_eligible_operand(op),
        }
    }

    #[cfg(feature = "std")]
    fn check_refined_cranelift_eligibility(
        refined: &MirFunction,
        struct_meta: Option<&hashbrown::HashMap<String, crate::vm::RuntimeStructInfo>>,
        check_struct: &dyn Fn(&str) -> bool,
    ) -> bool {
        let check_type = |ty: &MirType| -> bool { Self::check_refined_type(ty, check_struct) };
        if !check_type(&refined.signature.return_type)
            || !refined
                .params
                .iter()
                .all(|p| check_type(&refined.locals[p.0].ty))
            || !refined.locals.iter().all(|l| check_type(&l.ty))
        {
            return false;
        }
        for block in &refined.blocks {
            for stmt in &block.statements {
                match stmt {
                    Statement::Assign(place, rval, _) => {
                        if !Self::check_refined_place(
                            place,
                            &refined.locals,
                            struct_meta,
                            check_struct,
                        ) {
                            return false;
                        }
                        match rval {
                            Rvalue::Use(op) => {
                                if !Self::check_refined_operand(
                                    op,
                                    &refined.locals,
                                    struct_meta,
                                    check_struct,
                                ) {
                                    return false;
                                }
                            }
                            Rvalue::BinaryOp(op_kind, lhs, rhs) => {
                                if !Self::check_refined_operand(
                                    lhs,
                                    &refined.locals,
                                    struct_meta,
                                    check_struct,
                                ) || !Self::check_refined_operand(
                                    rhs,
                                    &refined.locals,
                                    struct_meta,
                                    check_struct,
                                ) {
                                    return false;
                                }
                                if matches!(op_kind, crate::mir::ir::MirBinaryOp::Concat) {
                                    let l_ty = Self::mir_operand_type_in_vm(
                                        lhs,
                                        &refined.locals,
                                        struct_meta,
                                    );
                                    let r_ty = Self::mir_operand_type_in_vm(
                                        rhs,
                                        &refined.locals,
                                        struct_meta,
                                    );
                                    let ok_ty = |t: &Option<MirType>| {
                                        matches!(
                                            t,
                                            Some(
                                                MirType::String
                                                    | MirType::Int
                                                    | MirType::Float
                                                    | MirType::Bool
                                            )
                                        )
                                    };
                                    if !ok_ty(&l_ty) || !ok_ty(&r_ty) {
                                        return false;
                                    }
                                }
                            }
                            Rvalue::UnaryOp(op_kind, op) => {
                                if !Self::check_refined_operand(
                                    op,
                                    &refined.locals,
                                    struct_meta,
                                    check_struct,
                                ) {
                                    return false;
                                }
                                if matches!(op_kind, MirUnaryOp::Len) {
                                    let op_ty = Self::mir_operand_type_in_vm(
                                        op,
                                        &refined.locals,
                                        struct_meta,
                                    );
                                    if !matches!(op_ty, Some(MirType::Array(_) | MirType::String)) {
                                        return false;
                                    }
                                }
                            }
                            Rvalue::Cast(op, target_ty) => {
                                if !Self::check_refined_operand(
                                    op,
                                    &refined.locals,
                                    struct_meta,
                                    check_struct,
                                ) || !check_type(target_ty)
                                {
                                    return false;
                                }
                            }
                            Rvalue::Aggregate(agg_ty, elems) => {
                                if Self::type_contains_unknown(agg_ty)
                                    || !check_type(agg_ty)
                                    || !elems.iter().all(|e| {
                                        Self::check_refined_operand(
                                            e,
                                            &refined.locals,
                                            struct_meta,
                                            check_struct,
                                        )
                                    })
                                {
                                    return false;
                                }
                                match agg_ty {
                                    MirType::Array(elem_ty) => {
                                        let elem_ok = match &**elem_ty {
                                            MirType::Struct(name) => check_struct(name),
                                            other => {
                                                Self::is_cranelift_supported_primitive_elem_type(
                                                    other,
                                                )
                                            }
                                        };
                                        if !elem_ok {
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
                                        for (elem, expected_ty) in
                                            elems.iter().zip(elem_types.iter())
                                        {
                                            if !Self::is_cranelift_supported_primitive_elem_type(
                                                expected_ty,
                                            ) || Self::mir_operand_type_in_vm(
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
                            Rvalue::Closure { .. } => return false,
                        }
                    }
                    Statement::StorageLive(_, _)
                    | Statement::StorageDead(_, _)
                    | Statement::ResourceCheck(_, _, _) => {}
                }
            }
            if let Some(term) = &block.terminator {
                match term {
                    Terminator::Goto(_, _) | Terminator::Unreachable(_) => {}
                    Terminator::Cond { cond, .. } => {
                        if !Self::check_refined_operand(
                            cond,
                            &refined.locals,
                            struct_meta,
                            check_struct,
                        ) {
                            return false;
                        }
                    }
                    Terminator::Switch { discr, .. } => {
                        if !Self::check_refined_operand(
                            discr,
                            &refined.locals,
                            struct_meta,
                            check_struct,
                        ) {
                            return false;
                        }
                    }
                    Terminator::Call {
                        callee,
                        args,
                        destination,
                        ..
                    } => {
                        if !Self::check_refined_place(
                            destination,
                            &refined.locals,
                            struct_meta,
                            check_struct,
                        ) || !args.iter().all(|a| {
                            Self::check_refined_operand(
                                a,
                                &refined.locals,
                                struct_meta,
                                check_struct,
                            )
                        }) {
                            return false;
                        }
                        if callee == "print" || callee == "println" {
                            let dest_ok = matches!(
                                destination,
                                crate::mir::ir::Place::Local(loc)
                                    if matches!(
                                        refined.locals.get(loc.0).map(|l| &l.ty),
                                        Some(MirType::Unit | MirType::Int)
                                    )
                            );
                            if !dest_ok {
                                return false;
                            }
                            for arg in args {
                                let arg_ty = Self::mir_operand_type_in_vm(
                                    arg,
                                    &refined.locals,
                                    struct_meta,
                                );
                                if !matches!(
                                    arg_ty,
                                    Some(
                                        MirType::Int
                                            | MirType::Float
                                            | MirType::Bool
                                            | MirType::String
                                    )
                                ) {
                                    return false;
                                }
                            }
                        } else if callee == "tostring" {
                            if args.len() != 1 {
                                return false;
                            }
                            let arg_ty = Self::mir_operand_type_in_vm(
                                &args[0],
                                &refined.locals,
                                struct_meta,
                            );
                            if !matches!(
                                arg_ty,
                                Some(
                                    MirType::Int
                                        | MirType::Float
                                        | MirType::Bool
                                        | MirType::String
                                )
                            ) {
                                return false;
                            }
                            let dest_ok = matches!(
                                destination,
                                crate::mir::ir::Place::Local(loc)
                                    if refined.locals.get(loc.0).map(|l| &l.ty) == Some(&MirType::String)
                            );
                            if !dest_ok {
                                return false;
                            }
                        } else if matches!(callee.as_str(), "table.len" | "array.len") {
                            if args.len() != 1 {
                                return false;
                            }
                            let arg_ty = Self::mir_operand_type_in_vm(
                                &args[0],
                                &refined.locals,
                                struct_meta,
                            );
                            if !matches!(arg_ty, Some(MirType::Array(_))) {
                                return false;
                            }
                        } else if callee == "string.len" {
                            if args.len() != 1 {
                                return false;
                            }
                            let arg_ty = Self::mir_operand_type_in_vm(
                                &args[0],
                                &refined.locals,
                                struct_meta,
                            );
                            if arg_ty != Some(MirType::String) {
                                return false;
                            }
                        } else if callee == "string.sub" {
                            if args.len() != 2 && args.len() != 3 {
                                return false;
                            }
                            let s_ty = Self::mir_operand_type_in_vm(
                                &args[0],
                                &refined.locals,
                                struct_meta,
                            );
                            let i_ty = Self::mir_operand_type_in_vm(
                                &args[1],
                                &refined.locals,
                                struct_meta,
                            );
                            let j_ok = args.get(2).is_none_or(|a| {
                                Self::mir_operand_type_in_vm(a, &refined.locals, struct_meta)
                                    == Some(MirType::Int)
                            });
                            let dest_ty = Self::mir_place_type_in_vm(
                                destination,
                                &refined.locals,
                                struct_meta,
                            );
                            if s_ty != Some(MirType::String)
                                || i_ty != Some(MirType::Int)
                                || !j_ok
                                || dest_ty != Some(MirType::String)
                            {
                                return false;
                            }
                        } else if callee == "string.byte" {
                            if !(1..=3).contains(&args.len()) {
                                return false;
                            }
                            let s_ty = Self::mir_operand_type_in_vm(
                                &args[0],
                                &refined.locals,
                                struct_meta,
                            );
                            let bounds_ok = args[1..].iter().all(|a| {
                                Self::mir_operand_type_in_vm(a, &refined.locals, struct_meta)
                                    == Some(MirType::Int)
                            });
                            let dest_ty = Self::mir_place_type_in_vm(
                                destination,
                                &refined.locals,
                                struct_meta,
                            );
                            if s_ty != Some(MirType::String)
                                || !bounds_ok
                                || !matches!(dest_ty, Some(MirType::Int | MirType::Array(_)))
                            {
                                return false;
                            }
                        } else if callee == "string.gsub" {
                            if args.len() != 3 {
                                return false;
                            }
                            let args_ok = args.iter().all(|a| {
                                Self::mir_operand_type_in_vm(a, &refined.locals, struct_meta)
                                    == Some(MirType::String)
                            });
                            let dest_ty = Self::mir_place_type_in_vm(
                                destination,
                                &refined.locals,
                                struct_meta,
                            );
                            let dest_ok = match dest_ty {
                                Some(MirType::String) => true,
                                Some(MirType::Array(inner)) => *inner == MirType::String,
                                _ => false,
                            };
                            if !args_ok || !dest_ok {
                                return false;
                            }
                        } else if callee == "string.split" {
                            if args.len() != 2 {
                                return false;
                            }
                            let args_ok = args.iter().all(|a| {
                                Self::mir_operand_type_in_vm(a, &refined.locals, struct_meta)
                                    == Some(MirType::String)
                            });
                            let dest_ty = Self::mir_place_type_in_vm(
                                destination,
                                &refined.locals,
                                struct_meta,
                            );
                            if !args_ok
                                || !matches!(dest_ty, Some(MirType::Array(inner)) if *inner == MirType::String)
                            {
                                return false;
                            }
                        } else if callee == "string.find" {
                            if !(2..=4).contains(&args.len()) {
                                return false;
                            }
                            let s_ok = Self::mir_operand_type_in_vm(
                                &args[0],
                                &refined.locals,
                                struct_meta,
                            ) == Some(MirType::String)
                                && Self::mir_operand_type_in_vm(
                                    &args[1],
                                    &refined.locals,
                                    struct_meta,
                                ) == Some(MirType::String);
                            let start_ok = args.get(2).is_none_or(|a| {
                                Self::mir_operand_type_in_vm(a, &refined.locals, struct_meta)
                                    == Some(MirType::Int)
                            });
                            let plain_ok = args.get(3).is_none_or(|a| {
                                matches!(
                                    Self::mir_operand_type_in_vm(a, &refined.locals, struct_meta),
                                    Some(MirType::Bool | MirType::Int)
                                )
                            });
                            let dest_ty = Self::mir_place_type_in_vm(
                                destination,
                                &refined.locals,
                                struct_meta,
                            );
                            if !s_ok
                                || !start_ok
                                || !plain_ok
                                || !matches!(dest_ty, Some(MirType::Int | MirType::Array(_)))
                            {
                                return false;
                            }
                        } else if callee == "map.get_unwrap" {
                            if args.len() != 2 {
                                return false;
                            }
                            let Some(MirType::Map(k_ty, v_ty)) = Self::mir_operand_type_in_vm(
                                &args[0],
                                &refined.locals,
                                struct_meta,
                            ) else {
                                return false;
                            };
                            let arg_k = Self::mir_operand_type_in_vm(
                                &args[1],
                                &refined.locals,
                                struct_meta,
                            );
                            let dest_ty = Self::mir_place_type_in_vm(
                                destination,
                                &refined.locals,
                                struct_meta,
                            );
                            if arg_k.as_ref() != Some(&*k_ty) || dest_ty.as_ref() != Some(&*v_ty) {
                                return false;
                            }
                        } else if callee == "map.get_unwrap_or" {
                            if args.len() != 3 {
                                return false;
                            }
                            let Some(MirType::Map(k_ty, v_ty)) = Self::mir_operand_type_in_vm(
                                &args[0],
                                &refined.locals,
                                struct_meta,
                            ) else {
                                return false;
                            };
                            let arg_k = Self::mir_operand_type_in_vm(
                                &args[1],
                                &refined.locals,
                                struct_meta,
                            );
                            let arg_def = Self::mir_operand_type_in_vm(
                                &args[2],
                                &refined.locals,
                                struct_meta,
                            );
                            let dest_ty = Self::mir_place_type_in_vm(
                                destination,
                                &refined.locals,
                                struct_meta,
                            );
                            if arg_k.as_ref() != Some(&*k_ty)
                                || arg_def.as_ref() != Some(&*v_ty)
                                || dest_ty.as_ref() != Some(&*v_ty)
                            {
                                return false;
                            }
                        }
                    }
                    Terminator::Return(opt_op, _) => {
                        if let Some(op) = opt_op
                            && !Self::check_refined_operand(
                                op,
                                &refined.locals,
                                struct_meta,
                                check_struct,
                            )
                        {
                            return false;
                        }
                    }
                    Terminator::Raise(_, _) => {
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
            Some(self.struct_metadata.as_map()),
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
                | "array.push"
                | "table.push"
                | "string.len"
                | "string.sub"
                | "string.byte"
                | "string.byte_at"
                | "string.gsub"
                | "string.split"
                | "string.find"
                | "tostring"
                | "map.len"
                | "map.has"
                | "map.get"
                | "map.get_unwrap"
                | "map.get_unwrap_or"
                | "map.set"
                | "map.new"
                | "enum.tag"
                | "enum.payload"
                | "enum.payload_at"
                | "enum.new"
                | "enum.alloc"
                | "enum.new2"
                | "enum.alloc2"
        )
    }

    #[cfg(feature = "std")]
    pub fn is_supported_deopt_recipe_type(&self, ty: &MirType) -> bool {
        if Self::type_contains_unknown(ty) {
            return false;
        }
        match ty {
            MirType::Int | MirType::Bool | MirType::Float | MirType::Unit | MirType::String => true,
            MirType::Array(inner) => match &**inner {
                MirType::Int | MirType::Bool | MirType::Float | MirType::String => true,
                MirType::Struct(name) => self.is_cranelift_supported_struct(name),
                _ => false,
            },
            MirType::Struct(name) => self.is_cranelift_supported_struct(name),
            MirType::Option(inner) => match &**inner {
                MirType::Int | MirType::Bool | MirType::Float | MirType::String => true,
                MirType::Struct(name) => self.is_cranelift_supported_struct(name),
                _ => false,
            },
            MirType::Map(k, v) => {
                self.is_cranelift_supported_type_in_vm(k)
                    && self.is_cranelift_supported_type_in_vm(v)
            }
            MirType::Enum(name) => !name.is_empty(),
            _ => false,
        }
    }

    /// Rewrites MIR-VM continuation bytecode operations on `MirType::Struct` locals from
    /// `NewArray` / `GetIndex` / `SetIndex` to `NewStruct` / `GetField` / `SetField` 1-for-1
    /// in place (preserving exact instruction PCs, `block_entry_pcs`, `terminator_pcs`, and
    /// jump offsets) so post-deopt interpreter resumption operates directly on `Value::Struct`.
    #[cfg(feature = "std")]
    fn rewrite_continuation_struct_ops(
        &self,
        candidate: &MirFunction,
        continuation_bc: &mut crate::bytecode::Function,
    ) -> core::result::Result<(), String> {
        use crate::bytecode::Instruction;

        let refined = self.refine_mir_function_in_vm(candidate);
        if !refined
            .locals
            .iter()
            .any(|l| matches!(l.ty, MirType::Struct(_)))
        {
            return Ok(());
        }

        let func_name = &candidate.signature.name;
        let len = continuation_bc.chunk.instructions.len();
        for pc in 0..len {
            let insn = continuation_bc.chunk.instructions[pc];
            match insn {
                Instruction::NewArray(dst, first_field, count) => {
                    let Some(MirType::Struct(struct_name)) =
                        refined.locals.get(dst as usize).map(|l| &l.ty)
                    else {
                        continue;
                    };
                    let info = self.struct_metadata.get(struct_name).ok_or_else(|| {
                        format!(
                            "Missing struct layout metadata for '{struct_name}' rewriting continuation of '{func_name}'"
                        )
                    })?;
                    let field_names = info.layout.field_names();
                    if field_names.len() != count as usize {
                        // Leave synthetic/malformed struct constructors untouched so runtime
                        // validate_and_decode_struct enforces fail-closed field-count rejection.
                        continue;
                    }
                    let name_val = Value::string(struct_name.clone());
                    let name_exists = continuation_bc
                        .chunk
                        .constants
                        .iter()
                        .any(|v| v == &name_val);
                    if !name_exists && continuation_bc.chunk.constants.len() >= u16::MAX as usize {
                        return Err(format!(
                            "Constant pool overflow rewriting struct '{struct_name}' constructor in '{func_name}'"
                        ));
                    }
                    let name_idx = continuation_bc.chunk.add_constant(name_val);
                    let first_field_name_idx = if count == 0 {
                        0u16
                    } else {
                        let start = continuation_bc.chunk.constants.len();
                        if start
                            .checked_add(field_names.len())
                            .is_none_or(|end| end > u16::MAX as usize)
                        {
                            return Err(format!(
                                "Constant pool overflow appending struct '{struct_name}' field names in '{func_name}'"
                            ));
                        }
                        for fname in field_names {
                            continuation_bc
                                .chunk
                                .constants
                                .push(Value::String(Rc::clone(fname)));
                        }
                        start as u16
                    };
                    continuation_bc.chunk.instructions[pc] = Instruction::NewStruct(
                        dst,
                        name_idx,
                        first_field_name_idx,
                        first_field,
                        count,
                    );
                }
                Instruction::GetIndex(dst, base_reg, idx_reg) => {
                    let Some(MirType::Struct(struct_name)) =
                        refined.locals.get(base_reg as usize).map(|l| &l.ty)
                    else {
                        continue;
                    };
                    let field_idx = match pc
                        .checked_sub(1)
                        .and_then(|prev| continuation_bc.chunk.instructions.get(prev))
                    {
                        Some(Instruction::LoadConst(reg, c_idx)) if *reg == idx_reg => {
                            match continuation_bc.chunk.constants.get(*c_idx as usize) {
                                Some(Value::Int(n)) if *n >= 0 => *n as usize,
                                other => {
                                    return Err(format!(
                                        "Invalid struct field index constant {other:?} at pc {pc} in '{func_name}'"
                                    ));
                                }
                            }
                        }
                        other => {
                            return Err(format!(
                                "Expected LoadConst before struct GetIndex at pc {pc} in '{func_name}', found {other:?}"
                            ));
                        }
                    };
                    let info = self.struct_metadata.get(struct_name).ok_or_else(|| {
                        format!(
                            "Missing struct layout metadata for '{struct_name}' rewriting GetIndex in '{func_name}'"
                        )
                    })?;
                    let fname = info.layout.field_names().get(field_idx).ok_or_else(|| {
                        format!(
                            "Struct '{struct_name}' field index {field_idx} out of bounds in '{func_name}'"
                        )
                    })?;
                    if struct_name != "LuaTable" && field_idx <= u8::MAX as usize {
                        continuation_bc.chunk.instructions[pc] =
                            Instruction::GetFieldSlot(dst, base_reg, field_idx as u8);
                        continue;
                    }
                    let fname_val = Value::String(Rc::clone(fname));
                    let fname_exists = continuation_bc
                        .chunk
                        .constants
                        .iter()
                        .any(|v| v == &fname_val);
                    if !fname_exists && continuation_bc.chunk.constants.len() >= u16::MAX as usize {
                        return Err(format!(
                            "Constant pool overflow rewriting struct GetField in '{func_name}'"
                        ));
                    }
                    let field_const_idx = continuation_bc.chunk.add_constant(fname_val);
                    continuation_bc.chunk.instructions[pc] =
                        Instruction::GetField(dst, base_reg, field_const_idx);
                }
                Instruction::SetIndex(base_reg, idx_reg, val_reg) => {
                    let Some(MirType::Struct(struct_name)) =
                        refined.locals.get(base_reg as usize).map(|l| &l.ty)
                    else {
                        continue;
                    };
                    let field_idx = match pc
                        .checked_sub(1)
                        .and_then(|prev| continuation_bc.chunk.instructions.get(prev))
                    {
                        Some(Instruction::LoadConst(reg, c_idx)) if *reg == idx_reg => {
                            match continuation_bc.chunk.constants.get(*c_idx as usize) {
                                Some(Value::Int(n)) if *n >= 0 => *n as usize,
                                other => {
                                    return Err(format!(
                                        "Invalid struct field index constant {other:?} at pc {pc} in '{func_name}'"
                                    ));
                                }
                            }
                        }
                        other => {
                            return Err(format!(
                                "Expected LoadConst before struct SetIndex at pc {pc} in '{func_name}', found {other:?}"
                            ));
                        }
                    };
                    let info = self.struct_metadata.get(struct_name).ok_or_else(|| {
                        format!(
                            "Missing struct layout metadata for '{struct_name}' rewriting SetIndex in '{func_name}'"
                        )
                    })?;
                    let fname = info.layout.field_names().get(field_idx).ok_or_else(|| {
                        format!(
                            "Struct '{struct_name}' field index {field_idx} out of bounds in '{func_name}'"
                        )
                    })?;
                    if struct_name != "LuaTable" && field_idx <= u8::MAX as usize {
                        continuation_bc.chunk.instructions[pc] =
                            Instruction::SetFieldSlot(base_reg, field_idx as u8, val_reg);
                        continue;
                    }
                    let fname_val = Value::String(Rc::clone(fname));
                    let fname_exists = continuation_bc
                        .chunk
                        .constants
                        .iter()
                        .any(|v| v == &fname_val);
                    if !fname_exists && continuation_bc.chunk.constants.len() >= u16::MAX as usize {
                        return Err(format!(
                            "Constant pool overflow rewriting struct SetField in '{func_name}'"
                        ));
                    }
                    let field_const_idx = continuation_bc.chunk.add_constant(fname_val);
                    continuation_bc.chunk.instructions[pc] =
                        Instruction::SetField(base_reg, field_const_idx, val_reg);
                }
                _ => {}
            }
        }
        continuation_bc.chunk.compute_bb_gas_weights();
        continuation_bc.chunk.build_packed_code();
        Ok(())
    }

    #[cfg(feature = "std")]
    fn build_function_deopt_recipes(
        &self,
        func_name: &str,
        mir_fn: &MirFunction,
        bc_fn: &crate::bytecode::Function,
        compilation_generation: u64,
        hook_authorized_for_fn: bool,
    ) -> core::result::Result<(Vec<CompiledDeoptRecipe>, bool), String> {
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
                    if Self::type_contains_unknown(&ty) || !self.is_supported_deopt_recipe_type(&ty) {
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
                if Self::type_contains_unknown(&ty) || !self.is_supported_deopt_recipe_type(&ty) {
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

    /// Syncs `self.struct_metadata` into `self.mir_context.struct_fields` ONLY when
    /// `self.mir_context_struct_gen != self.struct_metadata.generation()`.
    #[cfg(feature = "std")]
    pub(super) fn sync_mir_context_struct_metadata(&mut self) {
        let current_gen = self.struct_metadata.generation();
        if self.mir_context_struct_gen == current_gen {
            return;
        }
        {
            let mut sf = self.mir_context.struct_fields.borrow_mut();
            sf.clear();
            let meta_map = self.struct_metadata.as_map();
            for (name, info) in self.struct_metadata.iter() {
                self.mir_context.intern_struct_type_name(name);
                if !info.layout.name().is_empty() && info.layout.name() != name.as_str() {
                    self.mir_context.intern_struct_type_name(info.layout.name());
                }
                let field_types: Vec<MirType> = (0..info.layout.field_names().len())
                    .map(|i| {
                        Self::resolve_struct_field_mir_type_in_map(
                            name,
                            info.layout.field_type(i),
                            meta_map,
                        )
                    })
                    .collect();
                sf.insert(name.clone(), field_types);
            }
        }
        self.mir_context_struct_gen = current_gen;
        self.cranelift_struct_sync_count += 1;
    }

    /// Invalidates all Cranelift compiled artifacts, deopt recipes, and metadata cache entries
    /// after a `struct_metadata` generation change so stale struct layouts can never execute.
    #[cfg(feature = "std")]
    pub(super) fn invalidate_cranelift_for_struct_change(&mut self) {
        self.restore_all_ast_bytecode_backups();
        self.cranelift_jit = None;
        self.cranelift_compiled.clear();
        self.cranelift_recipes.clear();
        self.cranelift_fn_cache.clear();
        self.cranelift_compiled_struct_gen = 0;
        self.cranelift_generation = self.cranelift_generation.wrapping_add(1).max(1);
    }

    /// Collects all VM function names referenced by non-Cranelift bytecode functions (indirect calls,
    /// closures, globals, `CallMethod`), runtime global closures, or MIR `Rvalue::Closure` nodes so
    /// dead-helper DCE never prunes an indirectly-callable or closure-captured helper.
    #[cfg(feature = "std")]
    fn collect_bytecode_referenced_functions(
        &self,
        eligible_names: &hashbrown::HashSet<String>,
    ) -> hashbrown::HashSet<String> {
        use crate::bytecode::Instruction;

        let mut referenced = hashbrown::HashSet::new();

        for func in &self.functions {
            if eligible_names.contains(&func.name) {
                continue;
            }
            for constant in &func.chunk.constants {
                match constant {
                    Value::Function(idx) => {
                        if let Some(target) = self.functions.get(*idx) {
                            referenced.insert(target.name.clone());
                        }
                    }
                    Value::Closure(cl) => {
                        if let Some(target) = self.functions.get(cl.function_idx) {
                            referenced.insert(target.name.clone());
                        }
                    }
                    _ => {}
                }
            }
            for insn in &func.chunk.instructions {
                match *insn {
                    Instruction::Closure(_, fn_idx, _, _) => {
                        if let Some(target) = self.functions.get(fn_idx as usize) {
                            referenced.insert(target.name.clone());
                        }
                    }
                    Instruction::LoadGlobal(_, const_idx) => {
                        if let Some(Value::String(gname)) =
                            func.chunk.constants.get(const_idx as usize)
                        {
                            let gname_str = gname.as_str();
                            if let Some(val) = self.globals.get(gname_str) {
                                match val {
                                    Value::Function(idx) => {
                                        if let Some(target) = self.functions.get(*idx) {
                                            referenced.insert(target.name.clone());
                                        }
                                    }
                                    Value::Closure(cl) => {
                                        if let Some(target) = self.functions.get(cl.function_idx) {
                                            referenced.insert(target.name.clone());
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            let dot_suffix = format!(".{gname_str}");
                            let colon_suffix = format!("::{gname_str}");
                            for target in &self.functions {
                                if target.name == gname_str
                                    || target.name.ends_with(&dot_suffix)
                                    || target.name.ends_with(&colon_suffix)
                                {
                                    referenced.insert(target.name.clone());
                                }
                            }
                        }
                    }
                    Instruction::CallMethod(_, method_const_idx, _, _, _) => {
                        if let Some(Value::String(mname)) =
                            func.chunk.constants.get(method_const_idx as usize)
                        {
                            let method_suffix = format!(":{}", mname.as_str());
                            for target in &self.functions {
                                if target.is_method
                                    && (target.name == mname.as_str()
                                        || target.name.ends_with(&method_suffix))
                                {
                                    referenced.insert(target.name.clone());
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Preserve any function captured in a live global closure value.
        for val in self.globals.values() {
            if let Value::Closure(cl) = val
                && let Some(target) = self.functions.get(cl.function_idx)
            {
                referenced.insert(target.name.clone());
            }
        }

        // Preserve any function targeted by an Rvalue::Closure in any registered MIR function
        // (including ineligible MIR functions).
        for mir_fn in self.mir_functions.values() {
            for block in &mir_fn.blocks {
                for stmt in &block.statements {
                    if let Statement::Assign(_, Rvalue::Closure { func_name, .. }, _) = stmt {
                        referenced.insert(func_name.clone());
                    }
                }
            }
        }

        referenced
    }

    #[cfg(feature = "std")]
    fn refine_mir_function_with_callee_rets(
        mir_fn: &MirFunction,
        callee_rets: &hashbrown::HashMap<String, MirType>,
    ) -> MirFunction {
        let mut f = mir_fn.clone();
        apply_module_callee_return_types(&mut f, callee_rets);
        crate::mir::cranelift_backend::jit_engine::refine_mir_function(&f)
    }

    /// Lightweight compile-time fail-closed deopt eligibility validation without compiling
    /// continuation bytecode or allocating `CompiledDeoptRecipe` entries (`NG-W14-7`).
    /// Returns `Ok((has_effect_site, has_loop_header))`.
    #[cfg(feature = "std")]
    fn validate_function_deopt_eligibility(
        &self,
        func_name: &str,
        refined: &MirFunction,
        hook_authorized_for_fn: bool,
    ) -> core::result::Result<(bool, bool), String> {
        if refined.locals.len() > u8::MAX as usize {
            return Err(format!(
                "MIR local count {} exceeds 255-register VM limit in '{}'",
                refined.locals.len(),
                func_name
            ));
        }

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
                for idx in proven_indices {
                    let ty = refined.locals.get(idx).map(|l| &l.ty).ok_or_else(|| {
                        format!("Local index {} out of bounds in '{}'", idx, func_name)
                    })?;
                    if Self::type_contains_unknown(ty) || !self.is_supported_deopt_recipe_type(ty) {
                        return Err(format!(
                            "Unsupported deopt local {} type {:?} at site {} in '{}'",
                            idx, ty, b_idx, func_name
                        ));
                    }
                }
            }
        }

        let loop_headers = find_loop_headers(&refined.blocks);
        let has_loop_header = !loop_headers.is_empty();
        for b_idx in loop_headers {
            let proven_indices = refined
                .derive_proven_site_metadata_at_entry(b_idx)
                .map_err(|err| {
                    format!(
                        "Cannot derive authoritative deopt metadata for loop header safepoint {}: {}",
                        b_idx, err
                    )
                })?;
            for idx in proven_indices {
                let ty = refined.locals.get(idx).map(|l| &l.ty).ok_or_else(|| {
                    format!("Local index {} out of bounds in '{}'", idx, func_name)
                })?;
                if Self::type_contains_unknown(ty) || !self.is_supported_deopt_recipe_type(ty) {
                    return Err(format!(
                        "Unsupported deopt local {} type {:?} at loop header {} in '{}'",
                        idx, ty, b_idx, func_name
                    ));
                }
            }
        }

        if has_effect_site || has_loop_header {
            for local in &refined.locals {
                if let MirType::Struct(struct_name) = &local.ty
                    && self.struct_metadata.get(struct_name).is_none()
                {
                    return Err(format!(
                        "Missing struct layout metadata for '{struct_name}' in deopt-capable function '{func_name}'"
                    ));
                }
            }
        }

        Ok((has_effect_site, has_loop_header))
    }

    #[cfg(feature = "std")]
    fn compile_continuation_bytecode_for_mir(
        &self,
        name: &str,
        func_idx: usize,
        candidate: &MirFunction,
        direct_callees: &hashbrown::HashSet<String>,
        direct_callee_indices: &hashbrown::HashMap<String, usize>,
    ) -> core::result::Result<crate::bytecode::Function, String> {
        if candidate.locals.len() > u8::MAX as usize {
            return Err(format!(
                "MIR local count {} exceeds 255-register VM limit in '{}'",
                candidate.locals.len(),
                name
            ));
        }
        let mut continuation_bc =
            crate::mir::vm_backend::MirToBytecodeCompiler::compile_with_direct_callees_and_indices(
                candidate,
                direct_callees,
                direct_callee_indices,
            )
            .map_err(|err| {
                format!(
                    "Failed to compile MIR continuation bytecode for '{}': {:?}",
                    name, err
                )
            })?;
        if (continuation_bc.register_count as usize) < candidate.locals.len() {
            return Err(format!(
                "MIR continuation register_count overflow for '{}' (locals={}, register_count={})",
                name,
                candidate.locals.len(),
                continuation_bc.register_count
            ));
        }
        if let Some(orig_fn) = self.functions.get(func_idx) {
            continuation_bc.is_method = orig_fn.is_method;
            continuation_bc.signature = orig_fn.signature.clone();
        }
        self.rewrite_continuation_struct_ops(candidate, &mut continuation_bc)?;
        Ok(continuation_bc)
    }

    /// Builds the candidate post-inline Cranelift MIR module with:
    /// - Step A (`PERF-T1-4`): JIT-safe MIR inlining + SROA + tier1 optimizations.
    /// - Dead-Helper DCE (`NG-W14-7`): computes reachable native functions from active roots +
    ///   exported/indirectly-referenced functions after inlining and prunes 100%-inlined helpers
    ///   with 0 remaining callers before Cranelift codegen.
    /// - Lazy Deopt Preparation (`NG-W14-7`): validates fail-closed deopt eligibility while
    ///   deferring `CompiledDeoptRecipe` materialization (and script/non-entry continuation
    ///   bytecode compilation) until first actual deopt event.
    #[cfg(feature = "std")]
    fn prepare_cranelift_module(
        &self,
        requested_roots: Option<&hashbrown::HashSet<String>>,
    ) -> Result<PreparedCraneliftModule> {
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

        // Pre-Step A: validate bytecode function presence and effect-hook authorization, and
        // seed initial `may_deopt_effect` set before running the initial call-graph fixpoint.
        let mut func_idx_by_name: hashbrown::HashMap<String, usize> = hashbrown::HashMap::new();
        let mut pre_may_deopt_effect: hashbrown::HashSet<String> = hashbrown::HashSet::new();
        let pre_candidates: Vec<String> = eligible.iter().cloned().collect();
        for name in pre_candidates {
            let Some(mir_fn) = self.mir_functions.get(&name) else {
                eligible.remove(&name);
                continue;
            };
            let hook_auth = hook_target.as_deref() == Some(name.as_str());
            let Some(func_idx) = self.functions.iter().position(|f| f.name == name) else {
                let err_msg = format!("Missing bytecode function for '{}'", name);
                if hook_auth {
                    return Err(LustError::RuntimeError { message: err_msg });
                }
                eligible.remove(&name);
                continue;
            };
            let mut unauthorized_effect: Option<String> = None;
            let mut has_effect = false;
            for block in &mir_fn.blocks {
                if let Some(Terminator::Call { callee, .. }) = &block.terminator
                    && is_effect_or_handler_callee(callee)
                {
                    if !is_vm_effect_intrinsic_callee(callee) && !hook_auth {
                        unauthorized_effect = Some(format!(
                            "Callee '{}' in '{}' requires test-deopt hook authorization",
                            callee, name
                        ));
                        break;
                    }
                    has_effect = true;
                }
            }
            if let Some(err_msg) = unauthorized_effect {
                if hook_auth {
                    return Err(LustError::RuntimeError { message: err_msg });
                }
                eligible.remove(&name);
                continue;
            }
            if has_effect {
                pre_may_deopt_effect.insert(name.clone());
            }
            func_idx_by_name.insert(name, func_idx);
        }

        // Initial call-graph eligibility fixpoint so only Cranelift-eligible functions enter Step A.
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
                        if Self::is_cranelift_builtin_callee(callee)
                            || is_effect_or_handler_callee(callee)
                        {
                            return false;
                        }
                        if !eligible.contains(callee) || pre_may_deopt_effect.contains(callee) {
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
            return Ok(PreparedCraneliftModule {
                module: crate::mir::ir::MirModule::new("vm_cranelift_module".into()),
                eligible,
                func_idx_by_name,
                eager_continuations: hashbrown::HashMap::new(),
                deferred_continuations: hashbrown::HashSet::new(),
                deferred_dead_helpers: hashbrown::HashSet::new(),
                effective_roots: hashbrown::HashSet::new(),
                skipped_dead_helper_count: 0,
            });
        }

        // Step A (Hazard 3 / PERF-T1-4): Clone raw eligible functions into candidate module
        // and run JIT-safe MIR inlining + post-inline optimization BEFORE reachability & Steps B/C.
        let mut sorted_names: Vec<String> = eligible.iter().cloned().collect();
        sorted_names.sort();

        let mut module = crate::mir::ir::MirModule::new("vm_cranelift_module".into());
        let mut pre_inline_fns: hashbrown::HashMap<String, MirFunction> =
            hashbrown::HashMap::with_capacity(sorted_names.len());
        for name in &sorted_names {
            if let Some(raw_fn) = self.mir_functions.get(name) {
                pre_inline_fns.insert(name.clone(), raw_fn.clone());
                module.add_function(raw_fn.clone());
            }
        }

        let mut jit_inline_cfg = crate::mir::opt::JitInlineConfig::default();
        if let Some(ref target) = hook_target {
            jit_inline_cfg.excluded_functions.insert(target.clone());
        }
        let sroa_layouts = self.build_sroa_struct_layouts();
        let mut inlined_names =
            crate::mir::opt::MirOptimizer::optimize_module_for_jit_with_layouts(
                &mut module,
                &jit_inline_cfg,
                &sroa_layouts,
            );

        // Fail-closed post-inline eligibility check: revert any inlined function that became ineligible.
        for func in &mut module.functions {
            if inlined_names.contains(&func.signature.name)
                && !self.is_cranelift_eligible_in_vm(func)
            {
                if let Some(orig) = pre_inline_fns.get(&func.signature.name) {
                    *func = orig.clone();
                }
                inlined_names.remove(&func.signature.name);
            }
        }

        // Compute effective reachability roots AFTER MIR inlining (`NG-W14-7` Dead-Helper DCE).
        let bytecode_referenced = self.collect_bytecode_referenced_functions(&eligible);
        let mut effective_roots: hashbrown::HashSet<String> = match requested_roots {
            Some(req) => req.clone(),
            None => compute_default_module_roots(&module),
        };
        for name in bytecode_referenced {
            if eligible.contains(&name) {
                effective_roots.insert(name);
            }
        }
        if let Some(ref target) = hook_target
            && eligible.contains(target)
        {
            effective_roots.insert(target.clone());
        }

        // Build callee return type map once for refinement reuse.
        let callee_rets: hashbrown::HashMap<String, MirType> = self
            .mir_functions
            .iter()
            .map(|(k, v)| (k.clone(), v.signature.return_type.clone()))
            .collect();

        // Step B & Step C (`NG-W14-7` Lazy Deopt Preparation + Dead-Helper DCE):
        // Validate deopt eligibility on reachable functions, reverting to pre-inline MIR on failure
        // (and re-computing reachability if a revert re-exposes previously-inlined callees).
        let mut reachable = compute_reachable_native_functions(&module, &effective_roots);
        let mut validated: hashbrown::HashSet<String> = hashbrown::HashSet::new();
        let mut may_deopt_effect: hashbrown::HashSet<String> = hashbrown::HashSet::new();
        let mut eager_continuations: hashbrown::HashMap<String, crate::bytecode::Function> =
            hashbrown::HashMap::new();
        let mut deferred_continuations: hashbrown::HashSet<String> = hashbrown::HashSet::new();
        let mut direct_callees_cache: Option<hashbrown::HashSet<String>> = None;
        let mut direct_callee_indices_cache: Option<hashbrown::HashMap<String, usize>> = None;

        loop {
            let mut reverted_any = false;
            let mut progress = false;

            for func in &mut module.functions {
                let name = func.signature.name.clone();
                if !reachable.contains(&name) || !eligible.contains(&name) || validated.contains(&name)
                {
                    continue;
                }
                progress = true;
                let hook_auth = hook_target.as_deref() == Some(name.as_str());
                let Some(&func_idx) = func_idx_by_name.get(&name) else {
                    eligible.remove(&name);
                    continue;
                };

                let is_script_entry = name == "__script" || name.starts_with("__init@");
                let is_nested_only_callee = requested_roots
                    .is_some_and(|req| !req.contains(&name))
                    && !hook_auth;

                let validate_or_compile = |candidate: &MirFunction,
                                           direct_callees_cache: &mut Option<
                    hashbrown::HashSet<String>,
                >,
                                           direct_callee_indices_cache: &mut Option<
                    hashbrown::HashMap<String, usize>,
                >|
                 -> core::result::Result<
                    (bool, bool, Option<crate::bytecode::Function>),
                    String,
                > {
                    let refined = Self::refine_mir_function_with_callee_rets(candidate, &callee_rets);
                    let (has_effect, has_loop) =
                        self.validate_function_deopt_eligibility(&name, &refined, hook_auth)?;
                    if !has_effect && !has_loop {
                        // Zero deopt sites: never needs continuation bytecode or recipes.
                        return Ok((false, false, None));
                    }
                    let defer_bc =
                        (is_script_entry || is_nested_only_callee) && candidate.locals.len() <= 220;
                    if defer_bc {
                        return Ok((has_effect, true, None));
                    }
                    let direct_callees = direct_callees_cache
                        .get_or_insert_with(|| self.collect_direct_callees());
                    let direct_callee_indices = direct_callee_indices_cache.get_or_insert_with(|| {
                        self.functions
                            .iter()
                            .enumerate()
                            .map(|(idx, f)| (f.name.clone(), idx))
                            .collect()
                    });
                    let continuation_bc = self.compile_continuation_bytecode_for_mir(
                        &name,
                        func_idx,
                        candidate,
                        direct_callees,
                        direct_callee_indices,
                    )?;
                    Ok((has_effect, false, Some(continuation_bc)))
                };

                let attempt = match validate_or_compile(
                    func,
                    &mut direct_callees_cache,
                    &mut direct_callee_indices_cache,
                ) {
                    Ok(ok) => Ok(ok),
                    Err(_) if inlined_names.contains(&name) => {
                        if let Some(orig) = pre_inline_fns.get(&name) {
                            *func = orig.clone();
                        }
                        inlined_names.remove(&name);
                        reverted_any = true;
                        validate_or_compile(
                            func,
                            &mut direct_callees_cache,
                            &mut direct_callee_indices_cache,
                        )
                    }
                    Err(err) => Err(err),
                };

                match attempt {
                    Ok((has_effect, is_deferred, opt_bc)) => {
                        if has_effect {
                            may_deopt_effect.insert(name.clone());
                        }
                        if is_deferred {
                            deferred_continuations.insert(name.clone());
                        } else if let Some(bc) = opt_bc {
                            eager_continuations.insert(name.clone(), bc);
                        }
                        validated.insert(name);
                    }
                    Err(err_msg) => {
                        if hook_auth {
                            return Err(LustError::RuntimeError { message: err_msg });
                        }
                        eligible.remove(&name);
                    }
                }
            }

            if reverted_any {
                reachable = compute_reachable_native_functions(&module, &effective_roots);
                continue;
            }
            if !progress {
                break;
            }
        }

        // Partition post-inline eligible functions into live reachable vs dead-inlined helpers.
        let mut deferred_dead_helpers: hashbrown::HashSet<String> = hashbrown::HashSet::new();
        for f in &module.functions {
            if eligible.contains(&f.signature.name) && !reachable.contains(&f.signature.name) {
                deferred_dead_helpers.insert(f.signature.name.clone());
            }
        }
        let skipped_dead_helper_count = deferred_dead_helpers.len();

        // Retain only live reachable & eligible functions for call-graph fixpoint & codegen.
        eligible.retain(|name| reachable.contains(name));
        module
            .functions
            .retain(|f| eligible.contains(&f.signature.name));

        // Post-Step C call-graph eligibility fixpoint on the FINAL live post-inline module.
        loop {
            let param_counts: hashbrown::HashMap<&str, usize> = module
                .functions
                .iter()
                .map(|f| (f.signature.name.as_str(), f.params.len()))
                .collect();
            let to_remove: Vec<String> = module
                .functions
                .iter()
                .filter(|mir_fn| {
                    mir_fn.blocks.iter().any(|block| {
                        let Some(Terminator::Call { callee, args, .. }) = &block.terminator else {
                            return false;
                        };
                        if Self::is_cranelift_builtin_callee(callee)
                            || is_effect_or_handler_callee(callee)
                        {
                            return false;
                        }
                        if !eligible.contains(callee) || may_deopt_effect.contains(callee) {
                            return true;
                        }
                        param_counts
                            .get(callee.as_str())
                            .is_none_or(|&cnt| cnt != args.len())
                    })
                })
                .map(|f| f.signature.name.clone())
                .collect();
            if to_remove.is_empty() {
                break;
            }
            drop(param_counts);
            for name in to_remove {
                eligible.remove(&name);
                eager_continuations.remove(&name);
                deferred_continuations.remove(&name);
            }
            module
                .functions
                .retain(|f| eligible.contains(&f.signature.name));
        }

        // Prune any newly-orphaned helpers after fixpoint removal.
        if !module.functions.is_empty() {
            let post_fixpoint_reachable =
                compute_reachable_native_functions(&module, &effective_roots);
            let mut extra_skipped = 0usize;
            for f in &module.functions {
                if !post_fixpoint_reachable.contains(&f.signature.name) {
                    deferred_dead_helpers.insert(f.signature.name.clone());
                    extra_skipped += 1;
                }
            }
            if extra_skipped > 0 {
                eligible.retain(|name| post_fixpoint_reachable.contains(name));
                eager_continuations.retain(|name, _| eligible.contains(name));
                deferred_continuations.retain(|name| eligible.contains(name));
                module
                    .functions
                    .retain(|f| eligible.contains(&f.signature.name));
            }
        }

        let skipped_dead_helper_count = deferred_dead_helpers.len().max(skipped_dead_helper_count);

        Ok(PreparedCraneliftModule {
            module,
            eligible,
            func_idx_by_name,
            eager_continuations,
            deferred_continuations,
            deferred_dead_helpers,
            effective_roots,
            skipped_dead_helper_count,
        })
    }

    /// Read-only inspection helper returning the exact post-inline compiled `MirFunction`
    /// for a function currently compiled in Cranelift JIT (zero runtime dispatch overhead).
    #[cfg(feature = "std")]
    pub fn cranelift_compiled_mir(&self, func_name: &str) -> Option<MirFunction> {
        if !self.cranelift_compiled.contains_key(func_name) {
            return None;
        }
        if let Some(mir) = self
            .cranelift_jit
            .as_ref()
            .and_then(|jit| jit.post_inline_mir(func_name))
        {
            return Some(mir.clone());
        }
        let prepared = self.prepare_cranelift_module(None).ok()?;
        prepared
            .module
            .functions
            .into_iter()
            .find(|f| f.signature.name == func_name)
    }

    /// Read-only inspection helper returning the number of dead inlined helpers skipped from
    /// Cranelift codegen during the most recent compilation (`NG-W14-7`).
    #[cfg(feature = "std")]
    pub fn cranelift_skipped_dead_helper_count(&self) -> usize {
        self.cranelift_jit
            .as_ref()
            .map_or(0, |jit| jit.skipped_dead_helper_count())
    }

    /// Read-only inspection helper returning the number of on-demand lazy deopt materializations
    /// performed for the current Cranelift compilation (`NG-W14-7`).
    #[cfg(feature = "std")]
    pub fn cranelift_lazy_deopt_materialization_count(&self) -> u64 {
        self.cranelift_jit
            .as_ref()
            .map_or(0, |jit| jit.lazy_deopt_materialization_count())
    }

    /// Read-only inspection helper returning the number of currently materialized deopt recipes
    /// in `self.cranelift_recipes` (`NG-W14-7`).
    #[cfg(feature = "std")]
    pub fn cranelift_materialized_recipe_count(&self) -> usize {
        self.cranelift_recipes.len()
    }

    /// Read-only inspection helper returning whether `func_name` currently has deferred
    /// continuation bytecode awaiting first actual deopt (`NG-W14-7`).
    #[cfg(feature = "std")]
    pub fn cranelift_has_deferred_continuation(&self, func_name: &str) -> bool {
        self.cranelift_jit
            .as_ref()
            .is_some_and(|jit| jit.is_deferred_continuation(func_name))
    }

    /// Read-only inspection helper returning whether `func_name` was pruned as a dead inlined helper
    /// for the active compilation root set (`NG-W14-7`).
    #[cfg(feature = "std")]
    pub fn cranelift_is_deferred_dead_helper(&self, func_name: &str) -> bool {
        self.cranelift_jit
            .as_ref()
            .is_some_and(|jit| jit.is_deferred_dead_helper(func_name))
    }

    /// Read-only inspection helper returning the current compiled Cranelift generation.
    #[cfg(feature = "std")]
    pub fn cranelift_compiled_generation(&self) -> u64 {
        self.cranelift_compiled_generation
    }

    /// Read-only inspection helper returning the per-thread count of non-empty Cranelift
    /// array/struct writeback commits.
    #[cfg(feature = "std")]
    pub fn cranelift_writeback_commit_count(&self) -> u64 {
        cranelift_writeback_commit_count()
    }

    /// Read-only inspection helper returning the proven `readonly_params` flags for a compiled
    /// Cranelift function.
    #[cfg(feature = "std")]
    pub fn cranelift_readonly_params(&self, func_name: &str) -> Option<Vec<bool>> {
        let func_idx = self.functions.iter().position(|f| f.name == func_name)?;
        self.cranelift_fn_cache
            .get(func_idx)?
            .as_ref()
            .map(|entry| entry.readonly_params.clone())
    }

    /// Read-only inspection helper returning the compile-time `has_external_or_host_calls` flag
    /// for a compiled Cranelift function (`PERF-T4-3`).
    #[cfg(feature = "std")]
    pub fn cranelift_has_external_or_host_calls(&self, func_name: &str) -> Option<bool> {
        let func_idx = self.functions.iter().position(|f| f.name == func_name)?;
        self.cranelift_fn_cache
            .get(func_idx)?
            .as_ref()
            .map(|entry| entry.has_external_or_host_calls)
    }

    /// Builds scalar-only `MirStructLayout` descriptors from VM `struct_metadata` for MIR SROA
    /// (`NG-W11-1`) without polluting `MirModule::struct_layouts` (which controls C-flat codegen).
    #[cfg(feature = "std")]
    fn build_sroa_struct_layouts(
        &self,
    ) -> hashbrown::HashMap<String, crate::mir::ir::MirStructLayout> {
        use crate::data::layout::{CanonicalTypeTag, LogicalFieldDesc, LogicalLayoutV1};

        let mut layouts = hashbrown::HashMap::new();
        for (name, info) in self.struct_metadata.iter() {
            if !self.is_cranelift_supported_struct(name) {
                continue;
            }
            let field_names = info.layout.field_names();
            if field_names.is_empty() {
                continue;
            }
            let mut fields = Vec::with_capacity(field_names.len());
            let mut all_scalar = true;
            for (idx, fname) in field_names.iter().enumerate() {
                let tag = match &info.layout.field_type(idx).kind {
                    TypeKind::Int => CanonicalTypeTag::Int64,
                    TypeKind::Float => CanonicalTypeTag::Float64,
                    TypeKind::Bool => CanonicalTypeTag::Bool,
                    TypeKind::String => CanonicalTypeTag::StringSlice,
                    _ => {
                        all_scalar = false;
                        break;
                    }
                };
                fields.push(LogicalFieldDesc::new(idx as u16, fname.as_str(), tag));
            }
            if all_scalar {
                let logical = LogicalLayoutV1::new(0, name.clone(), fields);
                layouts.insert(name.clone(), crate::mir::ir::MirStructLayout::from(&logical));
            }
        }
        layouts
    }

    /// Read-only inspection helper returning the SROA struct recipes for a compiled Cranelift
    /// function (`NG-W11-1`).
    #[cfg(feature = "std")]
    pub fn cranelift_sroa_recipes(
        &self,
        func_name: &str,
    ) -> Option<Vec<crate::mir::ir::SroaStructRecipe>> {
        let func_idx = self.functions.iter().position(|f| f.name == func_name)?;
        self.cranelift_fn_cache
            .get(func_idx)?
            .as_ref()
            .map(|entry| entry.sroa_recipes.clone())
    }

    /// Read-only inspection helper returning whether a compiled Cranelift function qualifies for
    /// the 0..=6-arg direct scalar C-ABI fast path (`NG-W10-3` + `NG-W15-7`).
    #[cfg(feature = "std")]
    pub fn cranelift_is_pure_scalar_abi_6(&self, func_name: &str) -> Option<bool> {
        let func_idx = self.functions.iter().position(|f| f.name == func_name)?;
        self.cranelift_fn_cache
            .get(func_idx)?
            .as_ref()
            .map(|entry| {
                entry.is_pure_scalar_abi
                    || Self::entry_is_5_or_6_uniform_scalar_abi(entry)
            })
    }

    #[cfg(feature = "std")]
    #[inline]
    fn entry_is_5_or_6_uniform_scalar_abi(entry: &crate::vm::CompiledFnEntry) -> bool {
        !entry.has_external_or_host_calls
            && matches!(entry.param_types.len(), 5 | 6)
            && (entry.param_types.iter().all(|t| matches!(t, MirType::Int))
                || entry.param_types.iter().all(|t| matches!(t, MirType::Float)))
            && matches!(entry.ret_type, MirType::Int | MirType::Float)
    }

    /// Returns true when `func_idx` can use the 5- or 6-arg direct scalar Cranelift C-ABI fast
    /// path (`NG-W15-7`), mirroring `VM::has_native_entry` without changing `is_pure_scalar_abi`
    /// (`<= 4` args contract tested by `ng_w10_3_direct_scalar_call_abi`).
    #[cfg(feature = "std")]
    #[inline]
    pub(crate) fn has_cranelift_direct_scalar_entry_5_6(&self, func_idx: usize) -> bool {
        if self.backend != crate::vm::VmBackend::Cranelift
            || (self.slow_path_flags & !crate::vm::SLOW_PATH_GAS_LIMIT) != 0
            || crate::vm::MUTATION_M1_SKIP_GAS.load(core::sync::atomic::Ordering::Relaxed)
            || !self.call_stack.is_empty()
            || self.active_jit_depth != 0
            || self.active_native_depth != 0
            || self.budgets.memory.limit_bytes.is_some()
            || self.has_active_finalizers
            || crate::vm::tracing::has_trace_hook()
        {
            return false;
        }
        #[cfg(feature = "test-deopt")]
        if self.test_deopt_spike_hook.is_some() {
            return false;
        }
        let Some(func) = self.functions.get(func_idx) else {
            return false;
        };
        if !matches!(func.param_count, 5 | 6) {
            return false;
        }
        let current_struct_gen = self.struct_metadata.generation();
        if self.cranelift_compiled_generation == self.cranelift_generation
            && self.cranelift_compiled_struct_gen == current_struct_gen
            && self.mir_context_struct_gen == current_struct_gen
        {
            if let Some(Some(entry)) = self.cranelift_fn_cache.get(func_idx) {
                return Self::entry_is_5_or_6_uniform_scalar_abi(entry)
                    && !entry.code_ptr.is_null()
                    && entry.compiled_generation == self.cranelift_compiled_generation
                    && entry.struct_metadata_generation == current_struct_gen;
            }
            return self
                .cranelift_jit
                .as_ref()
                .is_some_and(|jit| jit.is_deferred_dead_helper(&func.name));
        }
        !self.mir_functions.is_empty() && self.mir_functions.contains_key(&func.name)
    }

    /// Read-only inspection helper returning the compile-time `is_pure_scalar_abi` flag
    /// for a compiled Cranelift function (`NG-W10-3`).
    #[cfg(feature = "std")]
    pub fn cranelift_is_pure_scalar_abi(&self, func_name: &str) -> Option<bool> {
        let func_idx = self.functions.iter().position(|f| f.name == func_name)?;
        self.cranelift_fn_cache
            .get(func_idx)?
            .as_ref()
            .map(|entry| entry.is_pure_scalar_abi)
    }

    /// Read-only inspection helper returning the total Cranelift native entry count.
    #[cfg(feature = "std")]
    pub fn cranelift_entry_count(&self) -> u64 {
        self.cranelift_entry_count
    }

    /// Read-only inspection helper returning the number of borrowed read-only flat-array cache hits.
    #[cfg(feature = "std")]
    pub fn cranelift_array_cache_hits(&self) -> u64 {
        self.cranelift_array_cache_hits
    }

    /// Read-only inspection helper returning the number of borrowed read-only flat-array cache builds.
    #[cfg(feature = "std")]
    pub fn cranelift_array_cache_builds(&self) -> u64 {
        self.cranelift_array_cache_builds
    }

    /// Read-only inspection helper returning the cached `local_types.len()` for a compiled
    /// Cranelift function.
    #[cfg(feature = "std")]
    pub fn cranelift_compiled_local_count(&self, func_name: &str) -> Option<usize> {
        let func_idx = self.functions.iter().position(|f| f.name == func_name)?;
        self.cranelift_fn_cache
            .get(func_idx)?
            .as_ref()
            .map(|entry| entry.local_types.len())
    }

    /// Read-only inspection helper returning the compiled deopt recipe sites
    /// `(block_id, site_kind, precise_pc)` for `func_name`.
    /// When recipes are lazily deferred (`NG-W14-7`), derives the sites read-only from the stored
    /// post-inline MIR snapshot without mutating VM state.
    #[cfg(feature = "std")]
    pub fn cranelift_deopt_recipe_sites(&self, func_name: &str) -> Vec<(usize, DeoptSiteKind, usize)> {
        let mut sites: Vec<(usize, DeoptSiteKind, usize)> = self
            .cranelift_recipes
            .values()
            .filter(|r| r.func_name == func_name)
            .map(|r| (r.block_id, r.site_kind, r.precise_pc))
            .collect();
        if sites.is_empty()
            && let Some(jit) = self.cranelift_jit.as_ref()
            && let Some(mir_snapshot) = jit.post_inline_mir(func_name)
            && let Some(func_idx) = self.functions.iter().position(|f| f.name == func_name)
        {
            #[cfg(feature = "test-deopt")]
            let hook_auth = self
                .test_deopt_spike_hook
                .as_ref()
                .is_some_and(|h| h.target_function == func_name);
            #[cfg(not(feature = "test-deopt"))]
            let hook_auth = false;

            let use_installed = !jit.is_deferred_continuation(func_name)
                && !self.functions[func_idx].block_entry_pcs.is_empty();
            let temp_bc;
            let bc_ref = if use_installed {
                &self.functions[func_idx]
            } else {
                let direct_callees = self.collect_direct_callees();
                let direct_callee_indices: hashbrown::HashMap<String, usize> = self
                    .functions
                    .iter()
                    .enumerate()
                    .map(|(idx, f)| (f.name.clone(), idx))
                    .collect();
                match self.compile_continuation_bytecode_for_mir(
                    func_name,
                    func_idx,
                    mir_snapshot,
                    &direct_callees,
                    &direct_callee_indices,
                ) {
                    Ok(bc) => {
                        temp_bc = bc;
                        &temp_bc
                    }
                    Err(_) => return sites,
                }
            };
            if let Ok((recipes, _)) = self.build_function_deopt_recipes(
                func_name,
                mir_snapshot,
                bc_ref,
                self.cranelift_compiled_generation,
                hook_auth,
            ) {
                sites.extend(
                    recipes
                        .into_iter()
                        .map(|r| (r.block_id, r.site_kind, r.precise_pc)),
                );
            }
        }
        sites.sort_by_key(|&(block_id, _, precise_pc)| (block_id, precise_pc));
        sites
    }

    /// Compiles all uncompiled reachable eligible registered MIR functions using CraneliftJitCompiler.
    #[cfg(feature = "std")]
    pub fn compile_cranelift(&mut self) -> Result<()> {
        self.compile_cranelift_internal(None)
    }

    #[cfg(feature = "std")]
    fn compile_cranelift_internal(&mut self, requested_root: Option<&str>) -> Result<()> {
        let _guard = self.mir_context.enter();
        let current_struct_gen = self.struct_metadata.generation();
        if self.cranelift_compiled_struct_gen != current_struct_gen
            && (!self.cranelift_compiled.is_empty()
                || !self.cranelift_recipes.is_empty()
                || !self.cranelift_fn_cache.is_empty())
        {
            self.invalidate_cranelift_for_struct_change();
        }
        self.sync_mir_context_struct_metadata();

        let next_gen = if self.cranelift_compiled_generation == self.cranelift_generation {
            self.cranelift_generation.wrapping_add(1).max(1)
        } else {
            self.cranelift_generation.max(1)
        };

        let requested_roots: Option<hashbrown::HashSet<String>> = requested_root.map(|root| {
            let mut roots = hashbrown::HashSet::new();
            roots.insert(root.to_string());
            if self.cranelift_compiled_generation == self.cranelift_generation
                && self.cranelift_compiled_struct_gen == current_struct_gen
                && let Some(jit) = self.cranelift_jit.as_ref()
            {
                for prev_root in jit.compiled_roots() {
                    roots.insert(prev_root.clone());
                }
            }
            roots
        });

        let mut prepared = self.prepare_cranelift_module(requested_roots.as_ref())?;

        if prepared.eligible.is_empty() {
            if requested_root.is_none() || prepared.deferred_dead_helpers.is_empty() {
                self.restore_all_ast_bytecode_backups();
                self.cranelift_compiled.clear();
                self.cranelift_recipes.clear();
                self.cranelift_fn_cache.clear();
                self.cranelift_fn_cache.resize(self.functions.len(), None);
                self.cranelift_compiled_generation = self.cranelift_generation;
                self.cranelift_compiled_struct_gen = current_struct_gen;
            }
            return Ok(());
        }

        let all_compiled = self.cranelift_jit.is_some()
            && self.cranelift_compiled_generation == self.cranelift_generation
            && self.cranelift_compiled_struct_gen == current_struct_gen
            && prepared.eligible.len() == self.cranelift_compiled.len()
            && prepared
                .eligible
                .iter()
                .all(|name| self.cranelift_compiled.contains_key(name));
        if all_compiled {
            return Ok(());
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

        let compiled_map =
            match compiler.compile_module_with_roots(&prepared.module, &prepared.effective_roots) {
                Ok(map) => map,
                Err(e) => {
                    return Err(LustError::RuntimeError {
                        message: format!("Cranelift JIT module compilation failed: {:?}", e),
                    });
                }
            };

        let backend_skipped = compiler.skipped_dead_helper_count();
        let total_skipped_dead_helpers = prepared
            .skipped_dead_helper_count
            .saturating_add(backend_skipped);

        // Precompute per-function refined metadata and read-only parameter masks strictly from
        // the FINAL live post-inline compiled module functions before mutating VM state.
        let callee_rets: hashbrown::HashMap<String, MirType> = self
            .mir_functions
            .iter()
            .map(|(k, v)| (k.clone(), v.signature.return_type.clone()))
            .collect();
        let mut refined_module = crate::mir::ir::MirModule::new(prepared.module.name.clone());
        let mut post_inline_mirs: hashbrown::HashMap<String, MirFunction> =
            hashbrown::HashMap::with_capacity(prepared.module.functions.len());
        let mut entry_meta_by_name: hashbrown::HashMap<
            String,
            (
                Vec<MirType>,
                MirType,
                Vec<MirType>,
                Vec<crate::mir::ir::SroaStructRecipe>,
            ),
        > = hashbrown::HashMap::with_capacity(prepared.module.functions.len());
        for f in &prepared.module.functions {
            if !compiled_map.contains_key(&f.signature.name) {
                continue;
            }
            let refined = Self::refine_mir_function_with_callee_rets(f, &callee_rets);
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
            entry_meta_by_name.insert(
                f.signature.name.clone(),
                (
                    param_types,
                    f.signature.return_type.clone(),
                    local_types,
                    f.sroa_decomposed_structs.clone(),
                ),
            );
            post_inline_mirs.insert(f.signature.name.clone(), f.clone());
            refined_module.add_function(refined);
        }
        let mut readonly_params_by_name = compute_readonly_params_with_structs(
            &refined_module,
            Some(self.struct_metadata.as_map()),
        );
        let mut has_ext_calls_by_name = compute_has_external_or_host_calls(&refined_module);

        compiler.set_post_inline_snapshot(
            post_inline_mirs,
            prepared.deferred_continuations,
            prepared.deferred_dead_helpers,
            prepared.effective_roots,
            total_skipped_dead_helpers,
        );

        // Transactional publish ONLY after compile_module_with_roots returns Ok.
        let mut installed_cont_indices: hashbrown::HashSet<usize> = hashbrown::HashSet::new();
        for name in compiled_map.keys() {
            if prepared.eager_continuations.contains_key(name)
                && let Some(&func_idx) = prepared.func_idx_by_name.get(name)
            {
                installed_cont_indices.insert(func_idx);
            }
        }
        // Restore any previously-compiled function that was dropped or deferred in this recompilation.
        let dropped_indices: Vec<usize> = self
            .ast_bytecode_backup
            .keys()
            .copied()
            .filter(|idx| !installed_cont_indices.contains(idx))
            .collect();
        for func_idx in dropped_indices {
            if let Some(orig_fn) = self.ast_bytecode_backup.remove(&func_idx)
                && func_idx < self.functions.len()
            {
                let meta = crate::vm::CallMeta::of(&orig_fn);
                self.functions[func_idx] = orig_fn;
                if func_idx < self.call_meta.len() {
                    self.call_meta[func_idx] = meta;
                }
            }
        }

        self.cranelift_jit = Some(compiler);
        self.cranelift_compiled.clear();
        self.cranelift_recipes.clear();
        self.cranelift_fn_cache.clear();
        self.cranelift_fn_cache.resize(self.functions.len(), None);
        for (name, ptr) in compiled_map {
            if let Some(&func_idx) = prepared.func_idx_by_name.get(&name)
                && func_idx < self.functions.len()
            {
                if let Some(continuation_bc) = prepared.eager_continuations.remove(&name) {
                    let meta = crate::vm::CallMeta::of(&continuation_bc);
                    let prev_fn = core::mem::replace(&mut self.functions[func_idx], continuation_bc);
                    self.ast_bytecode_backup.entry(func_idx).or_insert(prev_fn);
                    if func_idx < self.call_meta.len() {
                        self.call_meta[func_idx] = meta;
                    }
                }
                if let Some((param_types, ret_type, local_types, sroa_recipes)) =
                    entry_meta_by_name.remove(&name)
                    && param_types.len() == self.functions[func_idx].param_count as usize
                    && !Self::type_contains_unknown(&ret_type)
                    && self.is_cranelift_supported_type_in_vm(&ret_type)
                {
                    let readonly_params = readonly_params_by_name
                        .remove(&name)
                        .filter(|flags| flags.len() == param_types.len())
                        .unwrap_or_else(|| vec![false; param_types.len()]);
                    let has_external_or_host_calls = has_ext_calls_by_name
                        .remove(&name)
                        .unwrap_or(true);
                    let is_scalar = |t: &MirType| {
                        matches!(t, MirType::Int | MirType::Float | MirType::Bool)
                    };
                    let is_pure_scalar_abi = !has_external_or_host_calls
                        && param_types.len() <= 4
                        && param_types.iter().all(is_scalar)
                        && (is_scalar(&ret_type) || matches!(ret_type, MirType::Unit));
                    self.cranelift_fn_cache[func_idx] = Some(Rc::new(crate::vm::CompiledFnEntry {
                        func_name: alloc::rc::Rc::<str>::from(name.as_str()),
                        code_ptr: ptr,
                        param_types,
                        ret_type,
                        local_types,
                        compiled_generation: next_gen,
                        struct_metadata_generation: current_struct_gen,
                        readonly_params,
                        has_external_or_host_calls,
                        is_pure_scalar_abi,
                        sroa_recipes,
                    }));
                }
            }
            self.cranelift_compiled.insert(name, ptr);
            self.cranelift_compilation_count += 1;
        }
        self.cranelift_generation = next_gen;
        self.cranelift_compiled_generation = next_gen;
        self.cranelift_compiled_struct_gen = current_struct_gen;
        Ok(())
    }

    #[cfg(not(feature = "std"))]
    pub fn compile_cranelift(&mut self) -> Result<()> {
        Ok(())
    }

    /// Sub-frame guard (`NG-W15-7`): detects trivial loop-free leaf methods whose VM<->Cranelift
    /// struct-marshaling cost far exceeds running the few bytecode instructions in `'frame_fast`.
    #[cfg(feature = "std")]
    #[inline]
    fn is_trivial_leaf_method_for_vm_subframe(func: &crate::bytecode::Function) -> bool {
        use crate::bytecode::Instruction;
        if !func.is_method || func.chunk.instructions.len() > 24 {
            return false;
        }
        for insn in &func.chunk.instructions {
            match insn {
                Instruction::Jump(off)
                | Instruction::JumpIf(_, off)
                | Instruction::JumpIfNot(_, off)
                | Instruction::BranchLtInt(_, _, off)
                | Instruction::BranchLeInt(_, _, off) => {
                    if *off < 0 {
                        return false;
                    }
                }
                Instruction::Call(..) | Instruction::CallMethod(..) | Instruction::Closure(..) => {
                    return false;
                }
                _ => {}
            }
        }
        true
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
        if func_idx >= self.functions.len() {
            return Err(LustError::RuntimeError {
                message: format!("Function index {} not found in VM", func_idx),
            });
        }

        #[cfg(feature = "test-deopt")]
        let has_spike_hook = self.test_deopt_spike_hook.is_some();
        #[cfg(not(feature = "test-deopt"))]
        let has_spike_hook = false;

        if self.call_stack.len() > 1
            && !has_spike_hook
            && Self::is_trivial_leaf_method_for_vm_subframe(&self.functions[func_idx])
        {
            return Ok(None);
        }

        let current_struct_gen = self.struct_metadata.generation();

        let entry = if self.cranelift_compiled_generation == self.cranelift_generation
            && self.cranelift_compiled_struct_gen == current_struct_gen
            && self.mir_context_struct_gen == current_struct_gen
            && func_idx < self.cranelift_fn_cache.len()
        {
            match self.cranelift_fn_cache[func_idx].as_ref() {
                Some(cached)
                    if cached.compiled_generation == self.cranelift_compiled_generation
                        && cached.struct_metadata_generation == current_struct_gen
                        && cached.param_types.len() == param_count =>
                {
                    Rc::clone(cached)
                }
                None if !has_spike_hook
                    && !self.cranelift_jit.as_ref().is_some_and(|jit| {
                        jit.is_deferred_dead_helper(&self.functions[func_idx].name)
                    }) =>
                {
                    // Already compiled for current (cranelift_generation, struct_metadata_generation)
                    // and this func_idx was ineligible (not a DCE-deferred helper).
                    return Ok(None);
                }
                _ => {
                    // Fallback to slow path if hook active, DCE-deferred helper invoked directly,
                    // or metadata mismatch.
                    let Some(e) = self.resolve_cranelift_fn_entry_slow(func_idx, param_count)? else {
                        return Ok(None);
                    };
                    e
                }
            }
        } else {
            let Some(e) = self.resolve_cranelift_fn_entry_slow(func_idx, param_count)? else {
                return Ok(None);
            };
            e
        };

        // Step E (Hazard 3 / Hazard 10): Ensure top_frame register window capacity covers installed
        // continuation bytecode register_count, param_count, and MIR local_types before native entry.
        let required_regs = self
            .functions
            .get(func_idx)
            .map(|f| f.register_count as usize)
            .unwrap_or(0)
            .max(entry.local_types.len())
            .max(param_count);
        if required_regs > 256 || self.call_stack.is_empty() {
            return Ok(None);
        }
        let cur_reg_count = self.call_stack.last().unwrap().reg_count as usize;
        if cur_reg_count < param_count {
            return Ok(None);
        }
        if required_regs > cur_reg_count {
            self.ensure_top_frame_reg_capacity(required_regs)?;
        }
        if required_regs > param_count {
            let base_reg = self.call_stack.last().unwrap().base_reg;
            for slot in &mut self.vm_registers[base_reg + param_count..base_reg + required_regs] {
                *slot = Value::Nil;
            }
        }

        let baseline = self.mir_context.capture_baseline();
        let mut native_entered = false;
        let outcome = self.try_run_cranelift_top_frame_scoped(
            func_idx,
            &entry.func_name,
            &entry.param_types,
            &entry.ret_type,
            &entry.local_types,
            &entry.readonly_params,
            &entry.sroa_recipes,
            entry.has_external_or_host_calls,
            entry.code_ptr,
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
                    let _ = self.pop_call_frame();
                }
                Err(err)
            }
        }
    }

    #[cfg(feature = "std")]
    fn resolve_cranelift_fn_entry_slow(
        &mut self,
        func_idx: usize,
        param_count: usize,
    ) -> Result<Option<Rc<crate::vm::CompiledFnEntry>>> {
        let current_struct_gen = self.struct_metadata.generation();
        if self.cranelift_compiled_struct_gen != current_struct_gen
            && (!self.cranelift_compiled.is_empty()
                || !self.cranelift_recipes.is_empty()
                || !self.cranelift_fn_cache.is_empty())
        {
            self.invalidate_cranelift_for_struct_change();
        }

        let func_name = match self.functions.get(func_idx) {
            Some(f) => f.name.clone(),
            None => {
                return Err(LustError::RuntimeError {
                    message: format!("Function index {} not found in VM", func_idx),
                });
            }
        };

        let is_eligible = match self.mir_functions.get(&func_name) {
            Some(f) => {
                self.cranelift_dispatch_refine_count += 1;
                self.is_cranelift_eligible_in_vm(f)
            }
            None => return Ok(None),
        };
        if !is_eligible {
            return Ok(None);
        }

        // Compile reachable closure from `func_name` before entering MirContext invocation scope
        // and packing arguments (`NG-W14-7` demand-driven dead-helper DCE).
        if !self.cranelift_compiled.contains_key(&func_name)
            || self.cranelift_compiled_generation != self.cranelift_generation
            || self.cranelift_compiled_struct_gen != self.struct_metadata.generation()
        {
            self.compile_cranelift_internal(Some(&func_name))?;
        }

        let post_struct_gen = self.struct_metadata.generation();
        if self.mir_context_struct_gen != post_struct_gen {
            self.sync_mir_context_struct_metadata();
        }

        let entry = self
            .cranelift_fn_cache
            .get(func_idx)
            .and_then(|slot| slot.as_ref())
            .filter(|entry| {
                entry.compiled_generation == self.cranelift_compiled_generation
                    && entry.struct_metadata_generation == post_struct_gen
                    && entry.param_types.len() == param_count
            })
            .cloned();
        Ok(entry)
    }

    #[cfg(feature = "std")]
    #[allow(clippy::too_many_arguments)]
    fn marshal_struct_recursive(
        &self,
        rc: &Rc<StructObject>,
        expected_name: &str,
        expected_info_hint: Option<&crate::vm::RuntimeStructInfo>,
        param_idx: usize,
        marshalled_structs: &mut Vec<MarshalledStruct>,
        marshalled_struct_by_ptr: &mut hashbrown::HashMap<*const StructObject, usize>,
        visiting: &mut hashbrown::HashSet<*const StructObject>,
        depth: usize,
    ) -> Result<Option<*mut i64>> {
        if depth > MAX_STRUCT_DECODE_DEPTH {
            return Err(LustError::RuntimeError {
                message: format!(
                    "Struct marshaling depth limit ({MAX_STRUCT_DECODE_DEPTH}) exceeded for '{expected_name}'"
                ),
            });
        }
        let rc_key = Rc::as_ptr(rc);
        if visiting.contains(&rc_key) {
            return Err(LustError::RuntimeError {
                message: format!(
                    "Cyclic struct reference detected while marshaling '{expected_name}' at {:p}",
                    rc_key
                ),
            });
        }
        if let Some(&existing_idx) = marshalled_struct_by_ptr.get(&rc_key) {
            let existing = &mut marshalled_structs[existing_idx];
            if !existing.param_indices.contains(&param_idx) {
                existing.param_indices.push(param_idx);
            }
            return Ok(Some(existing.ptr));
        }

        // `is_cranelift_supported_struct` validates the full transitive closure of
        // `Option<Named(sub)>` targets at `depth == 0`, so recursive steps do not need
        // to re-run the global-lock + HashSet traversal on every node.
        if depth == 0 && !self.is_cranelift_supported_struct(expected_name) {
            return Ok(None);
        }
        let expected_info = match expected_info_hint {
            Some(info) => info,
            None => match self.struct_metadata.get(expected_name) {
                Some(info) => info,
                None => return Ok(None),
            },
        };
        if rc.name.as_str() != expected_name
            && rc.name.as_str() != expected_info.layout.name()
        {
            return Ok(None);
        }
        let fields = rc.fields.try_borrow().map_err(|_| LustError::RuntimeError {
            message: "Failed to acquire borrow for struct parameter packing: mutable borrow active".into(),
        })?;
        let expected_field_count = expected_info.layout.field_names().len();
        if rc.layout.field_names() != expected_info.layout.field_names()
            || fields.len() != expected_field_count
        {
            return Ok(None);
        }

        enum RawFieldBuf {
            Inline([i64; 8], usize),
            Heap(Vec<i64>),
        }
        impl RawFieldBuf {
            #[inline]
            fn as_slice(&self) -> &[i64] {
                match self {
                    Self::Inline(arr, len) => &arr[..*len],
                    Self::Heap(v) => v.as_slice(),
                }
            }
        }

        visiting.insert(rc_key);
        let marshal_fields_res: Result<Option<RawFieldBuf>> = (|| {
            let mut raw_fields = if fields.len() <= 8 {
                RawFieldBuf::Inline([0; 8], 0)
            } else {
                RawFieldBuf::Heap(Vec::with_capacity(fields.len()))
            };
            for (idx, fld) in fields.iter().enumerate() {
                let raw_fld = match (fld, &expected_info.layout.field_type(idx).kind) {
                    (Value::Int(n), TypeKind::Int) => *n,
                    (Value::Bool(b), TypeKind::Bool) => i64::from(*b),
                    (Value::Float(f), TypeKind::Float) => f.to_bits() as i64,
                    (Value::String(_), TypeKind::String)
                    | (Value::Map(_), TypeKind::Map(_, _))
                    | (Value::Enum(_), TypeKind::Result(_, _)) => {
                        self.mir_context.register_handle(fld.clone())
                    }
                    (Value::Enum(_), TypeKind::Option(inner)) => match &inner.kind {
                        TypeKind::Named(sub) => {
                            let sub_name = resolve_option_struct_target_name(
                                expected_name,
                                sub,
                                Some(self.struct_metadata.as_map()),
                            );
                            match fld.as_enum() {
                                Some(("Option", "None", None))
                                | Some(("Option", "None", Some([]))) => 0i64,
                                Some(("Option", "Some", Some([Value::Struct(sub_rc)]))) => {
                                    let sub_hint = if sub_name.as_ref() == expected_name {
                                        Some(expected_info)
                                    } else {
                                        None
                                    };
                                    match self.marshal_struct_recursive(
                                        sub_rc,
                                        sub_name.as_ref(),
                                        sub_hint,
                                        param_idx,
                                        marshalled_structs,
                                        marshalled_struct_by_ptr,
                                        visiting,
                                        depth + 1,
                                    )? {
                                        Some(sub_ptr) => sub_ptr as i64,
                                        None => return Ok(None),
                                    }
                                }
                                _ => return Ok(None),
                            }
                        }
                        _ => self.mir_context.register_handle(fld.clone()),
                    },
                    _ => return Ok(None),
                };
                match &mut raw_fields {
                    RawFieldBuf::Inline(arr, len) => {
                        arr[*len] = raw_fld;
                        *len += 1;
                    }
                    RawFieldBuf::Heap(v) => v.push(raw_fld),
                }
            }
            Ok(Some(raw_fields))
        })();
        visiting.remove(&rc_key);
        let Some(raw_fields) = marshal_fields_res? else {
            return Ok(None);
        };

        let raw_slice = raw_fields.as_slice();
        let len = raw_slice.len();
        let ptr = self
            .mir_context
            .alloc_struct_with_fields(expected_name, raw_slice);
        if ptr.is_null() {
            return Err(LustError::RuntimeError {
                message: "Out of memory allocating struct buffer for Cranelift JIT".into(),
            });
        }
        let new_idx = marshalled_structs.len();
        marshalled_structs.push(MarshalledStruct {
            rc: rc.clone(),
            ptr,
            field_count: len,
            param_indices: vec![param_idx],
        });
        marshalled_struct_by_ptr.insert(rc_key, new_idx);
        Ok(Some(ptr))
    }

    #[cfg(feature = "std")]
    fn collect_readonly_borrowable_array_rcs(
        &self,
        param_types: &[MirType],
        readonly_params: &[bool],
    ) -> (Vec<Rc<LustArray>>, hashbrown::HashMap<*const LustArray, usize>) {
        let Some(frame) = self.call_stack.last() else {
            return (Vec::new(), hashbrown::HashMap::new());
        };
        if (frame.reg_count as usize) < param_types.len() {
            return (Vec::new(), hashbrown::HashMap::new());
        }
        let regs = &self.vm_registers[frame.base_reg..frame.base_reg + frame.reg_count as usize];
        let mut state: hashbrown::HashMap<
            *const LustArray,
            (Rc<LustArray>, Option<FlatArrayElemTag>, bool),
        > = hashbrown::HashMap::new();
        let mut order: Vec<*const LustArray> = Vec::new();
        for (i, expected_ty) in param_types.iter().enumerate() {
            if let Value::Array(rc) = &regs[i] {
                let key = Rc::as_ptr(rc);
                let is_ro = readonly_params.get(i).copied() == Some(true);
                let tag_opt = match expected_ty {
                    MirType::Array(elem_ty) => flat_array_elem_tag(elem_ty),
                    _ => None,
                };
                match state.get_mut(&key) {
                    Some((_, existing_tag, eligible)) => {
                        if !is_ro || tag_opt.is_none() || *existing_tag != tag_opt {
                            *eligible = false;
                        }
                    }
                    None => {
                        order.push(key);
                        state.insert(key, (Rc::clone(rc), tag_opt, is_ro && tag_opt.is_some()));
                    }
                }
            }
        }
        let mut rcs = Vec::new();
        let mut index_by_ptr = hashbrown::HashMap::new();
        for key in order {
            if let Some((rc, _, true)) = state.remove(&key) {
                let idx = rcs.len();
                rcs.push(rc);
                index_by_ptr.insert(key, idx);
            }
        }
        (rcs, index_by_ptr)
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
        readonly_params: &[bool],
        sroa_recipes: &[crate::mir::ir::SroaStructRecipe],
        has_external_or_host_calls: bool,
        code_ptr: *const u8,
        native_entered: &mut bool,
    ) -> Result<CraneliftOutcome> {
        let param_count = param_types.len();

        // Fast-path sync for specialized tracing-JIT caches (single thread-local Cell load == 0 in common case)
        // BEFORE acquiring shared Ref guards on borrowed read-only arrays.
        if crate::bytecode::value::specialized_cache_active_entries() > 0 {
            let _sync_vm_guard = crate::vm::CurrentVmGuard::from_vm(self);
            if let Some(frame) = self.call_stack.last() {
                let regs = &self.vm_registers[frame.base_reg..frame.base_reg + frame.reg_count as usize];
                for (i, expected_ty) in param_types.iter().enumerate().take(regs.len()) {
                    if let (Value::Array(rc), MirType::Array(_)) = (&regs[i], expected_ty) {
                        crate::bytecode::value::try_sync_specialized_array(rc)
                            .map_err(|msg| LustError::RuntimeError { message: msg })?;
                    }
                }
            }
        }

        // --- INVOCATION CONTEXT ACTIVATION, PINNING & DROP-ORDER INVARIANT (PERF-T4-3) ---
        // Rust drops locals strictly in REVERSE declaration order:
        //   Declared 1st -> Dropped 4th (LAST):  `pinned_borrowed_arrays: Vec<Rc<FlatArrayBuffer>>`
        //   Declared 2nd -> Dropped 3rd:         `borrowed_array_rcs: Vec<Rc<LustArray>>`
        //   Declared 3rd -> Dropped 2nd:         `_borrowed_array_guards: Vec<Ref<'_, Vec<Value>>>`
        //   Declared 4th -> Dropped 1st (FIRST): `_vm_guard` (drops `MirContextGuard`, unregistering
        //                                        all borrowed provenance records BEFORE buffers drop)
        let mut pinned_borrowed_arrays: Vec<Rc<FlatArrayBuffer>> = Vec::new();
        let (borrowed_array_rcs, borrowed_array_guard_idx) = if !has_external_or_host_calls
            && param_types.iter().any(|t| matches!(t, MirType::Array(_)))
        {
            self.collect_readonly_borrowable_array_rcs(param_types, readonly_params)
        } else {
            (Vec::new(), hashbrown::HashMap::new())
        };
        let _borrowed_array_guards: Vec<core::cell::Ref<'_, Vec<Value>>> = borrowed_array_rcs
            .iter()
            .map(|rc| {
                if rc.is_unboxed() {
                    Ok(rc.borrow_elements_without_sync())
                } else {
                    rc.try_borrow().map_err(|_| LustError::RuntimeError {
                        message:
                            "Failed to acquire borrow for array parameter packing: mutable borrow active"
                                .into(),
                    })
                }
            })
            .collect::<Result<Vec<_>>>()?;

        // Activate VM and MIR invocation context before any argument packing allocations.
        // CurrentVmGuard enters both CURRENT_VM_STACK (for console routing) and mir_context
        // once, keeping host marshaling allocations inside this scope's ownership floor.
        let _vm_guard = crate::vm::CurrentVmGuard::from_vm(self);

        let vm_gas_budget = self.gas_remaining();
        let vm_has_mem_budget = self.memory_budget_bytes().is_some();
        let mem_limit = match self.memory_remaining_bytes() {
            Some(rem_mem) => i64::try_from(rem_mem).unwrap_or(i64::MAX),
            None => i64::MAX,
        };

        self.mir_context.set_status(MIR_STATUS_OK);
        self.mir_context.set_memory_used(0);
        self.mir_context.set_max_memory_limit(i64::MAX);
        let gas_guard = crate::vm::GasScopeGuard::enter(
            Rc::clone(&self.mir_context),
            vm_gas_budget,
            self.active_jit_depth > 0,
        );

        self.sync_mir_context_struct_metadata();

        // Convert parameters to RawArg with alias-preserving tracking
        let mut raw_args = Vec::with_capacity(param_count);
        let mut marshalled_arrays: Vec<MarshalledArray> = Vec::new();
        let mut marshalled_array_by_ptr: hashbrown::HashMap<*const LustArray, usize> =
            hashbrown::HashMap::new();
        let mut marshalled_structs: Vec<MarshalledStruct> = Vec::new();
        let mut marshalled_struct_by_ptr: hashbrown::HashMap<*const StructObject, usize> =
            hashbrown::HashMap::new();
        let mut borrowed_cache_bytes_used: i64 = 0;
        let mut array_cache_hits_delta: u64 = 0;
        let mut array_cache_builds_delta: u64 = 0;

        {
            let frame = self.call_stack.last().unwrap();
            if (frame.reg_count as usize) < param_count {
                return Ok(CraneliftOutcome::Fallback);
            }
            let regs = &self.vm_registers[frame.base_reg..frame.base_reg + frame.reg_count as usize];
            for (i, expected_ty) in param_types.iter().enumerate() {
                let val = &regs[i];
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

                        // Borrowed read-only flat-view boundary fast-path (PERF-T4-3):
                        // Eligible ONLY when !has_external_or_host_calls, all aliased param
                        // positions are proven readonly, elem_ty is homogeneous Int/Float/Bool,
                        // and a shared Ref guard is pinned for the entire invocation.
                        if let Some(&guard_idx) = borrowed_array_guard_idx.get(&rc_key)
                            && let Some(flat_tag) = flat_array_elem_tag(elem_ty)
                        {
                            let elements_slice: &[Value] = &_borrowed_array_guards[guard_idx];
                            let saved_used = self.mir_context.memory_used();
                            self.mir_context.set_memory_used(borrowed_cache_bytes_used);
                            self.mir_context.set_max_memory_limit(mem_limit);
                            let build_res = build_or_get_cached_flat_array(
                                &self.mir_context,
                                rc,
                                elements_slice,
                                flat_tag,
                            );
                            borrowed_cache_bytes_used = self.mir_context.memory_used();
                            self.mir_context.set_memory_used(saved_used);
                            self.mir_context.set_max_memory_limit(i64::MAX);

                            match build_res? {
                                Some((buf, was_hit)) => {
                                    if was_hit {
                                        array_cache_hits_delta += 1;
                                    } else {
                                        array_cache_builds_delta += 1;
                                    }
                                    let buf_ptr = buf.as_ptr();
                                    self.mir_context
                                        .register_borrowed_readonly_array(
                                            buf_ptr,
                                            buf.word_len(),
                                            (**elem_ty).clone(),
                                        )
                                        .map_err(|message| LustError::RuntimeError { message })?;
                                    pinned_borrowed_arrays.push(buf);
                                    let new_idx = marshalled_arrays.len();
                                    marshalled_arrays.push(MarshalledArray {
                                        rc: rc.clone(),
                                        ptr: buf_ptr as *mut i64,
                                        elem_ty: (**elem_ty).clone(),
                                        param_indices: vec![i],
                                    });
                                    marshalled_array_by_ptr.insert(rc_key, new_idx);
                                    raw_args.push(RawArg::Int(buf_ptr as i64));
                                    continue;
                                }
                                None => {
                                    // Element type mismatch (e.g. mixed `[1, "two"]`): reserved
                                    // bytes already refunded & flat_cache untouched; fall back
                                    // transparently to interpreter just like standard marshalling.
                                    return Ok(CraneliftOutcome::Fallback);
                                }
                            }
                        }

                        let Some(elem_tag) = mir_array_elem_tag(elem_ty) else {
                            if matches!(&**elem_ty, MirType::Struct(_)) {
                                return Ok(CraneliftOutcome::Fallback);
                            }
                            return Err(LustError::RuntimeError {
                                message: format!(
                                    "Unsupported non-primitive array element type {:?}",
                                    elem_ty
                                ),
                            });
                        };
                        let unboxed_ptr_opt: Option<Result<Option<*mut i64>>> =
                            if rc.is_unboxed() {
                                rc.with_unboxed_data(|unb| {
                                    let len = match (unb, &**elem_ty) {
                                        (UnboxedArrayData::None, _) => return None,
                                        (UnboxedArrayData::Int(v), MirType::Int) => v.len(),
                                        (UnboxedArrayData::Float(v), MirType::Float) => v.len(),
                                        (UnboxedArrayData::Bool(v), MirType::Bool) => v.len(),
                                        (UnboxedArrayData::Int(v), _) if v.is_empty() => 0,
                                        (UnboxedArrayData::Float(v), _) if v.is_empty() => 0,
                                        (UnboxedArrayData::Bool(v), _) if v.is_empty() => 0,
                                        _ => return Some(Ok(None)),
                                    };
                                    let ptr = lust_mir_alloc_array_typed(len as i64, elem_tag);
                                    if ptr.is_null() {
                                        return Some(Err(LustError::RuntimeError {
                                            message: "Out of memory allocating array buffer for Cranelift JIT".into(),
                                        }));
                                    }
                                    if len > 0 {
                                        unsafe {
                                            let dst = ptr.add(2);
                                            match (unb, &**elem_ty) {
                                                (UnboxedArrayData::Int(v), MirType::Int) => {
                                                    core::ptr::copy_nonoverlapping(v.as_ptr(), dst, len);
                                                }
                                                (UnboxedArrayData::Float(v), MirType::Float) => {
                                                    core::ptr::copy_nonoverlapping(
                                                        v.as_ptr() as *const i64,
                                                        dst,
                                                        len,
                                                    );
                                                }
                                                (UnboxedArrayData::Bool(v), MirType::Bool) => {
                                                    core::ptr::copy_nonoverlapping(
                                                        v.as_ptr(),
                                                        dst as *mut u8,
                                                        len,
                                                    );
                                                }
                                                _ => unreachable!("unboxed array element type pre-verified"),
                                            }
                                        }
                                    }
                                    Some(Ok(Some(ptr)))
                                })
                            } else {
                                None
                            };

                        let ptr = if let Some(unb_res) = unboxed_ptr_opt {
                            match unb_res? {
                                Some(ptr) => ptr,
                                None => return Ok(CraneliftOutcome::Fallback),
                            }
                        } else {
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
                            let ptr = lust_mir_alloc_array_typed(len as i64, elem_tag);
                            if ptr.is_null() {
                                return Err(LustError::RuntimeError {
                                    message: "Out of memory allocating array buffer for Cranelift JIT".into(),
                                });
                            }
                            if matches!(&**elem_ty, MirType::Bool) {
                                let dst_bytes = unsafe { ptr.add(2) as *mut u8 };
                                for (idx, elem) in elements.iter().enumerate() {
                                    let Value::Bool(b) = elem else {
                                        unreachable!("bool array elements pre-verified");
                                    };
                                    unsafe {
                                        *dst_bytes.add(idx) = u8::from(*b);
                                    }
                                }
                            } else {
                                for (idx, elem) in elements.iter().enumerate() {
                                    let raw_elem = match (elem, &**elem_ty) {
                                        (Value::Int(n), MirType::Int) => *n,
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
                            }
                            ptr
                        };
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
                        let mut visiting = hashbrown::HashSet::new();
                        match self.marshal_struct_recursive(
                            rc,
                            expected_name,
                            None,
                            i,
                            &mut marshalled_structs,
                            &mut marshalled_struct_by_ptr,
                            &mut visiting,
                            0,
                        )? {
                            Some(ptr) => RawArg::Int(ptr as i64),
                            None => return Ok(CraneliftOutcome::Fallback),
                        }
                    }
                    (Value::Enum(_), MirType::Option(inner)) => match &**inner {
                        MirType::Struct(expected_name) => match val.as_enum() {
                            Some(("Option", "None", None))
                            | Some(("Option", "None", Some([]))) => RawArg::Int(0),
                            Some(("Option", "Some", Some([Value::Struct(rc)]))) => {
                                let mut visiting = hashbrown::HashSet::new();
                                match self.marshal_struct_recursive(
                                    rc,
                                    expected_name,
                                    None,
                                    i,
                                    &mut marshalled_structs,
                                    &mut marshalled_struct_by_ptr,
                                    &mut visiting,
                                    0,
                                )? {
                                    Some(ptr) => RawArg::Int(ptr as i64),
                                    None => return Ok(CraneliftOutcome::Fallback),
                                }
                            }
                            _ => return Ok(CraneliftOutcome::Fallback),
                        },
                        MirType::Int | MirType::Bool | MirType::Float | MirType::String => {
                            match val.as_enum() {
                                Some(("Option", "None", None))
                                | Some(("Option", "None", Some([]))) => RawArg::Int(0),
                                Some(("Option", "Some", Some([payload])))
                                    if matches!(
                                        (payload, &**inner),
                                        (Value::Int(_), MirType::Int)
                                            | (Value::Bool(_), MirType::Bool)
                                            | (Value::Float(_), MirType::Float)
                                            | (Value::String(_), MirType::String)
                                    ) =>
                                {
                                    let h = self.mir_context.register_handle(val.clone());
                                    RawArg::Int(h)
                                }
                                _ => return Ok(CraneliftOutcome::Fallback),
                            }
                        }
                        _ => return Ok(CraneliftOutcome::Fallback),
                    },
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

        self.cranelift_array_cache_hits += array_cache_hits_delta;
        self.cranelift_array_cache_builds += array_cache_builds_delta;

        // Apply guest memory budget immediately before native entry, carrying forward
        // any bytes newly allocated during borrowed read-only flat-cache builds.
        self.mir_context.set_memory_used(borrowed_cache_bytes_used);
        self.mir_context.set_max_memory_limit(mem_limit);

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

        let Some((raw_int, raw_float)) =
            Self::invoke_cranelift_raw_code_ptr(code_ptr, &raw_args, ret_type)
        else {
            *native_entered = false;
            self.cranelift_entry_count = self.cranelift_entry_count.saturating_sub(1);
            return Ok(CraneliftOutcome::Fallback);
        };

        let status = crate::mir::cranelift_backend::jit_engine::lust_mir_get_status();
        drop(_jit_guard);
        drop(_inv_guard);

        self.finish_cranelift_top_frame_scoped(
            func_name,
            ret_type,
            local_types,
            readonly_params,
            sroa_recipes,
            frame_depth,
            invocation_id,
            vm_has_mem_budget,
            gas_guard,
            &marshalled_arrays,
            &marshalled_structs,
            raw_int,
            raw_float,
            status,
        )
    }

    #[cfg(feature = "std")]
    fn invoke_cranelift_raw_code_ptr(
        code_ptr: *const u8,
        raw_args: &[RawArg],
        ret_type: &MirType,
    ) -> Option<(i64, f64)> {
        // Execute JIT compiled code matching argument types and return type
        let (raw_int, raw_float): (i64, f64) = match (raw_args, ret_type) {
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

            _ => return None,
        };
        Some((raw_int, raw_float))
    }

    /// Direct 5- and 6-argument scalar C-ABI fast path into Cranelift JIT without allocating a VM
    /// `CallFrame` (`NG-W15-7`, `BENCH-NG-15`), preserving `SEC-01` gas enforcement.
    #[cfg(feature = "std")]
    pub(crate) fn try_call_cranelift_scalar_direct_5_6(
        &mut self,
        func_idx: usize,
        args: &[Value],
    ) -> Option<Result<Value>> {
        if self.backend != crate::vm::VmBackend::Cranelift
            || (self.slow_path_flags & !crate::vm::SLOW_PATH_GAS_LIMIT) != 0
            || crate::vm::MUTATION_M1_SKIP_GAS.load(core::sync::atomic::Ordering::Relaxed)
            || !self.call_stack.is_empty()
            || self.active_jit_depth != 0
            || self.active_native_depth != 0
            || self.budgets.memory.limit_bytes.is_some()
            || self.has_active_finalizers
            || super::execution::get_effect_finalizer_count() > 0
            || crate::vm::tracing::has_trace_hook()
            || !matches!(args.len(), 5 | 6)
        {
            return None;
        }
        #[cfg(feature = "test-deopt")]
        if self.test_deopt_spike_hook.is_some() {
            return None;
        }
        if crate::mir::cranelift_backend::safepoint_stats().mask_depth != 0 {
            return None;
        }

        let current_struct_gen = self.struct_metadata.generation();
        let needs_compile = self.cranelift_compiled_generation != self.cranelift_generation
            || self.cranelift_compiled_struct_gen != current_struct_gen
            || self.mir_context_struct_gen != current_struct_gen
            || func_idx >= self.cranelift_fn_cache.len();
        if needs_compile {
            if self.mir_functions.is_empty() {
                return None;
            }
            self.compile_cranelift().ok()?;
        } else if self
            .cranelift_fn_cache
            .get(func_idx)
            .and_then(|s| s.as_ref())
            .is_none()
            && let Some(func_name) = self.functions.get(func_idx).map(|f| f.name.clone())
            && self
                .cranelift_jit
                .as_ref()
                .is_some_and(|jit| jit.is_deferred_dead_helper(&func_name))
        {
            self.compile_cranelift_internal(Some(&func_name)).ok()?;
        }

        let post_struct_gen = self.struct_metadata.generation();
        let entry = self.cranelift_fn_cache.get(func_idx)?.as_ref()?;
        if !Self::entry_is_5_or_6_uniform_scalar_abi(entry)
            || entry.code_ptr.is_null()
            || entry.compiled_generation != self.cranelift_compiled_generation
            || entry.struct_metadata_generation != post_struct_gen
            || args.len() != entry.param_types.len()
        {
            return None;
        }

        let arg_len = args.len();
        let mut raw_args = [RawArg::Int(0); 6];
        for (i, (val, expected_ty)) in args.iter().zip(entry.param_types.iter()).enumerate() {
            raw_args[i] = match (val, expected_ty) {
                (Value::Int(n), MirType::Int) => RawArg::Int(*n),
                (Value::Float(f), MirType::Float) => RawArg::Float(*f),
                _ => return None,
            };
        }
        let code_ptr = entry.code_ptr;
        let ret_type = entry.ret_type.clone();

        let baseline = self.mir_context.capture_baseline();
        let _mir_guard = self.mir_context.enter();

        let vm_gas_budget = self.gas_remaining();
        self.mir_context.set_status(MIR_STATUS_OK);
        self.mir_context.set_memory_used(0);
        self.mir_context.set_max_memory_limit(i64::MAX);
        let mut gas_guard = crate::vm::GasScopeGuard::enter(
            Rc::clone(&self.mir_context),
            vm_gas_budget,
            false,
        );

        self.cranelift_entry_count += 1;
        let _jit_guard = self.enter_jit_scope();

        // Charge base entry gas (matches `try_call_cranelift_scalar_direct` & `try_run_cranelift_top_frame_scoped`).
        self.mir_context.charge_gas(1);

        let invoked = if self.mir_context.last_status() == MIR_STATUS_OK {
            match Self::invoke_cranelift_raw_code_ptr(code_ptr, &raw_args[..arg_len], &ret_type) {
                Some(pair) => Some(pair),
                None => {
                    drop(_jit_guard);
                    self.cranelift_entry_count = self.cranelift_entry_count.saturating_sub(1);
                    if self.mir_context.dirty_allocations() {
                        self.mir_context
                            .clear_to_baselines(baseline.handles, baseline.buffers);
                    }
                    drop(gas_guard);
                    drop(_mir_guard);
                    return None;
                }
            }
        } else {
            Some((0, 0.0))
        };

        let status = self.mir_context.last_status();
        drop(_jit_guard);

        if status == MIR_STATUS_DEOPT {
            crate::mir::cranelift_backend::take_deopt_payload();
            self.mir_context.set_status(MIR_STATUS_OK);
            self.cranelift_entry_count = self.cranelift_entry_count.saturating_sub(1);
            if self.mir_context.dirty_allocations() {
                self.mir_context
                    .clear_to_baselines(baseline.handles, baseline.buffers);
            }
            drop(gas_guard);
            drop(_mir_guard);
            return None;
        }

        let gas_charge_res = gas_guard.finish_and_charge(&mut self.budgets);

        if self.mir_context.dirty_allocations() {
            self.mir_context
                .clear_to_baselines(baseline.handles, baseline.buffers);
        }
        drop(_mir_guard);

        if status == MIR_STATUS_OK {
            let (raw_int, raw_float) = invoked.unwrap_or((0, 0.0));
            let result_val = match ret_type {
                MirType::Float => Value::Float(raw_float),
                MirType::Int => Value::Int(raw_int),
                _ => Value::Nil,
            };
            return Some(gas_charge_res.map(|()| result_val));
        }

        if let Some(sticky_err) = crate::mir::cranelift_backend::take_current_sticky_error() {
            crate::mir::cranelift_backend::take_deopt_payload();
            return Some(Err(LustError::RuntimeError {
                message: format!("Native deopt helper error: {}", sticky_err),
            }));
        }

        crate::mir::cranelift_backend::take_deopt_payload();
        match status {
            MIR_STATUS_ERR_OUT_OF_GAS => Some(Err(match gas_charge_res {
                Err(e) => e,
                Ok(()) => LustError::RuntimeError {
                    message: "Out of gas during Cranelift JIT execution".into(),
                },
            })),
            MIR_STATUS_ERR_OOM => Some(Err(LustError::RuntimeError {
                message: "Out of memory during Cranelift JIT execution".into(),
            })),
            MIR_STATUS_ERR_DIV_BY_ZERO => Some(Err(LustError::RuntimeError {
                message: "Division or modulo by zero during Cranelift JIT execution".into(),
            })),
            MIR_STATUS_ERR_NEG_EXPONENT => Some(Err(LustError::RuntimeError {
                message: "Negative exponent in integer power during Cranelift JIT execution".into(),
            })),
            MIR_STATUS_ERR_OUT_OF_BOUNDS => Some(Err(LustError::RuntimeError {
                message: "Array index or pointer out of bounds during Cranelift JIT execution"
                    .into(),
            })),
            _ => {
                self.mir_context.set_status(MIR_STATUS_OK);
                None
            }
        }
    }

    #[cfg(feature = "std")]
    #[allow(clippy::too_many_arguments)]
    fn finish_cranelift_top_frame_scoped(
        &mut self,
        func_name: &str,
        ret_type: &MirType,
        local_types: &[MirType],
        readonly_params: &[bool],
        sroa_recipes: &[crate::mir::ir::SroaStructRecipe],
        frame_depth: usize,
        invocation_id: u64,
        vm_has_mem_budget: bool,
        mut gas_guard: crate::vm::GasScopeGuard,
        marshalled_arrays: &[MarshalledArray],
        marshalled_structs: &[MarshalledStruct],
        raw_int: i64,
        raw_float: f64,
        status: i64,
    ) -> Result<CraneliftOutcome> {
        let gas_charge_res = gas_guard.finish_and_charge(&mut self.budgets);

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
                    if let Err(wb_err) = stage_and_commit_all_writebacks_with_readonly(
                        &self.mir_context,
                        &marshalled_arrays,
                        &marshalled_structs,
                        readonly_params,
                        Some(self.struct_metadata.as_map()),
                    ) {
                        return Err(LustError::RuntimeError {
                            message: format!("{}; writeback failed: {}", err, wb_err),
                        });
                    }
                    return Err(err);
                }
                if let Err(err) = mem_charge_res {
                    crate::mir::cranelift_backend::take_deopt_payload();
                    if let Err(wb_err) = stage_and_commit_all_writebacks_with_readonly(
                        &self.mir_context,
                        &marshalled_arrays,
                        &marshalled_structs,
                        readonly_params,
                        Some(self.struct_metadata.as_map()),
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
                    readonly_params,
                    sroa_recipes,
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
                if let Err(wb_err) = stage_and_commit_all_writebacks_with_readonly(
                    &self.mir_context,
                    &marshalled_arrays,
                    &marshalled_structs,
                    readonly_params,
                    Some(self.struct_metadata.as_map()),
                ) {
                    return Err(LustError::RuntimeError {
                        message: format!("{}; writeback failed: {}", primary_err, wb_err),
                    });
                }
            }
            return Err(primary_err);
        }

        if let Err(err) = gas_charge_res {
            if let Err(wb_err) = stage_and_commit_all_writebacks_with_readonly(
                &self.mir_context,
                &marshalled_arrays,
                &marshalled_structs,
                readonly_params,
                Some(self.struct_metadata.as_map()),
            ) {
                return Err(LustError::RuntimeError {
                    message: format!("{}; writeback failed: {}", err, wb_err),
                });
            }
            return Err(err);
        }
        if let Err(err) = mem_charge_res {
            if let Err(wb_err) = stage_and_commit_all_writebacks_with_readonly(
                &self.mir_context,
                &marshalled_arrays,
                &marshalled_structs,
                readonly_params,
                Some(self.struct_metadata.as_map()),
            ) {
                return Err(LustError::RuntimeError {
                    message: format!("{}; writeback failed: {}", err, wb_err),
                });
            }
            return Err(err);
        }

        // Stage writebacks for all non-readonly marshalled input arrays and structs BEFORE mutating any caller state
        let staged_arrays = stage_input_writebacks_with_readonly(
            &self.mir_context,
            &marshalled_arrays,
            readonly_params,
        )?;
        let staged_structs = stage_struct_writebacks_with_readonly_and_meta(
            &self.mir_context,
            &marshalled_structs,
            readonly_params,
            Some(self.struct_metadata.as_map()),
        )?;

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
                if matches!(&**elem_ty, MirType::String | MirType::Struct(_))
                    && self
                        .mir_context
                        .get_active_allocation_record(raw_int as usize)
                        .is_none()
                    && let Some(Value::Array(rc)) = self.mir_context.get_handle(raw_int)
                {
                    Value::Array(rc)
                } else {
                    let decoded = validate_and_decode_array(
                        &self.mir_context,
                        marshalled_arrays,
                        raw_int,
                        elem_ty,
                    )?;
                    if let Some(orig_rc) = decoded.matched_input_rc {
                        Value::Array(orig_rc)
                    } else {
                        Value::Array(Rc::new(LustArray::new(decoded.elements)))
                    }
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
                validate_and_decode_struct_with_metadata(
                    &self.mir_context,
                    &marshalled_structs,
                    raw_int,
                    name,
                    Some(layout),
                    Some(self.struct_metadata.as_map()),
                )?
            }
            MirType::Option(inner) => match &**inner {
                MirType::Struct(name) => {
                    if raw_int == 0 {
                        Value::none()
                    } else {
                        let layout = self
                            .struct_metadata
                            .get(name)
                            .map(|info| &info.layout)
                            .ok_or_else(|| LustError::RuntimeError {
                                message: format!(
                                    "Missing struct layout metadata for '{name}' during Cranelift Option return decoding"
                                ),
                            })?;
                        let decoded_struct = validate_and_decode_struct_with_metadata(
                            &self.mir_context,
                            &marshalled_structs,
                            raw_int,
                            name,
                            Some(layout),
                            Some(self.struct_metadata.as_map()),
                        )?;
                        Value::some(decoded_struct)
                    }
                }
                MirType::Int | MirType::Bool | MirType::Float | MirType::String => {
                    if raw_int == 0 {
                        Value::none()
                    } else if let Some(val) = self.mir_context.get_handle(raw_int) {
                        if matches!(&val, Value::Enum(_)) {
                            val
                        } else {
                            Value::some(val)
                        }
                    } else {
                        return Err(LustError::RuntimeError {
                            message: format!(
                                "Missing Option<{:?}> handle {} in Cranelift return decoding",
                                inner, raw_int
                            ),
                        });
                    }
                }
                _ => {
                    return Err(LustError::RuntimeError {
                        message: format!(
                            "Unsupported Option return element type {:?} in Cranelift return decoding",
                            inner
                        ),
                    });
                }
            },
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
        readonly_params: &[bool],
        sroa_recipes: &[crate::mir::ir::SroaStructRecipe],
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

        // Lazy deopt preparation (`NG-W14-7`): materialize continuation bytecode (if deferred)
        // and `CompiledDeoptRecipe`s on demand from the stored post-inline MIR snapshot.
        // Keep new bytecode/recipes staged until all payload & writeback validations succeed
        // (preserving zero-mutation fail-closed atomicity).
        let mut staged_continuation_bc: Option<(usize, crate::bytecode::Function)> = None;
        let mut staged_recipes: Vec<CompiledDeoptRecipe> = Vec::new();
        let recipe = if let Some(cached) = self
            .cranelift_recipes
            .get(&(func_name.to_string(), target_block_id, payload.site_kind))
            .cloned()
        {
            cached
        } else {
            let mir_snapshot = self
                .cranelift_jit
                .as_ref()
                .and_then(|jit| jit.post_inline_mir(func_name).cloned());
            let is_deferred_bc = self
                .cranelift_jit
                .as_ref()
                .is_some_and(|jit| jit.is_deferred_continuation(func_name));
            if let Some(mir_snapshot) = mir_snapshot
                && let Some(func_idx) = self.functions.iter().position(|f| f.name == func_name)
            {
                #[cfg(feature = "test-deopt")]
                let hook_auth = self
                    .test_deopt_spike_hook
                    .as_ref()
                    .is_some_and(|h| h.target_function == func_name);
                #[cfg(not(feature = "test-deopt"))]
                let hook_auth = false;

                let need_bc = is_deferred_bc || self.functions[func_idx].block_entry_pcs.is_empty();
                let recipes = if need_bc {
                    let direct_callees = self.collect_direct_callees();
                    let direct_callee_indices: hashbrown::HashMap<String, usize> = self
                        .functions
                        .iter()
                        .enumerate()
                        .map(|(idx, f)| (f.name.clone(), idx))
                        .collect();
                    let continuation_bc = self
                        .compile_continuation_bytecode_for_mir(
                            func_name,
                            func_idx,
                            &mir_snapshot,
                            &direct_callees,
                            &direct_callee_indices,
                        )
                        .map_err(|message| LustError::RuntimeError { message })?;
                    let (recipes, _) = self
                        .build_function_deopt_recipes(
                            func_name,
                            &mir_snapshot,
                            &continuation_bc,
                            compiled_gen,
                            hook_auth,
                        )
                        .map_err(|message| LustError::RuntimeError { message })?;
                    staged_continuation_bc = Some((func_idx, continuation_bc));
                    recipes
                } else {
                    let (recipes, _) = self
                        .build_function_deopt_recipes(
                            func_name,
                            &mir_snapshot,
                            &self.functions[func_idx],
                            compiled_gen,
                            hook_auth,
                        )
                        .map_err(|message| LustError::RuntimeError { message })?;
                    recipes
                };
                let matched = recipes
                    .iter()
                    .find(|r| r.block_id == target_block_id && r.site_kind == payload.site_kind)
                    .cloned()
                    .ok_or_else(|| LustError::RuntimeError {
                        message: format!(
                            "Missing compiled deopt recipe for ({}, block {}, {:?}) ({source_label})",
                            func_name, target_block_id, payload.site_kind
                        ),
                    })?;
                staged_recipes = recipes;
                matched
            } else {
                return Err(LustError::RuntimeError {
                    message: format!(
                        "Missing compiled deopt recipe for ({}, block {}, {:?}) ({source_label})",
                        func_name, target_block_id, payload.site_kind
                    ),
                });
            }
        };
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
            .map(|frame| frame.reg_count as usize)
            .ok_or_else(|| LustError::RuntimeError {
                message: "Call stack empty during deopt reconstruction".into(),
            })?;
        // Hazard 10: MIR inlining / SROA / lazily-materialized continuation bytecode can require
        // more locals/scratch registers than initial AST `reg_count`. Compute effective capacity
        // required; actual window expansion happens ONLY after all payload/writeback validations
        // succeed (preserving zero-mutation fail-closed atomicity).
        let continuation_reg_count = if let Some((_, ref bc)) = staged_continuation_bc {
            bc.register_count as usize
        } else {
            self.call_stack
                .last()
                .and_then(|frame| self.functions.get(frame.function_idx))
                .map(|f| f.register_count as usize)
                .unwrap_or(0)
        };
        let max_sroa_orig = sroa_recipes
            .iter()
            .map(|r| r.orig_local.0 + 1)
            .max()
            .unwrap_or(0);
        let required_locals = local_types
            .len()
            .max(max_sroa_orig)
            .max(continuation_reg_count);
        let effective_reg_capacity = top_frame_reg_count.max(required_locals);

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

        let mut materialized_fresh_arrays: hashbrown::HashMap<usize, Rc<LustArray>> =
            hashbrown::HashMap::new();
        let mut materialized_fresh_structs: hashbrown::HashMap<usize, Rc<StructObject>> =
            hashbrown::HashMap::new();
        let mut validated_restorations = Vec::with_capacity(payload.live_locals.len());
        for &(loc_idx, raw_val) in &payload.live_locals {
            if loc_idx >= local_types.len() || loc_idx >= effective_reg_capacity {
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
                    if matches!(
                        &**inner,
                        MirType::Int
                            | MirType::Bool
                            | MirType::Float
                            | MirType::String
                            | MirType::Struct(_)
                    ) =>
                {
                    if matches!(&**inner, MirType::String | MirType::Struct(_))
                        && self
                            .mir_context
                            .get_active_allocation_record(raw_val as usize)
                            .is_none()
                        && let Some(Value::Array(rc)) = self.mir_context.get_handle(raw_val)
                    {
                        Value::Array(Rc::clone(&rc))
                    } else {
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
                                .or_insert_with(|| Rc::new(LustArray::new(decoded.elements)))
                                .clone();
                            Value::Array(rc)
                        }
                    }
                }
                MirType::Struct(name) => {
                    if !self.is_cranelift_supported_struct(name) {
                        return Err(LustError::RuntimeError {
                            message: format!(
                                "Unsupported struct type '{}' for deopt local {} in ({source_label})",
                                name, loc_idx
                            ),
                        });
                    }
                    let layout = self
                        .struct_metadata
                        .get(name)
                        .map(|info| &info.layout)
                        .ok_or_else(|| LustError::RuntimeError {
                            message: format!(
                                "Missing struct layout metadata for '{name}' during deopt reconstruction ({source_label})"
                            ),
                        })?;
                    let mut visited_ptrs = hashbrown::HashSet::new();
                    let decoded_val = validate_and_decode_struct_inner(
                        &self.mir_context,
                        marshalled_structs,
                        raw_val,
                        name,
                        Some(layout),
                        Some(self.struct_metadata.as_map()),
                        &mut visited_ptrs,
                        &mut materialized_fresh_structs,
                        0,
                    )?;
                    match decoded_val {
                        Value::Struct(rc) => Value::Struct(rc),
                        other => {
                            return Err(LustError::RuntimeError {
                                message: format!(
                                    "Expected Struct value for deopt local {} in ({source_label}), got {:?}",
                                    loc_idx, other
                                ),
                            });
                        }
                    }
                }
                MirType::Option(inner) => match &**inner {
                    MirType::Struct(name) => {
                        if !self.is_cranelift_supported_struct(name) {
                            return Err(LustError::RuntimeError {
                                message: format!(
                                    "Unsupported Option<Struct('{}')> type for deopt local {} in ({source_label})",
                                    name, loc_idx
                                ),
                            });
                        }
                        if raw_val == 0 {
                            Value::none()
                        } else {
                            let layout = self
                                .struct_metadata
                                .get(name)
                                .map(|info| &info.layout)
                                .ok_or_else(|| LustError::RuntimeError {
                                    message: format!(
                                        "Missing struct layout metadata for '{name}' during deopt Option reconstruction ({source_label})"
                                    ),
                                })?;
                            let mut visited_ptrs = hashbrown::HashSet::new();
                            let decoded_val = validate_and_decode_struct_inner(
                                &self.mir_context,
                                marshalled_structs,
                                raw_val,
                                name,
                                Some(layout),
                                Some(self.struct_metadata.as_map()),
                                &mut visited_ptrs,
                                &mut materialized_fresh_structs,
                                0,
                            )?;
                            Value::some(decoded_val)
                        }
                    }
                    MirType::Int | MirType::Bool | MirType::Float | MirType::String => {
                        if raw_val == 0 {
                            Value::none()
                        } else {
                            match self.mir_context.get_handle(raw_val) {
                                Some(Value::Enum(e_rc)) if e_rc.enum_name.as_str() == "Option" => {
                                    Value::Enum(Rc::clone(&e_rc))
                                }
                                Some(other) => {
                                    return Err(LustError::RuntimeError {
                                        message: format!(
                                            "Deopt payload ({source_label}) invalid handle {} for Option<{:?}> local {}: expected Option enum, got {:?}",
                                            raw_val, inner, loc_idx, other
                                        ),
                                    });
                                }
                                None => {
                                    return Err(LustError::RuntimeError {
                                        message: format!(
                                            "Deopt payload ({source_label}) missing Option<{:?}> handle {} for local {}",
                                            inner, raw_val, loc_idx
                                        ),
                                    });
                                }
                            }
                        }
                    }
                    _ => {
                        return Err(LustError::RuntimeError {
                            message: format!(
                                "Unsupported Option type {:?} for deopt local {} in ({source_label})",
                                mir_ty, loc_idx
                            ),
                        });
                    }
                },
                MirType::Map(_, _) => match self.mir_context.get_handle(raw_val) {
                    Some(Value::Map(m_rc)) => Value::Map(Rc::clone(&m_rc)),
                    Some(other) => {
                        return Err(LustError::RuntimeError {
                            message: format!(
                                "Deopt payload ({source_label}) invalid handle {} for Map local {}: expected Map, got {:?}",
                                raw_val, loc_idx, other
                            ),
                        });
                    }
                    None => {
                        return Err(LustError::RuntimeError {
                            message: format!(
                                "Deopt payload ({source_label}) missing Map handle {} for local {}",
                                raw_val, loc_idx
                            ),
                        });
                    }
                },
                MirType::Enum(name) => {
                    if name == "Option" && raw_val == 0 {
                        Value::none()
                    } else {
                        match self.mir_context.get_handle(raw_val) {
                            Some(Value::Enum(e_rc)) => Value::Enum(Rc::clone(&e_rc)),
                            Some(other) => {
                                return Err(LustError::RuntimeError {
                                    message: format!(
                                        "Deopt payload ({source_label}) invalid handle {} for Enum('{}') local {}: expected Enum, got {:?}",
                                        raw_val, name, loc_idx, other
                                    ),
                                });
                            }
                            None => {
                                return Err(LustError::RuntimeError {
                                    message: format!(
                                        "Deopt payload ({source_label}) missing Enum('{}') handle {} for local {}",
                                        name, raw_val, loc_idx
                                    ),
                                });
                            }
                        }
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

        // Cold SROA struct materialization (`NG-W11-1`): reconstruct `Rc<StructObject>` from
        // restored scalar `field_locals` registers for every decomposed struct recipe so `orig_local`
        // holds a well-typed `Value::Struct` post-deopt.
        let mut sroa_materializations: Vec<(usize, Value)> = Vec::new();
        if !sroa_recipes.is_empty() {
            let mut restored_by_idx: hashbrown::HashMap<usize, Value> =
                hashbrown::HashMap::with_capacity(validated_restorations.len());
            for (loc_idx, val) in &validated_restorations {
                restored_by_idx.insert(*loc_idx, val.clone());
            }
            for sroa in sroa_recipes {
                if sroa.orig_local.0 >= effective_reg_capacity {
                    continue;
                }
                let Some(info) = self.struct_metadata.get(&sroa.struct_name) else {
                    continue;
                };
                let layout = &info.layout;
                if layout.field_names().len() != sroa.field_locals.len() {
                    continue;
                }
                let mut fields = Vec::with_capacity(sroa.field_locals.len());
                for (field_idx, field_local) in sroa.field_locals.iter().enumerate() {
                    let val = if let Some(restored) = restored_by_idx.get(&field_local.0) {
                        restored.clone()
                    } else {
                        match &layout.field_type(field_idx).kind {
                            TypeKind::Int => Value::Int(0),
                            TypeKind::Float => Value::Float(0.0),
                            TypeKind::Bool => Value::Bool(false),
                            TypeKind::String => Value::string(""),
                            _ => Value::Nil,
                        }
                    };
                    fields.push(val);
                }
                let struct_name = if layout.name().is_empty() {
                    sroa.struct_name.clone()
                } else {
                    layout.name().to_string()
                };
                let rc = StructObject::new(struct_name, Rc::clone(layout), fields);
                sroa_materializations.push((sroa.orig_local.0, Value::Struct(rc)));
            }
        }

        stage_and_commit_all_writebacks_with_readonly(
            &self.mir_context,
            marshalled_arrays,
            marshalled_structs,
            readonly_params,
            Some(self.struct_metadata.as_map()),
        )?;

        #[cfg(feature = "test-deopt")]
        {
            LAST_DEOPT_SROA_MATERIALIZATIONS.with(|slot| {
                *slot.borrow_mut() = sroa_materializations.clone();
            });
        }

        // Hazard 10: Expand top frame register window if `reg_count < required_locals`.
        if required_locals > top_frame_reg_count {
            self.ensure_top_frame_reg_capacity(required_locals)?;
        }

        // Commit lazily-materialized continuation bytecode & deopt recipes atomically
        // ONLY after all payload, writeback, and register-window checks have succeeded (`NG-W14-7`).
        if let Some((func_idx, continuation_bc)) = staged_continuation_bc
            && func_idx < self.functions.len()
        {
            let meta = crate::vm::CallMeta::of(&continuation_bc);
            let prev_fn = core::mem::replace(&mut self.functions[func_idx], continuation_bc);
            self.ast_bytecode_backup.entry(func_idx).or_insert(prev_fn);
            if func_idx < self.call_meta.len() {
                self.call_meta[func_idx] = meta;
            }
            if let Some(jit) = self.cranelift_jit.as_mut() {
                jit.mark_continuation_materialized(func_name);
            }
        }
        if !staged_recipes.is_empty() {
            for r in staged_recipes {
                self.cranelift_recipes
                    .insert((r.func_name.clone(), r.block_id, r.site_kind), r);
            }
            if let Some(jit) = self.cranelift_jit.as_mut() {
                jit.record_lazy_deopt_materialization();
            }
        }

        let top_frame = self.call_stack.last_mut().expect("validated non-empty");
        top_frame.ip = recipe.precise_pc;
        let base_reg = top_frame.base_reg;
        for (loc_idx, val) in validated_restorations {
            if !val.is_plain() {
                top_frame.mark_non_plain(loc_idx);
            }
            self.vm_registers[base_reg + loc_idx] = val;
        }
        for (orig_idx, struct_val) in sroa_materializations {
            if !struct_val.is_plain() {
                top_frame.mark_non_plain(orig_idx);
            }
            self.vm_registers[base_reg + orig_idx] = struct_val;
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
                self.sync_slow_path_flags();
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

#[cfg(all(test, feature = "std"))]
mod eligibility_walker_tests {
    use super::*;
    use crate::ast::{Span, Type, TypeKind};
    use crate::bytecode::value::FieldStorage;
    use crate::mir::ir::{
        FunctionSignature, MirBinaryOp, MirConstant, MirFunction, MirType, MirUnaryOp, Operand,
        Place, Rvalue, Statement, Terminator,
    };
    use alloc::rc::Rc;

    fn make_base_fn(name: &str) -> (MirFunction, crate::mir::ir::LocalId, crate::mir::ir::LocalId) {
        let span = Span::dummy();
        let mut f = MirFunction::new(
            FunctionSignature {
                name: name.to_string(),
                param_types: vec![MirType::Int],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            span,
        );
        let p0 = f.params[0];
        let dst = f.add_local(Some("dst".to_string()), MirType::Int, true, span);
        (f, p0, dst)
    }

    #[test]
    fn test_eligibility_walker_rejects_bad_field_projection_in_every_ir_variant() {
        let vm = VM::new();
        let span = Span::dummy();

        // 0. Baseline valid function MUST be eligible (proves assertions have teeth).
        {
            let (mut f, p0, _) = make_base_fn("valid_base");
            f.blocks[0].terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(p0))), span));
            assert!(vm.is_cranelift_eligible_in_vm(&f), "baseline scalar fn must be eligible");
        }

        // 1. Statement::Assign LHS bad Place::Field(Local(int), 0)
        {
            let (mut f, p0, _) = make_base_fn("bad_assign_lhs");
            let bad = Place::Field(Box::new(Place::Local(p0)), 0);
            f.blocks[0].statements.push(Statement::Assign(
                bad,
                Rvalue::Use(Operand::Constant(MirConstant::Int(1))),
                span,
            ));
            f.blocks[0].terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(p0))), span));
            assert!(!vm.is_cranelift_eligible_in_vm(&f), "Assign LHS bad field must be rejected");
        }

        // 2. Rvalue::Use bad field
        {
            let (mut f, p0, dst) = make_base_fn("bad_rvalue_use");
            let bad = Operand::Copy(Place::Field(Box::new(Place::Local(p0)), 0));
            f.blocks[0].statements.push(Statement::Assign(Place::Local(dst), Rvalue::Use(bad), span));
            f.blocks[0].terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(dst))), span));
            assert!(!vm.is_cranelift_eligible_in_vm(&f), "Rvalue::Use bad field must be rejected");
        }

        // 3. Rvalue::BinaryOp (lhs & rhs)
        for in_rhs in [false, true] {
            let (mut f, p0, dst) = make_base_fn("bad_rvalue_binop");
            let bad = Operand::Copy(Place::Field(Box::new(Place::Local(p0)), 0));
            let good = Operand::Constant(MirConstant::Int(1));
            let (l, r) = if in_rhs { (good, bad) } else { (bad, good) };
            f.blocks[0].statements.push(Statement::Assign(
                Place::Local(dst),
                Rvalue::BinaryOp(MirBinaryOp::Add, l, r),
                span,
            ));
            f.blocks[0].terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(dst))), span));
            assert!(!vm.is_cranelift_eligible_in_vm(&f), "BinaryOp bad field (rhs={in_rhs}) must be rejected");
        }

        // 4. Rvalue::UnaryOp
        {
            let (mut f, p0, dst) = make_base_fn("bad_rvalue_unop");
            let bad = Operand::Copy(Place::Field(Box::new(Place::Local(p0)), 0));
            f.blocks[0].statements.push(Statement::Assign(
                Place::Local(dst),
                Rvalue::UnaryOp(MirUnaryOp::Neg, bad),
                span,
            ));
            f.blocks[0].terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(dst))), span));
            assert!(!vm.is_cranelift_eligible_in_vm(&f), "UnaryOp bad field must be rejected");
        }

        // 5. Rvalue::Cast
        {
            let (mut f, p0, _) = make_base_fn("bad_rvalue_cast");
            let fdst = f.add_local(Some("fdst".to_string()), MirType::Float, true, span);
            let bad = Operand::Copy(Place::Field(Box::new(Place::Local(p0)), 0));
            f.blocks[0].statements.push(Statement::Assign(
                Place::Local(fdst),
                Rvalue::Cast(bad, MirType::Float),
                span,
            ));
            f.blocks[0].terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(p0))), span));
            assert!(!vm.is_cranelift_eligible_in_vm(&f), "Cast bad field must be rejected");
        }

        // 6. Rvalue::Aggregate
        {
            let (mut f, p0, _) = make_base_fn("bad_rvalue_agg");
            let arr = f.add_local(Some("arr".to_string()), MirType::Array(Box::new(MirType::Int)), true, span);
            let bad = Operand::Copy(Place::Field(Box::new(Place::Local(p0)), 0));
            f.blocks[0].statements.push(Statement::Assign(
                Place::Local(arr),
                Rvalue::Aggregate(MirType::Array(Box::new(MirType::Int)), vec![bad]),
                span,
            ));
            f.blocks[0].terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(p0))), span));
            assert!(!vm.is_cranelift_eligible_in_vm(&f), "Aggregate bad field must be rejected");
        }

        // 7. Terminator::Cond
        {
            let (mut f, p0, _) = make_base_fn("bad_term_cond");
            let b1 = f.new_block();
            let bad = Operand::Copy(Place::Field(Box::new(Place::Local(p0)), 0));
            f.blocks[0].terminator = Some(Terminator::Cond {
                cond: bad,
                then_block: b1,
                else_block: b1,
                span,
            });
            f.blocks[b1.0].terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(p0))), span));
            assert!(!vm.is_cranelift_eligible_in_vm(&f), "Cond bad field must be rejected");
        }

        // 8. Terminator::Switch
        {
            let (mut f, p0, _) = make_base_fn("bad_term_switch");
            let b1 = f.new_block();
            let bad = Operand::Copy(Place::Field(Box::new(Place::Local(p0)), 0));
            f.blocks[0].terminator = Some(Terminator::Switch {
                discr: bad,
                cases: vec![(0, b1)],
                default: b1,
                span,
            });
            f.blocks[b1.0].terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(p0))), span));
            assert!(!vm.is_cranelift_eligible_in_vm(&f), "Switch bad field must be rejected");
        }

        // 9. Terminator::Call (bad arg AND bad destination)
        for bad_dest in [false, true] {
            let (mut f, p0, dst) = make_base_fn("bad_term_call");
            let b1 = f.new_block();
            let bad_place = Place::Field(Box::new(Place::Local(p0)), 0);
            let (args, destination) = if bad_dest {
                (vec![Operand::Copy(Place::Local(p0))], bad_place)
            } else {
                (vec![Operand::Copy(bad_place)], Place::Local(dst))
            };
            f.blocks[0].terminator = Some(Terminator::Call {
                callee: "abs".to_string(),
                args,
                destination,
                success: b1,
                error: None,
                span,
            });
            f.blocks[b1.0].terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(dst))), span));
            assert!(
                !vm.is_cranelift_eligible_in_vm(&f),
                "Call bad field (bad_dest={bad_dest}) must be rejected"
            );
        }

        // 10. Terminator::Return
        {
            let (mut f, p0, _) = make_base_fn("bad_term_return");
            let bad = Operand::Copy(Place::Field(Box::new(Place::Local(p0)), 0));
            f.blocks[0].terminator = Some(Terminator::Return(Some(bad), span));
            assert!(!vm.is_cranelift_eligible_in_vm(&f), "Return bad field must be rejected");
        }

        // 11. Terminator::Raise (always rejected)
        {
            let (mut f, p0, _) = make_base_fn("bad_term_raise");
            f.blocks[0].terminator = Some(Terminator::Raise(Operand::Copy(Place::Local(p0)), span));
            assert!(!vm.is_cranelift_eligible_in_vm(&f), "Raise must be rejected");
        }
    }

    #[test]
    fn test_eligibility_walker_preserves_valid_option_result_struct_handles() {
        let mut vm = VM::new();
        let span = Span::dummy();
        let layout = crate::bytecode::value::StructLayout::new(
            "Envelope".to_string(),
            vec![
                Rc::new("id".to_string()),
                Rc::new("opt".to_string()),
                Rc::new("res".to_string()),
            ],
            vec![FieldStorage::Strong, FieldStorage::Strong, FieldStorage::Strong],
            vec![
                Type::new(TypeKind::Int, span),
                Type::new(TypeKind::Option(Box::new(Type::new(TypeKind::Int, span))), span),
                Type::new(
                    TypeKind::Result(
                        Box::new(Type::new(TypeKind::Int, span)),
                        Box::new(Type::new(TypeKind::String, span)),
                    ),
                    span,
                ),
            ],
            vec![None, None, None],
        );
        vm.struct_metadata.insert(
            "Envelope".to_string(),
            crate::vm::RuntimeStructInfo {
                layout: Rc::new(layout),
            },
        );
        assert!(
            vm.is_cranelift_supported_struct("Envelope"),
            "Struct with Int/Option/Result fields must remain supported"
        );

        let mut f = MirFunction::new(
            FunctionSignature {
                name: "read_env_id".to_string(),
                param_types: vec![MirType::Struct("Envelope".to_string())],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            span,
        );
        let p0 = f.params[0];
        let dst = f.add_local(Some("id".to_string()), MirType::Int, true, span);
        f.blocks[0].statements.push(Statement::Assign(
            Place::Local(dst),
            Rvalue::Use(Operand::Copy(Place::Field(Box::new(Place::Local(p0)), 0))),
            span,
        ));
        f.blocks[0].terminator = Some(Terminator::Return(Some(Operand::Copy(Place::Local(dst))), span));
        assert!(
            vm.is_cranelift_eligible_in_vm(&f),
            "Valid struct field projection on Option/Result-bearing struct must stay eligible"
        );

        // Out-of-bounds field index on the same valid struct MUST be rejected.
        let mut f_oob = f.clone();
        f_oob.signature.name = "read_env_oob".to_string();
        f_oob.blocks[0].statements[0] = Statement::Assign(
            Place::Local(dst),
            Rvalue::Use(Operand::Copy(Place::Field(Box::new(Place::Local(p0)), 9))),
            span,
        );
        assert!(
            !vm.is_cranelift_eligible_in_vm(&f_oob),
            "Out-of-bounds field projection on valid struct must be rejected"
        );
    }

    fn register_point_struct(vm: &mut VM) -> Rc<StructLayout> {
        let span = Span::dummy();
        let layout = Rc::new(StructLayout::new(
            "Point".to_string(),
            vec![Rc::new("x".to_string()), Rc::new("y".to_string())],
            vec![FieldStorage::Strong, FieldStorage::Strong],
            vec![Type::new(TypeKind::Int, span), Type::new(TypeKind::Int, span)],
            vec![None, None],
        ));
        vm.struct_metadata.insert(
            "Point".to_string(),
            crate::vm::RuntimeStructInfo {
                layout: layout.clone(),
            },
        );
        layout
    }

    #[test]
    fn test_perf_t2_1_unit_is_supported_deopt_recipe_type_struct_filtering() {
        let mut vm = VM::new();
        let span = Span::dummy();
        // Unregistered struct must be rejected
        assert!(!vm.is_supported_deopt_recipe_type(&MirType::Struct("Point".into())));
        assert!(!vm.is_supported_deopt_recipe_type(&MirType::Struct(String::new())));

        // Register valid Point { x: int, y: int }
        register_point_struct(&mut vm);
        assert!(vm.is_supported_deopt_recipe_type(&MirType::Struct("Point".into())));

        // Struct with unsupported nested Array field must be rejected
        let bad_layout = Rc::new(StructLayout::new(
            "BadStruct".to_string(),
            vec![Rc::new("items".to_string())],
            vec![FieldStorage::Strong],
            vec![Type::new(
                TypeKind::Array(Box::new(Type::new(TypeKind::Int, span))),
                span,
            )],
            vec![None],
        ));
        vm.struct_metadata.insert(
            "BadStruct".to_string(),
            crate::vm::RuntimeStructInfo { layout: bad_layout },
        );
        assert!(!vm.is_supported_deopt_recipe_type(&MirType::Struct("BadStruct".into())));
    }

    #[test]
    fn test_perf_t2_1_unit_restore_deopt_fresh_struct_alias_and_continuation_resume() {
        let mut vm = VM::new();
        let _layout = register_point_struct(&mut vm);
        let span = Span::dummy();

        // MIR function:
        //   bb0: a = Point { 1, 10 }; b = a; goto bb1
        //   bb1 (loop / resume target): a.x = a.x + 5; tmp = b.x; out = tmp + a.y; return out
        // After deopt at bb1, mutating a.x MUST be observed through b.x (shared Rc::ptr_eq)!
        let mut f = MirFunction::new(
            FunctionSignature {
                name: "alias_resume".to_string(),
                param_types: vec![],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            span,
        );
        let a = f.add_local(Some("a".into()), MirType::Struct("Point".into()), true, span);
        let b = f.add_local(Some("b".into()), MirType::Struct("Point".into()), true, span);
        let ax = f.add_local(Some("ax".into()), MirType::Int, true, span);
        let bx = f.add_local(Some("bx".into()), MirType::Int, true, span);
        let ay = f.add_local(Some("ay".into()), MirType::Int, true, span);
        let out = f.add_local(Some("out".into()), MirType::Int, true, span);
        let bb1 = f.new_block();

        f.blocks[0].statements.push(Statement::Assign(
            Place::Local(a),
            Rvalue::Aggregate(
                MirType::Struct("Point".into()),
                vec![
                    Operand::Constant(MirConstant::Int(1)),
                    Operand::Constant(MirConstant::Int(10)),
                ],
            ),
            span,
        ));
        f.blocks[0].statements.push(Statement::Assign(
            Place::Local(b),
            Rvalue::Use(Operand::Copy(Place::Local(a))),
            span,
        ));
        f.blocks[0].terminator = Some(Terminator::Goto(bb1, span));

        f.blocks[bb1.0].statements.push(Statement::Assign(
            Place::Local(ax),
            Rvalue::Use(Operand::Copy(Place::Field(Box::new(Place::Local(a)), 0))),
            span,
        ));
        f.blocks[bb1.0].statements.push(Statement::Assign(
            Place::Local(ax),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(ax)),
                Operand::Constant(MirConstant::Int(5)),
            ),
            span,
        ));
        f.blocks[bb1.0].statements.push(Statement::Assign(
            Place::Field(Box::new(Place::Local(a)), 0),
            Rvalue::Use(Operand::Copy(Place::Local(ax))),
            span,
        ));
        f.blocks[bb1.0].statements.push(Statement::Assign(
            Place::Local(bx),
            Rvalue::Use(Operand::Copy(Place::Field(Box::new(Place::Local(b)), 0))),
            span,
        ));
        f.blocks[bb1.0].statements.push(Statement::Assign(
            Place::Local(ay),
            Rvalue::Use(Operand::Copy(Place::Field(Box::new(Place::Local(a)), 1))),
            span,
        ));
        f.blocks[bb1.0].statements.push(Statement::Assign(
            Place::Local(out),
            Rvalue::BinaryOp(
                MirBinaryOp::Add,
                Operand::Copy(Place::Local(bx)),
                Operand::Copy(Place::Local(ay)),
            ),
            span,
        ));
        f.blocks[bb1.0].terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(out))),
            span,
        ));

        let mut bc = crate::mir::vm_backend::MirToBytecodeCompiler::compile(&f).expect("compile bc");
        let orig_len = bc.chunk.instructions.len();
        let orig_block_pcs = bc.block_entry_pcs.clone();
        vm.rewrite_continuation_struct_ops(&f, &mut bc)
            .expect("rewrite_continuation_struct_ops");
        assert_eq!(
            bc.chunk.instructions.len(),
            orig_len,
            "1-for-1 struct rewrite must preserve exact instruction count and PCs"
        );
        assert_eq!(bc.block_entry_pcs, orig_block_pcs);
        assert!(
            bc.chunk
                .instructions
                .iter()
                .any(|i| matches!(i, crate::bytecode::Instruction::NewStruct(..))),
            "expected NewStruct after continuation rewrite"
        );
        assert!(
            bc.chunk
                .instructions
                .iter()
                .any(|i| matches!(
                    i,
                    crate::bytecode::Instruction::GetField(..)
                        | crate::bytecode::Instruction::GetFieldSlot(..)
                ))
                && bc
                    .chunk
                    .instructions
                    .iter()
                    .any(|i| matches!(
                        i,
                        crate::bytecode::Instruction::SetField(..)
                            | crate::bytecode::Instruction::SetFieldSlot(..)
                    )),
            "expected GetField/GetFieldSlot and SetField/SetFieldSlot after continuation rewrite"
        );

        let resume_pc = bc.block_entry_pcs[&bb1.0];
        let func_idx = vm.functions.len();
        vm.functions.push(bc);
        vm.call_meta.push(crate::vm::CallMeta::of(&vm.functions[func_idx]));
        vm.cranelift_compiled_generation = 7;
        vm.cranelift_recipes.insert(
            ("alias_resume".to_string(), bb1.0, DeoptSiteKind::BlockEntry),
            CompiledDeoptRecipe {
                func_name: "alias_resume".to_string(),
                compilation_generation: 7,
                block_id: bb1.0,
                site_kind: DeoptSiteKind::BlockEntry,
                precise_pc: resume_pc,
                proven_locals: vec![
                    (a.0, MirType::Struct("Point".into())),
                    (b.0, MirType::Struct("Point".into())),
                ],
            },
        );

        let local_types: Vec<MirType> = f.locals.iter().map(|l| l.ty.clone()).collect();
        let frame_idx = vm.push_call_frame(func_idx, 256, None);
        vm.frame_registers_mut(frame_idx)[0] = Value::Int(-999);

        let _ctx_guard = vm.mir_context.enter();
        let fresh_ptr = lust_mir_alloc_struct_named(2, "Point".as_ptr(), "Point".len() as i64);
        assert!(!fresh_ptr.is_null());
        unsafe {
            *fresh_ptr.add(2) = 100; // x = 100
            *fresh_ptr.add(3) = 20; // y = 20
        }

        let payload = {
            let _inv = crate::mir::cranelift_backend::push_native_invocation_with_generation(
                "alias_resume".to_string(),
                1,
                1,
                7,
            );
            crate::mir::cranelift_backend::DeoptPayload::new(
                bb1.0,
                vec![(a.0, fresh_ptr as i64), (b.0, fresh_ptr as i64)],
            )
        };
        crate::mir::cranelift_backend::set_current_deopt_payload(payload);

        let outcome = vm
            .restore_deopt_top_frame("alias_resume", 1, 1, &local_types, &[], &[], &[], &[])
            .expect("restore_deopt_top_frame must succeed for aliased fresh structs");
        assert!(matches!(outcome, CraneliftOutcome::Fallback));

        // Verify exact Rc::ptr_eq identity between restored locals `a` and `b`!
        let (rc_a, rc_b) = {
            let top = vm.call_stack.last().unwrap();
            assert_eq!(top.ip, resume_pc);
            let regs = vm.top_frame_registers();
            match (&regs[a.0], &regs[b.0]) {
                (Value::Struct(ra), Value::Struct(rb)) => (Rc::clone(ra), Rc::clone(rb)),
                other => panic!("expected (Struct, Struct), got {other:?}"),
            }
        };
        assert!(
            Rc::ptr_eq(&rc_a, &rc_b),
            "two live locals aliasing the same fresh struct MUST share Rc::ptr_eq identity post-deopt"
        );

        // Resume interpreter execution on rewritten continuation bytecode:
        // a.x becomes 100 + 5 = 105; b.x observes 105; out = 105 + 20 = 125!
        drop(_ctx_guard);
        let resumed_val = vm.run_inner::<false>().expect("interpreter continuation resume");
        assert_eq!(resumed_val, Value::Int(125));
    }

    #[test]
    fn test_perf_t2_1_unit_restore_deopt_rejects_corrupted_ptr_and_field_count_mismatch() {
        let mut vm = VM::new();
        let layout = register_point_struct(&mut vm);

        let local_types = vec![MirType::Struct("Point".into())];
        vm.cranelift_compiled_generation = 3;
        vm.cranelift_recipes.insert(
            ("neg_struct_deopt".to_string(), 1, DeoptSiteKind::BlockEntry),
            CompiledDeoptRecipe {
                func_name: "neg_struct_deopt".to_string(),
                compilation_generation: 3,
                block_id: 1,
                site_kind: DeoptSiteKind::BlockEntry,
                precise_pc: 42,
                proven_locals: vec![(0, MirType::Struct("Point".into()))],
            },
        );

        let frame_idx = vm.push_call_frame(0, 256, None);
        vm.call_stack[frame_idx].ip = 7;
        vm.frame_registers_mut(frame_idx)[0] = Value::Int(12345);

        let caller_struct = StructObject::new(
            "Point".to_string(),
            layout,
            vec![Value::Int(10), Value::Int(20)],
        );

        let _ctx_guard = vm.mir_context.enter();
        let input_ptr = lust_mir_alloc_struct_named(2, "Point".as_ptr(), "Point".len() as i64);
        assert!(!input_ptr.is_null());
        unsafe {
            *input_ptr.add(2) = 999; // staged mutation that MUST NOT commit on error!
            *input_ptr.add(3) = 888;
        }
        let marshalled_structs = vec![MarshalledStruct {
            rc: caller_struct.clone(),
            ptr: input_ptr,
            field_count: 2,
            param_indices: vec![0],
        }];

        // Case 1: Corrupted / unowned pointers (null & 0xdeadbeef)
        for bad_ptr in [0_i64, 0x0000_dead_beef_i64] {
            let payload = {
                let _inv = crate::mir::cranelift_backend::push_native_invocation_with_generation(
                    "neg_struct_deopt".to_string(),
                    1,
                    1,
                    3,
                );
                crate::mir::cranelift_backend::DeoptPayload::new(1, vec![(0, bad_ptr)])
            };
            crate::mir::cranelift_backend::set_current_deopt_payload(payload);
            let err = vm
                .restore_deopt_top_frame(
                    "neg_struct_deopt",
                    1,
                    1,
                    &local_types,
                    &[],
                    &marshalled_structs,
                    &[],
                    &[],
                )
                .expect_err("corrupted/unowned struct pointer must fail closed");
            assert!(
                err.to_string().contains("provenance") || err.to_string().contains("null pointer"),
                "unexpected error for {bad_ptr:#x}: {err}"
            );
            // Zero frame mutation AND zero input-struct writeback mutation!
            assert_eq!(vm.call_stack.last().unwrap().ip, 7);
            assert_eq!(vm.top_frame_registers()[0], Value::Int(12345));
            assert_eq!(*caller_struct.fields.borrow(), vec![Value::Int(10), Value::Int(20)]);
        }

        // Case 2: Owned `Point` allocation with mismatched field count (len=1 != expected 2)
        let mismatch_ptr = lust_mir_alloc_struct_named(1, "Point".as_ptr(), "Point".len() as i64);
        assert!(!mismatch_ptr.is_null());
        let payload = {
            let _inv = crate::mir::cranelift_backend::push_native_invocation_with_generation(
                "neg_struct_deopt".to_string(),
                1,
                2,
                3,
            );
            crate::mir::cranelift_backend::DeoptPayload::new(1, vec![(0, mismatch_ptr as i64)])
        };
        crate::mir::cranelift_backend::set_current_deopt_payload(payload);
        let err = vm
            .restore_deopt_top_frame(
                "neg_struct_deopt",
                1,
                2,
                &local_types,
                &[],
                &marshalled_structs,
                &[],
                &[],
            )
            .expect_err("struct field count mismatch must fail closed");
        assert!(
            err.to_string().contains("field count") || err.to_string().contains("Malformed struct"),
            "unexpected error: {err}"
        );
        // Zero frame mutation AND zero input-struct writeback mutation!
        assert_eq!(vm.call_stack.last().unwrap().ip, 7);
        assert_eq!(vm.top_frame_registers()[0], Value::Int(12345));
        assert_eq!(*caller_struct.fields.borrow(), vec![Value::Int(10), Value::Int(20)]);
    }
}
