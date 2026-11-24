#[cfg(feature = "std")]
pub use jit_engine::{
    analyze_module_call_effects, clear_registered_struct_layouts, compute_default_module_roots,
    compute_reachable_native_functions, find_loop_backedges, find_loop_headers,
    get_registered_struct_layout, get_registered_struct_layout_by_name, ic_dispatch,
    ic_invalidate, ic_invalidate_all, ic_probe, ic_register_site, ic_reset, ic_site_stats,
    ic_update, is_loop_backedge, lust_mir_abs_int, lust_mir_alloc_array,
    lust_mir_alloc_array_typed, lust_mir_alloc_cflat_struct, lust_mir_alloc_enum,
    lust_mir_alloc_enum2, lust_mir_alloc_map, lust_mir_alloc_struct, lust_mir_alloc_struct_named,
    lust_mir_alloc_tuple, lust_mir_alloc_tuple_typed, lust_mir_array_get, lust_mir_array_len,
    lust_mir_array_len_checked, lust_mir_array_push, lust_mir_array_set, MirArrayHeader,
    mir_array_elem_tag, MIR_ARRAY_ELEM_TAG_BOOL, MIR_ARRAY_ELEM_TAG_FLOAT,
    MIR_ARRAY_ELEM_TAG_INT, MIR_ARRAY_ELEM_TAG_STRING,
    lust_mir_charge_gas, lust_mir_clamp_int, lust_mir_div_int, lust_mir_enum_payload,
    lust_mir_enum_payload_at, lust_mir_enum_tag, lust_mir_floordiv_float, lust_mir_floordiv_int,
    lust_mir_gas_fast_ptr, lust_mir_gas_slow, lust_mir_get_gas,
    lust_mir_get_memory_used, lust_mir_get_status, lust_mir_ic_dispatch, lust_mir_ic_miss,
    lust_mir_ic_probe, lust_mir_init_deopt, lust_mir_init_deopt_at_terminator, lust_mir_map_get,
    lust_mir_map_get_int_unwrap, lust_mir_map_get_typed, lust_mir_map_has_typed, lust_mir_map_len,
    lust_mir_map_set, lust_mir_map_set_typed, lust_mir_max_int, lust_mir_min_int,
    lust_mir_pow_float, lust_mir_pow_int, lust_mir_print_bool_part, lust_mir_print_float_part,
    lust_mir_print_int, lust_mir_print_int_part, lust_mir_print_nl, lust_mir_print_sep,
    lust_mir_print_string_part, lust_mir_queue_new, lust_mir_queue_offer_int,
    lust_mir_queue_take_int, lust_mir_record_deopt,
    lust_mir_record_deopt_local,
    lust_mir_record_native_entry, lust_mir_rem_float, lust_mir_rem_int, lust_mir_safepoint,
    lust_mir_safepoint_slow, lust_mir_safepoint_state_ptr,
    lust_mir_set_gas, lust_mir_set_max_memory, lust_mir_set_memory_limit, lust_mir_set_status,
    lust_mir_string_append_inplace, lust_mir_string_byte, lust_mir_string_byte_at,
    lust_mir_string_concat, lust_mir_string_eq, lust_mir_string_find, lust_mir_string_gsub,
    lust_mir_string_len, lust_mir_string_split, lust_mir_string_sub, lust_mir_tostring_bool,
    lust_mir_tostring_float, lust_mir_tostring_int, lust_mir_unknown_is_int,
    lust_mir_unknown_unwrap_int,
    lust_mir_struct_decode_binary, lust_mir_struct_decode_contiguous, publish_deopt_payload,
    publish_deopt_payload_with_func, publish_deopt_payload_name_only,
    publish_deopt_payload_at_terminator, publish_deopt_payload_with_site_kind,
    refine_locals, refine_locals_with_params, refine_locals_with_context, refine_mir_function, refine_mir_function_with_layouts, apply_module_callee_return_types, register_struct_layout, reset_native_instrumentation_counters, safepoint_enter_mask,
    safepoint_exit_mask, safepoint_register_finalizer, safepoint_request_deopt,
    safepoint_request_interrupt, safepoint_reset, safepoint_set_gas_budget_active,
    safepoint_stats, take_deopt_payload,
    set_current_deopt_payload, with_current_deopt_payload_mut,
    take_current_sticky_error, get_current_sticky_error, set_current_sticky_error,
    take_last_schema_decode_error, set_current_native_function, get_current_native_function,
    push_native_invocation, push_native_invocation_with_generation, get_current_native_invocation, native_invocation_depth,
    is_vm_effect_intrinsic_callee, is_effect_or_handler_callee,
    lust_mir_enter_native_callee, lust_mir_exit_native_callee,
    NativeInvocationIdentity, NativeInvocationGuard, DeoptSiteKind,
    CraneliftJitCompiler, DeoptPayload, GasFastState, IcEntry, IcSiteStats,
    IcStateKind, InlineCacheSite, MirCraneliftBackendError, OsrPromotionTracker,
    SafepointFastState, SafepointRuntimeState, SafepointStatsSnapshot, SchemaDecodeError,
    SLOW_FLAG_DEOPT, SLOW_FLAG_FINALIZERS, SLOW_FLAG_GAS_BUDGET, SLOW_FLAG_INTERRUPT,
    SLOW_FLAG_MASKED, UNBOUNDED_GAS_CHUNK, UnboundedGasChunkGuard,
    reset_unbounded_gas_replenish_count, set_unbounded_gas_chunk_override,
    unbounded_gas_chunk, unbounded_gas_replenish_count,
    CRANELIFT_NATIVE_ENTRY_COUNT, CRANELIFT_SIMD_SCALAR_COUNT, CRANELIFT_SIMD_VECTOR_COUNT,
    DIRECT_DECODE_COUNT, DYNAMIC_LOOKUP_COUNT, PROMOTED_LOOP_CFG_VERIFICATIONS,
    promoted_loop_cfg_verifications, reset_promoted_loop_cfg_verifications,
    lust_mir_is_simd_scalar_forced, lust_mir_record_simd_scalar_op,
    lust_mir_record_simd_vector_op, MIR_STATUS_DEOPT, MIR_STATUS_ERR_DIV_BY_ZERO,
    MIR_STATUS_ERR_INTERRUPTED, MIR_STATUS_ERR_NEG_EXPONENT, MIR_STATUS_ERR_OOM,
    MIR_STATUS_ERR_OUT_OF_BOUNDS, MIR_STATUS_ERR_OUT_OF_GAS, MIR_STATUS_OK,
};

#[cfg(all(feature = "std", feature = "test-deopt"))]
pub use jit_engine::{
    refine_locals_with_seam, refine_mir_function_with_seam, safepoint_request_deopt_on_poll,
    SLOW_FLAG_TEST_DEOPT,
};

#[cfg(feature = "std")]
pub mod jit_engine {
    use crate::mir::ir::*;
    pub use crate::mir::runtime_helpers::*;
    use crate::CanonicalTypeTag;
    use alloc::{format, string::String, vec::Vec};
    use cranelift_codegen::Context;
    use cranelift_codegen::ir::{
        condcodes::{FloatCC, IntCC},
        types, AbiParam, Block, BlockArg, FuncRef, InstBuilder, MemFlagsData, Signature, StackSlot,
        StackSlotData, StackSlotKind, Value as ClifValue,
    };
    use cranelift_codegen::settings::{self, Configurable};
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::{FuncId, Linkage, Module};

    #[derive(Debug, Clone, PartialEq, thiserror::Error)]
    pub enum MirCraneliftBackendError {
        #[error("Binary operator '{0:?}' is not supported by Cranelift backend")]
        UnsupportedBinaryOp(MirBinaryOp),
        #[error("Constant '{0:?}' is not supported by Cranelift backend")]
        UnsupportedConstant(MirConstant),
        #[error("MIR type '{0:?}' is not supported by Cranelift backend")]
        UnsupportedType(MirType),
        #[error("Cranelift codegen error: {0}")]
        CodegenError(String),
        #[error("Failed or unlowerable value cannot be replaced with silent Unit: {0}")]
        UnlowerableValue(String),
        #[error("Invalid SIMD call '{callee}': {reason}")]
        InvalidSimdCall { callee: String, reason: String },
    }

    pub const MIR_STATUS_ERR_INTERRUPTED: i64 = crate::effect_abi::LUST_STATUS_ERR_INTERRUPTED as i64;
    pub const MIR_STATUS_ERR_GENERIC: i64 = crate::effect_abi::LUST_STATUS_ERR_GENERIC as i64;
    pub const MIR_STATUS_DEOPT: i64 = -100;

    pub const SLOW_FLAG_DEOPT: u64 = 1 << 0;
    pub const SLOW_FLAG_INTERRUPT: u64 = 1 << 1;
    pub const SLOW_FLAG_MASKED: u64 = 1 << 2;
    pub const SLOW_FLAG_FINALIZERS: u64 = 1 << 3;
    pub const SLOW_FLAG_GAS_BUDGET: u64 = 1 << 4;
    #[cfg(feature = "test-deopt")]
    pub const SLOW_FLAG_TEST_DEOPT: u64 = 1 << 5;

    #[repr(C)]
    pub struct SafepointFastState {
        /// Non-zero bitmask iff ANY slow-path condition is armed on this thread.
        pub slow_flags: u64,
        /// Single authoritative per-thread counter for `SafepointStatsSnapshot::safepoints_checked`.
        pub safepoints_checked: u64,
        /// Result word written by `CallConv::PreserveAll` cold safepoint trampoline (`NG-W16-3`).
        pub slow_result: i64,
    }

    #[derive(Default)]
    pub struct SafepointRuntimeState {
        pub mask_depth: usize,
        pub pending_interrupt: Option<crate::effect_abi::InterruptReason>,
        pub safepoints_masked_deferred: u64,
        pub safepoints_delivered: u64,
        pub deopt_requested: bool,
        pub gas_budget_active: bool,
        #[cfg(feature = "test-deopt")]
        pub deopt_at_check_ordinal: Option<u64>,
        #[cfg(feature = "test-deopt")]
        pub deopt_target_func: Option<String>,
        pub cleanups_run: usize,
        pub finalizers: Vec<alloc::boxed::Box<dyn FnOnce() + Send + 'static>>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct NativeInvocationIdentity {
        pub func_name: String,
        pub frame_depth: usize,
        pub invocation_id: u64,
        pub compilation_generation: u64,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum DeoptSiteKind {
        BlockEntry,
        PreTerminator,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum DeoptConversionError {
        MissingTypeMetadata { local_idx: usize },
        DuplicateLocal { local_idx: usize },
        DuplicateTypeMetadata { local_idx: usize },
        UnsupportedType { local_idx: usize, ty: MirType },
        InvalidReferenceHandle { local_idx: usize, handle: i64, ty: MirType },
    }

    impl core::fmt::Display for DeoptConversionError {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            match self {
                Self::MissingTypeMetadata { local_idx } => {
                    write!(f, "Missing type metadata for local index {}", local_idx)
                }
                Self::DuplicateLocal { local_idx } => {
                    write!(f, "Duplicate local index {} in deopt payload", local_idx)
                }
                Self::DuplicateTypeMetadata { local_idx } => {
                    write!(f, "Duplicate type metadata for local index {}", local_idx)
                }
                Self::UnsupportedType { local_idx, ty } => {
                    write!(f, "Unsupported type {:?} for local index {}", ty, local_idx)
                }
                Self::InvalidReferenceHandle { local_idx, handle, ty } => {
                    write!(
                        f,
                        "Invalid or absent reference handle {} for local index {} of type {:?}",
                        handle, local_idx, ty
                    )
                }
            }
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct DeoptPayload {
        pub target_pc: usize,
        pub live_locals: Vec<(usize, i64)>,
        pub func_name: Option<String>,
        pub frame_identity: Option<NativeInvocationIdentity>,
        pub site_kind: DeoptSiteKind,
    }

    impl DeoptPayload {
        pub fn new(target_pc: usize, live_locals: Vec<(usize, i64)>) -> Self {
            let ident = get_current_native_invocation();
            let fn_name = ident.as_ref().map(|i| i.func_name.clone());
            Self {
                target_pc,
                live_locals,
                func_name: fn_name,
                frame_identity: ident,
                site_kind: DeoptSiteKind::BlockEntry,
            }
        }

        pub fn at_block_entry(target_pc: usize, live_locals: Vec<(usize, i64)>) -> Self {
            Self::new(target_pc, live_locals)
        }

        pub fn at_pre_terminator(target_pc: usize, live_locals: Vec<(usize, i64)>) -> Self {
            let mut p = Self::new(target_pc, live_locals);
            p.site_kind = DeoptSiteKind::PreTerminator;
            p
        }

        pub fn with_site_kind(mut self, site_kind: DeoptSiteKind) -> Self {
            self.site_kind = site_kind;
            self
        }

        pub fn with_func(
            target_pc: usize,
            live_locals: Vec<(usize, i64)>,
            func_name: Option<String>,
        ) -> Self {
            let ident = func_name.as_ref().map(|f| NativeInvocationIdentity {
                func_name: f.clone(),
                frame_depth: 1,
                invocation_id: 1,
                compilation_generation: 0,
            });
            Self {
                target_pc,
                live_locals,
                func_name,
                frame_identity: ident,
                site_kind: DeoptSiteKind::BlockEntry,
            }
        }

        pub fn with_identity(
            target_pc: usize,
            live_locals: Vec<(usize, i64)>,
            frame_identity: Option<NativeInvocationIdentity>,
        ) -> Self {
            let fn_name = frame_identity.as_ref().map(|i| i.func_name.clone());
            Self {
                target_pc,
                live_locals,
                func_name: fn_name,
                frame_identity,
                site_kind: DeoptSiteKind::BlockEntry,
            }
        }

        pub fn try_to_vm_values_with_types(
            &self,
            types: &[(usize, MirType)],
        ) -> Result<Vec<(usize, crate::bytecode::Value)>, DeoptConversionError> {
            let mut seen_types = hashbrown::HashSet::new();
            for &(idx, _) in types {
                if !seen_types.insert(idx) {
                    return Err(DeoptConversionError::DuplicateTypeMetadata { local_idx: idx });
                }
            }

            let mut res = Vec::with_capacity(self.live_locals.len());
            let mut seen_locals = hashbrown::HashSet::new();

            for &(loc_idx, raw_val) in &self.live_locals {
                if !seen_locals.insert(loc_idx) {
                    return Err(DeoptConversionError::DuplicateLocal { local_idx: loc_idx });
                }

                let opt_ty = types
                    .iter()
                    .find(|(i, _)| *i == loc_idx)
                    .map(|(_, ty)| ty)
                    .ok_or(DeoptConversionError::MissingTypeMetadata { local_idx: loc_idx })?;

                let val = match opt_ty {
                    MirType::Int => crate::bytecode::Value::Int(raw_val),
                    MirType::Float => crate::bytecode::Value::Float(f64::from_bits(raw_val as u64)),
                    MirType::Bool => crate::bytecode::Value::Bool(raw_val != 0),
                    MirType::Unit => crate::bytecode::Value::Nil,
                    MirType::String
                    | MirType::Array(_)
                    | MirType::Tuple(_)
                    | MirType::Map(_, _)
                    | MirType::Struct(_)
                    | MirType::Enum(_) => {
                        let handle_val = with_current_mir_context(|ctx| ctx.get_handle(raw_val));
                        match handle_val {
                            Some(v) => v,
                            None => {
                                return Err(DeoptConversionError::InvalidReferenceHandle {
                                    local_idx: loc_idx,
                                    handle: raw_val,
                                    ty: opt_ty.clone(),
                                });
                            }
                        }
                    }
                    MirType::Option(_) => {
                        if raw_val == 0 {
                            crate::bytecode::Value::none()
                        } else {
                            let handle_val = with_current_mir_context(|ctx| ctx.get_handle(raw_val));
                            match handle_val {
                                Some(v) => v,
                                None => {
                                    return Err(DeoptConversionError::InvalidReferenceHandle {
                                        local_idx: loc_idx,
                                        handle: raw_val,
                                        ty: opt_ty.clone(),
                                    });
                                }
                            }
                        }
                    }
                    other => {
                        return Err(DeoptConversionError::UnsupportedType {
                            local_idx: loc_idx,
                            ty: other.clone(),
                        });
                    }
                };
                res.push((loc_idx, val));
            }
            Ok(res)
        }

        pub fn try_to_vm_values(
            &self,
            mir_fn: &MirFunction,
        ) -> Result<Vec<(usize, crate::bytecode::Value)>, DeoptConversionError> {
            let types: Vec<(usize, MirType)> = mir_fn
                .locals
                .iter()
                .enumerate()
                .map(|(i, l)| (i, l.ty.clone()))
                .collect();
            self.try_to_vm_values_with_types(&types)
        }

        pub fn to_vm_values_with_types(
            &self,
            types: &[(usize, MirType)],
        ) -> Vec<(usize, crate::bytecode::Value)> {
            let mut res = Vec::with_capacity(self.live_locals.len());
            for &(loc_idx, raw_val) in &self.live_locals {
                let opt_ty = types
                    .iter()
                    .find(|(i, _)| *i == loc_idx)
                    .map(|(_, ty)| ty);
                let val = match opt_ty {
                    Some(MirType::Int) => crate::bytecode::Value::Int(raw_val),
                    Some(MirType::Float) => crate::bytecode::Value::Float(f64::from_bits(raw_val as u64)),
                    Some(MirType::Bool) => crate::bytecode::Value::Bool(raw_val != 0),
                    Some(MirType::Unit) => crate::bytecode::Value::Nil,
                    Some(MirType::String)
                    | Some(MirType::Array(_))
                    | Some(MirType::Tuple(_))
                    | Some(MirType::Map(_, _))
                    | Some(MirType::Struct(_))
                    | Some(MirType::Enum(_)) => {
                        let handle_val = with_current_mir_context(|ctx| ctx.get_handle(raw_val));
                        handle_val.unwrap_or(crate::bytecode::Value::Nil)
                    }
                    Some(MirType::Option(_)) => {
                        if raw_val == 0 {
                            crate::bytecode::Value::none()
                        } else {
                            let handle_val = with_current_mir_context(|ctx| ctx.get_handle(raw_val));
                            handle_val.unwrap_or(crate::bytecode::Value::Nil)
                        }
                    }
                    _ => crate::bytecode::Value::Nil,
                };
                res.push((loc_idx, val));
            }
            res
        }

        pub fn to_vm_values(&self, mir_fn: &MirFunction) -> Vec<(usize, crate::bytecode::Value)> {
            let types: Vec<(usize, MirType)> = mir_fn
                .locals
                .iter()
                .enumerate()
                .map(|(i, l)| (i, l.ty.clone()))
                .collect();
            self.to_vm_values_with_types(&types)
        }
    }

    pub fn publish_deopt_payload(target_pc: usize, live_locals: Vec<(usize, i64)>) {
        publish_deopt_payload_with_func(target_pc, live_locals, None);
    }

    pub fn publish_deopt_payload_at_terminator(target_pc: usize, live_locals: Vec<(usize, i64)>) {
        publish_deopt_payload_with_site_kind(target_pc, live_locals, None, DeoptSiteKind::PreTerminator);
    }

    pub fn publish_deopt_payload_with_func(
        target_pc: usize,
        live_locals: Vec<(usize, i64)>,
        func_name: Option<String>,
    ) {
        publish_deopt_payload_with_site_kind(target_pc, live_locals, func_name, DeoptSiteKind::BlockEntry);
    }

    pub fn publish_deopt_payload_with_site_kind(
        target_pc: usize,
        live_locals: Vec<(usize, i64)>,
        func_name: Option<String>,
        site_kind: DeoptSiteKind,
    ) {
        let cur_ident = get_current_native_invocation();
        let ident = if let Some(cur) = cur_ident {
            Some(cur)
        } else {
            func_name.as_ref().map(|f| NativeInvocationIdentity {
                func_name: f.clone(),
                frame_depth: 1,
                invocation_id: 1,
                compilation_generation: 0,
            })
        };
        let fn_name = ident.as_ref().map(|i| i.func_name.clone()).or(func_name);
        set_current_deopt_payload(DeoptPayload {
            target_pc,
            live_locals,
            func_name: fn_name,
            frame_identity: ident,
            site_kind,
        });
    }

    pub fn publish_deopt_payload_name_only(
        target_pc: usize,
        live_locals: Vec<(usize, i64)>,
        func_name: Option<String>,
    ) {
        set_current_deopt_payload(DeoptPayload {
            target_pc,
            live_locals,
            func_name,
            frame_identity: None,
            site_kind: DeoptSiteKind::BlockEntry,
        });
    }

    // =========================================================================
    // J-IC: Inline Cache runtime definitions & registry
    // =========================================================================

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum IcStateKind {
        Uninitialized,
        Monomorphic,
        Polymorphic,
        MegamorphicFallback,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct IcEntry {
        pub type_tag: u64,
        pub target_val: u64,
    }

    pub const IC_POLYMORPHIC_CAPACITY: usize = 4;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct InlineCacheSite {
        pub site_id: u32,
        pub state: IcStateKind,
        pub entries: [Option<IcEntry>; IC_POLYMORPHIC_CAPACITY],
        pub entry_count: usize,
        pub hits: u64,
        pub misses: u64,
        pub invalidations: u64,
        pub fallbacks: u64,
    }

    impl InlineCacheSite {
        pub fn new(site_id: u32) -> Self {
            Self {
                site_id,
                state: IcStateKind::Uninitialized,
                entries: [None; IC_POLYMORPHIC_CAPACITY],
                entry_count: 0,
                hits: 0,
                misses: 0,
                invalidations: 0,
                fallbacks: 0,
            }
        }

        pub fn probe(&mut self, type_tag: u64) -> Option<u64> {
            for entry in self.entries.iter().flatten() {
                if entry.type_tag == type_tag {
                    self.hits += 1;
                    return Some(entry.target_val);
                }
            }
            self.misses += 1;
            None
        }

        pub fn update(&mut self, type_tag: u64, target_val: u64) -> u64 {
            for entry in self.entries.iter_mut().flatten() {
                if entry.type_tag == type_tag {
                    self.hits += 1;
                    return entry.target_val;
                }
            }
            match self.state {
                IcStateKind::Uninitialized => {
                    self.entries[0] = Some(IcEntry { type_tag, target_val });
                    self.entry_count = 1;
                    self.state = IcStateKind::Monomorphic;
                }
                IcStateKind::Monomorphic => {
                    self.entries[1] = Some(IcEntry { type_tag, target_val });
                    self.entry_count = 2;
                    self.state = IcStateKind::Polymorphic;
                }
                IcStateKind::Polymorphic => {
                    if self.entry_count < IC_POLYMORPHIC_CAPACITY {
                        self.entries[self.entry_count] = Some(IcEntry { type_tag, target_val });
                        self.entry_count += 1;
                    } else {
                        self.state = IcStateKind::MegamorphicFallback;
                        self.fallbacks += 1;
                    }
                }
                IcStateKind::MegamorphicFallback => {
                    self.fallbacks += 1;
                }
            }
            target_val
        }

        pub fn invalidate(&mut self) {
            self.entries = [None; IC_POLYMORPHIC_CAPACITY];
            self.entry_count = 0;
            self.state = IcStateKind::Uninitialized;
            self.invalidations += 1;
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct IcSiteStats {
        pub site_id: u32,
        pub state: IcStateKind,
        pub entry_count: usize,
        pub hits: u64,
        pub misses: u64,
        pub invalidations: u64,
        pub fallbacks: u64,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct NativeInvocationEntry {
        pub identity: NativeInvocationIdentity,
        pub deopt_payload: Option<DeoptPayload>,
        pub sticky_error: Option<String>,
        pub native_callee_depth: usize,
        #[cfg(feature = "test-deopt")]
        pub safepoint_polls: u64,
    }

    thread_local! {
        static IC_REGISTRY: core::cell::RefCell<hashbrown::HashMap<u32, InlineCacheSite>> = core::cell::RefCell::new(hashbrown::HashMap::new());
        static SAFEPOINT_STATE: core::cell::RefCell<SafepointRuntimeState> = core::cell::RefCell::new(SafepointRuntimeState::default());
        static SAFEPOINT_FAST_STATE: core::cell::UnsafeCell<SafepointFastState> =
            const { core::cell::UnsafeCell::new(SafepointFastState { slow_flags: 0, safepoints_checked: 0, slow_result: 0 }) };
        static STANDALONE_DEOPT_PAYLOAD: core::cell::RefCell<Option<DeoptPayload>> = const { core::cell::RefCell::new(None) };
        static STANDALONE_STICKY_ERROR: core::cell::RefCell<Option<String>> = const { core::cell::RefCell::new(None) };
        static CURRENT_NATIVE_INVOCATION_STACK: core::cell::RefCell<Vec<NativeInvocationEntry>> = const { core::cell::RefCell::new(Vec::new()) };
        static STRUCT_LAYOUT_REGISTRY: core::cell::RefCell<hashbrown::HashMap<u64, MirStructLayout>> = core::cell::RefCell::new(hashbrown::HashMap::new());
        static STRUCT_NAME_MAP: core::cell::RefCell<hashbrown::HashMap<String, u64>> = core::cell::RefCell::new(hashbrown::HashMap::new());
    }

    #[inline(always)]
    fn sync_safepoint_fast_flags(st: &SafepointRuntimeState) {
        let mut flags: u64 = 0;
        if st.deopt_requested {
            flags |= SLOW_FLAG_DEOPT;
        }
        if st.pending_interrupt.is_some() {
            flags |= SLOW_FLAG_INTERRUPT;
        }
        if st.mask_depth > 0 {
            flags |= SLOW_FLAG_MASKED;
        }
        if !st.finalizers.is_empty() {
            flags |= SLOW_FLAG_FINALIZERS;
        }
        if st.gas_budget_active {
            flags |= SLOW_FLAG_GAS_BUDGET;
        }
        #[cfg(feature = "test-deopt")]
        if st.deopt_at_check_ordinal.is_some() {
            flags |= SLOW_FLAG_TEST_DEOPT;
        }
        SAFEPOINT_FAST_STATE.with(|cell| unsafe {
            (*cell.get()).slow_flags = flags;
        });
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn lust_mir_safepoint_state_ptr() -> *mut SafepointFastState {
        SAFEPOINT_FAST_STATE.with(|c| c.get())
    }

    pub struct NativeInvocationGuard;

    impl Drop for NativeInvocationGuard {
        fn drop(&mut self) {
            // Invocation-scoped cleanup: popping the native invocation stack automatically
            // drops invocation state. If popping the root invocation and it has an unconsumed
            // deopt payload or sticky error, preserve it into standalone storage for external
            // consumers (such as resume_from_deopt or take_deopt_payload).
            CURRENT_NATIVE_INVOCATION_STACK.with(|stack| {
                let mut s = stack.borrow_mut();
                if let Some(popped) = s.pop() {
                    #[cfg(feature = "test-deopt")]
                    {
                        let popped_fn = popped.identity.func_name.as_str();
                        SAFEPOINT_STATE.with(|state_cell| {
                            let mut st = state_cell.borrow_mut();
                            if st.deopt_at_check_ordinal.is_some()
                                && st
                                    .deopt_target_func
                                    .as_deref()
                                    .is_none_or(|target| target == popped_fn)
                            {
                                st.deopt_at_check_ordinal = None;
                                st.deopt_target_func = None;
                                sync_safepoint_fast_flags(&st);
                            }
                        });
                    }
                    if s.is_empty() {
                        if let Some(payload) = popped.deopt_payload {
                            STANDALONE_DEOPT_PAYLOAD.with(|sp| {
                                let mut st = sp.borrow_mut();
                                if st.is_none() {
                                    *st = Some(payload);
                                }
                            });
                        }
                        if let Some(err) = popped.sticky_error {
                            STANDALONE_STICKY_ERROR.with(|se| {
                                let mut st = se.borrow_mut();
                                if st.is_none() {
                                    *st = Some(err);
                                }
                            });
                        }
                    }
                }
            });
        }
    }

    pub fn push_native_invocation(
        func_name: String,
        frame_depth: usize,
        invocation_id: u64,
    ) -> NativeInvocationGuard {
        push_native_invocation_with_generation(func_name, frame_depth, invocation_id, 0)
    }

    pub fn push_native_invocation_with_generation(
        func_name: String,
        frame_depth: usize,
        invocation_id: u64,
        compilation_generation: u64,
    ) -> NativeInvocationGuard {
        CURRENT_NATIVE_INVOCATION_STACK.with(|stack| {
            let mut s = stack.borrow_mut();
            s.push(NativeInvocationEntry {
                identity: NativeInvocationIdentity {
                    func_name,
                    frame_depth,
                    invocation_id,
                    compilation_generation,
                },
                deopt_payload: None,
                sticky_error: None,
                native_callee_depth: 0,
                #[cfg(feature = "test-deopt")]
                safepoint_polls: 0,
            });
        });
        NativeInvocationGuard
    }

    pub fn get_current_native_invocation() -> Option<NativeInvocationIdentity> {
        CURRENT_NATIVE_INVOCATION_STACK.with(|stack| stack.borrow().last().map(|e| e.identity.clone()))
    }

    pub fn get_current_native_function() -> Option<String> {
        get_current_native_invocation().map(|i| i.func_name)
    }

    pub fn native_invocation_depth() -> usize {
        CURRENT_NATIVE_INVOCATION_STACK.with(|stack| stack.borrow().len())
    }

    pub fn set_current_native_function(name: Option<String>) {
        CURRENT_NATIVE_INVOCATION_STACK.with(|stack| {
            let mut s = stack.borrow_mut();
            if let Some(n) = name {
                s.push(NativeInvocationEntry {
                    identity: NativeInvocationIdentity {
                        func_name: n,
                        frame_depth: 1,
                        invocation_id: 1,
                        compilation_generation: 0,
                    },
                    deopt_payload: None,
                    sticky_error: None,
                    native_callee_depth: 0,
                    #[cfg(feature = "test-deopt")]
                    safepoint_polls: 0,
                });
            } else {
                s.clear();
            }
        });
    }

    pub fn set_current_deopt_payload(payload: DeoptPayload) {
        CURRENT_NATIVE_INVOCATION_STACK.with(|stack| {
            let mut s = stack.borrow_mut();
            if let Some(top) = s.last_mut() {
                top.deopt_payload = Some(payload);
            } else {
                STANDALONE_DEOPT_PAYLOAD.with(|sp| {
                    *sp.borrow_mut() = Some(payload);
                });
            }
        });
    }

    pub fn with_current_deopt_payload_mut<F, R>(f: F) -> Option<R>
    where
        F: FnOnce(&mut DeoptPayload) -> R,
    {
        CURRENT_NATIVE_INVOCATION_STACK.with(|stack| {
            let mut s = stack.borrow_mut();
            if let Some(top) = s.last_mut() {
                top.deopt_payload.as_mut().map(f)
            } else {
                STANDALONE_DEOPT_PAYLOAD.with(|sp| {
                    sp.borrow_mut().as_mut().map(f)
                })
            }
        })
    }

    pub fn ic_register_site(site_id: u32) {
        IC_REGISTRY.with(|reg| {
            reg.borrow_mut().entry(site_id).or_insert_with(|| InlineCacheSite::new(site_id));
        });
    }

    pub fn ic_probe(site_id: u32, type_tag: u64) -> Option<u64> {
        IC_REGISTRY.with(|reg| {
            let mut map = reg.borrow_mut();
            let site = map.entry(site_id).or_insert_with(|| InlineCacheSite::new(site_id));
            site.probe(type_tag)
        })
    }

    pub fn ic_update(site_id: u32, type_tag: u64, target_val: u64) -> u64 {
        IC_REGISTRY.with(|reg| {
            let mut map = reg.borrow_mut();
            let site = map.entry(site_id).or_insert_with(|| InlineCacheSite::new(site_id));
            site.update(type_tag, target_val)
        })
    }

    pub fn ic_dispatch(site_id: u32, type_tag: u64, default_target: u64) -> u64 {
        IC_REGISTRY.with(|reg| {
            let mut map = reg.borrow_mut();
            let site = map.entry(site_id).or_insert_with(|| InlineCacheSite::new(site_id));
            if let Some(target) = site.probe(type_tag) {
                target
            } else {
                site.update(type_tag, default_target)
            }
        })
    }

    pub fn ic_invalidate(site_id: u32) {
        IC_REGISTRY.with(|reg| {
            if let Some(site) = reg.borrow_mut().get_mut(&site_id) {
                site.invalidate();
            }
        });
    }

    pub fn ic_invalidate_all() {
        IC_REGISTRY.with(|reg| {
            for site in reg.borrow_mut().values_mut() {
                site.invalidate();
            }
        });
    }

    pub fn ic_site_stats(site_id: u32) -> Option<IcSiteStats> {
        IC_REGISTRY.with(|reg| {
            reg.borrow().get(&site_id).map(|s| IcSiteStats {
                site_id: s.site_id,
                state: s.state,
                entry_count: s.entry_count,
                hits: s.hits,
                misses: s.misses,
                invalidations: s.invalidations,
                fallbacks: s.fallbacks,
            })
        })
    }

    pub fn ic_reset() {
        IC_REGISTRY.with(|reg| {
            reg.borrow_mut().clear();
        });
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn lust_mir_ic_probe(site_id: u64, type_tag: u64) -> u64 {
        ic_probe(site_id as u32, type_tag).unwrap_or(0)
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn lust_mir_ic_miss(site_id: u64, type_tag: u64, target_val: u64) -> u64 {
        ic_update(site_id as u32, type_tag, target_val)
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn lust_mir_ic_dispatch(site_id: u64, type_tag: u64, default_target: u64) -> u64 {
        ic_dispatch(site_id as u32, type_tag, default_target)
    }

    // =========================================================================
    // Gate-SCHEMA-09/13: Struct layout registry & contiguous decoder
    // =========================================================================

    pub fn register_struct_layout(layout: MirStructLayout) {
        STRUCT_NAME_MAP.with(|m| {
            m.borrow_mut().insert(layout.type_name.clone(), layout.type_id);
        });
        STRUCT_LAYOUT_REGISTRY.with(|reg| {
            reg.borrow_mut().insert(layout.type_id, layout);
        });
    }

    pub fn get_registered_struct_layout(type_id: u64) -> Option<MirStructLayout> {
        STRUCT_LAYOUT_REGISTRY.with(|reg| reg.borrow().get(&type_id).cloned())
    }

    pub fn get_registered_struct_layout_by_name(name: &str) -> Option<MirStructLayout> {
        let type_id = STRUCT_NAME_MAP.with(|m| m.borrow().get(name).copied())?;
        get_registered_struct_layout(type_id)
    }

    pub fn clear_registered_struct_layouts() {
        STRUCT_LAYOUT_REGISTRY.with(|r| r.borrow_mut().clear());
        STRUCT_NAME_MAP.with(|m| m.borrow_mut().clear());
    }

    pub static CRANELIFT_NATIVE_ENTRY_COUNT: core::sync::atomic::AtomicUsize =
        core::sync::atomic::AtomicUsize::new(0);
    pub static CRANELIFT_SIMD_VECTOR_COUNT: core::sync::atomic::AtomicUsize =
        core::sync::atomic::AtomicUsize::new(0);
    pub static CRANELIFT_SIMD_SCALAR_COUNT: core::sync::atomic::AtomicUsize =
        core::sync::atomic::AtomicUsize::new(0);
    pub static DIRECT_DECODE_COUNT: core::sync::atomic::AtomicUsize =
        core::sync::atomic::AtomicUsize::new(0);
    pub static DYNAMIC_LOOKUP_COUNT: core::sync::atomic::AtomicUsize =
        core::sync::atomic::AtomicUsize::new(0);

    pub fn reset_native_instrumentation_counters() {
        CRANELIFT_NATIVE_ENTRY_COUNT.store(0, core::sync::atomic::Ordering::SeqCst);
        CRANELIFT_SIMD_VECTOR_COUNT.store(0, core::sync::atomic::Ordering::SeqCst);
        CRANELIFT_SIMD_SCALAR_COUNT.store(0, core::sync::atomic::Ordering::SeqCst);
        DIRECT_DECODE_COUNT.store(0, core::sync::atomic::Ordering::SeqCst);
        DYNAMIC_LOOKUP_COUNT.store(0, core::sync::atomic::Ordering::SeqCst);
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn lust_mir_record_native_entry() {
        CRANELIFT_NATIVE_ENTRY_COUNT.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn lust_mir_record_simd_vector_op() {
        CRANELIFT_SIMD_VECTOR_COUNT.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        crate::mir::opt::simd::SIMD_VECTOR_OP_COUNT.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn lust_mir_record_simd_scalar_op() {
        CRANELIFT_SIMD_SCALAR_COUNT.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        crate::mir::opt::simd::SIMD_SCALAR_FALLBACK_COUNT.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn lust_mir_is_simd_scalar_forced() -> i64 {
        i64::from(crate::mir::opt::simd::is_force_simd_scalar_fallback())
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum SchemaDecodeError {
        TruncatedPayload { expected: usize, actual: usize },
        NullPointer,
        UnknownLayout(u64),
        InvalidField(String),
    }

    std::thread_local! {
        static LAST_SCHEMA_DECODE_ERROR: core::cell::RefCell<Option<SchemaDecodeError>> =
            const { core::cell::RefCell::new(None) };
    }

    pub fn take_last_schema_decode_error() -> Option<SchemaDecodeError> {
        LAST_SCHEMA_DECODE_ERROR.with(|e| e.borrow_mut().take())
    }

    fn set_last_schema_decode_error(err: SchemaDecodeError) {
        LAST_SCHEMA_DECODE_ERROR.with(|e| {
            *e.borrow_mut() = Some(err);
        });
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn lust_mir_struct_decode_binary(
        layout_id: u64,
        ptr: *const u8,
        len: usize,
    ) -> i64 {
        let Some(layout) = get_registered_struct_layout(layout_id) else {
            set_last_schema_decode_error(SchemaDecodeError::UnknownLayout(layout_id));
            with_current_mir_context(|ctx| {
                ctx.last_status.store(
                    crate::effect_abi::LUST_STATUS_ERR_GENERIC as i64,
                    core::sync::atomic::Ordering::SeqCst,
                );
            });
            return 0;
        };

        let expected = layout.size_bytes;
        if len < expected {
            set_last_schema_decode_error(SchemaDecodeError::TruncatedPayload {
                expected,
                actual: len,
            });
            with_current_mir_context(|ctx| {
                ctx.last_status
                    .store(-1, core::sync::atomic::Ordering::SeqCst);
            });
            return 0;
        }

        if ptr.is_null() {
            set_last_schema_decode_error(SchemaDecodeError::NullPointer);
            with_current_mir_context(|ctx| {
                ctx.last_status
                    .store(-1, core::sync::atomic::Ordering::SeqCst);
            });
            return 0;
        }

        DIRECT_DECODE_COUNT.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        let count_i64 = (expected + 7) / 8;
        let count = count_i64.max(1);
        let mut boxed = alloc::vec![0i64; count].into_boxed_slice();
        let out_ptr = boxed.as_mut_ptr();
        unsafe {
            core::ptr::copy_nonoverlapping(ptr, out_ptr as *mut u8, expected);
        }
        let raw_box = alloc::boxed::Box::into_raw(boxed);
        with_current_mir_context(|ctx| {
            ctx.push_owned_buffer(
                raw_box as *mut i64,
                count,
                MirAllocationKind::CFlatStruct("BinaryDecode".into()),
            );
        });
        out_ptr as i64
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn lust_mir_struct_decode_contiguous(layout_id: u64, input_handle: i64) -> i64 {
        let input_bytes = with_current_mir_context(|ctx| {
            ctx.get_handle(input_handle).and_then(|val| val.as_string().map(|s| s.as_bytes().to_vec()))
        });
        let Some(bytes) = input_bytes else {
            with_current_mir_context(|ctx| {
                ctx.last_status.store(crate::effect_abi::LUST_STATUS_ERR_GENERIC as i64, core::sync::atomic::Ordering::SeqCst);
            });
            return 0;
        };

        let layout_opt = get_registered_struct_layout(layout_id);
        let Some(layout) = layout_opt else {
            with_current_mir_context(|ctx| {
                ctx.last_status.store(crate::effect_abi::LUST_STATUS_ERR_GENERIC as i64, core::sync::atomic::Ordering::SeqCst);
            });
            return 0;
        };

        let logical = layout.to_logical_layout();
        let mut counters = crate::data::materialize::LayoutPerfCounters::default();
        let opts = crate::data::materialize::ContiguousDecodeOptions {
            cancel_check: None,
            nested_layouts: None,
            require_nominal_tag: false,
        };

        match crate::data::materialize::decode_json_into_contiguous(&logical, &bytes, &mut counters, &opts) {
            Ok(contiguous) => {
                DIRECT_DECODE_COUNT.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
                let count_i64 = (contiguous.raw_bytes().len() + 7) / 8;
                let count = count_i64.max(1);
                let mut boxed = alloc::vec![0i64; count].into_boxed_slice();
                let ptr = boxed.as_mut_ptr();
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        contiguous.raw_bytes().as_ptr(),
                        ptr as *mut u8,
                        contiguous.raw_bytes().len(),
                    );
                }
                let raw_box = alloc::boxed::Box::into_raw(boxed);
                with_current_mir_context(|ctx| {
                    ctx.push_owned_buffer(
                        raw_box as *mut i64,
                        count,
                        MirAllocationKind::CFlatStruct(layout.type_name.clone()),
                    );
                });
                ptr as i64
            }
            Err(_e) => {
                with_current_mir_context(|ctx| {
                    ctx.last_status.store(-1, core::sync::atomic::Ordering::SeqCst);
                });
                0
            }
        }
    }

    const NESTED_NATIVE_DEOPT_ERR: &str =
        "Nested native-to-native deopt without multi-frame reconstruction is disabled";

    #[inline(always)]
    fn is_nested_native_callee() -> bool {
        CURRENT_NATIVE_INVOCATION_STACK.with(|stack| {
            stack
                .borrow()
                .last()
                .is_some_and(|e| e.native_callee_depth > 0)
        })
    }

    #[inline(always)]
    fn fail_nested_native_deopt() -> i64 {
        set_current_sticky_error(NESTED_NATIVE_DEOPT_ERR.into());
        with_current_mir_context(|ctx| {
            ctx.last_status
                .store(MIR_STATUS_ERR_GENERIC, core::sync::atomic::Ordering::SeqCst);
        });
        MIR_STATUS_ERR_GENERIC
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn lust_mir_enter_native_callee() {
        CURRENT_NATIVE_INVOCATION_STACK.with(|stack| {
            if let Some(top) = stack.borrow_mut().last_mut() {
                top.native_callee_depth = top.native_callee_depth.saturating_add(1);
            }
        });
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn lust_mir_exit_native_callee() {
        CURRENT_NATIVE_INVOCATION_STACK.with(|stack| {
            if let Some(top) = stack.borrow_mut().last_mut() {
                top.native_callee_depth = top.native_callee_depth.saturating_sub(1);
            }
        });
    }

    #[inline(always)]
    pub fn is_vm_effect_intrinsic_callee(callee: &str) -> bool {
        matches!(
            callee,
            "__lust_handle"
                | "__lust_perform"
                | "perform"
                | "__lust_resume"
                | "resume"
                | "__lust_register_finalizer"
                | "register_finalizer"
        )
    }

    #[inline(always)]
    pub fn is_effect_or_handler_callee(callee: &str) -> bool {
        is_vm_effect_intrinsic_callee(callee) || callee == "effect.perform"
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn lust_mir_safepoint_slow() -> i64 {
        #[cfg(feature = "test-deopt")]
        let fast_checked = SAFEPOINT_FAST_STATE.with(|c| unsafe { (*c.get()).safepoints_checked });

        enum SlowOutcome {
            Continue,
            Deopt,
            NestedDeoptErr,
            DeliverInterrupt(Vec<alloc::boxed::Box<dyn FnOnce() + Send + 'static>>),
        }

        let outcome = SAFEPOINT_STATE.with(|state_cell| {
            let mut st = state_cell.borrow_mut();

            #[cfg(feature = "test-deopt")]
            if let Some(target_poll) = st.deopt_at_check_ordinal {
                let target_func = st.deopt_target_func.as_deref();
                let (matches_fn, current_poll) = CURRENT_NATIVE_INVOCATION_STACK.with(|stack| {
                    let mut s = stack.borrow_mut();
                    if let Some(top) = s.last_mut() {
                        let fn_ok = target_func.is_none_or(|t| t == top.identity.func_name.as_str());
                        if fn_ok {
                            top.safepoint_polls += 1;
                        }
                        (fn_ok, top.safepoint_polls)
                    } else {
                        (target_func.is_none(), fast_checked)
                    }
                });
                if matches_fn && current_poll == target_poll {
                    st.deopt_at_check_ordinal = None;
                    st.deopt_target_func = None;
                    sync_safepoint_fast_flags(&st);
                    if is_nested_native_callee() {
                        return SlowOutcome::NestedDeoptErr;
                    }
                    return SlowOutcome::Deopt;
                }
            }

            if st.deopt_requested {
                if is_nested_native_callee() {
                    st.deopt_requested = false;
                    sync_safepoint_fast_flags(&st);
                    return SlowOutcome::NestedDeoptErr;
                }
                return SlowOutcome::Deopt;
            }

            if st.mask_depth > 0 {
                if st.pending_interrupt.is_some() {
                    st.safepoints_masked_deferred += 1;
                }
                return SlowOutcome::Continue;
            }

            if let Some(_reason) = st.pending_interrupt.take() {
                st.safepoints_delivered += 1;
                let finalizers = core::mem::take(&mut st.finalizers);
                sync_safepoint_fast_flags(&st);
                return SlowOutcome::DeliverInterrupt(finalizers);
            }

            SlowOutcome::Continue
        });

        match outcome {
            SlowOutcome::Continue => 0,
            SlowOutcome::NestedDeoptErr => fail_nested_native_deopt(),
            SlowOutcome::Deopt => {
                with_current_mir_context(|ctx| {
                    ctx.last_status
                        .store(MIR_STATUS_DEOPT, core::sync::atomic::Ordering::SeqCst);
                });
                MIR_STATUS_DEOPT
            }
            SlowOutcome::DeliverInterrupt(finalizers) => {
                // SAFEPOINT_STATE borrow is dropped before invoking finalizers so reentrant
                // calls to safepoint_stats() or safepoint_* mutators never RefCell-panic.
                for f in finalizers {
                    f();
                    SAFEPOINT_STATE.with(|state_cell| {
                        let mut st = state_cell.borrow_mut();
                        st.cleanups_run += 1;
                        sync_safepoint_fast_flags(&st);
                    });
                }
                SAFEPOINT_STATE.with(|state_cell| {
                    let st = state_cell.borrow();
                    sync_safepoint_fast_flags(&st);
                });
                with_current_mir_context(|ctx| {
                    ctx.last_status
                        .store(MIR_STATUS_ERR_INTERRUPTED, core::sync::atomic::Ordering::SeqCst);
                });
                MIR_STATUS_ERR_INTERRUPTED
            }
        }
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn lust_mir_safepoint() -> i64 {
        SAFEPOINT_FAST_STATE.with(|cell| unsafe {
            (*cell.get()).safepoints_checked += 1;
        });
        lust_mir_safepoint_slow()
    }

    /// Safe cold quantized refill helper for multi-header inline-queue loops (`NG-W16-8`):
    /// preserves a 100% reducible CFG (eliminating cross-loop multi-header SSA merges)
    /// while enforcing bit-exact safepoint/deopt/SEC-01 gas semantics.
    /// Returns:
    ///   - `>= 0`: armed `quantum_rem` (`0` or `max_q`)
    ///   - `MIR_STATUS_DEOPT` (`-100`): deopt requested at this safepoint
    ///   - `-1`: non-deopt error (e.g. `OutOfGas` / `Interrupted`; status already stored)
    pub extern "C" fn lust_mir_quantized_refill_slow(amt: i64, max_q: i64) -> i64 {
        let mut allow_arm = true;
        let sp_slow_needed = SAFEPOINT_STATE.with(|state_cell| {
            let st = state_cell.borrow();
            let base = st.deopt_requested
                || st.pending_interrupt.is_some()
                || st.mask_depth > 0
                || !st.finalizers.is_empty()
                || st.gas_budget_active;
            #[cfg(feature = "test-deopt")]
            {
                base || st.deopt_at_check_ordinal.is_some()
            }
            #[cfg(not(feature = "test-deopt"))]
            {
                base
            }
        });
        if sp_slow_needed {
            let sp_res = lust_mir_safepoint_slow();
            if sp_res != 0 {
                return if sp_res == MIR_STATUS_DEOPT {
                    MIR_STATUS_DEOPT
                } else {
                    -1
                };
            }
            allow_arm = false;
        }
        let neg_gas = with_current_mir_context(|ctx| {
            let next = ctx.gas_remaining().saturating_sub(amt);
            ctx.set_gas(next);
            next < 0
        });
        if neg_gas {
            let gas_res = lust_mir_gas_slow();
            if gas_res != 0 {
                return -1;
            }
            return 0;
        }
        if allow_arm && max_q > 0 {
            let threshold = max_q.saturating_mul(amt);
            let armed = with_current_mir_context(|ctx| {
                let rem = ctx.gas_remaining();
                if rem >= threshold {
                    ctx.set_gas(rem - threshold);
                    true
                } else {
                    false
                }
            });
            if armed {
                return max_q;
            }
        }
        0
    }

    fn init_deopt_impl(target_pc: i64, site_kind: DeoptSiteKind, label: &str) {
        if is_nested_native_callee() {
            fail_nested_native_deopt();
            return;
        }
        if target_pc < 0 {
            set_current_sticky_error(format!("Invalid negative target_pc {} in {}", target_pc, label));
            return;
        }
        let catch_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let ident = get_current_native_invocation();
            let fn_name = ident.as_ref().map(|i| i.func_name.clone());
            set_current_deopt_payload(DeoptPayload {
                target_pc: target_pc as usize,
                live_locals: Vec::new(),
                func_name: fn_name,
                frame_identity: ident,
                site_kind,
            });
        }));
        if catch_res.is_err() {
            set_current_sticky_error(format!("Panic in {}", label));
        }
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn lust_mir_init_deopt(target_pc: i64) {
        init_deopt_impl(target_pc, DeoptSiteKind::BlockEntry, "lust_mir_init_deopt");
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn lust_mir_init_deopt_at_terminator(target_pc: i64) {
        init_deopt_impl(
            target_pc,
            DeoptSiteKind::PreTerminator,
            "lust_mir_init_deopt_at_terminator",
        );
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn lust_mir_record_deopt_local(target_pc: i64, local_idx: i64, val: i64) {
        if target_pc < 0 {
            set_current_sticky_error(format!("Invalid negative target_pc {} in lust_mir_record_deopt_local", target_pc));
            return;
        }
        if local_idx < 0 {
            set_current_sticky_error(format!("Invalid negative local_idx {} in lust_mir_record_deopt_local", local_idx));
            return;
        }
        let catch_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let target_pc = target_pc as usize;
            let local_idx = local_idx as usize;
            let mut err = None;
            CURRENT_NATIVE_INVOCATION_STACK.with(|stack| {
                let mut s = stack.borrow_mut();
                if let Some(top) = s.last_mut() {
                    if let Some(payload) = top.deopt_payload.as_mut() {
                        if payload.target_pc != target_pc {
                            err = Some(format!(
                                "Deopt local target_pc mismatch: payload initialized for {}, recorded for {}",
                                payload.target_pc, target_pc
                            ));
                            return;
                        }
                        if payload.frame_identity.as_ref() != Some(&top.identity) {
                            err = Some("Deopt local frame identity mismatch".into());
                            return;
                        }
                        if payload.live_locals.iter().any(|(idx, _)| *idx == local_idx) {
                            err = Some(format!("Duplicate local_idx {} in deopt payload", local_idx));
                            return;
                        }
                        payload.live_locals.push((local_idx, val));
                    } else {
                        err = Some("lust_mir_record_deopt_local called before lust_mir_init_deopt".into());
                    }
                } else {
                    STANDALONE_DEOPT_PAYLOAD.with(|sp| {
                        let mut opt = sp.borrow_mut();
                        if let Some(payload) = opt.as_mut() {
                            if payload.target_pc != target_pc {
                                err = Some(format!(
                                    "Standalone deopt local target_pc mismatch: payload initialized for {}, recorded for {}",
                                    payload.target_pc, target_pc
                                ));
                                return;
                            }
                            if payload.live_locals.iter().any(|(idx, _)| *idx == local_idx) {
                                err = Some(format!("Duplicate local_idx {} in standalone deopt payload", local_idx));
                                return;
                            }
                            payload.live_locals.push((local_idx, val));
                        } else {
                            err = Some("lust_mir_record_deopt_local called before lust_mir_init_deopt (standalone)".into());
                        }
                    });
                }
            });
            if let Some(e) = err {
                set_current_sticky_error(e);
            }
        }));
        if catch_res.is_err() {
            set_current_sticky_error("Panic in lust_mir_record_deopt_local".into());
        }
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn lust_mir_record_deopt(target_pc: i64, num_locals: i64, locals_ptr: *const i64) -> i64 {
        if is_nested_native_callee() {
            return fail_nested_native_deopt();
        }
        if target_pc < 0 {
            return MIR_STATUS_ERR_GENERIC;
        }
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut locals = Vec::new();
            if !locals_ptr.is_null() && num_locals > 0 {
                for i in 0..num_locals as usize {
                    let val = unsafe { *locals_ptr.add(i) };
                    locals.push((i, val));
                }
            }
            let ident = get_current_native_invocation();
            let fn_name = ident.as_ref().map(|i| i.func_name.clone());
            set_current_deopt_payload(DeoptPayload {
                target_pc: target_pc as usize,
                live_locals: locals,
                func_name: fn_name,
                frame_identity: ident,
                site_kind: DeoptSiteKind::BlockEntry,
            });
            MIR_STATUS_DEOPT
        }));
        res.unwrap_or(MIR_STATUS_ERR_GENERIC)
    }

    pub fn safepoint_enter_mask() {
        SAFEPOINT_STATE.with(|st| {
            let mut state = st.borrow_mut();
            state.mask_depth += 1;
            sync_safepoint_fast_flags(&state);
        });
    }

    pub fn safepoint_exit_mask() -> Option<crate::effect_abi::InterruptReason> {
        SAFEPOINT_STATE.with(|st| {
            let mut state = st.borrow_mut();
            state.mask_depth = state.mask_depth.saturating_sub(1);
            sync_safepoint_fast_flags(&state);
            if state.mask_depth == 0 {
                state.pending_interrupt
            } else {
                None
            }
        })
    }

    pub fn safepoint_request_interrupt(reason: crate::effect_abi::InterruptReason) {
        SAFEPOINT_STATE.with(|st| {
            let mut state = st.borrow_mut();
            if state.pending_interrupt.is_none() {
                state.pending_interrupt = Some(reason);
            }
            sync_safepoint_fast_flags(&state);
        });
    }

    pub fn safepoint_register_finalizer<F: FnOnce() + Send + 'static>(f: F) {
        SAFEPOINT_STATE.with(|st| {
            let mut state = st.borrow_mut();
            state.finalizers.push(alloc::boxed::Box::new(f));
            sync_safepoint_fast_flags(&state);
        });
    }

    pub fn safepoint_request_deopt(deopt: bool) {
        SAFEPOINT_STATE.with(|st| {
            let mut state = st.borrow_mut();
            state.deopt_requested = deopt;
            sync_safepoint_fast_flags(&state);
        });
    }

    pub fn safepoint_set_gas_budget_active(active: bool) {
        SAFEPOINT_STATE.with(|st| {
            let mut state = st.borrow_mut();
            state.gas_budget_active = active;
            sync_safepoint_fast_flags(&state);
        });
    }

    #[cfg(feature = "test-deopt")]
    pub fn safepoint_request_deopt_on_poll(func_name: Option<String>, poll_ordinal: u64) {
        SAFEPOINT_STATE.with(|st| {
            let mut state = st.borrow_mut();
            state.deopt_target_func = func_name;
            state.deopt_at_check_ordinal = Some(poll_ordinal);
            sync_safepoint_fast_flags(&state);
        });
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct SafepointStatsSnapshot {
        pub mask_depth: usize,
        pub safepoints_checked: u64,
        pub safepoints_masked_deferred: u64,
        pub safepoints_delivered: u64,
        pub cleanups_run: usize,
    }

    pub fn safepoint_stats() -> SafepointStatsSnapshot {
        let checked = SAFEPOINT_FAST_STATE.with(|c| unsafe { (*c.get()).safepoints_checked });
        SAFEPOINT_STATE.with(|st| {
            let s = st.borrow();
            SafepointStatsSnapshot {
                mask_depth: s.mask_depth,
                safepoints_checked: checked,
                safepoints_masked_deferred: s.safepoints_masked_deferred,
                safepoints_delivered: s.safepoints_delivered,
                cleanups_run: s.cleanups_run,
            }
        })
    }

    pub fn safepoint_reset() {
        SAFEPOINT_STATE.with(|st| {
            let mut state = st.borrow_mut();
            *state = SafepointRuntimeState::default();
            SAFEPOINT_FAST_STATE.with(|cell| unsafe {
                (*cell.get()).safepoints_checked = 0;
            });
            sync_safepoint_fast_flags(&state);
        });
        CURRENT_NATIVE_INVOCATION_STACK.with(|stack| {
            let mut s = stack.borrow_mut();
            if let Some(top) = s.last_mut() {
                top.deopt_payload = None;
                top.sticky_error = None;
            }
        });
        STANDALONE_DEOPT_PAYLOAD.with(|sp| {
            *sp.borrow_mut() = None;
        });
        STANDALONE_STICKY_ERROR.with(|sp| {
            *sp.borrow_mut() = None;
        });
    }

    pub fn take_deopt_payload() -> Option<DeoptPayload> {
        CURRENT_NATIVE_INVOCATION_STACK.with(|stack| {
            let mut s = stack.borrow_mut();
            if let Some(top) = s.last_mut() {
                return top.deopt_payload.take();
            }
            STANDALONE_DEOPT_PAYLOAD.with(|sp| sp.borrow_mut().take())
        })
    }

    pub fn take_current_sticky_error() -> Option<String> {
        CURRENT_NATIVE_INVOCATION_STACK.with(|stack| {
            let mut s = stack.borrow_mut();
            if let Some(top) = s.last_mut() {
                return top.sticky_error.take();
            }
            STANDALONE_STICKY_ERROR.with(|sp| sp.borrow_mut().take())
        })
    }

    pub fn get_current_sticky_error() -> Option<String> {
        CURRENT_NATIVE_INVOCATION_STACK.with(|stack| {
            let s = stack.borrow();
            if let Some(top) = s.last() {
                return top.sticky_error.clone();
            }
            STANDALONE_STICKY_ERROR.with(|sp| sp.borrow().clone())
        })
    }

    pub fn set_current_sticky_error(err: String) {
        with_current_mir_context(|ctx| {
            ctx.last_status.store(MIR_STATUS_ERR_GENERIC, core::sync::atomic::Ordering::SeqCst);
        });
        CURRENT_NATIVE_INVOCATION_STACK.with(|stack| {
            let mut s = stack.borrow_mut();
            if let Some(top) = s.last_mut() {
                if top.sticky_error.is_none() {
                    top.sticky_error = Some(err);
                }
                return;
            }
            STANDALONE_STICKY_ERROR.with(|sp| {
                let mut st = sp.borrow_mut();
                if st.is_none() {
                    *st = Some(err);
                }
            });
        });
    }

    /// Materializes a handle-backed `Value::Array` (returned by `lust_mir_string_gsub`,
    /// `lust_mir_string_split`, `lust_mir_string_byte`, `lust_mir_string_find`) into a
    /// Cranelift `*mut MirArrayHeader` pointer so `array.len` and `arr[idx]` execute natively.
    fn materialize_handle_array_to_mir_ptr(handle: i64, elem_tag: i64) -> i64 {
        if handle == 0 {
            return 0;
        }
        let arr_val = with_current_mir_context(|ctx| ctx.get_handle(handle));
        let Some(crate::bytecode::Value::Array(arr)) = arr_val else {
            return 0;
        };
        let elems: Vec<i64> = with_current_mir_context(|ctx| {
            arr.borrow()
                .iter()
                .map(|v| match elem_tag {
                    MIR_ARRAY_ELEM_TAG_INT => match v {
                        crate::bytecode::Value::Int(n) => *n,
                        crate::bytecode::Value::Bool(b) => i64::from(*b),
                        crate::bytecode::Value::Float(f) => *f as i64,
                        _ => 0,
                    },
                    MIR_ARRAY_ELEM_TAG_BOOL => match v {
                        crate::bytecode::Value::Bool(b) => i64::from(*b),
                        crate::bytecode::Value::Int(n) => i64::from(*n != 0),
                        _ => 0,
                    },
                    MIR_ARRAY_ELEM_TAG_FLOAT => match v {
                        crate::bytecode::Value::Float(f) => f.to_bits() as i64,
                        crate::bytecode::Value::Int(n) => (*n as f64).to_bits() as i64,
                        _ => 0,
                    },
                    MIR_ARRAY_ELEM_TAG_STRING => match v {
                        crate::bytecode::Value::String(_) => ctx.register_handle(v.clone()),
                        crate::bytecode::Value::Int(n) => {
                            ctx.register_handle(crate::bytecode::Value::string(alloc::format!("{n}")))
                        }
                        crate::bytecode::Value::Bool(b) => ctx.register_handle(
                            crate::bytecode::Value::string(if *b { "true" } else { "false" }),
                        ),
                        crate::bytecode::Value::Float(f) => {
                            ctx.register_handle(crate::bytecode::Value::string(alloc::format!("{f}")))
                        }
                        _ => ctx.register_handle(v.clone()),
                    },
                    _ => ctx.register_handle(v.clone()),
                })
                .collect()
        });
        let ptr = if elem_tag > 0 {
            lust_mir_alloc_array_typed(elems.len() as i64, elem_tag)
        } else {
            lust_mir_alloc_array(elems.len() as i64)
        };
        if ptr.is_null() {
            return 0;
        }
        for (idx, val) in elems.into_iter().enumerate() {
            lust_mir_array_set(ptr, idx as i64, val);
        }
        ptr as i64
    }

    extern "C" fn lust_mir_clif_string_gsub(
        _vm_ctx: i64,
        handle: i64,
        pattern: i64,
        replacement: i64,
        elem_tag: i64,
    ) -> i64 {
        with_current_mir_context(|ctx| {
            let res = ctx.with_str_pair(handle, pattern, |text, pat| {
                ctx.with_str_handle(replacement, |repl| {
                    crate::vm::stdlib::string_gsub_fast(text, pat, repl)
                })
            });
            let Some(Ok((output, count))) = res.flatten() else {
                ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
                return 0;
            };
            if elem_tag == -1 {
                if !crate::mir::runtime_helpers::charge_handle_bytes(ctx, output.len()) {
                    return 0;
                }
                return ctx.intern_or_register_string(output);
            }
            if elem_tag == MIR_ARRAY_ELEM_TAG_STRING {
                let mut count_buf = [0u8; 24];
                let (count_handle, count_bytes) =
                    if let Some(h) = ctx.tostring_small_int_handle(count) {
                        let d = match count {
                            0..=9 => 1,
                            10..=99 => 2,
                            100..=999 => 3,
                            _ => 4,
                        };
                        (h, d)
                    } else {
                        let s = crate::mir::runtime_helpers::format_i64_stack(count, &mut count_buf);
                        let len = s.len();
                        let h = ctx.register_handle(crate::bytecode::Value::string(s));
                        (h, len)
                    };
                if !crate::mir::runtime_helpers::charge_handle_bytes(
                    ctx,
                    output.len().saturating_add(count_bytes),
                ) {
                    return 0;
                }
                let out_handle = ctx.intern_or_register_string(output);
                let ptr = ctx.alloc_array_with_words(MirType::String, &[out_handle, count_handle]);
                return ptr as i64;
            }
            // Non-string array destination fallback: materialize directly without intermediate Value::Array.
            let out_bytes = output.len();
            if !crate::mir::runtime_helpers::charge_handle_bytes(ctx, out_bytes) {
                return 0;
            }
            let elems = match elem_tag {
                MIR_ARRAY_ELEM_TAG_INT => [0i64, count],
                MIR_ARRAY_ELEM_TAG_BOOL => [0i64, i64::from(count != 0)],
                MIR_ARRAY_ELEM_TAG_FLOAT => [0i64, (count as f64).to_bits() as i64],
                _ => [
                    ctx.intern_or_register_string(output),
                    ctx.register_handle(crate::bytecode::Value::Int(count)),
                ],
            };
            let elem_ty =
                crate::mir::runtime_helpers::mir_array_elem_type_from_tag(elem_tag).unwrap_or(MirType::Unknown);
            if elem_ty == MirType::Bool {
                let ptr = lust_mir_alloc_array_typed(2, elem_tag);
                if ptr.is_null() {
                    return 0;
                }
                lust_mir_array_set(ptr, 0, elems[0]);
                lust_mir_array_set(ptr, 1, elems[1]);
                return ptr as i64;
            }
            ctx.alloc_array_with_words(elem_ty, &elems) as i64
        })
    }

    extern "C" fn lust_mir_clif_string_split(
        vm_ctx: i64,
        handle: i64,
        separator: i64,
        elem_tag: i64,
    ) -> i64 {
        if elem_tag != MIR_ARRAY_ELEM_TAG_STRING {
            let arr_handle = lust_mir_string_split(vm_ctx, handle, separator);
            return materialize_handle_array_to_mir_ptr(arr_handle, elem_tag);
        }
        with_current_mir_context(|ctx| {
            let Some(text_src) = ctx.resolve_str_source(handle) else {
                ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
                return 0;
            };
            let Some(sep_src) = ctx.resolve_str_source(separator) else {
                ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
                return 0;
            };
            let text = text_src.as_str();
            let sep = sep_src.as_str();

            // Upper-bound memory check: exact sum of split piece lengths is always <= text.len().
            // If `memory_used + text.len() <= limit`, we avoid scanning `text.split(sep)` twice
            // and charge the exact `total_bytes` accumulated in the single pass.
            let mem_used = ctx.memory_used();
            let mem_limit = ctx.max_memory_limit();
            let text_len_i64 = i64::try_from(text.len()).unwrap_or(i64::MAX);
            let fast_budget_ok = mem_used
                .checked_add(text_len_i64)
                .is_some_and(|total| total <= mem_limit);
            if !fast_budget_ok {
                let mut exact_bytes = 0usize;
                if sep.is_empty() {
                    exact_bytes = text.len();
                } else {
                    for piece in text.split(sep) {
                        exact_bytes = exact_bytes.saturating_add(piece.len());
                    }
                }
                if !crate::mir::runtime_helpers::charge_handle_bytes(ctx, exact_bytes) {
                    return 0;
                }
            }

            let mut inline_handles = [0i64; 16];
            let mut inline_len = 0usize;
            let mut overflow_handles: Option<Vec<i64>> = None;
            let mut pushed_heap = false;
            let mut total_bytes = 0usize;

            {
                let interned = ctx.interned_strings.borrow();
                let has_interned = !interned.is_empty();
                let mut dyn_map = ctx.dyn_interned_map.borrow_mut();
                let mut push_piece = |piece: &str| {
                    let bytes = piece.as_bytes();
                    total_bytes = total_bytes.saturating_add(bytes.len());
                    let h = if bytes.is_empty() {
                        crate::mir::runtime_helpers::EMPTY_STR_HANDLE
                    } else if bytes.len() == 1 && bytes[0] < 128 {
                        crate::mir::runtime_helpers::ascii_byte_handle(bytes[0])
                    } else if bytes.len() <= crate::mir::runtime_helpers::DYN_INTERN_MAX_BYTES {
                        if has_interned && let Some(&existing) = interned.get(piece) {
                            existing
                        } else if let Some(&existing) = dyn_map.get(piece) {
                            existing
                        } else {
                            let mut dyn_list = ctx.dyn_interned_list.borrow_mut();
                            if dyn_list.len() < crate::mir::runtime_helpers::DYN_INTERN_CAP {
                                let owned = alloc::string::ToString::to_string(piece);
                                let idx = dyn_list.len();
                                let h = crate::mir::runtime_helpers::DYN_INTERN_BASE - (idx as i64);
                                dyn_list.push(alloc::rc::Rc::new(owned.clone()));
                                dyn_map.insert(owned, h);
                                pushed_heap = true;
                                h
                            } else {
                                drop(dyn_list);
                                let mut hv = ctx.heap_values.borrow_mut();
                                hv.push(Some(crate::bytecode::Value::string(piece)));
                                pushed_heap = true;
                                hv.len() as i64
                            }
                        }
                    } else {
                        let mut hv = ctx.heap_values.borrow_mut();
                        hv.push(Some(crate::bytecode::Value::string(piece)));
                        pushed_heap = true;
                        hv.len() as i64
                    };
                    if let Some(vec) = overflow_handles.as_mut() {
                        vec.push(h);
                    } else if inline_len < inline_handles.len() {
                        inline_handles[inline_len] = h;
                        inline_len += 1;
                    } else {
                        let mut vec = Vec::with_capacity(32);
                        vec.extend_from_slice(&inline_handles);
                        vec.push(h);
                        overflow_handles = Some(vec);
                    }
                };
                if sep.is_empty() {
                    let mut utf8_buf = [0u8; 4];
                    for ch in text.chars() {
                        push_piece(ch.encode_utf8(&mut utf8_buf));
                    }
                } else if sep.len() == 1 && sep.as_bytes()[0] < 128 {
                    let delim = sep.as_bytes()[0];
                    let tbytes = text.as_bytes();
                    let mut start = 0usize;
                    let mut i = 0usize;
                    while i < tbytes.len() {
                        if tbytes[i] == delim {
                            push_piece(&text[start..i]);
                            start = i + 1;
                        }
                        i += 1;
                    }
                    push_piece(&text[start..]);
                } else {
                    for piece in text.split(sep) {
                        push_piece(piece);
                    }
                }
            }
            if fast_budget_ok {
                ctx.set_memory_used(mem_used.saturating_add(total_bytes as i64));
            }
            if pushed_heap {
                ctx.mark_dirty_allocation();
            }
            let slice: &[i64] = match overflow_handles.as_deref() {
                Some(v) => v,
                None => &inline_handles[..inline_len],
            };
            ctx.alloc_array_with_words(MirType::String, slice) as i64
        })
    }

    fn intern_static_struct_type_id(name: &str) -> i64 {
        use std::sync::Mutex;
        static TYPE_IDS: Mutex<Option<hashbrown::HashMap<String, i64>>> = Mutex::new(None);
        let mut guard = TYPE_IDS.lock().unwrap_or_else(|e| e.into_inner());
        let map = guard.get_or_insert_with(hashbrown::HashMap::new);
        if let Some(&id) = map.get(name) {
            return id;
        }
        let next_id = (map.len() as i64) + 1;
        map.insert(name.to_string(), next_id);
        next_id
    }

    extern "C" fn lust_mir_clif_unknown_type_id(obj: i64) -> i64 {
        if obj == 0 {
            return 0;
        }
        with_current_mir_context(|ctx| {
            let Some(rec) = ctx.get_active_allocation_record(obj as usize) else {
                return 0;
            };
            match &rec.kind {
                MirAllocationKind::Struct(name) => intern_static_struct_type_id(name),
                MirAllocationKind::CFlatStruct(name) => intern_static_struct_type_id(name),
                _ => 0,
            }
        })
    }

    extern "C" fn lust_mir_clif_unknown_get_field_fallback(obj: i64, field_handle: i64) -> i64 {
        with_current_mir_context(|ctx| {
            let Some(field_name) = ctx.with_str_handle(field_handle, |s| s.to_string()) else {
                ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
                return 0;
            };
            let Some(val) = ctx.get_handle(obj) else {
                ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
                return 0;
            };
            let found = match val {
                crate::bytecode::Value::Struct(_) => val.struct_get_field(&field_name),
                crate::bytecode::Value::Map(m) => m
                    .borrow()
                    .get(&crate::bytecode::ValueKey::from(
                        crate::bytecode::Value::string(field_name),
                    ))
                    .cloned(),
                _ => None,
            };
            match found {
                Some(crate::bytecode::Value::Int(n)) => n,
                Some(crate::bytecode::Value::Bool(b)) => i64::from(b),
                Some(crate::bytecode::Value::Float(f)) => f.to_bits() as i64,
                Some(other) => ctx.register_handle(other),
                None => {
                    ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
                    0
                }
            }
        })
    }

    extern "C" fn lust_mir_clif_string_sub_byte_array(
        base_handle: i64,
        sub_start: i64,
        sub_end: i64,
        byte_pos: i64,
    ) -> i64 {
        with_current_mir_context(|ctx| {
            let mut single = [0i64; 1];
            let mut count = 0usize;
            let ok = ctx.with_str_handle(base_handle, |text| {
                let sub = crate::vm::stdlib::string_sub_slice(text, sub_start, Some(sub_end));
                let raw = sub.as_bytes();
                let len = raw.len() as i64;
                let idx = if byte_pos < 0 {
                    len.saturating_add(byte_pos).saturating_add(1)
                } else {
                    byte_pos
                };
                if idx >= 1 && idx <= len {
                    single[0] = raw[(idx - 1) as usize] as i64;
                    count = 1;
                }
            });
            if ok.is_none() {
                ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
                return 0;
            }
            ctx.alloc_array_with_words(MirType::Int, &single[..count]) as i64
        })
    }

    extern "C" fn lust_mir_clif_string_byte(
        vm_ctx: i64,
        handle: i64,
        start: i64,
        end: i64,
        elem_tag: i64,
    ) -> i64 {
        if elem_tag == MIR_ARRAY_ELEM_TAG_INT {
            return with_current_mir_context(|ctx| {
                let mut inline_words = [0i64; 16];
                let mut inline_len = 0usize;
                let mut overflow_words: Option<Vec<i64>> = None;
                let ok = ctx.with_str_handle(handle, |text| {
                    let raw = text.as_bytes();
                    let len = raw.len() as i64;
                    let norm_idx = |idx: i64| -> i64 {
                        if idx < 0 { len.saturating_add(idx).saturating_add(1) } else { idx }
                    };
                    let s = norm_idx(start).max(1);
                    let e = norm_idx(end).min(len);
                    if s > e || s > len || e < 1 {
                        return;
                    }
                    let lo = (s - 1) as usize;
                    let hi = e as usize;
                    let slice = &raw[lo..hi];
                    if slice.len() <= inline_words.len() {
                        for (dst, &b) in inline_words.iter_mut().zip(slice.iter()) {
                            *dst = b as i64;
                        }
                        inline_len = slice.len();
                    } else {
                        overflow_words = Some(slice.iter().map(|&b| b as i64).collect());
                    }
                });
                if ok.is_none() {
                    ctx.set_status(MIR_STATUS_ERR_OUT_OF_BOUNDS);
                    return 0;
                }
                let words: &[i64] = match overflow_words.as_deref() {
                    Some(v) => v,
                    None => &inline_words[..inline_len],
                };
                ctx.alloc_array_with_words(MirType::Int, words) as i64
            });
        }
        let arr_handle = lust_mir_string_byte(vm_ctx, handle, start, end);
        materialize_handle_array_to_mir_ptr(arr_handle, elem_tag)
    }

    extern "C" fn lust_mir_clif_string_find(
        vm_ctx: i64,
        handle: i64,
        pattern: i64,
        start: i64,
        plain: i64,
        elem_tag: i64,
    ) -> i64 {
        let arr_handle = lust_mir_string_find(vm_ctx, handle, pattern, start, plain);
        if elem_tag == -2 {
            if arr_handle == 0 {
                return 0;
            }
            return with_current_mir_context(|ctx| {
                let Some(crate::bytecode::Value::Array(arr)) = ctx.get_handle(arr_handle) else {
                    return 0;
                };
                match arr.borrow().first() {
                    Some(crate::bytecode::Value::Int(n)) => *n,
                    _ => 0,
                }
            });
        }
        materialize_handle_array_to_mir_ptr(arr_handle, elem_tag)
    }

    pub fn find_loop_headers(blocks: &[BasicBlock]) -> hashbrown::HashSet<usize> {
        crate::mir::opt::Safepoints::find_loop_headers(blocks)
            .into_iter()
            .map(|b| b.0)
            .collect()
    }

    pub fn find_loop_backedges(blocks: &[BasicBlock]) -> Vec<(usize, usize)> {
        crate::mir::opt::Safepoints::find_loop_backedges(blocks)
            .into_iter()
            .map(|(from, to)| (from.0, to.0))
            .collect()
    }

    pub fn is_loop_backedge(blocks: &[BasicBlock], from: usize, to: usize) -> bool {
        crate::mir::opt::Safepoints::find_loop_backedges(blocks)
            .into_iter()
            .any(|(f, t)| f.0 == from && t.0 == to)
    }

    #[derive(Debug, Clone)]
    pub struct OsrPromotionTracker {
        pub hot_threshold: u32,
        pub backedge_hits: hashbrown::HashMap<(String, usize), u32>,
        pub promoted_entries: hashbrown::HashMap<(String, usize), *const u8>,
    }

    impl Default for OsrPromotionTracker {
        fn default() -> Self {
            Self::new(100)
        }
    }

    impl OsrPromotionTracker {
        pub fn new(hot_threshold: u32) -> Self {
            Self {
                hot_threshold,
                backedge_hits: hashbrown::HashMap::new(),
                promoted_entries: hashbrown::HashMap::new(),
            }
        }

        pub fn record_backedge(&mut self, func_name: &str, target_block: usize) -> bool {
            let entry = self
                .backedge_hits
                .entry((func_name.to_string(), target_block))
                .or_insert(0);
            *entry = entry.saturating_add(1);
            *entry >= self.hot_threshold
        }

        pub fn is_promoted(&self, func_name: &str, target_block: usize) -> bool {
            self.promoted_entries
                .contains_key(&(func_name.to_string(), target_block))
        }

        pub fn get_promoted(&self, func_name: &str, target_block: usize) -> Option<*const u8> {
            self.promoted_entries
                .get(&(func_name.to_string(), target_block))
                .copied()
        }

        pub fn get_promoted_entry(&self, func_name: &str, target_block: usize) -> Option<*const u8> {
            self.get_promoted(func_name, target_block)
        }

        pub fn register_promoted(
            &mut self,
            func_name: &str,
            target_block: usize,
            code_ptr: *const u8,
        ) {
            self.promoted_entries
                .insert((func_name.to_string(), target_block), code_ptr);
        }

        pub fn register_promotion(
            &mut self,
            func_name: &str,
            target_block: usize,
            code_ptr: *const u8,
        ) {
            self.register_promoted(func_name, target_block, code_ptr);
        }

        pub fn reset(&mut self) {
            self.backedge_hits.clear();
            self.promoted_entries.clear();
        }

        pub fn stats(&self) -> (usize, u32) {
            let promoted = self.promoted_entries.len();
            let total_hits: u32 = self.backedge_hits.values().sum();
            (promoted, total_hits)
        }
    }

    /// Horizontally sums the four `i64` lanes held across two `I64X2` vector accumulators.
    fn fold_vec_acc_lanes(
        builder: &mut FunctionBuilder,
        vacc0: Variable,
        vacc1: Variable,
    ) -> ClifValue {
        let fv0 = builder.use_var(vacc0);
        let fv1 = builder.use_var(vacc1);
        let sum_v = builder.ins().iadd(fv0, fv1);
        let l0 = builder.ins().extractlane(sum_v, 0);
        let l1 = builder.ins().extractlane(sum_v, 1);
        builder.ins().iadd(l0, l1)
    }

    /// Counter incremented each time `verify_promoted_loop_exit_completeness` mechanically
    /// verifies a promoted loop's Cranelift CFG before finalization (`PERF-T4-2` Barrier 5).
    pub static PROMOTED_LOOP_CFG_VERIFICATIONS: core::sync::atomic::AtomicU64 =
        core::sync::atomic::AtomicU64::new(0);

    pub fn promoted_loop_cfg_verifications() -> u64 {
        PROMOTED_LOOP_CFG_VERIFICATIONS.load(core::sync::atomic::Ordering::SeqCst)
    }

    pub fn reset_promoted_loop_cfg_verifications() {
        PROMOTED_LOOP_CFG_VERIFICATIONS.store(0, core::sync::atomic::Ordering::SeqCst);
    }

    #[derive(Clone, Copy)]
    struct PromotedLoopCounters {
        gas: Option<(Variable, Variable)>, // (gas_ptr_var, gas_var)
        sp: Option<(Variable, Variable)>,  // (sp_ptr_var, sp_cnt_var)
        quantum: Option<(Variable, u64)>,  // NG-W14-1: (quantum_rem_var, header_gas_amt) for Q=64 batch reconciliation
    }

    #[derive(Clone, Copy)]
    struct HoistedArrayVars {
        len_var: Variable,
        data_ptr_var: Variable,
    }

    struct PromotedLoopSpec {
        hdr: usize,
        loop_blocks: hashbrown::HashSet<usize>,
        promote_gas: bool,
        promote_sp: bool,
        pub quantized_header_gas_amt: Option<u64>,
    }

    /// NG-W14-1: Strictly nested innermost natural loop quantized (`Q=64`) inside
    /// an enclosing non-quantized counter-promoted loop (`parent_cl_idx`).
    struct InnerQuantizedLoopSpec {
        hdr: usize,
        loop_blocks: hashbrown::HashSet<usize>,
        parent_cl_idx: usize,
        header_gas_amt: u64,
    }

    struct HoistLoopSpec {
        hdr: usize,
        body_entry: usize,
        loop_exit: usize,
        then_is_body: bool,
        loop_blocks: hashbrown::HashSet<usize>,
        hoist_arrays: Vec<LocalId>,
        hoist_array_aliases: Vec<(LocalId, LocalId)>,
        hoist_structs: Vec<LocalId>,
    }

    struct LoopPromotionPlan {
        counter_loops: Vec<PromotedLoopSpec>,
        block_to_counter_loop: hashbrown::HashMap<usize, usize>,
        hdr_to_counter_loop: hashbrown::HashMap<usize, usize>,
        inner_quantized_loops: Vec<InnerQuantizedLoopSpec>,
        block_to_inner_quantized: hashbrown::HashMap<usize, usize>,
        hoist_loops: Vec<HoistLoopSpec>,
        hdr_to_hoist_loop: hashbrown::HashMap<usize, usize>,
        inline_push_blocks: hashbrown::HashSet<usize>,
    }

    /// Identifies non-escaping local integer queues created by a single `queue.new(const_cap)`
    /// (with optional 1-hop `_tmp -> q` move) that are accessed ONLY by `queue.offer_int` and
    /// `queue.take_int`. These lower to an inline MIR ring buffer with register-held `(head, len)`
    /// and zero per-operation C-ABI calls (`BENCH-NG-12`).
    fn compute_inline_local_queues(
        mir_fn: &MirFunction,
        osr_info: Option<(BlockId, &[LocalId])>,
        stack_slots: &hashbrown::HashMap<usize, StackSlot>,
        const_int_locals: &hashbrown::HashMap<usize, i64>,
        local_read_counts: &[usize],
    ) -> (
        hashbrown::HashMap<usize, (usize, i64, usize)>, // q_local -> (def_block, cap, alloc_tmp_local)
        hashbrown::HashSet<usize>,                      // inline_queue_blocks (offer/take)
    ) {
        let mut excluded: hashbrown::HashSet<usize> = stack_slots.keys().copied().collect();
        for &p in &mir_fn.params {
            excluded.insert(p.0);
        }
        if let Some((_, live)) = osr_info {
            for &l in live {
                excluded.insert(l.0);
            }
        }
        let mut def_count = vec![0usize; mir_fn.locals.len()];
        let mut copy_dst_of_tmp: hashbrown::HashMap<usize, usize> = hashbrown::HashMap::new();
        let mut tmp_use_count = vec![0usize; mir_fn.locals.len()];
        let mut q_new_defs: Vec<(usize, usize, i64)> = Vec::new(); // (tmp_loc, def_bb, cap)

        for (b_idx, block) in mir_fn.blocks.iter().enumerate() {
            for stmt in &block.statements {
                match stmt {
                    Statement::Assign(Place::Local(dst), rval, _) => {
                        if dst.0 < def_count.len() {
                            def_count[dst.0] += 1;
                        }
                        if let Rvalue::Use(
                            Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src)),
                        ) = rval
                        {
                            if src.0 < tmp_use_count.len() {
                                tmp_use_count[src.0] += 1;
                            }
                            copy_dst_of_tmp.insert(src.0, dst.0);
                        }
                    }
                    _ => {}
                }
            }
            if let Some(Terminator::Call {
                callee,
                args,
                destination: Place::Local(dst),
                error: None,
                ..
            }) = &block.terminator
            {
                if dst.0 < def_count.len() {
                    def_count[dst.0] += 1;
                }
                if callee == "queue.new"
                    && args.len() == 1
                    && let Some(cap) = resolve_const_int_operand(&args[0], const_int_locals)
                    && (1..=4096).contains(&cap)
                {
                    q_new_defs.push((dst.0, b_idx, cap));
                }
            }
        }

        let mut candidates: hashbrown::HashMap<usize, (usize, i64, usize)> = hashbrown::HashMap::new();
        for (tmp_loc, def_bb, cap) in q_new_defs {
            if excluded.contains(&tmp_loc) || def_count[tmp_loc] != 1 {
                continue;
            }
            let target_loc = if tmp_use_count[tmp_loc] == 1
                && let Some(&dst_loc) = copy_dst_of_tmp.get(&tmp_loc)
                && !excluded.contains(&dst_loc)
                && def_count[dst_loc] == 1
            {
                dst_loc
            } else if tmp_use_count[tmp_loc] == 0 {
                tmp_loc
            } else {
                continue;
            };
            candidates.insert(target_loc, (def_bb, cap, tmp_loc));
        }

        if candidates.is_empty() {
            return (candidates, hashbrown::HashSet::new());
        }

        // Reject any candidate (or its alloc_tmp) that escapes into ANY statement/terminator
        // other than `q = Copy/Move(tmp)` and `queue.offer_int(q, ..)` / `queue.take_int(q)`.
        let mut bad_q: hashbrown::HashSet<usize> = hashbrown::HashSet::new();
        let mut tmp_to_q: hashbrown::HashMap<usize, usize> = hashbrown::HashMap::new();
        for (&q_loc, &(_, _, tmp_loc)) in &candidates {
            tmp_to_q.insert(tmp_loc, q_loc);
        }
        let mut mark_if_cand = |loc_id: usize| {
            if candidates.contains_key(&loc_id) {
                bad_q.insert(loc_id);
            }
            if let Some(&q_loc) = tmp_to_q.get(&loc_id) {
                bad_q.insert(q_loc);
            }
        };
        let check_op = |op: &Operand, mark: &mut dyn FnMut(usize)| {
            if let Operand::Copy(p) | Operand::Move(p) = op {
                mark(p.root_local().0);
                if let Place::Index(_, idx) = p {
                    mark(idx.0);
                }
            }
        };
        for block in &mir_fn.blocks {
            for stmt in &block.statements {
                if let Statement::Assign(lhs, rval, _) = stmt {
                    // Allow the single `q = Move/Copy(tmp)` statement, and allow dead alias
                    // copies (`offer = Copy(q)`, `take = Copy(q)`) whose destination has 0 reads.
                    let is_allowed_init_copy = matches!(
                        (lhs, rval),
                        (
                            Place::Local(dst),
                            Rvalue::Use(
                                Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src))
                            )
                        ) if candidates.get(&dst.0).is_some_and(|&(_, _, t)| t == src.0)
                            || (local_read_counts.get(dst.0).copied().unwrap_or(1) == 0
                                && (candidates.contains_key(&src.0) || tmp_to_q.contains_key(&src.0)))
                    );
                    if is_allowed_init_copy {
                        continue;
                    }
                    if !matches!(lhs, Place::Local(_)) {
                        mark_if_cand(lhs.root_local().0);
                    }
                    match rval {
                        Rvalue::Use(op) | Rvalue::Cast(op, _) | Rvalue::UnaryOp(_, op) => {
                            check_op(op, &mut mark_if_cand);
                        }
                        Rvalue::BinaryOp(_, l, r) => {
                            check_op(l, &mut mark_if_cand);
                            check_op(r, &mut mark_if_cand);
                        }
                        Rvalue::Aggregate(_, ops) => {
                            for op in ops {
                                check_op(op, &mut mark_if_cand);
                            }
                        }
                        Rvalue::Closure { captures, .. } => {
                            for (_, op, _) in captures {
                                check_op(op, &mut mark_if_cand);
                            }
                        }
                    }
                }
            }
            match &block.terminator {
                Some(Terminator::Call {
                    callee,
                    args,
                    destination,
                    error,
                    ..
                }) => {
                    if !matches!(destination, Place::Local(_)) {
                        mark_if_cand(destination.root_local().0);
                    }
                    let is_q_op = matches!(callee.as_str(), "queue.offer_int" | "queue.take_int")
                        && error.is_none();
                    for (idx, a) in args.iter().enumerate() {
                        if is_q_op && idx == 0 {
                            continue;
                        }
                        check_op(a, &mut mark_if_cand);
                    }
                }
                Some(Terminator::Cond { cond, .. }) => check_op(cond, &mut mark_if_cand),
                Some(Terminator::Switch { discr, .. }) => check_op(discr, &mut mark_if_cand),
                Some(Terminator::Return(Some(op), _)) | Some(Terminator::Raise(op, _)) => {
                    check_op(op, &mut mark_if_cand);
                }
                _ => {}
            }
        }
        for b in bad_q {
            candidates.remove(&b);
        }

        let mut inline_q_blocks = hashbrown::HashSet::new();
        for (b_idx, block) in mir_fn.blocks.iter().enumerate() {
            if let Some(Terminator::Call {
                callee,
                args,
                error: None,
                ..
            }) = &block.terminator
                && matches!(callee.as_str(), "queue.offer_int" | "queue.take_int")
                && let Some(Operand::Copy(Place::Local(q_loc)) | Operand::Move(Place::Local(q_loc))) =
                    args.first()
                && candidates.contains_key(&q_loc.0)
            {
                inline_q_blocks.insert(b_idx);
            }
        }
        (candidates, inline_q_blocks)
    }

    /// Recognized `I64X2` vectorizable inner reduction (`s += arr[offset + k] <op> k`)
    /// where `arr` is a hoisted `Array<int>`, `offset` is loop-invariant, and all temporaries
    /// are dead outside the loop (`BENCH-NG-14`).
    struct VecI64x2ReductionKernel {
        acc_local: LocalId,
        arr_local: LocalId,
        offset_local: LocalId,
        op: MirBinaryOp,
        arr_on_lhs: bool,
    }

    /// Recognized pure counted innermost quantized loop (`k < bound`, `k += 1`, straight-line
    /// pure body with proven-in-bounds hoisted int-array loads) eligible for an 8x unrolled
    /// fast-path gated on `quantum_rem >= 8 && 0 <= k <= bound - 8` (`BENCH-NG-14`).
    struct Unroll4InnerLoopSpec {
        cond_local: LocalId,
        idx_local: LocalId,
        bound_op: Operand,
        exit_block: BlockId,
        path_blocks: Vec<usize>,
        /// Optional associative `i64` additive reduction (`acc = acc + delta` or 1-hop via tmp)
        /// eligible for a balanced 8-way binary reduction tree (breaking serial `iadd` latency).
        add_reduction: Option<(LocalId, usize, usize, Operand, Option<(usize, usize)>)>,
        /// Statements belonging to `k = k + 1` (safe to skip per-step when `k` is not read after).
        idx_inc_stmts: Vec<(usize, usize)>,
        /// Optional 4x`I64X2` (8-lane) SIMD reduction kernel replacing the 8 scalar steps.
        vec_i64x2_kernel: Option<VecI64x2ReductionKernel>,
    }

    #[allow(clippy::too_many_arguments)]
    fn detect_unroll4_inner_loop(
        hdr: usize,
        mir_fn: &MirFunction,
        loop_headers: &hashbrown::HashSet<usize>,
        stack_slots: &hashbrown::HashMap<usize, StackSlot>,
        active_hoisted_arrays: &hashbrown::HashMap<usize, HoistedArrayVars>,
        local_read_counts: &[usize],
        bounds_excluded_locals: &hashbrown::HashSet<usize>,
        loop_bounds_analysis: &crate::mir::ir::LoopBoundsAnalysis,
        validated_in: &[hashbrown::HashSet<usize>],
    ) -> Option<Unroll4InnerLoopSpec> {
        let hdr_block = &mir_fn.blocks[hdr];
        if !matches!(
            hdr_block.statements.first(),
            Some(Statement::ResourceCheck(ResourceKind::Gas, ..))
        ) {
            return None;
        }
        let mut cond_def: Option<(LocalId, LocalId, Operand)> = None;
        for stmt in &hdr_block.statements[1..] {
            match stmt {
                Statement::StorageLive(..) | Statement::StorageDead(..) => {}
                Statement::Assign(
                    Place::Local(cond_loc),
                    Rvalue::BinaryOp(
                        MirBinaryOp::Lt,
                        Operand::Copy(Place::Local(idx_loc)) | Operand::Move(Place::Local(idx_loc)),
                        bound_op,
                    ),
                    _,
                ) if cond_def.is_none() => {
                    cond_def = Some((*cond_loc, *idx_loc, bound_op.clone()));
                }
                _ => return None,
            }
        }
        let (cond_loc, idx_local, bound_op) = cond_def?;
        if mir_fn.locals.get(idx_local.0).map(|l| &l.ty) != Some(&MirType::Int)
            || stack_slots.contains_key(&idx_local.0)
        {
            return None;
        }
        let bound_local = match &bound_op {
            Operand::Constant(MirConstant::Int(_)) => None,
            Operand::Copy(Place::Local(b)) | Operand::Move(Place::Local(b)) => {
                if *b == idx_local
                    || mir_fn.locals.get(b.0).map(|l| &l.ty) != Some(&MirType::Int)
                    || stack_slots.contains_key(&b.0)
                {
                    return None;
                }
                Some(*b)
            }
            _ => return None,
        };
        let (body_start, exit_block) = match &hdr_block.terminator {
            Some(Terminator::Cond {
                cond: Operand::Copy(Place::Local(c)) | Operand::Move(Place::Local(c)),
                then_block,
                else_block,
                ..
            }) if *c == cond_loc && then_block.0 != hdr && else_block.0 != hdr => {
                (then_block.0, *else_block)
            }
            _ => return None,
        };

        let mut path_blocks = Vec::new();
        let mut cur = body_start;
        let mut idx_update_count = 0usize;
        let mut idx_updated_by_plus_one = false;
        let mut idx_read_after_update = false;
        let mut idx_inc_stmts: Vec<(usize, usize)> = Vec::new();
        let mut local_body_reads: hashbrown::HashMap<usize, usize> = hashbrown::HashMap::new();
        let mut local_body_writes: hashbrown::HashMap<usize, usize> = hashbrown::HashMap::new();
        let mut read_before_write: hashbrown::HashSet<usize> = hashbrown::HashSet::new();
        // Track potential `tmp = acc + delta` (or `acc = acc + delta`) for balanced reduction tree
        let mut add_defs: hashbrown::HashMap<usize, (usize, usize, LocalId, Operand)> =
            hashbrown::HashMap::new();
        let mut copy_from_add: Vec<(LocalId, usize, usize, usize)> = Vec::new();
        // Track `offset + idx_local` -> `arr[idx]` -> `v <op> idx_local` for I64X2 SIMD kernel
        let mut offset_plus_idx: hashbrown::HashMap<usize, LocalId> = hashbrown::HashMap::new();
        let mut arr_load_defs: hashbrown::HashMap<usize, (LocalId, LocalId)> =
            hashbrown::HashMap::new();
        let mut vec_op_defs: hashbrown::HashMap<usize, (MirBinaryOp, LocalId, LocalId, bool)> =
            hashbrown::HashMap::new();
        let mut folded_rc_dsts: hashbrown::HashSet<usize> = hashbrown::HashSet::new();

        let count_op_reads = |op: &Operand,
                              reads: &mut hashbrown::HashMap<usize, usize>,
                              writes: &hashbrown::HashMap<usize, usize>,
                              rbw: &mut hashbrown::HashSet<usize>,
                              idx_upd_done: bool,
                              idx_read_after: &mut bool| {
            let mut note = |l: LocalId| {
                *reads.entry(l.0).or_insert(0) += 1;
                if !writes.contains_key(&l.0) {
                    rbw.insert(l.0);
                }
                if idx_upd_done && l == idx_local {
                    *idx_read_after = true;
                }
            };
            match op {
                Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l)) => note(*l),
                Operand::Copy(Place::Index(base, i_loc))
                | Operand::Move(Place::Index(base, i_loc)) => {
                    if let Place::Local(arr) = base.as_ref() {
                        note(*arr);
                    }
                    note(*i_loc);
                }
                _ => {}
            }
        };

        let check_pure_op = |op: &Operand, stmt_proven: &[(LocalId, LocalId)]| -> bool {
            match op {
                Operand::Constant(MirConstant::Int(_) | MirConstant::Bool(_)) => true,
                Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l)) => {
                    !stack_slots.contains_key(&l.0)
                        && matches!(
                            mir_fn.locals.get(l.0).map(|d| &d.ty),
                            Some(MirType::Int | MirType::Bool)
                        )
                }
                Operand::Copy(Place::Index(base, i_loc))
                | Operand::Move(Place::Index(base, i_loc)) => {
                    if let Place::Local(arr) = base.as_ref() {
                        active_hoisted_arrays.contains_key(&arr.0)
                            && !stack_slots.contains_key(&i_loc.0)
                            && mir_fn.locals.get(arr.0).map(|d| &d.ty)
                                == Some(&MirType::Array(Box::new(MirType::Int)))
                            && stmt_proven.contains(&(*arr, *i_loc))
                    } else {
                        false
                    }
                }
                _ => false,
            }
        };

        for _ in 0..8 {
            if cur == hdr {
                break;
            }
            if loop_headers.contains(&cur) || path_blocks.contains(&cur) {
                return None;
            }
            path_blocks.push(cur);
            let blk = &mir_fn.blocks[cur];
            let rf = mir_fn.detect_unsigned_range_checks_in_block(
                cur,
                local_read_counts,
                bounds_excluded_locals,
                &loop_bounds_analysis.non_neg_locals_in[cur],
            );
            let mut cur_proven: Vec<(LocalId, LocalId)> =
                mir_fn.proven_in_bounds.get(&cur).cloned().unwrap_or_default();
            if let Some(ind_pairs) = loop_bounds_analysis.proven_in_bounds.get(&cur) {
                for &pair in ind_pairs {
                    if !cur_proven.contains(&pair) {
                        cur_proven.push(pair);
                    }
                }
            }
            let mut folded_rc: hashbrown::HashSet<usize> = hashbrown::HashSet::new();
            let mut folded_len_skip: hashbrown::HashSet<usize> = hashbrown::HashSet::new();
            let mut proven_true_conds: hashbrown::HashSet<usize> = hashbrown::HashSet::new();
            for (&rc_stmt_idx, rc) in &rf.fused_checks {
                let Some(arr_loc) = rc.arr_local else {
                    continue;
                };
                let proven_at_stmt = loop_bounds_analysis
                    .proven_in_bounds_at_stmt
                    .get(&(cur, rc_stmt_idx))
                    .is_some_and(|v| v.contains(&(arr_loc, rc.idx_local)));
                let proven_from_entry = if cur_proven.contains(&(arr_loc, rc.idx_local)) {
                    !blk.statements[..rc_stmt_idx].iter().any(|s| match s {
                        Statement::Assign(Place::Local(d), _, _) => {
                            *d == arr_loc || *d == rc.idx_local
                        }
                        Statement::StorageDead(d, _) => *d == arr_loc || *d == rc.idx_local,
                        _ => false,
                    })
                } else {
                    false
                };
                if proven_at_stmt || proven_from_entry {
                    folded_rc.insert(rc_stmt_idx);
                    if let Some(s_len_idx) = rc.s_len
                        && (validated_in[cur].contains(&arr_loc.0)
                            || active_hoisted_arrays.contains_key(&arr_loc.0))
                    {
                        folded_len_skip.insert(s_len_idx);
                    }
                }
            }

            let mut plus_one_tmps: hashbrown::HashMap<usize, (usize, usize)> =
                hashbrown::HashMap::new();
            for (stmt_idx, stmt) in blk.statements.iter().enumerate() {
                if rf.skip_stmts.contains(&stmt_idx) || folded_len_skip.contains(&stmt_idx) {
                    continue;
                }
                if folded_rc.contains(&stmt_idx) {
                    if let Some(rc) = rf.fused_checks.get(&stmt_idx) {
                        proven_true_conds.insert(rc.dst.0);
                        folded_rc_dsts.insert(rc.dst.0);
                        cur_proven.retain(|&(a, i)| a != rc.dst && i != rc.dst);
                    }
                    continue;
                }
                let mut stmt_proven = cur_proven.clone();
                if let Some(extra) = loop_bounds_analysis.proven_in_bounds_at_stmt.get(&(cur, stmt_idx))
                {
                    for &pair in extra {
                        if !stmt_proven.contains(&pair) {
                            stmt_proven.push(pair);
                        }
                    }
                }
                match stmt {
                    Statement::StorageLive(..) | Statement::StorageDead(..) => {}
                    Statement::Assign(Place::Local(dst), rval, _) => {
                        if stack_slots.contains_key(&dst.0) || bound_local == Some(*dst) {
                            return None;
                        }
                        if !matches!(
                            mir_fn.locals.get(dst.0).map(|d| &d.ty),
                            Some(MirType::Int | MirType::Bool)
                        ) {
                            return None;
                        }
                        *local_body_writes.entry(dst.0).or_insert(0) += 1;
                        let is_idx_plus_one = matches!(
                            rval,
                            Rvalue::BinaryOp(
                                MirBinaryOp::Add,
                                Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l)),
                                Operand::Constant(MirConstant::Int(1)),
                            ) | Rvalue::BinaryOp(
                                MirBinaryOp::Add,
                                Operand::Constant(MirConstant::Int(1)),
                                Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l)),
                            ) if *l == idx_local
                        );
                        if *dst == idx_local {
                            idx_update_count += 1;
                            let via_tmp = match rval {
                                Rvalue::Use(
                                    Operand::Copy(Place::Local(t))
                                        | Operand::Move(Place::Local(t)),
                                ) => plus_one_tmps.get(&t.0).copied().map(|pos| (*t, pos)),
                                _ => None,
                            };
                            idx_updated_by_plus_one = is_idx_plus_one || via_tmp.is_some();
                            if is_idx_plus_one {
                                idx_inc_stmts.push((cur, stmt_idx));
                            } else if let Some((t_loc, t_pos)) = via_tmp
                                && local_read_counts.get(t_loc.0).copied().unwrap_or(2) == 1
                            {
                                idx_inc_stmts.push(t_pos);
                                idx_inc_stmts.push((cur, stmt_idx));
                            }
                            plus_one_tmps.clear();
                        } else if is_idx_plus_one {
                            plus_one_tmps.insert(dst.0, (cur, stmt_idx));
                        } else {
                            plus_one_tmps.remove(&dst.0);
                        }

                        let idx_upd_done = idx_update_count > 0;
                        match rval {
                            Rvalue::Use(op) => {
                                if !check_pure_op(op, &stmt_proven) {
                                    return None;
                                }
                                count_op_reads(
                                    op,
                                    &mut local_body_reads,
                                    &local_body_writes,
                                    &mut read_before_write,
                                    idx_upd_done,
                                    &mut idx_read_after_update,
                                );
                                if let Operand::Copy(Place::Local(src))
                                | Operand::Move(Place::Local(src)) = op
                                {
                                    if add_defs.contains_key(&src.0) {
                                        copy_from_add.push((*dst, cur, stmt_idx, src.0));
                                    }
                                    if let Some(&off_l) = offset_plus_idx.get(&src.0) {
                                        offset_plus_idx.insert(dst.0, off_l);
                                    }
                                    if let Some(&al) = arr_load_defs.get(&src.0) {
                                        arr_load_defs.insert(dst.0, al);
                                    }
                                    if proven_true_conds.contains(&src.0) {
                                        proven_true_conds.insert(dst.0);
                                    } else {
                                        proven_true_conds.remove(&dst.0);
                                    }
                                } else if let Operand::Copy(Place::Index(base, i_loc))
                                | Operand::Move(Place::Index(base, i_loc)) = op
                                {
                                    if let Place::Local(arr) = base.as_ref()
                                        && let Some(&off_l) = offset_plus_idx.get(&i_loc.0)
                                    {
                                        arr_load_defs.insert(dst.0, (*arr, off_l));
                                    }
                                    proven_true_conds.remove(&dst.0);
                                } else {
                                    proven_true_conds.remove(&dst.0);
                                }
                            }
                            Rvalue::UnaryOp(
                                MirUnaryOp::Neg | MirUnaryOp::Not | MirUnaryOp::BitNot,
                                op,
                            ) => {
                                if !check_pure_op(op, &stmt_proven) {
                                    return None;
                                }
                                count_op_reads(
                                    op,
                                    &mut local_body_reads,
                                    &local_body_writes,
                                    &mut read_before_write,
                                    idx_upd_done,
                                    &mut idx_read_after_update,
                                );
                                proven_true_conds.remove(&dst.0);
                            }
                            Rvalue::BinaryOp(op_kind, l, r)
                                if !matches!(
                                    op_kind,
                                    MirBinaryOp::Div
                                        | MirBinaryOp::Mod
                                        | MirBinaryOp::FloorDiv
                                        | MirBinaryOp::Pow
                                        | MirBinaryOp::Concat
                                ) =>
                            {
                                if !check_pure_op(l, &stmt_proven)
                                    || !check_pure_op(r, &stmt_proven)
                                {
                                    return None;
                                }
                                count_op_reads(
                                    l,
                                    &mut local_body_reads,
                                    &local_body_writes,
                                    &mut read_before_write,
                                    idx_upd_done,
                                    &mut idx_read_after_update,
                                );
                                count_op_reads(
                                    r,
                                    &mut local_body_reads,
                                    &local_body_writes,
                                    &mut read_before_write,
                                    idx_upd_done,
                                    &mut idx_read_after_update,
                                );
                                if *op_kind == MirBinaryOp::Add
                                    && mir_fn.locals.get(dst.0).map(|d| &d.ty)
                                        == Some(&MirType::Int)
                                {
                                    if let (
                                        Operand::Copy(Place::Local(la))
                                        | Operand::Move(Place::Local(la)),
                                        Operand::Copy(Place::Local(lb))
                                        | Operand::Move(Place::Local(lb)),
                                    ) = (l, r)
                                    {
                                        if *lb == idx_local && *la != idx_local {
                                            offset_plus_idx.insert(dst.0, *la);
                                        } else if *la == idx_local && *lb != idx_local {
                                            offset_plus_idx.insert(dst.0, *lb);
                                        }
                                    }
                                    if let Operand::Copy(Place::Local(acc_l))
                                    | Operand::Move(Place::Local(acc_l)) = l
                                    {
                                        add_defs.insert(dst.0, (cur, stmt_idx, *acc_l, r.clone()));
                                    } else if let Operand::Copy(Place::Local(acc_r))
                                    | Operand::Move(Place::Local(acc_r)) = r
                                    {
                                        add_defs.insert(dst.0, (cur, stmt_idx, *acc_r, l.clone()));
                                    }
                                }
                                if matches!(
                                    op_kind,
                                    MirBinaryOp::BitXor
                                        | MirBinaryOp::Add
                                        | MirBinaryOp::Sub
                                        | MirBinaryOp::BitAnd
                                        | MirBinaryOp::BitOr
                                ) && mir_fn.locals.get(dst.0).map(|d| &d.ty)
                                    == Some(&MirType::Int)
                                    && let (
                                        Operand::Copy(Place::Local(la))
                                        | Operand::Move(Place::Local(la)),
                                        Operand::Copy(Place::Local(lb))
                                        | Operand::Move(Place::Local(lb)),
                                    ) = (l, r)
                                {
                                    if *lb == idx_local
                                        && let Some(&(arr, off_l)) = arr_load_defs.get(&la.0)
                                    {
                                        vec_op_defs.insert(dst.0, (op_kind.clone(), arr, off_l, true));
                                    } else if *la == idx_local
                                        && let Some(&(arr, off_l)) = arr_load_defs.get(&lb.0)
                                    {
                                        vec_op_defs.insert(dst.0, (op_kind.clone(), arr, off_l, false));
                                    }
                                }
                                proven_true_conds.remove(&dst.0);
                            }
                            _ => return None,
                        }
                        cur_proven.retain(|&(a, i)| a != *dst && i != *dst);
                    }
                    _ => return None,
                }
            }

            match &blk.terminator {
                Some(Terminator::Goto(next, _)) => cur = next.0,
                Some(Terminator::Cond {
                    cond: Operand::Copy(Place::Local(c)) | Operand::Move(Place::Local(c)),
                    then_block,
                    ..
                }) if proven_true_conds.contains(&c.0) => cur = then_block.0,
                _ => return None,
            }
        }

        if cur != hdr
            || path_blocks.is_empty()
            || idx_update_count != 1
            || !idx_updated_by_plus_one
        {
            return None;
        }
        if idx_read_after_update {
            idx_inc_stmts.clear();
        }
        // Identify an associative `i64` additive reduction accumulator (`acc = acc + delta` or
        // `tmp = acc + delta; acc = tmp`) whose ONLY read in `path_blocks` is that single add.
        let mut add_reduction = None;
        let is_valid_acc = |acc: LocalId, delta: &Operand| -> bool {
            acc != idx_local
                && bound_local != Some(acc)
                && mir_fn.locals.get(acc.0).map(|d| &d.ty) == Some(&MirType::Int)
                && local_body_writes.get(&acc.0).copied().unwrap_or(0) == 1
                && local_body_reads.get(&acc.0).copied().unwrap_or(0) == 1
                && !matches!(
                    delta,
                    Operand::Copy(Place::Local(d)) | Operand::Move(Place::Local(d)) if *d == acc
                )
        };
        // Check 1-hop `tmp = acc + delta; acc = tmp` first (standard MIR lowering form)
        for &(acc_dst, copy_bb, copy_si, tmp_id) in &copy_from_add {
            if let Some(&(add_bb, add_si, acc_src, ref delta_op)) = add_defs.get(&tmp_id)
                && acc_src == acc_dst
                && tmp_id != acc_dst.0
                && local_body_writes.get(&tmp_id).copied().unwrap_or(0) == 1
                && local_body_reads.get(&tmp_id).copied().unwrap_or(0) == 1
                && local_read_counts.get(tmp_id).copied().unwrap_or(2) == 1
                && is_valid_acc(acc_dst, delta_op)
            {
                add_reduction = Some((
                    acc_dst,
                    add_bb,
                    add_si,
                    delta_op.clone(),
                    Some((copy_bb, copy_si)),
                ));
                break;
            }
        }
        // Fallback: direct `acc = acc + delta`
        if add_reduction.is_none() {
            let mut direct_keys: Vec<usize> = add_defs.keys().copied().collect();
            direct_keys.sort_unstable();
            for dst_id in direct_keys {
                let (add_bb, add_si, acc_src, ref delta_op) = add_defs[&dst_id];
                if acc_src.0 == dst_id && is_valid_acc(acc_src, delta_op) {
                    add_reduction = Some((acc_src, add_bb, add_si, delta_op.clone(), None));
                    break;
                }
            }
        }
        // Detect 4x I64X2 (8-lane) SIMD reduction kernel (`s += arr[offset + k] <op> k`)
        // when all intermediate temporaries are dead outside `path_blocks` and not loop-carried.
        let vec_i64x2_kernel = if cfg!(any(target_arch = "aarch64", target_arch = "x86_64"))
            && !idx_inc_stmts.is_empty()
            && let Some((acc_local, _, _, ref delta_op, _)) = add_reduction
            && let Operand::Copy(Place::Local(d_loc)) | Operand::Move(Place::Local(d_loc)) = delta_op
            && let Some((op, arr_local, offset_local, arr_on_lhs)) = vec_op_defs.get(&d_loc.0)
            && !local_body_writes.contains_key(&offset_local.0)
            && !local_body_writes.contains_key(&arr_local.0)
            && !stack_slots.contains_key(&offset_local.0)
            && active_hoisted_arrays.contains_key(&arr_local.0)
        {
            let path_block_set: hashbrown::HashSet<usize> = path_blocks.iter().copied().collect();
            let mut read_outside_path: hashbrown::HashSet<usize> = hashbrown::HashSet::new();
            let mark_op_outside = |op: &Operand, out: &mut hashbrown::HashSet<usize>| match op {
                Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l)) => {
                    out.insert(l.0);
                }
                Operand::Copy(Place::Index(base, i_loc))
                | Operand::Move(Place::Index(base, i_loc)) => {
                    if let Place::Local(arr) = base.as_ref() {
                        out.insert(arr.0);
                    }
                    out.insert(i_loc.0);
                }
                Operand::Copy(Place::Field(base, _)) | Operand::Move(Place::Field(base, _)) => {
                    out.insert(base.root_local().0);
                }
                _ => {}
            };
            for (ob_idx, ob) in mir_fn.blocks.iter().enumerate() {
                if path_block_set.contains(&ob_idx) {
                    continue;
                }
                for stmt in &ob.statements {
                    if let Statement::Assign(lhs, rval, _) = stmt {
                        if !matches!(lhs, Place::Local(_)) {
                            read_outside_path.insert(lhs.root_local().0);
                            if let Place::Index(_, i_loc) = lhs {
                                read_outside_path.insert(i_loc.0);
                            }
                        }
                        match rval {
                            Rvalue::Use(op) | Rvalue::Cast(op, _) | Rvalue::UnaryOp(_, op) => {
                                mark_op_outside(op, &mut read_outside_path);
                            }
                            Rvalue::BinaryOp(_, l, r) => {
                                mark_op_outside(l, &mut read_outside_path);
                                mark_op_outside(r, &mut read_outside_path);
                            }
                            Rvalue::Aggregate(_, ops) => {
                                for op in ops {
                                    mark_op_outside(op, &mut read_outside_path);
                                }
                            }
                            Rvalue::Closure { captures, .. } => {
                                for (_, op, _) in captures {
                                    mark_op_outside(op, &mut read_outside_path);
                                }
                            }
                        }
                    }
                }
                match &ob.terminator {
                    Some(Terminator::Call {
                        args, destination, ..
                    }) => {
                        if !matches!(destination, Place::Local(_)) {
                            read_outside_path.insert(destination.root_local().0);
                        }
                        for a in args {
                            mark_op_outside(a, &mut read_outside_path);
                        }
                    }
                    Some(Terminator::Cond { cond, .. }) => {
                        mark_op_outside(cond, &mut read_outside_path);
                    }
                    Some(Terminator::Switch { discr, .. }) => {
                        mark_op_outside(discr, &mut read_outside_path);
                    }
                    Some(Terminator::Return(Some(op), _)) | Some(Terminator::Raise(op, _)) => {
                        mark_op_outside(op, &mut read_outside_path);
                    }
                    _ => {}
                }
            }
            let temps_dead_outside = local_body_writes.keys().all(|&w| {
                w == acc_local.0
                    || w == idx_local.0
                    || (!read_outside_path.contains(&w) && !read_before_write.contains(&w))
            }) && folded_rc_dsts
                .iter()
                .all(|&rc_w| !read_outside_path.contains(&rc_w));
            if temps_dead_outside {
                Some(VecI64x2ReductionKernel {
                    acc_local,
                    arr_local: *arr_local,
                    offset_local: *offset_local,
                    op: op.clone(),
                    arr_on_lhs: *arr_on_lhs,
                })
            } else {
                None
            }
        } else {
            None
        };
        Some(Unroll4InnerLoopSpec {
            cond_local: cond_loc,
            idx_local,
            bound_op,
            exit_block,
            path_blocks,
            add_reduction,
            idx_inc_stmts,
            vec_i64x2_kernel,
        })
    }

    enum VecQueueLoopKind {
        /// `while k < bound { offer(q, base + k); k += 1 }`
        OfferLinear {
            q_loc: usize,
            offer_dst: Option<LocalId>,
            pre_stmts: Vec<(usize, usize)>,
            val_op: Operand,
        },
        /// `while r < bound { msg = take(q); acc = (acc + msg) % mod_const; r += 1 }`
        TakeAddMod {
            q_loc: usize,
            acc_local: LocalId,
            mod_const: i64,
        },
    }

    struct VecQueueLoopSpec {
        cond_local: LocalId,
        idx_local: LocalId,
        bound_op: Operand,
        exit_block: BlockId,
        kind: VecQueueLoopKind,
    }

    fn detect_vec_queue_inner_loop(
        hdr: usize,
        mir_fn: &MirFunction,
        loop_headers: &hashbrown::HashSet<usize>,
        stack_slots: &hashbrown::HashMap<usize, StackSlot>,
        const_int_locals: &hashbrown::HashMap<usize, i64>,
        inline_local_queues: &hashbrown::HashMap<usize, (usize, i64, usize)>,
    ) -> Option<VecQueueLoopSpec> {
        if inline_local_queues.is_empty() {
            return None;
        }
        let hdr_block = &mir_fn.blocks[hdr];
        if !matches!(
            hdr_block.statements.first(),
            Some(Statement::ResourceCheck(ResourceKind::Gas, ..))
        ) {
            return None;
        }
        let mut cond_def: Option<(LocalId, LocalId, Operand)> = None;
        for stmt in &hdr_block.statements[1..] {
            match stmt {
                Statement::StorageLive(..) | Statement::StorageDead(..) => {}
                Statement::Assign(
                    Place::Local(cond_loc),
                    Rvalue::BinaryOp(
                        MirBinaryOp::Lt,
                        Operand::Copy(Place::Local(idx_loc)) | Operand::Move(Place::Local(idx_loc)),
                        bound_op,
                    ),
                    _,
                ) if cond_def.is_none() => {
                    cond_def = Some((*cond_loc, *idx_loc, bound_op.clone()));
                }
                _ => return None,
            }
        }
        let (cond_local, idx_local, bound_op) = cond_def?;
        if mir_fn.locals.get(idx_local.0).map(|l| &l.ty) != Some(&MirType::Int)
            || stack_slots.contains_key(&idx_local.0)
        {
            return None;
        }
        let bound_local = match &bound_op {
            Operand::Constant(MirConstant::Int(_)) => None,
            Operand::Copy(Place::Local(b)) | Operand::Move(Place::Local(b)) => {
                if *b == idx_local
                    || mir_fn.locals.get(b.0).map(|l| &l.ty) != Some(&MirType::Int)
                    || stack_slots.contains_key(&b.0)
                {
                    return None;
                }
                Some(*b)
            }
            _ => return None,
        };
        let (body_start, exit_block) = match &hdr_block.terminator {
            Some(Terminator::Cond {
                cond: Operand::Copy(Place::Local(c)) | Operand::Move(Place::Local(c)),
                then_block,
                else_block,
                ..
            }) if *c == cond_local && then_block.0 != hdr && else_block.0 != hdr => {
                (then_block.0, *else_block)
            }
            _ => return None,
        };

        // Walk single-path body blocks back to `hdr`, finding exactly one inline queue op.
        let mut path_blocks: Vec<usize> = Vec::new();
        let mut call_pos: Option<usize> = None;
        let mut cur = body_start;
        while cur != hdr {
            if path_blocks.len() >= 16
                || path_blocks.contains(&cur)
                || loop_headers.contains(&cur)
            {
                return None;
            }
            let pos = path_blocks.len();
            path_blocks.push(cur);
            match &mir_fn.blocks[cur].terminator {
                Some(Terminator::Goto(next, _)) => cur = next.0,
                Some(Terminator::Call {
                    callee,
                    success,
                    error: None,
                    ..
                }) if matches!(callee.as_str(), "queue.offer_int" | "queue.take_int")
                    && call_pos.is_none() =>
                {
                    call_pos = Some(pos);
                    cur = success.0;
                }
                _ => return None,
            }
        }
        let call_pos = call_pos?;
        let call_b = path_blocks[call_pos];
        let Some(Terminator::Call {
            callee,
            args,
            destination,
            ..
        }) = &mir_fn.blocks[call_b].terminator
        else {
            return None;
        };
        let q_loc = match args.first() {
            Some(Operand::Copy(Place::Local(ql)) | Operand::Move(Place::Local(ql))) => ql.0,
            _ => return None,
        };
        let &(_, cap, _) = inline_local_queues.get(&q_loc)?;
        if cap < 8 {
            return None;
        }

        // Collect all locals read outside `path_blocks` so we only elide dead temporaries.
        let mut read_outside_path: hashbrown::HashSet<usize> = hashbrown::HashSet::new();
        let mark_op_outside = |op: &Operand, out: &mut hashbrown::HashSet<usize>| {
            if let Operand::Copy(p) | Operand::Move(p) = op {
                out.insert(p.root_local().0);
                if let Place::Index(_, idx_l) = p {
                    out.insert(idx_l.0);
                }
            }
        };
        for (bi, ob) in mir_fn.blocks.iter().enumerate() {
            if path_blocks.contains(&bi) {
                continue;
            }
            for s in &ob.statements {
                if let Statement::Assign(dst, rv, _) = s {
                    if !matches!(dst, Place::Local(_)) {
                        read_outside_path.insert(dst.root_local().0);
                    }
                    if let Place::Index(_, idx_l) = dst {
                        read_outside_path.insert(idx_l.0);
                    }
                    match rv {
                        Rvalue::Use(op) | Rvalue::UnaryOp(_, op) | Rvalue::Cast(op, _) => {
                            mark_op_outside(op, &mut read_outside_path);
                        }
                        Rvalue::BinaryOp(_, l, r) => {
                            mark_op_outside(l, &mut read_outside_path);
                            mark_op_outside(r, &mut read_outside_path);
                        }
                        Rvalue::Aggregate(_, ops) => {
                            for op in ops {
                                mark_op_outside(op, &mut read_outside_path);
                            }
                        }
                        Rvalue::Closure { captures, .. } => {
                            for (_, op, _) in captures {
                                mark_op_outside(op, &mut read_outside_path);
                            }
                        }
                    }
                }
            }
            match &ob.terminator {
                Some(Terminator::Call {
                    args: c_args,
                    destination: c_dst,
                    ..
                }) => {
                    if !matches!(c_dst, Place::Local(_)) {
                        read_outside_path.insert(c_dst.root_local().0);
                    }
                    for a in c_args {
                        mark_op_outside(a, &mut read_outside_path);
                    }
                }
                Some(Terminator::Cond { cond, .. }) => mark_op_outside(cond, &mut read_outside_path),
                Some(Terminator::Switch { discr, .. }) => {
                    mark_op_outside(discr, &mut read_outside_path);
                }
                Some(Terminator::Return(Some(op), _)) | Some(Terminator::Raise(op, _)) => {
                    mark_op_outside(op, &mut read_outside_path);
                }
                _ => {}
            }
        }

        let resolve_body_const =
            |op: &Operand, body_consts: &hashbrown::HashMap<usize, i64>| -> Option<i64> {
                match op {
                    Operand::Constant(MirConstant::Int(v)) => Some(*v),
                    Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l)) => body_consts
                        .get(&l.0)
                        .copied()
                        .or_else(|| const_int_locals.get(&l.0).copied()),
                    _ => None,
                }
            };
        let op_local = |op: &Operand| -> Option<LocalId> {
            match op {
                Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l)) => Some(*l),
                _ => None,
            }
        };

        if callee == "queue.offer_int" {
            if args.len() != 2 {
                return None;
            }
            let Place::Local(offer_dst_loc) = destination else {
                return None;
            };
            if *offer_dst_loc == idx_local
                || bound_local == Some(*offer_dst_loc)
                || stack_slots.contains_key(&offer_dst_loc.0)
            {
                return None;
            }
            let mut pre_stmts: Vec<(usize, usize)> = Vec::new();
            let mut written_pre: hashbrown::HashSet<usize> = hashbrown::HashSet::new();
            let mut read_before_write: hashbrown::HashSet<usize> = hashbrown::HashSet::new();
            let mut depends_on_idx: hashbrown::HashSet<usize> = hashbrown::HashSet::new();
            let mut linear_plus1: hashbrown::HashSet<usize> = hashbrown::HashSet::new();
            depends_on_idx.insert(idx_local.0);
            linear_plus1.insert(idx_local.0);

            for &pb in &path_blocks[..=call_pos] {
                for (si, stmt) in mir_fn.blocks[pb].statements.iter().enumerate() {
                    match stmt {
                        Statement::StorageLive(..) | Statement::StorageDead(..) => {}
                        Statement::Assign(Place::Local(dst), rv, _) => {
                            if *dst == idx_local
                                || bound_local == Some(*dst)
                                || dst.0 == q_loc
                                || stack_slots.contains_key(&dst.0)
                                || read_outside_path.contains(&dst.0)
                                || mir_fn.locals.get(dst.0).map(|l| &l.ty) != Some(&MirType::Int)
                            {
                                return None;
                            }
                            let mark_read = |op: &Operand,
                                             wp: &hashbrown::HashSet<usize>,
                                             rbw: &mut hashbrown::HashSet<usize>| {
                                if let Some(l) = op_local(op)
                                    && !wp.contains(&l.0)
                                {
                                    rbw.insert(l.0);
                                }
                            };
                            match rv {
                                Rvalue::Use(op) => {
                                    mark_read(op, &written_pre, &mut read_before_write);
                                    if let Some(l) = op_local(op) {
                                        if linear_plus1.contains(&l.0) {
                                            linear_plus1.insert(dst.0);
                                            depends_on_idx.insert(dst.0);
                                        } else if depends_on_idx.contains(&l.0) {
                                            depends_on_idx.insert(dst.0);
                                            linear_plus1.remove(&dst.0);
                                        } else {
                                            depends_on_idx.remove(&dst.0);
                                            linear_plus1.remove(&dst.0);
                                        }
                                    } else if !matches!(op, Operand::Constant(MirConstant::Int(_))) {
                                        return None;
                                    } else {
                                        depends_on_idx.remove(&dst.0);
                                        linear_plus1.remove(&dst.0);
                                    }
                                }
                                Rvalue::BinaryOp(op_kind, l, r)
                                    if matches!(
                                        op_kind,
                                        MirBinaryOp::Add
                                            | MirBinaryOp::Sub
                                            | MirBinaryOp::Mul
                                            | MirBinaryOp::BitAnd
                                            | MirBinaryOp::BitOr
                                            | MirBinaryOp::BitXor
                                            | MirBinaryOp::Shl
                                            | MirBinaryOp::Shr
                                    ) =>
                                {
                                    mark_read(l, &written_pre, &mut read_before_write);
                                    mark_read(r, &written_pre, &mut read_before_write);
                                    let l_dep =
                                        op_local(l).is_some_and(|x| depends_on_idx.contains(&x.0));
                                    let r_dep =
                                        op_local(r).is_some_and(|x| depends_on_idx.contains(&x.0));
                                    let l_lin =
                                        op_local(l).is_some_and(|x| linear_plus1.contains(&x.0));
                                    let r_lin =
                                        op_local(r).is_some_and(|x| linear_plus1.contains(&x.0));
                                    if *op_kind == MirBinaryOp::Add && ((l_lin && !r_dep) || (r_lin && !l_dep)) {
                                        linear_plus1.insert(dst.0);
                                        depends_on_idx.insert(dst.0);
                                    } else if *op_kind == MirBinaryOp::Sub && l_lin && !r_dep {
                                        linear_plus1.insert(dst.0);
                                        depends_on_idx.insert(dst.0);
                                    } else if l_dep || r_dep {
                                        depends_on_idx.insert(dst.0);
                                        linear_plus1.remove(&dst.0);
                                    } else {
                                        depends_on_idx.remove(&dst.0);
                                        linear_plus1.remove(&dst.0);
                                    }
                                }
                                _ => return None,
                            }
                            written_pre.insert(dst.0);
                            pre_stmts.push((pb, si));
                        }
                        _ => return None,
                    }
                }
            }
            if written_pre.iter().any(|w| read_before_write.contains(w)) {
                return None;
            }
            let val_local = op_local(&args[1])?;
            if !linear_plus1.contains(&val_local.0) {
                return None;
            }

            // Post-call blocks: only `idx_local = idx_local + 1` (direct or 1-hop temp).
            let mut body_consts: hashbrown::HashMap<usize, i64> = hashbrown::HashMap::new();
            let mut idx_plus_one_tmps: hashbrown::HashSet<usize> = hashbrown::HashSet::new();
            let mut idx_updates = 0usize;
            for &pb in &path_blocks[(call_pos + 1)..] {
                for stmt in &mir_fn.blocks[pb].statements {
                    match stmt {
                        Statement::StorageLive(..) | Statement::StorageDead(..) => {}
                        Statement::Assign(Place::Local(dst), rv, _) => {
                            if *dst == idx_local {
                                let is_inc = match rv {
                                    Rvalue::BinaryOp(MirBinaryOp::Add, l, r) => {
                                        (op_local(l) == Some(idx_local)
                                            && resolve_body_const(r, &body_consts) == Some(1))
                                            || (op_local(r) == Some(idx_local)
                                                && resolve_body_const(l, &body_consts) == Some(1))
                                    }
                                    Rvalue::Use(op) => {
                                        op_local(op).is_some_and(|t| idx_plus_one_tmps.contains(&t.0))
                                    }
                                    _ => false,
                                };
                                if !is_inc {
                                    return None;
                                }
                                idx_updates += 1;
                                idx_plus_one_tmps.clear();
                            } else {
                                if bound_local == Some(*dst)
                                    || dst.0 == q_loc
                                    || stack_slots.contains_key(&dst.0)
                                    || read_outside_path.contains(&dst.0)
                                {
                                    return None;
                                }
                                match rv {
                                    Rvalue::Use(Operand::Constant(MirConstant::Int(v))) => {
                                        body_consts.insert(dst.0, *v);
                                    }
                                    Rvalue::BinaryOp(MirBinaryOp::Add, l, r)
                                        if idx_updates == 0
                                            && ((op_local(l) == Some(idx_local)
                                                && resolve_body_const(r, &body_consts) == Some(1))
                                                || (op_local(r) == Some(idx_local)
                                                    && resolve_body_const(l, &body_consts)
                                                        == Some(1))) =>
                                    {
                                        idx_plus_one_tmps.insert(dst.0);
                                    }
                                    _ => return None,
                                }
                            }
                        }
                        _ => return None,
                    }
                }
            }
            if idx_updates != 1 {
                return None;
            }
            let offer_dst = if read_outside_path.contains(&offer_dst_loc.0) {
                Some(*offer_dst_loc)
            } else {
                None
            };
            return Some(VecQueueLoopSpec {
                cond_local,
                idx_local,
                bound_op,
                exit_block,
                kind: VecQueueLoopKind::OfferLinear {
                    q_loc,
                    offer_dst,
                    pre_stmts,
                    val_op: args[1].clone(),
                },
            });
        }

        if callee == "queue.take_int" {
            if args.len() != 1 {
                return None;
            }
            let Place::Local(take_dst_loc) = destination else {
                return None;
            };
            if *take_dst_loc == idx_local
                || bound_local == Some(*take_dst_loc)
                || stack_slots.contains_key(&take_dst_loc.0)
                || read_outside_path.contains(&take_dst_loc.0)
            {
                return None;
            }
            // Pre-call blocks must be side-effect/assignment free.
            for &pb in &path_blocks[..=call_pos] {
                for stmt in &mir_fn.blocks[pb].statements {
                    if !matches!(stmt, Statement::StorageLive(..) | Statement::StorageDead(..)) {
                        return None;
                    }
                }
            }

            let mut msg_aliases: hashbrown::HashSet<usize> = hashbrown::HashSet::new();
            msg_aliases.insert(take_dst_loc.0);
            let mut body_consts: hashbrown::HashMap<usize, i64> = hashbrown::HashMap::new();
            let mut sum_defs: hashbrown::HashMap<usize, LocalId> = hashbrown::HashMap::new();
            let mut rem_defs: hashbrown::HashMap<usize, (LocalId, i64)> = hashbrown::HashMap::new();
            let mut idx_plus_one_tmps: hashbrown::HashSet<usize> = hashbrown::HashSet::new();
            let mut acc_result: Option<(LocalId, i64)> = None;
            let mut idx_updates = 0usize;

            for &pb in &path_blocks[(call_pos + 1)..] {
                for stmt in &mir_fn.blocks[pb].statements {
                    match stmt {
                        Statement::StorageLive(..) | Statement::StorageDead(..) => {}
                        Statement::Assign(Place::Local(dst), rv, _) => {
                            if *dst == idx_local {
                                let is_inc = match rv {
                                    Rvalue::BinaryOp(MirBinaryOp::Add, l, r) => {
                                        (op_local(l) == Some(idx_local)
                                            && resolve_body_const(r, &body_consts) == Some(1))
                                            || (op_local(r) == Some(idx_local)
                                                && resolve_body_const(l, &body_consts) == Some(1))
                                    }
                                    Rvalue::Use(op) => {
                                        op_local(op).is_some_and(|t| idx_plus_one_tmps.contains(&t.0))
                                    }
                                    _ => false,
                                };
                                if !is_inc {
                                    return None;
                                }
                                idx_updates += 1;
                                idx_plus_one_tmps.clear();
                                continue;
                            }
                            if bound_local == Some(*dst)
                                || dst.0 == q_loc
                                || stack_slots.contains_key(&dst.0)
                            {
                                return None;
                            }
                            match rv {
                                Rvalue::Use(Operand::Constant(MirConstant::Int(v))) => {
                                    if read_outside_path.contains(&dst.0) {
                                        return None;
                                    }
                                    body_consts.insert(dst.0, *v);
                                }
                                Rvalue::Use(op)
                                    if op_local(op).is_some_and(|t| msg_aliases.contains(&t.0)) =>
                                {
                                    if read_outside_path.contains(&dst.0) {
                                        return None;
                                    }
                                    msg_aliases.insert(dst.0);
                                }
                                Rvalue::Use(op)
                                    if op_local(op)
                                        .and_then(|t| rem_defs.get(&t.0).copied())
                                        .is_some_and(|(acc, _)| acc == *dst) =>
                                {
                                    let (acc, mod_c) = rem_defs[&op_local(op).unwrap().0];
                                    if acc_result.is_some() {
                                        return None;
                                    }
                                    acc_result = Some((acc, mod_c));
                                }
                                Rvalue::BinaryOp(MirBinaryOp::Add, l, r) => {
                                    let is_idx_inc_tmp = idx_updates == 0
                                        && !read_outside_path.contains(&dst.0)
                                        && ((op_local(l) == Some(idx_local)
                                            && resolve_body_const(r, &body_consts) == Some(1))
                                            || (op_local(r) == Some(idx_local)
                                                && resolve_body_const(l, &body_consts) == Some(1)));
                                    if is_idx_inc_tmp {
                                        idx_plus_one_tmps.insert(dst.0);
                                    } else {
                                        let acc_opt = match (op_local(l), op_local(r)) {
                                            (Some(a), Some(m))
                                                if msg_aliases.contains(&m.0)
                                                    && !msg_aliases.contains(&a.0) =>
                                            {
                                                Some(a)
                                            }
                                            (Some(m), Some(a))
                                                if msg_aliases.contains(&m.0)
                                                    && !msg_aliases.contains(&a.0) =>
                                            {
                                                Some(a)
                                            }
                                            _ => None,
                                        };
                                        let acc = acc_opt?;
                                        if acc == idx_local
                                            || bound_local == Some(acc)
                                            || acc.0 == q_loc
                                            || stack_slots.contains_key(&acc.0)
                                            || mir_fn.locals.get(acc.0).map(|d| &d.ty)
                                                != Some(&MirType::Int)
                                            || read_outside_path.contains(&dst.0)
                                        {
                                            return None;
                                        }
                                        sum_defs.insert(dst.0, acc);
                                    }
                                }
                                Rvalue::BinaryOp(MirBinaryOp::Mod, l, r) => {
                                    let sum_l = op_local(l)?;
                                    let &acc = sum_defs.get(&sum_l.0)?;
                                    let mod_c = resolve_body_const(r, &body_consts)?;
                                    if !(1_000_000..(1i64 << 57)).contains(&mod_c) {
                                        return None;
                                    }
                                    if *dst == acc {
                                        if acc_result.is_some() {
                                            return None;
                                        }
                                        acc_result = Some((acc, mod_c));
                                    } else {
                                        if read_outside_path.contains(&dst.0) {
                                            return None;
                                        }
                                        rem_defs.insert(dst.0, (acc, mod_c));
                                    }
                                }
                                _ => return None,
                            }
                        }
                        _ => return None,
                    }
                }
            }
            let (acc_local, mod_const) = acc_result?;
            if idx_updates != 1 {
                return None;
            }
            return Some(VecQueueLoopSpec {
                cond_local,
                idx_local,
                bound_op,
                exit_block,
                kind: VecQueueLoopKind::TakeAddMod {
                    q_loc,
                    acc_local,
                    mod_const,
                },
            });
        }

        None
    }

    /// Identifies non-escaping `string.sub(base, s, e)` locals (with optional 1-hop copy) that are
    /// dead at all loop headers and read ONLY by single-byte `string.byte(chunk, k, k)` calls
    /// while `base` is never mutated, enabling zero-allocation `lust_mir_clif_string_sub_byte_array`
    /// fusion (`BENCH-NG-09`).
    fn compute_fused_string_sub_locals(
        mir_fn: &MirFunction,
        loop_headers: &hashbrown::HashSet<usize>,
        osr_info: Option<(BlockId, &[LocalId])>,
        stack_slots: &hashbrown::HashMap<usize, StackSlot>,
    ) -> hashbrown::HashMap<usize, usize> {
        // Returns map: `sub_local -> alloc_tmp_local`
        let mut excluded: hashbrown::HashSet<usize> = stack_slots.keys().copied().collect();
        for &p in &mir_fn.params {
            excluded.insert(p.0);
        }
        if let Some((_, live)) = osr_info {
            for &l in live {
                excluded.insert(l.0);
            }
        }
        for &hdr in loop_headers {
            if let Ok(live) = mir_fn.derive_proven_site_metadata_at_entry(hdr) {
                for l in live {
                    excluded.insert(l);
                }
            }
        }
        let mut def_count = vec![0usize; mir_fn.locals.len()];
        let mut copy_dst_of_tmp: hashbrown::HashMap<usize, usize> = hashbrown::HashMap::new();
        let mut tmp_use_count = vec![0usize; mir_fn.locals.len()];
        let mut sub_defs: Vec<(usize, usize)> = Vec::new(); // (tmp_loc, base_loc)
        let mut base_mutated_in_loop_blocks: hashbrown::HashSet<usize> = hashbrown::HashSet::new();

        for (b_idx, block) in mir_fn.blocks.iter().enumerate() {
            for stmt in &block.statements {
                if let Statement::Assign(Place::Local(dst), rval, _) = stmt {
                    if dst.0 < def_count.len() {
                        def_count[dst.0] += 1;
                    }
                    if let Rvalue::Use(
                        Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src)),
                    ) = rval
                    {
                        if src.0 < tmp_use_count.len() {
                            tmp_use_count[src.0] += 1;
                        }
                        copy_dst_of_tmp.insert(src.0, dst.0);
                    }
                }
            }
            if let Some(Terminator::Call {
                callee,
                args,
                destination: Place::Local(dst),
                error: None,
                ..
            }) = &block.terminator
            {
                if dst.0 < def_count.len() {
                    def_count[dst.0] += 1;
                }
                if callee == "string.sub"
                    && (args.len() == 2 || args.len() == 3)
                    && let Operand::Copy(Place::Local(base_loc))
                    | Operand::Move(Place::Local(base_loc)) = &args[0]
                {
                    sub_defs.push((dst.0, base_loc.0));
                }
            }
            let _ = b_idx;
        }
        let _ = &mut base_mutated_in_loop_blocks;

        let mut candidates: hashbrown::HashMap<usize, (usize, usize)> = hashbrown::HashMap::new();
        for (tmp_loc, base_loc) in sub_defs {
            if excluded.contains(&tmp_loc) || def_count[tmp_loc] != 1 {
                continue;
            }
            let target_loc = if tmp_use_count[tmp_loc] == 1
                && let Some(&dst_loc) = copy_dst_of_tmp.get(&tmp_loc)
                && !excluded.contains(&dst_loc)
                && def_count[dst_loc] == 1
            {
                dst_loc
            } else if tmp_use_count[tmp_loc] == 0 {
                tmp_loc
            } else {
                continue;
            };
            candidates.insert(target_loc, (tmp_loc, base_loc));
        }
        if candidates.is_empty() {
            return hashbrown::HashMap::new();
        }

        // Verify every use of `sub_local` (and `tmp_local`) is strictly `sub = Move(tmp)` or
        // `string.byte(sub, k, k)` with `Array<int>` destination!
        let mut bad: hashbrown::HashSet<usize> = hashbrown::HashSet::new();
        let mut tmp_to_sub: hashbrown::HashMap<usize, usize> = hashbrown::HashMap::new();
        let mut byte_use_count: hashbrown::HashMap<usize, usize> = hashbrown::HashMap::new();
        for (&sub_loc, &(tmp_loc, _)) in &candidates {
            tmp_to_sub.insert(tmp_loc, sub_loc);
        }
        let mut mark_bad = |loc_id: usize| {
            if candidates.contains_key(&loc_id) {
                bad.insert(loc_id);
            }
            if let Some(&sub_loc) = tmp_to_sub.get(&loc_id) {
                bad.insert(sub_loc);
            }
        };
        let check_op = |op: &Operand, mark: &mut dyn FnMut(usize)| {
            if let Operand::Copy(p) | Operand::Move(p) = op {
                mark(p.root_local().0);
                if let Place::Index(_, idx) = p {
                    mark(idx.0);
                }
            }
        };
        for block in &mir_fn.blocks {
            for stmt in &block.statements {
                if let Statement::Assign(lhs, rval, _) = stmt {
                    let is_init_copy = matches!(
                        (lhs, rval),
                        (
                            Place::Local(dst),
                            Rvalue::Use(
                                Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src))
                            )
                        ) if candidates.get(&dst.0).is_some_and(|&(t, _)| t == src.0)
                    );
                    if is_init_copy {
                        continue;
                    }
                    if !matches!(lhs, Place::Local(_)) {
                        mark_bad(lhs.root_local().0);
                    }
                    match rval {
                        Rvalue::Use(op) | Rvalue::Cast(op, _) | Rvalue::UnaryOp(_, op) => {
                            check_op(op, &mut mark_bad);
                        }
                        Rvalue::BinaryOp(_, l, r) => {
                            check_op(l, &mut mark_bad);
                            check_op(r, &mut mark_bad);
                        }
                        Rvalue::Aggregate(_, ops) => {
                            for op in ops {
                                check_op(op, &mut mark_bad);
                            }
                        }
                        Rvalue::Closure { captures, .. } => {
                            for (_, op, _) in captures {
                                check_op(op, &mut mark_bad);
                            }
                        }
                    }
                }
            }
            match &block.terminator {
                Some(Terminator::Call {
                    callee,
                    args,
                    destination,
                    error,
                    ..
                }) => {
                    if !matches!(destination, Place::Local(_)) {
                        mark_bad(destination.root_local().0);
                    }
                    let is_single_byte = callee == "string.byte"
                        && error.is_none()
                        && (args.len() == 2 || (args.len() == 3 && args[1] == args[2]))
                        && matches!(
                            destination,
                            Place::Local(dl)
                                if matches!(
                                    mir_fn.locals.get(dl.0).map(|d| &d.ty),
                                    Some(MirType::Array(inner)) if **inner == MirType::Int
                                )
                        );
                    for (idx, a) in args.iter().enumerate() {
                        if is_single_byte && idx == 0 {
                            if let Operand::Copy(Place::Local(sl)) | Operand::Move(Place::Local(sl)) = a
                                && candidates.contains_key(&sl.0)
                            {
                                *byte_use_count.entry(sl.0).or_insert(0) += 1;
                            }
                            continue;
                        }
                        check_op(a, &mut mark_bad);
                    }
                }
                Some(Terminator::Cond { cond, .. }) => check_op(cond, &mut mark_bad),
                Some(Terminator::Switch { discr, .. }) => check_op(discr, &mut mark_bad),
                Some(Terminator::Return(Some(op), _)) | Some(Terminator::Raise(op, _)) => {
                    check_op(op, &mut mark_bad);
                }
                _ => {}
            }
        }

        let mut out = hashbrown::HashMap::new();
        for (sub_loc, (tmp_loc, _base_loc)) in candidates {
            if !bad.contains(&sub_loc) && byte_use_count.get(&sub_loc).copied().unwrap_or(0) >= 1 {
                out.insert(sub_loc, tmp_loc);
            }
        }
        out
    }

    struct PromotedLoopCfgRecord {
        hdr: usize,
        promote_gas: bool,
        promote_sp: bool,
        preheader_bb: Block,
        loop_clif_blocks: hashbrown::HashSet<Block>,
        flush_clif_blocks: hashbrown::HashSet<Block>,
        post_flush_clif_blocks: hashbrown::HashSet<Block>,
        reload_clif_blocks: hashbrown::HashSet<Block>,
    }

    struct InnerQuantizedCfgRecord {
        hdr: usize,
        loop_clif_blocks: hashbrown::HashSet<Block>,
        reconcile_clif_blocks: hashbrown::HashSet<Block>,
        post_flush_clif_blocks: hashbrown::HashSet<Block>,
        reload_clif_blocks: hashbrown::HashSet<Block>,
    }

    fn emit_promoted_counters_load(
        builder: &mut FunctionBuilder,
        counters: PromotedLoopCounters,
    ) {
        const OFF_GAS_REMAINING: i32 = core::mem::offset_of!(GasFastState, gas_remaining) as i32;
        const OFF_SAFEPOINTS_CHECKED: i32 =
            core::mem::offset_of!(SafepointFastState, safepoints_checked) as i32;

        if let Some((gas_ptr_var, gas_var)) = counters.gas {
            let gas_ptr = builder.use_var(gas_ptr_var);
            let g = builder
                .ins()
                .load(types::I64, MemFlagsData::trusted(), gas_ptr, OFF_GAS_REMAINING);
            builder.def_var(gas_var, g);
        }
        if let Some((sp_ptr_var, sp_cnt_var)) = counters.sp {
            let sp_ptr = builder.use_var(sp_ptr_var);
            let s = builder
                .ins()
                .load(types::I64, MemFlagsData::trusted(), sp_ptr, OFF_SAFEPOINTS_CHECKED);
            builder.def_var(sp_cnt_var, s);
        }
        if let Some((quantum_rem_var, _)) = counters.quantum {
            let zero = builder.ins().iconst(types::I64, 0);
            builder.def_var(quantum_rem_var, zero);
        }
    }

    /// NG-W14-1: Reconciles any unused pre-reserved `Q=64` quantum steps back into
    /// register-promoted `(gas_var, sp_cnt_var)` with zero memory stores and resets `quantum_rem = 0`.
    /// Invariant inside quantized loop:
    ///   `gas_var == actual_gas_remaining - quantum_rem * amt`
    ///   `sp_cnt_var == actual_safepoints_checked + quantum_rem`
    fn emit_quantum_reconcile_to_regs(
        builder: &mut FunctionBuilder,
        counters: PromotedLoopCounters,
    ) {
        if let Some((quantum_rem_var, amt)) = counters.quantum {
            let q = builder.use_var(quantum_rem_var);
            if let Some((_, gas_var)) = counters.gas {
                let g = builder.use_var(gas_var);
                let refund = if amt == 1 {
                    q
                } else {
                    let amt_val = builder.ins().iconst(types::I64, amt as i64);
                    builder.ins().imul(q, amt_val)
                };
                let g_rec = builder.ins().iadd(g, refund);
                builder.def_var(gas_var, g_rec);
            }
            if let Some((_, sp_cnt_var)) = counters.sp {
                let s = builder.use_var(sp_cnt_var);
                let s_rec = builder.ins().isub(s, q);
                builder.def_var(sp_cnt_var, s_rec);
            }
            let zero = builder.ins().iconst(types::I64, 0);
            builder.def_var(quantum_rem_var, zero);
        }
    }

    /// Flushes `(gas_var, sp_cnt_var)` directly to memory when `quantum_rem == 0` is statically guaranteed.
    fn emit_promoted_counters_flush_exact(
        builder: &mut FunctionBuilder,
        counters: PromotedLoopCounters,
    ) {
        const OFF_GAS_REMAINING: i32 = core::mem::offset_of!(GasFastState, gas_remaining) as i32;
        const OFF_SAFEPOINTS_CHECKED: i32 =
            core::mem::offset_of!(SafepointFastState, safepoints_checked) as i32;

        if let Some((gas_ptr_var, gas_var)) = counters.gas {
            let gas_ptr = builder.use_var(gas_ptr_var);
            let g = builder.use_var(gas_var);
            builder
                .ins()
                .store(MemFlagsData::trusted(), g, gas_ptr, OFF_GAS_REMAINING);
        }
        if let Some((sp_ptr_var, sp_cnt_var)) = counters.sp {
            let sp_ptr = builder.use_var(sp_ptr_var);
            let s = builder.use_var(sp_cnt_var);
            builder
                .ins()
                .store(MemFlagsData::trusted(), s, sp_ptr, OFF_SAFEPOINTS_CHECKED);
        }
    }

    fn emit_promoted_counters_flush(
        builder: &mut FunctionBuilder,
        counters: PromotedLoopCounters,
    ) {
        emit_quantum_reconcile_to_regs(builder, counters);
        emit_promoted_counters_flush_exact(builder, counters);
    }

    /// Barrier 5 (`PERF-T4-2` + `NG-W14-1`): Mechanically verifies on the emitted Cranelift IR CFG that:
    /// 1. Every CFG edge leaving a promoted loop's interior block set routes strictly through
    ///    an authorized flush block (and no interior block executes `return`).
    /// 2. Every authorized flush block stores all promoted counters BEFORE any call or exit.
    /// 3. Every reload block loads all promoted counters BEFORE re-entering the loop.
    /// 4. Every preheader block loads all promoted counters.
    /// 5. Every CFG edge leaving an inner quantized loop's interior block set routes strictly
    ///    through an authorized reconcile/flush block (and no interior block executes `return`).
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    fn verify_promoted_loop_exit_completeness(
        func: &cranelift_codegen::ir::Function,
        records: &[PromotedLoopCfgRecord],
        iq_records: &[InnerQuantizedCfgRecord],
    ) -> Result<(), MirCraneliftBackendError> {
        use cranelift_codegen::ir::InstructionData;

        const OFF_GAS_REMAINING: i32 = core::mem::offset_of!(GasFastState, gas_remaining) as i32;
        const OFF_SAFEPOINTS_CHECKED: i32 =
            core::mem::offset_of!(SafepointFastState, safepoints_checked) as i32;

        for rec in records {
            // 1. Preheader must load all promoted counters.
            {
                let mut loaded_gas = !rec.promote_gas;
                let mut loaded_sp = !rec.promote_sp;
                for inst in func.layout.block_insts(rec.preheader_bb) {
                    if let InstructionData::Load { offset, .. } = func.dfg.insts[inst] {
                        let off = i32::from(offset);
                        if off == OFF_GAS_REMAINING {
                            loaded_gas = true;
                        }
                        if off == OFF_SAFEPOINTS_CHECKED {
                            loaded_sp = true;
                        }
                    }
                }
                if !loaded_gas || !loaded_sp {
                    return Err(MirCraneliftBackendError::CodegenError(format!(
                        "PERF-T4-2 CFG verifier: preheader {:?} for loop hdr {} missing counter load (gas={}, sp={})",
                        rec.preheader_bb, rec.hdr, loaded_gas, loaded_sp
                    )));
                }
            }

            // 2. Every interior block's outgoing edges must stay inside interior or land in flush_clif_blocks.
            for &bb in &rec.loop_clif_blocks {
                if !func.layout.is_block_inserted(bb) {
                    continue;
                }
                for inst in func.layout.block_insts(bb) {
                    let data = &func.dfg.insts[inst];
                    if data.opcode().is_return() {
                        return Err(MirCraneliftBackendError::CodegenError(format!(
                            "PERF-T4-2 CFG verifier: interior block {:?} of loop hdr {} executes return without flush block",
                            bb, rec.hdr
                        )));
                    }
                    for bcall in data.branch_destination(&func.dfg.jump_tables, &func.dfg.exception_tables) {
                        let dst = bcall.block(&func.dfg.value_lists);
                        if !rec.loop_clif_blocks.contains(&dst) && !rec.flush_clif_blocks.contains(&dst) {
                            return Err(MirCraneliftBackendError::CodegenError(format!(
                                "PERF-T4-2 CFG verifier: leaked exit edge {:?} -> {:?} from promoted loop hdr {} bypassing flush blocks",
                                bb, dst, rec.hdr
                            )));
                        }
                    }
                }
            }

            // 3. Every flush block must store all promoted counters BEFORE its first call/branch/return.
            for &fbb in &rec.flush_clif_blocks {
                if !func.layout.is_block_inserted(fbb) {
                    continue;
                }
                let mut stored_gas = !rec.promote_gas;
                let mut stored_sp = !rec.promote_sp;
                for inst in func.layout.block_insts(fbb) {
                    let data = &func.dfg.insts[inst];
                    if data.opcode().is_call() || data.opcode().is_branch() || data.opcode().is_return() {
                        break;
                    }
                    if let InstructionData::Store { offset, .. } = *data {
                        let off = i32::from(offset);
                        if off == OFF_GAS_REMAINING {
                            stored_gas = true;
                        }
                        if off == OFF_SAFEPOINTS_CHECKED {
                            stored_sp = true;
                        }
                    }
                }
                if !stored_gas || !stored_sp {
                    return Err(MirCraneliftBackendError::CodegenError(format!(
                        "PERF-T4-2 CFG verifier: flush block {:?} for loop hdr {} missing counter store before call/exit (gas={}, sp={})",
                        fbb, rec.hdr, stored_gas, stored_sp
                    )));
                }
            }

            // 4. Every reload block must load all promoted counters before jumping back into loop.
            for &rbb in &rec.reload_clif_blocks {
                if !func.layout.is_block_inserted(rbb) {
                    continue;
                }
                let mut loaded_gas = !rec.promote_gas;
                let mut loaded_sp = !rec.promote_sp;
                for inst in func.layout.block_insts(rbb) {
                    if let InstructionData::Load { offset, .. } = func.dfg.insts[inst] {
                        let off = i32::from(offset);
                        if off == OFF_GAS_REMAINING {
                            loaded_gas = true;
                        }
                        if off == OFF_SAFEPOINTS_CHECKED {
                            loaded_sp = true;
                        }
                    }
                }
                if !loaded_gas || !loaded_sp {
                    return Err(MirCraneliftBackendError::CodegenError(format!(
                        "PERF-T4-2 CFG verifier: reload block {:?} for loop hdr {} missing counter reload (gas={}, sp={})",
                        rbb, rec.hdr, loaded_gas, loaded_sp
                    )));
                }
            }

            PROMOTED_LOOP_CFG_VERIFICATIONS.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        }

        for iq_rec in iq_records {
            for &bb in &iq_rec.loop_clif_blocks {
                if !func.layout.is_block_inserted(bb) {
                    continue;
                }
                for inst in func.layout.block_insts(bb) {
                    let data = &func.dfg.insts[inst];
                    if data.opcode().is_return() {
                        return Err(MirCraneliftBackendError::CodegenError(format!(
                            "NG-W14-1 CFG verifier: interior block {:?} of inner quantized loop hdr {} executes return without reconcile/flush block",
                            bb, iq_rec.hdr
                        )));
                    }
                    for bcall in data.branch_destination(&func.dfg.jump_tables, &func.dfg.exception_tables) {
                        let dst = bcall.block(&func.dfg.value_lists);
                        if !iq_rec.loop_clif_blocks.contains(&dst)
                            && !iq_rec.reconcile_clif_blocks.contains(&dst)
                        {
                            return Err(MirCraneliftBackendError::CodegenError(format!(
                                "NG-W14-1 CFG verifier: leaked exit edge {:?} -> {:?} from inner quantized loop hdr {} bypassing reconcile/flush blocks",
                                bb, dst, iq_rec.hdr
                            )));
                        }
                    }
                }
            }
            PROMOTED_LOOP_CFG_VERIFICATIONS.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        }

        Ok(())
    }

    fn collect_place_hoist_candidates(
        place: &Place,
        locals: &[LocalDecl],
        arrays: &mut Vec<usize>,
        structs: &mut Vec<usize>,
    ) {
        match place {
            Place::Local(_) => {}
            Place::Index(base, _) => {
                if let Place::Local(loc) = base.as_ref() {
                    if matches!(locals.get(loc.0).map(|l| &l.ty), Some(MirType::Array(_)))
                        && !arrays.contains(&loc.0)
                    {
                        arrays.push(loc.0);
                    }
                } else {
                    collect_place_hoist_candidates(base, locals, arrays, structs);
                }
            }
            Place::Field(base, _) => {
                if let Place::Local(loc) = base.as_ref() {
                    if matches!(
                        locals.get(loc.0).map(|l| &l.ty),
                        Some(MirType::Struct(_) | MirType::Tuple(_))
                    ) && !structs.contains(&loc.0)
                    {
                        structs.push(loc.0);
                    }
                } else {
                    collect_place_hoist_candidates(base, locals, arrays, structs);
                }
            }
        }
    }

    fn collect_operand_hoist_candidates(
        op: &Operand,
        locals: &[LocalDecl],
        arrays: &mut Vec<usize>,
        structs: &mut Vec<usize>,
    ) {
        if let Operand::Copy(p) | Operand::Move(p) = op {
            collect_place_hoist_candidates(p, locals, arrays, structs);
        }
    }

    /// NG-W14-5: Identifies function-wide single-definition constant `MirType::Int` locals
    /// whose defining block strictly dominates every read in the function (fail-closed).
    fn compute_const_int_locals(
        mir_fn: &MirFunction,
        osr_info: Option<(BlockId, &[LocalId])>,
        stack_slots: &hashbrown::HashMap<usize, StackSlot>,
    ) -> hashbrown::HashMap<usize, i64> {
        let n = mir_fn.locals.len();
        if n == 0 || mir_fn.blocks.is_empty() {
            return hashbrown::HashMap::new();
        }
        let mut def_state = vec![0u8; n]; // 0 = unseen, 1 = single const int def, 2 = ineligible
        let mut def_val = vec![0i64; n];
        let mut def_loc = vec![(0usize, 0usize); n]; // (block_idx, stmt_idx)

        for (idx, decl) in mir_fn.locals.iter().enumerate() {
            if decl.ty != MirType::Int || stack_slots.contains_key(&idx) {
                def_state[idx] = 2;
            }
        }
        for param in &mir_fn.params {
            if param.0 < n {
                def_state[param.0] = 2;
            }
        }
        if let Some((_, live_locals)) = osr_info {
            for loc in live_locals {
                if loc.0 < n {
                    def_state[loc.0] = 2;
                }
            }
        }

        for (b_idx, block) in mir_fn.blocks.iter().enumerate() {
            for (s_idx, stmt) in block.statements.iter().enumerate() {
                if let Statement::Assign(place, rval, _) = stmt {
                    match (place, rval) {
                        (
                            Place::Local(dst),
                            Rvalue::Use(Operand::Constant(MirConstant::Int(c))),
                        ) => {
                            if dst.0 < n {
                                if def_state[dst.0] == 0 {
                                    def_state[dst.0] = 1;
                                    def_val[dst.0] = *c;
                                    def_loc[dst.0] = (b_idx, s_idx);
                                } else {
                                    def_state[dst.0] = 2;
                                }
                            }
                        }
                        _ => {
                            let root = place.root_local().0;
                            if root < n {
                                def_state[root] = 2;
                            }
                        }
                    }
                }
            }
            if let Some(Terminator::Call { destination, .. }) = &block.terminator {
                let root = destination.root_local().0;
                if root < n {
                    def_state[root] = 2;
                }
            }
        }

        let entry_idx = osr_info
            .map(|(tb, _)| tb.0)
            .unwrap_or(mir_fn.entry.0)
            .min(mir_fn.blocks.len() - 1);

        let mut out = hashbrown::HashMap::new();
        for idx in 0..n {
            if def_state[idx] != 1 {
                continue;
            }
            let (def_bb, def_stmt_idx) = def_loc[idx];

            // 1. Fail closed if `idx` is read before its definition inside `def_bb`.
            let mut read_before_def = false;
            for stmt in &mir_fn.blocks[def_bb].statements[..def_stmt_idx] {
                let mut uses = hashbrown::HashSet::new();
                stmt.collect_uses(&mut uses);
                if uses.contains(&idx) {
                    read_before_def = true;
                    break;
                }
            }
            if read_before_def {
                continue;
            }

            // 2. If not defined in `entry_idx`, verify `def_bb` dominates every block that reads `idx`
            //    by checking whether any block reading `idx` is reachable from `entry_idx` without
            //    passing through `def_bb`.
            if def_bb != entry_idx {
                let mut visited = hashbrown::HashSet::new();
                let mut queue = std::collections::VecDeque::new();
                visited.insert(entry_idx);
                queue.push_back(entry_idx);
                let mut bypassed = false;
                while let Some(b) = queue.pop_front() {
                    let mut uses = hashbrown::HashSet::new();
                    for stmt in &mir_fn.blocks[b].statements {
                        stmt.collect_uses(&mut uses);
                    }
                    if let Some(term) = &mir_fn.blocks[b].terminator {
                        term.collect_uses(&mut uses);
                    }
                    if uses.contains(&idx) {
                        bypassed = true;
                        break;
                    }
                    if let Some(term) = &mir_fn.blocks[b].terminator {
                        for succ in term.successors() {
                            if succ < mir_fn.blocks.len() && succ != def_bb && visited.insert(succ) {
                                queue.push_back(succ);
                            }
                        }
                    }
                }
                if bypassed {
                    continue;
                }
            }

            out.insert(idx, def_val[idx]);
        }
        out
    }

    #[inline]
    fn resolve_const_int_operand(
        op: &Operand,
        const_int_locals: &hashbrown::HashMap<usize, i64>,
    ) -> Option<i64> {
        match op {
            Operand::Constant(MirConstant::Int(c)) => Some(*c),
            Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => {
                const_int_locals.get(&loc.0).copied()
            }
            _ => None,
        }
    }

    fn rvalue_may_fail_non_oob(
        rval: &Rvalue,
        locals: &[LocalDecl],
        validated_at_hdr: &hashbrown::HashSet<usize>,
        const_int_locals: &hashbrown::HashMap<usize, i64>,
    ) -> bool {
        match rval {
            Rvalue::Use(_) | Rvalue::Cast(..) => false,
            Rvalue::BinaryOp(op, lhs, rhs) => match op {
                MirBinaryOp::Concat | MirBinaryOp::FloorDiv | MirBinaryOp::Pow => true,
                MirBinaryOp::Div | MirBinaryOp::Mod => {
                    let is_float = matches!(
                        lhs,
                        Operand::Constant(MirConstant::Float(_))
                    ) || matches!(
                        rhs,
                        Operand::Constant(MirConstant::Float(_))
                    ) || matches!(
                        lhs,
                        Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l))
                            if locals.get(l.0).is_some_and(|d| d.ty == MirType::Float)
                    ) || matches!(
                        rhs,
                        Operand::Copy(Place::Local(r)) | Operand::Move(Place::Local(r))
                            if locals.get(r.0).is_some_and(|d| d.ty == MirType::Float)
                    );
                    if is_float {
                        matches!(op, MirBinaryOp::Mod)
                    } else {
                        !matches!(resolve_const_int_operand(rhs, const_int_locals), Some(c) if c != 0)
                    }
                }
                MirBinaryOp::Add => {
                    matches!(lhs, Operand::Constant(MirConstant::String(_)))
                        || matches!(
                            lhs,
                            Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l))
                                if locals.get(l.0).is_some_and(|d| d.ty == MirType::String)
                        )
                }
                _ => false,
            },
            Rvalue::UnaryOp(MirUnaryOp::Len, op) => match op {
                Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l)) => {
                    matches!(locals.get(l.0).map(|d| &d.ty), Some(MirType::Array(_)))
                        && !validated_at_hdr.contains(&l.0)
                }
                _ => true,
            },
            Rvalue::UnaryOp(..) => false,
            Rvalue::Aggregate(..) | Rvalue::Closure { .. } => true,
        }
    }

    fn is_inline_local_array_push_terminator(
        term: &Option<Terminator>,
        b_idx: usize,
        mir_fn: &MirFunction,
        struct_layouts: &hashbrown::HashMap<String, MirStructLayout>,
        locally_allocated_arrays: &hashbrown::HashSet<usize>,
        def_out: &[hashbrown::HashSet<usize>],
    ) -> bool {
        let Some(Terminator::Call {
            callee,
            args,
            destination,
            ..
        }) = term
        else {
            return false;
        };
        if (callee != "array.push" && callee != "table.push") || args.len() != 2 {
            return false;
        }
        if !matches!(destination, Place::Local(_)) {
            return false;
        }
        let (Operand::Copy(Place::Local(arr_loc)) | Operand::Move(Place::Local(arr_loc))) =
            &args[0]
        else {
            return false;
        };
        if !locally_allocated_arrays.contains(&arr_loc.0) {
            return false;
        }
        if !def_out.get(b_idx).is_some_and(|s| s.contains(&arr_loc.0)) {
            return false;
        }
        let Ok(MirType::Array(elem_ty)) = try_operand_type(&args[0], &mir_fn.locals, struct_layouts)
        else {
            return false;
        };
        if mir_array_elem_tag(&elem_ty).is_none()
            && !matches!(&*elem_ty, MirType::Struct(name) if !name.is_empty())
        {
            return false;
        }
        let Ok(val_ty) = try_operand_type(&args[1], &mir_fn.locals, struct_layouts) else {
            return false;
        };
        val_ty == *elem_ty
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_loop_promotion_plan(
        mir_fn: &MirFunction,
        loop_headers: &hashbrown::HashSet<usize>,
        osr_info: Option<(BlockId, &[LocalId])>,
        vec_pre_map: &hashbrown::HashMap<usize, usize>,
        vec_hdr_map: &hashbrown::HashMap<usize, usize>,
        vec_loops: &[crate::mir::opt::simd::VectorizedLoopInfo],
        prov_plan: &ArrayLenProvenancePlan,
        stack_slots: &hashbrown::HashMap<usize, StackSlot>,
        const_int_locals: &hashbrown::HashMap<usize, i64>,
        struct_layouts: &hashbrown::HashMap<String, MirStructLayout>,
        has_sp_ctx: bool,
        has_gas_ctx: bool,
        inline_queue_blocks: &hashbrown::HashSet<usize>,
    ) -> LoopPromotionPlan {
        let empty = LoopPromotionPlan {
            counter_loops: Vec::new(),
            block_to_counter_loop: hashbrown::HashMap::new(),
            hdr_to_counter_loop: hashbrown::HashMap::new(),
            inner_quantized_loops: Vec::new(),
            block_to_inner_quantized: hashbrown::HashMap::new(),
            hoist_loops: Vec::new(),
            hdr_to_hoist_loop: hashbrown::HashMap::new(),
            inline_push_blocks: hashbrown::HashSet::new(),
        };
        let num_blocks = mir_fn.blocks.len();
        if num_blocks == 0 {
            return empty;
        }
        let entry_idx = osr_info
            .map(|(tb, _)| tb.0)
            .unwrap_or(mir_fn.entry.0)
            .min(num_blocks - 1);

        // 1. Reachability & predecessors over reachable blocks
        let mut reachable = hashbrown::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        reachable.insert(entry_idx);
        queue.push_back(entry_idx);
        while let Some(b) = queue.pop_front() {
            if let Some(term) = &mir_fn.blocks[b].terminator {
                for succ in term.successors() {
                    if succ < num_blocks && reachable.insert(succ) {
                        queue.push_back(succ);
                    }
                }
            }
        }
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];
        for &b in &reachable {
            if let Some(term) = &mir_fn.blocks[b].terminator {
                for succ in term.successors() {
                    if succ < num_blocks && reachable.contains(&succ) && !preds[succ].contains(&b) {
                        preds[succ].push(b);
                    }
                }
            }
        }

        // 2. Exact Dominators over reachable blocks
        let mut dom: Vec<hashbrown::HashSet<usize>> = (0..num_blocks)
            .map(|b| {
                if b == entry_idx {
                    let mut s = hashbrown::HashSet::new();
                    s.insert(entry_idx);
                    s
                } else if reachable.contains(&b) {
                    reachable.clone()
                } else {
                    hashbrown::HashSet::new()
                }
            })
            .collect();
        loop {
            let mut changed = false;
            for b in 0..num_blocks {
                if b == entry_idx || !reachable.contains(&b) || preds[b].is_empty() {
                    continue;
                }
                let mut iter = preds[b].iter();
                let first = *iter.next().unwrap();
                let mut new_dom = dom[first].clone();
                for &p in iter {
                    new_dom.retain(|x| dom[p].contains(x));
                }
                new_dom.insert(b);
                if new_dom != dom[b] {
                    dom[b] = new_dom;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        // 3. True natural loops from dominating CFG backedges (u -> hdr where hdr dom u)
        let mut nat_loops: hashbrown::HashMap<usize, hashbrown::HashSet<usize>> =
            hashbrown::HashMap::new();
        for &u in &reachable {
            if let Some(term) = &mir_fn.blocks[u].terminator {
                for v in term.successors() {
                    if v < num_blocks && reachable.contains(&v) && dom[u].contains(&v) {
                        let body = nat_loops.entry(v).or_default();
                        body.insert(v);
                        if body.insert(u) {
                            let mut stack = vec![u];
                            while let Some(cur) = stack.pop() {
                                for &p in &preds[cur] {
                                    if body.insert(p) {
                                        stack.push(p);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 3b. Compute per-block assignments, locally allocated arrays (strictly excluding params
        // and OSR live-ins), definite initialization, and inline-push-eligible blocks (NG-W10-5).
        let mut assigned_in_block: Vec<hashbrown::HashSet<usize>> =
            vec![hashbrown::HashSet::new(); num_blocks];
        let mut local_def_only_array_agg: Vec<bool> = vec![true; mir_fn.locals.len()];
        let mut local_has_any_def: Vec<bool> = vec![false; mir_fn.locals.len()];
        let mut local_copy_sources: Vec<Vec<usize>> = vec![Vec::new(); mir_fn.locals.len()];
        for &p in &mir_fn.params {
            if p.0 < local_def_only_array_agg.len() {
                local_def_only_array_agg[p.0] = false;
            }
        }
        if let Some((_, live)) = osr_info {
            for &l in live {
                if l.0 < local_def_only_array_agg.len() {
                    local_def_only_array_agg[l.0] = false;
                }
            }
        }
        for (b_idx, block) in mir_fn.blocks.iter().enumerate() {
            for stmt in &block.statements {
                if let Statement::Assign(Place::Local(dst), rval, _) = stmt {
                    assigned_in_block[b_idx].insert(dst.0);
                    if dst.0 < local_def_only_array_agg.len() {
                        local_has_any_def[dst.0] = true;
                        match rval {
                            Rvalue::Aggregate(MirType::Array(_), _) => {}
                            Rvalue::Use(
                                Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src)),
                            ) => {
                                local_copy_sources[dst.0].push(src.0);
                            }
                            _ => {
                                local_def_only_array_agg[dst.0] = false;
                            }
                        }
                    }
                }
            }
            if let Some(Terminator::Call {
                destination: Place::Local(dst),
                ..
            }) = &block.terminator
            {
                assigned_in_block[b_idx].insert(dst.0);
                if dst.0 < local_def_only_array_agg.len() {
                    local_has_any_def[dst.0] = true;
                    local_def_only_array_agg[dst.0] = false;
                }
            }
        }
        loop {
            let mut changed = false;
            for i in 0..mir_fn.locals.len() {
                if !local_def_only_array_agg[i] {
                    continue;
                }
                if !local_has_any_def[i] {
                    local_def_only_array_agg[i] = false;
                    changed = true;
                    continue;
                }
                if local_copy_sources[i]
                    .iter()
                    .any(|&src| src >= local_def_only_array_agg.len() || !local_def_only_array_agg[src])
                {
                    local_def_only_array_agg[i] = false;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let locally_allocated_arrays: hashbrown::HashSet<usize> = (0..mir_fn.locals.len())
            .filter(|&i| local_has_any_def[i] && local_def_only_array_agg[i])
            .collect();

        let init_at_entry: hashbrown::HashSet<usize> = if let Some((_, live)) = osr_info {
            live.iter().map(|l| l.0).collect()
        } else {
            mir_fn.params.iter().map(|l| l.0).collect()
        };
        let all_locals: hashbrown::HashSet<usize> = (0..mir_fn.locals.len()).collect();
        let mut def_in: Vec<hashbrown::HashSet<usize>> = (0..num_blocks)
            .map(|b| {
                if b == entry_idx {
                    init_at_entry.clone()
                } else {
                    all_locals.clone()
                }
            })
            .collect();
        let mut def_out: Vec<hashbrown::HashSet<usize>> = (0..num_blocks)
            .map(|b| {
                if b == entry_idx {
                    let mut s = init_at_entry.clone();
                    s.extend(assigned_in_block[b].iter().copied());
                    s
                } else {
                    all_locals.clone()
                }
            })
            .collect();
        loop {
            let mut changed = false;
            for b in 0..num_blocks {
                if !reachable.contains(&b) {
                    continue;
                }
                let new_in = if b == entry_idx {
                    let mut s = init_at_entry.clone();
                    for &p in &preds[b] {
                        s.retain(|x| def_out[p].contains(x));
                    }
                    s
                } else if preds[b].is_empty() {
                    hashbrown::HashSet::new()
                } else {
                    let mut iter = preds[b].iter();
                    let first = *iter.next().unwrap();
                    let mut s = def_out[first].clone();
                    for &p in iter {
                        s.retain(|x| def_out[p].contains(x));
                    }
                    s
                };
                let mut new_out = new_in.clone();
                new_out.extend(assigned_in_block[b].iter().copied());
                if new_in != def_in[b] || new_out != def_out[b] {
                    def_in[b] = new_in;
                    def_out[b] = new_out;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        let mut inline_push_blocks: hashbrown::HashSet<usize> = hashbrown::HashSet::new();
        for &b in &reachable {
            if is_inline_local_array_push_terminator(
                &mir_fn.blocks[b].terminator,
                b,
                mir_fn,
                struct_layouts,
                &locally_allocated_arrays,
                &def_out,
            ) {
                inline_push_blocks.insert(b);
            }
        }

        // 4. Filter for strict observer-free single-header reducible natural loops
        let mut eligible: Vec<(usize, hashbrown::HashSet<usize>)> = Vec::new();
        for (hdr, body) in nat_loops {
            // (a) Single-header strict dominance & zero side entries (rejects irreducible CFGs)
            let all_dominated = body.iter().all(|&b| dom[b].contains(&hdr));
            if !all_dominated {
                continue;
            }
            let single_entry = body
                .iter()
                .all(|&b| b == hdr || preds[b].iter().all(|p| body.contains(p)));
            if !single_entry {
                continue;
            }
            let has_external_entry =
                hdr == entry_idx || preds[hdr].iter().any(|p| !body.contains(p));
            if !has_external_entry {
                continue;
            }
            // (b) ZERO non-inline Terminator::Call, ZERO Raise, ZERO Unreachable, ZERO None, ZERO vec_pre_map
            // Inline local `array.push` / `table.push` calls are allowed (cold grow flushes/reloads counters).
            let observer_free = body.iter().all(|&b| {
                if vec_pre_map.contains_key(&b) {
                    return false;
                }
                matches!(
                    mir_fn.blocks[b].terminator,
                    Some(
                        Terminator::Goto(..)
                            | Terminator::Cond { .. }
                            | Terminator::Switch { .. }
                            | Terminator::Return(..)
                    )
                ) || inline_push_blocks.contains(&b)
                    || inline_queue_blocks.contains(&b)
            });
            if !observer_free {
                continue;
            }
            eligible.push((hdr, body));
        }

        // Sort outermost-first (largest body first, deterministic tiebreak by hdr)
        eligible.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then_with(|| a.0.cmp(&b.0)));

        // 5. Select disjoint counter-promoted loops (outermost-first)
        let mut counter_loops: Vec<PromotedLoopSpec> = Vec::new();
        let mut block_to_counter_loop: hashbrown::HashMap<usize, usize> =
            hashbrown::HashMap::new();
        let mut hdr_to_counter_loop: hashbrown::HashMap<usize, usize> =
            hashbrown::HashMap::new();

        for (hdr, body) in &eligible {
            if body.iter().any(|b| block_to_counter_loop.contains_key(b)) {
                continue;
            }
            let promote_sp = has_sp_ctx && body.iter().any(|b| loop_headers.contains(b));
            let promote_gas = has_gas_ctx
                && (body.iter().any(|b| {
                    mir_fn.blocks[*b]
                        .statements
                        .iter()
                        .any(|s| matches!(s, Statement::ResourceCheck(ResourceKind::Gas, ..)))
                }) || body.iter().any(|b| {
                    vec_hdr_map
                        .get(b)
                        .is_some_and(|&vl_idx| vec_loops[vl_idx].gas_check.is_some())
                }));
            if !promote_sp && !promote_gas {
                continue;
            }
            // NG-W11-2 (`F-06`, `F-20`) + NG-W16-8 (persistent nested Q=64 quantum):
            // Enable header quantization (Q = 64) across the entire counter-promoted loop nest
            // ONLY when:
            //   1. Loop is observer_free (already enforced by `eligible`) and has no SIMD vec header.
            //   2. Every loop header `h` inside `body` has `statements[0] == ResourceCheck(Gas, amt)`
            //      with the EXACT same uniform `amt` (`0 < amt <= 1024`), and there are NO other
            //      `ResourceCheck(Gas, ..)` statements anywhere in `body`.
            // Sharing a single `quantum_rem_var` across outer/mid/inner headers preserves bit-exact
            // SEC-01 gas & safepoint counts while eliminating per-inner-loop refills and exit
            // reconciliations on short-trip nested loops (`BENCH-NG-14`).
            let total_gas_checks: usize = body
                .iter()
                .map(|&b| {
                    mir_fn.blocks[b]
                        .statements
                        .iter()
                        .filter(|s| matches!(s, Statement::ResourceCheck(ResourceKind::Gas, ..)))
                        .count()
                })
                .sum();
            let hdrs_in_body: Vec<usize> = body
                .iter()
                .copied()
                .filter(|b| loop_headers.contains(b))
                .collect();
            let has_vec_hdr = body.iter().any(|b| vec_hdr_map.contains_key(b));
            let quantized_header_gas_amt = if promote_sp
                && promote_gas
                && !has_vec_hdr
                && loop_headers.contains(hdr)
                && !hdrs_in_body.is_empty()
                && total_gas_checks == hdrs_in_body.len()
            {
                let mut uniform_amt: Option<u64> = None;
                let mut all_match = true;
                for &h in &hdrs_in_body {
                    match mir_fn.blocks[h].statements.first() {
                        Some(Statement::ResourceCheck(ResourceKind::Gas, amt, _))
                            if *amt > 0 && *amt <= 1024 =>
                        {
                            match uniform_amt {
                                None => uniform_amt = Some(*amt),
                                Some(prev) if prev == *amt => {}
                                Some(_) => {
                                    all_match = false;
                                    break;
                                }
                            }
                        }
                        _ => {
                            all_match = false;
                            break;
                        }
                    }
                }
                if all_match { uniform_amt } else { None }
            } else {
                None
            };
            let cl_idx = counter_loops.len();
            for &b in body {
                block_to_counter_loop.insert(b, cl_idx);
            }
            hdr_to_counter_loop.insert(*hdr, cl_idx);
            counter_loops.push(PromotedLoopSpec {
                hdr: *hdr,
                loop_blocks: body.clone(),
                promote_gas,
                promote_sp,
                quantized_header_gas_amt,
            });
        }

        // 5b. NG-W14-1: Prioritize innermost natural loops (smallest block set first)
        // for Q=64 quantization inside enclosing non-quantized counter-promoted loops.
        // Outer loops stay on standard non-quantized counter promotion (zero ssub_overflow),
        // while innermost leaf loops get Q=64 single-counter batch strip-mining.
        let mut eligible_innermost = eligible.clone();
        eligible_innermost.sort_by(|a, b| a.1.len().cmp(&b.1.len()).then_with(|| a.0.cmp(&b.0)));

        let mut inner_quantized_loops: Vec<InnerQuantizedLoopSpec> = Vec::new();
        let mut block_to_inner_quantized: hashbrown::HashMap<usize, usize> =
            hashbrown::HashMap::new();

        for (hdr, body) in &eligible_innermost {
            let Some(&parent_cl_idx) = block_to_counter_loop.get(hdr) else {
                continue;
            };
            let parent_spec = &counter_loops[parent_cl_idx];
            if parent_spec.hdr == *hdr
                || parent_spec.quantized_header_gas_amt.is_some()
                || !parent_spec.promote_gas
                || !parent_spec.promote_sp
            {
                continue;
            }
            if !body.iter().all(|b| {
                block_to_counter_loop.get(b) == Some(&parent_cl_idx)
                    && !block_to_inner_quantized.contains_key(b)
            }) {
                continue;
            }
            let total_gas_checks: usize = body
                .iter()
                .map(|&b| {
                    mir_fn.blocks[b]
                        .statements
                        .iter()
                        .filter(|s| matches!(s, Statement::ResourceCheck(ResourceKind::Gas, ..)))
                        .count()
                })
                .sum();
            let loop_header_count = body.iter().filter(|&&b| loop_headers.contains(&b)).count();
            let has_vec_hdr = body.iter().any(|b| vec_hdr_map.contains_key(b));
            if has_vec_hdr
                || !loop_headers.contains(hdr)
                || loop_header_count != 1
                || total_gas_checks != 1
            {
                continue;
            }
            let header_gas_amt = match mir_fn.blocks[*hdr].statements.first() {
                Some(Statement::ResourceCheck(ResourceKind::Gas, amt, _))
                    if *amt > 0 && *amt <= 1024 =>
                {
                    *amt
                }
                _ => continue,
            };
            let iq_idx = inner_quantized_loops.len();
            for &b in body {
                block_to_inner_quantized.insert(b, iq_idx);
            }
            inner_quantized_loops.push(InnerQuantizedLoopSpec {
                hdr: *hdr,
                loop_blocks: body.clone(),
                parent_cl_idx,
                header_gas_amt,
            });
        }

        // 6. Pillar 2: Non-Speculative Guarded-Preheader Array (data_ptr, len) & Struct Base Hoisting
        // (`eligible` is already sorted outermost-first so outer loops hoist before inner loops.)
        let mut hoist_loops: Vec<HoistLoopSpec> = Vec::new();
        let mut hdr_to_hoist_loop: hashbrown::HashMap<usize, usize> = hashbrown::HashMap::new();

        for (hdr, body) in &eligible {
            if vec_hdr_map.contains_key(hdr) {
                continue;
            }
            // Loops containing `array.push` / `table.push` may grow/redirect array backing buffers;
            // never hoist `(len, data_ptr)` across a loop body that can resize an array.
            let loop_has_array_push = body.iter().any(|b| inline_push_blocks.contains(b));
            // Require single-condition loop header branching to (body_entry inside loop, loop_exit outside loop)
            let Some(Terminator::Cond {
                then_block,
                else_block,
                ..
            }) = &mir_fn.blocks[*hdr].terminator
            else {
                continue;
            };
            let (body_entry, loop_exit, then_is_body) = if body.contains(&then_block.0)
                && then_block.0 != *hdr
                && !body.contains(&else_block.0)
            {
                (then_block.0, else_block.0, true)
            } else if body.contains(&else_block.0)
                && else_block.0 != *hdr
                && !body.contains(&then_block.0)
            {
                (else_block.0, then_block.0, false)
            } else {
                continue;
            };

            // Collect written index roots across the entire loop body (for shifted-RawWords alias safety)
            let mut written_index_roots = hashbrown::HashSet::new();
            let mut has_non_local_index_write = false;
            let mut used_arrays_anywhere = Vec::new();
            for &b in body {
                for stmt in &mir_fn.blocks[b].statements {
                    if let Statement::Assign(lhs, rval, _) = stmt {
                        if let Place::Index(base, _) = lhs {
                            if let Place::Local(loc) = base.as_ref() {
                                written_index_roots.insert(loc.0);
                            } else {
                                has_non_local_index_write = true;
                            }
                        }
                        let mut dummy_s = Vec::new();
                        collect_place_hoist_candidates(lhs, &mir_fn.locals, &mut used_arrays_anywhere, &mut dummy_s);
                        match rval {
                            Rvalue::Use(op) | Rvalue::Cast(op, _) | Rvalue::UnaryOp(_, op) => {
                                collect_operand_hoist_candidates(op, &mir_fn.locals, &mut used_arrays_anywhere, &mut dummy_s);
                            }
                            Rvalue::BinaryOp(_, l, r) => {
                                collect_operand_hoist_candidates(l, &mir_fn.locals, &mut used_arrays_anywhere, &mut dummy_s);
                                collect_operand_hoist_candidates(r, &mir_fn.locals, &mut used_arrays_anywhere, &mut dummy_s);
                            }
                            Rvalue::Aggregate(_, ops) => {
                                for op in ops {
                                    collect_operand_hoist_candidates(op, &mir_fn.locals, &mut used_arrays_anywhere, &mut dummy_s);
                                }
                            }
                            Rvalue::Closure { captures, .. } => {
                                for (_, op, _) in captures {
                                    collect_operand_hoist_candidates(op, &mir_fn.locals, &mut used_arrays_anywhere, &mut dummy_s);
                                }
                            }
                        }
                    }
                }
            }

            // Walk straight-line path from body_entry to find accesses that post-dominate body_entry
            // before any non-OOB failing statement (guarantees zero speculation and exact error ordering!).
            let mut guaranteed_arrays: Vec<usize> = Vec::new();
            let mut guaranteed_structs: Vec<usize> = Vec::new();
            let mut walk_bb = body_entry;
            let mut walk_visited = hashbrown::HashSet::new();
            let validated_at_hdr = &prov_plan.validated_in[*hdr];
            'walk: while walk_visited.insert(walk_bb) {
                for stmt in &mir_fn.blocks[walk_bb].statements {
                    match stmt {
                        Statement::ResourceCheck(ResourceKind::Gas, ..) => break 'walk,
                        Statement::Assign(lhs, rval, _) => {
                            // RHS executes before LHS
                            match rval {
                                Rvalue::UnaryOp(MirUnaryOp::Len, op) => {
                                    collect_operand_hoist_candidates(
                                        op,
                                        &mir_fn.locals,
                                        &mut guaranteed_arrays,
                                        &mut guaranteed_structs,
                                    );
                                    if let Operand::Copy(Place::Local(loc))
                                        | Operand::Move(Place::Local(loc)) = op
                                        && matches!(
                                            mir_fn.locals.get(loc.0).map(|l| &l.ty),
                                            Some(MirType::Array(_))
                                        )
                                        && !guaranteed_arrays.contains(&loc.0)
                                    {
                                        guaranteed_arrays.push(loc.0);
                                    }
                                }
                                Rvalue::Use(op) | Rvalue::Cast(op, _) | Rvalue::UnaryOp(_, op) => {
                                    collect_operand_hoist_candidates(
                                        op,
                                        &mir_fn.locals,
                                        &mut guaranteed_arrays,
                                        &mut guaranteed_structs,
                                    );
                                }
                                Rvalue::BinaryOp(_, l, r) => {
                                    collect_operand_hoist_candidates(
                                        l,
                                        &mir_fn.locals,
                                        &mut guaranteed_arrays,
                                        &mut guaranteed_structs,
                                    );
                                    collect_operand_hoist_candidates(
                                        r,
                                        &mir_fn.locals,
                                        &mut guaranteed_arrays,
                                        &mut guaranteed_structs,
                                    );
                                }
                                Rvalue::Aggregate(_, ops) => {
                                    for op in ops {
                                        collect_operand_hoist_candidates(
                                            op,
                                            &mir_fn.locals,
                                            &mut guaranteed_arrays,
                                            &mut guaranteed_structs,
                                        );
                                    }
                                }
                                Rvalue::Closure { captures, .. } => {
                                    for (_, op, _) in captures {
                                        collect_operand_hoist_candidates(
                                            op,
                                            &mir_fn.locals,
                                            &mut guaranteed_arrays,
                                            &mut guaranteed_structs,
                                        );
                                    }
                                }
                            }
                            if rvalue_may_fail_non_oob(
                                rval,
                                &mir_fn.locals,
                                validated_at_hdr,
                                const_int_locals,
                            ) {
                                break 'walk;
                            }
                            collect_place_hoist_candidates(
                                lhs,
                                &mir_fn.locals,
                                &mut guaranteed_arrays,
                                &mut guaranteed_structs,
                            );
                        }
                        _ => {}
                    }
                }
                match &mir_fn.blocks[walk_bb].terminator {
                    Some(Terminator::Goto(next, _))
                        if body.contains(&next.0) && next.0 != *hdr =>
                    {
                        walk_bb = next.0;
                    }
                    _ => break 'walk,
                }
            }

            // Combine guaranteed_arrays with already-proven-valid arrays used anywhere in the loop.
            // Also resolve single-definition local copy aliases (`_inlined_arr = Copy(data)`) back
            // to their loop-invariant root so outer loops hoist `data` once for inlined inner loops
            // (`BENCH-NG-14`).
            let mut single_def_copy_src: Vec<Option<usize>> = vec![None; mir_fn.locals.len()];
            let mut def_counts: Vec<usize> = vec![0; mir_fn.locals.len()];
            for &p in &mir_fn.params {
                if p.0 < def_counts.len() {
                    def_counts[p.0] += 1;
                }
            }
            if let Some((_, live)) = osr_info {
                for &l in live {
                    if l.0 < def_counts.len() {
                        def_counts[l.0] += 1;
                    }
                }
            }
            for block in &mir_fn.blocks {
                for stmt in &block.statements {
                    if let Statement::Assign(Place::Local(dst), rval, _) = stmt
                        && dst.0 < def_counts.len()
                    {
                        def_counts[dst.0] += 1;
                        if def_counts[dst.0] == 1 {
                            if let Rvalue::Use(
                                Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src)),
                            ) = rval
                            {
                                single_def_copy_src[dst.0] = Some(src.0);
                            }
                        } else {
                            single_def_copy_src[dst.0] = None;
                        }
                    }
                }
                if let Some(Terminator::Call {
                    destination: Place::Local(dst),
                    ..
                }) = &block.terminator
                    && dst.0 < def_counts.len()
                {
                    def_counts[dst.0] += 1;
                    single_def_copy_src[dst.0] = None;
                }
            }
            let resolve_copy_root = |mut cur: usize| -> usize {
                for _ in 0..8 {
                    if let Some(next) = single_def_copy_src.get(cur).copied().flatten() {
                        if next == cur {
                            break;
                        }
                        cur = next;
                    } else {
                        break;
                    }
                }
                cur
            };

            let mut candidate_arrays = guaranteed_arrays;
            let mut alias_pairs: Vec<(usize, usize)> = Vec::new();
            for loc_idx in used_arrays_anywhere {
                let root_idx = resolve_copy_root(loc_idx);
                if (validated_at_hdr.contains(&loc_idx)
                    || locally_allocated_arrays.contains(&loc_idx))
                    && !candidate_arrays.contains(&loc_idx)
                {
                    candidate_arrays.push(loc_idx);
                }
                if root_idx != loc_idx
                    && !stack_slots.contains_key(&loc_idx)
                    && (validated_at_hdr.contains(&root_idx)
                        || locally_allocated_arrays.contains(&root_idx))
                {
                    if !candidate_arrays.contains(&root_idx) {
                        candidate_arrays.push(root_idx);
                    }
                    if !alias_pairs.contains(&(loc_idx, root_idx)) {
                        alias_pairs.push((loc_idx, root_idx));
                    }
                }
            }

            let is_proven_mir_array = |loc_idx: usize| -> bool {
                validated_at_hdr.contains(&loc_idx) || locally_allocated_arrays.contains(&loc_idx)
            };

            let mut hoist_arrays = Vec::new();
            let mut hoist_array_aliases: Vec<(LocalId, LocalId)> = Vec::new();
            if !loop_has_array_push {
                for loc_idx in candidate_arrays {
                    if !def_in[*hdr].contains(&loc_idx)
                        || stack_slots.contains_key(&loc_idx)
                        || body.iter().any(|b| assigned_in_block[*b].contains(&loc_idx))
                    {
                        continue;
                    }
                    // Skip if already hoisted by an enclosing outer loop (or aliased by one)
                    if hoist_loops.iter().any(|outer| {
                        outer.loop_blocks.contains(hdr)
                            && (outer.hoist_arrays.iter().any(|a| a.0 == loc_idx)
                                || outer.hoist_array_aliases.iter().any(|(al, _)| al.0 == loc_idx))
                    }) {
                        continue;
                    }
                    // Shifted-overlap RawWords safety: if any OTHER array is written in this loop,
                    // require both this array and all written arrays to be proven MirAllocationKind::Array.
                    let other_array_written = has_non_local_index_write
                        || written_index_roots
                            .iter()
                            .any(|&w| w != loc_idx && resolve_copy_root(w) != loc_idx);
                    if other_array_written {
                        let all_proven = !has_non_local_index_write
                            && is_proven_mir_array(loc_idx)
                            && written_index_roots.iter().all(|&w| is_proven_mir_array(w));
                        if !all_proven {
                            continue;
                        }
                    }
                    hoist_arrays.push(LocalId(loc_idx));
                }
                for (alias_loc, root_loc) in alias_pairs {
                    if hoist_arrays.iter().any(|a| a.0 == root_loc) {
                        hoist_array_aliases.push((LocalId(alias_loc), LocalId(root_loc)));
                    }
                }
            }

            let mut hoist_structs = Vec::new();
            for loc_idx in guaranteed_structs {
                if !def_in[*hdr].contains(&loc_idx)
                    || stack_slots.contains_key(&loc_idx)
                    || body.iter().any(|b| assigned_in_block[*b].contains(&loc_idx))
                {
                    continue;
                }
                if hoist_loops.iter().any(|outer| {
                    outer.loop_blocks.contains(hdr)
                        && outer.hoist_structs.iter().any(|s| s.0 == loc_idx)
                }) {
                    continue;
                }
                hoist_structs.push(LocalId(loc_idx));
            }

            if !hoist_arrays.is_empty() || !hoist_structs.is_empty() {
                let hl_idx = hoist_loops.len();
                hdr_to_hoist_loop.insert(*hdr, hl_idx);
                hoist_loops.push(HoistLoopSpec {
                    hdr: *hdr,
                    body_entry,
                    loop_exit,
                    then_is_body,
                    loop_blocks: body.clone(),
                    hoist_arrays,
                    hoist_array_aliases,
                    hoist_structs,
                });
            }
        }

        LoopPromotionPlan {
            counter_loops,
            block_to_counter_loop,
            hdr_to_counter_loop,
            inner_quantized_loops,
            block_to_inner_quantized,
            hoist_loops,
            hdr_to_hoist_loop,
            inline_push_blocks,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_safepoint_check(
        builder: &mut FunctionBuilder,
        sp_ptr_var: Variable,
        safepoint_slow_fn: FuncRef,
        init_deopt_fn: FuncRef,
        record_deopt_local_fn: FuncRef,
        target_block_idx: usize,
        live_locals: &[LocalId],
        local_vars: &[Variable],
        stack_slots: &hashbrown::HashMap<usize, StackSlot>,
        locals: &[LocalDecl],
        raw_emergency_exit_bb: Block,
        pending_vec_reduction: Option<(LocalId, Variable, Variable)>,
        promoted: Option<PromotedLoopCounters>,
        mut cfg_rec: Option<&mut PromotedLoopCfgRecord>,
    ) {
        const OFF_SLOW_FLAGS: i32 = core::mem::offset_of!(SafepointFastState, slow_flags) as i32;
        const OFF_SAFEPOINTS_CHECKED: i32 =
            core::mem::offset_of!(SafepointFastState, safepoints_checked) as i32;

        let sp_ptr = builder.use_var(sp_ptr_var);
        if let Some((_, sp_cnt_var)) = promoted.and_then(|p| p.sp) {
            let cnt = builder.use_var(sp_cnt_var);
            let cnt1 = builder.ins().iadd_imm_s(cnt, 1);
            builder.def_var(sp_cnt_var, cnt1);
        } else {
            let cnt = builder
                .ins()
                .load(types::I64, MemFlagsData::trusted(), sp_ptr, OFF_SAFEPOINTS_CHECKED);
            let cnt1 = builder.ins().iadd_imm_s(cnt, 1);
            builder
                .ins()
                .store(MemFlagsData::trusted(), cnt1, sp_ptr, OFF_SAFEPOINTS_CHECKED);
        }
        let flags = builder
            .ins()
            .load(types::I64, MemFlagsData::trusted(), sp_ptr, OFF_SLOW_FLAGS);
        let is_slow = builder.ins().icmp_imm_s(IntCC::NotEqual, flags, 0);

        let slow_safepoint_bb = builder.create_block();
        let cont_bb = builder.create_block();
        let err_dispatch_bb = builder.create_block();
        let deopt_capture_bb = builder.create_block();
        builder.set_cold_block(slow_safepoint_bb);
        builder.set_cold_block(err_dispatch_bb);
        builder.set_cold_block(deopt_capture_bb);

        builder.ins().brif(is_slow, slow_safepoint_bb, &[], cont_bb, &[]);

        builder.switch_to_block(slow_safepoint_bb);
        if let Some(p) = promoted {
            // Barrier 1: flush BOTH promoted counters before calling lust_mir_safepoint_slow()
            emit_promoted_counters_flush(builder, p);
        }
        let sp_call = builder.ins().call(safepoint_slow_fn, &[]);
        let sp_res = if builder.inst_results(sp_call).is_empty() {
            const OFF_SLOW_RESULT: i32 = core::mem::offset_of!(SafepointFastState, slow_result) as i32;
            let sp_ptr = builder.use_var(sp_ptr_var);
            builder
                .ins()
                .load(types::I64, MemFlagsData::trusted(), sp_ptr, OFF_SLOW_RESULT)
        } else {
            builder.inst_results(sp_call)[0]
        };
        let is_err = builder.ins().icmp_imm_s(IntCC::NotEqual, sp_res, 0);
        if let Some(p) = promoted {
            let reload_bb = builder.create_block();
            builder.set_cold_block(reload_bb);
            if let Some(rec) = cfg_rec.as_deref_mut() {
                rec.flush_clif_blocks.insert(slow_safepoint_bb);
                rec.post_flush_clif_blocks.insert(err_dispatch_bb);
                rec.post_flush_clif_blocks.insert(deopt_capture_bb);
                rec.reload_clif_blocks.insert(reload_bb);
            }
            builder.ins().brif(is_err, err_dispatch_bb, &[], reload_bb, &[]);

            builder.switch_to_block(reload_bb);
            // Reload BOTH promoted counters after slow call returns 0 (continue)
            emit_promoted_counters_load(builder, p);
            builder.ins().jump(cont_bb, &[]);
        } else {
            builder.ins().brif(is_err, err_dispatch_bb, &[], cont_bb, &[]);
        }

        builder.switch_to_block(err_dispatch_bb);
        let is_deopt = builder.ins().icmp_imm_s(IntCC::Equal, sp_res, MIR_STATUS_DEOPT);
        builder.ins().brif(is_deopt, deopt_capture_bb, &[], raw_emergency_exit_bb, &[]);

        builder.switch_to_block(deopt_capture_bb);
        let target_pc_val = builder.ins().iconst(types::I64, target_block_idx as i64);
        builder.ins().call(init_deopt_fn, &[target_pc_val]);
        for &loc_id in live_locals {
            if loc_id.0 < local_vars.len() {
                let loc_idx_val = builder.ins().iconst(types::I64, loc_id.0 as i64);
                let cur_val = if let Some(&slot) = stack_slots.get(&loc_id.0) {
                    let mir_ty = locals.get(loc_id.0).map(|l| l.ty.clone()).unwrap_or(MirType::Int);
                    let clif_ty = mir_to_clif_type(&mir_ty).unwrap_or(types::I64);
                    builder.ins().stack_load(types::I64, clif_ty, slot, 0)
                } else {
                    builder.use_var(local_vars[loc_id.0])
                };
                let val_ty = builder.func.dfg.value_type(cur_val);
                let val_i64 = if val_ty == types::I8 {
                    builder.ins().uextend(types::I64, cur_val)
                } else if val_ty == types::F64 {
                    builder.ins().bitcast(types::I64, MemFlagsData::new(), cur_val)
                } else {
                    cur_val
                };
                // Fold pending SIMD reduction vector lanes only on the cold deopt edge so the hot loop stays untouched.
                let val_i64 = if let Some((acc_id, vacc0, vacc1)) = pending_vec_reduction
                    && loc_id == acc_id
                {
                    let lane_sum = fold_vec_acc_lanes(builder, vacc0, vacc1);
                    builder.ins().iadd(val_i64, lane_sum)
                } else {
                    val_i64
                };
                builder.ins().call(record_deopt_local_fn, &[target_pc_val, loc_idx_val, val_i64]);
            }
        }
        builder.ins().jump(raw_emergency_exit_bb, &[]);

        builder.switch_to_block(cont_bb);
    }

    /// NG-W11-2 + NG-W14-1: Atomic quantized loop header check (Q = 64, SEC-01 sound)
    /// with true single-counter batch gas & safepoint reconciliation.
    ///
    /// Invariant maintained throughout the quantized loop:
    ///   - `gas_var == actual_gas_remaining - quantum_rem * amt`
    ///   - `sp_cnt_var == actual_safepoints_checked + quantum_rem`
    ///   - `0 <= quantum_rem <= 63` (and `quantum_rem > 0 => slow_flags == 0 && gas_var >= 0`)
    ///
    /// Fast path (63 of 64 iterations): updates ONLY `quantum_rem -= 1` and falls through
    /// directly into `cont_bb` (zero `sp_cnt` update, zero `gas_var` update, zero `slow_flags`
    /// memory load, zero 4-block diamond). Exact `(gas_remaining, safepoints_checked)` are
    /// reconciled on refill, every normal loop exit edge, early break/return, OOG/OOB trap,
    /// and forced deopt exit.
    #[allow(clippy::too_many_arguments)]
    fn emit_quantized_loop_header_check(
        builder: &mut FunctionBuilder,
        sp_ptr_var: Variable,
        safepoint_slow_fn: FuncRef,
        gas_slow_fn: FuncRef,
        init_deopt_fn: FuncRef,
        record_deopt_local_fn: FuncRef,
        target_block_idx: usize,
        amt: u64,
        live_locals: &[LocalId],
        local_vars: &[Variable],
        stack_slots: &hashbrown::HashMap<usize, StackSlot>,
        locals: &[LocalDecl],
        raw_emergency_exit_bb: Block,
        promoted: PromotedLoopCounters,
        mut cfg_rec: Option<&mut PromotedLoopCfgRecord>,
        mut iq_cfg_rec: Option<&mut InnerQuantizedCfgRecord>,
    ) {
        const OFF_SLOW_FLAGS: i32 = core::mem::offset_of!(SafepointFastState, slow_flags) as i32;
        const QUANTUM_FAST_ITERS: i64 = 63;

        debug_assert!(amt > 0 && amt <= 1024);
        let (_, sp_cnt_var) = promoted
            .sp
            .expect("quantized loop header check requires promoted sp counter");
        let (_, gas_var) = promoted
            .gas
            .expect("quantized loop header check requires promoted gas counter");
        let (quantum_rem_var, q_amt) = promoted
            .quantum
            .expect("quantized loop header check requires quantum_rem_var");
        debug_assert_eq!(q_amt, amt);

        // Fast path (63 of 64 iterations): single register decrement + branch straight to cont_bb.
        // Decrementing `quantum_rem` by 1 preserves:
        //   actual_gas_remaining = gas_var + quantum_rem * amt  (decreased by amt)
        //   actual_safepoints    = sp_cnt_var - quantum_rem     (increased by 1)
        let q0 = builder.use_var(quantum_rem_var);
        let q1 = builder.ins().iadd_imm_s(q0, -1);
        let needs_refill = builder.ins().icmp_imm_s(IntCC::SignedLessThan, q1, 0);
        builder.def_var(quantum_rem_var, q1);

        let check_reload_bb = builder.create_block();
        let slow_safepoint_bb = builder.create_block();
        let sp_reload_bb = builder.create_block();
        let err_dispatch_bb = builder.create_block();
        let deopt_capture_bb = builder.create_block();
        let gas_check_bb = builder.create_block();
        let allow_arm_param = builder.append_block_param(gas_check_bb, types::I8);
        let gas_slow_bb = builder.create_block();
        let gas_reload_bb = builder.create_block();
        let arm_quantum_bb = builder.create_block();
        let cont_bb = builder.create_block();

        // Refill / check / arm blocks run at most 1/64 iterations — mark cold so hot loop
        // body lays out straight-line with zero taken jumps on the 63/64 fast path.
        builder.set_cold_block(check_reload_bb);
        builder.set_cold_block(slow_safepoint_bb);
        builder.set_cold_block(sp_reload_bb);
        builder.set_cold_block(err_dispatch_bb);
        builder.set_cold_block(deopt_capture_bb);
        builder.set_cold_block(gas_check_bb);
        builder.set_cold_block(gas_slow_bb);
        builder.set_cold_block(gas_reload_bb);
        builder.set_cold_block(arm_quantum_bb);

        if let Some(rec) = cfg_rec.as_deref_mut() {
            rec.flush_clif_blocks.insert(slow_safepoint_bb);
            rec.post_flush_clif_blocks.insert(err_dispatch_bb);
            rec.post_flush_clif_blocks.insert(deopt_capture_bb);
            rec.reload_clif_blocks.insert(sp_reload_bb);
            rec.flush_clif_blocks.insert(gas_slow_bb);
            rec.reload_clif_blocks.insert(gas_reload_bb);
        }
        if let Some(iq_rec) = iq_cfg_rec.as_deref_mut() {
            iq_rec.reconcile_clif_blocks.insert(slow_safepoint_bb);
            iq_rec.post_flush_clif_blocks.insert(err_dispatch_bb);
            iq_rec.post_flush_clif_blocks.insert(deopt_capture_bb);
            iq_rec.reload_clif_blocks.insert(sp_reload_bb);
            iq_rec.reconcile_clif_blocks.insert(gas_slow_bb);
            iq_rec.reload_clif_blocks.insert(gas_reload_bb);
        }

        builder
            .ins()
            .brif(needs_refill, check_reload_bb, &[], cont_bb, &[]);

        // Refill path (1 of 64 iterations, or every iteration when gas < 64*amt or flags != 0).
        // Entered ONLY when q0 == 0, so gas_var and sp_cnt_var are already exact pre-iteration values.
        // Evaluates slow_flags BEFORE decrementing gas_var so interrupts and deopt observe exact
        // pre-ResourceCheck gas_remaining.
        builder.switch_to_block(check_reload_bb);
        let zero_q = builder.ins().iconst(types::I64, 0);
        builder.def_var(quantum_rem_var, zero_q);
        let sp0 = builder.use_var(sp_cnt_var);
        let sp1 = builder.ins().iadd_imm_s(sp0, 1);
        builder.def_var(sp_cnt_var, sp1);
        let sp_ptr = builder.use_var(sp_ptr_var);
        let flags = builder
            .ins()
            .load(types::I64, MemFlagsData::trusted(), sp_ptr, OFF_SLOW_FLAGS);
        let is_slow = builder.ins().icmp_imm_s(IntCC::NotEqual, flags, 0);
        let allow_arm_yes = builder.ins().iconst(types::I8, 1);
        builder.ins().brif(
            is_slow,
            slow_safepoint_bb,
            &[],
            gas_check_bb,
            &[BlockArg::Value(allow_arm_yes)],
        );

        builder.switch_to_block(slow_safepoint_bb);
        // quantum_rem_var == 0 here, so gas_var (pre-ResourceCheck) & sp_cnt_var (+1) are exact.
        emit_promoted_counters_flush_exact(builder, promoted);
        let sp_call = builder.ins().call(safepoint_slow_fn, &[]);
        let sp_res = if builder.inst_results(sp_call).is_empty() {
            const OFF_SLOW_RESULT: i32 = core::mem::offset_of!(SafepointFastState, slow_result) as i32;
            let sp_ptr = builder.use_var(sp_ptr_var);
            builder
                .ins()
                .load(types::I64, MemFlagsData::trusted(), sp_ptr, OFF_SLOW_RESULT)
        } else {
            builder.inst_results(sp_call)[0]
        };
        let is_sp_err = builder.ins().icmp_imm_s(IntCC::NotEqual, sp_res, 0);
        builder
            .ins()
            .brif(is_sp_err, err_dispatch_bb, &[], sp_reload_bb, &[]);

        builder.switch_to_block(sp_reload_bb);
        emit_promoted_counters_load(builder, promoted);
        let allow_arm_no = builder.ins().iconst(types::I8, 0);
        builder
            .ins()
            .jump(gas_check_bb, &[BlockArg::Value(allow_arm_no)]);

        builder.switch_to_block(err_dispatch_bb);
        let is_deopt = builder.ins().icmp_imm_s(IntCC::Equal, sp_res, MIR_STATUS_DEOPT);
        builder
            .ins()
            .brif(is_deopt, deopt_capture_bb, &[], raw_emergency_exit_bb, &[]);

        builder.switch_to_block(deopt_capture_bb);
        let target_pc_val = builder.ins().iconst(types::I64, target_block_idx as i64);
        builder.ins().call(init_deopt_fn, &[target_pc_val]);
        for &loc_id in live_locals {
            if loc_id.0 < local_vars.len() {
                let loc_idx_val = builder.ins().iconst(types::I64, loc_id.0 as i64);
                let cur_val = if let Some(&slot) = stack_slots.get(&loc_id.0) {
                    let mir_ty = locals
                        .get(loc_id.0)
                        .map(|l| l.ty.clone())
                        .unwrap_or(MirType::Int);
                    let clif_ty = mir_to_clif_type(&mir_ty).unwrap_or(types::I64);
                    builder.ins().stack_load(types::I64, clif_ty, slot, 0)
                } else {
                    builder.use_var(local_vars[loc_id.0])
                };
                let val_ty = builder.func.dfg.value_type(cur_val);
                let val_i64 = if val_ty == types::I8 {
                    builder.ins().uextend(types::I64, cur_val)
                } else if val_ty == types::F64 {
                    builder.ins().bitcast(types::I64, MemFlagsData::new(), cur_val)
                } else {
                    cur_val
                };
                builder
                    .ins()
                    .call(record_deopt_local_fn, &[target_pc_val, loc_idx_val, val_i64]);
            }
        }
        builder.ins().jump(raw_emergency_exit_bb, &[]);

        // Gas check (2-instruction iadd_imm_s + icmp_imm_s since 0 < amt <= 1024).
        builder.switch_to_block(gas_check_bb);
        let gas0 = builder.use_var(gas_var);
        let gas1 = builder.ins().iadd_imm_s(gas0, -(amt as i64));
        builder.def_var(gas_var, gas1);
        let is_neg = builder.ins().icmp_imm_s(IntCC::SignedLessThan, gas1, 0);
        builder
            .ins()
            .brif(is_neg, gas_slow_bb, &[], arm_quantum_bb, &[]);

        builder.switch_to_block(gas_slow_bb);
        // quantum_rem_var == 0 here, so gas_var (gas1 < 0) & sp_cnt_var are exact.
        emit_promoted_counters_flush_exact(builder, promoted);
        let slow_call = builder.ins().call(gas_slow_fn, &[]);
        let slow_res = if builder.inst_results(slow_call).is_empty() {
            const OFF_GAS_SLOW_RESULT: i32 = core::mem::offset_of!(MirContext, gas_slow_result) as i32;
            let (gas_ptr_var, _) = promoted
                .gas
                .expect("quantized loop header check requires promoted gas ptr");
            let gas_ptr = builder.use_var(gas_ptr_var);
            builder
                .ins()
                .load(types::I64, MemFlagsData::trusted(), gas_ptr, OFF_GAS_SLOW_RESULT)
        } else {
            builder.inst_results(slow_call)[0]
        };
        let is_gas_err = builder.ins().icmp_imm_s(IntCC::NotEqual, slow_res, 0);
        builder
            .ins()
            .brif(is_gas_err, raw_emergency_exit_bb, &[], gas_reload_bb, &[]);

        builder.switch_to_block(gas_reload_bb);
        // Reloads replenished gas_var, sp_cnt_var, and resets quantum_rem_var = 0.
        emit_promoted_counters_load(builder, promoted);
        builder.ins().jump(cont_bb, &[]);

        // Arm next 63 fast iterations ONLY when slow_flags == 0 AND gas1 >= 63 * amt.
        // Pre-reserve 63 * amt from gas_var and +63 into sp_cnt_var so the fast path
        // updates ONLY quantum_rem_var!
        builder.switch_to_block(arm_quantum_bb);
        let threshold = QUANTUM_FAST_ITERS * (amt as i64);
        let can_arm_gas = builder
            .ins()
            .icmp_imm_s(IntCC::SignedGreaterThanOrEqual, gas1, threshold);
        let can_arm = builder.ins().band(allow_arm_param, can_arm_gas);
        let c63 = builder.ins().iconst(types::I64, QUANTUM_FAST_ITERS);
        let c0 = builder.ins().iconst(types::I64, 0);
        let next_q = builder.ins().select(can_arm, c63, c0);
        let gas_reserved = builder.ins().iadd_imm_s(gas1, -threshold);
        let next_gas = builder.ins().select(can_arm, gas_reserved, gas1);
        let sp_cur = builder.use_var(sp_cnt_var);
        let sp_reserved = builder.ins().iadd_imm_s(sp_cur, QUANTUM_FAST_ITERS);
        let next_sp = builder.ins().select(can_arm, sp_reserved, sp_cur);
        builder.def_var(quantum_rem_var, next_q);
        builder.def_var(gas_var, next_gas);
        builder.def_var(sp_cnt_var, next_sp);
        builder.ins().jump(cont_bb, &[]);

        builder.switch_to_block(cont_bb);
    }

    /// Compact, 100% reducible quantized loop header check calling `lust_mir_quantized_refill_slow`.
    /// Eliminates cross-loop irreducible SCCs and keeps cold CLIF block count minimal (~4 tiny blocks).
    #[allow(clippy::too_many_arguments)]
    fn emit_quantized_loop_header_check_compact(
        builder: &mut FunctionBuilder,
        refill_slow_fn: FuncRef,
        init_deopt_fn: FuncRef,
        record_deopt_local_fn: FuncRef,
        target_block_idx: usize,
        amt: u64,
        live_locals: &[LocalId],
        local_vars: &[Variable],
        stack_slots: &hashbrown::HashMap<usize, StackSlot>,
        locals: &[LocalDecl],
        raw_emergency_exit_bb: Block,
        promoted: PromotedLoopCounters,
        mut cfg_rec: Option<&mut PromotedLoopCfgRecord>,
        mut iq_cfg_rec: Option<&mut InnerQuantizedCfgRecord>,
    ) {
        const QUANTUM_FAST_ITERS: i64 = 63;
        debug_assert!(amt > 0 && amt <= 1024);
        let (_, sp_cnt_var) = promoted
            .sp
            .expect("compact quantized header requires promoted sp counter");
        let (quantum_rem_var, q_amt) = promoted
            .quantum
            .expect("compact quantized header requires quantum_rem_var");
        debug_assert_eq!(q_amt, amt);

        let q0 = builder.use_var(quantum_rem_var);
        let q1 = builder.ins().iadd_imm_s(q0, -1);
        let needs_refill = builder.ins().icmp_imm_s(IntCC::SignedLessThan, q1, 0);
        builder.def_var(quantum_rem_var, q1);

        let refill_bb = builder.create_block();
        let reload_bb = builder.create_block();
        let err_bb = builder.create_block();
        let deopt_bb = builder.create_block();
        let cont_bb = builder.create_block();

        builder.set_cold_block(refill_bb);
        builder.set_cold_block(reload_bb);
        builder.set_cold_block(err_bb);
        builder.set_cold_block(deopt_bb);

        if let Some(rec) = cfg_rec.as_deref_mut() {
            rec.flush_clif_blocks.insert(refill_bb);
            rec.reload_clif_blocks.insert(reload_bb);
            rec.post_flush_clif_blocks.insert(err_bb);
            rec.post_flush_clif_blocks.insert(deopt_bb);
            rec.loop_clif_blocks.insert(cont_bb);
        }
        if let Some(iq_rec) = iq_cfg_rec.as_deref_mut() {
            iq_rec.reconcile_clif_blocks.insert(refill_bb);
            iq_rec.loop_clif_blocks.insert(cont_bb);
        }

        builder.ins().brif(needs_refill, refill_bb, &[], cont_bb, &[]);

        builder.switch_to_block(refill_bb);
        let sp0 = builder.use_var(sp_cnt_var);
        let sp1 = builder.ins().iadd_imm_s(sp0, 1);
        builder.def_var(sp_cnt_var, sp1);
        emit_promoted_counters_flush_exact(builder, promoted);
        let amt_v = builder.ins().iconst(types::I64, amt as i64);
        let max_q_v = builder.ins().iconst(types::I64, QUANTUM_FAST_ITERS);
        let call = builder.ins().call(refill_slow_fn, &[amt_v, max_q_v]);
        let res = builder.inst_results(call)[0];
        let is_err = builder.ins().icmp_imm_s(IntCC::SignedLessThan, res, 0);
        builder.ins().brif(is_err, err_bb, &[], reload_bb, &[]);

        builder.switch_to_block(reload_bb);
        emit_promoted_counters_load(builder, promoted);
        let sp_r = builder.use_var(sp_cnt_var);
        let sp_next = builder.ins().iadd(sp_r, res);
        builder.def_var(sp_cnt_var, sp_next);
        builder.def_var(quantum_rem_var, res);
        builder.ins().jump(cont_bb, &[]);

        builder.switch_to_block(err_bb);
        let is_deopt = builder.ins().icmp_imm_s(IntCC::Equal, res, MIR_STATUS_DEOPT);
        builder
            .ins()
            .brif(is_deopt, deopt_bb, &[], raw_emergency_exit_bb, &[]);

        builder.switch_to_block(deopt_bb);
        let target_pc_val = builder.ins().iconst(types::I64, target_block_idx as i64);
        builder.ins().call(init_deopt_fn, &[target_pc_val]);
        for &loc_id in live_locals {
            if loc_id.0 < local_vars.len() {
                let loc_idx_val = builder.ins().iconst(types::I64, loc_id.0 as i64);
                let cur_val = if let Some(&slot) = stack_slots.get(&loc_id.0) {
                    let mir_ty = locals
                        .get(loc_id.0)
                        .map(|l| l.ty.clone())
                        .unwrap_or(MirType::Int);
                    let clif_ty = mir_to_clif_type(&mir_ty).unwrap_or(types::I64);
                    builder.ins().stack_load(types::I64, clif_ty, slot, 0)
                } else {
                    builder.use_var(local_vars[loc_id.0])
                };
                let val_ty = builder.func.dfg.value_type(cur_val);
                let val_i64 = if val_ty == types::I8 {
                    builder.ins().uextend(types::I64, cur_val)
                } else if val_ty == types::F64 {
                    builder.ins().bitcast(types::I64, MemFlagsData::new(), cur_val)
                } else {
                    cur_val
                };
                builder
                    .ins()
                    .call(record_deopt_local_fn, &[target_pc_val, loc_idx_val, val_i64]);
            }
        }
        builder.ins().jump(raw_emergency_exit_bb, &[]);

        builder.switch_to_block(cont_bb);
    }

    fn function_has_gas_checks(
        mir_fn: &MirFunction,
        vec_loops: &[crate::mir::opt::simd::VectorizedLoopInfo],
    ) -> bool {
        if vec_loops.iter().any(|vl| vl.gas_check.is_some()) {
            return true;
        }
        mir_fn.blocks.iter().any(|b| {
            b.statements
                .iter()
                .any(|s| matches!(s, Statement::ResourceCheck(ResourceKind::Gas, ..)))
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_inline_gas_check(
        builder: &mut FunctionBuilder,
        gas_ptr_var: Variable,
        gas_slow_fn: FuncRef,
        amt: u64,
        raw_emergency_exit_bb: Block,
        promoted: Option<PromotedLoopCounters>,
        mut cfg_rec: Option<&mut PromotedLoopCfgRecord>,
    ) {
        const OFF_GAS_REMAINING: i32 = core::mem::offset_of!(GasFastState, gas_remaining) as i32;

        let gas_slow_bb = builder.create_block();
        let cont_bb = builder.create_block();
        builder.set_cold_block(gas_slow_bb);

        if let Some(p) = promoted
            && let Some((_, gas_var)) = p.gas
        {
            // NG-W11-2 Stage 1: 2-instruction `iadd_imm_s` + `icmp_imm_s` for 0 < amt <= i32::MAX.
            // Prior check guarantees rem0 >= 0, so rem0 - amt >= -i32::MAX > i64::MIN (never wraps).
            if amt > 0 && amt <= (i32::MAX as u64) {
                let rem0 = builder.use_var(gas_var);
                let rem1 = builder.ins().iadd_imm_s(rem0, -(amt as i64));
                builder.def_var(gas_var, rem1);
                let is_neg = builder.ins().icmp_imm_s(IntCC::SignedLessThan, rem1, 0);
                builder.ins().brif(is_neg, gas_slow_bb, &[], cont_bb, &[]);
            } else if let Ok(amt_i64) = i64::try_from(amt) {
                let rem0 = builder.use_var(gas_var);
                let amt_val = builder.ins().iconst(types::I64, amt_i64);
                let (rem1, overflowed) = builder.ins().ssub_overflow(rem0, amt_val);
                let min_val = builder.ins().iconst(types::I64, i64::MIN);
                let rem_st = builder.ins().select(overflowed, min_val, rem1);
                builder.def_var(gas_var, rem_st);
                let is_lt_zero = builder.ins().icmp_imm_s(IntCC::SignedLessThan, rem_st, 0);
                let is_neg = builder.ins().bor(overflowed, is_lt_zero);
                builder.ins().brif(is_neg, gas_slow_bb, &[], cont_bb, &[]);
            } else {
                let min_val = builder.ins().iconst(types::I64, i64::MIN);
                builder.def_var(gas_var, min_val);
                builder.ins().jump(gas_slow_bb, &[]);
            }

            builder.switch_to_block(gas_slow_bb);
            // Barrier 2: flush BOTH promoted counters before calling lust_mir_gas_slow()
            emit_promoted_counters_flush(builder, p);
            let slow_call = builder.ins().call(gas_slow_fn, &[]);
            let slow_res = if builder.inst_results(slow_call).is_empty() {
                const OFF_GAS_SLOW_RESULT: i32 = core::mem::offset_of!(MirContext, gas_slow_result) as i32;
                let gas_ptr = builder.use_var(gas_ptr_var);
                builder
                    .ins()
                    .load(types::I64, MemFlagsData::trusted(), gas_ptr, OFF_GAS_SLOW_RESULT)
            } else {
                builder.inst_results(slow_call)[0]
            };
            let is_err = builder.ins().icmp_imm_s(IntCC::NotEqual, slow_res, 0);
            let reload_bb = builder.create_block();
            builder.set_cold_block(reload_bb);
            if let Some(rec) = cfg_rec.as_deref_mut() {
                rec.flush_clif_blocks.insert(gas_slow_bb);
                rec.reload_clif_blocks.insert(reload_bb);
            }
            builder.ins().brif(is_err, raw_emergency_exit_bb, &[], reload_bb, &[]);

            builder.switch_to_block(reload_bb);
            // Reload BOTH promoted counters (picks up unbounded gas replenishment!)
            emit_promoted_counters_load(builder, p);
            builder.ins().jump(cont_bb, &[]);

            builder.switch_to_block(cont_bb);
            return;
        }

        let gas_ptr = builder.use_var(gas_ptr_var);
        if let Ok(amt_i64) = i64::try_from(amt) {
            let rem0 = builder
                .ins()
                .load(types::I64, MemFlagsData::trusted(), gas_ptr, OFF_GAS_REMAINING);
            let amt_val = builder.ins().iconst(types::I64, amt_i64);
            let (rem1, overflowed) = builder.ins().ssub_overflow(rem0, amt_val);
            let min_val = builder.ins().iconst(types::I64, i64::MIN);
            let rem_st = builder.ins().select(overflowed, min_val, rem1);
            builder
                .ins()
                .store(MemFlagsData::trusted(), rem_st, gas_ptr, OFF_GAS_REMAINING);
            let is_lt_zero = builder.ins().icmp_imm_s(IntCC::SignedLessThan, rem_st, 0);
            let is_neg = builder.ins().bor(overflowed, is_lt_zero);
            builder.ins().brif(is_neg, gas_slow_bb, &[], cont_bb, &[]);
        } else {
            let min_val = builder.ins().iconst(types::I64, i64::MIN);
            builder
                .ins()
                .store(MemFlagsData::trusted(), min_val, gas_ptr, OFF_GAS_REMAINING);
            builder.ins().jump(gas_slow_bb, &[]);
        }

        builder.switch_to_block(gas_slow_bb);
        let slow_call = builder.ins().call(gas_slow_fn, &[]);
        let slow_res = if builder.inst_results(slow_call).is_empty() {
            const OFF_GAS_SLOW_RESULT: i32 = core::mem::offset_of!(MirContext, gas_slow_result) as i32;
            let gas_ptr = builder.use_var(gas_ptr_var);
            builder
                .ins()
                .load(types::I64, MemFlagsData::trusted(), gas_ptr, OFF_GAS_SLOW_RESULT)
        } else {
            builder.inst_results(slow_call)[0]
        };
        let is_err = builder.ins().icmp_imm_s(IntCC::NotEqual, slow_res, 0);
        builder.ins().brif(is_err, raw_emergency_exit_bb, &[], cont_bb, &[]);

        builder.switch_to_block(cont_bb);
    }

    /// Computes the default reachable root set for a `MirModule`:
    /// - Module entrypoints (`__script`, `__init@*`),
    /// - Exported / public functions (`signature.is_public == true`),
    /// - Any function referenced by an `Rvalue::Closure` in the module.
    ///
    /// If a synthetic module has no public/entry/closure roots at all, falls back conservatively
    /// to treating all functions in `module` as roots.
    pub fn compute_default_module_roots(module: &MirModule) -> hashbrown::HashSet<String> {
        let mut roots = hashbrown::HashSet::new();
        for f in &module.functions {
            if f.signature.is_public
                || f.signature.name == "__script"
                || f.signature.name.starts_with("__init@")
            {
                roots.insert(f.signature.name.clone());
            }
            for block in &f.blocks {
                for stmt in &block.statements {
                    if let Statement::Assign(_, Rvalue::Closure { func_name, .. }, _) = stmt {
                        roots.insert(func_name.clone());
                    }
                }
            }
        }
        if roots.is_empty() {
            for f in &module.functions {
                roots.insert(f.signature.name.clone());
            }
        }
        roots
    }

    /// Computes the transitive set of module-local functions reachable from `explicit_roots`
    /// via direct calls (`Terminator::Call`) and closure captures (`Rvalue::Closure`).
    ///
    /// Helper functions that were 100% inlined with 0 remaining callers and are not in
    /// `explicit_roots` are excluded from the returned set so Cranelift skips their codegen.
    pub fn compute_reachable_native_functions(
        module: &MirModule,
        explicit_roots: &hashbrown::HashSet<String>,
    ) -> hashbrown::HashSet<String> {
        let fn_by_name: hashbrown::HashMap<&str, &MirFunction> = module
            .functions
            .iter()
            .map(|f| (f.signature.name.as_str(), f))
            .collect();

        let mut reachable: hashbrown::HashSet<String> = hashbrown::HashSet::new();
        let mut worklist: Vec<&str> = Vec::new();

        for root in explicit_roots {
            if let Some(&f) = fn_by_name.get(root.as_str()) {
                if reachable.insert(f.signature.name.clone()) {
                    worklist.push(f.signature.name.as_str());
                }
            }
        }

        while let Some(cur_name) = worklist.pop() {
            let Some(&cur_fn) = fn_by_name.get(cur_name) else {
                continue;
            };
            for block in &cur_fn.blocks {
                for stmt in &block.statements {
                    if let Statement::Assign(_, Rvalue::Closure { func_name, .. }, _) = stmt
                        && let Some(&target_fn) = fn_by_name.get(func_name.as_str())
                        && reachable.insert(target_fn.signature.name.clone())
                    {
                        worklist.push(target_fn.signature.name.as_str());
                    }
                }
                if let Some(Terminator::Call { callee, args, .. }) = &block.terminator {
                    if callee == "__mir_unknown_call_method" {
                        if let Some(Operand::Constant(MirConstant::String(method_name))) =
                            args.get(1)
                        {
                            let colon_suffix = format!(":{method_name}");
                            let dot_suffix = format!(".{method_name}");
                            for cand in &module.functions {
                                if (cand.signature.name.ends_with(&colon_suffix)
                                    || cand.signature.name.ends_with(&dot_suffix))
                                    && reachable.insert(cand.signature.name.clone())
                                {
                                    worklist.push(cand.signature.name.as_str());
                                }
                            }
                        }
                    } else if !is_builtin_or_special_callee(callee)
                        && let Some(&target_fn) = fn_by_name.get(callee.as_str())
                        && reachable.insert(target_fn.signature.name.clone())
                    {
                        worklist.push(target_fn.signature.name.as_str());
                    }
                }
            }
        }

        reachable
    }

    #[derive(Default)]
    struct ModuleDispatchTables {
        fn_ids: hashbrown::HashMap<String, i64>,
        closure_capture_counts: hashbrown::HashMap<String, usize>,
        fn_signatures: hashbrown::HashMap<String, (Vec<MirType>, MirType)>,
        pic_struct_layouts: hashbrown::HashMap<String, MirStructLayout>,
        dynamic_unknown_params: hashbrown::HashMap<String, Vec<bool>>,
    }

    fn collect_explicit_dynamic_unknown_locals(
        mir_fn: &MirFunction,
        dispatch_tables: Option<&ModuleDispatchTables>,
    ) -> hashbrown::HashSet<usize> {
        let mut dynamic: hashbrown::HashSet<usize> = hashbrown::HashSet::new();
        for block in &mir_fn.blocks {
            if let Some(Terminator::Call {
                callee,
                args,
                destination,
                ..
            }) = &block.terminator
            {
                if matches!(
                    callee.as_str(),
                    "__mir_unknown_get_field"
                        | "__mir_unknown_call_method"
                        | "__mir_unknown_is_int"
                        | "__mir_unknown_unwrap_int"
                ) {
                    if let Some(Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc))) =
                        args.first()
                    {
                        dynamic.insert(loc.0);
                    }
                    if matches!(
                        callee.as_str(),
                        "__mir_unknown_get_field" | "__mir_unknown_call_method"
                    ) && let Place::Local(dst) = destination
                    {
                        dynamic.insert(dst.0);
                    }
                } else if let Some(dt) = dispatch_tables
                    && let Some(param_flags) = dt.dynamic_unknown_params.get(callee)
                {
                    for (idx, arg) in args.iter().enumerate() {
                        if param_flags.get(idx).copied().unwrap_or(false)
                            && let Operand::Copy(Place::Local(loc))
                            | Operand::Move(Place::Local(loc)) = arg
                        {
                            dynamic.insert(loc.0);
                        }
                    }
                }
            }
        }
        if dynamic.is_empty() {
            return dynamic;
        }
        loop {
            let mut changed = false;
            for block in &mir_fn.blocks {
                for stmt in &block.statements {
                    if let Statement::Assign(Place::Local(dst), rval, _) = stmt {
                        match rval {
                            Rvalue::Use(
                                Operand::Copy(Place::Local(src))
                                | Operand::Move(Place::Local(src)),
                            ) => {
                                if dynamic.contains(&src.0) && dynamic.insert(dst.0) {
                                    changed = true;
                                }
                                if dynamic.contains(&dst.0) && dynamic.insert(src.0) {
                                    changed = true;
                                }
                            }
                            Rvalue::BinaryOp(_, lhs, rhs) => {
                                let l_dyn = matches!(
                                    lhs,
                                    Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l))
                                        if dynamic.contains(&l.0)
                                );
                                let r_dyn = matches!(
                                    rhs,
                                    Operand::Copy(Place::Local(r)) | Operand::Move(Place::Local(r))
                                        if dynamic.contains(&r.0)
                                );
                                if (l_dyn || r_dyn) && dynamic.insert(dst.0) {
                                    changed = true;
                                }
                            }
                            Rvalue::UnaryOp(_, op) => {
                                let o_dyn = matches!(
                                    op,
                                    Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l))
                                        if dynamic.contains(&l.0)
                                );
                                if o_dyn && dynamic.insert(dst.0) {
                                    changed = true;
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        dynamic
    }

    fn build_module_dispatch_tables(
        refined_functions: &[MirFunction],
        module_struct_layouts: &[MirStructLayout],
        compiler_struct_layouts: &hashbrown::HashMap<String, MirStructLayout>,
    ) -> ModuleDispatchTables {
        let mut tables = ModuleDispatchTables::default();
        for layout in module_struct_layouts {
            tables
                .pic_struct_layouts
                .insert(layout.type_name.clone(), layout.clone());
        }
        for (k, v) in compiler_struct_layouts {
            tables.pic_struct_layouts.insert(k.clone(), v.clone());
        }
        for f in refined_functions {
            for layout in &f.struct_layouts {
                tables
                    .pic_struct_layouts
                    .entry(layout.type_name.clone())
                    .or_insert_with(|| layout.clone());
            }
        }

        let mut sorted_names: Vec<String> = refined_functions
            .iter()
            .map(|f| f.signature.name.clone())
            .collect();
        sorted_names.sort();
        for (idx, name) in sorted_names.into_iter().enumerate() {
            tables.fn_ids.insert(name, (idx as i64) + 1);
        }

        for f in refined_functions {
            let param_tys: Vec<MirType> = f
                .params
                .iter()
                .map(|pid| f.locals[pid.0].ty.clone())
                .collect();
            tables
                .fn_signatures
                .insert(f.signature.name.clone(), (param_tys, f.signature.return_type.clone()));
            for b in &f.blocks {
                for stmt in &b.statements {
                    if let Statement::Assign(
                        _,
                        Rvalue::Closure {
                            func_name,
                            captures,
                        },
                        _,
                    ) = stmt
                    {
                        tables
                            .closure_capture_counts
                            .insert(func_name.clone(), captures.len());
                    }
                }
            }
        }

        for f in refined_functions {
            tables
                .dynamic_unknown_params
                .insert(f.signature.name.clone(), vec![false; f.params.len()]);
        }
        loop {
            let mut changed = false;
            for f in refined_functions {
                let dyn_locals = collect_explicit_dynamic_unknown_locals(f, Some(&tables));
                let flags = tables
                    .dynamic_unknown_params
                    .get_mut(&f.signature.name)
                    .expect("initialized");
                for (i, pid) in f.params.iter().enumerate() {
                    if dyn_locals.contains(&pid.0) && !flags[i] {
                        flags[i] = true;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        tables
    }

    fn mir_type_from_canonical_tag_clif(tag: CanonicalTypeTag) -> types::Type {
        match tag {
            CanonicalTypeTag::Bool => types::I8,
            CanonicalTypeTag::Float64 => types::F64,
            _ => types::I64,
        }
    }

    pub struct CraneliftJitCompiler {
        builder_ctx: FunctionBuilderContext,
        ctx: Context,
        module: JITModule,
        pub struct_layouts: hashbrown::HashMap<String, MirStructLayout>,
        compiled_code_sizes: hashbrown::HashMap<FuncId, usize>,
        compiled_code_ranges: hashbrown::HashMap<String, (usize, usize)>,
        pub compiled_osr_entries: hashbrown::HashMap<String, *const u8>,
        osr_request_counter: u64,
        capture_ir: bool,
        captured_ir: hashbrown::HashMap<String, String>,
        captured_opt_ir: hashbrown::HashMap<String, String>,
        host_callbacks: hashbrown::HashMap<String, (Vec<MirType>, MirType)>,
        gas_budget_active: bool,
        post_inline_mirs: hashbrown::HashMap<String, MirFunction>,
        deferred_continuation_fns: hashbrown::HashSet<String>,
        deferred_dead_helpers: hashbrown::HashSet<String>,
        compiled_roots: hashbrown::HashSet<String>,
        skipped_dead_helper_count: usize,
        lazy_deopt_materialization_count: u64,
        preserve_all_safepoint_slow_id: Option<FuncId>,
        preserve_all_gas_slow_id: Option<FuncId>,
        #[cfg(feature = "test-deopt")]
        refine_max_iterations_override: Option<usize>,
    }

    impl CraneliftJitCompiler {
        pub fn new() -> Result<Self, String> {
            // SAFETY: Empty external callback slice registers no extra pointers.
            unsafe { Self::new_with_host_callbacks(&[]) }
        }

        /// Creates a `CraneliftJitCompiler` with additional C-ABI host callback symbols
        /// callable via `Terminator::Call` from compiled MIR functions.
        ///
        /// # Safety
        /// Every `(name, param_types, ret_type, fn_ptr)` entry must point to a valid
        /// `extern "C"` function matching the Cranelift ABI for `(param_types) -> ret_type`
        /// that outlives this compiler and any compiled code pointers.
        pub unsafe fn new_with_host_callbacks(
            callbacks: &[(&str, Vec<MirType>, MirType, *const u8)],
        ) -> Result<Self, String> {
            let mut flag_builder = settings::builder();
            flag_builder
                .set("opt_level", "speed")
                .map_err(|e| format!("{e}"))?;
            if !cfg!(debug_assertions) {
                flag_builder
                    .set("enable_verifier", "false")
                    .map_err(|e| format!("{e}"))?;
            }
            flag_builder
                .set("use_colocated_libcalls", "false")
                .map_err(|e| format!("{e}"))?;
            if cfg!(target_arch = "x86_64") {
                flag_builder
                    .set("is_pic", "true")
                    .map_err(|e| format!("{e}"))?;
            }

            let isa_builder = cranelift_native::builder()
                .map_err(|msg| format!("Host machine not supported by cranelift: {msg}"))?;
            let isa = isa_builder
                .finish(settings::Flags::new(flag_builder))
                .map_err(|e| format!("{e}"))?;

            let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
            builder.symbol_lookup_fn(Box::new(|name: &str| -> Option<*const u8> {
                match name {
                    "lust_mir_set_gas" => Some(lust_mir_set_gas as *const u8),
                    "lust_mir_get_gas" => Some(lust_mir_get_gas as *const u8),
                    "lust_mir_charge_gas" => Some(lust_mir_charge_gas as *const u8),
                    "lust_mir_gas_fast_ptr" => Some(lust_mir_gas_fast_ptr as *const u8),
                    "lust_mir_gas_slow" => Some(lust_mir_gas_slow as *const u8),
                    "lust_mir_alloc_array" => Some(lust_mir_alloc_array as *const u8),
                    "lust_mir_alloc_array_typed" => Some(lust_mir_alloc_array_typed as *const u8),
                    "lust_mir_alloc_tuple" => Some(lust_mir_alloc_tuple as *const u8),
                    "lust_mir_alloc_tuple_typed" => Some(lust_mir_alloc_tuple_typed as *const u8),
                    "lust_mir_alloc_struct" => Some(lust_mir_alloc_struct as *const u8),
                    "lust_mir_alloc_struct_named" => Some(lust_mir_alloc_struct_named as *const u8),
                    "lust_mir_alloc_cflat_struct" => Some(lust_mir_alloc_cflat_struct as *const u8),
                    "lust_mir_array_len" => Some(lust_mir_array_len as *const u8),
                    "lust_mir_array_len_checked" => Some(lust_mir_array_len_checked as *const u8),
                    "lust_mir_array_get" => Some(lust_mir_array_get as *const u8),
                    "lust_mir_array_set" => Some(lust_mir_array_set as *const u8),
                    "lust_mir_array_push" => Some(lust_mir_array_push as *const u8),
                    "lust_mir_floordiv_int" => Some(lust_mir_floordiv_int as *const u8),
                    "lust_mir_pow_int" => Some(lust_mir_pow_int as *const u8),
                    "lust_mir_abs_int" => Some(lust_mir_abs_int as *const u8),
                    "lust_mir_min_int" => Some(lust_mir_min_int as *const u8),
                    "lust_mir_max_int" => Some(lust_mir_max_int as *const u8),
                    "lust_mir_clamp_int" => Some(lust_mir_clamp_int as *const u8),
                    "lust_mir_get_status" => Some(lust_mir_get_status as *const u8),
                    "lust_mir_set_status" => Some(lust_mir_set_status as *const u8),
                    "lust_mir_print_int" => Some(lust_mir_print_int as *const u8),
                    "lust_mir_print_int_part" => Some(lust_mir_print_int_part as *const u8),
                    "lust_mir_print_float_part" => Some(lust_mir_print_float_part as *const u8),
                    "lust_mir_print_bool_part" => Some(lust_mir_print_bool_part as *const u8),
                    "lust_mir_print_string_part" => Some(lust_mir_print_string_part as *const u8),
                    "lust_mir_print_sep" => Some(lust_mir_print_sep as *const u8),
                    "lust_mir_print_nl" => Some(lust_mir_print_nl as *const u8),
                    "lust_mir_div_int" => Some(lust_mir_div_int as *const u8),
                    "lust_mir_rem_int" => Some(lust_mir_rem_int as *const u8),
                    "lust_mir_floordiv_float" => Some(lust_mir_floordiv_float as *const u8),
                    "lust_mir_pow_float" => Some(lust_mir_pow_float as *const u8),
                    "lust_mir_rem_float" => Some(lust_mir_rem_float as *const u8),
                    "lust_mir_string_concat" => Some(lust_mir_string_concat as *const u8),
                    "lust_mir_string_append_inplace" => Some(lust_mir_string_append_inplace as *const u8),
                    "lust_mir_string_eq" => Some(lust_mir_string_eq as *const u8),
                    "lust_mir_string_len" => Some(lust_mir_string_len as *const u8),
                    "lust_mir_string_sub" => Some(lust_mir_string_sub as *const u8),
                    "lust_mir_string_byte" => Some(lust_mir_string_byte as *const u8),
                    "lust_mir_string_byte_at" => Some(lust_mir_string_byte_at as *const u8),
                    "lust_mir_string_gsub" => Some(lust_mir_string_gsub as *const u8),
                    "lust_mir_string_find" => Some(lust_mir_string_find as *const u8),
                    "lust_mir_string_split" => Some(lust_mir_string_split as *const u8),
                    "lust_mir_clif_string_gsub" => Some(lust_mir_clif_string_gsub as *const u8),
                    "lust_mir_clif_string_split" => Some(lust_mir_clif_string_split as *const u8),
                    "lust_mir_clif_string_byte" => Some(lust_mir_clif_string_byte as *const u8),
                    "lust_mir_clif_string_sub_byte_array" => Some(lust_mir_clif_string_sub_byte_array as *const u8),
                    "lust_mir_clif_string_find" => Some(lust_mir_clif_string_find as *const u8),
                    "lust_mir_tostring_int" => Some(lust_mir_tostring_int as *const u8),
                    "lust_mir_tostring_float" => Some(lust_mir_tostring_float as *const u8),
                    "lust_mir_tostring_bool" => Some(lust_mir_tostring_bool as *const u8),
                    "lust_mir_alloc_map" => Some(lust_mir_alloc_map as *const u8),
                    "lust_mir_map_get" => Some(lust_mir_map_get as *const u8),
                    "lust_mir_map_set" => Some(lust_mir_map_set as *const u8),
                    "lust_mir_map_len" => Some(lust_mir_map_len as *const u8),
                    "lust_mir_map_set_typed" => Some(lust_mir_map_set_typed as *const u8),
                    "lust_mir_map_get_typed" => Some(lust_mir_map_get_typed as *const u8),
                    "lust_mir_map_has_typed" => Some(lust_mir_map_has_typed as *const u8),
                    "lust_mir_map_get_int_unwrap" => Some(lust_mir_map_get_int_unwrap as *const u8),
                    "lust_mir_alloc_enum" => Some(lust_mir_alloc_enum as *const u8),
                    "lust_mir_alloc_enum2" => Some(lust_mir_alloc_enum2 as *const u8),
                    "lust_mir_enum_tag" => Some(lust_mir_enum_tag as *const u8),
                    "lust_mir_enum_payload" => Some(lust_mir_enum_payload as *const u8),
                    "lust_mir_enum_payload_at" => Some(lust_mir_enum_payload_at as *const u8),
                    "lust_mir_queue_new" => Some(lust_mir_queue_new as *const u8),
                    "lust_mir_queue_offer_int" => Some(lust_mir_queue_offer_int as *const u8),
                    "lust_mir_queue_take_int" => Some(lust_mir_queue_take_int as *const u8),
                    "lust_mir_unknown_is_int" => Some(lust_mir_unknown_is_int as *const u8),
                    "lust_mir_unknown_unwrap_int" => Some(lust_mir_unknown_unwrap_int as *const u8),
                    "lust_mir_clif_unknown_type_id" => Some(lust_mir_clif_unknown_type_id as *const u8),
                    "lust_mir_clif_unknown_get_field_fallback" => Some(lust_mir_clif_unknown_get_field_fallback as *const u8),
                    "lust_mir_safepoint" => Some(lust_mir_safepoint as *const u8),
                    "lust_mir_quantized_refill_slow" => Some(lust_mir_quantized_refill_slow as *const u8),
                    "lust_mir_safepoint_state_ptr" => Some(lust_mir_safepoint_state_ptr as *const u8),
                    "lust_mir_safepoint_slow" => Some(lust_mir_safepoint_slow as *const u8),
                    "lust_mir_init_deopt" => Some(lust_mir_init_deopt as *const u8),
                    "lust_mir_init_deopt_at_terminator" => Some(lust_mir_init_deopt_at_terminator as *const u8),
                    "lust_mir_record_deopt_local" => Some(lust_mir_record_deopt_local as *const u8),
                    "lust_mir_record_deopt" => Some(lust_mir_record_deopt as *const u8),
                    "lust_mir_enter_native_callee" => Some(lust_mir_enter_native_callee as *const u8),
                    "lust_mir_exit_native_callee" => Some(lust_mir_exit_native_callee as *const u8),
                    "lust_mir_ic_probe" => Some(lust_mir_ic_probe as *const u8),
                    "lust_mir_ic_miss" => Some(lust_mir_ic_miss as *const u8),
                    "lust_mir_ic_dispatch" => Some(lust_mir_ic_dispatch as *const u8),
                    "lust_mir_struct_decode_contiguous" => Some(lust_mir_struct_decode_contiguous as *const u8),
                    "lust_mir_struct_decode_binary" => Some(lust_mir_struct_decode_binary as *const u8),
                    "lust_mir_record_native_entry" => Some(lust_mir_record_native_entry as *const u8),
                    "lust_mir_record_simd_vector_op" => Some(lust_mir_record_simd_vector_op as *const u8),
                    "lust_mir_record_simd_scalar_op" => Some(lust_mir_record_simd_scalar_op as *const u8),
                    "lust_mir_is_simd_scalar_forced" => Some(lust_mir_is_simd_scalar_forced as *const u8),
                    _ => None,
                }
            }));

            let mut host_callbacks = hashbrown::HashMap::new();
            for (name, params, ret, ptr) in callbacks {
                builder.symbol(*name, *ptr);
                host_callbacks.insert((*name).to_string(), (params.clone(), ret.clone()));
            }

            let module = JITModule::new(builder);

            Ok(Self {
                builder_ctx: FunctionBuilderContext::new(),
                ctx: module.make_context(),
                module,
                struct_layouts: hashbrown::HashMap::new(),
                compiled_code_sizes: hashbrown::HashMap::new(),
                compiled_code_ranges: hashbrown::HashMap::new(),
                compiled_osr_entries: hashbrown::HashMap::new(),
                osr_request_counter: 0,
                capture_ir: false,
                captured_ir: hashbrown::HashMap::new(),
                captured_opt_ir: hashbrown::HashMap::new(),
                host_callbacks,
                gas_budget_active: false,
                post_inline_mirs: hashbrown::HashMap::new(),
                deferred_continuation_fns: hashbrown::HashSet::new(),
                deferred_dead_helpers: hashbrown::HashSet::new(),
                compiled_roots: hashbrown::HashSet::new(),
                skipped_dead_helper_count: 0,
                lazy_deopt_materialization_count: 0,
                preserve_all_safepoint_slow_id: None,
                preserve_all_gas_slow_id: None,
                #[cfg(feature = "test-deopt")]
                refine_max_iterations_override: None,
            })
        }

        fn compile_preserve_all_slow_trampoline(
            &mut self,
            tramp_name: &str,
            target_c_sym: &str,
            state_ptr_sym: &str,
            result_offset: i32,
        ) -> Result<FuncId, MirCraneliftBackendError> {
            // Cranelift `CallConv::PreserveAll` forbids return values (all registers are preserved),
            // so the cold trampoline stores the C-ABI i64 return value into the thread-confined
            // fast state struct (`SafepointFastState::slow_result` / `MirContext::gas_slow_result`).
            let tramp_sig = Signature::new(cranelift_codegen::isa::CallConv::PreserveAll);
            let tramp_id = self
                .module
                .declare_function(tramp_name, Linkage::Local, &tramp_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let mut c_sig = self.module.make_signature();
            c_sig.returns.push(AbiParam::new(types::I64));
            let c_callee = self
                .module
                .declare_function(target_c_sym, Linkage::Import, &c_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;
            let state_ptr_callee = self
                .module
                .declare_function(state_ptr_sym, Linkage::Import, &c_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            self.module.clear_context(&mut self.ctx);
            self.builder_ctx = FunctionBuilderContext::new();
            self.ctx.func.signature = tramp_sig;

            {
                let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);
                let c_fn = self.module.declare_func_in_func(c_callee, &mut builder.func);
                let state_ptr_fn = self
                    .module
                    .declare_func_in_func(state_ptr_callee, &mut builder.func);
                let entry_bb = builder.create_block();
                builder.switch_to_block(entry_bb);
                let call_inst = builder.ins().call(c_fn, &[]);
                let res = builder.inst_results(call_inst)[0];
                let ptr_call = builder.ins().call(state_ptr_fn, &[]);
                let state_ptr = builder.inst_results(ptr_call)[0];
                builder
                    .ins()
                    .store(MemFlagsData::trusted(), res, state_ptr, result_offset);
                builder.ins().return_(&[]);
                builder.seal_all_blocks();
                builder.finalize(self.module.target_config());
            }

            self.module
                .define_function(tramp_id, &mut self.ctx)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e:?}")))?;
            let code_size = self
                .ctx
                .compiled_code()
                .map(|c| c.code_info().total_size as usize)
                .unwrap_or(1);
            self.compiled_code_sizes.insert(tramp_id, code_size);
            self.module.clear_context(&mut self.ctx);
            Ok(tramp_id)
        }

        fn ensure_preserve_all_slow_trampolines(
            &mut self,
        ) -> Result<(FuncId, FuncId), MirCraneliftBackendError> {
            if let (Some(sp_id), Some(gas_id)) = (
                self.preserve_all_safepoint_slow_id,
                self.preserve_all_gas_slow_id,
            ) {
                return Ok((sp_id, gas_id));
            }
            const OFF_SP_RESULT: i32 = core::mem::offset_of!(SafepointFastState, slow_result) as i32;
            const OFF_GAS_RESULT: i32 = core::mem::offset_of!(MirContext, gas_slow_result) as i32;
            let sp_id = self.compile_preserve_all_slow_trampoline(
                "__clif_preserve_all_safepoint_slow",
                "lust_mir_safepoint_slow",
                "lust_mir_safepoint_state_ptr",
                OFF_SP_RESULT,
            )?;
            let gas_id = self.compile_preserve_all_slow_trampoline(
                "__clif_preserve_all_gas_slow",
                "lust_mir_gas_slow",
                "lust_mir_gas_fast_ptr",
                OFF_GAS_RESULT,
            )?;
            self.preserve_all_safepoint_slow_id = Some(sp_id);
            self.preserve_all_gas_slow_id = Some(gas_id);
            Ok((sp_id, gas_id))
        }

        pub fn set_gas_budget_active(&mut self, active: bool) {
            self.gas_budget_active = active;
        }

        /// Stores the lightweight post-inline MIR snapshot, deferred continuation/dead-helper
        /// indices, and active root set for lazy deopt materialization and on-demand DCE expansion.
        pub fn set_post_inline_snapshot(
            &mut self,
            post_inline_mirs: hashbrown::HashMap<String, MirFunction>,
            deferred_continuation_fns: hashbrown::HashSet<String>,
            deferred_dead_helpers: hashbrown::HashSet<String>,
            compiled_roots: hashbrown::HashSet<String>,
            skipped_dead_helper_count: usize,
        ) {
            self.post_inline_mirs = post_inline_mirs;
            self.deferred_continuation_fns = deferred_continuation_fns;
            self.deferred_dead_helpers = deferred_dead_helpers;
            self.compiled_roots = compiled_roots;
            self.skipped_dead_helper_count = skipped_dead_helper_count;
            self.lazy_deopt_materialization_count = 0;
        }

        pub fn post_inline_mir(&self, name: &str) -> Option<&MirFunction> {
            self.post_inline_mirs.get(name)
        }

        pub fn is_deferred_continuation(&self, name: &str) -> bool {
            self.deferred_continuation_fns.contains(name)
        }

        pub fn mark_continuation_materialized(&mut self, name: &str) {
            self.deferred_continuation_fns.remove(name);
        }

        pub fn is_deferred_dead_helper(&self, name: &str) -> bool {
            self.deferred_dead_helpers.contains(name)
        }

        pub fn compiled_roots(&self) -> &hashbrown::HashSet<String> {
            &self.compiled_roots
        }

        pub fn skipped_dead_helper_count(&self) -> usize {
            self.skipped_dead_helper_count
        }

        pub fn lazy_deopt_materialization_count(&self) -> u64 {
            self.lazy_deopt_materialization_count
        }

        pub fn record_lazy_deopt_materialization(&mut self) {
            self.lazy_deopt_materialization_count += 1;
        }

        /// Configures an explicit type refinement maximum iteration override for testing convergence boundaries.
        /// Available only under the `test-deopt` feature.
        #[cfg(feature = "test-deopt")]
        pub fn set_refine_max_iterations_override_for_test(&mut self, max_iterations: Option<usize>) {
            self.refine_max_iterations_override = max_iterations;
        }

        /// Helper to refine a MIR function using this compiler's configured struct layouts and test seam overrides.
        #[inline]
        fn refine_function_for_compiler(&self, f: &MirFunction) -> MirFunction {
            #[cfg(feature = "test-deopt")]
            {
                refine_mir_function_internal(f, &self.struct_layouts, self.refine_max_iterations_override)
            }
            #[cfg(not(feature = "test-deopt"))]
            {
                refine_mir_function_internal(f, &self.struct_layouts, None)
            }
        }

        pub fn compiled_code_range(&self, name: &str) -> Option<(usize, usize)> {
            self.compiled_code_ranges.get(name).copied()
        }

        pub fn isa_opt_level(&self) -> settings::OptLevel {
            self.module.isa().flags().opt_level()
        }

        pub fn isa_opt_level_str(&self) -> &'static str {
            match self.isa_opt_level() {
                settings::OptLevel::None => "none",
                settings::OptLevel::Speed => "speed",
                settings::OptLevel::SpeedAndSize => "speed_and_size",
            }
        }

        pub fn dump_function_ir(&mut self, mir_fn: &MirFunction) -> Result<String, MirCraneliftBackendError> {
            let prev = self.capture_ir;
            self.capture_ir = true;
            let compile_res = self.compile_function(mir_fn);
            self.capture_ir = prev;
            compile_res?;
            let fast_key = format!("{}$fast", mir_fn.signature.name);
            self.captured_ir
                .get(&fast_key)
                .or_else(|| self.captured_ir.get(&mir_fn.signature.name))
                .cloned()
                .ok_or_else(|| {
                    MirCraneliftBackendError::CodegenError(format!(
                        "No IR captured for function '{}'",
                        mir_fn.signature.name
                    ))
                })
        }

        pub fn dump_optimized_function_ir(
            &mut self,
            mir_fn: &MirFunction,
        ) -> Result<String, MirCraneliftBackendError> {
            let prev = self.capture_ir;
            self.capture_ir = true;
            let compile_res = self.compile_function(mir_fn);
            self.capture_ir = prev;
            compile_res?;
            let fast_key = format!("{}$fast", mir_fn.signature.name);
            self.captured_opt_ir
                .get(&fast_key)
                .or_else(|| self.captured_opt_ir.get(&mir_fn.signature.name))
                .cloned()
                .ok_or_else(|| {
                    MirCraneliftBackendError::CodegenError(format!(
                        "No optimized IR captured for function '{}'",
                        mir_fn.signature.name
                    ))
                })
        }

        pub fn dump_module_ir(
            &mut self,
            module: &MirModule,
        ) -> Result<hashbrown::HashMap<String, String>, MirCraneliftBackendError> {
            let prev = self.capture_ir;
            self.captured_ir.clear();
            self.capture_ir = true;
            let compile_res = self.compile_module(module);
            self.capture_ir = prev;
            compile_res?;
            Ok(self.captured_ir.clone())
        }

        pub fn dump_osr_entry_ir(
            &mut self,
            mir_fn: &MirFunction,
            target_block: BlockId,
            live_locals: &[LocalId],
        ) -> Result<String, MirCraneliftBackendError> {
            let prev = self.capture_ir;
            self.captured_ir.clear();
            self.capture_ir = true;
            let compile_res = self.compile_osr_entry(mir_fn, target_block, live_locals);
            self.capture_ir = prev;
            compile_res?;
            self.captured_ir
                .get(&mir_fn.signature.name)
                .cloned()
                .ok_or_else(|| {
                    MirCraneliftBackendError::CodegenError(format!(
                        "No OSR IR captured for function '{}'",
                        mir_fn.signature.name
                    ))
                })
        }

        fn format_captured_ir(&self) -> String {
            let mut ir_text = format!("{}", self.ctx.func.display());
            let decls = self.module.declarations();
            for (func_ref, ext_data) in self.ctx.func.dfg.ext_funcs.iter() {
                if let cranelift_codegen::ir::ExternalName::User(uref) = ext_data.name {
                    let uname = &self.ctx.func.params.user_named_funcs()[uref];
                    if uname.namespace == 0 {
                        let fid = FuncId::from_u32(uname.index);
                        if let Some(sym_name) = &decls.get_function_decl(fid).name {
                            ir_text.push_str(&format!("\n; sym {func_ref} = {sym_name}"));
                        }
                    }
                }
            }
            ir_text
        }

        fn compile_export_wrapper(
            &mut self,
            export_name: &str,
            sig: &Signature,
            export_func_id: FuncId,
            fast_func_id: FuncId,
        ) -> Result<(), MirCraneliftBackendError> {
            self.module.clear_context(&mut self.ctx);
            self.builder_ctx = FunctionBuilderContext::new();
            self.ctx.func.signature = sig.clone();

            let sig_void0 = self.module.make_signature();
            let native_entry_callee = self
                .module
                .declare_function("lust_mir_record_native_entry", Linkage::Import, &sig_void0)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            {
                let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);
                let native_entry_fn = self
                    .module
                    .declare_func_in_func(native_entry_callee, &mut builder.func);
                let fast_fn = self
                    .module
                    .declare_func_in_func(fast_func_id, &mut builder.func);

                let entry_bb = builder.create_block();
                builder.append_block_params_for_function_params(entry_bb);
                builder.switch_to_block(entry_bb);
                builder.ins().call(native_entry_fn, &[]);

                let params: Vec<ClifValue> = builder.block_params(entry_bb).to_vec();
                let call_inst = builder.ins().call(fast_fn, &params);
                let results: Vec<ClifValue> = builder.inst_results(call_inst).to_vec();
                builder.ins().return_(&results);

                builder.seal_all_blocks();
                builder.finalize(self.module.target_config());
            }

            if self.capture_ir {
                let ir_text = self.format_captured_ir();
                self.captured_ir.insert(export_name.to_string(), ir_text);
            }

            self.module
                .define_function(export_func_id, &mut self.ctx)
                .map_err(|e| {
                    eprintln!("Verifier error in define_function (wrapper {export_name}): {:?}", e);
                    MirCraneliftBackendError::CodegenError(format!("{e:?}"))
                })?;
            let code_size = self
                .ctx
                .compiled_code()
                .map(|c| c.code_info().total_size as usize)
                .unwrap_or(1);
            self.compiled_code_sizes.insert(export_func_id, code_size);
            self.module.clear_context(&mut self.ctx);
            Ok(())
        }

        pub fn register_struct_layouts(&mut self, layouts: &[MirStructLayout]) {
            for layout in layouts {
                self.struct_layouts.insert(layout.type_name.clone(), layout.clone());
                register_struct_layout(layout.clone());
            }
        }

        pub fn compile_function(&mut self, mir_fn: &MirFunction) -> Result<*const u8, MirCraneliftBackendError> {
            let mut module = MirModule::new(mir_fn.signature.name.clone());
            module.add_function(mir_fn.clone());
            let mut map = self.compile_module(&module)?;
            map.remove(&mir_fn.signature.name)
                .ok_or_else(|| MirCraneliftBackendError::CodegenError("Failed to get compiled function pointer".into()))
        }

        pub fn compile_module(
            &mut self,
            module: &MirModule,
        ) -> Result<hashbrown::HashMap<String, *const u8>, MirCraneliftBackendError> {
            let roots = compute_default_module_roots(module);
            self.compile_module_with_roots(module, &roots)
        }

        pub fn compile_module_with_roots(
            &mut self,
            module: &MirModule,
            roots: &hashbrown::HashSet<String>,
        ) -> Result<hashbrown::HashMap<String, *const u8>, MirCraneliftBackendError> {
            self.register_struct_layouts(&module.struct_layouts);

            let reachable = compute_reachable_native_functions(module, roots);
            let skipped = module
                .functions
                .iter()
                .filter(|f| !reachable.contains(&f.signature.name))
                .count();
            self.skipped_dead_helper_count = skipped;

            let live_functions: Vec<&MirFunction> = module
                .functions
                .iter()
                .filter(|f| reachable.contains(&f.signature.name))
                .collect();

            let mut fast_funcs = hashbrown::HashMap::new();
            let callee_rets: hashbrown::HashMap<String, MirType> = live_functions
                .iter()
                .map(|f| (f.signature.name.clone(), f.signature.return_type.clone()))
                .collect();
            let mut refined_functions = Vec::with_capacity(live_functions.len());
            for f in &live_functions {
                let mut f_with_calls = (*f).clone();
                apply_module_callee_return_types(&mut f_with_calls, &callee_rets);
                refined_functions.push(self.refine_function_for_compiler(&f_with_calls));
            }

            // Compute compile-time fixed-point transitive deopt & fail effects before lowering.
            let call_effects =
                analyze_refined_module_call_effects(&refined_functions, self.gas_budget_active);

            // Identify functions that are targets of at least one module-local `Terminator::Call`,
            // `Rvalue::Closure`, or `__mir_unknown_call_method` dynamic dispatch site.
            // Functions with internal callers get a dual entrypoint (`F` Export wrapper -> `F$fast` Local body)
            // so internal calls bypass `lust_mir_record_native_entry`. Top-level-only functions (including
            // standalone SIMD kernels inspected via `compiled_code_range`) compile directly as `F` with
            // zero wrapper trampoline overhead while still calling internal callees via `G$fast`.
            let known_fn_names: hashbrown::HashSet<&str> = refined_functions
                .iter()
                .map(|f| f.signature.name.as_str())
                .collect();
            let mut internally_called: hashbrown::HashSet<String> = hashbrown::HashSet::new();
            for f in &refined_functions {
                for b in &f.blocks {
                    for stmt in &b.statements {
                        if let Statement::Assign(_, Rvalue::Closure { func_name, .. }, _) = stmt
                            && known_fn_names.contains(func_name.as_str())
                        {
                            internally_called.insert(func_name.clone());
                        }
                    }
                    if let Some(Terminator::Call { callee, args, .. }) = &b.terminator {
                        if callee == "__mir_unknown_call_method" {
                            if let Some(Operand::Constant(MirConstant::String(method_name))) =
                                args.get(1)
                            {
                                let colon_suffix = format!(":{method_name}");
                                let dot_suffix = format!(".{method_name}");
                                for &cand_name in &known_fn_names {
                                    if cand_name.ends_with(&colon_suffix)
                                        || cand_name.ends_with(&dot_suffix)
                                    {
                                        internally_called.insert(cand_name.to_string());
                                    }
                                }
                            }
                        } else if !is_builtin_or_special_callee(callee)
                            && known_fn_names.contains(callee.as_str())
                        {
                            internally_called.insert(callee.clone());
                        }
                    }
                }
            }

            let dispatch_tables = build_module_dispatch_tables(
                &refined_functions,
                &module.struct_layouts,
                &self.struct_layouts,
            );
            // Ensure non-CFlat struct field types from `MirFunction::struct_layouts` are visible
            // to `try_place_type` even during standalone `CraneliftJitCompiler::compile_module` calls.
            with_current_mir_context(|ctx| {
                for (sname, layout) in &dispatch_tables.pic_struct_layouts {
                    if !self.struct_layouts.contains_key(sname)
                        && ctx.get_struct_field_type(sname, 0).is_none()
                    {
                        let field_tys: Vec<MirType> = layout
                            .fields
                            .iter()
                            .map(|f| match f.tag {
                                CanonicalTypeTag::Bool => MirType::Bool,
                                CanonicalTypeTag::Float64 => MirType::Float,
                                CanonicalTypeTag::StringSlice => MirType::String,
                                _ => MirType::Int,
                            })
                            .collect();
                        ctx.register_struct_field_types(sname, field_tys);
                    }
                }
            });

            // 1. Declare all functions first so they can reference each other
            let mut export_entries: Vec<(String, Option<(String, FuncId)>, FuncId, Signature)> =
                Vec::with_capacity(refined_functions.len());
            for mir_fn in &refined_functions {
                let dyn_unknown_locals =
                    collect_explicit_dynamic_unknown_locals(mir_fn, Some(&dispatch_tables));
                let mut sig = self.module.make_signature();
                for param_id in &mir_fn.params {
                    let p_ty = &mir_fn.locals[param_id.0].ty;
                    if *p_ty == MirType::Unknown && !dyn_unknown_locals.contains(&param_id.0) {
                        return Err(MirCraneliftBackendError::UnsupportedType(MirType::Unknown));
                    }
                    sig.params.push(AbiParam::new(mir_to_clif_type(p_ty)?));
                }

                if mir_fn.signature.return_type == MirType::Unknown
                    && !dyn_unknown_locals.contains(&mir_fn.return_local.0)
                {
                    return Err(MirCraneliftBackendError::UnsupportedType(MirType::Unknown));
                }
                let ret_clif_ty = mir_to_clif_type(&mir_fn.signature.return_type)?;
                if ret_clif_ty != types::I8 || mir_fn.signature.return_type != MirType::Unit {
                    sig.returns.push(AbiParam::new(ret_clif_ty));
                }

                let export_func_id = self
                    .module
                    .declare_function(&mir_fn.signature.name, Linkage::Export, &sig)
                    .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

                if internally_called.contains(&mir_fn.signature.name) {
                    let fast_name = format!("{}$fast", mir_fn.signature.name);
                    let fast_func_id = self
                        .module
                        .declare_function(&fast_name, Linkage::Local, &sig)
                        .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;
                    fast_funcs.insert(mir_fn.signature.name.clone(), fast_func_id);
                    export_entries.push((
                        mir_fn.signature.name.clone(),
                        Some((fast_name, fast_func_id)),
                        export_func_id,
                        sig,
                    ));
                } else {
                    fast_funcs.insert(mir_fn.signature.name.clone(), export_func_id);
                    export_entries.push((
                        mir_fn.signature.name.clone(),
                        None,
                        export_func_id,
                        sig,
                    ));
                }
            }

            // 2. Compile each function definition
            for (mir_fn, (export_name, fast_opt, export_func_id, sig)) in
                refined_functions.iter().zip(export_entries.iter())
            {
                if let Some((fast_name, fast_func_id)) = fast_opt {
                    self.compile_function_with_options(
                        mir_fn,
                        *fast_func_id,
                        &fast_funcs,
                        None,
                        false,
                        Some(&call_effects),
                        Some(fast_name.as_str()),
                        Some(&dispatch_tables),
                    )?;
                    self.compile_export_wrapper(
                        export_name,
                        sig,
                        *export_func_id,
                        *fast_func_id,
                    )?;
                } else {
                    self.compile_function_with_options(
                        mir_fn,
                        *export_func_id,
                        &fast_funcs,
                        None,
                        true,
                        Some(&call_effects),
                        Some(export_name.as_str()),
                        Some(&dispatch_tables),
                    )?;
                }
            }

            // 3. Finalize definitions
            self.module
                .finalize_definitions()
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            // 4. Collect finalized code pointers (public export `F`) and code ranges
            let mut res = hashbrown::HashMap::new();
            for (export_name, fast_opt, export_func_id, _) in export_entries {
                let code_ptr = self.module.get_finalized_function(export_func_id);
                let size = self
                    .compiled_code_sizes
                    .get(&export_func_id)
                    .copied()
                    .unwrap_or(1);
                let start = code_ptr as usize;
                self.compiled_code_ranges.insert(
                    export_name.clone(),
                    (start, start.saturating_add(size.max(1))),
                );

                if let Some((fast_name, fast_func_id)) = fast_opt {
                    let fast_ptr = self.module.get_finalized_function(fast_func_id);
                    let fast_size = self
                        .compiled_code_sizes
                        .get(&fast_func_id)
                        .copied()
                        .unwrap_or(1);
                    let fast_start = fast_ptr as usize;
                    self.compiled_code_ranges.insert(
                        fast_name,
                        (fast_start, fast_start.saturating_add(fast_size.max(1))),
                    );
                }

                res.insert(export_name, code_ptr);
            }

            Ok(res)
        }

        pub fn compile_osr_entry(
            &mut self,
            mir_fn: &MirFunction,
            target_block: BlockId,
            live_locals: &[LocalId],
        ) -> Result<*const u8, MirCraneliftBackendError> {
            let mut module = MirModule::new(mir_fn.signature.name.clone());
            module.add_function(mir_fn.clone());
            self.compile_osr_entry_in_module(&module, &mir_fn.signature.name, target_block, live_locals)
        }

        pub fn compile_osr_entry_in_module(
            &mut self,
            module: &MirModule,
            func_name: &str,
            target_block: BlockId,
            live_locals: &[LocalId],
        ) -> Result<*const u8, MirCraneliftBackendError> {
            self.register_struct_layouts(&module.struct_layouts);
            let mir_fn = module
                .functions
                .iter()
                .find(|f| f.signature.name == func_name)
                .ok_or_else(|| {
                    MirCraneliftBackendError::CodegenError(format!("Function '{func_name}' not found in module"))
                })?;

            // Pre-validation: ensure function names in module are unique
            let mut seen_fn_names = hashbrown::HashSet::new();
            for f in &module.functions {
                if !seen_fn_names.insert(&f.signature.name) {
                    return Err(MirCraneliftBackendError::CodegenError(format!(
                        "Duplicate function name '{}' in module",
                        f.signature.name
                    )));
                }
            }

            // Pre-validation: validate call terminators to module functions for arity and type matching
            self.validate_module_calls_prior_to_mutation(module)?;

            let refined = self.refine_function_for_compiler(mir_fn);

            // Fresh uniquely named compilation per request to eliminate stale code reuse
            // and prevent Cranelift DuplicateDefinition errors on repeated compilations of same symbol.
            let req_id = self.osr_request_counter;
            self.osr_request_counter = self.osr_request_counter.wrapping_add(1);
            let osr_name = format!("{}_osr_{}_req{}", func_name, target_block.0, req_id);

            // Check if func_name is called by any function in the module (direct/mutual recursion).
            let func_is_called = module.functions.iter().any(|f| {
                f.blocks.iter().any(|b| {
                    if let Some(Terminator::Call { callee, .. }) = &b.terminator {
                        callee == func_name
                    } else {
                        false
                    }
                })
            });

            let mut module_funcs = hashbrown::HashMap::new();
            let mut declared_sigs = hashbrown::HashMap::new();
            let mut defined_fids = hashbrown::HashSet::new();

            // 1a. If func_name is called recursively, declare a versioned baseline entry for it.
            if func_is_called {
                let mut base_sig = self.module.make_signature();
                for param_id in &mir_fn.params {
                    let p_ty = &mir_fn.locals[param_id.0].ty;
                    base_sig.params.push(AbiParam::new(mir_to_clif_type(p_ty)?));
                }
                let ret_clif_ty = mir_to_clif_type(&mir_fn.signature.return_type)?;
                if ret_clif_ty != types::I8 || mir_fn.signature.return_type != MirType::Unit {
                    base_sig.returns.push(AbiParam::new(ret_clif_ty));
                }
                let base_name = format!("{}_base_req{}", func_name, req_id);
                let base_func_id = self
                    .module
                    .declare_function(&base_name, Linkage::Export, &base_sig)
                    .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;
                declared_sigs.insert(base_func_id, base_sig);
                module_funcs.insert(func_name.to_string(), base_func_id);
            }

            // 1b. Declare other module functions with versioned symbols so OSR entry can call them
            for f in &module.functions {
                if f.signature.name == func_name {
                    continue;
                }
                let refined_f = self.refine_function_for_compiler(f);
                let mut sig = self.module.make_signature();
                for param_id in &refined_f.params {
                    let p_ty = &refined_f.locals[param_id.0].ty;
                    sig.params.push(AbiParam::new(mir_to_clif_type(p_ty)?));
                }
                let ret_clif_ty = mir_to_clif_type(&refined_f.signature.return_type)?;
                if ret_clif_ty != types::I8 || refined_f.signature.return_type != MirType::Unit {
                    sig.returns.push(AbiParam::new(ret_clif_ty));
                }
                let aux_name = format!("{}_osr_req{}_aux_{}", func_name, req_id, f.signature.name);
                let func_id = self
                    .module
                    .declare_function(&aux_name, Linkage::Export, &sig)
                    .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;
                declared_sigs.insert(func_id, sig);
                module_funcs.insert(f.signature.name.clone(), func_id);
            }

            // 2. Declare OSR entry function taking i64 parameters for all live locals
            let mut osr_sig = self.module.make_signature();
            for _ in live_locals {
                osr_sig.params.push(AbiParam::new(types::I64));
            }
            let ret_clif_ty = mir_to_clif_type(&refined.signature.return_type)?;
            if ret_clif_ty != types::I8 || refined.signature.return_type != MirType::Unit {
                osr_sig.returns.push(AbiParam::new(ret_clif_ty));
            }
            let osr_func_id = self
                .module
                .declare_function(&osr_name, Linkage::Export, &osr_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;
            declared_sigs.insert(osr_func_id, osr_sig);

            // 3a. Compile base entry of func_name if called recursively
            if func_is_called {
                let base_func_id = module_funcs[func_name];
                let refined_base = self.refine_function_for_compiler(mir_fn);
                if let Err(err) = self.compile_function_internal(&refined_base, base_func_id, &module_funcs) {
                    self.cleanup_unfinalized_declarations(&declared_sigs, &defined_fids);
                    return Err(err);
                }
                defined_fids.insert(base_func_id);
            }

            // 3b. Compile other module functions
            for other_fn in &module.functions {
                if other_fn.signature.name != func_name {
                    let fid = module_funcs[&other_fn.signature.name];
                    let refined_other = self.refine_function_for_compiler(other_fn);
                    if let Err(err) = self.compile_function_internal(&refined_other, fid, &module_funcs) {
                        self.cleanup_unfinalized_declarations(&declared_sigs, &defined_fids);
                        return Err(err);
                    }
                    defined_fids.insert(fid);
                }
            }

            // 4. Compile OSR entry
            if let Err(err) = self.compile_function_with_osr(
                &refined,
                osr_func_id,
                &module_funcs,
                Some((target_block, live_locals)),
            ) {
                self.cleanup_unfinalized_declarations(&declared_sigs, &defined_fids);
                return Err(err);
            }
            defined_fids.insert(osr_func_id);

            // 5. Finalize definitions
            if let Err(e) = self.module.finalize_definitions() {
                self.cleanup_unfinalized_declarations(&declared_sigs, &defined_fids);
                return Err(MirCraneliftBackendError::CodegenError(format!("{e}")));
            }

            let code_ptr = self.module.get_finalized_function(osr_func_id);
            let osr_size = self.compiled_code_sizes.get(&osr_func_id).copied().unwrap_or(1);
            let osr_start = code_ptr as usize;
            let osr_range = (osr_start, osr_start.saturating_add(osr_size.max(1)));
            self.compiled_code_ranges.insert(
                osr_name.clone(),
                osr_range,
            );
            self.compiled_code_ranges.insert(
                func_name.to_string(),
                osr_range,
            );
            self.compiled_osr_entries.insert(osr_name, code_ptr);

            // Record finalized code ranges for auxiliary functions
            for other_fn in &module.functions {
                if other_fn.signature.name != func_name {
                    let fid = module_funcs[&other_fn.signature.name];
                    let aux_ptr = self.module.get_finalized_function(fid);
                    let size = self.compiled_code_sizes.get(&fid).copied().unwrap_or(1);
                    let start = aux_ptr as usize;
                    let range = (start, start.saturating_add(size.max(1)));
                    let aux_name = format!("{}_osr_req{}_aux_{}", func_name, req_id, other_fn.signature.name);
                    self.compiled_code_ranges.insert(
                        aux_name,
                        range,
                    );
                    self.compiled_code_ranges.insert(
                        other_fn.signature.name.clone(),
                        range,
                    );
                }
            }

            if func_is_called {
                let fid = module_funcs[func_name];
                let base_ptr = self.module.get_finalized_function(fid);
                let size = self.compiled_code_sizes.get(&fid).copied().unwrap_or(1);
                let start = base_ptr as usize;
                let range = (start, start.saturating_add(size.max(1)));
                let base_name = format!("{}_base_req{}", func_name, req_id);
                self.compiled_code_ranges.insert(
                    base_name,
                    range,
                );
                self.compiled_code_ranges.insert(
                    func_name.to_string(),
                    range,
                );
            }

            Ok(code_ptr)
        }

        pub fn compile_struct_decoder(&mut self, layout: &MirStructLayout) -> Result<*const u8, MirCraneliftBackendError> {
            self.register_struct_layouts(core::slice::from_ref(layout));
            let fn_name = format!("dyn_decode_{}_{}", layout.type_name, layout.type_id);
            let mut sig = self.module.make_signature();
            sig.params.push(AbiParam::new(types::I64));
            sig.returns.push(AbiParam::new(types::I64));

            let mut decode_sig = self.module.make_signature();
            decode_sig.params.push(AbiParam::new(types::I64));
            decode_sig.params.push(AbiParam::new(types::I64));
            decode_sig.returns.push(AbiParam::new(types::I64));
            let decode_callee = self
                .module
                .declare_function("lust_mir_struct_decode_contiguous", Linkage::Import, &decode_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let entry_sig = self.module.make_signature();
            let native_entry_callee = self
                .module
                .declare_function("lust_mir_record_native_entry", Linkage::Import, &entry_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let func_id = self
                .module
                .declare_function(&fn_name, Linkage::Export, &sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            self.ctx.func.signature = sig;
            {
                let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);
                let decode_fn = self.module.declare_func_in_func(decode_callee, &mut builder.func);
                let native_entry_fn = self.module.declare_func_in_func(native_entry_callee, &mut builder.func);
                let entry = builder.create_block();
                builder.append_block_params_for_function_params(entry);
                builder.switch_to_block(entry);
                builder.ins().call(native_entry_fn, &[]);
                let input_val = builder.block_params(entry)[0];
                let layout_id_val = builder.ins().iconst(types::I64, layout.type_id as i64);
                let call = builder.ins().call(decode_fn, &[layout_id_val, input_val]);
                let res = builder.inst_results(call)[0];
                builder.ins().return_(&[res]);
                builder.seal_all_blocks();
                builder.finalize(self.module.target_config());
            }

            self.module
                .define_function(func_id, &mut self.ctx)
                .map_err(|e| {
                    eprintln!("Verifier error in define_function: {:?}", e);
                    MirCraneliftBackendError::CodegenError(format!("{e:?}"))
                })?;
            self.module.clear_context(&mut self.ctx);
            self.module
                .finalize_definitions()
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;
            Ok(self.module.get_finalized_function(func_id))
        }

        pub fn compile_binary_struct_decoder(
            &mut self,
            layout: &MirStructLayout,
        ) -> Result<*const u8, MirCraneliftBackendError> {
            self.register_struct_layouts(core::slice::from_ref(layout));
            let fn_name = format!("dyn_decode_bin_{}_{}", layout.type_name, layout.type_id);
            let mut sig = self.module.make_signature();
            sig.params.push(AbiParam::new(types::I64));
            sig.params.push(AbiParam::new(types::I64));
            sig.returns.push(AbiParam::new(types::I64));

            let mut decode_sig = self.module.make_signature();
            decode_sig.params.push(AbiParam::new(types::I64));
            decode_sig.params.push(AbiParam::new(types::I64));
            decode_sig.params.push(AbiParam::new(types::I64));
            decode_sig.returns.push(AbiParam::new(types::I64));
            let decode_callee = self
                .module
                .declare_function("lust_mir_struct_decode_binary", Linkage::Import, &decode_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let entry_sig = self.module.make_signature();
            let native_entry_callee = self
                .module
                .declare_function("lust_mir_record_native_entry", Linkage::Import, &entry_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let func_id = self
                .module
                .declare_function(&fn_name, Linkage::Export, &sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            self.ctx.func.signature = sig;
            {
                let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);
                let decode_fn = self.module.declare_func_in_func(decode_callee, &mut builder.func);
                let native_entry_fn = self.module.declare_func_in_func(native_entry_callee, &mut builder.func);
                let entry = builder.create_block();
                builder.append_block_params_for_function_params(entry);
                builder.switch_to_block(entry);
                builder.ins().call(native_entry_fn, &[]);
                let ptr_val = builder.block_params(entry)[0];
                let len_val = builder.block_params(entry)[1];
                let layout_id_val = builder.ins().iconst(types::I64, layout.type_id as i64);
                let call = builder.ins().call(decode_fn, &[layout_id_val, ptr_val, len_val]);
                let res = builder.inst_results(call)[0];
                builder.ins().return_(&[res]);
                builder.seal_all_blocks();
                builder.finalize(self.module.target_config());
            }

            self.module
                .define_function(func_id, &mut self.ctx)
                .map_err(|e| {
                    eprintln!("Verifier error in define_function: {:?}", e);
                    MirCraneliftBackendError::CodegenError(format!("{e:?}"))
                })?;
            self.module.clear_context(&mut self.ctx);
            self.module
                .finalize_definitions()
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;
            Ok(self.module.get_finalized_function(func_id))
        }

        pub fn compile_struct_field_loader(
            &mut self,
            layout: &MirStructLayout,
            field_idx: usize,
        ) -> Result<*const u8, MirCraneliftBackendError> {
            self.register_struct_layouts(core::slice::from_ref(layout));
            let f = layout.fields.get(field_idx).ok_or_else(|| {
                MirCraneliftBackendError::CodegenError(format!("Field index {field_idx} out of bounds"))
            })?;
            let fn_name = format!("dyn_load_field_{}_{}_{}", layout.type_name, layout.type_id, field_idx);
            let mut sig = self.module.make_signature();
            sig.params.push(AbiParam::new(types::I64));

            let clif_ty = match f.tag {
                CanonicalTypeTag::Float64 => types::F64,
                CanonicalTypeTag::Bool => types::I8,
                _ => types::I64,
            };
            sig.returns.push(AbiParam::new(clif_ty));

            let entry_sig = self.module.make_signature();
            let native_entry_callee = self
                .module
                .declare_function("lust_mir_record_native_entry", Linkage::Import, &entry_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let func_id = self
                .module
                .declare_function(&fn_name, Linkage::Export, &sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            self.ctx.func.signature = sig;
            {
                let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);
                let native_entry_fn = self.module.declare_func_in_func(native_entry_callee, &mut builder.func);
                let entry = builder.create_block();
                builder.append_block_params_for_function_params(entry);
                builder.switch_to_block(entry);
                builder.ins().call(native_entry_fn, &[]);
                let base_ptr = builder.block_params(entry)[0];
                let loaded = builder.ins().load(clif_ty, MemFlagsData::trusted(), base_ptr, f.byte_offset as i32);
                builder.ins().return_(&[loaded]);
                builder.seal_all_blocks();
                builder.finalize(self.module.target_config());
            }

            self.module
                .define_function(func_id, &mut self.ctx)
                .map_err(|e| {
                    eprintln!("Verifier error in define_function: {:?}", e);
                    MirCraneliftBackendError::CodegenError(format!("{e:?}"))
                })?;
            self.module.clear_context(&mut self.ctx);
            self.module
                .finalize_definitions()
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;
            Ok(self.module.get_finalized_function(func_id))
        }

        pub fn compile_ic_dispatch_tester(
            &mut self,
            site_id: u32,
        ) -> Result<*const u8, MirCraneliftBackendError> {
            let fn_name = format!("dyn_ic_dispatch_{site_id}");
            let mut sig = self.module.make_signature();
            sig.params.push(AbiParam::new(types::I64));
            sig.params.push(AbiParam::new(types::I64));
            sig.returns.push(AbiParam::new(types::I64));

            let mut ic_sig = self.module.make_signature();
            ic_sig.params.push(AbiParam::new(types::I64));
            ic_sig.params.push(AbiParam::new(types::I64));
            ic_sig.params.push(AbiParam::new(types::I64));
            ic_sig.returns.push(AbiParam::new(types::I64));
            let ic_callee = self
                .module
                .declare_function("lust_mir_ic_dispatch", Linkage::Import, &ic_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let func_id = self
                .module
                .declare_function(&fn_name, Linkage::Export, &sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            self.ctx.func.signature = sig;
            {
                let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);
                let ic_fn = self.module.declare_func_in_func(ic_callee, &mut builder.func);
                let entry = builder.create_block();
                builder.append_block_params_for_function_params(entry);
                builder.switch_to_block(entry);
                let type_tag = builder.block_params(entry)[0];
                let default_target = builder.block_params(entry)[1];
                let site_val = builder.ins().iconst(types::I64, site_id as i64);
                let call = builder.ins().call(ic_fn, &[site_val, type_tag, default_target]);
                let res = builder.inst_results(call)[0];
                builder.ins().return_(&[res]);
                builder.seal_all_blocks();
                builder.finalize(self.module.target_config());
            }

            self.module
                .define_function(func_id, &mut self.ctx)
                .map_err(|e| {
                    eprintln!("Verifier error in define_function: {:?}", e);
                    MirCraneliftBackendError::CodegenError(format!("{e:?}"))
                })?;
            self.module.clear_context(&mut self.ctx);
            self.module
                .finalize_definitions()
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;
            Ok(self.module.get_finalized_function(func_id))
        }

        pub fn compile_simd_add_kernel(&mut self) -> Result<*const u8, MirCraneliftBackendError> {
            let sig = FunctionSignature {
                name: "simd_add_kernel".into(),
                param_types: vec![MirType::Int, MirType::Int, MirType::Int, MirType::Int, MirType::Int],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            };
            let mut func = MirFunction::new(sig, crate::ast::Span::dummy());
            let a = func.params[0];
            let b = func.params[1];
            let out = func.params[2];
            let len = func.params[3];
            let force = func.params[4];
            let dst = func.add_local(Some("dst".into()), MirType::Int, false, crate::ast::Span::dummy());
            let exit_bb = func.new_block();

            func.blocks[0].terminator = Some(Terminator::Call {
                callee: "simd.add".into(),
                args: vec![
                    Operand::Copy(Place::Local(a)),
                    Operand::Copy(Place::Local(b)),
                    Operand::Copy(Place::Local(out)),
                    Operand::Copy(Place::Local(len)),
                    Operand::Copy(Place::Local(force)),
                ],
                destination: Place::Local(dst),
                success: exit_bb,
                error: None,
                span: crate::ast::Span::dummy(),
            });
            func.blocks[exit_bb.0].terminator = Some(Terminator::Return(
                Some(Operand::Copy(Place::Local(dst))),
                crate::ast::Span::dummy(),
            ));

            self.compile_function(&func)
        }

        pub fn compile_simd_mul_kernel(&mut self) -> Result<*const u8, MirCraneliftBackendError> {
            let sig = FunctionSignature {
                name: "simd_mul_kernel".into(),
                param_types: vec![MirType::Int, MirType::Int, MirType::Int, MirType::Int, MirType::Int],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            };
            let mut func = MirFunction::new(sig, crate::ast::Span::dummy());
            let a = func.params[0];
            let b = func.params[1];
            let out = func.params[2];
            let len = func.params[3];
            let force = func.params[4];
            let dst = func.add_local(Some("dst".into()), MirType::Int, false, crate::ast::Span::dummy());
            let exit_bb = func.new_block();

            func.blocks[0].terminator = Some(Terminator::Call {
                callee: "simd.mul".into(),
                args: vec![
                    Operand::Copy(Place::Local(a)),
                    Operand::Copy(Place::Local(b)),
                    Operand::Copy(Place::Local(out)),
                    Operand::Copy(Place::Local(len)),
                    Operand::Copy(Place::Local(force)),
                ],
                destination: Place::Local(dst),
                success: exit_bb,
                error: None,
                span: crate::ast::Span::dummy(),
            });
            func.blocks[exit_bb.0].terminator = Some(Terminator::Return(
                Some(Operand::Copy(Place::Local(dst))),
                crate::ast::Span::dummy(),
            ));

            self.compile_function(&func)
        }

        pub fn compile_simd_sum_kernel(&mut self) -> Result<*const u8, MirCraneliftBackendError> {
            let sig = FunctionSignature {
                name: "simd_sum_kernel".into(),
                param_types: vec![MirType::Int, MirType::Int, MirType::Int],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            };
            let mut func = MirFunction::new(sig, crate::ast::Span::dummy());
            let a = func.params[0];
            let len = func.params[1];
            let force = func.params[2];
            let dst = func.add_local(Some("dst".into()), MirType::Int, false, crate::ast::Span::dummy());
            let exit_bb = func.new_block();

            func.blocks[0].terminator = Some(Terminator::Call {
                callee: "simd.sum".into(),
                args: vec![
                    Operand::Copy(Place::Local(a)),
                    Operand::Copy(Place::Local(len)),
                    Operand::Copy(Place::Local(force)),
                ],
                destination: Place::Local(dst),
                success: exit_bb,
                error: None,
                span: crate::ast::Span::dummy(),
            });
            func.blocks[exit_bb.0].terminator = Some(Terminator::Return(
                Some(Operand::Copy(Place::Local(dst))),
                crate::ast::Span::dummy(),
            ));

            self.compile_function(&func)
        }

        pub fn compile_simd_dot_kernel(&mut self) -> Result<*const u8, MirCraneliftBackendError> {
            let sig = FunctionSignature {
                name: "simd_dot_kernel".into(),
                param_types: vec![MirType::Int, MirType::Int, MirType::Int, MirType::Int],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            };
            let mut func = MirFunction::new(sig, crate::ast::Span::dummy());
            let a = func.params[0];
            let b = func.params[1];
            let len = func.params[2];
            let force = func.params[3];
            let dst = func.add_local(Some("dst".into()), MirType::Int, false, crate::ast::Span::dummy());
            let exit_bb = func.new_block();

            func.blocks[0].terminator = Some(Terminator::Call {
                callee: "simd.dot".into(),
                args: vec![
                    Operand::Copy(Place::Local(a)),
                    Operand::Copy(Place::Local(b)),
                    Operand::Copy(Place::Local(len)),
                    Operand::Copy(Place::Local(force)),
                ],
                destination: Place::Local(dst),
                success: exit_bb,
                error: None,
                span: crate::ast::Span::dummy(),
            });
            func.blocks[exit_bb.0].terminator = Some(Terminator::Return(
                Some(Operand::Copy(Place::Local(dst))),
                crate::ast::Span::dummy(),
            ));

            self.compile_function(&func)
        }

        fn is_simd_callee_name(callee: &str) -> bool {
            callee.starts_with("simd.")
                || callee.starts_with("simd_")
                || callee.starts_with("vector.")
        }

        fn validate_simd_place_type(
            callee: &str,
            place: &Place,
            locals: &[LocalDecl],
            struct_layouts: &hashbrown::HashMap<String, MirStructLayout>,
        ) -> Result<MirType, MirCraneliftBackendError> {
            match place {
                Place::Local(loc) => locals.get(loc.0).map(|l| l.ty.clone()).ok_or_else(|| {
                    MirCraneliftBackendError::InvalidSimdCall {
                        callee: callee.to_string(),
                        reason: format!(
                            "Place local {loc:?} out of bounds (locals len {})",
                            locals.len()
                        ),
                    }
                }),
                Place::Index(base, idx) => {
                    let idx_decl = locals.get(idx.0).ok_or_else(|| {
                        MirCraneliftBackendError::InvalidSimdCall {
                            callee: callee.to_string(),
                            reason: format!(
                                "Index local {idx:?} out of bounds (locals len {})",
                                locals.len()
                            ),
                        }
                    })?;
                    if idx_decl.ty != MirType::Int {
                        return Err(MirCraneliftBackendError::InvalidSimdCall {
                            callee: callee.to_string(),
                            reason: format!(
                                "Index local {idx:?} must have Int type, got '{}'",
                                idx_decl.ty
                            ),
                        });
                    }
                    let base_ty = Self::validate_simd_place_type(callee, base, locals, struct_layouts)?;
                    match base_ty {
                        MirType::Array(inner) => Ok(*inner),
                        other => Err(MirCraneliftBackendError::InvalidSimdCall {
                            callee: callee.to_string(),
                            reason: format!(
                                "Cannot index non-array place of type '{other}' in SIMD call"
                            ),
                        }),
                    }
                }
                Place::Field(base, field_idx) => {
                    let base_ty = Self::validate_simd_place_type(callee, base, locals, struct_layouts)?;
                    match base_ty {
                        MirType::Tuple(elems) => {
                            elems.get(*field_idx).cloned().ok_or_else(|| {
                                MirCraneliftBackendError::InvalidSimdCall {
                                    callee: callee.to_string(),
                                    reason: format!(
                                        "Tuple field index {field_idx} out of bounds (tuple has {} fields)",
                                        elems.len()
                                    ),
                                }
                            })
                        }
                        MirType::Struct(name) => {
                            if let Some(layout) = struct_layouts.get(&name) {
                                let field = layout.fields.get(*field_idx).ok_or_else(|| {
                                    MirCraneliftBackendError::InvalidSimdCall {
                                        callee: callee.to_string(),
                                        reason: format!(
                                            "Struct '{name}' field index {field_idx} out of bounds (struct has {} fields)",
                                            layout.fields.len()
                                        ),
                                    }
                                })?;
                                return match field.tag {
                                    CanonicalTypeTag::Bool => Ok(MirType::Bool),
                                    CanonicalTypeTag::Int64
                                    | CanonicalTypeTag::UInt64
                                    | CanonicalTypeTag::DateTimeUtc => Ok(MirType::Int),
                                    CanonicalTypeTag::Float64 => Ok(MirType::Float),
                                    CanonicalTypeTag::StringSlice => Ok(MirType::String),
                                    other => with_current_mir_context(|ctx| {
                                        ctx.get_struct_field_type(&name, *field_idx)
                                    })
                                    .ok_or_else(|| {
                                        MirCraneliftBackendError::InvalidSimdCall {
                                            callee: callee.to_string(),
                                            reason: format!(
                                                "Unsupported struct field tag '{other:?}' at '{name}.{field_idx}'"
                                            ),
                                        }
                                    }),
                                };
                            }
                            with_current_mir_context(|ctx| {
                                ctx.get_struct_field_type(&name, *field_idx)
                            })
                            .ok_or_else(|| MirCraneliftBackendError::InvalidSimdCall {
                                callee: callee.to_string(),
                                reason: format!(
                                    "Unknown struct layout or out-of-bounds field {field_idx} for struct '{name}'"
                                ),
                            })
                        }
                        other => Err(MirCraneliftBackendError::InvalidSimdCall {
                            callee: callee.to_string(),
                            reason: format!(
                                "Cannot access field {field_idx} on non-aggregate type '{other}' in SIMD call"
                            ),
                        }),
                    }
                }
            }
        }

        fn validate_simd_operand_type(
            callee: &str,
            op: &Operand,
            locals: &[LocalDecl],
            struct_layouts: &hashbrown::HashMap<String, MirStructLayout>,
        ) -> Result<MirType, MirCraneliftBackendError> {
            match op {
                Operand::Copy(p) | Operand::Move(p) => {
                    Self::validate_simd_place_type(callee, p, locals, struct_layouts)
                }
                Operand::Constant(c) => match c {
                    MirConstant::Int(_) => Ok(MirType::Int),
                    MirConstant::Float(_) => Ok(MirType::Float),
                    MirConstant::Bool(_) => Ok(MirType::Bool),
                    MirConstant::String(_) => Ok(MirType::String),
                    MirConstant::Unit => Ok(MirType::Unit),
                },
            }
        }

        fn validate_simd_call(
            callee: &str,
            args: &[Operand],
            destination: &Place,
            success: BlockId,
            error: Option<BlockId>,
            mir_fn: &MirFunction,
            struct_layouts: &hashbrown::HashMap<String, MirStructLayout>,
        ) -> Result<(), MirCraneliftBackendError> {
            #[derive(Clone, Copy)]
            enum SimdKind {
                BinaryVector,
                UnaryReduce,
                BinaryReduce,
            }

            let (expected_arity, kind) = match callee {
                "simd.add" | "simd_add" | "vector.add" => (5, SimdKind::BinaryVector),
                "simd.mul" | "simd_mul" | "vector.mul" => (5, SimdKind::BinaryVector),
                "simd.sum" | "simd_sum" | "vector.sum" => (3, SimdKind::UnaryReduce),
                "simd.dot" | "simd_dot" | "vector.dot" => (4, SimdKind::BinaryReduce),
                _ => {
                    return Err(MirCraneliftBackendError::InvalidSimdCall {
                        callee: callee.to_string(),
                        reason: format!("Unsupported or unknown SIMD operation '{callee}'"),
                    });
                }
            };

            if args.len() != expected_arity {
                return Err(MirCraneliftBackendError::InvalidSimdCall {
                    callee: callee.to_string(),
                    reason: format!(
                        "SIMD operation '{callee}' requires exactly {expected_arity} operands, got {}",
                        args.len()
                    ),
                });
            }

            if success.0 >= mir_fn.blocks.len() {
                return Err(MirCraneliftBackendError::InvalidSimdCall {
                    callee: callee.to_string(),
                    reason: format!("Success block {:?} out of bounds", success),
                });
            }
            if let Some(err_bb) = error {
                if err_bb.0 >= mir_fn.blocks.len() {
                    return Err(MirCraneliftBackendError::InvalidSimdCall {
                        callee: callee.to_string(),
                        reason: format!("Error block {:?} out of bounds", err_bb),
                    });
                }
            }

            // Validate destination: must be a Local of exact type Int (nested places are unsupported shapes)
            match destination {
                Place::Local(loc) => {
                    let local_decl = mir_fn.locals.get(loc.0).ok_or_else(|| {
                        MirCraneliftBackendError::InvalidSimdCall {
                            callee: callee.to_string(),
                            reason: format!("Destination local {loc:?} out of bounds"),
                        }
                    })?;
                    if local_decl.ty != MirType::Int {
                        return Err(MirCraneliftBackendError::InvalidSimdCall {
                            callee: callee.to_string(),
                            reason: format!(
                                "Destination local {loc:?} has invalid type '{}'; SIMD operations require Int destination",
                                local_decl.ty
                            ),
                        });
                    }
                }
                Place::Index(..) | Place::Field(..) => {
                    // Validate nested place first for precise diagnostics, then reject unsupported shape
                    let _ = Self::validate_simd_place_type(
                        callee,
                        destination,
                        &mir_fn.locals,
                        struct_layouts,
                    )?;
                    return Err(MirCraneliftBackendError::InvalidSimdCall {
                        callee: callee.to_string(),
                        reason: format!(
                            "Unsupported SIMD destination place '{destination}'; destination must be a local Int variable"
                        ),
                    });
                }
            }

            let mut arg_tys = Vec::with_capacity(args.len());
            for arg in args {
                arg_tys.push(Self::validate_simd_operand_type(
                    callee,
                    arg,
                    &mir_fn.locals,
                    struct_layouts,
                )?);
            }

            let check_ptr = |idx: usize, name: &str, ty: &MirType| -> Result<(), MirCraneliftBackendError> {
                if *ty != MirType::Int {
                    return Err(MirCraneliftBackendError::InvalidSimdCall {
                        callee: callee.to_string(),
                        reason: format!(
                            "SIMD buffer pointer operand {idx} ('{name}') must have Int type, got '{ty}'"
                        ),
                    });
                }
                Ok(())
            };
            let check_len = |idx: usize, ty: &MirType| -> Result<(), MirCraneliftBackendError> {
                if *ty != MirType::Int {
                    return Err(MirCraneliftBackendError::InvalidSimdCall {
                        callee: callee.to_string(),
                        reason: format!(
                            "SIMD length operand {idx} must have Int type, got '{ty}'"
                        ),
                    });
                }
                Ok(())
            };
            let check_force = |idx: usize, ty: &MirType| -> Result<(), MirCraneliftBackendError> {
                if !matches!(ty, MirType::Int | MirType::Bool) {
                    return Err(MirCraneliftBackendError::InvalidSimdCall {
                        callee: callee.to_string(),
                        reason: format!(
                            "SIMD force_scalar operand {idx} must have Int or Bool type, got '{ty}'"
                        ),
                    });
                }
                Ok(())
            };

            match kind {
                SimdKind::BinaryVector => {
                    check_ptr(0, "a", &arg_tys[0])?;
                    check_ptr(1, "b", &arg_tys[1])?;
                    check_ptr(2, "out", &arg_tys[2])?;
                    check_len(3, &arg_tys[3])?;
                    check_force(4, &arg_tys[4])?;
                }
                SimdKind::UnaryReduce => {
                    check_ptr(0, "a", &arg_tys[0])?;
                    check_len(1, &arg_tys[1])?;
                    check_force(2, &arg_tys[2])?;
                }
                SimdKind::BinaryReduce => {
                    check_ptr(0, "a", &arg_tys[0])?;
                    check_ptr(1, "b", &arg_tys[1])?;
                    check_len(2, &arg_tys[2])?;
                    check_force(3, &arg_tys[3])?;
                }
            }

            Ok(())
        }

        fn define_trap_stub(&mut self, func_id: FuncId, sig: Signature) {
            self.module.clear_context(&mut self.ctx);
            self.builder_ctx = FunctionBuilderContext::new();
            self.ctx.func.signature = sig;
            {
                let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);
                let entry_bb = builder.create_block();
                builder.append_block_params_for_function_params(entry_bb);
                builder.switch_to_block(entry_bb);
                builder.ins().trap(cranelift_codegen::ir::TrapCode::unwrap_user(1));
                builder.seal_all_blocks();
                builder.finalize(self.module.target_config());
            }
            if let Err(e) = self.module.define_function(func_id, &mut self.ctx) {
                eprintln!("Failed to define trap stub for func_id {func_id:?}: {e:?}");
            }
            self.module.clear_context(&mut self.ctx);
            self.builder_ctx = FunctionBuilderContext::new();
        }

        fn mir_types_compatible_for_call(actual: &MirType, expected: &MirType) -> bool {
            // Unresolved types cannot be proven semantically compatible.
            // ABI bitwidth equality (e.g. types::I64) does not prove semantic compatibility.
            if *actual == MirType::Unknown || *expected == MirType::Unknown {
                return false;
            }
            actual == expected
        }

        fn validate_module_calls_prior_to_mutation(
            &self,
            module: &MirModule,
        ) -> Result<(), MirCraneliftBackendError> {
            let callee_rets: hashbrown::HashMap<String, MirType> = module
                .functions
                .iter()
                .map(|f| (f.signature.name.clone(), f.signature.return_type.clone()))
                .collect();
            // First, refine a copy of each function to resolve local types accurately
            let mut refined_funcs: hashbrown::HashMap<String, MirFunction> = module
                .functions
                .iter()
                .map(|f| {
                    let mut f_with_calls = f.clone();
                    apply_module_callee_return_types(&mut f_with_calls, &callee_rets);
                    (
                        f.signature.name.clone(),
                        self.refine_function_for_compiler(&f_with_calls),
                    )
                })
                .collect();

            // Refine anonymous or unknown destination locals from callee return types
            for f in refined_funcs.values_mut() {
                for b in &mut f.blocks {
                    if let Some(Terminator::Call { callee, destination, .. }) = &b.terminator {
                        if let Some(target_fn) = module.functions.iter().find(|other| &other.signature.name == callee) {
                            if let Place::Local(loc) = destination {
                                if let Some(decl) = f.locals.get_mut(loc.0) {
                                    if decl.ty == MirType::Unknown
                                        && target_fn.signature.return_type != MirType::Unknown
                                    {
                                        decl.ty = target_fn.signature.return_type.clone();
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let refined_list: Vec<MirFunction> = refined_funcs.values().cloned().collect();
            let dispatch_tables = build_module_dispatch_tables(
                &refined_list,
                &module.struct_layouts,
                &self.struct_layouts,
            );

            for f in refined_funcs.values() {
                for b in &f.blocks {
                    if let Some(Terminator::Call { callee, args, destination, .. }) = &b.terminator {
                        if matches!(callee.as_str(), "table.len" | "array.len") {
                            if args.len() != 1 {
                                return Err(MirCraneliftBackendError::CodegenError(format!(
                                    "Call to '{callee}' arity mismatch: got {} arguments, expected 1",
                                    args.len()
                                )));
                            }
                            let arg_ty = try_operand_type(&args[0], &f.locals, &self.struct_layouts)?;
                            if !matches!(arg_ty, MirType::Array(_)) {
                                return Err(MirCraneliftBackendError::CodegenError(format!(
                                    "Call to '{callee}' requires Array argument, got '{arg_ty}'"
                                )));
                            }
                        } else if let Some(target_fn) = refined_funcs.get(callee) {
                            // 1. Arity check
                            let expected_arity = target_fn.params.len();
                            if args.len() != expected_arity {
                                return Err(MirCraneliftBackendError::CodegenError(format!(
                                    "Call to '{}' arity mismatch: got {} arguments, expected {}",
                                    callee, args.len(), expected_arity
                                )));
                            }

                            // 2. Argument types check
                            let dyn_param_flags = dispatch_tables.dynamic_unknown_params.get(callee);
                            for (i, arg) in args.iter().enumerate() {
                                let arg_ty = try_operand_type(arg, &f.locals, &self.struct_layouts)?;
                                let expected_ty = target_fn
                                    .locals
                                    .get(target_fn.params[i].0)
                                    .map(|l| &l.ty)
                                    .unwrap_or_else(|| &target_fn.signature.param_types[i]);
                                let is_explicit_dyn_unknown_param = *expected_ty == MirType::Unknown
                                    && dyn_param_flags
                                        .and_then(|flags| flags.get(i).copied())
                                        .unwrap_or(false)
                                    && matches!(arg_ty, MirType::Unknown | MirType::Struct(_));
                                if !is_explicit_dyn_unknown_param
                                    && !Self::mir_types_compatible_for_call(&arg_ty, expected_ty)
                                {
                                    return Err(MirCraneliftBackendError::CodegenError(format!(
                                        "Call to '{}' argument {} type mismatch: got '{}', expected '{}'",
                                        callee, i, arg_ty, expected_ty
                                    )));
                                }
                            }

                            // 3. Destination return type check
                            let callee_ret_ty = &target_fn.signature.return_type;
                            let dst_ty = try_place_type(destination, &f.locals, &self.struct_layouts)?;
                            if !Self::mir_types_compatible_for_call(callee_ret_ty, &dst_ty) {
                                return Err(MirCraneliftBackendError::CodegenError(format!(
                                    "Call to '{}' return type mismatch: callee returns '{}', but destination has type '{}'",
                                    callee, callee_ret_ty, dst_ty
                                )));
                            }
                        }
                    }
                }
            }
            Ok(())
        }

        fn cleanup_unfinalized_declarations(
            &mut self,
            declared_sigs: &hashbrown::HashMap<FuncId, Signature>,
            defined_fids: &hashbrown::HashSet<FuncId>,
        ) {
            for (&fid, sig) in declared_sigs {
                if !defined_fids.contains(&fid) {
                    self.define_trap_stub(fid, sig.clone());
                }
            }
            let _ = self.module.finalize_definitions();
        }

        fn compile_function_internal(
            &mut self,
            mir_fn: &MirFunction,
            func_id: FuncId,
            module_funcs: &hashbrown::HashMap<String, FuncId>,
        ) -> Result<(), MirCraneliftBackendError> {
            self.compile_function_with_options(
                mir_fn,
                func_id,
                module_funcs,
                None,
                true,
                None,
                None,
                None,
            )
        }

        fn compile_function_with_osr(
            &mut self,
            mir_fn: &MirFunction,
            func_id: FuncId,
            module_funcs: &hashbrown::HashMap<String, FuncId>,
            osr_info: Option<(BlockId, &[LocalId])>,
        ) -> Result<(), MirCraneliftBackendError> {
            self.compile_function_with_options(
                mir_fn,
                func_id,
                module_funcs,
                osr_info,
                true,
                None,
                None,
                None,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn compile_function_with_options(
            &mut self,
            mir_fn: &MirFunction,
            func_id: FuncId,
            module_funcs: &hashbrown::HashMap<String, FuncId>,
            osr_info: Option<(BlockId, &[LocalId])>,
            emit_native_entry: bool,
            call_effects: Option<&hashbrown::HashMap<String, (bool, bool)>>,
            ir_capture_key: Option<&str>,
            dispatch_tables: Option<&ModuleDispatchTables>,
        ) -> Result<(), MirCraneliftBackendError> {
            self.module.clear_context(&mut self.ctx);
            self.builder_ctx = FunctionBuilderContext::new();
            let res = self.compile_function_with_osr_inner(
                mir_fn,
                func_id,
                module_funcs,
                osr_info,
                emit_native_entry,
                call_effects,
                ir_capture_key,
                dispatch_tables,
            );
            if res.is_err() {
                self.module.clear_context(&mut self.ctx);
                self.builder_ctx = FunctionBuilderContext::new();
            }
            res
        }

        fn declare_ext_fn(
            module: &mut JITModule,
            func: &mut cranelift_codegen::ir::Function,
            cache: &mut hashbrown::HashMap<&'static str, FuncRef>,
            name: &'static str,
            params: &[types::Type],
            ret: Option<types::Type>,
        ) -> Result<FuncRef, MirCraneliftBackendError> {
            if let Some(&fref) = cache.get(name) {
                return Ok(fref);
            }
            let mut sig = module.make_signature();
            for &p in params {
                sig.params.push(AbiParam::new(p));
            }
            if let Some(r) = ret {
                sig.returns.push(AbiParam::new(r));
            }
            let fid = module
                .declare_function(name, Linkage::Import, &sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;
            let fref = module.declare_func_in_func(fid, func);
            cache.insert(name, fref);
            Ok(fref)
        }

        #[allow(clippy::too_many_arguments)]
        fn compile_function_with_osr_inner(
            &mut self,
            mir_fn: &MirFunction,
            func_id: FuncId,
            module_funcs: &hashbrown::HashMap<String, FuncId>,
            osr_info: Option<(BlockId, &[LocalId])>,
            emit_native_entry: bool,
            call_effects: Option<&hashbrown::HashMap<String, (bool, bool)>>,
            ir_capture_key: Option<&str>,
            dispatch_tables: Option<&ModuleDispatchTables>,
        ) -> Result<(), MirCraneliftBackendError> {
            // Fail-closed validation for SIMD calls before lowering
            for block in &mir_fn.blocks {
                if let Some(Terminator::Call {
                    callee,
                    args,
                    destination,
                    success,
                    error,
                    ..
                }) = &block.terminator
                {
                    if Self::is_simd_callee_name(callee) {
                        Self::validate_simd_call(
                            callee,
                            args,
                            destination,
                            *success,
                            *error,
                            mir_fn,
                            &self.struct_layouts,
                        )?;
                    }
                }
            }

            // Fail-closed compile boundary: ensure all locals have proven concrete types,
            // allowing `MirType::Unknown` ONLY for locals explicitly participating in dynamic
            // `__mir_unknown_*` operations (preserving strict fail-closed rejection on unproven locals).
            let explicit_dyn_unknown_locals =
                collect_explicit_dynamic_unknown_locals(mir_fn, dispatch_tables);
            for (idx, local) in mir_fn.locals.iter().enumerate() {
                if local.ty == MirType::Unknown && !explicit_dyn_unknown_locals.contains(&idx) {
                    return Err(MirCraneliftBackendError::CodegenError(format!(
                        "Function '{}' local {} ('{}') has unproven or conflicting type 'unknown'",
                        mir_fn.signature.name,
                        idx,
                        local.name.as_deref().unwrap_or("<anonymous>")
                    )));
                }
            }

            // NG-W16-3: Pre-compile CallConv::PreserveAll cold trampolines BEFORE building
            // `self.ctx.func` when the function uses F64 locals/returns so AArch64 AAPCS64
            // default clobbers (`v0..v31`) never spill loop-carried floats on hot iterations.
            let use_preserve_all_slow = cfg!(any(target_arch = "aarch64", target_arch = "x86_64"))
                && (mir_fn.locals.iter().any(|l| matches!(l.ty, MirType::Float))
                    || matches!(mir_fn.signature.return_type, MirType::Float));
            let preserve_all_trampolines = if use_preserve_all_slow {
                Some(self.ensure_preserve_all_slow_trampolines()?)
            } else {
                None
            };

            self.ctx.func.signature.params.clear();
            self.ctx.func.signature.returns.clear();

            if let Some((_, live_locals)) = osr_info {
                for _ in live_locals {
                    self.ctx.func.signature.params.push(AbiParam::new(types::I64));
                }
            } else {
                for param_id in &mir_fn.params {
                    let p_ty = &mir_fn.locals[param_id.0].ty;
                    self.ctx.func.signature.params.push(AbiParam::new(mir_to_clif_type(p_ty)?));
                }
            }

            let ret_clif_ty = mir_to_clif_type(&mir_fn.signature.return_type)?;
            if ret_clif_ty != types::I8 || mir_fn.signature.return_type != MirType::Unit {
                self.ctx.func.signature.returns.push(AbiParam::new(ret_clif_ty));
            }

            // Declare status helpers
            let mut sig_void1 = self.module.make_signature();
            sig_void1.params.push(AbiParam::new(types::I64));
            let status_set_callee = self
                .module
                .declare_function("lust_mir_set_status", Linkage::Import, &sig_void1)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let mut sig0_i64 = self.module.make_signature();
            sig0_i64.returns.push(AbiParam::new(types::I64));
            let status_callee = self
                .module
                .declare_function("lust_mir_get_status", Linkage::Import, &sig0_i64)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            // Declare gas helpers
            let mut sig1 = self.module.make_signature();
            sig1.params.push(AbiParam::new(types::I64));
            sig1.returns.push(AbiParam::new(types::I64));

            let gas_fast_ptr_callee = self
                .module
                .declare_function("lust_mir_gas_fast_ptr", Linkage::Import, &sig0_i64)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let gas_slow_callee = self
                .module
                .declare_function("lust_mir_gas_slow", Linkage::Import, &sig0_i64)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let mut alloc_typed_sig = self.module.make_signature();
            alloc_typed_sig.params.push(AbiParam::new(types::I64));
            alloc_typed_sig.params.push(AbiParam::new(types::I64));
            alloc_typed_sig.returns.push(AbiParam::new(types::I64));

            let alloc_array_callee = self
                .module
                .declare_function("lust_mir_alloc_array", Linkage::Import, &sig1)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let alloc_array_typed_callee = self
                .module
                .declare_function("lust_mir_alloc_array_typed", Linkage::Import, &alloc_typed_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let alloc_tuple_typed_callee = self
                .module
                .declare_function("lust_mir_alloc_tuple_typed", Linkage::Import, &alloc_typed_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let mut alloc_named_sig = self.module.make_signature();
            alloc_named_sig.params.push(AbiParam::new(types::I64));
            alloc_named_sig.params.push(AbiParam::new(types::I64));
            alloc_named_sig.params.push(AbiParam::new(types::I64));
            alloc_named_sig.returns.push(AbiParam::new(types::I64));

            let alloc_struct_named_callee = self
                .module
                .declare_function("lust_mir_alloc_struct_named", Linkage::Import, &alloc_named_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let alloc_cflat_struct_callee = self
                .module
                .declare_function("lust_mir_alloc_cflat_struct", Linkage::Import, &alloc_named_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            // Declare safepoint and deopt imports
            let mut sp_sig = self.module.make_signature();
            sp_sig.returns.push(AbiParam::new(types::I64));
            let safepoint_state_ptr_callee = self
                .module
                .declare_function("lust_mir_safepoint_state_ptr", Linkage::Import, &sp_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;
            let safepoint_slow_callee = self
                .module
                .declare_function("lust_mir_safepoint_slow", Linkage::Import, &sp_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let mut init_deopt_sig = self.module.make_signature();
            init_deopt_sig.params.push(AbiParam::new(types::I64));
            let init_deopt_callee = self
                .module
                .declare_function("lust_mir_init_deopt", Linkage::Import, &init_deopt_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let mut rec_deopt_sig = self.module.make_signature();
            rec_deopt_sig.params.push(AbiParam::new(types::I64));
            rec_deopt_sig.params.push(AbiParam::new(types::I64));
            rec_deopt_sig.params.push(AbiParam::new(types::I64));
            let rec_deopt_callee = self
                .module
                .declare_function("lust_mir_record_deopt_local", Linkage::Import, &rec_deopt_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            // Declare int binop helpers
            let mut binop_sig = self.module.make_signature();
            binop_sig.params.push(AbiParam::new(types::I64));
            binop_sig.params.push(AbiParam::new(types::I64));
            binop_sig.returns.push(AbiParam::new(types::I64));

            let floordiv_callee = self
                .module
                .declare_function("lust_mir_floordiv_int", Linkage::Import, &binop_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let pow_callee = self
                .module
                .declare_function("lust_mir_pow_int", Linkage::Import, &binop_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            // Declare float binop helpers
            let mut float_binop_sig = self.module.make_signature();
            float_binop_sig.params.push(AbiParam::new(types::F64));
            float_binop_sig.params.push(AbiParam::new(types::F64));
            float_binop_sig.returns.push(AbiParam::new(types::F64));

            let floordiv_float_callee = self
                .module
                .declare_function("lust_mir_floordiv_float", Linkage::Import, &float_binop_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let pow_float_callee = self
                .module
                .declare_function("lust_mir_pow_float", Linkage::Import, &float_binop_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let rem_float_callee = self
                .module
                .declare_function("lust_mir_rem_float", Linkage::Import, &float_binop_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let string_concat_callee = self
                .module
                .declare_function("lust_mir_string_concat", Linkage::Import, &binop_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let string_append_inplace_callee = self
                .module
                .declare_function("lust_mir_string_append_inplace", Linkage::Import, &binop_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let mut str_eq_sig = self.module.make_signature();
            str_eq_sig.params.push(AbiParam::new(types::I64));
            str_eq_sig.params.push(AbiParam::new(types::I64));
            str_eq_sig.returns.push(AbiParam::new(types::I8));
            let string_eq_callee = self
                .module
                .declare_function("lust_mir_string_eq", Linkage::Import, &str_eq_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let string_len_callee = self
                .module
                .declare_function("lust_mir_string_len", Linkage::Import, &sig1)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let tostring_int_callee = self
                .module
                .declare_function("lust_mir_tostring_int", Linkage::Import, &sig1)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let mut tostring_float_sig = self.module.make_signature();
            tostring_float_sig.params.push(AbiParam::new(types::F64));
            tostring_float_sig.returns.push(AbiParam::new(types::I64));
            let tostring_float_callee = self
                .module
                .declare_function("lust_mir_tostring_float", Linkage::Import, &tostring_float_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let tostring_bool_callee = self
                .module
                .declare_function("lust_mir_tostring_bool", Linkage::Import, &sig1)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let len_checked_callee = self
                .module
                .declare_function("lust_mir_array_len_checked", Linkage::Import, &sig1)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let sig_void0 = self.module.make_signature();
            let native_entry_callee = self
                .module
                .declare_function("lust_mir_record_native_entry", Linkage::Import, &sig_void0)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let fn_has_array = mir_fn
                .locals
                .iter()
                .any(|l| matches!(l.ty, MirType::Array(_) | MirType::Unknown));
            let fn_has_string = mir_fn
                .locals
                .iter()
                .any(|l| matches!(l.ty, MirType::String | MirType::Unknown))
                || mir_fn.blocks.iter().any(|b| {
                    matches!(
                        &b.terminator,
                        Some(Terminator::Call { callee, .. }) if callee.starts_with("string.")
                    )
                });
            let vec_loops = if osr_info.is_none() && fn_has_array {
                crate::mir::opt::simd::recognize_vectorized_loop_blocks(mir_fn)
            } else {
                Vec::new()
            };
            let has_simd = !vec_loops.is_empty()
                || mir_fn.blocks.iter().any(|b| {
                    matches!(
                        &b.terminator,
                        Some(Terminator::Call { callee, .. }) if Self::is_simd_callee_name(callee)
                    )
                });
            let (simd_vector_callee, simd_scalar_callee, simd_scalar_forced_callee) = if has_simd {
                (
                    Some(
                        self.module
                            .declare_function(
                                "lust_mir_record_simd_vector_op",
                                Linkage::Import,
                                &sig_void0,
                            )
                            .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?,
                    ),
                    Some(
                        self.module
                            .declare_function(
                                "lust_mir_record_simd_scalar_op",
                                Linkage::Import,
                                &sig_void0,
                            )
                            .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?,
                    ),
                    Some(
                        self.module
                            .declare_function(
                                "lust_mir_is_simd_scalar_forced",
                                Linkage::Import,
                                &sig0_i64,
                            )
                            .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?,
                    ),
                )
            } else {
                (None, None, None)
            };
            let mut vec_pre_map: hashbrown::HashMap<usize, usize> = hashbrown::HashMap::new();
            let mut vec_hdr_map: hashbrown::HashMap<usize, usize> = hashbrown::HashMap::new();
            let mut vec_body_map: hashbrown::HashMap<usize, usize> = hashbrown::HashMap::new();
            for (idx, vl) in vec_loops.iter().enumerate() {
                vec_pre_map.insert(vl.pre_block.0, idx);
                vec_hdr_map.insert(vl.hdr_block.0, idx);
                vec_body_map.insert(vl.body_block.0, idx);
            }

            {
                let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);
                let mut ext_fn_cache: hashbrown::HashMap<&'static str, FuncRef> =
                    hashbrown::HashMap::new();
                let alloc_array_fn = self.module.declare_func_in_func(alloc_array_callee, &mut builder.func);
                let alloc_array_typed_fn = self.module.declare_func_in_func(alloc_array_typed_callee, &mut builder.func);
                let alloc_tuple_typed_fn = self.module.declare_func_in_func(alloc_tuple_typed_callee, &mut builder.func);
                let alloc_struct_named_fn = self.module.declare_func_in_func(alloc_struct_named_callee, &mut builder.func);
                let alloc_cflat_struct_fn = self.module.declare_func_in_func(alloc_cflat_struct_callee, &mut builder.func);
                let len_fn = self.module.declare_func_in_func(len_checked_callee, &mut builder.func);
                let status_fn = self.module.declare_func_in_func(status_callee, &mut builder.func);
                let status_set_fn = self.module.declare_func_in_func(status_set_callee, &mut builder.func);
                let floordiv_fn = self.module.declare_func_in_func(floordiv_callee, &mut builder.func);
                let pow_fn = self.module.declare_func_in_func(pow_callee, &mut builder.func);
                let floordiv_float_fn = self.module.declare_func_in_func(floordiv_float_callee, &mut builder.func);
                let pow_float_fn = self.module.declare_func_in_func(pow_float_callee, &mut builder.func);
                let rem_float_fn = self.module.declare_func_in_func(rem_float_callee, &mut builder.func);
                let string_concat_fn = self.module.declare_func_in_func(string_concat_callee, &mut builder.func);
                let string_append_inplace_fn = self.module.declare_func_in_func(string_append_inplace_callee, &mut builder.func);
                let string_eq_fn = self.module.declare_func_in_func(string_eq_callee, &mut builder.func);
                let string_len_fn = self.module.declare_func_in_func(string_len_callee, &mut builder.func);
                let closure_fn_ids_ref = dispatch_tables.map(|dt| &dt.fn_ids);
                let tostring_int_fn = self.module.declare_func_in_func(tostring_int_callee, &mut builder.func);
                let tostring_float_fn = self.module.declare_func_in_func(tostring_float_callee, &mut builder.func);
                let tostring_bool_fn = self.module.declare_func_in_func(tostring_bool_callee, &mut builder.func);
                let string_helpers = StringHelperFns {
                    concat_fn: string_concat_fn,
                    append_inplace_fn: string_append_inplace_fn,
                    eq_fn: string_eq_fn,
                    len_fn: string_len_fn,
                    tostring_int_fn,
                    tostring_float_fn,
                    tostring_bool_fn,
                };
                let unaliased_string_locals = if fn_has_string {
                    compute_unaliased_string_locals(mir_fn)
                } else {
                    hashbrown::HashSet::new()
                };
                let init_deopt_fn = self.module.declare_func_in_func(init_deopt_callee, &mut builder.func);
                let record_deopt_local_fn = self.module.declare_func_in_func(rec_deopt_callee, &mut builder.func);
                let native_entry_fn = self.module.declare_func_in_func(native_entry_callee, &mut builder.func);
                let simd_vector_fn = simd_vector_callee
                    .map(|cid| self.module.declare_func_in_func(cid, &mut builder.func));
                let simd_scalar_fn = simd_scalar_callee
                    .map(|cid| self.module.declare_func_in_func(cid, &mut builder.func));
                let simd_scalar_forced_fn = simd_scalar_forced_callee
                    .map(|cid| self.module.declare_func_in_func(cid, &mut builder.func));

                let emergency_exit_bb = builder.create_block();
                let empty_hoisted_arrays: hashbrown::HashMap<usize, HoistedArrayVars> =
                    hashbrown::HashMap::new();
                let empty_hoisted_structs: hashbrown::HashMap<usize, Variable> =
                    hashbrown::HashMap::new();
                let empty_pow2_mod_defs: hashbrown::HashMap<usize, (ClifValue, i64)> =
                    hashbrown::HashMap::new();

                let mut vec_acc_vars: Vec<Option<(Variable, Variable)>> = Vec::with_capacity(vec_loops.len());
                for vl in &vec_loops {
                    if matches!(
                        vl.kind,
                        crate::mir::opt::simd::VectorLoopKernelKind::Sum
                            | crate::mir::opt::simd::VectorLoopKernelKind::Dot
                    ) {
                        vec_acc_vars.push(Some((
                            builder.declare_var(types::I64X2),
                            builder.declare_var(types::I64X2),
                        )));
                    } else {
                        vec_acc_vars.push(None);
                    }
                }

                // Create blocks for each MIR block
                let mut clif_blocks = Vec::with_capacity(mir_fn.blocks.len());
                for _ in &mir_fn.blocks {
                    clif_blocks.push(builder.create_block());
                }

                // Declare SSA variables and Cranelift StackSlots for locals requiring explicit slots
                let mut local_vars = Vec::with_capacity(mir_fn.locals.len());
                let mut stack_slots: hashbrown::HashMap<usize, StackSlot> = hashbrown::HashMap::new();
                for local in &mir_fn.locals {
                    let clif_ty = mir_to_clif_type(&local.ty)?;
                    let var = builder.declare_var(clif_ty);
                    local_vars.push(var);
                    if crate::mir::opt::escape_analysis::requires_explicit_stack_slot(local) {
                        let size = if let MirType::Struct(name) = &local.ty {
                            self.struct_layouts.get(name).map(|l| l.size_bytes as u32).unwrap_or(8)
                        } else {
                            8
                        };
                        let slot = builder.create_sized_stack_slot(StackSlotData::new(
                            StackSlotKind::ExplicitSlot,
                            size.max(8),
                            3,
                        ));
                        stack_slots.insert(local.id.0, slot);
                    }
                }

                // NG-W14-5: Function-wide single-definition constant integer local map.
                let const_int_locals = compute_const_int_locals(mir_fn, osr_info, &stack_slots);

                let loop_headers: hashbrown::HashSet<usize> =
                    crate::mir::opt::Safepoints::find_loop_headers_in_fn(mir_fn)
                        .into_iter()
                        .map(|b| b.0)
                        .collect();
                let prov_plan = if fn_has_array {
                    compute_array_len_provenance_plan(mir_fn, &loop_headers, osr_info)
                } else {
                    ArrayLenProvenancePlan {
                        validated_in: vec![hashbrown::HashSet::new(); mir_fn.blocks.len()],
                        hoist_at_header_entry: hashbrown::HashMap::new(),
                        loop_bodies: hashbrown::HashMap::new(),
                    }
                };
                let mut pre_header_blocks: hashbrown::HashMap<usize, Block> =
                    hashbrown::HashMap::new();
                for (&hdr_idx, locs) in &prov_plan.hoist_at_header_entry {
                    if !locs.is_empty() {
                        pre_header_blocks.insert(hdr_idx, builder.create_block());
                    }
                }

                let needs_safepoint = !loop_headers.is_empty() || osr_info.is_some();
                let safepoint_ctx: Option<(Variable, FuncRef, FuncRef)> = if needs_safepoint {
                    let sp_ptr_fn = self
                        .module
                        .declare_func_in_func(safepoint_state_ptr_callee, &mut builder.func);
                    let sp_slow_callee_eff = preserve_all_trampolines
                        .map(|(sp_id, _)| sp_id)
                        .unwrap_or(safepoint_slow_callee);
                    let sp_slow_fn = self
                        .module
                        .declare_func_in_func(sp_slow_callee_eff, &mut builder.func);
                    let sp_ptr_var = builder.declare_var(types::I64);
                    Some((sp_ptr_var, sp_ptr_fn, sp_slow_fn))
                } else {
                    None
                };

                let needs_gas_fast = function_has_gas_checks(mir_fn, &vec_loops);
                let gas_ctx: Option<(Variable, FuncRef, FuncRef)> = if needs_gas_fast {
                    let gas_fast_ptr_fn = self
                        .module
                        .declare_func_in_func(gas_fast_ptr_callee, &mut builder.func);
                    let gas_slow_callee_eff = preserve_all_trampolines
                        .map(|(_, gas_id)| gas_id)
                        .unwrap_or(gas_slow_callee);
                    let gas_slow_fn = self
                        .module
                        .declare_func_in_func(gas_slow_callee_eff, &mut builder.func);
                    let gas_ptr_var = builder.declare_var(types::I64);
                    Some((gas_ptr_var, gas_fast_ptr_fn, gas_slow_fn))
                } else {
                    None
                };
                let inline_status_ptr_var: Option<Variable> = gas_ctx.map(|(v, _, _)| v);

                let bc_ctx = BoundsCheckCtx {
                    status_set_fn,
                    emergency_exit_bb,
                    hoisted_arrays: &empty_hoisted_arrays,
                    hoisted_structs: &empty_hoisted_structs,
                    proven_in_bounds: &[],
                    const_int_locals: &const_int_locals,
                    pow2_mod_defs: &empty_pow2_mod_defs,
                    inline_status_ptr_var,
                    closure_fn_ids: closure_fn_ids_ref,
                };

                // Cache `type_id` once in `fn_entry_bb` for read-only `MirType::Unknown` parameters
                // used across >= 2 dynamic `__mir_unknown_get_field` / `__mir_unknown_call_method` sites.
                let mut cached_unknown_tid_vars: hashbrown::HashMap<usize, Variable> =
                    hashbrown::HashMap::new();
                if osr_info.is_none() {
                    let mut dyn_use_counts: hashbrown::HashMap<usize, usize> =
                        hashbrown::HashMap::new();
                    let mut mutated_locals: hashbrown::HashSet<usize> =
                        hashbrown::HashSet::new();
                    for b in &mir_fn.blocks {
                        for stmt in &b.statements {
                            if let Statement::Assign(Place::Local(dst), _, _) = stmt {
                                mutated_locals.insert(dst.0);
                            }
                        }
                        if let Some(Terminator::Call {
                            callee,
                            args,
                            destination,
                            ..
                        }) = &b.terminator
                        {
                            if let Place::Local(dst) = destination {
                                mutated_locals.insert(dst.0);
                            }
                            if matches!(
                                callee.as_str(),
                                "__mir_unknown_get_field" | "__mir_unknown_call_method"
                            ) && let Some(
                                Operand::Copy(Place::Local(loc))
                                | Operand::Move(Place::Local(loc)),
                            ) = args.first()
                            {
                                *dyn_use_counts.entry(loc.0).or_insert(0) += 1;
                            }
                        }
                    }
                    for &p_id in &mir_fn.params {
                        if mir_fn.locals[p_id.0].ty == MirType::Unknown
                            && !mutated_locals.contains(&p_id.0)
                            && dyn_use_counts.get(&p_id.0).copied().unwrap_or(0) >= 2
                        {
                            cached_unknown_tid_vars.insert(p_id.0, builder.declare_var(types::I64));
                        }
                    }
                }

                // NG-W16-8: Inline non-escaping local integer queues & fused string.sub+string.byte locals.
                let local_read_counts = mir_fn.compute_local_read_counts();
                let (inline_local_queues, inline_queue_blocks) = compute_inline_local_queues(
                    mir_fn,
                    osr_info,
                    &stack_slots,
                    &const_int_locals,
                    &local_read_counts,
                );
                struct InlineQueueVars {
                    cap: i64,
                    stack_slot: Option<StackSlot>,
                    buf_var: Variable,
                    head_var: Variable,
                    tail_var: Variable,
                    len_var: Variable,
                }
                let mut inline_queue_vars: hashbrown::HashMap<usize, InlineQueueVars> =
                    hashbrown::HashMap::new();
                let mut inline_queue_alloc_tmp_to_q: hashbrown::HashMap<usize, usize> =
                    hashbrown::HashMap::new();
                for (&q_loc, &(_, cap, alloc_tmp)) in &inline_local_queues {
                    let stack_slot = if cap <= 128 {
                        // Reserve 1 extra scratch word at index `cap` so branchless `queue.offer_int`
                        // can redirect full-queue writes without branching or extra CLIF blocks.
                        Some(builder.create_sized_stack_slot(StackSlotData::new(
                            StackSlotKind::ExplicitSlot,
                            ((cap as u32) + 1) * 8,
                            3,
                        )))
                    } else {
                        None
                    };
                    inline_queue_vars.insert(
                        q_loc,
                        InlineQueueVars {
                            cap,
                            stack_slot,
                            buf_var: builder.declare_var(types::I64),
                            head_var: builder.declare_var(types::I64),
                            tail_var: builder.declare_var(types::I64),
                            len_var: builder.declare_var(types::I64),
                        },
                    );
                    inline_queue_alloc_tmp_to_q.insert(alloc_tmp, q_loc);
                }

                let fused_string_sub_map = if fn_has_string {
                    compute_fused_string_sub_locals(
                        mir_fn,
                        &loop_headers,
                        osr_info,
                        &stack_slots,
                    )
                } else {
                    hashbrown::HashMap::new()
                };
                struct FusedStringSubVars {
                    base_var: Variable,
                    start_var: Variable,
                    end_var: Variable,
                }
                let mut fused_sub_vars: hashbrown::HashMap<usize, FusedStringSubVars> =
                    hashbrown::HashMap::new();
                let mut fused_sub_tmp_to_sub: hashbrown::HashMap<usize, usize> =
                    hashbrown::HashMap::new();
                for (&sub_loc, &tmp_loc) in &fused_string_sub_map {
                    fused_sub_vars.insert(
                        sub_loc,
                        FusedStringSubVars {
                            base_var: builder.declare_var(types::I64),
                            start_var: builder.declare_var(types::I64),
                            end_var: builder.declare_var(types::I64),
                        },
                    );
                    fused_sub_tmp_to_sub.insert(tmp_loc, sub_loc);
                }

                // PERF-T4-2: Pure inner-loop counter register promotion & non-speculative
                // guarded-preheader array (data_ptr, len) + struct base hoisting.
                let promo_plan = compute_loop_promotion_plan(
                    mir_fn,
                    &loop_headers,
                    osr_info,
                    &vec_pre_map,
                    &vec_hdr_map,
                    &vec_loops,
                    &prov_plan,
                    &stack_slots,
                    &const_int_locals,
                    &self.struct_layouts,
                    safepoint_ctx.is_some(),
                    gas_ctx.is_some(),
                    &inline_queue_blocks,
                );

                // NG-W14-8: Unsigned range-check fusion (`(j as u64) < (len as u64)`) and
                // conservative loop-induction bounds proof feeding `proven_in_bounds`.
                let mut bounds_excluded_locals: hashbrown::HashSet<usize> =
                    stack_slots.keys().copied().collect();
                if let Some((_, live_locals)) = osr_info {
                    for &loc in live_locals {
                        bounds_excluded_locals.insert(loc.0);
                    }
                }
                let bounds_entry_idx = osr_info
                    .map(|(tb, _)| tb.0)
                    .unwrap_or(mir_fn.entry.0);
                let loop_bounds_analysis = if fn_has_array {
                    mir_fn.compute_loop_induction_bounds_analysis(
                        bounds_entry_idx,
                        &bounds_excluded_locals,
                        osr_info.is_none(),
                    )
                } else {
                    crate::mir::ir::LoopBoundsAnalysis {
                        non_neg_locals_in: vec![hashbrown::HashSet::new(); mir_fn.blocks.len()],
                        proven_in_bounds: hashbrown::HashMap::new(),
                        proven_in_bounds_at_stmt: hashbrown::HashMap::new(),
                    }
                };

                struct CounterLoopLoweringState {
                    counters: PromotedLoopCounters,
                    preheader_bb: Block,
                    emergency_flush_bb: Block,
                }
                struct SharedQuantizedRefillHub {
                    check_reload_bb: Block,
                    site_id_param: ClifValue,
                    amt: u64,
                    sites: Vec<(i64, usize, Vec<LocalId>, Block)>,
                }
                struct InnerQuantizedLoopLoweringState {
                    counters: PromotedLoopCounters,
                    emergency_flush_bb: Block,
                }
                struct HoistLoopLoweringState {
                    peeled_header_bb: Block,
                    guarded_preheader_bb: Block,
                    hoisted_arrays: hashbrown::HashMap<usize, HoistedArrayVars>,
                    hoisted_structs: hashbrown::HashMap<usize, Variable>,
                }

                let mut cl_states: Vec<CounterLoopLoweringState> =
                    Vec::with_capacity(promo_plan.counter_loops.len());
                let mut shared_q_hubs: Vec<Option<SharedQuantizedRefillHub>> =
                    Vec::with_capacity(promo_plan.counter_loops.len());
                let mut cfg_records: Vec<PromotedLoopCfgRecord> =
                    Vec::with_capacity(promo_plan.counter_loops.len());
                for (cl_idx, spec) in promo_plan.counter_loops.iter().enumerate() {
                    let gas_pair = if spec.promote_gas {
                        let (gas_ptr_var, _, _) =
                            gas_ctx.expect("promote_gas implies gas_ctx");
                        let gas_var = builder.declare_var(types::I64);
                        Some((gas_ptr_var, gas_var))
                    } else {
                        None
                    };
                    let sp_pair = if spec.promote_sp {
                        let (sp_ptr_var, _, _) =
                            safepoint_ctx.expect("promote_sp implies safepoint_ctx");
                        let sp_cnt_var = builder.declare_var(types::I64);
                        Some((sp_ptr_var, sp_cnt_var))
                    } else {
                        None
                    };
                    let quantum = spec
                        .quantized_header_gas_amt
                        .map(|amt| (builder.declare_var(types::I64), amt));
                    let preheader_bb = builder.create_block();
                    let emergency_flush_bb = builder.create_block();
                    builder.set_cold_block(emergency_flush_bb);
                    let mut flush_clif_blocks = hashbrown::HashSet::new();
                    flush_clif_blocks.insert(emergency_flush_bb);
                    let mut q_hdr_sites = spec
                        .loop_blocks
                        .iter()
                        .filter(|b| loop_headers.contains(*b))
                        .count();
                    for hspec in &promo_plan.hoist_loops {
                        if promo_plan.block_to_counter_loop.get(&hspec.hdr) == Some(&cl_idx)
                            && hspec.hdr != spec.hdr
                            && loop_headers.contains(&hspec.hdr)
                        {
                            q_hdr_sites += 1;
                        }
                    }
                    let hub = if let Some(amt) = spec.quantized_header_gas_amt
                        && q_hdr_sites >= 2
                        && inline_local_queues.is_empty()
                    {
                        let check_reload_bb = builder.create_block();
                        builder.set_cold_block(check_reload_bb);
                        let site_id_param =
                            builder.append_block_param(check_reload_bb, types::I64);
                        Some(SharedQuantizedRefillHub {
                            check_reload_bb,
                            site_id_param,
                            amt,
                            sites: Vec::with_capacity(q_hdr_sites),
                        })
                    } else {
                        None
                    };
                    shared_q_hubs.push(hub);
                    cl_states.push(CounterLoopLoweringState {
                        counters: PromotedLoopCounters {
                            gas: gas_pair,
                            sp: sp_pair,
                            quantum,
                        },
                        preheader_bb,
                        emergency_flush_bb,
                    });
                    cfg_records.push(PromotedLoopCfgRecord {
                        hdr: spec.hdr,
                        promote_gas: spec.promote_gas,
                        promote_sp: spec.promote_sp,
                        preheader_bb,
                        loop_clif_blocks: hashbrown::HashSet::new(),
                        flush_clif_blocks,
                        post_flush_clif_blocks: hashbrown::HashSet::new(),
                        reload_clif_blocks: hashbrown::HashSet::new(),
                    });
                }
                let has_any_shared_q_hub = shared_q_hubs.iter().any(Option::is_some);

                let mut iq_states: Vec<InnerQuantizedLoopLoweringState> =
                    Vec::with_capacity(promo_plan.inner_quantized_loops.len());
                let mut iq_cfg_records: Vec<InnerQuantizedCfgRecord> =
                    Vec::with_capacity(promo_plan.inner_quantized_loops.len());
                for iq_spec in &promo_plan.inner_quantized_loops {
                    let parent_counters = cl_states[iq_spec.parent_cl_idx].counters;
                    let q_var = builder.declare_var(types::I64);
                    let emergency_flush_bb = builder.create_block();
                    builder.set_cold_block(emergency_flush_bb);
                    cfg_records[iq_spec.parent_cl_idx]
                        .flush_clif_blocks
                        .insert(emergency_flush_bb);
                    let mut reconcile_clif_blocks = hashbrown::HashSet::new();
                    reconcile_clif_blocks.insert(emergency_flush_bb);
                    iq_states.push(InnerQuantizedLoopLoweringState {
                        counters: PromotedLoopCounters {
                            gas: parent_counters.gas,
                            sp: parent_counters.sp,
                            quantum: Some((q_var, iq_spec.header_gas_amt)),
                        },
                        emergency_flush_bb,
                    });
                    iq_cfg_records.push(InnerQuantizedCfgRecord {
                        hdr: iq_spec.hdr,
                        loop_clif_blocks: hashbrown::HashSet::new(),
                        reconcile_clif_blocks,
                        post_flush_clif_blocks: hashbrown::HashSet::new(),
                        reload_clif_blocks: hashbrown::HashSet::new(),
                    });
                }

                let mut hl_states: Vec<HoistLoopLoweringState> =
                    Vec::with_capacity(promo_plan.hoist_loops.len());
                for hspec in &promo_plan.hoist_loops {
                    let peeled_header_bb = builder.create_block();
                    let guarded_preheader_bb = builder.create_block();
                    let mut hoisted_arrays = hashbrown::HashMap::new();
                    for &arr_loc in &hspec.hoist_arrays {
                        hoisted_arrays.insert(
                            arr_loc.0,
                            HoistedArrayVars {
                                len_var: builder.declare_var(types::I64),
                                data_ptr_var: builder.declare_var(types::I64),
                            },
                        );
                    }
                    let mut hoisted_structs = hashbrown::HashMap::new();
                    for &st_loc in &hspec.hoist_structs {
                        hoisted_structs.insert(st_loc.0, builder.declare_var(types::I64));
                    }
                    hl_states.push(HoistLoopLoweringState {
                        peeled_header_bb,
                        guarded_preheader_bb,
                        hoisted_arrays,
                        hoisted_structs,
                    });
                }

                // Helper: resolve the entry Cranelift block of `target` from `from_block`
                // BEFORE applying any outgoing exit-flush for `from_block`.
                let resolve_dest_entry_bb = |from_block: Option<usize>, target: usize| -> Result<Block, MirCraneliftBackendError> {
                    let direct_bb = clif_blocks.get(target).copied().ok_or_else(|| {
                        MirCraneliftBackendError::CodegenError(format!(
                            "Target block BlockId({target}) out of bounds"
                        ))
                    })?;
                    if let Some(&pre_bb) = pre_header_blocks.get(&target) {
                        let is_backedge_or_internal = from_block.is_some_and(|fb| {
                            prov_plan
                                .loop_bodies
                                .get(&target)
                                .is_some_and(|body| body.contains(&fb))
                        });
                        if !is_backedge_or_internal {
                            return Ok(pre_bb);
                        }
                    }
                    if let Some(&cl_idx) = promo_plan.hdr_to_counter_loop.get(&target) {
                        let is_internal = from_block.is_some_and(|fb| {
                            promo_plan.counter_loops[cl_idx].loop_blocks.contains(&fb)
                        });
                        if !is_internal {
                            return Ok(cl_states[cl_idx].preheader_bb);
                        }
                    }
                    if let Some(&hl_idx) = promo_plan.hdr_to_hoist_loop.get(&target) {
                        let is_internal = from_block.is_some_and(|fb| {
                            promo_plan.hoist_loops[hl_idx].loop_blocks.contains(&fb)
                        });
                        if !is_internal {
                            return Ok(hl_states[hl_idx].peeled_header_bb);
                        }
                    }
                    Ok(direct_bb)
                };

                // Pre-create dedicated critical-edge exit-flush blocks for every normal exit edge
                // leaving a counter-promoted loop (Barrier 4).
                let mut exit_flush_by_edge: hashbrown::HashMap<(usize, usize), Block> =
                    hashbrown::HashMap::new();
                let mut exit_flush_blocks_per_loop: Vec<Vec<(Block, Block)>> =
                    vec![Vec::new(); promo_plan.counter_loops.len()];
                for (cl_idx, spec) in promo_plan.counter_loops.iter().enumerate() {
                    let mut exit_targets: Vec<usize> = Vec::new();
                    let mut sorted_blocks: Vec<usize> = spec.loop_blocks.iter().copied().collect();
                    sorted_blocks.sort_unstable();
                    for b_idx in sorted_blocks {
                        if let Some(term) = &mir_fn.blocks[b_idx].terminator {
                            for succ in term.successors() {
                                if !spec.loop_blocks.contains(&succ) && !exit_targets.contains(&succ) {
                                    exit_targets.push(succ);
                                }
                            }
                        }
                        if let Some(&vl_idx) = vec_hdr_map.get(&b_idx) {
                            let tail = vec_loops[vl_idx].tail_block.0;
                            if !spec.loop_blocks.contains(&tail) && !exit_targets.contains(&tail) {
                                exit_targets.push(tail);
                            }
                        }
                    }
                    exit_targets.sort_unstable();
                    let mut by_dest_bb: hashbrown::HashMap<Block, Block> = hashbrown::HashMap::new();
                    for target in exit_targets {
                        let dest_bb = resolve_dest_entry_bb(Some(spec.hdr), target)?;
                        let flush_bb = if let Some(&existing) = by_dest_bb.get(&dest_bb) {
                            existing
                        } else {
                            let fbb = builder.create_block();
                            by_dest_bb.insert(dest_bb, fbb);
                            exit_flush_blocks_per_loop[cl_idx].push((dest_bb, fbb));
                            cfg_records[cl_idx].flush_clif_blocks.insert(fbb);
                            fbb
                        };
                        exit_flush_by_edge.insert((cl_idx, target), flush_bb);
                    }
                }

                // Pre-create dedicated critical-edge exit-reconcile / exit-flush blocks for every
                // normal exit edge leaving a nested inner quantized loop (NG-W14-1).
                let mut iq_exit_by_edge: hashbrown::HashMap<(usize, usize), Block> =
                    hashbrown::HashMap::new();
                let mut iq_exit_blocks_per_loop: Vec<Vec<(Block, Block, bool)>> =
                    vec![Vec::new(); promo_plan.inner_quantized_loops.len()];
                for (iq_idx, iq_spec) in promo_plan.inner_quantized_loops.iter().enumerate() {
                    let parent_cl_idx = iq_spec.parent_cl_idx;
                    let parent_loop_blocks = &promo_plan.counter_loops[parent_cl_idx].loop_blocks;
                    let mut exit_targets: Vec<usize> = Vec::new();
                    let mut sorted_blocks: Vec<usize> = iq_spec.loop_blocks.iter().copied().collect();
                    sorted_blocks.sort_unstable();
                    for b_idx in sorted_blocks {
                        if let Some(term) = &mir_fn.blocks[b_idx].terminator {
                            for succ in term.successors() {
                                if !iq_spec.loop_blocks.contains(&succ)
                                    && !exit_targets.contains(&succ)
                                {
                                    exit_targets.push(succ);
                                }
                            }
                        }
                    }
                    exit_targets.sort_unstable();
                    let mut by_dest_key: hashbrown::HashMap<(Block, bool), Block> =
                        hashbrown::HashMap::new();
                    for target in exit_targets {
                        let exits_outer_cl = !parent_loop_blocks.contains(&target);
                        let dest_bb = resolve_dest_entry_bb(Some(iq_spec.hdr), target)?;
                        let exit_bb = if let Some(&existing) =
                            by_dest_key.get(&(dest_bb, exits_outer_cl))
                        {
                            existing
                        } else {
                            let ebb = builder.create_block();
                            by_dest_key.insert((dest_bb, exits_outer_cl), ebb);
                            iq_exit_blocks_per_loop[iq_idx].push((dest_bb, ebb, exits_outer_cl));
                            iq_cfg_records[iq_idx].reconcile_clif_blocks.insert(ebb);
                            if exits_outer_cl {
                                cfg_records[parent_cl_idx].flush_clif_blocks.insert(ebb);
                            } else {
                                cfg_records[parent_cl_idx].loop_clif_blocks.insert(ebb);
                            }
                            ebb
                        };
                        iq_exit_by_edge.insert((iq_idx, target), exit_bb);
                    }
                }

                let resolve_target_bb = |from_block: Option<usize>, target: usize| -> Result<Block, MirCraneliftBackendError> {
                    if let Some(fb) = from_block {
                        if let Some(&iq_idx) = promo_plan.block_to_inner_quantized.get(&fb)
                            && !promo_plan.inner_quantized_loops[iq_idx].loop_blocks.contains(&target)
                        {
                            if let Some(&iq_exit_bb) = iq_exit_by_edge.get(&(iq_idx, target)) {
                                return Ok(iq_exit_bb);
                            }
                        }
                        if let Some(&cl_idx) = promo_plan.block_to_counter_loop.get(&fb)
                            && !promo_plan.counter_loops[cl_idx].loop_blocks.contains(&target)
                        {
                            if let Some(&flush_bb) = exit_flush_by_edge.get(&(cl_idx, target)) {
                                return Ok(flush_bb);
                            }
                        }
                    }
                    resolve_dest_entry_bb(from_block, target)
                };

                if let Some((target_block, live_locals)) = osr_info {
                    let osr_entry_bb = builder.create_block();
                    builder.append_block_params_for_function_params(osr_entry_bb);
                    builder.switch_to_block(osr_entry_bb);
                    builder.ins().call(native_entry_fn, &[]);

                    let (sp_ptr_var, sp_ptr_fn, sp_slow_fn) =
                        safepoint_ctx.expect("osr_info implies needs_safepoint");
                    let sp_ptr_call = builder.ins().call(sp_ptr_fn, &[]);
                    let sp_ptr_val = builder.inst_results(sp_ptr_call)[0];
                    builder.def_var(sp_ptr_var, sp_ptr_val);

                    if let Some((gas_ptr_var, gas_fast_ptr_fn, _)) = gas_ctx {
                        let gas_ptr_call = builder.ins().call(gas_fast_ptr_fn, &[]);
                        let gas_ptr_val = builder.inst_results(gas_ptr_call)[0];
                        builder.def_var(gas_ptr_var, gas_ptr_val);
                    }

                    for (idx, loc_id) in live_locals.iter().enumerate() {
                        let param_val = builder.block_params(osr_entry_bb)[idx];
                        let clif_ty = mir_to_clif_type(&mir_fn.locals[loc_id.0].ty)?;
                        let var_val = if clif_ty == types::F64 {
                            builder.ins().bitcast(types::F64, MemFlagsData::new(), param_val)
                        } else if clif_ty == types::I8 {
                            builder.ins().ireduce(types::I8, param_val)
                        } else {
                            param_val
                        };
                        builder.def_var(local_vars[loc_id.0], var_val);
                        if let Some(&slot) = stack_slots.get(&loc_id.0) {
                            builder.ins().stack_store(types::I64, var_val, slot, 0);
                        }
                    }
                    for (l_idx, local) in mir_fn.locals.iter().enumerate() {
                        if !live_locals.iter().any(|lid| lid.0 == l_idx) {
                            let clif_ty = mir_to_clif_type(&local.ty)?;
                            let zero = if clif_ty == types::F64 {
                                builder.ins().f64const(0.0)
                            } else {
                                builder.ins().iconst(clif_ty, 0)
                            };
                            builder.def_var(local_vars[l_idx], zero);
                            if let Some(&slot) = stack_slots.get(&l_idx) {
                                builder.ins().stack_store(types::I64, zero, slot, 0);
                            }
                        }
                    }

                    emit_safepoint_check(
                        &mut builder,
                        sp_ptr_var,
                        sp_slow_fn,
                        init_deopt_fn,
                        record_deopt_local_fn,
                        target_block.0,
                        live_locals,
                        &local_vars,
                        &stack_slots,
                        &mir_fn.locals,
                        emergency_exit_bb,
                        None,
                        None,
                        None,
                    );

                    let osr_target_bb = resolve_target_bb(None, target_block.0)?;
                    builder.ins().jump(osr_target_bb, &[]);
                } else {
                    let fn_entry_bb = builder.create_block();
                    builder.append_block_params_for_function_params(fn_entry_bb);
                    builder.switch_to_block(fn_entry_bb);
                    if emit_native_entry {
                        builder.ins().call(native_entry_fn, &[]);
                    }
                    if let Some((sp_ptr_var, sp_ptr_fn, _)) = safepoint_ctx {
                        let sp_ptr_call = builder.ins().call(sp_ptr_fn, &[]);
                        let sp_ptr_val = builder.inst_results(sp_ptr_call)[0];
                        builder.def_var(sp_ptr_var, sp_ptr_val);
                    }
                    if let Some((gas_ptr_var, gas_fast_ptr_fn, _)) = gas_ctx {
                        let gas_ptr_call = builder.ins().call(gas_fast_ptr_fn, &[]);
                        let gas_ptr_val = builder.inst_results(gas_ptr_call)[0];
                        builder.def_var(gas_ptr_var, gas_ptr_val);
                    }

                    for (idx, p_id) in mir_fn.params.iter().enumerate() {
                        let param_val = builder.block_params(fn_entry_bb)[idx];
                        builder.def_var(local_vars[p_id.0], param_val);
                        if let Some(&slot) = stack_slots.get(&p_id.0) {
                            builder.ins().stack_store(types::I64, param_val, slot, 0);
                        }
                        if let Some(&tid_var) = cached_unknown_tid_vars.get(&p_id.0) {
                            let unknown_type_id_fn = Self::declare_ext_fn(
                                &mut self.module,
                                &mut builder.func,
                                &mut ext_fn_cache,
                                "lust_mir_clif_unknown_type_id",
                                &[types::I64],
                                Some(types::I64),
                            )?;
                            let tid_call = builder.ins().call(unknown_type_id_fn, &[param_val]);
                            let tid_val = builder.inst_results(tid_call)[0];
                            builder.def_var(tid_var, tid_val);
                        }
                    }
                    if has_any_shared_q_hub {
                        // Ensure every non-param SSA variable has a dominating definition so
                        // cold merges in `SharedQuantizedRefillHub` remain 100% SSA-valid
                        // (unused initializers are eliminated by Cranelift's SSA builder).
                        let zero_i64 = builder.ins().iconst(types::I64, 0);
                        let zero_i8 = builder.ins().iconst(types::I8, 0);
                        let zero_f64 = builder.ins().f64const(0.0);
                        for (l_idx, local) in mir_fn.locals.iter().enumerate() {
                            if mir_fn.params.iter().any(|p| p.0 == l_idx)
                                || stack_slots.contains_key(&l_idx)
                            {
                                continue;
                            }
                            let clif_ty = mir_to_clif_type(&local.ty)?;
                            let z = if clif_ty == types::F64 {
                                zero_f64
                            } else if clif_ty == types::I8 {
                                zero_i8
                            } else {
                                zero_i64
                            };
                            builder.def_var(local_vars[l_idx], z);
                        }
                    }
                    for acc_pair in vec_acc_vars.iter().flatten() {
                        let zero_s = builder.ins().iconst(types::I64, 0);
                        let vzero = builder.ins().splat(types::I64X2, zero_s);
                        builder.def_var(acc_pair.0, vzero);
                        builder.def_var(acc_pair.1, vzero);
                    }
                    let entry_target_bb = resolve_target_bb(None, mir_fn.entry.0)?;
                    builder.ins().jump(entry_target_bb, &[]);
                }

                // Emit loop-preheader hoisted array provenance validation blocks (run once per loop entry)
                let mut hoisted_hdrs: Vec<usize> = pre_header_blocks.keys().copied().collect();
                hoisted_hdrs.sort_unstable();
                for hdr_idx in hoisted_hdrs {
                    // Check whether this pre_header_block sits strictly inside an outer counter-promoted loop
                    let enclosing_outer_cl = promo_plan
                        .block_to_counter_loop
                        .get(&hdr_idx)
                        .copied()
                        .filter(|&ci| promo_plan.counter_loops[ci].hdr != hdr_idx);
                    let blocks_before: hashbrown::HashSet<Block> = if enclosing_outer_cl.is_some() {
                        builder.func.layout.blocks().collect()
                    } else {
                        hashbrown::HashSet::new()
                    };
                    let pre_err_bb = enclosing_outer_cl
                        .map(|ci| cl_states[ci].emergency_flush_bb)
                        .unwrap_or(emergency_exit_bb);
                    let pre_bb = pre_header_blocks[&hdr_idx];
                    builder.switch_to_block(pre_bb);
                    if let Some(locs) = prov_plan.hoist_at_header_entry.get(&hdr_idx) {
                        for &loc in locs {
                            let arr_val =
                                load_local_i64(loc, &mut builder, &local_vars, &stack_slots)?;
                            let call_inst = builder.ins().call(len_fn, &[arr_val]);
                            let res = builder.inst_results(call_inst)[0];
                            let is_err = builder.ins().icmp_imm_s(IntCC::SignedLessThan, res, 0);
                            let ok_bb = builder.create_block();
                            builder.ins().brif(is_err, pre_err_bb, &[], ok_bb, &[]);
                            builder.switch_to_block(ok_bb);
                        }
                    }
                    let next_bb = if let Some(&cl_idx) = promo_plan.hdr_to_counter_loop.get(&hdr_idx) {
                        cl_states[cl_idx].preheader_bb
                    } else if let Some(&hl_idx) = promo_plan.hdr_to_hoist_loop.get(&hdr_idx) {
                        hl_states[hl_idx].peeled_header_bb
                    } else {
                        clif_blocks[hdr_idx]
                    };
                    builder.ins().jump(next_bb, &[]);
                    if let Some(ci) = enclosing_outer_cl {
                        let rec = &mut cfg_records[ci];
                        for bb in builder.func.layout.blocks() {
                            if !blocks_before.contains(&bb)
                                && !rec.flush_clif_blocks.contains(&bb)
                                && !rec.post_flush_clif_blocks.contains(&bb)
                                && !rec.reload_clif_blocks.contains(&bb)
                            {
                                rec.loop_clif_blocks.insert(bb);
                            }
                        }
                    }
                }

                // Emit PERF-T4-2 counter-promoted loop preheaders (load gas_var & sp_cnt_var once per entry)
                for (cl_idx, spec) in promo_plan.counter_loops.iter().enumerate() {
                    builder.switch_to_block(cl_states[cl_idx].preheader_bb);
                    emit_promoted_counters_load(&mut builder, cl_states[cl_idx].counters);
                    // NG-W14-1: Initialize quantum_rem = 0 for every nested inner quantized loop
                    // inside this counter-promoted loop so it dominates all inner loop entries.
                    for (iq_idx, iq_spec) in promo_plan.inner_quantized_loops.iter().enumerate() {
                        if iq_spec.parent_cl_idx == cl_idx {
                            if let Some((q_var, _)) = iq_states[iq_idx].counters.quantum {
                                let zero = builder.ins().iconst(types::I64, 0);
                                builder.def_var(q_var, zero);
                            }
                        }
                    }
                    let entry_hdr_bb = if let Some(&hl_idx) = promo_plan.hdr_to_hoist_loop.get(&spec.hdr) {
                        hl_states[hl_idx].peeled_header_bb
                    } else {
                        clif_blocks[spec.hdr]
                    };
                    builder.ins().jump(entry_hdr_bb, &[]);
                }

                // Emit PERF-T4-2 non-speculative peeled-entry headers & guarded preheaders
                for (hl_idx, hspec) in promo_plan.hoist_loops.iter().enumerate() {
                    let cl_opt = promo_plan.block_to_counter_loop.get(&hspec.hdr).copied();
                    let iq_opt = promo_plan.block_to_inner_quantized.get(&hspec.hdr).copied();
                    // NG-W15-6: `peeled_header_bb` runs a 1-iteration inline check and leaves
                    // `quantum_rem == 0` (deferring Q=64 arming to backedge header `clif_blocks[hspec.hdr]`).
                    // For a peeled inner quantized loop header, `peeled_header_bb` & `guarded_preheader_bb`
                    // belong strictly to the enclosing outer counter-promoted loop (zero cold refill and
                    // zero `iq_exit_bb` reconcile on 0-trip inner loops such as `sieve.lust`).
                    let is_peeled_iq_hdr = iq_opt
                        .is_some_and(|iqi| promo_plan.inner_quantized_loops[iqi].hdr == hspec.hdr);
                    let blocks_before: hashbrown::HashSet<Block> = if cl_opt.is_some() || iq_opt.is_some() {
                        builder.func.layout.blocks().collect()
                    } else {
                        hashbrown::HashSet::new()
                    };

                    // Hoisted vars visible inside peeled_header_bb come ONLY from strictly enclosing outer loops!
                    let mut outer_hoisted_arrays: hashbrown::HashMap<usize, HoistedArrayVars> =
                        hashbrown::HashMap::new();
                    let mut outer_hoisted_structs: hashbrown::HashMap<usize, Variable> =
                        hashbrown::HashMap::new();
                    for (other_idx, other_h) in promo_plan.hoist_loops.iter().enumerate() {
                        if other_idx != hl_idx && other_h.loop_blocks.contains(&hspec.hdr) {
                            outer_hoisted_arrays.extend(hl_states[other_idx].hoisted_arrays.iter().map(|(&k, &v)| (k, v)));
                            for &(alias_loc, root_loc) in &other_h.hoist_array_aliases {
                                if let Some(&hvars) = hl_states[other_idx].hoisted_arrays.get(&root_loc.0) {
                                    outer_hoisted_arrays.insert(alias_loc.0, hvars);
                                }
                            }
                            outer_hoisted_structs.extend(hl_states[other_idx].hoisted_structs.iter().map(|(&k, &v)| (k, v)));
                        }
                    }
                    let active_emergency_bb = if is_peeled_iq_hdr {
                        cl_opt
                            .map(|ci| cl_states[ci].emergency_flush_bb)
                            .unwrap_or(emergency_exit_bb)
                    } else if let Some(iqi) = iq_opt {
                        iq_states[iqi].emergency_flush_bb
                    } else if let Some(ci) = cl_opt {
                        cl_states[ci].emergency_flush_bb
                    } else {
                        emergency_exit_bb
                    };
                    let peeled_bc_ctx = BoundsCheckCtx {
                        status_set_fn,
                        emergency_exit_bb: active_emergency_bb,
                        hoisted_arrays: &outer_hoisted_arrays,
                        hoisted_structs: &outer_hoisted_structs,
                        proven_in_bounds: &[],
                        const_int_locals: &const_int_locals,
                        pow2_mod_defs: &empty_pow2_mod_defs,
                        inline_status_ptr_var,
                        closure_fn_ids: closure_fn_ids_ref,
                    };
                    let active_promoted = if is_peeled_iq_hdr {
                        cl_opt.map(|ci| cl_states[ci].counters)
                    } else if let Some(iqi) = iq_opt {
                        Some(iq_states[iqi].counters)
                    } else {
                        cl_opt.map(|ci| cl_states[ci].counters)
                    };

                    builder.switch_to_block(hl_states[hl_idx].peeled_header_bb);
                    let mut cur_validated = prov_plan.validated_in[hspec.hdr].clone();

                    // When `hspec.hdr == cl_spec.hdr` (the outermost entry of a quantized loop),
                    // `peeled_header_bb` executes ONCE with `quantum_rem == 0`, so simple 1-iteration
                    // safepoint+gas avoids duplicating 10 cold quantized blocks (saving compile time).
                    // For peeled INNER loops inside a quantized outer loop (`hspec.hdr != cl_spec.hdr`),
                    // `quantum_rem` may be > 0, so we MUST use `emit_quantized_loop_header_check`.
                    let peeled_quantized_amt = if !is_peeled_iq_hdr {
                        cl_opt.and_then(|ci| {
                            let spec = &promo_plan.counter_loops[ci];
                            if spec.hdr != hspec.hdr {
                                spec.quantized_header_gas_amt
                            } else {
                                None
                            }
                        })
                    } else {
                        None
                    };
                    let mut skip_first_header_gas_check = false;
                    if loop_headers.contains(&hspec.hdr) {
                        let proven_entry_storage;
                        let locals_to_record = if let Some((_, live_locals)) = osr_info {
                            live_locals
                        } else {
                            match mir_fn.derive_proven_site_metadata_at_entry(hspec.hdr) {
                                Ok(proven) => {
                                    proven_entry_storage = proven.into_iter().map(LocalId).collect::<Vec<_>>();
                                    &proven_entry_storage
                                }
                                Err(err) => {
                                    return Err(MirCraneliftBackendError::CodegenError(alloc::format!(
                                        "Cannot derive authoritative deopt metadata for peeled loop header safepoint {}: {}",
                                        hspec.hdr, err
                                    )));
                                }
                            }
                        };
                        let (sp_ptr_var, _, sp_slow_fn) =
                            safepoint_ctx.expect("loop header implies needs_safepoint");
                        if let Some(q_amt) = peeled_quantized_amt {
                            let promoted_counters =
                                active_promoted.expect("quantized loop header implies promoted counters");
                            if let Some(ci) = cl_opt
                                && let Some(hub) = shared_q_hubs[ci].as_mut()
                            {
                                let (quantum_rem_var, _) = promoted_counters
                                    .quantum
                                    .expect("quantized loop header requires quantum_rem_var");
                                let q0 = builder.use_var(quantum_rem_var);
                                let q1 = builder.ins().iadd_imm_s(q0, -1);
                                let needs_refill =
                                    builder.ins().icmp_imm_s(IntCC::SignedLessThan, q1, 0);
                                builder.def_var(quantum_rem_var, q1);
                                let cont_bb = builder.create_block();
                                let site_id = hub.sites.len() as i64;
                                let site_id_val = builder.ins().iconst(types::I64, site_id);
                                builder.ins().brif(
                                    needs_refill,
                                    hub.check_reload_bb,
                                    &[BlockArg::Value(site_id_val)],
                                    cont_bb,
                                    &[],
                                );
                                hub.sites.push((
                                    site_id,
                                    hspec.hdr,
                                    locals_to_record.to_vec(),
                                    cont_bb,
                                ));
                                builder.switch_to_block(cont_bb);
                            } else {
                                let (_, _, gas_slow_fn) =
                                    gas_ctx.expect("quantized loop header implies gas_ctx");
                                emit_quantized_loop_header_check(
                                    &mut builder,
                                    sp_ptr_var,
                                    sp_slow_fn,
                                    gas_slow_fn,
                                    init_deopt_fn,
                                    record_deopt_local_fn,
                                    hspec.hdr,
                                    q_amt,
                                    locals_to_record,
                                    &local_vars,
                                    &stack_slots,
                                    &mir_fn.locals,
                                    emergency_exit_bb,
                                    promoted_counters,
                                    cl_opt.map(|ci| &mut cfg_records[ci]),
                                    None,
                                );
                            }
                            skip_first_header_gas_check = true;
                        } else {
                            // NG-W15-6: Emit inline 1-iteration safepoint check in `peeled_header_bb`
                            // (statement 0 then runs `emit_inline_gas_check`), deferring Q=64 quantum arming
                            // to backedge header `clif_blocks[hspec.hdr]`.
                            emit_safepoint_check(
                                &mut builder,
                                sp_ptr_var,
                                sp_slow_fn,
                                init_deopt_fn,
                                record_deopt_local_fn,
                                hspec.hdr,
                                locals_to_record,
                                &local_vars,
                                &stack_slots,
                                &mir_fn.locals,
                                emergency_exit_bb,
                                None,
                                active_promoted,
                                cl_opt.map(|ci| &mut cfg_records[ci]),
                            );
                        }
                    }

                    let mut peeled_pow2_mod_defs: hashbrown::HashMap<usize, (ClifValue, i64)> =
                        hashbrown::HashMap::new();
                    let peeled_range_fusion = mir_fn.detect_unsigned_range_checks_in_block(
                        hspec.hdr,
                        &local_read_counts,
                        &bounds_excluded_locals,
                        &loop_bounds_analysis.non_neg_locals_in[hspec.hdr],
                    );
                    for (stmt_idx, stmt) in mir_fn.blocks[hspec.hdr].statements.iter().enumerate() {
                        if stmt_idx == 0 && skip_first_header_gas_check {
                            continue;
                        }
                        if peeled_range_fusion.skip_stmts.contains(&stmt_idx) {
                            continue;
                        }
                        let peeled_stmt_bc_ctx = BoundsCheckCtx {
                            status_set_fn,
                            emergency_exit_bb: active_emergency_bb,
                            hoisted_arrays: &outer_hoisted_arrays,
                            hoisted_structs: &outer_hoisted_structs,
                            proven_in_bounds: &[],
                            const_int_locals: &const_int_locals,
                            pow2_mod_defs: &peeled_pow2_mod_defs,
                            inline_status_ptr_var,
                            closure_fn_ids: closure_fn_ids_ref,
                        };
                        if let Some(rc) = peeled_range_fusion.fused_checks.get(&stmt_idx) {
                            let dst_decl = mir_fn.locals.get(rc.dst.0).ok_or_else(|| {
                                MirCraneliftBackendError::CodegenError(format!(
                                    "Destination local {:?} out of bounds",
                                    rc.dst
                                ))
                            })?;
                            let idx_val = load_local_i64(
                                rc.idx_local,
                                &mut builder,
                                &local_vars,
                                &stack_slots,
                            )?;
                            let mut len_val = lower_operand(
                                &rc.len_op,
                                &mut builder,
                                &local_vars,
                                &stack_slots,
                                &mir_fn.locals,
                                &self.struct_layouts,
                                peeled_stmt_bc_ctx,
                            )?;
                            if builder.func.dfg.value_type(len_val) != types::I64 {
                                len_val = cast_clif_type(&mut builder, len_val, types::I64);
                            }
                            let in_bounds = builder
                                .ins()
                                .icmp(IntCC::UnsignedLessThan, idx_val, len_val);
                            let target_ty = mir_to_clif_type(&dst_decl.ty)?;
                            let val = cast_clif_type(&mut builder, in_bounds, target_ty);
                            store_local_val(rc.dst, val, &mut builder, &local_vars, &stack_slots)?;
                            peeled_pow2_mod_defs.remove(&rc.dst.0);
                            cur_validated.remove(&rc.dst.0);
                            continue;
                        }
                        match stmt {
                            Statement::Assign(Place::Local(dst), rvalue, _) => {
                                let dst_decl = mir_fn.locals.get(dst.0).ok_or_else(|| {
                                    MirCraneliftBackendError::CodegenError(format!(
                                        "Destination local {dst:?} out of bounds"
                                    ))
                                })?;
                                let inplace_concat_dst =
                                    if unaliased_string_locals.contains(&dst.0) {
                                        Some(*dst)
                                    } else {
                                        None
                                    };
                                let mut pow2_mod_record: Option<(ClifValue, i64)> = None;
                                let val = lower_rvalue(
                                    rvalue,
                                    Some(&dst_decl.ty),
                                    inplace_concat_dst,
                                    &mut cur_validated,
                                    &mut builder,
                                    &local_vars,
                                    &stack_slots,
                                    &mir_fn.locals,
                                    &self.struct_layouts,
                                    alloc_array_fn,
                                    alloc_array_typed_fn,
                                    alloc_tuple_typed_fn,
                                    alloc_struct_named_fn,
                                    alloc_cflat_struct_fn,
                                    len_fn,
                                    floordiv_fn,
                                    pow_fn,
                                    floordiv_float_fn,
                                    pow_float_fn,
                                    rem_float_fn,
                                    string_helpers,
                                    status_fn,
                                    peeled_stmt_bc_ctx,
                                    Some(&mut pow2_mod_record),
                                )?;
                                let target_ty = mir_to_clif_type(&dst_decl.ty)?;
                                let val = cast_clif_type(&mut builder, val, target_ty);
                                store_local_val(*dst, val, &mut builder, &local_vars, &stack_slots)?;
                                if let Some(entry) = pow2_mod_record
                                    && !stack_slots.contains_key(&dst.0)
                                {
                                    peeled_pow2_mod_defs.insert(dst.0, entry);
                                } else {
                                    peeled_pow2_mod_defs.remove(&dst.0);
                                }
                                match rvalue {
                                    Rvalue::Aggregate(MirType::Array(_), _) => {
                                        cur_validated.insert(dst.0);
                                    }
                                    Rvalue::Use(
                                        Operand::Copy(Place::Local(src))
                                        | Operand::Move(Place::Local(src)),
                                    ) if cur_validated.contains(&src.0) => {
                                        cur_validated.insert(dst.0);
                                    }
                                    _ => {
                                        cur_validated.remove(&dst.0);
                                    }
                                }
                            }
                            Statement::Assign(lhs_place @ Place::Index(..), rvalue, _) => {
                                let val = lower_rvalue(
                                    rvalue,
                                    None,
                                    None,
                                    &mut cur_validated,
                                    &mut builder,
                                    &local_vars,
                                    &stack_slots,
                                    &mir_fn.locals,
                                    &self.struct_layouts,
                                    alloc_array_fn,
                                    alloc_array_typed_fn,
                                    alloc_tuple_typed_fn,
                                    alloc_struct_named_fn,
                                    alloc_cflat_struct_fn,
                                    len_fn,
                                    floordiv_fn,
                                    pow_fn,
                                    floordiv_float_fn,
                                    pow_float_fn,
                                    rem_float_fn,
                                    string_helpers,
                                    status_fn,
                                    peeled_stmt_bc_ctx,
                                    None,
                                )?;
                                let elem_addr = lower_place_addr(
                                    lhs_place,
                                    &mut builder,
                                    &local_vars,
                                    &stack_slots,
                                    &mir_fn.locals,
                                    &self.struct_layouts,
                                    peeled_stmt_bc_ctx,
                                )?;
                                let val_ty = builder.func.dfg.value_type(val);
                                let val_to_store = if is_bool_array_index_place(
                                    lhs_place,
                                    &mir_fn.locals,
                                    &self.struct_layouts,
                                ) {
                                    if val_ty == types::I8 {
                                        val
                                    } else {
                                        builder.ins().icmp_imm_s(IntCC::NotEqual, val, 0)
                                    }
                                } else if val_ty == types::I8 {
                                    builder.ins().uextend(types::I64, val)
                                } else {
                                    val
                                };
                                builder.ins().store(MemFlagsData::trusted(), val_to_store, elem_addr, 0);
                            }
                            Statement::Assign(lhs_place @ Place::Field(..), rvalue, _) => {
                                let val = lower_rvalue(
                                    rvalue,
                                    None,
                                    None,
                                    &mut cur_validated,
                                    &mut builder,
                                    &local_vars,
                                    &stack_slots,
                                    &mir_fn.locals,
                                    &self.struct_layouts,
                                    alloc_array_fn,
                                    alloc_array_typed_fn,
                                    alloc_tuple_typed_fn,
                                    alloc_struct_named_fn,
                                    alloc_cflat_struct_fn,
                                    len_fn,
                                    floordiv_fn,
                                    pow_fn,
                                    floordiv_float_fn,
                                    pow_float_fn,
                                    rem_float_fn,
                                    string_helpers,
                                    status_fn,
                                    peeled_stmt_bc_ctx,
                                    None,
                                )?;
                                let field_addr = lower_place_addr(
                                    lhs_place,
                                    &mut builder,
                                    &local_vars,
                                    &stack_slots,
                                    &mir_fn.locals,
                                    &self.struct_layouts,
                                    peeled_bc_ctx,
                                )?;
                                let val_ty = builder.func.dfg.value_type(val);
                                let val_to_store = if val_ty == types::I8 {
                                    builder.ins().uextend(types::I64, val)
                                } else {
                                    val
                                };
                                builder.ins().store(MemFlagsData::trusted(), val_to_store, field_addr, 0);
                            }
                            Statement::ResourceCheck(ResourceKind::Gas, amt, _) => {
                                let (gas_ptr_var, _, gas_slow_fn) =
                                    gas_ctx.expect("ResourceCheck(Gas) implies needs_gas_fast");
                                emit_inline_gas_check(
                                    &mut builder,
                                    gas_ptr_var,
                                    gas_slow_fn,
                                    *amt,
                                    emergency_exit_bb,
                                    active_promoted,
                                    cl_opt.map(|ci| &mut cfg_records[ci]),
                                );
                            }
                            _ => {}
                        }
                    }

                    // Peel branch: if entry condition holds -> guarded_preheader_bb; else -> exit!
                    let Some(Terminator::Cond { cond, .. }) = &mir_fn.blocks[hspec.hdr].terminator else {
                        unreachable!("HoistLoopSpec requires Terminator::Cond header");
                    };
                    let c_val = lower_operand(
                        cond,
                        &mut builder,
                        &local_vars,
                        &stack_slots,
                        &mir_fn.locals,
                        &self.struct_layouts,
                        peeled_bc_ctx,
                    )?;
                    let exit_bb = if is_peeled_iq_hdr {
                        if let Some(ci) = cl_opt
                            && !promo_plan.counter_loops[ci]
                                .loop_blocks
                                .contains(&hspec.loop_exit)
                        {
                            exit_flush_by_edge
                                .get(&(ci, hspec.loop_exit))
                                .copied()
                                .unwrap_or(resolve_dest_entry_bb(Some(hspec.hdr), hspec.loop_exit)?)
                        } else {
                            resolve_dest_entry_bb(Some(hspec.hdr), hspec.loop_exit)?
                        }
                    } else {
                        resolve_target_bb(Some(hspec.hdr), hspec.loop_exit)?
                    };
                    let guard_bb = hl_states[hl_idx].guarded_preheader_bb;
                    if hspec.then_is_body {
                        builder.ins().brif(c_val, guard_bb, &[], exit_bb, &[]);
                    } else {
                        builder.ins().brif(c_val, exit_bb, &[], guard_bb, &[]);
                    }

                    // Emit guarded_preheader_bb: runs ONLY when iteration 1 is guaranteed to enter body_entry!
                    builder.switch_to_block(guard_bb);
                    let data_off = core::mem::size_of::<MirArrayHeader>() as i64;
                    let len_off = core::mem::offset_of!(MirArrayHeader, len) as i32;
                    let cap_off = core::mem::offset_of!(MirArrayHeader, cap) as i32;
                    for &arr_loc in &hspec.hoist_arrays {
                        let base_ptr =
                            load_local_i64(arr_loc, &mut builder, &local_vars, &stack_slots)?;
                        if !cur_validated.contains(&arr_loc.0) {
                            let is_null = builder.ins().icmp_imm_s(IntCC::Equal, base_ptr, 0);
                            emit_oob_branch_if_bad(is_null, &mut builder, peeled_bc_ctx);
                        }
                        let arr_len = builder
                            .ins()
                            .load(types::I64, MemFlagsData::trusted(), base_ptr, len_off);
                        let len_neg = builder.ins().icmp_imm_s(IntCC::SignedLessThan, arr_len, 0);
                        emit_oob_branch_if_bad(len_neg, &mut builder, peeled_bc_ctx);

                        let cap_raw = builder
                            .ins()
                            .load(types::I64, MemFlagsData::trusted(), base_ptr, cap_off);
                        let is_redir = builder.ins().icmp_imm_s(IntCC::SignedLessThan, cap_raw, 0);
                        let redir_bb = builder.create_block();
                        let flat_bb = builder.create_block();
                        let data_merge_bb = builder.create_block();
                        let data_param = builder.append_block_param(data_merge_bb, types::I64);
                        builder.set_cold_block(redir_bb);
                        builder.ins().brif(is_redir, redir_bb, &[], flat_bb, &[]);

                        builder.switch_to_block(redir_bb);
                        let redir_ptr = builder.ins().load(
                            types::I64,
                            MemFlagsData::trusted(),
                            base_ptr,
                            data_off as i32,
                        );
                        builder.ins().jump(data_merge_bb, &[BlockArg::Value(redir_ptr)]);

                        builder.switch_to_block(flat_bb);
                        let flat_ptr = builder.ins().iadd_imm_s(base_ptr, data_off);
                        builder.ins().jump(data_merge_bb, &[BlockArg::Value(flat_ptr)]);

                        builder.switch_to_block(data_merge_bb);
                        let hvars = hl_states[hl_idx].hoisted_arrays[&arr_loc.0];
                        builder.def_var(hvars.len_var, arr_len);
                        builder.def_var(hvars.data_ptr_var, data_param);
                    }
                    for &st_loc in &hspec.hoist_structs {
                        let base_ptr =
                            load_local_i64(st_loc, &mut builder, &local_vars, &stack_slots)?;
                        let is_null = builder.ins().icmp_imm_s(IntCC::Equal, base_ptr, 0);
                        emit_oob_branch_if_bad(is_null, &mut builder, peeled_bc_ctx);
                        let st_var = hl_states[hl_idx].hoisted_structs[&st_loc.0];
                        builder.def_var(st_var, base_ptr);
                    }
                    builder.ins().jump(clif_blocks[hspec.body_entry], &[]);

                    if let Some(ci) = cl_opt {
                        let rec = &mut cfg_records[ci];
                        for bb in builder.func.layout.blocks() {
                            if !blocks_before.contains(&bb)
                                && !rec.flush_clif_blocks.contains(&bb)
                                && !rec.post_flush_clif_blocks.contains(&bb)
                                && !rec.reload_clif_blocks.contains(&bb)
                            {
                                rec.loop_clif_blocks.insert(bb);
                            }
                        }
                    }
                    if !is_peeled_iq_hdr
                        && let Some(iqi) = iq_opt
                    {
                        let iq_rec = &mut iq_cfg_records[iqi];
                        for bb in builder.func.layout.blocks() {
                            if !blocks_before.contains(&bb)
                                && !iq_rec.reconcile_clif_blocks.contains(&bb)
                                && !iq_rec.post_flush_clif_blocks.contains(&bb)
                                && !iq_rec.reload_clif_blocks.contains(&bb)
                            {
                                iq_rec.loop_clif_blocks.insert(bb);
                            }
                        }
                    }
                }

                // Lower each basic block
                for (b_idx, mir_block) in mir_fn.blocks.iter().enumerate() {
                    let cl_opt = promo_plan.block_to_counter_loop.get(&b_idx).copied();
                    let iq_opt = promo_plan.block_to_inner_quantized.get(&b_idx).copied();
                    let blocks_before: hashbrown::HashSet<Block> = if cl_opt.is_some() || iq_opt.is_some() {
                        builder.func.layout.blocks().collect()
                    } else {
                        hashbrown::HashSet::new()
                    };

                    let mut active_hoisted_arrays: hashbrown::HashMap<usize, HoistedArrayVars> =
                        hashbrown::HashMap::new();
                    let mut active_hoisted_structs: hashbrown::HashMap<usize, Variable> =
                        hashbrown::HashMap::new();
                    for (hl_idx, hspec) in promo_plan.hoist_loops.iter().enumerate() {
                        if hspec.loop_blocks.contains(&b_idx) {
                            active_hoisted_arrays.extend(hl_states[hl_idx].hoisted_arrays.iter().map(|(&k, &v)| (k, v)));
                            for &(alias_loc, root_loc) in &hspec.hoist_array_aliases {
                                if let Some(&hvars) = hl_states[hl_idx].hoisted_arrays.get(&root_loc.0) {
                                    active_hoisted_arrays.insert(alias_loc.0, hvars);
                                }
                            }
                            active_hoisted_structs.extend(hl_states[hl_idx].hoisted_structs.iter().map(|(&k, &v)| (k, v)));
                        }
                    }
                    let active_emergency_bb = if let Some(iqi) = iq_opt {
                        iq_states[iqi].emergency_flush_bb
                    } else if let Some(ci) = cl_opt {
                        cl_states[ci].emergency_flush_bb
                    } else {
                        emergency_exit_bb
                    };
                    let mut cur_proven_in_bounds: Vec<(LocalId, LocalId)> = mir_fn
                        .proven_in_bounds
                        .get(&b_idx)
                        .cloned()
                        .unwrap_or_default();
                    if let Some(ind_pairs) = loop_bounds_analysis.proven_in_bounds.get(&b_idx) {
                        for &pair in ind_pairs {
                            if !cur_proven_in_bounds.contains(&pair) {
                                cur_proven_in_bounds.push(pair);
                            }
                        }
                    }
                    let active_promoted = if let Some(iqi) = iq_opt {
                        Some(iq_states[iqi].counters)
                    } else {
                        cl_opt.map(|ci| cl_states[ci].counters)
                    };

                    let clif_bb = clif_blocks[b_idx];
                    builder.switch_to_block(clif_bb);
                    let mut cur_validated = prov_plan.validated_in[b_idx].clone();

                    let should_safepoint = loop_headers.contains(&b_idx);
                    let mut skip_first_header_gas_check = false;

                    if should_safepoint {
                        let proven_entry_storage;
                        let locals_to_record = if let Some((_, live_locals)) = osr_info {
                            live_locals
                        } else {
                            match mir_fn.derive_proven_site_metadata_at_entry(b_idx) {
                                Ok(proven) => {
                                    proven_entry_storage = proven.into_iter().map(LocalId).collect::<Vec<_>>();
                                    &proven_entry_storage
                                }
                                Err(err) => {
                                    return Err(MirCraneliftBackendError::CodegenError(alloc::format!(
                                        "Cannot derive authoritative deopt metadata for loop header safepoint {}: {}",
                                        b_idx, err
                                    )));
                                }
                            }
                        };
                        let pending_vec_reduction = vec_hdr_map.get(&b_idx).and_then(|&vl_idx| {
                            let vl = &vec_loops[vl_idx];
                            vec_acc_vars[vl_idx]
                                .and_then(|(v0, v1)| vl.acc_local.map(|acc_id| (acc_id, v0, v1)))
                        });
                        let (sp_ptr_var, _, sp_slow_fn) =
                            safepoint_ctx.expect("loop header implies needs_safepoint");
                        let quantized_amt = if vec_hdr_map.get(&b_idx).is_none() {
                            iq_opt
                                .and_then(|iqi| {
                                    let iq_spec = &promo_plan.inner_quantized_loops[iqi];
                                    if iq_spec.hdr == b_idx {
                                        Some(iq_spec.header_gas_amt)
                                    } else {
                                        None
                                    }
                                })
                                .or_else(|| {
                                    cl_opt.and_then(|ci| {
                                        promo_plan.counter_loops[ci].quantized_header_gas_amt
                                    })
                                })
                        } else {
                            None
                        };
                        if let Some(q_amt) = quantized_amt {
                            let promoted_counters =
                                active_promoted.expect("quantized loop header implies promoted counters");
                            let unroll4_spec = if osr_info.is_none()
                                && promoted_counters.quantum.is_some()
                            {
                                detect_unroll4_inner_loop(
                                    b_idx,
                                    mir_fn,
                                    &loop_headers,
                                    &stack_slots,
                                    &active_hoisted_arrays,
                                    &local_read_counts,
                                    &bounds_excluded_locals,
                                    &loop_bounds_analysis,
                                    &prov_plan.validated_in,
                                )
                            } else {
                                None
                            };
                            if let Some(uspec) = unroll4_spec {
                                let (quantum_rem_var, _) = promoted_counters
                                    .quantum
                                    .expect("checked quantum is_some");
                                let unroll_body_bb = builder.create_block();
                                let scalar_hdr_bb = builder.create_block();
                                let unroll_bc_ctx = BoundsCheckCtx {
                                    status_set_fn,
                                    emergency_exit_bb: active_emergency_bb,
                                    hoisted_arrays: &active_hoisted_arrays,
                                    hoisted_structs: &active_hoisted_structs,
                                    proven_in_bounds: &[],
                                    const_int_locals: &const_int_locals,
                                    pow2_mod_defs: &empty_pow2_mod_defs,
                                    inline_status_ptr_var,
                                    closure_fn_ids: closure_fn_ids_ref,
                                };
                                let q0 = builder.use_var(quantum_rem_var);
                                const UNROLL_K: i64 = 8;
                                let q_ge_k = builder
                                    .ins()
                                    .icmp_imm_s(IntCC::SignedGreaterThanOrEqual, q0, UNROLL_K);
                                let k0 = builder.use_var(local_vars[uspec.idx_local.0]);
                                let bound_v = lower_operand(
                                    &uspec.bound_op,
                                    &mut builder,
                                    &local_vars,
                                    &stack_slots,
                                    &mir_fn.locals,
                                    &self.struct_layouts,
                                    unroll_bc_ctx,
                                )?;
                                let bound_ge_k = builder
                                    .ins()
                                    .icmp_imm_s(IntCC::SignedGreaterThanOrEqual, bound_v, UNROLL_K);
                                let bound_minus_k = builder.ins().iadd_imm_s(bound_v, -UNROLL_K);
                                // When bound_v >= UNROLL_K, `(k0 as u64) <= (bound_v - UNROLL_K) as u64`
                                // simultaneously enforces `0 <= k0 <= bound_v - UNROLL_K` in 1 compare.
                                let k_fits = builder.ins().icmp(
                                    IntCC::UnsignedLessThanOrEqual,
                                    k0,
                                    bound_minus_k,
                                );
                                let c1 = builder.ins().band(q_ge_k, bound_ge_k);
                                let can_unroll = builder.ins().band(c1, k_fits);
                                builder.ins().brif(
                                    can_unroll,
                                    unroll_body_bb,
                                    &[],
                                    scalar_hdr_bb,
                                    &[],
                                );

                                // 8x unrolled pure body fast path: deducts UNROLL_K from quantum_rem once
                                // (sound because q0 >= UNROLL_K guarantees slow_flags == 0 and
                                // gas_remaining >= UNROLL_K * q_amt with zero deopt/trap possible).
                                // When `vec_i64x2_kernel` is recognized, emits 4 x 128-bit I64X2
                                // vector loads + vector ops + horizontal fold (~24 CLIF insns);
                                // otherwise uses independent `k_step = k0 + step` and a 4-way
                                // balanced binary reduction tree.
                                builder.switch_to_block(unroll_body_bb);
                                let q_after = builder.ins().iadd_imm_s(q0, -UNROLL_K);
                                builder.def_var(quantum_rem_var, q_after);
                                let use_independent_k = !uspec.idx_inc_stmts.is_empty();
                                if let Some(vk) = &uspec.vec_i64x2_kernel {
                                    let acc_init = builder.use_var(local_vars[vk.acc_local.0]);
                                    let off_v = builder.use_var(local_vars[vk.offset_local.0]);
                                    let start_idx = builder.ins().iadd(off_v, k0);
                                    let byte_off = builder.ins().ishl_imm_s(start_idx, 3);
                                    let data_ptr = builder.use_var(
                                        active_hoisted_arrays[&vk.arr_local.0].data_ptr_var,
                                    );
                                    let base_ptr = builder.ins().iadd(data_ptr, byte_off);

                                    let k1 = builder.ins().iadd_imm_s(k0, 1);
                                    let vk_splat = builder.ins().splat(types::I64X2, k0);
                                    let vk0 = builder.ins().insertlane(vk_splat, k1, 1);
                                    let two_s = builder.ins().iconst(types::I64, 2);
                                    let vtwo = builder.ins().splat(types::I64X2, two_s);
                                    let vk1 = builder.ins().iadd(vk0, vtwo);
                                    let vk2 = builder.ins().iadd(vk1, vtwo);
                                    let vk3 = builder.ins().iadd(vk2, vtwo);

                                    // Array elements are 8-byte aligned (not guaranteed 16-byte aligned),
                                    // so use `notrap` without `aligned` for sound 128-bit loads.
                                    let mem_flags = MemFlagsData::new().with_notrap();
                                    let d0 = builder.ins().load(types::I64X2, mem_flags, base_ptr, 0);
                                    let d1 = builder.ins().load(types::I64X2, mem_flags, base_ptr, 16);
                                    let d2 = builder.ins().load(types::I64X2, mem_flags, base_ptr, 32);
                                    let d3 = builder.ins().load(types::I64X2, mem_flags, base_ptr, 48);

                                    let emit_vop = |b: &mut FunctionBuilder,
                                                    dv: ClifValue,
                                                    kv: ClifValue|
                                     -> ClifValue {
                                        let (lhs, rhs) = if vk.arr_on_lhs {
                                            (dv, kv)
                                        } else {
                                            (kv, dv)
                                        };
                                        match vk.op {
                                            MirBinaryOp::BitXor => b.ins().bxor(lhs, rhs),
                                            MirBinaryOp::Add => b.ins().iadd(lhs, rhs),
                                            MirBinaryOp::Sub => b.ins().isub(lhs, rhs),
                                            MirBinaryOp::BitAnd => b.ins().band(lhs, rhs),
                                            MirBinaryOp::BitOr => b.ins().bor(lhs, rhs),
                                            _ => unreachable!(),
                                        }
                                    };
                                    let r0 = emit_vop(&mut builder, d0, vk0);
                                    let r1 = emit_vop(&mut builder, d1, vk1);
                                    let r2 = emit_vop(&mut builder, d2, vk2);
                                    let r3 = emit_vop(&mut builder, d3, vk3);

                                    let p0 = builder.ins().iadd(r0, r1);
                                    let p1 = builder.ins().iadd(r2, r3);
                                    let vsum = builder.ins().iadd(p0, p1);
                                    let lane0 = builder.ins().extractlane(vsum, 0);
                                    let lane1 = builder.ins().extractlane(vsum, 1);
                                    let sum8 = builder.ins().iadd(lane0, lane1);
                                    let s_final = builder.ins().iadd(acc_init, sum8);
                                    builder.def_var(local_vars[vk.acc_local.0], s_final);
                                } else {
                                let acc_init = uspec
                                    .add_reduction
                                    .as_ref()
                                    .map(|(acc_loc, ..)| builder.use_var(local_vars[acc_loc.0]));
                                let mut red_deltas: Vec<ClifValue> =
                                    Vec::with_capacity(UNROLL_K as usize);
                                for step in 0..UNROLL_K {
                                    if use_independent_k {
                                        let k_step = if step == 0 {
                                            k0
                                        } else {
                                            builder.ins().iadd_imm_s(k0, step)
                                        };
                                        builder.def_var(local_vars[uspec.idx_local.0], k_step);
                                    }
                                    for &pb in &uspec.path_blocks {
                                        let pblk = &mir_fn.blocks[pb];
                                        let mut pb_validated = prov_plan.validated_in[pb].clone();
                                        let mut pb_proven: Vec<(LocalId, LocalId)> = mir_fn
                                            .proven_in_bounds
                                            .get(&pb)
                                            .cloned()
                                            .unwrap_or_default();
                                        if let Some(ind_pairs) =
                                            loop_bounds_analysis.proven_in_bounds.get(&pb)
                                        {
                                            for &pair in ind_pairs {
                                                if !pb_proven.contains(&pair) {
                                                    pb_proven.push(pair);
                                                }
                                            }
                                        }
                                        let pb_rf = mir_fn.detect_unsigned_range_checks_in_block(
                                            pb,
                                            &local_read_counts,
                                            &bounds_excluded_locals,
                                            &loop_bounds_analysis.non_neg_locals_in[pb],
                                        );
                                        let mut pb_folded_rc: hashbrown::HashSet<usize> =
                                            hashbrown::HashSet::new();
                                        let mut pb_folded_len_skip: hashbrown::HashSet<usize> =
                                            hashbrown::HashSet::new();
                                        for (&rc_stmt_idx, rc) in &pb_rf.fused_checks {
                                            let Some(arr_loc) = rc.arr_local else {
                                                continue;
                                            };
                                            let proven_at_stmt = loop_bounds_analysis
                                                .proven_in_bounds_at_stmt
                                                .get(&(pb, rc_stmt_idx))
                                                .is_some_and(|v| {
                                                    v.contains(&(arr_loc, rc.idx_local))
                                                });
                                            let proven_from_entry = if pb_proven
                                                .contains(&(arr_loc, rc.idx_local))
                                            {
                                                !pblk.statements[..rc_stmt_idx].iter().any(|s| {
                                                    match s {
                                                        Statement::Assign(Place::Local(d), _, _) => {
                                                            *d == arr_loc || *d == rc.idx_local
                                                        }
                                                        Statement::StorageDead(d, _) => {
                                                            *d == arr_loc || *d == rc.idx_local
                                                        }
                                                        _ => false,
                                                    }
                                                })
                                            } else {
                                                false
                                            };
                                            if proven_at_stmt || proven_from_entry {
                                                pb_folded_rc.insert(rc_stmt_idx);
                                                if let Some(s_len_idx) = rc.s_len
                                                    && (pb_validated.contains(&arr_loc.0)
                                                        || active_hoisted_arrays
                                                            .contains_key(&arr_loc.0))
                                                {
                                                    pb_folded_len_skip.insert(s_len_idx);
                                                }
                                            }
                                        }
                                        for (stmt_idx, stmt) in pblk.statements.iter().enumerate() {
                                            if pb_rf.skip_stmts.contains(&stmt_idx)
                                                || pb_folded_len_skip.contains(&stmt_idx)
                                                || (use_independent_k
                                                    && uspec.idx_inc_stmts.contains(&(pb, stmt_idx)))
                                            {
                                                continue;
                                            }
                                            if let Some((_, add_bb, add_si, ref delta_op, copy_pos)) =
                                                uspec.add_reduction
                                            {
                                                if pb == add_bb && stmt_idx == add_si {
                                                    let step_bc_ctx = BoundsCheckCtx {
                                                        status_set_fn,
                                                        emergency_exit_bb: active_emergency_bb,
                                                        hoisted_arrays: &active_hoisted_arrays,
                                                        hoisted_structs: &active_hoisted_structs,
                                                        proven_in_bounds: &pb_proven,
                                                        const_int_locals: &const_int_locals,
                                                        pow2_mod_defs: &empty_pow2_mod_defs,
                                                        inline_status_ptr_var,
                                                        closure_fn_ids: closure_fn_ids_ref,
                                                    };
                                                    let d_val = lower_operand(
                                                        delta_op,
                                                        &mut builder,
                                                        &local_vars,
                                                        &stack_slots,
                                                        &mir_fn.locals,
                                                        &self.struct_layouts,
                                                        step_bc_ctx,
                                                    )?;
                                                    let d_i64 = cast_clif_type(
                                                        &mut builder,
                                                        d_val,
                                                        types::I64,
                                                    );
                                                    red_deltas.push(d_i64);
                                                    continue;
                                                }
                                                if copy_pos == Some((pb, stmt_idx)) {
                                                    continue;
                                                }
                                            }
                                            if pb_folded_rc.contains(&stmt_idx) {
                                                if let Some(rc) = pb_rf.fused_checks.get(&stmt_idx)
                                                {
                                                    let dst_decl = &mir_fn.locals[rc.dst.0];
                                                    let target_ty = mir_to_clif_type(&dst_decl.ty)?;
                                                    let one = builder.ins().iconst(types::I64, 1);
                                                    let val =
                                                        cast_clif_type(&mut builder, one, target_ty);
                                                    store_local_val(
                                                        rc.dst,
                                                        val,
                                                        &mut builder,
                                                        &local_vars,
                                                        &stack_slots,
                                                    )?;
                                                    pb_proven.retain(|&(a, i)| {
                                                        a != rc.dst && i != rc.dst
                                                    });
                                                }
                                                continue;
                                            }
                                            if let Statement::Assign(Place::Local(dst), rvalue, _) =
                                                stmt
                                            {
                                                let stmt_proven_storage: Vec<(LocalId, LocalId)>;
                                                let stmt_proven_slice: &[(LocalId, LocalId)] =
                                                    if let Some(extra) = loop_bounds_analysis
                                                        .proven_in_bounds_at_stmt
                                                        .get(&(pb, stmt_idx))
                                                    {
                                                        let mut combined = pb_proven.clone();
                                                        for &pair in extra {
                                                            if !combined.contains(&pair) {
                                                                combined.push(pair);
                                                            }
                                                        }
                                                        stmt_proven_storage = combined;
                                                        &stmt_proven_storage
                                                    } else {
                                                        &pb_proven
                                                    };
                                                let step_bc_ctx = BoundsCheckCtx {
                                                    status_set_fn,
                                                    emergency_exit_bb: active_emergency_bb,
                                                    hoisted_arrays: &active_hoisted_arrays,
                                                    hoisted_structs: &active_hoisted_structs,
                                                    proven_in_bounds: stmt_proven_slice,
                                                    const_int_locals: &const_int_locals,
                                                    pow2_mod_defs: &empty_pow2_mod_defs,
                                                    inline_status_ptr_var,
                                                    closure_fn_ids: closure_fn_ids_ref,
                                                };
                                                let dst_decl = &mir_fn.locals[dst.0];
                                                let val = lower_rvalue(
                                                    rvalue,
                                                    Some(&dst_decl.ty),
                                                    None,
                                                    &mut pb_validated,
                                                    &mut builder,
                                                    &local_vars,
                                                    &stack_slots,
                                                    &mir_fn.locals,
                                                    &self.struct_layouts,
                                                    alloc_array_fn,
                                                    alloc_array_typed_fn,
                                                    alloc_tuple_typed_fn,
                                                    alloc_struct_named_fn,
                                                    alloc_cflat_struct_fn,
                                                    len_fn,
                                                    floordiv_fn,
                                                    pow_fn,
                                                    floordiv_float_fn,
                                                    pow_float_fn,
                                                    rem_float_fn,
                                                    string_helpers,
                                                    status_fn,
                                                    step_bc_ctx,
                                                    None,
                                                )?;
                                                let target_ty = mir_to_clif_type(&dst_decl.ty)?;
                                                let val =
                                                    cast_clif_type(&mut builder, val, target_ty);
                                                store_local_val(
                                                    *dst,
                                                    val,
                                                    &mut builder,
                                                    &local_vars,
                                                    &stack_slots,
                                                )?;
                                                pb_proven
                                                    .retain(|&(a, i)| a != *dst && i != *dst);
                                            }
                                        }
                                    }
                                }
                                if let Some((acc_loc, ..)) = uspec.add_reduction {
                                    let s0 = acc_init.expect("acc_init present");
                                    debug_assert_eq!(red_deltas.len(), UNROLL_K as usize);
                                    let p0 = builder.ins().iadd(red_deltas[0], red_deltas[1]);
                                    let p1 = builder.ins().iadd(red_deltas[2], red_deltas[3]);
                                    let p2 = builder.ins().iadd(red_deltas[4], red_deltas[5]);
                                    let p3 = builder.ins().iadd(red_deltas[6], red_deltas[7]);
                                    let q_a = builder.ins().iadd(p0, p1);
                                    let q_b = builder.ins().iadd(p2, p3);
                                    let sum8 = builder.ins().iadd(q_a, q_b);
                                    let s_final = builder.ins().iadd(s0, sum8);
                                    builder.def_var(local_vars[acc_loc.0], s_final);
                                }
                                }
                                let k_next = if use_independent_k {
                                    let kn = builder.ins().iadd_imm_s(k0, UNROLL_K);
                                    builder.def_var(local_vars[uspec.idx_local.0], kn);
                                    kn
                                } else {
                                    builder.use_var(local_vars[uspec.idx_local.0])
                                };
                                // Direct exit when loop terminates at `k_next >= bound` with `q_after >= 1`
                                // (deducts the exact 1 exit-iteration header check from `quantum_rem`).
                                let done_loop =
                                    builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, k_next, bound_v);
                                let q_has_exit =
                                    builder.ins().icmp_imm_s(IntCC::SignedGreaterThanOrEqual, q_after, 1);
                                let can_fast_exit = builder.ins().band(done_loop, q_has_exit);
                                let fast_exit_bb = builder.create_block();
                                builder.ins().brif(can_fast_exit, fast_exit_bb, &[], clif_bb, &[]);

                                builder.switch_to_block(fast_exit_bb);
                                let q_exit = builder.ins().iadd_imm_s(q_after, -1);
                                builder.def_var(quantum_rem_var, q_exit);
                                let cond_ty = mir_to_clif_type(&mir_fn.locals[uspec.cond_local.0].ty)?;
                                let false_v = builder.ins().iconst(cond_ty, 0);
                                store_local_val(
                                    uspec.cond_local,
                                    false_v,
                                    &mut builder,
                                    &local_vars,
                                    &stack_slots,
                                )?;
                                let loop_exit_target =
                                    resolve_target_bb(Some(b_idx), uspec.exit_block.0)?;
                                builder.ins().jump(loop_exit_target, &[]);

                                builder.switch_to_block(scalar_hdr_bb);
                            } else if osr_info.is_none()
                                && promoted_counters.quantum.is_some()
                                && let Some(vq_spec) = detect_vec_queue_inner_loop(
                                    b_idx,
                                    mir_fn,
                                    &loop_headers,
                                    &stack_slots,
                                    &const_int_locals,
                                    &inline_local_queues,
                                )
                            {
                                let (quantum_rem_var, _) = promoted_counters
                                    .quantum
                                    .expect("checked quantum is_some");
                                const VQ_UNROLL_K: i64 = 8;
                                let vq_body_bb = builder.create_block();
                                let scalar_hdr_bb = builder.create_block();
                                let vq_bc_ctx = BoundsCheckCtx {
                                    status_set_fn,
                                    emergency_exit_bb: active_emergency_bb,
                                    hoisted_arrays: &active_hoisted_arrays,
                                    hoisted_structs: &active_hoisted_structs,
                                    proven_in_bounds: &[],
                                    const_int_locals: &const_int_locals,
                                    pow2_mod_defs: &empty_pow2_mod_defs,
                                    inline_status_ptr_var,
                                    closure_fn_ids: closure_fn_ids_ref,
                                };
                                let q0 = builder.use_var(quantum_rem_var);
                                let q_ge_k = builder.ins().icmp_imm_s(
                                    IntCC::SignedGreaterThanOrEqual,
                                    q0,
                                    VQ_UNROLL_K,
                                );
                                let k0 = builder.use_var(local_vars[vq_spec.idx_local.0]);
                                let bound_v = lower_operand(
                                    &vq_spec.bound_op,
                                    &mut builder,
                                    &local_vars,
                                    &stack_slots,
                                    &mir_fn.locals,
                                    &self.struct_layouts,
                                    vq_bc_ctx,
                                )?;
                                let bound_ge_k = builder.ins().icmp_imm_s(
                                    IntCC::SignedGreaterThanOrEqual,
                                    bound_v,
                                    VQ_UNROLL_K,
                                );
                                let bound_minus_k = builder.ins().iadd_imm_s(bound_v, -VQ_UNROLL_K);
                                let k_fits = builder.ins().icmp(
                                    IntCC::UnsignedLessThanOrEqual,
                                    k0,
                                    bound_minus_k,
                                );
                                let c0 = builder.ins().band(q_ge_k, bound_ge_k);
                                let c1 = builder.ins().band(c0, k_fits);

                                match &vq_spec.kind {
                                    VecQueueLoopKind::OfferLinear {
                                        q_loc,
                                        offer_dst,
                                        pre_stmts,
                                        val_op,
                                    } => {
                                        let qv = &inline_queue_vars[q_loc];
                                        let cur_len = builder.use_var(qv.len_var);
                                        let tail = builder.use_var(qv.tail_var);
                                        let len_fits = builder.ins().icmp_imm_s(
                                            IntCC::SignedLessThanOrEqual,
                                            cur_len,
                                            qv.cap - VQ_UNROLL_K,
                                        );
                                        let tail_fits = builder.ins().icmp_imm_s(
                                            IntCC::SignedLessThanOrEqual,
                                            tail,
                                            qv.cap - VQ_UNROLL_K,
                                        );
                                        let q_fits = builder.ins().band(len_fits, tail_fits);
                                        let can_vec = builder.ins().band(c1, q_fits);
                                        builder.ins().brif(
                                            can_vec,
                                            vq_body_bb,
                                            &[],
                                            scalar_hdr_bb,
                                            &[],
                                        );

                                        builder.switch_to_block(vq_body_bb);
                                        let mut dummy_val = hashbrown::HashSet::new();
                                        for &(pb, si) in pre_stmts {
                                            if let Statement::Assign(Place::Local(dst), rv, _) =
                                                &mir_fn.blocks[pb].statements[si]
                                            {
                                                let dst_decl = &mir_fn.locals[dst.0];
                                                let val = lower_rvalue(
                                                    rv,
                                                    Some(&dst_decl.ty),
                                                    None,
                                                    &mut dummy_val,
                                                    &mut builder,
                                                    &local_vars,
                                                    &stack_slots,
                                                    &mir_fn.locals,
                                                    &self.struct_layouts,
                                                    alloc_array_fn,
                                                    alloc_array_typed_fn,
                                                    alloc_tuple_typed_fn,
                                                    alloc_struct_named_fn,
                                                    alloc_cflat_struct_fn,
                                                    len_fn,
                                                    floordiv_fn,
                                                    pow_fn,
                                                    floordiv_float_fn,
                                                    pow_float_fn,
                                                    rem_float_fn,
                                                    string_helpers,
                                                    status_fn,
                                                    vq_bc_ctx,
                                                    None,
                                                )?;
                                                let target_ty = mir_to_clif_type(&dst_decl.ty)?;
                                                let val = cast_clif_type(&mut builder, val, target_ty);
                                                store_local_val(
                                                    *dst,
                                                    val,
                                                    &mut builder,
                                                    &local_vars,
                                                    &stack_slots,
                                                )?;
                                            }
                                        }
                                        let v_start = lower_operand(
                                            val_op,
                                            &mut builder,
                                            &local_vars,
                                            &stack_slots,
                                            &mir_fn.locals,
                                            &self.struct_layouts,
                                            vq_bc_ctx,
                                        )?;
                                        let v_start_i64 = cast_clif_type(&mut builder, v_start, types::I64);
                                        let v_start_p1 = builder.ins().iadd_imm_s(v_start_i64, 1);
                                        let v_splat = builder.ins().splat(types::I64X2, v_start_i64);
                                        let v0 = builder.ins().insertlane(v_splat, v_start_p1, 1);
                                        let two_s = builder.ins().iconst(types::I64, 2);
                                        let vtwo = builder.ins().splat(types::I64X2, two_s);
                                        let v1 = builder.ins().iadd(v0, vtwo);
                                        let v2 = builder.ins().iadd(v1, vtwo);
                                        let v3 = builder.ins().iadd(v2, vtwo);

                                        let buf_ptr = builder.use_var(qv.buf_var);
                                        let byte_off = builder.ins().ishl_imm_s(tail, 3);
                                        let base_ptr = builder.ins().iadd(buf_ptr, byte_off);
                                        let elem_off = if qv.stack_slot.is_some() { 0 } else { 16 };
                                        let mem_flags = MemFlagsData::new().with_notrap();
                                        builder.ins().store(mem_flags, v0, base_ptr, elem_off);
                                        builder.ins().store(mem_flags, v1, base_ptr, elem_off + 16);
                                        builder.ins().store(mem_flags, v2, base_ptr, elem_off + 32);
                                        builder.ins().store(mem_flags, v3, base_ptr, elem_off + 48);

                                        let tail_p8 = builder.ins().iadd_imm_s(tail, VQ_UNROLL_K);
                                        let next_tail = if (qv.cap as u64).is_power_of_two() {
                                            let mask = builder.ins().iconst(types::I64, qv.cap - 1);
                                            builder.ins().band(tail_p8, mask)
                                        } else {
                                            let zero = builder.ins().iconst(types::I64, 0);
                                            let eq_cap = builder.ins().icmp_imm_s(
                                                IntCC::Equal,
                                                tail_p8,
                                                qv.cap,
                                            );
                                            builder.ins().select(eq_cap, zero, tail_p8)
                                        };
                                        let next_len = builder.ins().iadd_imm_s(cur_len, VQ_UNROLL_K);
                                        let q_after = builder.ins().iadd_imm_s(q0, -VQ_UNROLL_K);
                                        let k_next = builder.ins().iadd_imm_s(k0, VQ_UNROLL_K);
                                        builder.def_var(qv.tail_var, next_tail);
                                        builder.def_var(qv.len_var, next_len);
                                        builder.def_var(quantum_rem_var, q_after);
                                        builder.def_var(local_vars[vq_spec.idx_local.0], k_next);
                                        if let Some(odst) = offer_dst {
                                            let target_ty = mir_to_clif_type(&mir_fn.locals[odst.0].ty)?;
                                            let one = builder.ins().iconst(types::I64, 1);
                                            let one_cast = cast_clif_type(&mut builder, one, target_ty);
                                            store_local_val(
                                                *odst,
                                                one_cast,
                                                &mut builder,
                                                &local_vars,
                                                &stack_slots,
                                            )?;
                                        }

                                        let done_loop = builder.ins().icmp(
                                            IntCC::SignedGreaterThanOrEqual,
                                            k_next,
                                            bound_v,
                                        );
                                        let q_has_exit = builder.ins().icmp_imm_s(
                                            IntCC::SignedGreaterThanOrEqual,
                                            q_after,
                                            1,
                                        );
                                        let can_fast_exit = builder.ins().band(done_loop, q_has_exit);
                                        let fast_exit_bb = builder.create_block();
                                        builder.ins().brif(
                                            can_fast_exit,
                                            fast_exit_bb,
                                            &[],
                                            clif_bb,
                                            &[],
                                        );

                                        builder.switch_to_block(fast_exit_bb);
                                        let q_exit = builder.ins().iadd_imm_s(q_after, -1);
                                        builder.def_var(quantum_rem_var, q_exit);
                                        let cond_ty =
                                            mir_to_clif_type(&mir_fn.locals[vq_spec.cond_local.0].ty)?;
                                        let false_v = builder.ins().iconst(cond_ty, 0);
                                        store_local_val(
                                            vq_spec.cond_local,
                                            false_v,
                                            &mut builder,
                                            &local_vars,
                                            &stack_slots,
                                        )?;
                                        let loop_exit_target =
                                            resolve_target_bb(Some(b_idx), vq_spec.exit_block.0)?;
                                        builder.ins().jump(loop_exit_target, &[]);
                                    }
                                    VecQueueLoopKind::TakeAddMod {
                                        q_loc,
                                        acc_local,
                                        mod_const,
                                    } => {
                                        let qv = &inline_queue_vars[q_loc];
                                        let cur_len = builder.use_var(qv.len_var);
                                        let head = builder.use_var(qv.head_var);
                                        let len_fits = builder.ins().icmp_imm_s(
                                            IntCC::SignedGreaterThanOrEqual,
                                            cur_len,
                                            VQ_UNROLL_K,
                                        );
                                        let head_fits = builder.ins().icmp_imm_s(
                                            IntCC::SignedLessThanOrEqual,
                                            head,
                                            qv.cap - VQ_UNROLL_K,
                                        );
                                        let q_fits = builder.ins().band(len_fits, head_fits);
                                        let can_vec = builder.ins().band(c1, q_fits);
                                        builder.ins().brif(
                                            can_vec,
                                            vq_body_bb,
                                            &[],
                                            scalar_hdr_bb,
                                            &[],
                                        );

                                        builder.switch_to_block(vq_body_bb);
                                        let acc0 = builder.use_var(local_vars[acc_local.0]);
                                        let buf_ptr = builder.use_var(qv.buf_var);
                                        let byte_off = builder.ins().ishl_imm_s(head, 3);
                                        let base_ptr = builder.ins().iadd(buf_ptr, byte_off);
                                        let elem_off = if qv.stack_slot.is_some() { 0 } else { 16 };
                                        let mem_flags = MemFlagsData::new().with_notrap();
                                        let d0 = builder.ins().load(types::I64X2, mem_flags, base_ptr, elem_off);
                                        let d1 = builder.ins().load(types::I64X2, mem_flags, base_ptr, elem_off + 16);
                                        let d2 = builder.ins().load(types::I64X2, mem_flags, base_ptr, elem_off + 32);
                                        let d3 = builder.ins().load(types::I64X2, mem_flags, base_ptr, elem_off + 48);

                                        // Exact non-negative & overflow-free proof:
                                        // `(acc0 | d0..d7) < 2^58` iff `0 <= acc0, d0..d7 < 2^58`,
                                        // so `acc0 + sum(d0..d7) < 9 * 2^58 < i64::MAX` (zero wrap),
                                        // and `< mod_const` guarantees all 8 prefix sums stay in `[0, mod_const)`.
                                        let o0 = builder.ins().bor(d0, d1);
                                        let o1 = builder.ins().bor(d2, d3);
                                        let vor = builder.ins().bor(o0, o1);
                                        let or0 = builder.ins().extractlane(vor, 0);
                                        let or1 = builder.ins().extractlane(vor, 1);
                                        let or01 = builder.ins().bor(or0, or1);
                                        let or_all = builder.ins().bor(or01, acc0);
                                        let bound58 = builder.ins().iconst(types::I64, 1i64 << 58);
                                        let all_small =
                                            builder.ins().icmp(IntCC::UnsignedLessThan, or_all, bound58);

                                        let s0 = builder.ins().iadd(d0, d1);
                                        let s1 = builder.ins().iadd(d2, d3);
                                        let vsum = builder.ins().iadd(s0, s1);
                                        let l0 = builder.ins().extractlane(vsum, 0);
                                        let l1 = builder.ins().extractlane(vsum, 1);
                                        let sum8 = builder.ins().iadd(l0, l1);
                                        let new_acc = builder.ins().iadd(acc0, sum8);
                                        let c_mod_val = builder.ins().iconst(types::I64, *mod_const);
                                        let in_mod = builder.ins().icmp(
                                            IntCC::UnsignedLessThan,
                                            new_acc,
                                            c_mod_val,
                                        );
                                        let fast_ok = builder.ins().band(all_small, in_mod);
                                        let commit_bb = builder.create_block();
                                        builder.ins().brif(
                                            fast_ok,
                                            commit_bb,
                                            &[],
                                            scalar_hdr_bb,
                                            &[],
                                        );

                                        builder.switch_to_block(commit_bb);
                                        let head_p8 = builder.ins().iadd_imm_s(head, VQ_UNROLL_K);
                                        let next_head = if (qv.cap as u64).is_power_of_two() {
                                            let mask = builder.ins().iconst(types::I64, qv.cap - 1);
                                            builder.ins().band(head_p8, mask)
                                        } else {
                                            let zero = builder.ins().iconst(types::I64, 0);
                                            let eq_cap = builder.ins().icmp_imm_s(
                                                IntCC::Equal,
                                                head_p8,
                                                qv.cap,
                                            );
                                            builder.ins().select(eq_cap, zero, head_p8)
                                        };
                                        let next_len = builder.ins().iadd_imm_s(cur_len, -VQ_UNROLL_K);
                                        let q_after = builder.ins().iadd_imm_s(q0, -VQ_UNROLL_K);
                                        let k_next = builder.ins().iadd_imm_s(k0, VQ_UNROLL_K);
                                        builder.def_var(local_vars[acc_local.0], new_acc);
                                        builder.def_var(qv.head_var, next_head);
                                        builder.def_var(qv.len_var, next_len);
                                        builder.def_var(quantum_rem_var, q_after);
                                        builder.def_var(local_vars[vq_spec.idx_local.0], k_next);

                                        let done_loop = builder.ins().icmp(
                                            IntCC::SignedGreaterThanOrEqual,
                                            k_next,
                                            bound_v,
                                        );
                                        let q_has_exit = builder.ins().icmp_imm_s(
                                            IntCC::SignedGreaterThanOrEqual,
                                            q_after,
                                            1,
                                        );
                                        let can_fast_exit = builder.ins().band(done_loop, q_has_exit);
                                        let fast_exit_bb = builder.create_block();
                                        builder.ins().brif(
                                            can_fast_exit,
                                            fast_exit_bb,
                                            &[],
                                            clif_bb,
                                            &[],
                                        );

                                        builder.switch_to_block(fast_exit_bb);
                                        let q_exit = builder.ins().iadd_imm_s(q_after, -1);
                                        builder.def_var(quantum_rem_var, q_exit);
                                        let cond_ty =
                                            mir_to_clif_type(&mir_fn.locals[vq_spec.cond_local.0].ty)?;
                                        let false_v = builder.ins().iconst(cond_ty, 0);
                                        store_local_val(
                                            vq_spec.cond_local,
                                            false_v,
                                            &mut builder,
                                            &local_vars,
                                            &stack_slots,
                                        )?;
                                        let loop_exit_target =
                                            resolve_target_bb(Some(b_idx), vq_spec.exit_block.0)?;
                                        builder.ins().jump(loop_exit_target, &[]);
                                    }
                                }
                                builder.switch_to_block(scalar_hdr_bb);
                            }

                            if iq_opt.is_none()
                                && let Some(ci) = cl_opt
                                && let Some(hub) = shared_q_hubs[ci].as_mut()
                            {
                                let (quantum_rem_var, _) = promoted_counters
                                    .quantum
                                    .expect("quantized loop header requires quantum_rem_var");
                                let q0 = builder.use_var(quantum_rem_var);
                                let q1 = builder.ins().iadd_imm_s(q0, -1);
                                let needs_refill =
                                    builder.ins().icmp_imm_s(IntCC::SignedLessThan, q1, 0);
                                builder.def_var(quantum_rem_var, q1);
                                let cont_bb = builder.create_block();
                                let site_id = hub.sites.len() as i64;
                                let site_id_val = builder.ins().iconst(types::I64, site_id);
                                builder.ins().brif(
                                    needs_refill,
                                    hub.check_reload_bb,
                                    &[BlockArg::Value(site_id_val)],
                                    cont_bb,
                                    &[],
                                );
                                hub.sites.push((
                                    site_id,
                                    b_idx,
                                    locals_to_record.to_vec(),
                                    cont_bb,
                                ));
                                builder.switch_to_block(cont_bb);
                            } else if !inline_local_queues.is_empty() {
                                let refill_slow_fn = Self::declare_ext_fn(
                                    &mut self.module,
                                    &mut builder.func,
                                    &mut ext_fn_cache,
                                    "lust_mir_quantized_refill_slow",
                                    &[types::I64, types::I64],
                                    Some(types::I64),
                                )?;
                                emit_quantized_loop_header_check_compact(
                                    &mut builder,
                                    refill_slow_fn,
                                    init_deopt_fn,
                                    record_deopt_local_fn,
                                    b_idx,
                                    q_amt,
                                    locals_to_record,
                                    &local_vars,
                                    &stack_slots,
                                    &mir_fn.locals,
                                    emergency_exit_bb,
                                    promoted_counters,
                                    cl_opt.map(|ci| &mut cfg_records[ci]),
                                    iq_opt.map(|iqi| &mut iq_cfg_records[iqi]),
                                );
                            } else {
                                let (_, _, gas_slow_fn) =
                                    gas_ctx.expect("quantized loop header implies gas_ctx");
                                emit_quantized_loop_header_check(
                                    &mut builder,
                                    sp_ptr_var,
                                    sp_slow_fn,
                                    gas_slow_fn,
                                    init_deopt_fn,
                                    record_deopt_local_fn,
                                    b_idx,
                                    q_amt,
                                    locals_to_record,
                                    &local_vars,
                                    &stack_slots,
                                    &mir_fn.locals,
                                    emergency_exit_bb,
                                    promoted_counters,
                                    cl_opt.map(|ci| &mut cfg_records[ci]),
                                    iq_opt.map(|iqi| &mut iq_cfg_records[iqi]),
                                );
                            }
                            skip_first_header_gas_check = true;
                        } else {
                            emit_safepoint_check(
                                &mut builder,
                                sp_ptr_var,
                                sp_slow_fn,
                                init_deopt_fn,
                                record_deopt_local_fn,
                                b_idx,
                                locals_to_record,
                                &local_vars,
                                &stack_slots,
                                &mir_fn.locals,
                                emergency_exit_bb,
                                pending_vec_reduction,
                                active_promoted,
                                cl_opt.map(|ci| &mut cfg_records[ci]),
                            );
                        }
                    }

                    // Native Cranelift hardware SIMD lowering for recognized vectorized loop header
                    if let Some(&vl_idx) = vec_hdr_map.get(&b_idx) {
                        let vl = &vec_loops[vl_idx];
                        if let Some(amt) = vl.gas_check {
                            let (gas_ptr_var, _, gas_slow_fn) =
                                gas_ctx.expect("SIMD vl.gas_check implies needs_gas_fast");
                            emit_inline_gas_check(
                                &mut builder,
                                gas_ptr_var,
                                gas_slow_fn,
                                amt,
                                emergency_exit_bb,
                                active_promoted,
                                cl_opt.map(|ci| &mut cfg_records[ci]),
                            );
                        }
                        let i_val = load_local_i64(vl.idx_local, &mut builder, &local_vars, &stack_slots)?;
                        let end_val = load_local_i64(vl.end_local, &mut builder, &local_vars, &stack_slots)?;
                        let cond = builder.ins().icmp(IntCC::SignedLessThan, i_val, end_val);
                        if let Some((vacc0, vacc1)) = vec_acc_vars[vl_idx] {
                            let acc_loc = vl.acc_local.ok_or_else(|| {
                                MirCraneliftBackendError::CodegenError("Missing acc_local in SIMD reduction".into())
                            })?;
                            let reduce_exit_bb = builder.create_block();
                            builder.ins().brif(cond, clif_blocks[vl.body_block.0], &[], reduce_exit_bb, &[]);
                            builder.switch_to_block(reduce_exit_bb);
                            let lane_sum = fold_vec_acc_lanes(&mut builder, vacc0, vacc1);
                            let cur_acc = load_local_i64(acc_loc, &mut builder, &local_vars, &stack_slots)?;
                            let new_acc = builder.ins().iadd(cur_acc, lane_sum);
                            store_local_val(acc_loc, new_acc, &mut builder, &local_vars, &stack_slots)?;
                            let tail_target = resolve_target_bb(Some(b_idx), vl.tail_block.0)?;
                            builder.ins().jump(tail_target, &[]);
                        } else {
                            let body_target = resolve_target_bb(Some(b_idx), vl.body_block.0)?;
                            let tail_target = resolve_target_bb(Some(b_idx), vl.tail_block.0)?;
                            builder.ins().brif(cond, body_target, &[], tail_target, &[]);
                        }
                        if let Some(ci) = cl_opt {
                            let rec = &mut cfg_records[ci];
                            for bb in builder.func.layout.blocks() {
                                if !blocks_before.contains(&bb)
                                    && !rec.flush_clif_blocks.contains(&bb)
                                    && !rec.post_flush_clif_blocks.contains(&bb)
                                    && !rec.reload_clif_blocks.contains(&bb)
                                {
                                    rec.loop_clif_blocks.insert(bb);
                                }
                            }
                        }
                        continue;
                    }

                    // Native Cranelift hardware SIMD lowering for recognized 4-lane vector body
                    if let Some(&vl_idx) = vec_body_map.get(&b_idx) {
                        let vl = &vec_loops[vl_idx];
                        let i_val = load_local_i64(vl.idx_local, &mut builder, &local_vars, &stack_slots)?;
                        let eight = builder.ins().iconst(types::I64, 8);
                        let byte_idx = builder.ins().imul(i_val, eight);
                        let elem_off = builder.ins().iadd_imm_s(byte_idx, 16);

                        let a_ptr = load_local_i64(vl.a_local, &mut builder, &local_vars, &stack_slots)?;
                        let cur_a = builder.ins().iadd(a_ptr, elem_off);
                        let va0 = builder.ins().load(types::I64X2, MemFlagsData::new(), cur_a, 0);
                        let va1 = builder.ins().load(types::I64X2, MemFlagsData::new(), cur_a, 16);

                        match vl.kind {
                            crate::mir::opt::simd::VectorLoopKernelKind::Add => {
                                let b_loc = vl.b_local.ok_or_else(|| {
                                    MirCraneliftBackendError::CodegenError("Missing b_local in SIMD Add".into())
                                })?;
                                let out_loc = vl.out_local.ok_or_else(|| {
                                    MirCraneliftBackendError::CodegenError("Missing out_local in SIMD Add".into())
                                })?;
                                let b_ptr = load_local_i64(b_loc, &mut builder, &local_vars, &stack_slots)?;
                                let out_ptr = load_local_i64(out_loc, &mut builder, &local_vars, &stack_slots)?;
                                let cur_b = builder.ins().iadd(b_ptr, elem_off);
                                let cur_out = builder.ins().iadd(out_ptr, elem_off);
                                let vb0 = builder.ins().load(types::I64X2, MemFlagsData::new(), cur_b, 0);
                                let vb1 = builder.ins().load(types::I64X2, MemFlagsData::new(), cur_b, 16);
                                let vr0 = builder.ins().iadd(va0, vb0);
                                let vr1 = builder.ins().iadd(va1, vb1);
                                builder.ins().store(MemFlagsData::new(), vr0, cur_out, 0);
                                builder.ins().store(MemFlagsData::new(), vr1, cur_out, 16);
                            }
                            crate::mir::opt::simd::VectorLoopKernelKind::Mul => {
                                let b_loc = vl.b_local.ok_or_else(|| {
                                    MirCraneliftBackendError::CodegenError("Missing b_local in SIMD Mul".into())
                                })?;
                                let out_loc = vl.out_local.ok_or_else(|| {
                                    MirCraneliftBackendError::CodegenError("Missing out_local in SIMD Mul".into())
                                })?;
                                let b_ptr = load_local_i64(b_loc, &mut builder, &local_vars, &stack_slots)?;
                                let out_ptr = load_local_i64(out_loc, &mut builder, &local_vars, &stack_slots)?;
                                let cur_b = builder.ins().iadd(b_ptr, elem_off);
                                let cur_out = builder.ins().iadd(out_ptr, elem_off);
                                let vb0 = builder.ins().load(types::I64X2, MemFlagsData::new(), cur_b, 0);
                                let vb1 = builder.ins().load(types::I64X2, MemFlagsData::new(), cur_b, 16);
                                let vr0 = emit_vec_mul_i64x2(&mut builder, va0, vb0);
                                let vr1 = emit_vec_mul_i64x2(&mut builder, va1, vb1);
                                builder.ins().store(MemFlagsData::new(), vr0, cur_out, 0);
                                builder.ins().store(MemFlagsData::new(), vr1, cur_out, 16);
                            }
                            crate::mir::opt::simd::VectorLoopKernelKind::Sum => {
                                let (vacc0, vacc1) = vec_acc_vars[vl_idx].ok_or_else(|| {
                                    MirCraneliftBackendError::CodegenError("Missing vacc in SIMD Sum".into())
                                })?;
                                let cur0 = builder.use_var(vacc0);
                                let cur1 = builder.use_var(vacc1);
                                let next0 = builder.ins().iadd(cur0, va0);
                                let next1 = builder.ins().iadd(cur1, va1);
                                builder.def_var(vacc0, next0);
                                builder.def_var(vacc1, next1);
                            }
                            crate::mir::opt::simd::VectorLoopKernelKind::Dot => {
                                let b_loc = vl.b_local.ok_or_else(|| {
                                    MirCraneliftBackendError::CodegenError("Missing b_local in SIMD Dot".into())
                                })?;
                                let (vacc0, vacc1) = vec_acc_vars[vl_idx].ok_or_else(|| {
                                    MirCraneliftBackendError::CodegenError("Missing vacc in SIMD Dot".into())
                                })?;
                                let b_ptr = load_local_i64(b_loc, &mut builder, &local_vars, &stack_slots)?;
                                let cur_b = builder.ins().iadd(b_ptr, elem_off);
                                let vb0 = builder.ins().load(types::I64X2, MemFlagsData::new(), cur_b, 0);
                                let vb1 = builder.ins().load(types::I64X2, MemFlagsData::new(), cur_b, 16);
                                let vp0 = emit_vec_mul_i64x2(&mut builder, va0, vb0);
                                let vp1 = emit_vec_mul_i64x2(&mut builder, va1, vb1);
                                let cur0 = builder.use_var(vacc0);
                                let cur1 = builder.use_var(vacc1);
                                let next0 = builder.ins().iadd(cur0, vp0);
                                let next1 = builder.ins().iadd(cur1, vp1);
                                builder.def_var(vacc0, next0);
                                builder.def_var(vacc1, next1);
                            }
                        }

                        let next_i = builder.ins().iadd_imm_s(i_val, 4);
                        store_local_val(vl.idx_local, next_i, &mut builder, &local_vars, &stack_slots)?;
                        builder.ins().jump(clif_blocks[vl.hdr_block.0], &[]);
                        if let Some(ci) = cl_opt {
                            let rec = &mut cfg_records[ci];
                            for bb in builder.func.layout.blocks() {
                                if !blocks_before.contains(&bb)
                                    && !rec.flush_clif_blocks.contains(&bb)
                                    && !rec.post_flush_clif_blocks.contains(&bb)
                                    && !rec.reload_clif_blocks.contains(&bb)
                                {
                                    rec.loop_clif_blocks.insert(bb);
                                }
                            }
                        }
                        continue;
                    }

                    let mut cur_pow2_mod_defs: hashbrown::HashMap<usize, (ClifValue, i64)> =
                        hashbrown::HashMap::new();
                    let block_range_fusion = mir_fn.detect_unsigned_range_checks_in_block(
                        b_idx,
                        &local_read_counts,
                        &bounds_excluded_locals,
                        &loop_bounds_analysis.non_neg_locals_in[b_idx],
                    );
                    // NG-W15-6: Pre-identify proven-in-bounds `UnsignedRangeCheck`s so both their
                    // preceding `Len(arr)` statement and their `icmp ult` + `brif` can be folded.
                    let mut folded_range_checks: hashbrown::HashSet<usize> =
                        hashbrown::HashSet::new();
                    let mut folded_len_skip_stmts: hashbrown::HashSet<usize> =
                        hashbrown::HashSet::new();
                    let mut proven_true_cond_locals: hashbrown::HashSet<usize> =
                        hashbrown::HashSet::new();
                    for (&rc_stmt_idx, rc) in &block_range_fusion.fused_checks {
                        let Some(arr_loc) = rc.arr_local else {
                            continue;
                        };
                        let proven_at_stmt = loop_bounds_analysis
                            .proven_in_bounds_at_stmt
                            .get(&(b_idx, rc_stmt_idx))
                            .is_some_and(|v| v.contains(&(arr_loc, rc.idx_local)));
                        let proven_from_entry = if cur_proven_in_bounds
                            .contains(&(arr_loc, rc.idx_local))
                        {
                            !mir_block.statements[..rc_stmt_idx].iter().any(|s| match s {
                                Statement::Assign(Place::Local(d), _, _) => {
                                    *d == arr_loc || *d == rc.idx_local
                                }
                                Statement::StorageDead(d, _) => {
                                    *d == arr_loc || *d == rc.idx_local
                                }
                                _ => false,
                            })
                        } else {
                            false
                        };
                        if proven_at_stmt || proven_from_entry {
                            folded_range_checks.insert(rc_stmt_idx);
                            if let Some(s_len_idx) = rc.s_len
                                && (cur_validated.contains(&arr_loc.0)
                                    || active_hoisted_arrays.contains_key(&arr_loc.0))
                            {
                                folded_len_skip_stmts.insert(s_len_idx);
                            }
                        }
                    }
                    for (stmt_idx, stmt) in mir_block.statements.iter().enumerate() {
                        if stmt_idx == 0 && skip_first_header_gas_check {
                            continue;
                        }
                        if block_range_fusion.skip_stmts.contains(&stmt_idx)
                            || folded_len_skip_stmts.contains(&stmt_idx)
                        {
                            continue;
                        }
                        let stmt_proven_storage: Vec<(LocalId, LocalId)>;
                        let stmt_proven_slice: &[(LocalId, LocalId)] =
                            if let Some(extra) =
                                loop_bounds_analysis.proven_in_bounds_at_stmt.get(&(b_idx, stmt_idx))
                            {
                                let mut combined = cur_proven_in_bounds.clone();
                                for &pair in extra {
                                    if !combined.contains(&pair) {
                                        combined.push(pair);
                                    }
                                }
                                stmt_proven_storage = combined;
                                &stmt_proven_storage
                            } else {
                                &cur_proven_in_bounds
                            };
                        let stmt_bc_ctx = BoundsCheckCtx {
                            status_set_fn,
                            emergency_exit_bb: active_emergency_bb,
                            hoisted_arrays: &active_hoisted_arrays,
                            hoisted_structs: &active_hoisted_structs,
                            proven_in_bounds: stmt_proven_slice,
                            const_int_locals: &const_int_locals,
                            pow2_mod_defs: &cur_pow2_mod_defs,
                            inline_status_ptr_var,
                            closure_fn_ids: closure_fn_ids_ref,
                        };
                        if let Some(rc) = block_range_fusion.fused_checks.get(&stmt_idx) {
                            let dst_decl = mir_fn.locals.get(rc.dst.0).ok_or_else(|| {
                                MirCraneliftBackendError::CodegenError(format!(
                                    "Destination local {:?} out of bounds",
                                    rc.dst
                                ))
                            })?;
                            let target_ty = mir_to_clif_type(&dst_decl.ty)?;
                            if folded_range_checks.contains(&stmt_idx) {
                                let one = builder.ins().iconst(types::I64, 1);
                                let val = cast_clif_type(&mut builder, one, target_ty);
                                store_local_val(rc.dst, val, &mut builder, &local_vars, &stack_slots)?;
                                proven_true_cond_locals.insert(rc.dst.0);
                            } else {
                                let idx_val = load_local_i64(
                                    rc.idx_local,
                                    &mut builder,
                                    &local_vars,
                                    &stack_slots,
                                )?;
                                let mut len_val = lower_operand(
                                    &rc.len_op,
                                    &mut builder,
                                    &local_vars,
                                    &stack_slots,
                                    &mir_fn.locals,
                                    &self.struct_layouts,
                                    stmt_bc_ctx,
                                )?;
                                if builder.func.dfg.value_type(len_val) != types::I64 {
                                    len_val = cast_clif_type(&mut builder, len_val, types::I64);
                                }
                                let in_bounds = builder
                                    .ins()
                                    .icmp(IntCC::UnsignedLessThan, idx_val, len_val);
                                let val = cast_clif_type(&mut builder, in_bounds, target_ty);
                                store_local_val(rc.dst, val, &mut builder, &local_vars, &stack_slots)?;
                                proven_true_cond_locals.remove(&rc.dst.0);
                            }
                            cur_pow2_mod_defs.remove(&rc.dst.0);
                            cur_proven_in_bounds
                                .retain(|&(arr_l, idx_l)| arr_l != rc.dst && idx_l != rc.dst);
                            cur_validated.remove(&rc.dst.0);
                            continue;
                        }
                        match stmt {
                            Statement::Assign(Place::Local(dst), rvalue, _) => {
                                let dst_decl = mir_fn.locals.get(dst.0).ok_or_else(|| {
                                    MirCraneliftBackendError::CodegenError(format!(
                                        "Destination local {dst:?} out of bounds"
                                    ))
                                })?;
                                let inplace_concat_dst =
                                    if unaliased_string_locals.contains(&dst.0) {
                                        Some(*dst)
                                    } else {
                                        None
                                    };
                                let mut pow2_mod_record: Option<(ClifValue, i64)> = None;
                                let val = lower_rvalue(
                                    rvalue,
                                    Some(&dst_decl.ty),
                                    inplace_concat_dst,
                                    &mut cur_validated,
                                    &mut builder,
                                    &local_vars,
                                    &stack_slots,
                                    &mir_fn.locals,
                                    &self.struct_layouts,
                                    alloc_array_fn,
                                    alloc_array_typed_fn,
                                    alloc_tuple_typed_fn,
                                    alloc_struct_named_fn,
                                    alloc_cflat_struct_fn,
                                    len_fn,
                                    floordiv_fn,
                                    pow_fn,
                                    floordiv_float_fn,
                                    pow_float_fn,
                                    rem_float_fn,
                                    string_helpers,
                                    status_fn,
                                    stmt_bc_ctx,
                                    Some(&mut pow2_mod_record),
                                )?;
                                let target_ty = mir_to_clif_type(&dst_decl.ty)?;
                                let val = cast_clif_type(&mut builder, val, target_ty);
                                store_local_val(*dst, val, &mut builder, &local_vars, &stack_slots)?;
                                if let Some(entry) = pow2_mod_record
                                    && !stack_slots.contains_key(&dst.0)
                                {
                                    cur_pow2_mod_defs.insert(dst.0, entry);
                                } else {
                                    cur_pow2_mod_defs.remove(&dst.0);
                                }
                                cur_proven_in_bounds
                                    .retain(|&(arr_l, idx_l)| arr_l != *dst && idx_l != *dst);
                                match rvalue {
                                    Rvalue::Use(
                                        Operand::Copy(Place::Local(src))
                                        | Operand::Move(Place::Local(src)),
                                    ) if proven_true_cond_locals.contains(&src.0) => {
                                        proven_true_cond_locals.insert(dst.0);
                                    }
                                    _ => {
                                        proven_true_cond_locals.remove(&dst.0);
                                    }
                                }
                                match rvalue {
                                    Rvalue::Aggregate(MirType::Array(_), _) => {
                                        cur_validated.insert(dst.0);
                                    }
                                    Rvalue::Use(
                                        Operand::Copy(Place::Local(src))
                                        | Operand::Move(Place::Local(src)),
                                    ) if cur_validated.contains(&src.0) => {
                                        cur_validated.insert(dst.0);
                                    }
                                    _ => {
                                        cur_validated.remove(&dst.0);
                                    }
                                }
                            }
                            Statement::Assign(lhs_place @ Place::Index(..), rvalue, _) => {
                                let val = lower_rvalue(
                                    rvalue,
                                    None,
                                    None,
                                    &mut cur_validated,
                                    &mut builder,
                                    &local_vars,
                                    &stack_slots,
                                    &mir_fn.locals,
                                    &self.struct_layouts,
                                    alloc_array_fn,
                                    alloc_array_typed_fn,
                                    alloc_tuple_typed_fn,
                                    alloc_struct_named_fn,
                                    alloc_cflat_struct_fn,
                                    len_fn,
                                    floordiv_fn,
                                    pow_fn,
                                    floordiv_float_fn,
                                    pow_float_fn,
                                    rem_float_fn,
                                    string_helpers,
                                    status_fn,
                                    stmt_bc_ctx,
                                    None,
                                )?;
                                let elem_addr = lower_place_addr(
                                    lhs_place,
                                    &mut builder,
                                    &local_vars,
                                    &stack_slots,
                                    &mir_fn.locals,
                                    &self.struct_layouts,
                                    stmt_bc_ctx,
                                )?;
                                let val_ty = builder.func.dfg.value_type(val);
                                let val_to_store = if is_bool_array_index_place(
                                    lhs_place,
                                    &mir_fn.locals,
                                    &self.struct_layouts,
                                ) {
                                    if val_ty == types::I8 {
                                        val
                                    } else {
                                        builder.ins().icmp_imm_s(IntCC::NotEqual, val, 0)
                                    }
                                } else if val_ty == types::I8 {
                                    builder.ins().uextend(types::I64, val)
                                } else {
                                    val
                                };
                                builder.ins().store(MemFlagsData::trusted(), val_to_store, elem_addr, 0);
                            }
                            Statement::Assign(lhs_place @ Place::Field(..), rvalue, _) => {
                                let val = lower_rvalue(
                                    rvalue,
                                    None,
                                    None,
                                    &mut cur_validated,
                                    &mut builder,
                                    &local_vars,
                                    &stack_slots,
                                    &mir_fn.locals,
                                    &self.struct_layouts,
                                    alloc_array_fn,
                                    alloc_array_typed_fn,
                                    alloc_tuple_typed_fn,
                                    alloc_struct_named_fn,
                                    alloc_cflat_struct_fn,
                                    len_fn,
                                    floordiv_fn,
                                    pow_fn,
                                    floordiv_float_fn,
                                    pow_float_fn,
                                    rem_float_fn,
                                    string_helpers,
                                    status_fn,
                                    stmt_bc_ctx,
                                    None,
                                )?;
                                let field_addr = lower_place_addr(
                                    lhs_place,
                                    &mut builder,
                                    &local_vars,
                                    &stack_slots,
                                    &mir_fn.locals,
                                    &self.struct_layouts,
                                    stmt_bc_ctx,
                                )?;
                                let val_ty = builder.func.dfg.value_type(val);
                                let val_to_store = if val_ty == types::I8 {
                                    builder.ins().uextend(types::I64, val)
                                } else {
                                    val
                                };
                                builder.ins().store(MemFlagsData::trusted(), val_to_store, field_addr, 0);
                            }
                            Statement::ResourceCheck(ResourceKind::Gas, amt, _) => {
                                let (gas_ptr_var, _, gas_slow_fn) =
                                    gas_ctx.expect("ResourceCheck(Gas) implies needs_gas_fast");
                                emit_inline_gas_check(
                                    &mut builder,
                                    gas_ptr_var,
                                    gas_slow_fn,
                                    *amt,
                                    emergency_exit_bb,
                                    active_promoted,
                                    cl_opt.map(|ci| &mut cfg_records[ci]),
                                );
                            }
                            Statement::StorageDead(loc, _) => {
                                proven_true_cond_locals.remove(&loc.0);
                            }
                            _ => {}
                        }
                    }

                    let active_bc_ctx = BoundsCheckCtx {
                        status_set_fn,
                        emergency_exit_bb: active_emergency_bb,
                        hoisted_arrays: &active_hoisted_arrays,
                        hoisted_structs: &active_hoisted_structs,
                        proven_in_bounds: &cur_proven_in_bounds,
                        const_int_locals: &const_int_locals,
                        pow2_mod_defs: &empty_pow2_mod_defs,
                        inline_status_ptr_var,
                        closure_fn_ids: closure_fn_ids_ref,
                    };

                    // If this block is the pre-entry to a vectorized loop, insert runtime scalar-forced
                    // and shifted-alias guards before entering the SIMD loop header
                    if let Some(&vl_idx) = vec_pre_map.get(&b_idx) {
                        let vl = &vec_loops[vl_idx];
                        let simd_scalar_forced_fn = simd_scalar_forced_fn.expect("has_simd implies simd_scalar_forced_fn");
                        let simd_vector_fn = simd_vector_fn.expect("has_simd implies simd_vector_fn");
                        let simd_scalar_fn = simd_scalar_fn.expect("has_simd implies simd_scalar_fn");
                        let forced_call = builder.ins().call(simd_scalar_forced_fn, &[]);
                        let forced_val = builder.inst_results(forced_call)[0];
                        let is_forced = if vl.gas_check.is_some() {
                            // Safepoints may be inserted after SIMD transformation.
                            // Enter the retained scalar header before any chunk gas
                            // charge or write, preserving each exhaustion boundary.
                            builder.ins().iconst(types::I8, 1)
                        } else {
                            builder.ins().icmp_imm_s(IntCC::NotEqual, forced_val, 0)
                        };

                        let end_val = load_local_i64(vl.end_local, &mut builder, &local_vars, &stack_slots)?;
                        let a_ptr = load_local_i64(vl.a_local, &mut builder, &local_vars, &stack_slots)?;
                        let a_zero = builder.ins().icmp_imm_s(IntCC::Equal, a_ptr, 0);
                        let end_lt_4 = builder.ins().icmp_imm_s(IntCC::SignedLessThan, end_val, 4);
                        let mut fallback_cond = builder.ins().bor(is_forced, a_zero);
                        fallback_cond = builder.ins().bor(fallback_cond, end_lt_4);

                        let b_ptr_opt = if let Some(b_loc) = vl.b_local {
                            let bp = load_local_i64(b_loc, &mut builder, &local_vars, &stack_slots)?;
                            let b_zero = builder.ins().icmp_imm_s(IntCC::Equal, bp, 0);
                            fallback_cond = builder.ins().bor(fallback_cond, b_zero);
                            Some(bp)
                        } else {
                            None
                        };

                        let out_ptr_opt = if let Some(out_loc) = vl.out_local {
                            let op = load_local_i64(out_loc, &mut builder, &local_vars, &stack_slots)?;
                            let out_zero = builder.ins().icmp_imm_s(IntCC::Equal, op, 0);
                            fallback_cond = builder.ins().bor(fallback_cond, out_zero);
                            Some(op)
                        } else {
                            None
                        };

                        let chk_alias_bb = builder.create_block();
                        let scalar_fallback_bb = builder.create_block();
                        let enter_vec_bb = builder.create_block();
                        builder.ins().brif(fallback_cond, scalar_fallback_bb, &[], chk_alias_bb, &[]);

                        builder.switch_to_block(chk_alias_bb);
                        // Redirected arrays (cap < 0) store elements in a separate backing buffer;
                        // divert them cleanly to the scalar loop whose Place::Index resolves redirection.
                        let cap_off = core::mem::offset_of!(MirArrayHeader, cap) as i32;
                        let a_cap = builder.ins().load(types::I64, MemFlagsData::trusted(), a_ptr, cap_off);
                        let mut any_redir = builder.ins().icmp_imm_s(IntCC::SignedLessThan, a_cap, 0);
                        if let Some(b_ptr) = b_ptr_opt {
                            let b_cap = builder.ins().load(types::I64, MemFlagsData::trusted(), b_ptr, cap_off);
                            let b_redir = builder.ins().icmp_imm_s(IntCC::SignedLessThan, b_cap, 0);
                            any_redir = builder.ins().bor(any_redir, b_redir);
                        }
                        if let Some(out_ptr) = out_ptr_opt {
                            let out_cap = builder.ins().load(types::I64, MemFlagsData::trusted(), out_ptr, cap_off);
                            let out_redir = builder.ins().icmp_imm_s(IntCC::SignedLessThan, out_cap, 0);
                            any_redir = builder.ins().bor(any_redir, out_redir);
                        }
                        if let Some(out_ptr) = out_ptr_opt {
                            let eight = builder.ins().iconst(types::I64, 8);
                            let vec_bytes = builder.ins().imul(end_val, eight);
                            let out_data = builder.ins().iadd_imm_s(out_ptr, 16);
                            let out_end = builder.ins().iadd(out_data, vec_bytes);

                            // Check writes [out_data, out_end) against full source objects [ptr, ptr + 16 + vec_bytes)
                            // so shifted overlaps hitting either input elements or 16-byte array headers divert to scalar.
                            let a_data = builder.ins().iadd_imm_s(a_ptr, 16);
                            let a_end = builder.ins().iadd(a_data, vec_bytes);
                            let a_diff = builder.ins().icmp(IntCC::NotEqual, out_data, a_data);
                            let a_ov1 = builder.ins().icmp(IntCC::UnsignedLessThan, out_data, a_end);
                            let a_ov2 = builder.ins().icmp(IntCC::UnsignedLessThan, a_ptr, out_end);
                            let a_ov = builder.ins().band(a_ov1, a_ov2);
                            let mut any_shifted_ov = builder.ins().band(a_diff, a_ov);

                            if let Some(b_ptr) = b_ptr_opt {
                                let b_data = builder.ins().iadd_imm_s(b_ptr, 16);
                                let b_end = builder.ins().iadd(b_data, vec_bytes);
                                let b_diff = builder.ins().icmp(IntCC::NotEqual, out_data, b_data);
                                let b_ov1 = builder.ins().icmp(IntCC::UnsignedLessThan, out_data, b_end);
                                let b_ov2 = builder.ins().icmp(IntCC::UnsignedLessThan, b_ptr, out_end);
                                let b_ov = builder.ins().band(b_ov1, b_ov2);
                                let b_shifted_ov = builder.ins().band(b_diff, b_ov);
                                any_shifted_ov = builder.ins().bor(any_shifted_ov, b_shifted_ov);
                            }

                            let must_fallback = builder.ins().bor(any_redir, any_shifted_ov);
                            builder.ins().brif(must_fallback, scalar_fallback_bb, &[], enter_vec_bb, &[]);
                        } else {
                            builder.ins().brif(any_redir, scalar_fallback_bb, &[], enter_vec_bb, &[]);
                        }

                        builder.switch_to_block(scalar_fallback_bb);
                        builder.ins().call(simd_scalar_fn, &[]);
                        let tail_target = resolve_target_bb(Some(b_idx), vl.tail_block.0)?;
                        builder.ins().jump(tail_target, &[]);

                        builder.switch_to_block(enter_vec_bb);
                        builder.ins().call(simd_vector_fn, &[]);
                        if let Some((vacc0, vacc1)) = vec_acc_vars[vl_idx] {
                            let zero_s = builder.ins().iconst(types::I64, 0);
                            let vzero = builder.ins().splat(types::I64X2, zero_s);
                            builder.def_var(vacc0, vzero);
                            builder.def_var(vacc1, vzero);
                        }
                        let hdr_target = resolve_target_bb(Some(b_idx), vl.hdr_block.0)?;
                        builder.ins().jump(hdr_target, &[]);
                        continue;
                    }

                    if let Some(term) = &mir_block.terminator {
                        match term {
                            Terminator::Goto(target, _) => {
                                let target_bb = resolve_target_bb(Some(b_idx), target.0)?;
                                builder.ins().jump(target_bb, &[]);
                            }
                            Terminator::Cond { cond, then_block, else_block, .. } => {
                                let is_proven_true = matches!(
                                    cond,
                                    Operand::Copy(Place::Local(d)) | Operand::Move(Place::Local(d))
                                        if proven_true_cond_locals.contains(&d.0)
                                );
                                if is_proven_true {
                                    let then_bb = resolve_target_bb(Some(b_idx), then_block.0)?;
                                    builder.ins().jump(then_bb, &[]);
                                } else {
                                    let c_val = lower_operand(cond, &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, active_bc_ctx)?;
                                    let then_bb = resolve_target_bb(Some(b_idx), then_block.0)?;
                                    let else_bb = resolve_target_bb(Some(b_idx), else_block.0)?;
                                    builder.ins().brif(c_val, then_bb, &[], else_bb, &[]);
                                }
                            }
                            Terminator::Switch { discr, cases, default, .. } => {
                                let d_val = lower_operand(discr, &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, active_bc_ctx)?;
                                for (val, target_bb) in cases {
                                    let is_eq = builder.ins().icmp_imm_s(IntCC::Equal, d_val, *val);
                                    let next_check = builder.create_block();
                                    let t_bb = resolve_target_bb(Some(b_idx), target_bb.0)?;
                                    builder.ins().brif(is_eq, t_bb, &[], next_check, &[]);
                                    builder.switch_to_block(next_check);
                                }
                                let def_bb = resolve_target_bb(Some(b_idx), default.0)?;
                                builder.ins().jump(def_bb, &[]);
                            }
                            Terminator::Call { callee, args, destination, success, .. } => {
                                if callee == "print" || callee == "println" {
                                    for (idx, arg) in args.iter().enumerate() {
                                        if idx > 0 {
                                            let print_sep_fn = Self::declare_ext_fn(
                                                &mut self.module,
                                                &mut builder.func,
                                                &mut ext_fn_cache,
                                                "lust_mir_print_sep",
                                                &[],
                                                None,
                                            )?;
                                            builder.ins().call(print_sep_fn, &[]);
                                        }
                                        let mir_ty = try_operand_type(arg, &mir_fn.locals, &self.struct_layouts)?;
                                        let arg_val = lower_operand(arg, &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                        let clif_ty = builder.func.dfg.value_type(arg_val);
                                        match (&mir_ty, clif_ty) {
                                            (MirType::Int, types::I64) => {
                                                let print_int_part_fn = Self::declare_ext_fn(
                                                    &mut self.module,
                                                    &mut builder.func,
                                                    &mut ext_fn_cache,
                                                    "lust_mir_print_int_part",
                                                    &[types::I64],
                                                    None,
                                                )?;
                                                builder.ins().call(print_int_part_fn, &[arg_val]);
                                            }
                                            (MirType::Float, types::F64) => {
                                                let print_float_part_fn = Self::declare_ext_fn(
                                                    &mut self.module,
                                                    &mut builder.func,
                                                    &mut ext_fn_cache,
                                                    "lust_mir_print_float_part",
                                                    &[types::F64],
                                                    None,
                                                )?;
                                                builder.ins().call(print_float_part_fn, &[arg_val]);
                                            }
                                            (MirType::Bool, types::I8) => {
                                                let print_bool_part_fn = Self::declare_ext_fn(
                                                    &mut self.module,
                                                    &mut builder.func,
                                                    &mut ext_fn_cache,
                                                    "lust_mir_print_bool_part",
                                                    &[types::I64],
                                                    None,
                                                )?;
                                                let ext = builder.ins().uextend(types::I64, arg_val);
                                                builder.ins().call(print_bool_part_fn, &[ext]);
                                            }
                                            (MirType::Bool, types::I64) => {
                                                let print_bool_part_fn = Self::declare_ext_fn(
                                                    &mut self.module,
                                                    &mut builder.func,
                                                    &mut ext_fn_cache,
                                                    "lust_mir_print_bool_part",
                                                    &[types::I64],
                                                    None,
                                                )?;
                                                builder.ins().call(print_bool_part_fn, &[arg_val]);
                                            }
                                            (MirType::String, types::I64) => {
                                                let print_string_part_fn = Self::declare_ext_fn(
                                                    &mut self.module,
                                                    &mut builder.func,
                                                    &mut ext_fn_cache,
                                                    "lust_mir_print_string_part",
                                                    &[types::I64],
                                                    None,
                                                )?;
                                                builder.ins().call(print_string_part_fn, &[arg_val]);
                                            }
                                            _ => {
                                                return Err(MirCraneliftBackendError::CodegenError(format!(
                                                    "Unsupported argument type ({mir_ty}, {clif_ty}) for Cranelift {callee}"
                                                )));
                                            }
                                        }
                                    }
                                    if callee == "println" {
                                        let print_nl_fn = Self::declare_ext_fn(
                                            &mut self.module,
                                            &mut builder.func,
                                            &mut ext_fn_cache,
                                            "lust_mir_print_nl",
                                            &[],
                                            None,
                                        )?;
                                        builder.ins().call(print_nl_fn, &[]);
                                    }
                                    match destination {
                                        Place::Local(loc) => {
                                            let dest_mir_ty = &mir_fn
                                                .locals
                                                .get(loc.0)
                                                .ok_or_else(|| {
                                                    MirCraneliftBackendError::CodegenError(format!(
                                                        "Destination local {loc:?} out of bounds for Cranelift {callee}"
                                                    ))
                                                })?
                                                .ty;
                                            let dest_clif_ty = mir_to_clif_type(dest_mir_ty)?;
                                            match (dest_mir_ty, dest_clif_ty) {
                                                (MirType::Unit, types::I8) => {
                                                    let zero = builder.ins().iconst(types::I8, 0);
                                                    store_local_val(*loc, zero, &mut builder, &local_vars, &stack_slots)?;
                                                }
                                                (MirType::Int, types::I64) => {
                                                    let zero = builder.ins().iconst(types::I64, 0);
                                                    store_local_val(*loc, zero, &mut builder, &local_vars, &stack_slots)?;
                                                }
                                                _ => {
                                                    return Err(MirCraneliftBackendError::CodegenError(format!(
                                                        "Unsupported destination type ({dest_mir_ty}, {dest_clif_ty}) for Cranelift {callee}"
                                                    )));
                                                }
                                            }
                                        }
                                        _ => {
                                            return Err(MirCraneliftBackendError::CodegenError(format!(
                                                "Unsupported non-local destination place for Cranelift {callee}"
                                            )));
                                        }
                                    }
                                    let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                    builder.ins().jump(succ_bb, &[]);
                                } else if let Some(struct_name) = crate::mir::ir::parse_mir_struct_decode_callee(callee) {
                                    let layout_id = if let Some(layout) = self.struct_layouts.get(struct_name) {
                                        layout.type_id as i64
                                    } else if let Some(type_id) = STRUCT_NAME_MAP.with(|m| m.borrow().get(struct_name).copied()) {
                                        type_id as i64
                                    } else {
                                        0
                                    };
                                    let layout_id_val = builder.ins().iconst(types::I64, layout_id);
                                    let arg_val = if let Some(first) = args.first() {
                                        lower_operand(first, &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?
                                    } else {
                                        builder.ins().iconst(types::I64, 0)
                                    };
                                    let struct_decode_fn = Self::declare_ext_fn(
                                        &mut self.module,
                                        &mut builder.func,
                                        &mut ext_fn_cache,
                                        "lust_mir_struct_decode_contiguous",
                                        &[types::I64, types::I64],
                                        Some(types::I64),
                                    )?;
                                    let call_inst = builder.ins().call(struct_decode_fn, &[layout_id_val, arg_val]);
                                    let res_val = builder.inst_results(call_inst)[0];
                                    if let Place::Local(loc) = destination {
                                        let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                        let res_cast = cast_clif_type(&mut builder, res_val, target_ty);
                                        store_local_val(*loc, res_cast, &mut builder, &local_vars, &stack_slots)?;
                                    }
                                    let stat_call = builder.ins().call(status_fn, &[]);
                                    let stat_val = builder.inst_results(stat_call)[0];
                                    let is_err = builder.ins().icmp_imm_s(IntCC::NotEqual, stat_val, 0);
                                    let call_cont_bb = builder.create_block();
                                    builder.ins().brif(is_err, emergency_exit_bb, &[], call_cont_bb, &[]);
                                    builder.switch_to_block(call_cont_bb);
                                    let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                    builder.ins().jump(succ_bb, &[]);
                                } else if is_effect_or_handler_callee(callee) {
                                    // 3A->3D deopt seam: publish DeoptPayload and exit with MIR_STATUS_DEOPT
                                    let proven_locals_storage;
                                    let locals_to_record = if let Some((_, live_locals)) = osr_info {
                                        live_locals
                                    } else {
                                        match mir_fn.derive_proven_site_metadata(b_idx) {
                                            Ok(proven) => {
                                                proven_locals_storage = proven.into_iter().map(LocalId).collect::<Vec<_>>();
                                                &proven_locals_storage
                                            }
                                            Err(err) => {
                                                return Err(MirCraneliftBackendError::CodegenError(alloc::format!(
                                                    "Cannot derive authoritative deopt metadata for site {}: {}",
                                                    b_idx, err
                                                )));
                                            }
                                        }
                                    };
                                    let target_pc_val = builder.ins().iconst(types::I64, b_idx as i64);
                                    let init_deopt_at_terminator_fn = Self::declare_ext_fn(
                                        &mut self.module,
                                        &mut builder.func,
                                        &mut ext_fn_cache,
                                        "lust_mir_init_deopt_at_terminator",
                                        &[types::I64],
                                        None,
                                    )?;
                                    builder.ins().call(init_deopt_at_terminator_fn, &[target_pc_val]);
                                    for &loc_id in locals_to_record {
                                        if loc_id.0 < local_vars.len() {
                                            let loc_idx_val = builder.ins().iconst(types::I64, loc_id.0 as i64);
                                            let cur_val = if let Some(&slot) = stack_slots.get(&loc_id.0) {
                                                let mir_ty = mir_fn.locals.get(loc_id.0).map(|l| l.ty.clone()).unwrap_or(MirType::Int);
                                                let clif_ty = mir_to_clif_type(&mir_ty).unwrap_or(types::I64);
                                                builder.ins().stack_load(types::I64, clif_ty, slot, 0)
                                            } else {
                                                builder.use_var(local_vars[loc_id.0])
                                            };
                                            let val_ty = builder.func.dfg.value_type(cur_val);
                                            let val_i64 = if val_ty == types::I8 {
                                                builder.ins().uextend(types::I64, cur_val)
                                            } else if val_ty == types::F64 {
                                                builder.ins().bitcast(types::I64, MemFlagsData::new(), cur_val)
                                            } else {
                                                cur_val
                                            };
                                            builder.ins().call(record_deopt_local_fn, &[target_pc_val, loc_idx_val, val_i64]);
                                        }
                                    }
                                    let deopt_status_val = builder.ins().iconst(types::I64, MIR_STATUS_DEOPT);
                                    builder.ins().call(status_set_fn, &[deopt_status_val]);
                                    builder.ins().jump(emergency_exit_bb, &[]);
                                } else if callee.starts_with("ic:") {
                                    let site_id_num: u32 = callee[3..].parse().unwrap_or(0);
                                    let site_id_val = builder.ins().iconst(types::I64, site_id_num as i64);
                                    let tag_val = if let Some(first) = args.first() {
                                        lower_operand(first, &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?
                                    } else {
                                        builder.ins().iconst(types::I64, 0)
                                    };
                                    let def_val = if args.len() > 1 {
                                        lower_operand(&args[1], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?
                                    } else {
                                        builder.ins().iconst(types::I64, 0)
                                    };
                                    let ic_dispatch_fn = Self::declare_ext_fn(
                                        &mut self.module,
                                        &mut builder.func,
                                        &mut ext_fn_cache,
                                        "lust_mir_ic_dispatch",
                                        &[types::I64, types::I64, types::I64],
                                        Some(types::I64),
                                    )?;
                                    let call_inst = builder.ins().call(ic_dispatch_fn, &[site_id_val, tag_val, def_val]);
                                    let res_val = builder.inst_results(call_inst)[0];
                                    if let Place::Local(loc) = destination {
                                        let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                        let res_cast = cast_clif_type(&mut builder, res_val, target_ty);
                                        store_local_val(*loc, res_cast, &mut builder, &local_vars, &stack_slots)?;
                                    }
                                    let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                    builder.ins().jump(succ_bb, &[]);
                                } else if callee == "simd.add" || callee == "simd_add" || callee == "vector.add" {
                                    let simd_vector_fn = simd_vector_fn.expect("has_simd implies simd_vector_fn");
                                    let simd_scalar_fn = simd_scalar_fn.expect("has_simd implies simd_scalar_fn");
                                    let a_val = lower_operand(&args[0], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let b_val = lower_operand(&args[1], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let out_val = lower_operand(&args[2], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let len_val = lower_operand(&args[3], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let force_raw = lower_operand(&args[4], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let force_val = cast_clif_type(&mut builder, force_raw, types::I64);

                                    let var_idx = builder.declare_var(types::I64);
                                    let var_chunks = builder.declare_var(types::I64);

                                    let null_err_bb = builder.create_block();
                                    let check_mode_bb = builder.create_block();
                                    let vector_bb = builder.create_block();
                                    let vector_loop_hdr = builder.create_block();
                                    let vector_loop_body = builder.create_block();
                                    let to_tail_bb = builder.create_block();
                                    let vector_tail_hdr = builder.create_block();
                                    let vector_tail_body = builder.create_block();
                                    let scalar_fwd_hdr = builder.create_block();
                                    let scalar_fwd_body = builder.create_block();
                                    let scalar_bwd_hdr = builder.create_block();
                                    let scalar_bwd_body = builder.create_block();
                                    let done_success_bb = builder.create_block();
                                    let next_mir_bb = resolve_target_bb(Some(b_idx), success.0)?;

                                    // Null checks: a == 0 || b == 0 || out == 0
                                    let a_is_zero = builder.ins().icmp_imm_s(IntCC::Equal, a_val, 0);
                                    let b_is_zero = builder.ins().icmp_imm_s(IntCC::Equal, b_val, 0);
                                    let out_is_zero = builder.ins().icmp_imm_s(IntCC::Equal, out_val, 0);
                                    let ab_zero = builder.ins().bor(a_is_zero, b_is_zero);
                                    let any_zero = builder.ins().bor(ab_zero, out_is_zero);
                                    builder.ins().brif(any_zero, null_err_bb, &[], check_mode_bb, &[]);

                                    // Null error block: return 0
                                    builder.switch_to_block(null_err_bb);
                                    let zero_res = builder.ins().iconst(types::I64, 0);
                                    if let Place::Local(loc) = destination {
                                        let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                        let res_cast = cast_clif_type(&mut builder, zero_res, target_ty);
                                        builder.def_var(local_vars[loc.0], res_cast);
                                        if let Some(&slot) = stack_slots.get(&loc.0) {
                                            builder.ins().stack_store(types::I64, res_cast, slot, 0);
                                        }
                                    }
                                    builder.ins().jump(next_mir_bb, &[]);

                                    // Check mode block: alias hazard detection & vector threshold
                                    builder.switch_to_block(check_mode_bb);
                                    let len_le_zero = builder.ins().icmp_imm_s(IntCC::SignedLessThanOrEqual, len_val, 0);
                                    let normal_exec_bb = builder.create_block();
                                    builder.ins().brif(len_le_zero, done_success_bb, &[], normal_exec_bb, &[]);

                                    builder.switch_to_block(normal_exec_bb);
                                    let eight = builder.ins().iconst(types::I64, 8);
                                    let len_bytes = builder.ins().imul(len_val, eight);
                                    let a_end = builder.ins().iadd(a_val, len_bytes);
                                    let b_end = builder.ins().iadd(b_val, len_bytes);
                                    let out_end = builder.ins().iadd(out_val, len_bytes);

                                    // Operand A shifted overlap & directional hazards:
                                    let out_lt_a_end = builder.ins().icmp(IntCC::UnsignedLessThan, out_val, a_end);
                                    let a_lt_out_end = builder.ins().icmp(IntCC::UnsignedLessThan, a_val, out_end);
                                    let a_overlaps = builder.ins().band(out_lt_a_end, a_lt_out_end);
                                    let a_diff = builder.ins().icmp(IntCC::NotEqual, out_val, a_val);
                                    let a_shifted = builder.ins().band(a_overlaps, a_diff);

                                    let out_gt_a = builder.ins().icmp(IntCC::UnsignedGreaterThan, out_val, a_val);
                                    let a_fwd_hazard = builder.ins().band(a_shifted, out_gt_a);
                                    let out_lt_a = builder.ins().icmp(IntCC::UnsignedLessThan, out_val, a_val);
                                    let a_bwd_hazard = builder.ins().band(a_shifted, out_lt_a);

                                    // Operand B shifted overlap & directional hazards:
                                    let out_lt_b_end = builder.ins().icmp(IntCC::UnsignedLessThan, out_val, b_end);
                                    let b_lt_out_end = builder.ins().icmp(IntCC::UnsignedLessThan, b_val, out_end);
                                    let b_overlaps = builder.ins().band(out_lt_b_end, b_lt_out_end);
                                    let b_diff = builder.ins().icmp(IntCC::NotEqual, out_val, b_val);
                                    let b_shifted = builder.ins().band(b_overlaps, b_diff);

                                    let out_gt_b = builder.ins().icmp(IntCC::UnsignedGreaterThan, out_val, b_val);
                                    let b_fwd_hazard = builder.ins().band(b_shifted, out_gt_b);
                                    let out_lt_b = builder.ins().icmp(IntCC::UnsignedLessThan, out_val, b_val);
                                    let b_bwd_hazard = builder.ins().band(b_shifted, out_lt_b);

                                    // Contradictory overlap: forward shift on one operand and backward shift on another.
                                    // Reject atomically by branching to null_err_bb (returns 0 with 0 writes).
                                    let has_fwd_hazard = builder.ins().bor(a_fwd_hazard, b_fwd_hazard);
                                    let has_bwd_hazard = builder.ins().bor(a_bwd_hazard, b_bwd_hazard);
                                    let is_contradictory = builder.ins().band(has_fwd_hazard, has_bwd_hazard);

                                    let check_contradictory_bb = builder.create_block();
                                    let fwd_hazard_bb = builder.create_block();
                                    let bwd_or_clean_bb = builder.create_block();
                                    let fwd_scalar_bb = builder.create_block();

                                    builder.ins().brif(is_contradictory, null_err_bb, &[], check_contradictory_bb, &[]);

                                    builder.switch_to_block(check_contradictory_bb);
                                    builder.ins().brif(has_fwd_hazard, fwd_hazard_bb, &[], bwd_or_clean_bb, &[]);

                                    // Forward hazard path (out > inputs): requires backward scalar traversal
                                    builder.switch_to_block(fwd_hazard_bb);
                                    builder.ins().call(simd_scalar_fn, &[]);
                                    let len_minus_one = builder.ins().iadd_imm_s(len_val, -1);
                                    builder.def_var(var_idx, len_minus_one);
                                    builder.ins().jump(scalar_bwd_hdr, &[]);

                                    // Backward hazard or clean path
                                    builder.switch_to_block(bwd_or_clean_bb);
                                    let is_force = builder.ins().icmp_imm_s(IntCC::NotEqual, force_val, 0);
                                    let is_short = builder.ins().icmp_imm_s(IntCC::SignedLessThan, len_val, 2);
                                    let force_or_short = builder.ins().bor(is_force, is_short);
                                    let need_fwd_scalar = builder.ins().bor(has_bwd_hazard, force_or_short);
                                    builder.ins().brif(need_fwd_scalar, fwd_scalar_bb, &[], vector_bb, &[]);

                                    builder.switch_to_block(fwd_scalar_bb);
                                    builder.ins().call(simd_scalar_fn, &[]);
                                    let zero = builder.ins().iconst(types::I64, 0);
                                    builder.def_var(var_idx, zero);
                                    builder.ins().jump(scalar_fwd_hdr, &[]);

                                    // Vector init
                                    builder.switch_to_block(vector_bb);
                                    builder.ins().call(simd_vector_fn, &[]);
                                    let zero = builder.ins().iconst(types::I64, 0);
                                    let two = builder.ins().iconst(types::I64, 2);
                                    let chunks = builder.ins().sdiv(len_val, two);
                                    builder.def_var(var_chunks, chunks);
                                    builder.def_var(var_idx, zero);
                                    builder.ins().jump(vector_loop_hdr, &[]);

                                    // Vector loop header
                                    builder.switch_to_block(vector_loop_hdr);
                                    let c = builder.use_var(var_idx);
                                    let max_c = builder.use_var(var_chunks);
                                    let vec_cond = builder.ins().icmp(IntCC::SignedLessThan, c, max_c);
                                    builder.ins().brif(vec_cond, vector_loop_body, &[], to_tail_bb, &[]);

                                    builder.switch_to_block(to_tail_bb);
                                    let tail_start = builder.ins().imul(max_c, two);
                                    builder.def_var(var_idx, tail_start);
                                    builder.ins().jump(vector_tail_hdr, &[]);

                                    // Vector loop body
                                    builder.switch_to_block(vector_loop_body);
                                    let sixteen = builder.ins().iconst(types::I64, 16);
                                    let byte_offset = builder.ins().imul(c, sixteen);
                                    let cur_a = builder.ins().iadd(a_val, byte_offset);
                                    let cur_b = builder.ins().iadd(b_val, byte_offset);
                                    let cur_out = builder.ins().iadd(out_val, byte_offset);
                                    let va = builder.ins().load(types::I64X2, MemFlagsData::new(), cur_a, 0);
                                    let vb = builder.ins().load(types::I64X2, MemFlagsData::new(), cur_b, 0);
                                    let vres = builder.ins().iadd(va, vb);
                                    builder.ins().store(MemFlagsData::new(), vres, cur_out, 0);
                                    let next_c = builder.ins().iadd_imm_s(c, 1);
                                    builder.def_var(var_idx, next_c);
                                    builder.ins().jump(vector_loop_hdr, &[]);

                                    // Vector tail header
                                    builder.switch_to_block(vector_tail_hdr);
                                    let tail_idx = builder.use_var(var_idx);
                                    let tail_cond = builder.ins().icmp(IntCC::SignedLessThan, tail_idx, len_val);
                                    builder.ins().brif(tail_cond, vector_tail_body, &[], done_success_bb, &[]);

                                    // Vector tail body
                                    builder.switch_to_block(vector_tail_body);
                                    let tail_offset = builder.ins().imul(tail_idx, eight);
                                    let ta = builder.ins().iadd(a_val, tail_offset);
                                    let tb = builder.ins().iadd(b_val, tail_offset);
                                    let tout = builder.ins().iadd(out_val, tail_offset);
                                    let ea = builder.ins().load(types::I64, MemFlagsData::new(), ta, 0);
                                    let eb = builder.ins().load(types::I64, MemFlagsData::new(), tb, 0);
                                    let eres = builder.ins().iadd(ea, eb);
                                    builder.ins().store(MemFlagsData::new(), eres, tout, 0);
                                    let next_tail = builder.ins().iadd_imm_s(tail_idx, 1);
                                    builder.def_var(var_idx, next_tail);
                                    builder.ins().jump(vector_tail_hdr, &[]);

                                    // Forward scalar loop header and body
                                    builder.switch_to_block(scalar_fwd_hdr);
                                    let cur_fwd = builder.use_var(var_idx);
                                    let fwd_cond = builder.ins().icmp(IntCC::SignedLessThan, cur_fwd, len_val);
                                    builder.ins().brif(fwd_cond, scalar_fwd_body, &[], done_success_bb, &[]);

                                    builder.switch_to_block(scalar_fwd_body);
                                    let fwd_offset = builder.ins().imul(cur_fwd, eight);
                                    let fa = builder.ins().iadd(a_val, fwd_offset);
                                    let fb = builder.ins().iadd(b_val, fwd_offset);
                                    let fout = builder.ins().iadd(out_val, fwd_offset);
                                    let ea = builder.ins().load(types::I64, MemFlagsData::new(), fa, 0);
                                    let eb = builder.ins().load(types::I64, MemFlagsData::new(), fb, 0);
                                    let eres = builder.ins().iadd(ea, eb);
                                    builder.ins().store(MemFlagsData::new(), eres, fout, 0);
                                    let next_fwd = builder.ins().iadd_imm_s(cur_fwd, 1);
                                    builder.def_var(var_idx, next_fwd);
                                    builder.ins().jump(scalar_fwd_hdr, &[]);

                                    // Backward scalar loop header and body
                                    builder.switch_to_block(scalar_bwd_hdr);
                                    let cur_bwd = builder.use_var(var_idx);
                                    let bwd_cond = builder.ins().icmp_imm_s(IntCC::SignedGreaterThanOrEqual, cur_bwd, 0);
                                    builder.ins().brif(bwd_cond, scalar_bwd_body, &[], done_success_bb, &[]);

                                    builder.switch_to_block(scalar_bwd_body);
                                    let bwd_offset = builder.ins().imul(cur_bwd, eight);
                                    let ba = builder.ins().iadd(a_val, bwd_offset);
                                    let bb = builder.ins().iadd(b_val, bwd_offset);
                                    let bout = builder.ins().iadd(out_val, bwd_offset);
                                    let ea = builder.ins().load(types::I64, MemFlagsData::new(), ba, 0);
                                    let eb = builder.ins().load(types::I64, MemFlagsData::new(), bb, 0);
                                    let eres = builder.ins().iadd(ea, eb);
                                    builder.ins().store(MemFlagsData::new(), eres, bout, 0);
                                    let next_bwd = builder.ins().iadd_imm_s(cur_bwd, -1);
                                    builder.def_var(var_idx, next_bwd);
                                    builder.ins().jump(scalar_bwd_hdr, &[]);

                                    // Done success
                                    builder.switch_to_block(done_success_bb);
                                    let one_res = builder.ins().iconst(types::I64, 1);
                                    if let Place::Local(loc) = destination {
                                        let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                        let res_cast = cast_clif_type(&mut builder, one_res, target_ty);
                                        builder.def_var(local_vars[loc.0], res_cast);
                                        if let Some(&slot) = stack_slots.get(&loc.0) {
                                            builder.ins().stack_store(types::I64, res_cast, slot, 0);
                                        }
                                    }
                                    builder.ins().jump(next_mir_bb, &[]);
                                } else if callee == "simd.mul" || callee == "simd_mul" || callee == "vector.mul" {
                                    let simd_vector_fn = simd_vector_fn.expect("has_simd implies simd_vector_fn");
                                    let simd_scalar_fn = simd_scalar_fn.expect("has_simd implies simd_scalar_fn");
                                    let a_val = lower_operand(&args[0], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let b_val = lower_operand(&args[1], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let out_val = lower_operand(&args[2], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let len_val = lower_operand(&args[3], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let force_raw = lower_operand(&args[4], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let force_val = cast_clif_type(&mut builder, force_raw, types::I64);

                                    let var_idx = builder.declare_var(types::I64);
                                    let var_chunks = builder.declare_var(types::I64);

                                    let null_err_bb = builder.create_block();
                                    let check_mode_bb = builder.create_block();
                                    let vector_bb = builder.create_block();
                                    let vector_loop_hdr = builder.create_block();
                                    let vector_loop_body = builder.create_block();
                                    let to_tail_bb = builder.create_block();
                                    let vector_tail_hdr = builder.create_block();
                                    let vector_tail_body = builder.create_block();
                                    let scalar_fwd_hdr = builder.create_block();
                                    let scalar_fwd_body = builder.create_block();
                                    let scalar_bwd_hdr = builder.create_block();
                                    let scalar_bwd_body = builder.create_block();
                                    let done_success_bb = builder.create_block();
                                    let next_mir_bb = resolve_target_bb(Some(b_idx), success.0)?;

                                    // Null checks: a == 0 || b == 0 || out == 0
                                    let a_is_zero = builder.ins().icmp_imm_s(IntCC::Equal, a_val, 0);
                                    let b_is_zero = builder.ins().icmp_imm_s(IntCC::Equal, b_val, 0);
                                    let out_is_zero = builder.ins().icmp_imm_s(IntCC::Equal, out_val, 0);
                                    let ab_zero = builder.ins().bor(a_is_zero, b_is_zero);
                                    let any_zero = builder.ins().bor(ab_zero, out_is_zero);
                                    builder.ins().brif(any_zero, null_err_bb, &[], check_mode_bb, &[]);

                                    // Null error block: return 0
                                    builder.switch_to_block(null_err_bb);
                                    let zero_res = builder.ins().iconst(types::I64, 0);
                                    if let Place::Local(loc) = destination {
                                        let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                        let res_cast = cast_clif_type(&mut builder, zero_res, target_ty);
                                        builder.def_var(local_vars[loc.0], res_cast);
                                        if let Some(&slot) = stack_slots.get(&loc.0) {
                                            builder.ins().stack_store(types::I64, res_cast, slot, 0);
                                        }
                                    }
                                    builder.ins().jump(next_mir_bb, &[]);

                                    // Check mode block: alias hazard detection & vector threshold
                                    builder.switch_to_block(check_mode_bb);
                                    let len_le_zero = builder.ins().icmp_imm_s(IntCC::SignedLessThanOrEqual, len_val, 0);
                                    let normal_exec_bb = builder.create_block();
                                    builder.ins().brif(len_le_zero, done_success_bb, &[], normal_exec_bb, &[]);

                                    builder.switch_to_block(normal_exec_bb);
                                    let eight = builder.ins().iconst(types::I64, 8);
                                    let len_bytes = builder.ins().imul(len_val, eight);
                                    let a_end = builder.ins().iadd(a_val, len_bytes);
                                    let b_end = builder.ins().iadd(b_val, len_bytes);
                                    let out_end = builder.ins().iadd(out_val, len_bytes);

                                    // Operand A shifted overlap & directional hazards:
                                    let out_lt_a_end = builder.ins().icmp(IntCC::UnsignedLessThan, out_val, a_end);
                                    let a_lt_out_end = builder.ins().icmp(IntCC::UnsignedLessThan, a_val, out_end);
                                    let a_overlaps = builder.ins().band(out_lt_a_end, a_lt_out_end);
                                    let a_diff = builder.ins().icmp(IntCC::NotEqual, out_val, a_val);
                                    let a_shifted = builder.ins().band(a_overlaps, a_diff);

                                    let out_gt_a = builder.ins().icmp(IntCC::UnsignedGreaterThan, out_val, a_val);
                                    let a_fwd_hazard = builder.ins().band(a_shifted, out_gt_a);
                                    let out_lt_a = builder.ins().icmp(IntCC::UnsignedLessThan, out_val, a_val);
                                    let a_bwd_hazard = builder.ins().band(a_shifted, out_lt_a);

                                    // Operand B shifted overlap & directional hazards:
                                    let out_lt_b_end = builder.ins().icmp(IntCC::UnsignedLessThan, out_val, b_end);
                                    let b_lt_out_end = builder.ins().icmp(IntCC::UnsignedLessThan, b_val, out_end);
                                    let b_overlaps = builder.ins().band(out_lt_b_end, b_lt_out_end);
                                    let b_diff = builder.ins().icmp(IntCC::NotEqual, out_val, b_val);
                                    let b_shifted = builder.ins().band(b_overlaps, b_diff);

                                    let out_gt_b = builder.ins().icmp(IntCC::UnsignedGreaterThan, out_val, b_val);
                                    let b_fwd_hazard = builder.ins().band(b_shifted, out_gt_b);
                                    let out_lt_b = builder.ins().icmp(IntCC::UnsignedLessThan, out_val, b_val);
                                    let b_bwd_hazard = builder.ins().band(b_shifted, out_lt_b);

                                    // Contradictory overlap: forward shift on one operand and backward shift on another.
                                    // Reject atomically by branching to null_err_bb (returns 0 with 0 writes).
                                    let has_fwd_hazard = builder.ins().bor(a_fwd_hazard, b_fwd_hazard);
                                    let has_bwd_hazard = builder.ins().bor(a_bwd_hazard, b_bwd_hazard);
                                    let is_contradictory = builder.ins().band(has_fwd_hazard, has_bwd_hazard);

                                    let check_contradictory_bb = builder.create_block();
                                    let fwd_hazard_bb = builder.create_block();
                                    let bwd_or_clean_bb = builder.create_block();
                                    let fwd_scalar_bb = builder.create_block();

                                    builder.ins().brif(is_contradictory, null_err_bb, &[], check_contradictory_bb, &[]);

                                    builder.switch_to_block(check_contradictory_bb);
                                    builder.ins().brif(has_fwd_hazard, fwd_hazard_bb, &[], bwd_or_clean_bb, &[]);

                                    // Forward hazard path (out > inputs): requires backward scalar traversal
                                    builder.switch_to_block(fwd_hazard_bb);
                                    builder.ins().call(simd_scalar_fn, &[]);
                                    let len_minus_one = builder.ins().iadd_imm_s(len_val, -1);
                                    builder.def_var(var_idx, len_minus_one);
                                    builder.ins().jump(scalar_bwd_hdr, &[]);

                                    // Backward hazard or clean path
                                    builder.switch_to_block(bwd_or_clean_bb);
                                    let is_force = builder.ins().icmp_imm_s(IntCC::NotEqual, force_val, 0);
                                    let is_short = builder.ins().icmp_imm_s(IntCC::SignedLessThan, len_val, 2);
                                    let force_or_short = builder.ins().bor(is_force, is_short);
                                    let need_fwd_scalar = builder.ins().bor(has_bwd_hazard, force_or_short);
                                    builder.ins().brif(need_fwd_scalar, fwd_scalar_bb, &[], vector_bb, &[]);

                                    builder.switch_to_block(fwd_scalar_bb);
                                    builder.ins().call(simd_scalar_fn, &[]);
                                    let zero = builder.ins().iconst(types::I64, 0);
                                    builder.def_var(var_idx, zero);
                                    builder.ins().jump(scalar_fwd_hdr, &[]);

                                    // Vector init
                                    builder.switch_to_block(vector_bb);
                                    builder.ins().call(simd_vector_fn, &[]);
                                    let zero = builder.ins().iconst(types::I64, 0);
                                    let two = builder.ins().iconst(types::I64, 2);
                                    let chunks = builder.ins().sdiv(len_val, two);
                                    builder.def_var(var_chunks, chunks);
                                    builder.def_var(var_idx, zero);
                                    builder.ins().jump(vector_loop_hdr, &[]);

                                    // Vector loop header
                                    builder.switch_to_block(vector_loop_hdr);
                                    let c = builder.use_var(var_idx);
                                    let max_c = builder.use_var(var_chunks);
                                    let vec_cond = builder.ins().icmp(IntCC::SignedLessThan, c, max_c);
                                    builder.ins().brif(vec_cond, vector_loop_body, &[], to_tail_bb, &[]);

                                    builder.switch_to_block(to_tail_bb);
                                    let tail_start = builder.ins().imul(max_c, two);
                                    builder.def_var(var_idx, tail_start);
                                    builder.ins().jump(vector_tail_hdr, &[]);

                                    // Vector loop body: load I64X2, extract lanes, imul, store
                                    builder.switch_to_block(vector_loop_body);
                                    let sixteen = builder.ins().iconst(types::I64, 16);
                                    let byte_offset = builder.ins().imul(c, sixteen);
                                    let cur_a = builder.ins().iadd(a_val, byte_offset);
                                    let cur_b = builder.ins().iadd(b_val, byte_offset);
                                    let cur_out = builder.ins().iadd(out_val, byte_offset);
                                    let va = builder.ins().load(types::I64X2, MemFlagsData::new(), cur_a, 0);
                                    let vb = builder.ins().load(types::I64X2, MemFlagsData::new(), cur_b, 0);
                                    let l0_a = builder.ins().extractlane(va, 0);
                                    let l1_a = builder.ins().extractlane(va, 1);
                                    let l0_b = builder.ins().extractlane(vb, 0);
                                    let l1_b = builder.ins().extractlane(vb, 1);
                                    let p0 = builder.ins().imul(l0_a, l0_b);
                                    let p1 = builder.ins().imul(l1_a, l1_b);
                                    builder.ins().store(MemFlagsData::new(), p0, cur_out, 0);
                                    builder.ins().store(MemFlagsData::new(), p1, cur_out, 8);
                                    let next_c = builder.ins().iadd_imm_s(c, 1);
                                    builder.def_var(var_idx, next_c);
                                    builder.ins().jump(vector_loop_hdr, &[]);

                                    // Vector tail header
                                    builder.switch_to_block(vector_tail_hdr);
                                    let tail_idx = builder.use_var(var_idx);
                                    let tail_cond = builder.ins().icmp(IntCC::SignedLessThan, tail_idx, len_val);
                                    builder.ins().brif(tail_cond, vector_tail_body, &[], done_success_bb, &[]);

                                    // Vector tail body
                                    builder.switch_to_block(vector_tail_body);
                                    let tail_offset = builder.ins().imul(tail_idx, eight);
                                    let ta = builder.ins().iadd(a_val, tail_offset);
                                    let tb = builder.ins().iadd(b_val, tail_offset);
                                    let tout = builder.ins().iadd(out_val, tail_offset);
                                    let ea = builder.ins().load(types::I64, MemFlagsData::new(), ta, 0);
                                    let eb = builder.ins().load(types::I64, MemFlagsData::new(), tb, 0);
                                    let eres = builder.ins().imul(ea, eb);
                                    builder.ins().store(MemFlagsData::new(), eres, tout, 0);
                                    let next_tail = builder.ins().iadd_imm_s(tail_idx, 1);
                                    builder.def_var(var_idx, next_tail);
                                    builder.ins().jump(vector_tail_hdr, &[]);

                                    // Forward scalar loop header and body
                                    builder.switch_to_block(scalar_fwd_hdr);
                                    let cur_fwd = builder.use_var(var_idx);
                                    let fwd_cond = builder.ins().icmp(IntCC::SignedLessThan, cur_fwd, len_val);
                                    builder.ins().brif(fwd_cond, scalar_fwd_body, &[], done_success_bb, &[]);

                                    builder.switch_to_block(scalar_fwd_body);
                                    let fwd_offset = builder.ins().imul(cur_fwd, eight);
                                    let fa = builder.ins().iadd(a_val, fwd_offset);
                                    let fb = builder.ins().iadd(b_val, fwd_offset);
                                    let fout = builder.ins().iadd(out_val, fwd_offset);
                                    let ea = builder.ins().load(types::I64, MemFlagsData::new(), fa, 0);
                                    let eb = builder.ins().load(types::I64, MemFlagsData::new(), fb, 0);
                                    let eres = builder.ins().imul(ea, eb);
                                    builder.ins().store(MemFlagsData::new(), eres, fout, 0);
                                    let next_fwd = builder.ins().iadd_imm_s(cur_fwd, 1);
                                    builder.def_var(var_idx, next_fwd);
                                    builder.ins().jump(scalar_fwd_hdr, &[]);

                                    // Backward scalar loop header and body
                                    builder.switch_to_block(scalar_bwd_hdr);
                                    let cur_bwd = builder.use_var(var_idx);
                                    let bwd_cond = builder.ins().icmp_imm_s(IntCC::SignedGreaterThanOrEqual, cur_bwd, 0);
                                    builder.ins().brif(bwd_cond, scalar_bwd_body, &[], done_success_bb, &[]);

                                    builder.switch_to_block(scalar_bwd_body);
                                    let bwd_offset = builder.ins().imul(cur_bwd, eight);
                                    let ba = builder.ins().iadd(a_val, bwd_offset);
                                    let bb = builder.ins().iadd(b_val, bwd_offset);
                                    let bout = builder.ins().iadd(out_val, bwd_offset);
                                    let ea = builder.ins().load(types::I64, MemFlagsData::new(), ba, 0);
                                    let eb = builder.ins().load(types::I64, MemFlagsData::new(), bb, 0);
                                    let eres = builder.ins().imul(ea, eb);
                                    builder.ins().store(MemFlagsData::new(), eres, bout, 0);
                                    let next_bwd = builder.ins().iadd_imm_s(cur_bwd, -1);
                                    builder.def_var(var_idx, next_bwd);
                                    builder.ins().jump(scalar_bwd_hdr, &[]);

                                    // Done success
                                    builder.switch_to_block(done_success_bb);
                                    let one_res = builder.ins().iconst(types::I64, 1);
                                    if let Place::Local(loc) = destination {
                                        let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                        let res_cast = cast_clif_type(&mut builder, one_res, target_ty);
                                        builder.def_var(local_vars[loc.0], res_cast);
                                        if let Some(&slot) = stack_slots.get(&loc.0) {
                                            builder.ins().stack_store(types::I64, res_cast, slot, 0);
                                        }
                                    }
                                    builder.ins().jump(next_mir_bb, &[]);
                                } else if callee == "simd.sum" || callee == "simd_sum" || callee == "vector.sum" {
                                    let simd_vector_fn = simd_vector_fn.expect("has_simd implies simd_vector_fn");
                                    let simd_scalar_fn = simd_scalar_fn.expect("has_simd implies simd_scalar_fn");
                                    let a_val = lower_operand(&args[0], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let len_val = lower_operand(&args[1], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let force_raw = lower_operand(&args[2], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let force_val = cast_clif_type(&mut builder, force_raw, types::I64);

                                    let var_idx = builder.declare_var(types::I64);
                                    let var_chunks = builder.declare_var(types::I64);
                                    let var_vacc = builder.declare_var(types::I64X2);
                                    let var_sum = builder.declare_var(types::I64);

                                    let zero = builder.ins().iconst(types::I64, 0);
                                    let null_or_empty_bb = builder.create_block();
                                    let check_mode_bb = builder.create_block();
                                    let vector_bb = builder.create_block();
                                    let vector_loop_hdr = builder.create_block();
                                    let vector_loop_body = builder.create_block();
                                    let vector_reduce_bb = builder.create_block();
                                    let vector_tail_hdr = builder.create_block();
                                    let vector_tail_body = builder.create_block();
                                    let scalar_hdr = builder.create_block();
                                    let scalar_body = builder.create_block();
                                    let done_bb = builder.create_block();
                                    let next_mir_bb = resolve_target_bb(Some(b_idx), success.0)?;

                                    let a_is_zero = builder.ins().icmp_imm_s(IntCC::Equal, a_val, 0);
                                    let len_le_zero = builder.ins().icmp_imm_s(IntCC::SignedLessThanOrEqual, len_val, 0);
                                    let null_or_empty = builder.ins().bor(a_is_zero, len_le_zero);
                                    builder.ins().brif(null_or_empty, null_or_empty_bb, &[], check_mode_bb, &[]);

                                    // Null or empty: return 0
                                    builder.switch_to_block(null_or_empty_bb);
                                    if let Place::Local(loc) = destination {
                                        let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                        let res_cast = cast_clif_type(&mut builder, zero, target_ty);
                                        builder.def_var(local_vars[loc.0], res_cast);
                                        if let Some(&slot) = stack_slots.get(&loc.0) {
                                            builder.ins().stack_store(types::I64, res_cast, slot, 0);
                                        }
                                    }
                                    builder.ins().jump(next_mir_bb, &[]);

                                    // Check mode
                                    builder.switch_to_block(check_mode_bb);
                                    let is_force = builder.ins().icmp_imm_s(IntCC::NotEqual, force_val, 0);
                                    let is_short = builder.ins().icmp_imm_s(IntCC::SignedLessThan, len_val, 2);
                                    let force_or_short = builder.ins().bor(is_force, is_short);
                                    let scalar_init_bb = builder.create_block();
                                    builder.ins().brif(force_or_short, scalar_init_bb, &[], vector_bb, &[]);

                                    builder.switch_to_block(scalar_init_bb);
                                    builder.ins().call(simd_scalar_fn, &[]);
                                    builder.def_var(var_idx, zero);
                                    builder.def_var(var_sum, zero);
                                    builder.ins().jump(scalar_hdr, &[]);

                                    // Vector init
                                    builder.switch_to_block(vector_bb);
                                    builder.ins().call(simd_vector_fn, &[]);
                                    let two = builder.ins().iconst(types::I64, 2);
                                    let chunks = builder.ins().sdiv(len_val, two);
                                    let v_zero = builder.ins().splat(types::I64X2, zero);
                                    builder.def_var(var_chunks, chunks);
                                    builder.def_var(var_idx, zero);
                                    builder.def_var(var_vacc, v_zero);
                                    builder.ins().jump(vector_loop_hdr, &[]);

                                    // Vector loop header
                                    builder.switch_to_block(vector_loop_hdr);
                                    let c = builder.use_var(var_idx);
                                    let max_c = builder.use_var(var_chunks);
                                    let cond = builder.ins().icmp(IntCC::SignedLessThan, c, max_c);
                                    builder.ins().brif(cond, vector_loop_body, &[], vector_reduce_bb, &[]);

                                    // Vector loop body
                                    builder.switch_to_block(vector_loop_body);
                                    let sixteen = builder.ins().iconst(types::I64, 16);
                                    let offset = builder.ins().imul(c, sixteen);
                                    let cur_a = builder.ins().iadd(a_val, offset);
                                    let va = builder.ins().load(types::I64X2, MemFlagsData::new(), cur_a, 0);
                                    let v_acc = builder.use_var(var_vacc);
                                    let next_vacc = builder.ins().iadd(v_acc, va);
                                    builder.def_var(var_vacc, next_vacc);
                                    let next_c = builder.ins().iadd_imm_s(c, 1);
                                    builder.def_var(var_idx, next_c);
                                    builder.ins().jump(vector_loop_hdr, &[]);

                                    // Vector reduce block: extract lanes and sum
                                    builder.switch_to_block(vector_reduce_bb);
                                    let final_vacc = builder.use_var(var_vacc);
                                    let l0 = builder.ins().extractlane(final_vacc, 0);
                                    let l1 = builder.ins().extractlane(final_vacc, 1);
                                    let init_sum = builder.ins().iadd(l0, l1);
                                    builder.def_var(var_sum, init_sum);
                                    let max_c = builder.use_var(var_chunks);
                                    let tail_start = builder.ins().imul(max_c, two);
                                    builder.def_var(var_idx, tail_start);
                                    builder.ins().jump(vector_tail_hdr, &[]);

                                    // Vector tail header
                                    builder.switch_to_block(vector_tail_hdr);
                                    let t_idx = builder.use_var(var_idx);
                                    let t_cond = builder.ins().icmp(IntCC::SignedLessThan, t_idx, len_val);
                                    builder.ins().brif(t_cond, vector_tail_body, &[], done_bb, &[]);

                                    // Vector tail body
                                    builder.switch_to_block(vector_tail_body);
                                    let eight = builder.ins().iconst(types::I64, 8);
                                    let t_offset = builder.ins().imul(t_idx, eight);
                                    let cur_ta = builder.ins().iadd(a_val, t_offset);
                                    let ea = builder.ins().load(types::I64, MemFlagsData::new(), cur_ta, 0);
                                    let cur_sum = builder.use_var(var_sum);
                                    let next_tsum = builder.ins().iadd(cur_sum, ea);
                                    builder.def_var(var_sum, next_tsum);
                                    let next_tidx = builder.ins().iadd_imm_s(t_idx, 1);
                                    builder.def_var(var_idx, next_tidx);
                                    builder.ins().jump(vector_tail_hdr, &[]);

                                    // Scalar header
                                    builder.switch_to_block(scalar_hdr);
                                    let s_idx = builder.use_var(var_idx);
                                    let s_cond = builder.ins().icmp(IntCC::SignedLessThan, s_idx, len_val);
                                    builder.ins().brif(s_cond, scalar_body, &[], done_bb, &[]);

                                    // Scalar body
                                    builder.switch_to_block(scalar_body);
                                    let cur_s_idx = builder.use_var(var_idx);
                                    let eight = builder.ins().iconst(types::I64, 8);
                                    let s_offset = builder.ins().imul(cur_s_idx, eight);
                                    let cur_sa = builder.ins().iadd(a_val, s_offset);
                                    let ea = builder.ins().load(types::I64, MemFlagsData::new(), cur_sa, 0);
                                    let cur_sum = builder.use_var(var_sum);
                                    let next_ssum = builder.ins().iadd(cur_sum, ea);
                                    builder.def_var(var_sum, next_ssum);
                                    let next_sidx = builder.ins().iadd_imm_s(cur_s_idx, 1);
                                    builder.def_var(var_idx, next_sidx);
                                    let s_check = builder.ins().icmp(IntCC::SignedLessThan, next_sidx, len_val);
                                    builder.ins().brif(s_check, scalar_body, &[], done_bb, &[]);

                                    // Done
                                    builder.switch_to_block(done_bb);
                                    let final_sum = builder.use_var(var_sum);
                                    if let Place::Local(loc) = destination {
                                        let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                        let res_cast = cast_clif_type(&mut builder, final_sum, target_ty);
                                        builder.def_var(local_vars[loc.0], res_cast);
                                        if let Some(&slot) = stack_slots.get(&loc.0) {
                                            builder.ins().stack_store(types::I64, res_cast, slot, 0);
                                        }
                                    }
                                    builder.ins().jump(next_mir_bb, &[]);
                                } else if callee == "simd.dot" || callee == "simd_dot" || callee == "vector.dot" {
                                    let simd_vector_fn = simd_vector_fn.expect("has_simd implies simd_vector_fn");
                                    let simd_scalar_fn = simd_scalar_fn.expect("has_simd implies simd_scalar_fn");
                                    let a_val = lower_operand(&args[0], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let b_val = lower_operand(&args[1], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let len_val = lower_operand(&args[2], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let force_raw = lower_operand(&args[3], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let force_val = cast_clif_type(&mut builder, force_raw, types::I64);

                                    let var_idx = builder.declare_var(types::I64);
                                    let var_chunks = builder.declare_var(types::I64);
                                    let var_dot = builder.declare_var(types::I64);

                                    let zero = builder.ins().iconst(types::I64, 0);
                                    let null_or_empty_bb = builder.create_block();
                                    let check_mode_bb = builder.create_block();
                                    let vector_bb = builder.create_block();
                                    let vector_loop_hdr = builder.create_block();
                                    let vector_loop_body = builder.create_block();
                                    let to_tail_bb = builder.create_block();
                                    let vector_tail_hdr = builder.create_block();
                                    let vector_tail_body = builder.create_block();
                                    let scalar_hdr = builder.create_block();
                                    let scalar_body = builder.create_block();
                                    let done_bb = builder.create_block();
                                    let next_mir_bb = resolve_target_bb(Some(b_idx), success.0)?;

                                    let a_is_zero = builder.ins().icmp_imm_s(IntCC::Equal, a_val, 0);
                                    let b_is_zero = builder.ins().icmp_imm_s(IntCC::Equal, b_val, 0);
                                    let any_null = builder.ins().bor(a_is_zero, b_is_zero);
                                    let len_le_zero = builder.ins().icmp_imm_s(IntCC::SignedLessThanOrEqual, len_val, 0);
                                    let null_or_empty = builder.ins().bor(any_null, len_le_zero);
                                    builder.ins().brif(null_or_empty, null_or_empty_bb, &[], check_mode_bb, &[]);

                                    // Null or empty: return 0
                                    builder.switch_to_block(null_or_empty_bb);
                                    if let Place::Local(loc) = destination {
                                        let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                        let res_cast = cast_clif_type(&mut builder, zero, target_ty);
                                        builder.def_var(local_vars[loc.0], res_cast);
                                        if let Some(&slot) = stack_slots.get(&loc.0) {
                                            builder.ins().stack_store(types::I64, res_cast, slot, 0);
                                        }
                                    }
                                    builder.ins().jump(next_mir_bb, &[]);

                                    // Check mode
                                    builder.switch_to_block(check_mode_bb);
                                    let is_force = builder.ins().icmp_imm_s(IntCC::NotEqual, force_val, 0);
                                    let is_short = builder.ins().icmp_imm_s(IntCC::SignedLessThan, len_val, 2);
                                    let force_or_short = builder.ins().bor(is_force, is_short);
                                    let scalar_init_bb = builder.create_block();
                                    builder.ins().brif(force_or_short, scalar_init_bb, &[], vector_bb, &[]);

                                    builder.switch_to_block(scalar_init_bb);
                                    builder.ins().call(simd_scalar_fn, &[]);
                                    builder.def_var(var_idx, zero);
                                    builder.def_var(var_dot, zero);
                                    builder.ins().jump(scalar_hdr, &[]);

                                    // Vector init
                                    builder.switch_to_block(vector_bb);
                                    builder.ins().call(simd_vector_fn, &[]);
                                    let two = builder.ins().iconst(types::I64, 2);
                                    let chunks = builder.ins().sdiv(len_val, two);
                                    builder.def_var(var_chunks, chunks);
                                    builder.def_var(var_idx, zero);
                                    builder.def_var(var_dot, zero);
                                    builder.ins().jump(vector_loop_hdr, &[]);

                                    // Vector loop header
                                    builder.switch_to_block(vector_loop_hdr);
                                    let c = builder.use_var(var_idx);
                                    let max_c = builder.use_var(var_chunks);
                                    let cond = builder.ins().icmp(IntCC::SignedLessThan, c, max_c);
                                    builder.ins().brif(cond, vector_loop_body, &[], to_tail_bb, &[]);

                                    builder.switch_to_block(to_tail_bb);
                                    let tail_start = builder.ins().imul(max_c, two);
                                    builder.def_var(var_idx, tail_start);
                                    builder.ins().jump(vector_tail_hdr, &[]);

                                    // Vector loop body
                                    builder.switch_to_block(vector_loop_body);
                                    let sixteen = builder.ins().iconst(types::I64, 16);
                                    let offset = builder.ins().imul(c, sixteen);
                                    let cur_a = builder.ins().iadd(a_val, offset);
                                    let cur_b = builder.ins().iadd(b_val, offset);
                                    let va = builder.ins().load(types::I64X2, MemFlagsData::new(), cur_a, 0);
                                    let vb = builder.ins().load(types::I64X2, MemFlagsData::new(), cur_b, 0);
                                    let l0_a = builder.ins().extractlane(va, 0);
                                    let l1_a = builder.ins().extractlane(va, 1);
                                    let l0_b = builder.ins().extractlane(vb, 0);
                                    let l1_b = builder.ins().extractlane(vb, 1);
                                    let p0 = builder.ins().imul(l0_a, l0_b);
                                    let p1 = builder.ins().imul(l1_a, l1_b);
                                    let p_sum = builder.ins().iadd(p0, p1);
                                    let cur_dot = builder.use_var(var_dot);
                                    let next_dot = builder.ins().iadd(cur_dot, p_sum);
                                    builder.def_var(var_dot, next_dot);
                                    let next_c = builder.ins().iadd_imm_s(c, 1);
                                    builder.def_var(var_idx, next_c);
                                    builder.ins().jump(vector_loop_hdr, &[]);

                                    // Vector tail header
                                    builder.switch_to_block(vector_tail_hdr);
                                    let t_idx = builder.use_var(var_idx);
                                    let t_cond = builder.ins().icmp(IntCC::SignedLessThan, t_idx, len_val);
                                    builder.ins().brif(t_cond, vector_tail_body, &[], done_bb, &[]);

                                    // Vector tail body
                                    builder.switch_to_block(vector_tail_body);
                                    let eight = builder.ins().iconst(types::I64, 8);
                                    let t_offset = builder.ins().imul(t_idx, eight);
                                    let cur_ta = builder.ins().iadd(a_val, t_offset);
                                    let cur_tb = builder.ins().iadd(b_val, t_offset);
                                    let ea = builder.ins().load(types::I64, MemFlagsData::new(), cur_ta, 0);
                                    let eb = builder.ins().load(types::I64, MemFlagsData::new(), cur_tb, 0);
                                    let prod = builder.ins().imul(ea, eb);
                                    let cur_dot = builder.use_var(var_dot);
                                    let next_tacc = builder.ins().iadd(cur_dot, prod);
                                    builder.def_var(var_dot, next_tacc);
                                    let next_tidx = builder.ins().iadd_imm_s(t_idx, 1);
                                    builder.def_var(var_idx, next_tidx);
                                    builder.ins().jump(vector_tail_hdr, &[]);

                                    // Scalar header
                                    builder.switch_to_block(scalar_hdr);
                                    let s_idx = builder.use_var(var_idx);
                                    let s_cond = builder.ins().icmp(IntCC::SignedLessThan, s_idx, len_val);
                                    builder.ins().brif(s_cond, scalar_body, &[], done_bb, &[]);

                                    // Scalar body
                                    builder.switch_to_block(scalar_body);
                                    let cur_s_idx = builder.use_var(var_idx);
                                    let eight = builder.ins().iconst(types::I64, 8);
                                    let s_offset = builder.ins().imul(cur_s_idx, eight);
                                    let cur_sa = builder.ins().iadd(a_val, s_offset);
                                    let cur_sb = builder.ins().iadd(b_val, s_offset);
                                    let ea = builder.ins().load(types::I64, MemFlagsData::new(), cur_sa, 0);
                                    let eb = builder.ins().load(types::I64, MemFlagsData::new(), cur_sb, 0);
                                    let prod = builder.ins().imul(ea, eb);
                                    let cur_dot = builder.use_var(var_dot);
                                    let next_sacc = builder.ins().iadd(cur_dot, prod);
                                    builder.def_var(var_dot, next_sacc);
                                    let next_sidx = builder.ins().iadd_imm_s(cur_s_idx, 1);
                                    builder.def_var(var_idx, next_sidx);
                                    let s_check = builder.ins().icmp(IntCC::SignedLessThan, next_sidx, len_val);
                                    builder.ins().brif(s_check, scalar_body, &[], done_bb, &[]);

                                    // Done
                                    builder.switch_to_block(done_bb);
                                    let final_res = builder.use_var(var_dot);
                                    if let Place::Local(loc) = destination {
                                        let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                        let res_cast = cast_clif_type(&mut builder, final_res, target_ty);
                                        builder.def_var(local_vars[loc.0], res_cast);
                                        if let Some(&slot) = stack_slots.get(&loc.0) {
                                            builder.ins().stack_store(types::I64, res_cast, slot, 0);
                                        }
                                    }
                                    let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                    builder.ins().jump(succ_bb, &[]);
                                } else if callee == "array.len" || callee == "table.len" {
                                    if args.len() != 1 {
                                        return Err(MirCraneliftBackendError::CodegenError(format!(
                                            "Call to '{callee}' arity mismatch: got {} arguments, expected 1",
                                            args.len()
                                        )));
                                    }
                                    let arg_ty = try_operand_type(&args[0], &mir_fn.locals, &self.struct_layouts)?;
                                    if !matches!(arg_ty, MirType::Array(_)) {
                                        return Err(MirCraneliftBackendError::CodegenError(format!(
                                            "Call to '{callee}' requires Array argument, got '{arg_ty}'"
                                        )));
                                    }
                                    let arr_val = lower_operand(
                                        &args[0],
                                        &mut builder,
                                        &local_vars,
                                        &stack_slots,
                                        &mir_fn.locals,
                                        &self.struct_layouts,
                                        bc_ctx,
                                    )?;
                                    let res_val = emit_array_len(
                                        &args[0],
                                        arr_val,
                                        &mut builder,
                                        len_fn,
                                        emergency_exit_bb,
                                        &mut cur_validated,
                                    );
                                    match destination {
                                        Place::Local(loc) => {
                                            let loc_decl = mir_fn.locals.get(loc.0).ok_or_else(|| {
                                                MirCraneliftBackendError::CodegenError(format!(
                                                    "Call destination local {loc:?} out of bounds"
                                                ))
                                            })?;
                                            let target_ty = mir_to_clif_type(&loc_decl.ty)?;
                                            let res_val = cast_clif_type(&mut builder, res_val, target_ty);
                                            store_local_val(*loc, res_val, &mut builder, &local_vars, &stack_slots)?;
                                        }
                                        Place::Index(..) | Place::Field(..) => {
                                            let dst_addr = lower_place_addr(
                                                destination,
                                                &mut builder,
                                                &local_vars,
                                                &stack_slots,
                                                &mir_fn.locals,
                                                &self.struct_layouts,
                                                bc_ctx,
                                            )?;
                                            builder.ins().store(MemFlagsData::trusted(), res_val, dst_addr, 0);
                                        }
                                    }
                                    let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                    builder.ins().jump(succ_bb, &[]);
                                } else if callee == "array.push" || callee == "table.push" {
                                    let array_push_fn = Self::declare_ext_fn(
                                        &mut self.module,
                                        &mut builder.func,
                                        &mut ext_fn_cache,
                                        "lust_mir_array_push",
                                        &[types::I64, types::I64],
                                        Some(types::I64),
                                    )?;
                                    if args.len() != 2 {
                                        return Err(MirCraneliftBackendError::CodegenError(format!(
                                            "Call to '{callee}' arity mismatch: got {} arguments, expected 2",
                                            args.len()
                                        )));
                                    }
                                    let arr_ty = try_operand_type(&args[0], &mir_fn.locals, &self.struct_layouts)?;
                                    let MirType::Array(elem_ty) = arr_ty else {
                                        return Err(MirCraneliftBackendError::CodegenError(format!(
                                            "Call to '{callee}' requires Array first argument, got '{arr_ty}'"
                                        )));
                                    };
                                    if mir_array_elem_tag(&elem_ty).is_none()
                                        && !matches!(&*elem_ty, MirType::Struct(name) if !name.is_empty())
                                    {
                                        return Err(MirCraneliftBackendError::UnsupportedType(*elem_ty));
                                    }
                                    let val_ty = try_operand_type(&args[1], &mir_fn.locals, &self.struct_layouts)?;
                                    if val_ty != *elem_ty {
                                        return Err(MirCraneliftBackendError::CodegenError(format!(
                                            "Call to '{callee}' element type mismatch: expected '{elem_ty}', got '{val_ty}'"
                                        )));
                                    }
                                    let arr_val = lower_operand(
                                        &args[0],
                                        &mut builder,
                                        &local_vars,
                                        &stack_slots,
                                        &mir_fn.locals,
                                        &self.struct_layouts,
                                        active_bc_ctx,
                                    )?;
                                    let val_raw = lower_operand(
                                        &args[1],
                                        &mut builder,
                                        &local_vars,
                                        &stack_slots,
                                        &mir_fn.locals,
                                        &self.struct_layouts,
                                        active_bc_ctx,
                                    )?;
                                    let val_clif_ty = builder.func.dfg.value_type(val_raw);
                                    if promo_plan.inline_push_blocks.contains(&b_idx) {
                                        // NG-W10-5 inline CLIF fast-path for locally-allocated arrays ONLY
                                        // (strictly excluded for function params / read_only_borrowed / unvalidated ptrs).
                                        let len_off = core::mem::offset_of!(MirArrayHeader, len) as i32;
                                        let cap_off = core::mem::offset_of!(MirArrayHeader, cap) as i32;
                                        let data_off = core::mem::size_of::<MirArrayHeader>() as i64;
                                        let len = builder
                                            .ins()
                                            .load(types::I64, MemFlagsData::trusted(), arr_val, len_off);
                                        let cap_raw = builder
                                            .ins()
                                            .load(types::I64, MemFlagsData::trusted(), arr_val, cap_off);
                                        let cap_mask = builder.ins().iconst(types::I64, i64::MAX);
                                        let actual_cap = builder.ins().band(cap_raw, cap_mask);
                                        let has_cap = builder
                                            .ins()
                                            .icmp(IntCC::UnsignedLessThan, len, actual_cap);

                                        let fast_push_bb = builder.create_block();
                                        let cold_grow_bb = builder.create_block();
                                        let cold_ok_bb = builder.create_block();
                                        builder.set_cold_block(cold_grow_bb);
                                        builder.set_cold_block(cold_ok_bb);
                                        builder
                                            .ins()
                                            .brif(has_cap, fast_push_bb, &[], cold_grow_bb, &[]);

                                        let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;

                                        // Hot fast-path: branchless flat/redirected data_ptr + element store + len += 1
                                        builder.switch_to_block(fast_push_bb);
                                        let is_redir =
                                            builder.ins().icmp_imm_s(IntCC::SignedLessThan, cap_raw, 0);
                                        let flat_ptr = builder.ins().iadd_imm_s(arr_val, data_off);
                                        // MirAllocationKind::Array always allocates >= 3 words, so word[2] is always readable.
                                        let word2 = builder.ins().load(
                                            types::I64,
                                            MemFlagsData::trusted(),
                                            arr_val,
                                            data_off as i32,
                                        );
                                        let data_ptr =
                                            builder.ins().select(is_redir, word2, flat_ptr);
                                        if *elem_ty == MirType::Bool {
                                            let elem_addr = builder.ins().iadd(data_ptr, len);
                                            let val_i8 = if val_clif_ty == types::I8 {
                                                val_raw
                                            } else {
                                                builder.ins().icmp_imm_s(IntCC::NotEqual, val_raw, 0)
                                            };
                                            builder.ins().store(
                                                MemFlagsData::trusted(),
                                                val_i8,
                                                elem_addr,
                                                0,
                                            );
                                        } else {
                                            let byte_off = builder.ins().ishl_imm_s(len, 3);
                                            let elem_addr = builder.ins().iadd(data_ptr, byte_off);
                                            builder.ins().store(
                                                MemFlagsData::trusted(),
                                                val_raw,
                                                elem_addr,
                                                0,
                                            );
                                        }
                                        let next_len = builder.ins().iadd_imm_s(len, 1);
                                        builder.ins().store(
                                            MemFlagsData::trusted(),
                                            next_len,
                                            arr_val,
                                            len_off,
                                        );
                                        if let Place::Local(loc) = destination {
                                            let loc_decl = mir_fn.locals.get(loc.0).ok_or_else(|| {
                                                MirCraneliftBackendError::CodegenError(format!(
                                                    "Call destination local {loc:?} out of bounds"
                                                ))
                                            })?;
                                            let target_ty = mir_to_clif_type(&loc_decl.ty)?;
                                            let zero = builder.ins().iconst(types::I64, 0);
                                            let zero_val = cast_clif_type(&mut builder, zero, target_ty);
                                            store_local_val(*loc, zero_val, &mut builder, &local_vars, &stack_slots)?;
                                        }
                                        builder.ins().jump(succ_bb, &[]);

                                        // Cold grow path: flush promoted counters (if in promoted loop), call lust_mir_array_push
                                        builder.switch_to_block(cold_grow_bb);
                                        if let Some(p_counters) = active_promoted {
                                            emit_promoted_counters_flush(&mut builder, p_counters);
                                        }
                                        if let Some(ci) = cl_opt {
                                            cfg_records[ci].flush_clif_blocks.insert(cold_grow_bb);
                                        }
                                        if let Some(iqi) = iq_opt {
                                            iq_cfg_records[iqi].reconcile_clif_blocks.insert(cold_grow_bb);
                                        }
                                        let val_i64 = if val_clif_ty == types::F64 {
                                            builder.ins().bitcast(types::I64, MemFlagsData::new(), val_raw)
                                        } else if val_clif_ty == types::I8 {
                                            builder.ins().uextend(types::I64, val_raw)
                                        } else {
                                            val_raw
                                        };
                                        let push_call = builder.ins().call(array_push_fn, &[arr_val, val_i64]);
                                        let push_status = builder.inst_results(push_call)[0];
                                        let is_err = builder.ins().icmp_imm_s(IntCC::NotEqual, push_status, 0);
                                        builder.ins().brif(is_err, emergency_exit_bb, &[], cold_ok_bb, &[]);

                                        // Cold reload & resume path
                                        builder.switch_to_block(cold_ok_bb);
                                        if let Some(p_counters) = active_promoted {
                                            emit_promoted_counters_load(&mut builder, p_counters);
                                        }
                                        if let Some(ci) = cl_opt {
                                            cfg_records[ci].reload_clif_blocks.insert(cold_ok_bb);
                                        }
                                        if let Some(iqi) = iq_opt {
                                            iq_cfg_records[iqi].reload_clif_blocks.insert(cold_ok_bb);
                                        }
                                        if let Place::Local(loc) = destination {
                                            let loc_decl = mir_fn.locals.get(loc.0).ok_or_else(|| {
                                                MirCraneliftBackendError::CodegenError(format!(
                                                    "Call destination local {loc:?} out of bounds"
                                                ))
                                            })?;
                                            let target_ty = mir_to_clif_type(&loc_decl.ty)?;
                                            let zero = builder.ins().iconst(types::I64, 0);
                                            let zero_val = cast_clif_type(&mut builder, zero, target_ty);
                                            store_local_val(*loc, zero_val, &mut builder, &local_vars, &stack_slots)?;
                                        }
                                        builder.ins().jump(succ_bb, &[]);
                                    } else {
                                        let val_i64 = if val_clif_ty == types::F64 {
                                            builder.ins().bitcast(types::I64, MemFlagsData::new(), val_raw)
                                        } else if val_clif_ty == types::I8 {
                                            builder.ins().uextend(types::I64, val_raw)
                                        } else {
                                            val_raw
                                        };
                                        let push_call = builder.ins().call(array_push_fn, &[arr_val, val_i64]);
                                        let push_status = builder.inst_results(push_call)[0];
                                        if let Place::Local(loc) = destination {
                                            let loc_decl = mir_fn.locals.get(loc.0).ok_or_else(|| {
                                                MirCraneliftBackendError::CodegenError(format!(
                                                    "Call destination local {loc:?} out of bounds"
                                                ))
                                            })?;
                                            let target_ty = mir_to_clif_type(&loc_decl.ty)?;
                                            let zero = builder.ins().iconst(types::I64, 0);
                                            let zero_val = cast_clif_type(&mut builder, zero, target_ty);
                                            store_local_val(*loc, zero_val, &mut builder, &local_vars, &stack_slots)?;
                                        }
                                        let is_err = builder.ins().icmp_imm_s(IntCC::NotEqual, push_status, 0);
                                        let push_ok_bb = builder.create_block();
                                        builder.ins().brif(is_err, emergency_exit_bb, &[], push_ok_bb, &[]);
                                        builder.switch_to_block(push_ok_bb);
                                        let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                        builder.ins().jump(succ_bb, &[]);
                                    }
                                } else if callee == "string.len" {
                                    if args.len() != 1 {
                                        return Err(MirCraneliftBackendError::CodegenError(format!(
                                            "Call to 'string.len' arity mismatch: got {} arguments, expected 1",
                                            args.len()
                                        )));
                                    }
                                    let arg_ty = try_operand_type(&args[0], &mir_fn.locals, &self.struct_layouts)?;
                                    if arg_ty != MirType::String {
                                        return Err(MirCraneliftBackendError::CodegenError(format!(
                                            "Call to 'string.len' requires String argument, got '{arg_ty}'"
                                        )));
                                    }
                                    let s_val = lower_operand(
                                        &args[0],
                                        &mut builder,
                                        &local_vars,
                                        &stack_slots,
                                        &mir_fn.locals,
                                        &self.struct_layouts,
                                        bc_ctx,
                                    )?;
                                    let len_call = builder.ins().call(string_len_fn, &[s_val]);
                                    let res_val = builder.inst_results(len_call)[0];
                                    match destination {
                                        Place::Local(loc) => {
                                            let loc_decl = mir_fn.locals.get(loc.0).ok_or_else(|| {
                                                MirCraneliftBackendError::CodegenError(format!(
                                                    "Call destination local {loc:?} out of bounds"
                                                ))
                                            })?;
                                            let target_ty = mir_to_clif_type(&loc_decl.ty)?;
                                            let res_val = cast_clif_type(&mut builder, res_val, target_ty);
                                            store_local_val(*loc, res_val, &mut builder, &local_vars, &stack_slots)?;
                                        }
                                        Place::Index(..) | Place::Field(..) => {
                                            let dst_addr = lower_place_addr(
                                                destination,
                                                &mut builder,
                                                &local_vars,
                                                &stack_slots,
                                                &mir_fn.locals,
                                                &self.struct_layouts,
                                                bc_ctx,
                                            )?;
                                            builder.ins().store(MemFlagsData::trusted(), res_val, dst_addr, 0);
                                        }
                                    }
                                    let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                    builder.ins().jump(succ_bb, &[]);
                                } else if callee == "tostring" {
                                    if args.len() != 1 {
                                        return Err(MirCraneliftBackendError::CodegenError(format!(
                                            "Call to 'tostring' arity mismatch: got {} arguments, expected 1",
                                            args.len()
                                        )));
                                    }
                                    let arg_ty = try_operand_type(&args[0], &mir_fn.locals, &self.struct_layouts)?;
                                    let arg_val = lower_operand(
                                        &args[0],
                                        &mut builder,
                                        &local_vars,
                                        &stack_slots,
                                        &mir_fn.locals,
                                        &self.struct_layouts,
                                        bc_ctx,
                                    )?;
                                    let res_val = lower_to_string_handle(
                                        &arg_ty,
                                        arg_val,
                                        &mut builder,
                                        string_helpers,
                                        status_fn,
                                        inline_status_ptr_var,
                                        emergency_exit_bb,
                                    )?;
                                    match destination {
                                        Place::Local(loc) => {
                                            let loc_decl = mir_fn.locals.get(loc.0).ok_or_else(|| {
                                                MirCraneliftBackendError::CodegenError(format!(
                                                    "Call destination local {loc:?} out of bounds"
                                                ))
                                            })?;
                                            let target_ty = mir_to_clif_type(&loc_decl.ty)?;
                                            let res_val = cast_clif_type(&mut builder, res_val, target_ty);
                                            store_local_val(*loc, res_val, &mut builder, &local_vars, &stack_slots)?;
                                        }
                                        Place::Index(..) | Place::Field(..) => {
                                            let dst_addr = lower_place_addr(
                                                destination,
                                                &mut builder,
                                                &local_vars,
                                                &stack_slots,
                                                &mir_fn.locals,
                                                &self.struct_layouts,
                                                bc_ctx,
                                            )?;
                                            builder.ins().store(MemFlagsData::trusted(), res_val, dst_addr, 0);
                                        }
                                    }
                                    let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                    builder.ins().jump(succ_bb, &[]);
                                } else if callee == "map.set" {
                                    let map_set_typed_fn = Self::declare_ext_fn(
                                        &mut self.module,
                                        &mut builder.func,
                                        &mut ext_fn_cache,
                                        "lust_mir_map_set_typed",
                                        &[types::I64; 6],
                                        Some(types::I64),
                                    )?;
                                    if args.len() != 3 {
                                        return Err(MirCraneliftBackendError::CodegenError(format!(
                                            "Call to 'map.set' arity mismatch: got {} arguments, expected 3",
                                            args.len()
                                        )));
                                    }
                                    let key_ty = try_operand_type(&args[1], &mir_fn.locals, &self.struct_layouts)?;
                                    let val_ty = try_operand_type(&args[2], &mir_fn.locals, &self.struct_layouts)?;
                                    let map_val = lower_operand(&args[0], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let key_val = lower_operand(&args[1], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let val_val = lower_operand(&args[2], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let key_raw = clif_val_to_raw_i64(&mut builder, key_val);
                                    let val_raw = clif_val_to_raw_i64(&mut builder, val_val);
                                    let zero_ctx = builder.ins().iconst(types::I64, 0);
                                    let key_tag = builder.ins().iconst(types::I64, mir_type_to_map_operand_tag(&key_ty));
                                    let val_tag = builder.ins().iconst(types::I64, mir_type_to_map_operand_tag(&val_ty));
                                    let set_call = builder.ins().call(
                                        map_set_typed_fn,
                                        &[zero_ctx, map_val, key_raw, val_raw, key_tag, val_tag],
                                    );
                                    let set_status = builder.inst_results(set_call)[0];
                                    if let Place::Local(loc) = destination {
                                        let loc_decl = mir_fn.locals.get(loc.0).ok_or_else(|| {
                                            MirCraneliftBackendError::CodegenError(format!(
                                                "Call destination local {loc:?} out of bounds"
                                            ))
                                        })?;
                                        let target_ty = mir_to_clif_type(&loc_decl.ty)?;
                                        let zero = builder.ins().iconst(types::I64, 0);
                                        let zero_val = cast_clif_type(&mut builder, zero, target_ty);
                                        store_local_val(*loc, zero_val, &mut builder, &local_vars, &stack_slots)?;
                                    }
                                    let is_err = builder.ins().icmp_imm_s(IntCC::NotEqual, set_status, 0);
                                    let ok_bb = builder.create_block();
                                    builder.ins().brif(is_err, emergency_exit_bb, &[], ok_bb, &[]);
                                    builder.switch_to_block(ok_bb);
                                    let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                    builder.ins().jump(succ_bb, &[]);
                                } else if callee == "map.get" || callee == "map.has" {
                                    if args.len() != 2 {
                                        return Err(MirCraneliftBackendError::CodegenError(format!(
                                            "Call to '{callee}' arity mismatch: got {} arguments, expected 2",
                                            args.len()
                                        )));
                                    }
                                    let key_ty = try_operand_type(&args[1], &mir_fn.locals, &self.struct_layouts)?;
                                    let map_val = lower_operand(&args[0], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let key_val = lower_operand(&args[1], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let key_raw = clif_val_to_raw_i64(&mut builder, key_val);
                                    let zero_ctx = builder.ins().iconst(types::I64, 0);
                                    let key_tag = builder.ins().iconst(types::I64, mir_type_to_map_operand_tag(&key_ty));
                                    let target_fn = if callee == "map.has" {
                                        Self::declare_ext_fn(
                                            &mut self.module,
                                            &mut builder.func,
                                            &mut ext_fn_cache,
                                            "lust_mir_map_has_typed",
                                            &[types::I64; 4],
                                            Some(types::I64),
                                        )?
                                    } else {
                                        Self::declare_ext_fn(
                                            &mut self.module,
                                            &mut builder.func,
                                            &mut ext_fn_cache,
                                            "lust_mir_map_get_typed",
                                            &[types::I64; 4],
                                            Some(types::I64),
                                        )?
                                    };
                                    let call_inst = builder.ins().call(target_fn, &[zero_ctx, map_val, key_raw, key_tag]);
                                    let res_val = builder.inst_results(call_inst)[0];
                                    if let Place::Local(loc) = destination {
                                        let loc_decl = mir_fn.locals.get(loc.0).ok_or_else(|| {
                                            MirCraneliftBackendError::CodegenError(format!(
                                                "Call destination local {loc:?} out of bounds"
                                            ))
                                        })?;
                                        let target_ty = mir_to_clif_type(&loc_decl.ty)?;
                                        let res_cast = cast_clif_type(&mut builder, res_val, target_ty);
                                        store_local_val(*loc, res_cast, &mut builder, &local_vars, &stack_slots)?;
                                    }
                                    emit_status_check(&mut builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                                    let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                    builder.ins().jump(succ_bb, &[]);
                                } else if callee == "map.get_unwrap" || callee == "map.get_unwrap_or" {
                                    let map_get_int_unwrap_fn = Self::declare_ext_fn(
                                        &mut self.module,
                                        &mut builder.func,
                                        &mut ext_fn_cache,
                                        "lust_mir_map_get_int_unwrap",
                                        &[types::I64; 5],
                                        Some(types::I64),
                                    )?;
                                    let map_get_typed_fn = Self::declare_ext_fn(
                                        &mut self.module,
                                        &mut builder.func,
                                        &mut ext_fn_cache,
                                        "lust_mir_map_get_typed",
                                        &[types::I64; 4],
                                        Some(types::I64),
                                    )?;
                                    let enum_payload_at_fn = Self::declare_ext_fn(
                                        &mut self.module,
                                        &mut builder.func,
                                        &mut ext_fn_cache,
                                        "lust_mir_enum_payload_at",
                                        &[types::I64; 3],
                                        Some(types::I64),
                                    )?;
                                    let expected_arity = if callee == "map.get_unwrap_or" { 3 } else { 2 };
                                    if args.len() != expected_arity {
                                        return Err(MirCraneliftBackendError::CodegenError(format!(
                                            "Call to '{callee}' arity mismatch: got {} arguments, expected {expected_arity}",
                                            args.len()
                                        )));
                                    }
                                    let key_ty = try_operand_type(&args[1], &mir_fn.locals, &self.struct_layouts)?;
                                    let dst_mir_ty = match destination {
                                        Place::Local(loc) => mir_fn
                                            .locals
                                            .get(loc.0)
                                            .map(|d| d.ty.clone())
                                            .unwrap_or(MirType::Int),
                                        _ => MirType::Int,
                                    };
                                    let map_val = lower_operand(&args[0], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let key_val = lower_operand(&args[1], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let key_raw = clif_val_to_raw_i64(&mut builder, key_val);
                                    let zero_ctx = builder.ins().iconst(types::I64, 0);
                                    let key_tag = builder.ins().iconst(types::I64, mir_type_to_map_operand_tag(&key_ty));
                                    let res_val = if dst_mir_ty == MirType::Int && callee == "map.get_unwrap" {
                                        let trap_sentinel = builder.ins().iconst(types::I64, i64::MIN);
                                        let call_inst = builder.ins().call(
                                            map_get_int_unwrap_fn,
                                            &[zero_ctx, map_val, key_raw, key_tag, trap_sentinel],
                                        );
                                        let v = builder.inst_results(call_inst)[0];
                                        emit_status_check(&mut builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                                        v
                                    } else if dst_mir_ty == MirType::Int && callee == "map.get_unwrap_or" {
                                        let def_val = lower_operand(&args[2], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                        let def_i64 = cast_clif_type(&mut builder, def_val, types::I64);
                                        let is_sentinel = builder.ins().icmp_imm_s(IntCC::Equal, def_i64, i64::MIN);
                                        let slow_bb = builder.create_block();
                                        let fast_bb = builder.create_block();
                                        let merge_bb = builder.create_block();
                                        let merged_param = builder.append_block_param(merge_bb, types::I64);
                                        builder.set_cold_block(slow_bb);
                                        builder.ins().brif(is_sentinel, slow_bb, &[], fast_bb, &[]);

                                        builder.switch_to_block(fast_bb);
                                        let fast_call = builder.ins().call(
                                            map_get_int_unwrap_fn,
                                            &[zero_ctx, map_val, key_raw, key_tag, def_i64],
                                        );
                                        let fast_v = builder.inst_results(fast_call)[0];
                                        emit_status_check(&mut builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                                        builder.ins().jump(merge_bb, &[BlockArg::Value(fast_v)]);

                                        builder.switch_to_block(slow_bb);
                                        let opt_call = builder.ins().call(
                                            map_get_typed_fn,
                                            &[zero_ctx, map_val, key_raw, key_tag],
                                        );
                                        let opt_h = builder.inst_results(opt_call)[0];
                                        emit_status_check(&mut builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                                        let is_none = builder.ins().icmp_imm_s(IntCC::Equal, opt_h, 0);
                                        let some_bb = builder.create_block();
                                        builder.ins().brif(is_none, merge_bb, &[BlockArg::Value(def_i64)], some_bb, &[]);
                                        builder.switch_to_block(some_bb);
                                        let f0 = builder.ins().iconst(types::I64, 0);
                                        let pay_call = builder.ins().call(enum_payload_at_fn, &[zero_ctx, opt_h, f0]);
                                        let pay_v = builder.inst_results(pay_call)[0];
                                        builder.ins().jump(merge_bb, &[BlockArg::Value(pay_v)]);

                                        builder.switch_to_block(merge_bb);
                                        merged_param
                                    } else {
                                        let opt_call = builder.ins().call(
                                            map_get_typed_fn,
                                            &[zero_ctx, map_val, key_raw, key_tag],
                                        );
                                        let opt_h = builder.inst_results(opt_call)[0];
                                        emit_status_check(&mut builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                                        let is_none = builder.ins().icmp_imm_s(IntCC::Equal, opt_h, 0);
                                        let target_ty = mir_to_clif_type(&dst_mir_ty)?;
                                        if callee == "map.get_unwrap_or" {
                                            let def_val = lower_operand(&args[2], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                            let def_cast = cast_clif_type(&mut builder, def_val, target_ty);
                                            let some_bb = builder.create_block();
                                            let merge_bb = builder.create_block();
                                            let merged_param = builder.append_block_param(merge_bb, target_ty);
                                            builder.ins().brif(is_none, merge_bb, &[BlockArg::Value(def_cast)], some_bb, &[]);
                                            builder.switch_to_block(some_bb);
                                            let f0 = builder.ins().iconst(types::I64, 0);
                                            let pay_call = builder.ins().call(enum_payload_at_fn, &[zero_ctx, opt_h, f0]);
                                            let pay_raw = builder.inst_results(pay_call)[0];
                                            let pay_val = if target_ty == types::F64 {
                                                builder.ins().bitcast(types::F64, MemFlagsData::new(), pay_raw)
                                            } else {
                                                cast_clif_type(&mut builder, pay_raw, target_ty)
                                            };
                                            builder.ins().jump(merge_bb, &[BlockArg::Value(pay_val)]);
                                            builder.switch_to_block(merge_bb);
                                            merged_param
                                        } else {
                                            let none_bb = builder.create_block();
                                            let some_bb = builder.create_block();
                                            builder.set_cold_block(none_bb);
                                            builder.ins().brif(is_none, none_bb, &[], some_bb, &[]);
                                            builder.switch_to_block(none_bb);
                                            let oob = builder.ins().iconst(types::I64, MIR_STATUS_ERR_OUT_OF_BOUNDS);
                                            builder.ins().call(status_set_fn, &[oob]);
                                            builder.ins().jump(emergency_exit_bb, &[]);
                                            builder.switch_to_block(some_bb);
                                            let f0 = builder.ins().iconst(types::I64, 0);
                                            let pay_call = builder.ins().call(enum_payload_at_fn, &[zero_ctx, opt_h, f0]);
                                            let pay_raw = builder.inst_results(pay_call)[0];
                                            if target_ty == types::F64 {
                                                builder.ins().bitcast(types::F64, MemFlagsData::new(), pay_raw)
                                            } else {
                                                cast_clif_type(&mut builder, pay_raw, target_ty)
                                            }
                                        }
                                    };
                                    if let Place::Local(loc) = destination {
                                        let loc_decl = mir_fn.locals.get(loc.0).ok_or_else(|| {
                                            MirCraneliftBackendError::CodegenError(format!(
                                                "Call destination local {loc:?} out of bounds"
                                            ))
                                        })?;
                                        let target_ty = mir_to_clif_type(&loc_decl.ty)?;
                                        let res_cast = cast_clif_type(&mut builder, res_val, target_ty);
                                        store_local_val(*loc, res_cast, &mut builder, &local_vars, &stack_slots)?;
                                    }
                                    let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                    builder.ins().jump(succ_bb, &[]);
                                } else if callee == "string.sub" {
                                    if args.len() != 2 && args.len() != 3 {
                                        return Err(MirCraneliftBackendError::CodegenError(format!(
                                            "Call to 'string.sub' arity mismatch: got {} arguments, expected 2 or 3",
                                            args.len()
                                        )));
                                    }
                                    let s_val = lower_operand(&args[0], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let start_val = lower_operand(&args[1], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let start_i64 = cast_clif_type(&mut builder, start_val, types::I64);
                                    let end_i64 = if args.len() == 3 {
                                        let ev = lower_operand(&args[2], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                        cast_clif_type(&mut builder, ev, types::I64)
                                    } else {
                                        builder.ins().iconst(types::I64, -1)
                                    };
                                    let fused_sub_target = match destination {
                                        Place::Local(loc) => fused_sub_tmp_to_sub.get(&loc.0).copied(),
                                        _ => None,
                                    };
                                    if let Some(sub_loc) = fused_sub_target {
                                        let fvars = &fused_sub_vars[&sub_loc];
                                        builder.def_var(fvars.base_var, s_val);
                                        builder.def_var(fvars.start_var, start_i64);
                                        builder.def_var(fvars.end_var, end_i64);
                                        if let Place::Local(loc) = destination {
                                            let zero = builder.ins().iconst(types::I64, 0);
                                            store_local_val(*loc, zero, &mut builder, &local_vars, &stack_slots)?;
                                        }
                                        let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                        builder.ins().jump(succ_bb, &[]);
                                    } else {
                                        let zero_ctx = builder.ins().iconst(types::I64, 0);
                                        let string_sub_fn = Self::declare_ext_fn(
                                            &mut self.module,
                                            &mut builder.func,
                                            &mut ext_fn_cache,
                                            "lust_mir_string_sub",
                                            &[types::I64; 4],
                                            Some(types::I64),
                                        )?;
                                        let call_inst = builder.ins().call(string_sub_fn, &[zero_ctx, s_val, start_i64, end_i64]);
                                        let res_val = builder.inst_results(call_inst)[0];
                                        if let Place::Local(loc) = destination {
                                            let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                            let res_cast = cast_clif_type(&mut builder, res_val, target_ty);
                                            store_local_val(*loc, res_cast, &mut builder, &local_vars, &stack_slots)?;
                                        }
                                        emit_status_check(&mut builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                                        let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                        builder.ins().jump(succ_bb, &[]);
                                    }
                                } else if callee == "string.byte" || callee == "string.byte_at" {
                                    if args.is_empty() || args.len() > 3 {
                                        return Err(MirCraneliftBackendError::CodegenError(format!(
                                            "Call to '{callee}' arity mismatch: got {} arguments",
                                            args.len()
                                        )));
                                    }
                                    let fused_sub_src = match &args[0] {
                                        Operand::Copy(Place::Local(sl)) | Operand::Move(Place::Local(sl))
                                            if callee == "string.byte"
                                                && (args.len() == 2 || (args.len() == 3 && args[1] == args[2]))
                                                && fused_sub_vars.contains_key(&sl.0) =>
                                        {
                                            Some(sl.0)
                                        }
                                        _ => None,
                                    };
                                    let start_i64 = if args.len() >= 2 {
                                        let sv = lower_operand(&args[1], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                        cast_clif_type(&mut builder, sv, types::I64)
                                    } else {
                                        builder.ins().iconst(types::I64, 1)
                                    };
                                    let dst_mir_ty = match destination {
                                        Place::Local(loc) => mir_fn.locals.get(loc.0).map(|d| d.ty.clone()),
                                        _ => None,
                                    };
                                    let res_val = if let Some(sub_loc) = fused_sub_src
                                        && matches!(&dst_mir_ty, Some(MirType::Array(inner)) if **inner == MirType::Int)
                                    {
                                        let fvars = &fused_sub_vars[&sub_loc];
                                        let base_v = builder.use_var(fvars.base_var);
                                        let sub_s = builder.use_var(fvars.start_var);
                                        let sub_e = builder.use_var(fvars.end_var);
                                        let clif_string_sub_byte_array_fn = Self::declare_ext_fn(
                                            &mut self.module,
                                            &mut builder.func,
                                            &mut ext_fn_cache,
                                            "lust_mir_clif_string_sub_byte_array",
                                            &[types::I64; 4],
                                            Some(types::I64),
                                        )?;
                                        let call_inst = builder.ins().call(
                                            clif_string_sub_byte_array_fn,
                                            &[base_v, sub_s, sub_e, start_i64],
                                        );
                                        builder.inst_results(call_inst)[0]
                                    } else {
                                        let zero_ctx = builder.ins().iconst(types::I64, 0);
                                        let s_val = lower_operand(&args[0], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                        if callee == "string.byte_at" || matches!(dst_mir_ty, Some(MirType::Int)) {
                                            let string_byte_at_fn = Self::declare_ext_fn(
                                                &mut self.module,
                                                &mut builder.func,
                                                &mut ext_fn_cache,
                                                "lust_mir_string_byte_at",
                                                &[types::I64; 3],
                                                Some(types::I64),
                                            )?;
                                            let call_inst = builder.ins().call(string_byte_at_fn, &[zero_ctx, s_val, start_i64]);
                                            builder.inst_results(call_inst)[0]
                                        } else {
                                            let end_i64 = if args.len() == 3 {
                                                let ev = lower_operand(&args[2], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                                cast_clif_type(&mut builder, ev, types::I64)
                                            } else {
                                                start_i64
                                            };
                                            if let Some(MirType::Array(inner)) = &dst_mir_ty {
                                                let elem_tag = mir_array_elem_tag(inner).unwrap_or(0);
                                                let tag_val = builder.ins().iconst(types::I64, elem_tag);
                                                let clif_string_byte_fn = Self::declare_ext_fn(
                                                    &mut self.module,
                                                    &mut builder.func,
                                                    &mut ext_fn_cache,
                                                    "lust_mir_clif_string_byte",
                                                    &[types::I64; 5],
                                                    Some(types::I64),
                                                )?;
                                                let call_inst = builder.ins().call(
                                                    clif_string_byte_fn,
                                                    &[zero_ctx, s_val, start_i64, end_i64, tag_val],
                                                );
                                                builder.inst_results(call_inst)[0]
                                            } else {
                                                let string_byte_fn = Self::declare_ext_fn(
                                                    &mut self.module,
                                                    &mut builder.func,
                                                    &mut ext_fn_cache,
                                                    "lust_mir_string_byte",
                                                    &[types::I64; 4],
                                                    Some(types::I64),
                                                )?;
                                                let call_inst = builder.ins().call(
                                                    string_byte_fn,
                                                    &[zero_ctx, s_val, start_i64, end_i64],
                                                );
                                                builder.inst_results(call_inst)[0]
                                            }
                                        }
                                    };
                                    if let Place::Local(loc) = destination {
                                        let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                        let res_cast = cast_clif_type(&mut builder, res_val, target_ty);
                                        store_local_val(*loc, res_cast, &mut builder, &local_vars, &stack_slots)?;
                                    }
                                    emit_status_check(&mut builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                                    let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                    builder.ins().jump(succ_bb, &[]);
                                } else if callee == "string.gsub" {
                                    if args.len() != 3 {
                                        return Err(MirCraneliftBackendError::CodegenError(format!(
                                            "Call to 'string.gsub' arity mismatch: got {} arguments, expected 3",
                                            args.len()
                                        )));
                                    }
                                    let zero_ctx = builder.ins().iconst(types::I64, 0);
                                    let s_val = lower_operand(&args[0], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let pat_val = lower_operand(&args[1], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let repl_val = lower_operand(&args[2], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let dst_mir_ty = match destination {
                                        Place::Local(loc) => mir_fn.locals.get(loc.0).map(|d| d.ty.clone()),
                                        _ => None,
                                    };
                                    let res_val = if let Some(MirType::Array(inner)) = &dst_mir_ty {
                                        let elem_tag = mir_array_elem_tag(inner).unwrap_or(MIR_ARRAY_ELEM_TAG_STRING);
                                        let tag_val = builder.ins().iconst(types::I64, elem_tag);
                                        let clif_string_gsub_fn = Self::declare_ext_fn(
                                            &mut self.module,
                                            &mut builder.func,
                                            &mut ext_fn_cache,
                                            "lust_mir_clif_string_gsub",
                                            &[types::I64; 5],
                                            Some(types::I64),
                                        )?;
                                        let call_inst = builder.ins().call(
                                            clif_string_gsub_fn,
                                            &[zero_ctx, s_val, pat_val, repl_val, tag_val],
                                        );
                                        builder.inst_results(call_inst)[0]
                                    } else if matches!(dst_mir_ty, Some(MirType::String)) {
                                        let tag_val = builder.ins().iconst(types::I64, -1);
                                        let clif_string_gsub_fn = Self::declare_ext_fn(
                                            &mut self.module,
                                            &mut builder.func,
                                            &mut ext_fn_cache,
                                            "lust_mir_clif_string_gsub",
                                            &[types::I64; 5],
                                            Some(types::I64),
                                        )?;
                                        let call_inst = builder.ins().call(
                                            clif_string_gsub_fn,
                                            &[zero_ctx, s_val, pat_val, repl_val, tag_val],
                                        );
                                        builder.inst_results(call_inst)[0]
                                    } else {
                                        let string_gsub_fn = Self::declare_ext_fn(
                                            &mut self.module,
                                            &mut builder.func,
                                            &mut ext_fn_cache,
                                            "lust_mir_string_gsub",
                                            &[types::I64; 4],
                                            Some(types::I64),
                                        )?;
                                        let call_inst = builder.ins().call(
                                            string_gsub_fn,
                                            &[zero_ctx, s_val, pat_val, repl_val],
                                        );
                                        builder.inst_results(call_inst)[0]
                                    };
                                    if let Place::Local(loc) = destination {
                                        let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                        let res_cast = cast_clif_type(&mut builder, res_val, target_ty);
                                        store_local_val(*loc, res_cast, &mut builder, &local_vars, &stack_slots)?;
                                    }
                                    emit_status_check(&mut builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                                    let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                    builder.ins().jump(succ_bb, &[]);
                                } else if callee == "string.split" {
                                    if args.len() != 2 {
                                        return Err(MirCraneliftBackendError::CodegenError(format!(
                                            "Call to 'string.split' arity mismatch: got {} arguments, expected 2",
                                            args.len()
                                        )));
                                    }
                                    let zero_ctx = builder.ins().iconst(types::I64, 0);
                                    let s_val = lower_operand(&args[0], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let sep_val = lower_operand(&args[1], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let dst_mir_ty = match destination {
                                        Place::Local(loc) => mir_fn.locals.get(loc.0).map(|d| d.ty.clone()),
                                        _ => None,
                                    };
                                    let res_val = if let Some(MirType::Array(inner)) = &dst_mir_ty {
                                        let elem_tag = mir_array_elem_tag(inner).unwrap_or(MIR_ARRAY_ELEM_TAG_STRING);
                                        let tag_val = builder.ins().iconst(types::I64, elem_tag);
                                        let clif_string_split_fn = Self::declare_ext_fn(
                                            &mut self.module,
                                            &mut builder.func,
                                            &mut ext_fn_cache,
                                            "lust_mir_clif_string_split",
                                            &[types::I64; 4],
                                            Some(types::I64),
                                        )?;
                                        let call_inst = builder.ins().call(
                                            clif_string_split_fn,
                                            &[zero_ctx, s_val, sep_val, tag_val],
                                        );
                                        builder.inst_results(call_inst)[0]
                                    } else {
                                        let string_split_fn = Self::declare_ext_fn(
                                            &mut self.module,
                                            &mut builder.func,
                                            &mut ext_fn_cache,
                                            "lust_mir_string_split",
                                            &[types::I64; 3],
                                            Some(types::I64),
                                        )?;
                                        let call_inst = builder.ins().call(
                                            string_split_fn,
                                            &[zero_ctx, s_val, sep_val],
                                        );
                                        builder.inst_results(call_inst)[0]
                                    };
                                    if let Place::Local(loc) = destination {
                                        let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                        let res_cast = cast_clif_type(&mut builder, res_val, target_ty);
                                        store_local_val(*loc, res_cast, &mut builder, &local_vars, &stack_slots)?;
                                    }
                                    emit_status_check(&mut builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                                    let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                    builder.ins().jump(succ_bb, &[]);
                                } else if callee == "string.find" {
                                    if args.len() < 2 || args.len() > 4 {
                                        return Err(MirCraneliftBackendError::CodegenError(format!(
                                            "Call to 'string.find' arity mismatch: got {} arguments, expected 2..=4",
                                            args.len()
                                        )));
                                    }
                                    let zero_ctx = builder.ins().iconst(types::I64, 0);
                                    let s_val = lower_operand(&args[0], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let pat_val = lower_operand(&args[1], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let start_i64 = if args.len() >= 3 {
                                        let sv = lower_operand(&args[2], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                        cast_clif_type(&mut builder, sv, types::I64)
                                    } else {
                                        builder.ins().iconst(types::I64, 1)
                                    };
                                    let plain_i64 = if args.len() == 4 {
                                        let pv = lower_operand(&args[3], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                        clif_val_to_raw_i64(&mut builder, pv)
                                    } else {
                                        builder.ins().iconst(types::I64, 0)
                                    };
                                    let dst_mir_ty = match destination {
                                        Place::Local(loc) => mir_fn.locals.get(loc.0).map(|d| d.ty.clone()),
                                        _ => None,
                                    };
                                    let res_val = if let Some(MirType::Array(inner)) = &dst_mir_ty {
                                        let elem_tag = mir_array_elem_tag(inner).unwrap_or(0);
                                        let tag_val = builder.ins().iconst(types::I64, elem_tag);
                                        let clif_string_find_fn = Self::declare_ext_fn(
                                            &mut self.module,
                                            &mut builder.func,
                                            &mut ext_fn_cache,
                                            "lust_mir_clif_string_find",
                                            &[types::I64; 6],
                                            Some(types::I64),
                                        )?;
                                        let call_inst = builder.ins().call(
                                            clif_string_find_fn,
                                            &[zero_ctx, s_val, pat_val, start_i64, plain_i64, tag_val],
                                        );
                                        builder.inst_results(call_inst)[0]
                                    } else if matches!(dst_mir_ty, Some(MirType::Int)) {
                                        let tag_val = builder.ins().iconst(types::I64, -2);
                                        let clif_string_find_fn = Self::declare_ext_fn(
                                            &mut self.module,
                                            &mut builder.func,
                                            &mut ext_fn_cache,
                                            "lust_mir_clif_string_find",
                                            &[types::I64; 6],
                                            Some(types::I64),
                                        )?;
                                        let call_inst = builder.ins().call(
                                            clif_string_find_fn,
                                            &[zero_ctx, s_val, pat_val, start_i64, plain_i64, tag_val],
                                        );
                                        builder.inst_results(call_inst)[0]
                                    } else {
                                        let string_find_fn = Self::declare_ext_fn(
                                            &mut self.module,
                                            &mut builder.func,
                                            &mut ext_fn_cache,
                                            "lust_mir_string_find",
                                            &[types::I64; 5],
                                            Some(types::I64),
                                        )?;
                                        let call_inst = builder.ins().call(
                                            string_find_fn,
                                            &[zero_ctx, s_val, pat_val, start_i64, plain_i64],
                                        );
                                        builder.inst_results(call_inst)[0]
                                    };
                                    if let Place::Local(loc) = destination {
                                        let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                        let res_cast = cast_clif_type(&mut builder, res_val, target_ty);
                                        store_local_val(*loc, res_cast, &mut builder, &local_vars, &stack_slots)?;
                                    }
                                    emit_status_check(&mut builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                                    let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                    builder.ins().jump(succ_bb, &[]);
                                } else if callee == "enum.alloc2"
                                    || callee == "enum.new2"
                                    || ((callee == "enum.alloc" || callee == "enum.new") && (args.len() == 1 || args.len() == 3))
                                {
                                    if args.is_empty() || args.len() > 4 {
                                        return Err(MirCraneliftBackendError::CodegenError(format!(
                                            "Call to '{callee}' arity mismatch: got {} arguments",
                                            args.len()
                                        )));
                                    }
                                    let zero_ctx = builder.ins().iconst(types::I64, 0);
                                    let tag_v = lower_operand(&args[0], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let tag_i64 = cast_clif_type(&mut builder, tag_v, types::I64);
                                    let p0_i64 = if args.len() >= 2 {
                                        let v = lower_operand(&args[1], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                        clif_val_to_raw_i64(&mut builder, v)
                                    } else {
                                        builder.ins().iconst(types::I64, 0)
                                    };
                                    let p1_i64 = if args.len() >= 3 {
                                        let v = lower_operand(&args[2], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                        clif_val_to_raw_i64(&mut builder, v)
                                    } else {
                                        builder.ins().iconst(types::I64, 0)
                                    };
                                    let arity_i64 = if args.len() == 4 {
                                        let v = lower_operand(&args[3], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                        cast_clif_type(&mut builder, v, types::I64)
                                    } else {
                                        builder.ins().iconst(types::I64, (args.len() - 1) as i64)
                                    };
                                    let alloc_enum2_fn = Self::declare_ext_fn(
                                        &mut self.module,
                                        &mut builder.func,
                                        &mut ext_fn_cache,
                                        "lust_mir_alloc_enum2",
                                        &[types::I64; 5],
                                        Some(types::I64),
                                    )?;
                                    let call_inst = builder.ins().call(
                                        alloc_enum2_fn,
                                        &[zero_ctx, tag_i64, p0_i64, p1_i64, arity_i64],
                                    );
                                    let res_val = builder.inst_results(call_inst)[0];
                                    if let Place::Local(loc) = destination {
                                        let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                        let res_cast = cast_clif_type(&mut builder, res_val, target_ty);
                                        store_local_val(*loc, res_cast, &mut builder, &local_vars, &stack_slots)?;
                                    }
                                    emit_status_check(&mut builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                                    let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                    builder.ins().jump(succ_bb, &[]);
                                } else if callee == "enum.payload_at"
                                    || (callee == "enum.payload" && args.len() == 2)
                                {
                                    if args.len() != 2 {
                                        return Err(MirCraneliftBackendError::CodegenError(format!(
                                            "Call to '{callee}' arity mismatch: got {} arguments, expected 2",
                                            args.len()
                                        )));
                                    }
                                    let zero_ctx = builder.ins().iconst(types::I64, 0);
                                    let enum_v = lower_operand(&args[0], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let idx_v = lower_operand(&args[1], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let idx_i64 = cast_clif_type(&mut builder, idx_v, types::I64);
                                    let enum_payload_at_fn = Self::declare_ext_fn(
                                        &mut self.module,
                                        &mut builder.func,
                                        &mut ext_fn_cache,
                                        "lust_mir_enum_payload_at",
                                        &[types::I64; 3],
                                        Some(types::I64),
                                    )?;
                                    let call_inst = builder.ins().call(enum_payload_at_fn, &[zero_ctx, enum_v, idx_i64]);
                                    let raw_val = builder.inst_results(call_inst)[0];
                                    if let Place::Local(loc) = destination {
                                        let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                        let res_val = if target_ty == types::F64 {
                                            builder.ins().bitcast(types::F64, MemFlagsData::new(), raw_val)
                                        } else {
                                            cast_clif_type(&mut builder, raw_val, target_ty)
                                        };
                                        store_local_val(*loc, res_val, &mut builder, &local_vars, &stack_slots)?;
                                    }
                                    let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                    builder.ins().jump(succ_bb, &[]);
                                } else if callee == "queue.new" {
                                    if args.len() != 1 {
                                        return Err(MirCraneliftBackendError::CodegenError(format!(
                                            "Call to 'queue.new' arity mismatch: got {} arguments, expected 1",
                                            args.len()
                                        )));
                                    }
                                    let inline_q_target = match destination {
                                        Place::Local(loc) => inline_queue_alloc_tmp_to_q.get(&loc.0).copied(),
                                        _ => None,
                                    };
                                    if let Some(q_loc) = inline_q_target {
                                        let qv = &inline_queue_vars[&q_loc];
                                        let buf_ptr = if let Some(slot) = qv.stack_slot {
                                            builder.ins().stack_addr(types::I64, slot, 0)
                                        } else {
                                            let cap_val = builder.ins().iconst(types::I64, qv.cap);
                                            let alloc_inst = builder.ins().call(alloc_array_fn, &[cap_val]);
                                            let ptr = builder.inst_results(alloc_inst)[0];
                                            emit_status_check(&mut builder, status_fn, inline_status_ptr_var, active_emergency_bb);
                                            ptr
                                        };
                                        let zero = builder.ins().iconst(types::I64, 0);
                                        builder.def_var(qv.buf_var, buf_ptr);
                                        builder.def_var(qv.head_var, zero);
                                        builder.def_var(qv.tail_var, zero);
                                        builder.def_var(qv.len_var, zero);
                                        if let Place::Local(loc) = destination {
                                            store_local_val(*loc, buf_ptr, &mut builder, &local_vars, &stack_slots)?;
                                        }
                                        let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                        builder.ins().jump(succ_bb, &[]);
                                    } else {
                                        let cap_v = lower_operand(&args[0], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                        let cap_i64 = cast_clif_type(&mut builder, cap_v, types::I64);
                                        let queue_new_fn = Self::declare_ext_fn(
                                            &mut self.module,
                                            &mut builder.func,
                                            &mut ext_fn_cache,
                                            "lust_mir_queue_new",
                                            &[types::I64],
                                            Some(types::I64),
                                        )?;
                                        let call_inst = builder.ins().call(queue_new_fn, &[cap_i64]);
                                        let res_val = builder.inst_results(call_inst)[0];
                                        if let Place::Local(loc) = destination {
                                            let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                            let res_cast = cast_clif_type(&mut builder, res_val, target_ty);
                                            store_local_val(*loc, res_cast, &mut builder, &local_vars, &stack_slots)?;
                                        }
                                        emit_status_check(&mut builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                                        let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                        builder.ins().jump(succ_bb, &[]);
                                    }
                                } else if callee == "queue.offer_int" {
                                    if args.len() != 2 {
                                        return Err(MirCraneliftBackendError::CodegenError(format!(
                                            "Call to 'queue.offer_int' arity mismatch: got {} arguments, expected 2",
                                            args.len()
                                        )));
                                    }
                                    let inline_q = match &args[0] {
                                        Operand::Copy(Place::Local(ql)) | Operand::Move(Place::Local(ql))
                                            if inline_queue_vars.contains_key(&ql.0) =>
                                        {
                                            Some(ql.0)
                                        }
                                        _ => None,
                                    };
                                    if let Some(q_loc) = inline_q {
                                        let qv = &inline_queue_vars[&q_loc];
                                        let val_v = lower_operand(&args[1], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, active_bc_ctx)?;
                                        let val_i64 = cast_clif_type(&mut builder, val_v, types::I64);
                                        let cur_len = builder.use_var(qv.len_var);
                                        let has_room = builder.ins().icmp_imm_s(IntCC::SignedLessThan, cur_len, qv.cap);
                                        let dest_is_live = match destination {
                                            Place::Local(loc) => local_read_counts.get(loc.0).copied().unwrap_or(1) > 0,
                                            _ => true,
                                        };
                                        let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                        if qv.stack_slot.is_some() {
                                            // Branchless inline ring-buffer offer using the extra scratch word at index `cap`
                                            // (zero extra CLIF blocks, zero branch misprediction, exact full-queue semantics).
                                            let tail = builder.use_var(qv.tail_var);
                                            let scratch_idx = builder.ins().iconst(types::I64, qv.cap);
                                            let write_idx = builder.ins().select(has_room, tail, scratch_idx);
                                            let buf_ptr = builder.use_var(qv.buf_var);
                                            let byte_off = builder.ins().ishl_imm_s(write_idx, 3);
                                            let elem_addr = builder.ins().iadd(buf_ptr, byte_off);
                                            builder.ins().store(MemFlagsData::trusted(), val_i64, elem_addr, 0);
                                            let inc = builder.ins().uextend(types::I64, has_room);
                                            let tail_adv = builder.ins().iadd(tail, inc);
                                            let next_tail = if (qv.cap as u64).is_power_of_two() {
                                                let mask = builder.ins().iconst(types::I64, qv.cap - 1);
                                                builder.ins().band(tail_adv, mask)
                                            } else {
                                                let zero = builder.ins().iconst(types::I64, 0);
                                                let eq_cap = builder.ins().icmp_imm_s(IntCC::Equal, tail_adv, qv.cap);
                                                builder.ins().select(eq_cap, zero, tail_adv)
                                            };
                                            builder.def_var(qv.tail_var, next_tail);
                                            let next_len = builder.ins().iadd(cur_len, inc);
                                            builder.def_var(qv.len_var, next_len);
                                            if dest_is_live
                                                && let Place::Local(loc) = destination
                                            {
                                                let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                                let res_cast = cast_clif_type(&mut builder, inc, target_ty);
                                                store_local_val(*loc, res_cast, &mut builder, &local_vars, &stack_slots)?;
                                            }
                                            builder.ins().jump(succ_bb, &[]);
                                        } else if dest_is_live {
                                            let push_bb = builder.create_block();
                                            let merge_bb = builder.create_block();
                                            let res_param = builder.append_block_param(merge_bb, types::I64);
                                            let zero_res = builder.ins().iconst(types::I64, 0);
                                            builder.ins().brif(has_room, push_bb, &[], merge_bb, &[BlockArg::Value(zero_res)]);

                                            builder.switch_to_block(push_bb);
                                            let tail = builder.use_var(qv.tail_var);
                                            let buf_ptr = builder.use_var(qv.buf_var);
                                            let byte_off = builder.ins().ishl_imm_s(tail, 3);
                                            let elem_addr = builder.ins().iadd(buf_ptr, byte_off);
                                            builder.ins().store(MemFlagsData::trusted(), val_i64, elem_addr, 16);
                                            let tail_plus1 = builder.ins().iadd_imm_s(tail, 1);
                                            let next_tail = if (qv.cap as u64).is_power_of_two() {
                                                let mask = builder.ins().iconst(types::I64, qv.cap - 1);
                                                builder.ins().band(tail_plus1, mask)
                                            } else {
                                                let zero = builder.ins().iconst(types::I64, 0);
                                                let eq_cap = builder.ins().icmp_imm_s(IntCC::Equal, tail_plus1, qv.cap);
                                                builder.ins().select(eq_cap, zero, tail_plus1)
                                            };
                                            builder.def_var(qv.tail_var, next_tail);
                                            let next_len = builder.ins().iadd_imm_s(cur_len, 1);
                                            builder.def_var(qv.len_var, next_len);
                                            let one_res = builder.ins().iconst(types::I64, 1);
                                            builder.ins().jump(merge_bb, &[BlockArg::Value(one_res)]);

                                            builder.switch_to_block(merge_bb);
                                            if let Place::Local(loc) = destination {
                                                let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                                let res_cast = cast_clif_type(&mut builder, res_param, target_ty);
                                                store_local_val(*loc, res_cast, &mut builder, &local_vars, &stack_slots)?;
                                            }
                                            builder.ins().jump(succ_bb, &[]);
                                        } else {
                                            let push_bb = builder.create_block();
                                            builder.ins().brif(has_room, push_bb, &[], succ_bb, &[]);
                                            builder.switch_to_block(push_bb);
                                            let tail = builder.use_var(qv.tail_var);
                                            let buf_ptr = builder.use_var(qv.buf_var);
                                            let byte_off = builder.ins().ishl_imm_s(tail, 3);
                                            let elem_addr = builder.ins().iadd(buf_ptr, byte_off);
                                            builder.ins().store(MemFlagsData::trusted(), val_i64, elem_addr, 16);
                                            let tail_plus1 = builder.ins().iadd_imm_s(tail, 1);
                                            let next_tail = if (qv.cap as u64).is_power_of_two() {
                                                let mask = builder.ins().iconst(types::I64, qv.cap - 1);
                                                builder.ins().band(tail_plus1, mask)
                                            } else {
                                                let zero = builder.ins().iconst(types::I64, 0);
                                                let eq_cap = builder.ins().icmp_imm_s(IntCC::Equal, tail_plus1, qv.cap);
                                                builder.ins().select(eq_cap, zero, tail_plus1)
                                            };
                                            builder.def_var(qv.tail_var, next_tail);
                                            let next_len = builder.ins().iadd_imm_s(cur_len, 1);
                                            builder.def_var(qv.len_var, next_len);
                                            builder.ins().jump(succ_bb, &[]);
                                        }
                                    } else {
                                        let q_v = lower_operand(&args[0], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                        let q_i64 = cast_clif_type(&mut builder, q_v, types::I64);
                                        let val_v = lower_operand(&args[1], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                        let val_i64 = cast_clif_type(&mut builder, val_v, types::I64);
                                        let queue_offer_int_fn = Self::declare_ext_fn(
                                            &mut self.module,
                                            &mut builder.func,
                                            &mut ext_fn_cache,
                                            "lust_mir_queue_offer_int",
                                            &[types::I64, types::I64],
                                            Some(types::I64),
                                        )?;
                                        let call_inst = builder.ins().call(queue_offer_int_fn, &[q_i64, val_i64]);
                                        let res_val = builder.inst_results(call_inst)[0];
                                        if let Place::Local(loc) = destination {
                                            let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                            let res_cast = cast_clif_type(&mut builder, res_val, target_ty);
                                            store_local_val(*loc, res_cast, &mut builder, &local_vars, &stack_slots)?;
                                        }
                                        emit_status_check(&mut builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                                        let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                        builder.ins().jump(succ_bb, &[]);
                                    }
                                } else if callee == "queue.take_int" {
                                    if args.len() != 1 {
                                        return Err(MirCraneliftBackendError::CodegenError(format!(
                                            "Call to 'queue.take_int' arity mismatch: got {} arguments, expected 1",
                                            args.len()
                                        )));
                                    }
                                    let inline_q = match &args[0] {
                                        Operand::Copy(Place::Local(ql)) | Operand::Move(Place::Local(ql))
                                            if inline_queue_vars.contains_key(&ql.0) =>
                                        {
                                            Some(ql.0)
                                        }
                                        _ => None,
                                    };
                                    if let Some(q_loc) = inline_q {
                                        let qv = &inline_queue_vars[&q_loc];
                                        let cur_len = builder.use_var(qv.len_var);
                                        let is_empty = builder.ins().icmp_imm_s(IntCC::SignedLessThanOrEqual, cur_len, 0);
                                        emit_oob_branch_if_bad(is_empty, &mut builder, active_bc_ctx);
                                        let cur_head = builder.use_var(qv.head_var);
                                        let buf_ptr = builder.use_var(qv.buf_var);
                                        let byte_off = builder.ins().ishl_imm_s(cur_head, 3);
                                        let elem_addr = builder.ins().iadd(buf_ptr, byte_off);
                                        let elem_off = if qv.stack_slot.is_some() { 0 } else { 16 };
                                        let item_val = builder.ins().load(types::I64, MemFlagsData::trusted(), elem_addr, elem_off);
                                        let head_plus1 = builder.ins().iadd_imm_s(cur_head, 1);
                                        let next_head = if (qv.cap as u64).is_power_of_two() {
                                            let mask = builder.ins().iconst(types::I64, qv.cap - 1);
                                            builder.ins().band(head_plus1, mask)
                                        } else {
                                            let zero = builder.ins().iconst(types::I64, 0);
                                            let eq_cap = builder.ins().icmp_imm_s(IntCC::Equal, head_plus1, qv.cap);
                                            builder.ins().select(eq_cap, zero, head_plus1)
                                        };
                                        let next_len = builder.ins().iadd_imm_s(cur_len, -1);
                                        builder.def_var(qv.head_var, next_head);
                                        builder.def_var(qv.len_var, next_len);
                                        if let Place::Local(loc) = destination {
                                            let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                            let res_cast = cast_clif_type(&mut builder, item_val, target_ty);
                                            store_local_val(*loc, res_cast, &mut builder, &local_vars, &stack_slots)?;
                                        }
                                        let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                        builder.ins().jump(succ_bb, &[]);
                                    } else {
                                        let q_v = lower_operand(&args[0], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                        let q_i64 = cast_clif_type(&mut builder, q_v, types::I64);
                                        let queue_take_int_fn = Self::declare_ext_fn(
                                            &mut self.module,
                                            &mut builder.func,
                                            &mut ext_fn_cache,
                                            "lust_mir_queue_take_int",
                                            &[types::I64],
                                            Some(types::I64),
                                        )?;
                                        let call_inst = builder.ins().call(queue_take_int_fn, &[q_i64]);
                                        let res_val = builder.inst_results(call_inst)[0];
                                        if let Place::Local(loc) = destination {
                                            let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                            let res_cast = cast_clif_type(&mut builder, res_val, target_ty);
                                            store_local_val(*loc, res_cast, &mut builder, &local_vars, &stack_slots)?;
                                        }
                                        emit_status_check(&mut builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                                        let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                        builder.ins().jump(succ_bb, &[]);
                                    }
                                } else if callee == "__mir_unknown_is_int" {
                                    if args.len() != 1 {
                                        return Err(MirCraneliftBackendError::CodegenError(format!(
                                            "Call to '__mir_unknown_is_int' arity mismatch: got {} arguments, expected 1",
                                            args.len()
                                        )));
                                    }
                                    let h_v = lower_operand(&args[0], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let h_i64 = clif_val_to_raw_i64(&mut builder, h_v);
                                    let unknown_is_int_fn = Self::declare_ext_fn(
                                        &mut self.module,
                                        &mut builder.func,
                                        &mut ext_fn_cache,
                                        "lust_mir_unknown_is_int",
                                        &[types::I64],
                                        Some(types::I64),
                                    )?;
                                    let call_inst = builder.ins().call(unknown_is_int_fn, &[h_i64]);
                                    let res_val = builder.inst_results(call_inst)[0];
                                    if let Place::Local(loc) = destination {
                                        let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                        let res_cast = cast_clif_type(&mut builder, res_val, target_ty);
                                        store_local_val(*loc, res_cast, &mut builder, &local_vars, &stack_slots)?;
                                    }
                                    let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                    builder.ins().jump(succ_bb, &[]);
                                } else if callee == "__mir_unknown_unwrap_int" {
                                    if args.len() != 1 {
                                        return Err(MirCraneliftBackendError::CodegenError(format!(
                                            "Call to '__mir_unknown_unwrap_int' arity mismatch: got {} arguments, expected 1",
                                            args.len()
                                        )));
                                    }
                                    let h_v = lower_operand(&args[0], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let h_i64 = clif_val_to_raw_i64(&mut builder, h_v);
                                    let unknown_unwrap_int_fn = Self::declare_ext_fn(
                                        &mut self.module,
                                        &mut builder.func,
                                        &mut ext_fn_cache,
                                        "lust_mir_unknown_unwrap_int",
                                        &[types::I64],
                                        Some(types::I64),
                                    )?;
                                    let call_inst = builder.ins().call(unknown_unwrap_int_fn, &[h_i64]);
                                    let res_val = builder.inst_results(call_inst)[0];
                                    if let Place::Local(loc) = destination {
                                        let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                        let res_cast = cast_clif_type(&mut builder, res_val, target_ty);
                                        store_local_val(*loc, res_cast, &mut builder, &local_vars, &stack_slots)?;
                                    }
                                    emit_status_check(&mut builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                                    let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                    builder.ins().jump(succ_bb, &[]);
                                } else if callee == "__mir_unknown_get_field" {
                                    if args.len() != 2 {
                                        return Err(MirCraneliftBackendError::CodegenError(format!(
                                            "Call to '__mir_unknown_get_field' arity mismatch: got {} arguments, expected 2",
                                            args.len()
                                        )));
                                    }
                                    let recv_v = lower_operand(&args[0], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let recv_i64 = clif_val_to_raw_i64(&mut builder, recv_v);
                                    let dst_clif_ty = match destination {
                                        Place::Local(loc) => mir_to_clif_type(&mir_fn.locals[loc.0].ty)?,
                                        _ => types::I64,
                                    };
                                    let field_name_opt = match &args[1] {
                                        Operand::Constant(MirConstant::String(s)) => Some(s.as_str()),
                                        _ => None,
                                    };
                                    let mut candidates: Vec<(String, i64, bool, i32, i32, types::Type)> = Vec::new();
                                    if let Some(fname) = field_name_opt {
                                        let mut layouts_iter: Vec<(&String, &MirStructLayout)> =
                                            if let Some(dt) = dispatch_tables {
                                                dt.pic_struct_layouts.iter().collect()
                                            } else {
                                                self.struct_layouts.iter().collect()
                                            };
                                        layouts_iter.sort_by(|a, b| a.0.cmp(b.0));
                                        for (sname, layout) in layouts_iter {
                                            if let Some(fdesc) = layout.field_by_name(fname) {
                                                let tid = intern_static_struct_type_id(sname);
                                                let is_cflat = self.struct_layouts.contains_key(sname);
                                                let word_off = 16 + (fdesc.ordinal as i32) * 8;
                                                let clif_ty = mir_type_from_canonical_tag_clif(fdesc.tag);
                                                candidates.push((
                                                    sname.to_string(),
                                                    tid,
                                                    is_cflat,
                                                    fdesc.byte_offset as i32,
                                                    word_off,
                                                    clif_ty,
                                                ));
                                            }
                                        }
                                    }
                                    let recv_mir_ty = try_operand_type(&args[0], &mir_fn.locals, &self.struct_layouts).ok();
                                    let static_hit = match recv_mir_ty {
                                        Some(MirType::Struct(ref sname)) => {
                                            candidates.iter().find(|c| &c.0 == sname).cloned()
                                        }
                                        _ => None,
                                    };
                                    let res_val = if let Some((_, _, is_cflat, byte_off, word_off, field_clif_ty)) = static_hit {
                                        let is_null = builder.ins().icmp_imm_s(IntCC::Equal, recv_i64, 0);
                                        emit_oob_branch_if_bad(is_null, &mut builder, bc_ctx);
                                        let loaded = if is_cflat {
                                            builder.ins().load(field_clif_ty, MemFlagsData::trusted(), recv_i64, byte_off)
                                        } else {
                                            let raw_w = builder.ins().load(types::I64, MemFlagsData::trusted(), recv_i64, word_off);
                                            if field_clif_ty == types::F64 {
                                                builder.ins().bitcast(types::F64, MemFlagsData::new(), raw_w)
                                            } else {
                                                cast_clif_type(&mut builder, raw_w, field_clif_ty)
                                            }
                                        };
                                        cast_clif_type(&mut builder, loaded, dst_clif_ty)
                                    } else {
                                        let tid_val = match &args[0] {
                                            Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc))
                                                if cached_unknown_tid_vars.contains_key(&loc.0) =>
                                            {
                                                builder.use_var(cached_unknown_tid_vars[&loc.0])
                                            }
                                            _ => {
                                                let unknown_type_id_fn = Self::declare_ext_fn(
                                                    &mut self.module,
                                                    &mut builder.func,
                                                    &mut ext_fn_cache,
                                                    "lust_mir_clif_unknown_type_id",
                                                    &[types::I64],
                                                    Some(types::I64),
                                                )?;
                                                let tid_call = builder.ins().call(unknown_type_id_fn, &[recv_i64]);
                                                builder.inst_results(tid_call)[0]
                                            }
                                        };
                                        let merge_bb = builder.create_block();
                                        let merged_param = builder.append_block_param(merge_bb, dst_clif_ty);
                                        let fallback_bb = builder.create_block();
                                        builder.set_cold_block(fallback_bb);
                                        for &(_, cand_tid, is_cflat, byte_off, word_off, field_clif_ty) in &candidates {
                                            let arm_bb = builder.create_block();
                                            let next_bb = builder.create_block();
                                            let is_match = builder.ins().icmp_imm_s(IntCC::Equal, tid_val, cand_tid);
                                            builder.ins().brif(is_match, arm_bb, &[], next_bb, &[]);
                                            builder.switch_to_block(arm_bb);
                                            let loaded = if is_cflat {
                                                builder.ins().load(field_clif_ty, MemFlagsData::trusted(), recv_i64, byte_off)
                                            } else {
                                                let raw_w = builder.ins().load(types::I64, MemFlagsData::trusted(), recv_i64, word_off);
                                                if field_clif_ty == types::F64 {
                                                    builder.ins().bitcast(types::F64, MemFlagsData::new(), raw_w)
                                                } else {
                                                    cast_clif_type(&mut builder, raw_w, field_clif_ty)
                                                }
                                            };
                                            let cast_v = cast_clif_type(&mut builder, loaded, dst_clif_ty);
                                            builder.ins().jump(merge_bb, &[BlockArg::Value(cast_v)]);
                                            builder.switch_to_block(next_bb);
                                        }
                                        builder.ins().jump(fallback_bb, &[]);
                                        builder.switch_to_block(fallback_bb);
                                        let field_h = lower_operand(&args[1], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                        let field_h_i64 = clif_val_to_raw_i64(&mut builder, field_h);
                                        let unknown_get_field_fallback_fn = Self::declare_ext_fn(
                                            &mut self.module,
                                            &mut builder.func,
                                            &mut ext_fn_cache,
                                            "lust_mir_clif_unknown_get_field_fallback",
                                            &[types::I64, types::I64],
                                            Some(types::I64),
                                        )?;
                                        let fb_call = builder.ins().call(unknown_get_field_fallback_fn, &[recv_i64, field_h_i64]);
                                        let fb_raw = builder.inst_results(fb_call)[0];
                                        emit_status_check(&mut builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                                        let fb_val = if dst_clif_ty == types::F64 {
                                            builder.ins().bitcast(types::F64, MemFlagsData::new(), fb_raw)
                                        } else {
                                            cast_clif_type(&mut builder, fb_raw, dst_clif_ty)
                                        };
                                        builder.ins().jump(merge_bb, &[BlockArg::Value(fb_val)]);
                                        builder.switch_to_block(merge_bb);
                                        merged_param
                                    };
                                    if let Place::Local(loc) = destination {
                                        store_local_val(*loc, res_val, &mut builder, &local_vars, &stack_slots)?;
                                    }
                                    let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                    builder.ins().jump(succ_bb, &[]);
                                } else if callee == "__mir_unknown_call_method" {
                                    if args.len() < 2 {
                                        return Err(MirCraneliftBackendError::CodegenError(format!(
                                            "Call to '__mir_unknown_call_method' arity mismatch: got {} arguments, expected >= 2",
                                            args.len()
                                        )));
                                    }
                                    let recv_v = lower_operand(&args[0], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                    let recv_i64 = clif_val_to_raw_i64(&mut builder, recv_v);
                                    let mut method_arg_vals = Vec::with_capacity(args.len() - 2);
                                    for a in &args[2..] {
                                        method_arg_vals.push(lower_operand(a, &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?);
                                    }
                                    let method_name = match &args[1] {
                                        Operand::Constant(MirConstant::String(s)) => s.as_str(),
                                        _ => {
                                            return Err(MirCraneliftBackendError::UnlowerableValue(
                                                "__mir_unknown_call_method requires constant string method name".into(),
                                            ));
                                        }
                                    };
                                    let colon_suffix = format!(":{method_name}");
                                    let dot_suffix = format!(".{method_name}");
                                    let mut candidates: Vec<(String, i64, String, FuncId, Vec<MirType>, MirType)> = Vec::new();
                                    let expected_params = 1 + method_arg_vals.len();
                                    for (fn_name, &fid) in module_funcs {
                                        let owner_opt = fn_name
                                            .strip_suffix(&colon_suffix)
                                            .or_else(|| fn_name.strip_suffix(&dot_suffix));
                                        let Some(owner) = owner_opt else {
                                            continue;
                                        };
                                        if owner.is_empty() {
                                            continue;
                                        }
                                        if let Some(dt) = dispatch_tables
                                            && let Some((ptys, rty)) = dt.fn_signatures.get(fn_name)
                                            && ptys.len() == expected_params
                                        {
                                            let tid = intern_static_struct_type_id(owner);
                                            candidates.push((
                                                owner.to_string(),
                                                tid,
                                                fn_name.clone(),
                                                fid,
                                                ptys.clone(),
                                                rty.clone(),
                                            ));
                                        }
                                    }
                                    candidates.sort_by(|a, b| a.0.cmp(&b.0));
                                    let dst_mir_ty = match destination {
                                        Place::Local(loc) => mir_fn.locals[loc.0].ty.clone(),
                                        _ => MirType::Int,
                                    };
                                    let dst_clif_ty = mir_to_clif_type(&dst_mir_ty)?;
                                    let tid_val = match &args[0] {
                                        Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc))
                                            if cached_unknown_tid_vars.contains_key(&loc.0) =>
                                        {
                                            builder.use_var(cached_unknown_tid_vars[&loc.0])
                                        }
                                        _ => {
                                            let unknown_type_id_fn = Self::declare_ext_fn(
                                                &mut self.module,
                                                &mut builder.func,
                                                &mut ext_fn_cache,
                                                "lust_mir_clif_unknown_type_id",
                                                &[types::I64],
                                                Some(types::I64),
                                            )?;
                                            let tid_call = builder.ins().call(unknown_type_id_fn, &[recv_i64]);
                                            builder.inst_results(tid_call)[0]
                                        }
                                    };
                                    let merge_bb = builder.create_block();
                                    let merged_param = builder.append_block_param(merge_bb, dst_clif_ty);
                                    let miss_bb = builder.create_block();
                                    builder.set_cold_block(miss_bb);
                                    for (_, cand_tid, cand_name, cand_fid, cand_ptys, _) in &candidates {
                                        let arm_bb = builder.create_block();
                                        let next_bb = builder.create_block();
                                        let is_match = builder.ins().icmp_imm_s(IntCC::Equal, tid_val, *cand_tid);
                                        builder.ins().brif(is_match, arm_bb, &[], next_bb, &[]);
                                        builder.switch_to_block(arm_bb);
                                        let cand_ref = self.module.declare_func_in_func(*cand_fid, &mut builder.func);
                                        let mut call_args = Vec::with_capacity(expected_params);
                                        call_args.push(recv_i64);
                                        for (idx, &av) in method_arg_vals.iter().enumerate() {
                                            let p_clif = mir_to_clif_type(&cand_ptys[1 + idx])?;
                                            call_args.push(cast_clif_type(&mut builder, av, p_clif));
                                        }
                                        let (callee_can_deopt, callee_can_fail) = call_effects
                                            .and_then(|m| m.get(cand_name).copied())
                                            .unwrap_or((true, true));
                                        if callee_can_deopt {
                                            let enter_fn = Self::declare_ext_fn(
                                                &mut self.module,
                                                &mut builder.func,
                                                &mut ext_fn_cache,
                                                "lust_mir_enter_native_callee",
                                                &[],
                                                None,
                                            )?;
                                            builder.ins().call(enter_fn, &[]);
                                        }
                                        let call_inst = builder.ins().call(cand_ref, &call_args);
                                        if callee_can_deopt {
                                            let exit_fn = Self::declare_ext_fn(
                                                &mut self.module,
                                                &mut builder.func,
                                                &mut ext_fn_cache,
                                                "lust_mir_exit_native_callee",
                                                &[],
                                                None,
                                            )?;
                                            builder.ins().call(exit_fn, &[]);
                                        }
                                        let arm_ret = if let Some(&rv) = builder.inst_results(call_inst).first() {
                                            cast_clif_type(&mut builder, rv, dst_clif_ty)
                                        } else if dst_clif_ty == types::F64 {
                                            builder.ins().f64const(0.0)
                                        } else {
                                            builder.ins().iconst(dst_clif_ty, 0)
                                        };
                                        if callee_can_fail {
                                            emit_status_check(&mut builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                                        }
                                        builder.ins().jump(merge_bb, &[BlockArg::Value(arm_ret)]);
                                        builder.switch_to_block(next_bb);
                                    }
                                    builder.ins().jump(miss_bb, &[]);
                                    builder.switch_to_block(miss_bb);
                                    let oob = builder.ins().iconst(types::I64, MIR_STATUS_ERR_OUT_OF_BOUNDS);
                                    builder.ins().call(status_set_fn, &[oob]);
                                    builder.ins().jump(emergency_exit_bb, &[]);
                                    builder.switch_to_block(merge_bb);
                                    if let Place::Local(loc) = destination {
                                        store_local_val(*loc, merged_param, &mut builder, &local_vars, &stack_slots)?;
                                    }
                                    let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                    builder.ins().jump(succ_bb, &[]);
                                } else if !is_builtin_or_special_callee(callee)
                                    && !module_funcs.contains_key(callee)
                                    && !self.host_callbacks.contains_key(callee)
                                    && let Some(fn_loc_decl) = mir_fn.locals.iter().find(|l| {
                                        l.name.as_deref() == Some(callee.as_str())
                                            && matches!(l.ty, MirType::Function { .. } | MirType::Unknown)
                                    })
                                {
                                    // Indirect call through a `MirType::Function` closure/function local (`W16-JIT-CLOSURE`):
                                    // Closure buffer layout: word[0] (offset 16) = fn_id, word[1 + i] = capture_i.
                                    let closure_ptr = load_local_i64(fn_loc_decl.id, &mut builder, &local_vars, &stack_slots)?;
                                    let is_null = builder.ins().icmp_imm_s(IntCC::Equal, closure_ptr, 0);
                                    emit_oob_branch_if_bad(is_null, &mut builder, bc_ctx);
                                    let fn_id_val = builder.ins().load(types::I64, MemFlagsData::trusted(), closure_ptr, 16);
                                    let mut lowered_args = Vec::with_capacity(args.len());
                                    for a in args {
                                        lowered_args.push(lower_operand(a, &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?);
                                    }
                                    let dst_mir_ty = match destination {
                                        Place::Local(loc) => mir_fn.locals[loc.0].ty.clone(),
                                        _ => MirType::Int,
                                    };
                                    let dst_clif_ty = mir_to_clif_type(&dst_mir_ty)?;
                                    let mut candidates: Vec<(i64, String, FuncId, usize, Vec<MirType>)> = Vec::new();
                                    if let Some(dt) = dispatch_tables {
                                        for (cand_name, &cand_fid) in module_funcs {
                                            let Some(&cand_id) = dt.fn_ids.get(cand_name) else {
                                                continue;
                                            };
                                            let Some((cand_ptys, _)) = dt.fn_signatures.get(cand_name) else {
                                                continue;
                                            };
                                            let cap_count = dt.closure_capture_counts.get(cand_name).copied().unwrap_or(0);
                                            if cand_ptys.len() == cap_count + lowered_args.len() {
                                                candidates.push((cand_id, cand_name.clone(), cand_fid, cap_count, cand_ptys.clone()));
                                            }
                                        }
                                    }
                                    candidates.sort_by_key(|c| c.0);
                                    let merge_bb = builder.create_block();
                                    let merged_param = builder.append_block_param(merge_bb, dst_clif_ty);
                                    let miss_bb = builder.create_block();
                                    builder.set_cold_block(miss_bb);
                                    for (cand_id, cand_name, cand_fid, cap_count, cand_ptys) in &candidates {
                                        let arm_bb = builder.create_block();
                                        let next_bb = builder.create_block();
                                        let is_match = builder.ins().icmp_imm_s(IntCC::Equal, fn_id_val, *cand_id);
                                        builder.ins().brif(is_match, arm_bb, &[], next_bb, &[]);
                                        builder.switch_to_block(arm_bb);
                                        let cand_ref = self.module.declare_func_in_func(*cand_fid, &mut builder.func);
                                        let mut call_args = Vec::with_capacity(cand_ptys.len());
                                        for c_idx in 0..*cap_count {
                                            let raw_cap = builder.ins().load(
                                                types::I64,
                                                MemFlagsData::trusted(),
                                                closure_ptr,
                                                16 + ((1 + c_idx) as i32) * 8,
                                            );
                                            let p_clif = mir_to_clif_type(&cand_ptys[c_idx])?;
                                            let cap_val = if p_clif == types::F64 {
                                                builder.ins().bitcast(types::F64, MemFlagsData::new(), raw_cap)
                                            } else {
                                                cast_clif_type(&mut builder, raw_cap, p_clif)
                                            };
                                            call_args.push(cap_val);
                                        }
                                        for (a_idx, &av) in lowered_args.iter().enumerate() {
                                            let p_clif = mir_to_clif_type(&cand_ptys[*cap_count + a_idx])?;
                                            call_args.push(cast_clif_type(&mut builder, av, p_clif));
                                        }
                                        let (callee_can_deopt, callee_can_fail) = call_effects
                                            .and_then(|m| m.get(cand_name).copied())
                                            .unwrap_or((true, true));
                                        if callee_can_deopt {
                                            let enter_fn = Self::declare_ext_fn(
                                                &mut self.module,
                                                &mut builder.func,
                                                &mut ext_fn_cache,
                                                "lust_mir_enter_native_callee",
                                                &[],
                                                None,
                                            )?;
                                            builder.ins().call(enter_fn, &[]);
                                        }
                                        let call_inst = builder.ins().call(cand_ref, &call_args);
                                        if callee_can_deopt {
                                            let exit_fn = Self::declare_ext_fn(
                                                &mut self.module,
                                                &mut builder.func,
                                                &mut ext_fn_cache,
                                                "lust_mir_exit_native_callee",
                                                &[],
                                                None,
                                            )?;
                                            builder.ins().call(exit_fn, &[]);
                                        }
                                        let arm_ret = if let Some(&rv) = builder.inst_results(call_inst).first() {
                                            cast_clif_type(&mut builder, rv, dst_clif_ty)
                                        } else if dst_clif_ty == types::F64 {
                                            builder.ins().f64const(0.0)
                                        } else {
                                            builder.ins().iconst(dst_clif_ty, 0)
                                        };
                                        if callee_can_fail {
                                            emit_status_check(&mut builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                                        }
                                        builder.ins().jump(merge_bb, &[BlockArg::Value(arm_ret)]);
                                        builder.switch_to_block(next_bb);
                                    }
                                    builder.ins().jump(miss_bb, &[]);
                                    builder.switch_to_block(miss_bb);
                                    let oob = builder.ins().iconst(types::I64, MIR_STATUS_ERR_OUT_OF_BOUNDS);
                                    builder.ins().call(status_set_fn, &[oob]);
                                    builder.ins().jump(emergency_exit_bb, &[]);
                                    builder.switch_to_block(merge_bb);
                                    if let Place::Local(loc) = destination {
                                        store_local_val(*loc, merged_param, &mut builder, &local_vars, &stack_slots)?;
                                    }
                                    let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                    builder.ins().jump(succ_bb, &[]);
                                } else {
                                    let mut is_native_mir_call = false;
                                    let mut is_host_callback = false;
                                    let (callee_ref, param_count) = match callee.as_str() {
                                        "math.abs" | "abs" => (
                                            Self::declare_ext_fn(&mut self.module, &mut builder.func, &mut ext_fn_cache, "lust_mir_abs_int", &[types::I64], Some(types::I64))?,
                                            1,
                                        ),
                                        "math.min" | "min" => (
                                            Self::declare_ext_fn(&mut self.module, &mut builder.func, &mut ext_fn_cache, "lust_mir_min_int", &[types::I64, types::I64], Some(types::I64))?,
                                            2,
                                        ),
                                        "math.max" | "max" => (
                                            Self::declare_ext_fn(&mut self.module, &mut builder.func, &mut ext_fn_cache, "lust_mir_max_int", &[types::I64, types::I64], Some(types::I64))?,
                                            2,
                                        ),
                                        "math.clamp" | "clamp" => (
                                            Self::declare_ext_fn(&mut self.module, &mut builder.func, &mut ext_fn_cache, "lust_mir_clamp_int", &[types::I64; 3], Some(types::I64))?,
                                            3,
                                        ),
                                        "map.len" => (
                                            Self::declare_ext_fn(&mut self.module, &mut builder.func, &mut ext_fn_cache, "lust_mir_map_len", &[types::I64], Some(types::I64))?,
                                            1,
                                        ),
                                        "map.get" => (
                                            Self::declare_ext_fn(&mut self.module, &mut builder.func, &mut ext_fn_cache, "lust_mir_map_get", &[types::I64, types::I64], Some(types::I64))?,
                                            2,
                                        ),
                                        "map.set" => (
                                            Self::declare_ext_fn(&mut self.module, &mut builder.func, &mut ext_fn_cache, "lust_mir_map_set", &[types::I64; 3], None)?,
                                            3,
                                        ),
                                        "map.new" => (
                                            Self::declare_ext_fn(&mut self.module, &mut builder.func, &mut ext_fn_cache, "lust_mir_alloc_map", &[], Some(types::I64))?,
                                            0,
                                        ),
                                        "enum.tag" => (
                                            Self::declare_ext_fn(&mut self.module, &mut builder.func, &mut ext_fn_cache, "lust_mir_enum_tag", &[types::I64], Some(types::I64))?,
                                            1,
                                        ),
                                        "enum.payload" => (
                                            Self::declare_ext_fn(&mut self.module, &mut builder.func, &mut ext_fn_cache, "lust_mir_enum_payload", &[types::I64], Some(types::I64))?,
                                            1,
                                        ),
                                        "enum.new" | "enum.alloc" => (
                                            Self::declare_ext_fn(&mut self.module, &mut builder.func, &mut ext_fn_cache, "lust_mir_alloc_enum", &[types::I64, types::I64], Some(types::I64))?,
                                            2,
                                        ),
                                        _ => {
                                            let (fid, pcnt) = if let Some(&fid) = module_funcs.get(callee) {
                                                is_native_mir_call = true;
                                                (fid, args.len())
                                            } else if let Some((cb_params, cb_ret)) =
                                                self.host_callbacks.get(callee)
                                            {
                                                is_host_callback = true;
                                                let mut cb_sig = self.module.make_signature();
                                                for p_ty in cb_params {
                                                    cb_sig.params.push(AbiParam::new(mir_to_clif_type(p_ty)?));
                                                }
                                                let cb_ret_clif = mir_to_clif_type(cb_ret)?;
                                                if cb_ret_clif != types::I8 || *cb_ret != MirType::Unit {
                                                    cb_sig.returns.push(AbiParam::new(cb_ret_clif));
                                                }
                                                let fid = self
                                                    .module
                                                    .declare_function(callee, Linkage::Import, &cb_sig)
                                                    .map_err(|e| {
                                                        MirCraneliftBackendError::CodegenError(format!("{e}"))
                                                    })?;
                                                (fid, cb_params.len())
                                            } else {
                                                return Err(MirCraneliftBackendError::UnlowerableValue(format!(
                                                    "External or unknown callee '{callee}' is not supported by Cranelift backend"
                                                )));
                                            };
                                            (self.module.declare_func_in_func(fid, &mut builder.func), pcnt)
                                        }
                                    };
                                    let mut arg_vals = Vec::with_capacity(param_count);
                                    for a in args {
                                        arg_vals.push(lower_operand(a, &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?);
                                    }
                                    while arg_vals.len() < param_count {
                                        arg_vals.push(builder.ins().iconst(types::I64, 0));
                                    }
                                    let (callee_can_deopt, callee_can_fail) = if is_native_mir_call {
                                        call_effects
                                            .and_then(|m| m.get(callee).copied())
                                            .unwrap_or((true, true))
                                    } else {
                                        (true, true)
                                    };
                                    if is_native_mir_call && callee_can_deopt {
                                        let enter_fn = Self::declare_ext_fn(
                                            &mut self.module,
                                            &mut builder.func,
                                            &mut ext_fn_cache,
                                            "lust_mir_enter_native_callee",
                                            &[],
                                            None,
                                        )?;
                                        builder.ins().call(enter_fn, &[]);
                                    }
                                    let call_inst = builder.ins().call(callee_ref, &arg_vals);
                                    if is_native_mir_call && callee_can_deopt {
                                        let exit_fn = Self::declare_ext_fn(
                                            &mut self.module,
                                            &mut builder.func,
                                            &mut ext_fn_cache,
                                            "lust_mir_exit_native_callee",
                                            &[],
                                            None,
                                        )?;
                                        builder.ins().call(exit_fn, &[]);
                                    }
                                    let ret_vals = builder.inst_results(call_inst);
                                    if !ret_vals.is_empty() {
                                        let res_val = ret_vals[0];
                                        match destination {
                                            Place::Local(loc) => {
                                                let loc_decl = mir_fn.locals.get(loc.0).ok_or_else(|| {
                                                    MirCraneliftBackendError::CodegenError(format!(
                                                        "Call destination local {loc:?} out of bounds"
                                                    ))
                                                })?;
                                                let target_ty = mir_to_clif_type(&loc_decl.ty)?;
                                                let res_val = cast_clif_type(&mut builder, res_val, target_ty);
                                                store_local_val(*loc, res_val, &mut builder, &local_vars, &stack_slots)?;
                                            }
                                            Place::Index(..) | Place::Field(..) => {
                                                let dst_addr = lower_place_addr(
                                                    destination,
                                                    &mut builder,
                                                    &local_vars,
                                                    &stack_slots,
                                                    &mir_fn.locals,
                                                    &self.struct_layouts,
                                                    bc_ctx,
                                                )?;
                                                let val_ty = builder.func.dfg.value_type(res_val);
                                                let val_to_store = if is_bool_array_index_place(
                                                    destination,
                                                    &mir_fn.locals,
                                                    &self.struct_layouts,
                                                ) {
                                                    if val_ty == types::I8 {
                                                        res_val
                                                    } else {
                                                        builder.ins().icmp_imm_s(IntCC::NotEqual, res_val, 0)
                                                    }
                                                } else if val_ty == types::I8 {
                                                    builder.ins().uextend(types::I64, res_val)
                                                } else {
                                                    res_val
                                                };
                                                builder.ins().store(MemFlagsData::trusted(), val_to_store, dst_addr, 0);
                                            }
                                        }
                                    } else if let Place::Local(loc) = destination {
                                        let loc_decl = mir_fn.locals.get(loc.0).ok_or_else(|| {
                                            MirCraneliftBackendError::CodegenError(format!(
                                                "Call destination local {loc:?} out of bounds"
                                            ))
                                        })?;
                                        let target_ty = mir_to_clif_type(&loc_decl.ty)?;
                                        let zero = builder.ins().iconst(types::I64, 0);
                                        let zero_val = cast_clif_type(&mut builder, zero, target_ty);
                                        store_local_val(*loc, zero_val, &mut builder, &local_vars, &stack_slots)?;
                                    }
                                    if !is_native_mir_call || callee_can_fail {
                                        let call_status_ptr = if is_host_callback {
                                            None
                                        } else {
                                            inline_status_ptr_var
                                        };
                                        emit_status_check(
                                            &mut builder,
                                            status_fn,
                                            call_status_ptr,
                                            emergency_exit_bb,
                                        );
                                    }
                                    if is_host_callback {
                                        if let Some((gas_ptr_var, gas_fast_ptr_fn, _)) = gas_ctx {
                                            let reload_call = builder.ins().call(gas_fast_ptr_fn, &[]);
                                            let reloaded_ptr = builder.inst_results(reload_call)[0];
                                            builder.def_var(gas_ptr_var, reloaded_ptr);
                                        }
                                    }

                                    let succ_bb = resolve_target_bb(Some(b_idx), success.0)?;
                                    builder.ins().jump(succ_bb, &[]);
                                }
                            }
                            Terminator::Return(opt_op, _) => {
                                if ret_clif_ty != types::I8 || mir_fn.signature.return_type != MirType::Unit {
                                    let ret_val = if let Some(op) = opt_op {
                                        let v = lower_operand(op, &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, active_bc_ctx)?;
                                        cast_clif_type(&mut builder, v, ret_clif_ty)
                                    } else if ret_clif_ty == types::F64 {
                                        builder.ins().f64const(0.0)
                                    } else {
                                        builder.ins().iconst(ret_clif_ty, 0)
                                    };
                                    if let Some(p_counters) = active_promoted {
                                        let ret_flush_bb = builder.create_block();
                                        if let Some(ci) = cl_opt {
                                            cfg_records[ci].flush_clif_blocks.insert(ret_flush_bb);
                                        }
                                        if let Some(iqi) = iq_opt {
                                            iq_cfg_records[iqi].reconcile_clif_blocks.insert(ret_flush_bb);
                                        }
                                        builder.ins().jump(ret_flush_bb, &[]);
                                        builder.switch_to_block(ret_flush_bb);
                                        emit_promoted_counters_flush(&mut builder, p_counters);
                                    }
                                    builder.ins().return_(&[ret_val]);
                                } else {
                                    if let Some(p_counters) = active_promoted {
                                        let ret_flush_bb = builder.create_block();
                                        if let Some(ci) = cl_opt {
                                            cfg_records[ci].flush_clif_blocks.insert(ret_flush_bb);
                                        }
                                        if let Some(iqi) = iq_opt {
                                            iq_cfg_records[iqi].reconcile_clif_blocks.insert(ret_flush_bb);
                                        }
                                        builder.ins().jump(ret_flush_bb, &[]);
                                        builder.switch_to_block(ret_flush_bb);
                                        emit_promoted_counters_flush(&mut builder, p_counters);
                                    }
                                    builder.ins().return_(&[]);
                                }
                            }
                            Terminator::Unreachable(_) => {
                                builder.ins().trap(cranelift_codegen::ir::TrapCode::unwrap_user(1));
                            }
                            _ => {
                                if ret_clif_ty != types::I8 || mir_fn.signature.return_type != MirType::Unit {
                                    let ret_val = if ret_clif_ty == types::F64 {
                                        builder.ins().f64const(0.0)
                                    } else {
                                        builder.ins().iconst(ret_clif_ty, 0)
                                    };
                                    builder.ins().return_(&[ret_val]);
                                } else {
                                    builder.ins().return_(&[]);
                                }
                            }
                        }
                    }

                    if let Some(ci) = cl_opt {
                        let rec = &mut cfg_records[ci];
                        for bb in builder.func.layout.blocks() {
                            if !blocks_before.contains(&bb)
                                && !rec.flush_clif_blocks.contains(&bb)
                                && !rec.post_flush_clif_blocks.contains(&bb)
                                && !rec.reload_clif_blocks.contains(&bb)
                            {
                                rec.loop_clif_blocks.insert(bb);
                            }
                        }
                    }
                    if let Some(iqi) = iq_opt {
                        let iq_rec = &mut iq_cfg_records[iqi];
                        for bb in builder.func.layout.blocks() {
                            if !blocks_before.contains(&bb)
                                && !iq_rec.reconcile_clif_blocks.contains(&bb)
                                && !iq_rec.post_flush_clif_blocks.contains(&bb)
                                && !iq_rec.reload_clif_blocks.contains(&bb)
                            {
                                iq_rec.loop_clif_blocks.insert(bb);
                            }
                        }
                    }
                }

                // Emit shared cold quantized refill hubs (one hub per multi-header quantized loop nest)
                for (cl_idx, hub_opt) in shared_q_hubs.iter().enumerate() {
                    let Some(hub) = hub_opt else {
                        continue;
                    };
                    if hub.sites.is_empty() {
                        continue;
                    }
                    const OFF_SLOW_FLAGS: i32 =
                        core::mem::offset_of!(SafepointFastState, slow_flags) as i32;
                    const QUANTUM_FAST_ITERS: i64 = 63;
                    let promoted = cl_states[cl_idx].counters;
                    let (sp_ptr_var, _, sp_slow_fn) =
                        safepoint_ctx.expect("quantized hub implies safepoint_ctx");
                    let (_, _, gas_slow_fn) =
                        gas_ctx.expect("quantized hub implies gas_ctx");
                    let (_, sp_cnt_var) = promoted
                        .sp
                        .expect("quantized hub requires promoted sp");
                    let (_, gas_var) = promoted
                        .gas
                        .expect("quantized hub requires promoted gas");
                    let (quantum_rem_var, _) = promoted
                        .quantum
                        .expect("quantized hub requires quantum_rem_var");
                    let amt = hub.amt;

                    let slow_safepoint_bb = builder.create_block();
                    let sp_reload_bb = builder.create_block();
                    let err_dispatch_bb = builder.create_block();
                    let deopt_dispatch_bb = builder.create_block();
                    let gas_check_bb = builder.create_block();
                    let allow_arm_param = builder.append_block_param(gas_check_bb, types::I8);
                    let gas_slow_bb = builder.create_block();
                    let gas_reload_bb = builder.create_block();
                    let arm_quantum_bb = builder.create_block();
                    let cont_dispatch_bb = builder.create_block();

                    builder.set_cold_block(slow_safepoint_bb);
                    builder.set_cold_block(sp_reload_bb);
                    builder.set_cold_block(err_dispatch_bb);
                    builder.set_cold_block(deopt_dispatch_bb);
                    builder.set_cold_block(gas_check_bb);
                    builder.set_cold_block(gas_slow_bb);
                    builder.set_cold_block(gas_reload_bb);
                    builder.set_cold_block(arm_quantum_bb);
                    builder.set_cold_block(cont_dispatch_bb);

                    let rec = &mut cfg_records[cl_idx];
                    rec.loop_clif_blocks.insert(hub.check_reload_bb);
                    rec.loop_clif_blocks.insert(gas_check_bb);
                    rec.loop_clif_blocks.insert(arm_quantum_bb);
                    rec.loop_clif_blocks.insert(cont_dispatch_bb);
                    rec.flush_clif_blocks.insert(slow_safepoint_bb);
                    rec.post_flush_clif_blocks.insert(err_dispatch_bb);
                    rec.post_flush_clif_blocks.insert(deopt_dispatch_bb);
                    rec.reload_clif_blocks.insert(sp_reload_bb);
                    rec.flush_clif_blocks.insert(gas_slow_bb);
                    rec.reload_clif_blocks.insert(gas_reload_bb);

                    builder.switch_to_block(hub.check_reload_bb);
                    let zero_q = builder.ins().iconst(types::I64, 0);
                    builder.def_var(quantum_rem_var, zero_q);
                    let sp0 = builder.use_var(sp_cnt_var);
                    let sp1 = builder.ins().iadd_imm_s(sp0, 1);
                    builder.def_var(sp_cnt_var, sp1);
                    let sp_ptr = builder.use_var(sp_ptr_var);
                    let flags = builder
                        .ins()
                        .load(types::I64, MemFlagsData::trusted(), sp_ptr, OFF_SLOW_FLAGS);
                    let is_slow = builder.ins().icmp_imm_s(IntCC::NotEqual, flags, 0);
                    let allow_arm_yes = builder.ins().iconst(types::I8, 1);
                    builder.ins().brif(
                        is_slow,
                        slow_safepoint_bb,
                        &[],
                        gas_check_bb,
                        &[BlockArg::Value(allow_arm_yes)],
                    );

                    builder.switch_to_block(slow_safepoint_bb);
                    emit_promoted_counters_flush_exact(&mut builder, promoted);
                    let sp_call = builder.ins().call(sp_slow_fn, &[]);
                    let sp_res = if builder.inst_results(sp_call).is_empty() {
                        const OFF_SLOW_RESULT: i32 =
                            core::mem::offset_of!(SafepointFastState, slow_result) as i32;
                        let sp_ptr = builder.use_var(sp_ptr_var);
                        builder
                            .ins()
                            .load(types::I64, MemFlagsData::trusted(), sp_ptr, OFF_SLOW_RESULT)
                    } else {
                        builder.inst_results(sp_call)[0]
                    };
                    let is_sp_err = builder.ins().icmp_imm_s(IntCC::NotEqual, sp_res, 0);
                    builder
                        .ins()
                        .brif(is_sp_err, err_dispatch_bb, &[], sp_reload_bb, &[]);

                    builder.switch_to_block(sp_reload_bb);
                    emit_promoted_counters_load(&mut builder, promoted);
                    let allow_arm_no = builder.ins().iconst(types::I8, 0);
                    builder
                        .ins()
                        .jump(gas_check_bb, &[BlockArg::Value(allow_arm_no)]);

                    builder.switch_to_block(err_dispatch_bb);
                    let is_deopt = builder.ins().icmp_imm_s(IntCC::Equal, sp_res, MIR_STATUS_DEOPT);
                    builder
                        .ins()
                        .brif(is_deopt, deopt_dispatch_bb, &[], emergency_exit_bb, &[]);

                    builder.switch_to_block(deopt_dispatch_bb);
                    let mut deopt_capture_bbs = Vec::with_capacity(hub.sites.len());
                    for _ in &hub.sites {
                        let dbb = builder.create_block();
                        builder.set_cold_block(dbb);
                        rec.post_flush_clif_blocks.insert(dbb);
                        deopt_capture_bbs.push(dbb);
                    }
                    if hub.sites.len() <= 4 {
                        for (idx, &(site_id, ..)) in hub.sites.iter().enumerate() {
                            let is_last = idx + 1 == hub.sites.len();
                            if is_last {
                                builder.ins().jump(deopt_capture_bbs[idx], &[]);
                            } else {
                                let next_chk = builder.create_block();
                                builder.set_cold_block(next_chk);
                                rec.post_flush_clif_blocks.insert(next_chk);
                                let is_site = builder.ins().icmp_imm_s(
                                    IntCC::Equal,
                                    hub.site_id_param,
                                    site_id,
                                );
                                builder.ins().brif(
                                    is_site,
                                    deopt_capture_bbs[idx],
                                    &[],
                                    next_chk,
                                    &[],
                                );
                                builder.switch_to_block(next_chk);
                            }
                        }
                    } else {
                        let blocks_before: hashbrown::HashSet<Block> =
                            builder.func.layout.blocks().collect();
                        let mut sw = cranelift_frontend::Switch::new();
                        for (idx, &(site_id, ..)) in hub.sites.iter().enumerate() {
                            if idx + 1 < hub.sites.len() {
                                sw.set_entry(site_id as u128, deopt_capture_bbs[idx]);
                            }
                        }
                        sw.emit(&mut builder, hub.site_id_param, *deopt_capture_bbs.last().unwrap());
                        let new_blocks: Vec<Block> = builder
                            .func
                            .layout
                            .blocks()
                            .filter(|b| !blocks_before.contains(b))
                            .collect();
                        for b in new_blocks {
                            builder.set_cold_block(b);
                            rec.post_flush_clif_blocks.insert(b);
                        }
                    }
                    for (idx, &(_, target_block_idx, ref live_locals, _)) in
                        hub.sites.iter().enumerate()
                    {
                        builder.switch_to_block(deopt_capture_bbs[idx]);
                        let target_pc_val = builder.ins().iconst(types::I64, target_block_idx as i64);
                        builder.ins().call(init_deopt_fn, &[target_pc_val]);
                        for &loc_id in live_locals {
                            if loc_id.0 < local_vars.len() {
                                let loc_idx_val =
                                    builder.ins().iconst(types::I64, loc_id.0 as i64);
                                let cur_val = if let Some(&slot) = stack_slots.get(&loc_id.0) {
                                    let mir_ty = mir_fn
                                        .locals
                                        .get(loc_id.0)
                                        .map(|l| l.ty.clone())
                                        .unwrap_or(MirType::Int);
                                    let clif_ty = mir_to_clif_type(&mir_ty).unwrap_or(types::I64);
                                    builder.ins().stack_load(types::I64, clif_ty, slot, 0)
                                } else {
                                    builder.use_var(local_vars[loc_id.0])
                                };
                                let val_ty = builder.func.dfg.value_type(cur_val);
                                let val_i64 = if val_ty == types::I8 {
                                    builder.ins().uextend(types::I64, cur_val)
                                } else if val_ty == types::F64 {
                                    builder.ins().bitcast(types::I64, MemFlagsData::new(), cur_val)
                                } else {
                                    cur_val
                                };
                                builder.ins().call(
                                    record_deopt_local_fn,
                                    &[target_pc_val, loc_idx_val, val_i64],
                                );
                            }
                        }
                        builder.ins().jump(emergency_exit_bb, &[]);
                    }

                    builder.switch_to_block(gas_check_bb);
                    let gas0 = builder.use_var(gas_var);
                    let gas1 = builder.ins().iadd_imm_s(gas0, -(amt as i64));
                    builder.def_var(gas_var, gas1);
                    let is_neg = builder.ins().icmp_imm_s(IntCC::SignedLessThan, gas1, 0);
                    builder
                        .ins()
                        .brif(is_neg, gas_slow_bb, &[], arm_quantum_bb, &[]);

                    builder.switch_to_block(gas_slow_bb);
                    emit_promoted_counters_flush_exact(&mut builder, promoted);
                    let slow_call = builder.ins().call(gas_slow_fn, &[]);
                    let slow_res = if builder.inst_results(slow_call).is_empty() {
                        const OFF_GAS_SLOW_RESULT: i32 =
                            core::mem::offset_of!(MirContext, gas_slow_result) as i32;
                        let (gas_ptr_var, _) = promoted
                            .gas
                            .expect("quantized hub requires promoted gas ptr");
                        let gas_ptr = builder.use_var(gas_ptr_var);
                        builder.ins().load(
                            types::I64,
                            MemFlagsData::trusted(),
                            gas_ptr,
                            OFF_GAS_SLOW_RESULT,
                        )
                    } else {
                        builder.inst_results(slow_call)[0]
                    };
                    let is_gas_err = builder.ins().icmp_imm_s(IntCC::NotEqual, slow_res, 0);
                    builder
                        .ins()
                        .brif(is_gas_err, emergency_exit_bb, &[], gas_reload_bb, &[]);

                    builder.switch_to_block(gas_reload_bb);
                    emit_promoted_counters_load(&mut builder, promoted);
                    builder.ins().jump(cont_dispatch_bb, &[]);

                    builder.switch_to_block(arm_quantum_bb);
                    let threshold = QUANTUM_FAST_ITERS * (amt as i64);
                    let can_arm_gas = builder
                        .ins()
                        .icmp_imm_s(IntCC::SignedGreaterThanOrEqual, gas1, threshold);
                    let can_arm = builder.ins().band(allow_arm_param, can_arm_gas);
                    let c63 = builder.ins().iconst(types::I64, QUANTUM_FAST_ITERS);
                    let c0 = builder.ins().iconst(types::I64, 0);
                    let next_q = builder.ins().select(can_arm, c63, c0);
                    let gas_reserved = builder.ins().iadd_imm_s(gas1, -threshold);
                    let next_gas = builder.ins().select(can_arm, gas_reserved, gas1);
                    let sp_cur = builder.use_var(sp_cnt_var);
                    let sp_reserved = builder.ins().iadd_imm_s(sp_cur, QUANTUM_FAST_ITERS);
                    let next_sp = builder.ins().select(can_arm, sp_reserved, sp_cur);
                    builder.def_var(quantum_rem_var, next_q);
                    builder.def_var(gas_var, next_gas);
                    builder.def_var(sp_cnt_var, next_sp);
                    builder.ins().jump(cont_dispatch_bb, &[]);

                    builder.switch_to_block(cont_dispatch_bb);
                    if hub.sites.len() <= 4 {
                        for (idx, &(site_id, _, _, cont_bb)) in hub.sites.iter().enumerate() {
                            let is_last = idx + 1 == hub.sites.len();
                            if is_last {
                                builder.ins().jump(cont_bb, &[]);
                            } else {
                                let next_chk = builder.create_block();
                                builder.set_cold_block(next_chk);
                                rec.loop_clif_blocks.insert(next_chk);
                                let is_site = builder.ins().icmp_imm_s(
                                    IntCC::Equal,
                                    hub.site_id_param,
                                    site_id,
                                );
                                builder.ins().brif(is_site, cont_bb, &[], next_chk, &[]);
                                builder.switch_to_block(next_chk);
                            }
                        }
                    } else {
                        let blocks_before: hashbrown::HashSet<Block> =
                            builder.func.layout.blocks().collect();
                        let mut sw = cranelift_frontend::Switch::new();
                        for (idx, &(site_id, _, _, cont_bb)) in hub.sites.iter().enumerate() {
                            if idx + 1 < hub.sites.len() {
                                sw.set_entry(site_id as u128, cont_bb);
                            }
                        }
                        sw.emit(&mut builder, hub.site_id_param, hub.sites.last().unwrap().3);
                        let new_blocks: Vec<Block> = builder
                            .func
                            .layout
                            .blocks()
                            .filter(|b| !blocks_before.contains(b))
                            .collect();
                        for b in new_blocks {
                            builder.set_cold_block(b);
                            rec.loop_clif_blocks.insert(b);
                        }
                    }
                }

                // Emit PERF-T4-2 Barrier 3 (`loop_emergency_flush_bb`) and Barrier 4 (`exit_flush_bb`) blocks
                for (cl_idx, state) in cl_states.iter().enumerate() {
                    builder.switch_to_block(state.emergency_flush_bb);
                    emit_promoted_counters_flush(&mut builder, state.counters);
                    builder.ins().jump(emergency_exit_bb, &[]);

                    for &(dest_bb, exit_flush_bb) in &exit_flush_blocks_per_loop[cl_idx] {
                        builder.switch_to_block(exit_flush_bb);
                        emit_promoted_counters_flush(&mut builder, state.counters);
                        builder.ins().jump(dest_bb, &[]);
                    }
                }

                // Emit NG-W14-1 inner quantized loop emergency flush & exit reconcile/flush blocks
                for (iq_idx, iq_state) in iq_states.iter().enumerate() {
                    builder.switch_to_block(iq_state.emergency_flush_bb);
                    emit_promoted_counters_flush(&mut builder, iq_state.counters);
                    builder.ins().jump(emergency_exit_bb, &[]);

                    for &(dest_bb, exit_bb, exits_outer_cl) in &iq_exit_blocks_per_loop[iq_idx] {
                        builder.switch_to_block(exit_bb);
                        if exits_outer_cl {
                            emit_promoted_counters_flush(&mut builder, iq_state.counters);
                        } else {
                            emit_quantum_reconcile_to_regs(&mut builder, iq_state.counters);
                        }
                        builder.ins().jump(dest_bb, &[]);
                    }
                }

                builder.switch_to_block(emergency_exit_bb);
                if ret_clif_ty != types::I8 || mir_fn.signature.return_type != MirType::Unit {
                    if ret_clif_ty == types::F64 {
                        let zero = builder.ins().f64const(0.0);
                        builder.ins().return_(&[zero]);
                    } else {
                        let stat_call = builder.ins().call(status_fn, &[]);
                        let stat_val = builder.inst_results(stat_call)[0];
                        let ret_val = cast_clif_type(&mut builder, stat_val, ret_clif_ty);
                        builder.ins().return_(&[ret_val]);
                    }
                } else {
                    builder.ins().return_(&[]);
                }

                builder.seal_all_blocks();
                // PERF-T4-2 Barrier 5 + NG-W14-1: compile-time CFG exit-completeness verification
                #[cfg(debug_assertions)]
                verify_promoted_loop_exit_completeness(&builder.func, &cfg_records, &iq_cfg_records)?;
                #[cfg(not(debug_assertions))]
                let _ = (&cfg_records, &iq_cfg_records);
                builder.finalize(self.module.target_config());
            }

            if self.capture_ir {
                let ir_text = self.format_captured_ir();
                let key = ir_capture_key.unwrap_or(&mir_fn.signature.name).to_string();
                self.captured_ir.insert(key, ir_text);
            }

            self.module
                .define_function(func_id, &mut self.ctx)
                .map_err(|e| {
                    eprintln!("Verifier error in define_function: {:?}", e);
                    MirCraneliftBackendError::CodegenError(format!("{e:?}"))
                })?;
            if self.capture_ir {
                let opt_ir_text = self.format_captured_ir();
                let key = ir_capture_key.unwrap_or(&mir_fn.signature.name).to_string();
                self.captured_opt_ir.insert(key, opt_ir_text);
            }
            let code_size = self
                .ctx
                .compiled_code()
                .map(|c| c.code_info().total_size as usize)
                .unwrap_or(1);
            self.compiled_code_sizes.insert(func_id, code_size);
            self.module.clear_context(&mut self.ctx);
            Ok(())
        }
    }

    #[inline(always)]
    fn is_strict_scalar_type(ty: &MirType) -> bool {
        matches!(ty, MirType::Int | MirType::Float | MirType::Bool)
    }

    #[inline(always)]
    fn is_builtin_or_special_callee(callee: &str) -> bool {
        is_effect_or_handler_callee(callee)
            || callee == "print"
            || callee == "println"
            || CraneliftJitCompiler::is_simd_callee_name(callee)
            || crate::mir::ir::parse_mir_struct_decode_callee(callee).is_some()
            || matches!(
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
                    | "table.push"
                    | "array.push"
                    | "string.len"
                    | "string.sub"
                    | "string.byte"
                    | "string.byte_at"
                    | "string.gsub"
                    | "string.split"
                    | "string.find"
                    | "tostring"
                    | "map.len"
                    | "map.get"
                    | "map.set"
                    | "map.has"
                    | "map.get_unwrap"
                    | "map.get_unwrap_or"
                    | "map.new"
                    | "enum.tag"
                    | "enum.payload"
                    | "enum.payload_at"
                    | "enum.new"
                    | "enum.alloc"
                    | "enum.new2"
                    | "enum.alloc2"
                    | "queue.new"
                    | "queue.offer_int"
                    | "queue.take_int"
                    | "__mir_unknown_is_int"
                    | "__mir_unknown_unwrap_int"
                    | "__mir_unknown_get_field"
                    | "__mir_unknown_call_method"
            )
    }

    #[inline(always)]
    fn scalar_operand_type(op: &Operand, locals: &[LocalDecl]) -> Option<MirType> {
        match op {
            Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => locals
                .get(loc.0)
                .map(|d| &d.ty)
                .filter(|ty| is_strict_scalar_type(ty))
                .cloned(),
            Operand::Constant(MirConstant::Int(_)) => Some(MirType::Int),
            Operand::Constant(MirConstant::Float(_)) => Some(MirType::Float),
            Operand::Constant(MirConstant::Bool(_)) => Some(MirType::Bool),
            _ => None,
        }
    }

    /// Strict closed-world allowlist check (fail-closed default: returns `false` on any
    /// non-whitelisted operation, place projection, allocation, non-scalar type, resource
    /// check, raise/unreachable, or non-module-local call).
    fn local_body_is_whitelisted_non_failing(
        f: &MirFunction,
        known_fns: &hashbrown::HashSet<&str>,
    ) -> bool {
        if f.blocks.is_empty() || !is_strict_scalar_type(&f.signature.return_type) {
            return false;
        }
        for &p in &f.params {
            if !f
                .locals
                .get(p.0)
                .is_some_and(|d| is_strict_scalar_type(&d.ty))
            {
                return false;
            }
        }
        for block in &f.blocks {
            for stmt in &block.statements {
                match stmt {
                    Statement::StorageLive(_, _) | Statement::StorageDead(_, _) => {}
                    Statement::Assign(Place::Local(dst), rvalue, _) => {
                        if !f
                            .locals
                            .get(dst.0)
                            .is_some_and(|d| is_strict_scalar_type(&d.ty))
                        {
                            return false;
                        }
                        match rvalue {
                            Rvalue::Use(op) => {
                                if scalar_operand_type(op, &f.locals).is_none() {
                                    return false;
                                }
                            }
                            Rvalue::BinaryOp(op, lhs, rhs) => {
                                let allowed_op = matches!(
                                    op,
                                    MirBinaryOp::Add
                                        | MirBinaryOp::Sub
                                        | MirBinaryOp::Mul
                                        | MirBinaryOp::BitAnd
                                        | MirBinaryOp::BitOr
                                        | MirBinaryOp::BitXor
                                        | MirBinaryOp::Shl
                                        | MirBinaryOp::Shr
                                        | MirBinaryOp::Eq
                                        | MirBinaryOp::NotEq
                                        | MirBinaryOp::Lt
                                        | MirBinaryOp::LtEq
                                        | MirBinaryOp::Gt
                                        | MirBinaryOp::GtEq
                                        | MirBinaryOp::And
                                        | MirBinaryOp::Or
                                );
                                if !allowed_op
                                    || scalar_operand_type(lhs, &f.locals).is_none()
                                    || scalar_operand_type(rhs, &f.locals).is_none()
                                {
                                    return false;
                                }
                            }
                            Rvalue::UnaryOp(op, operand) => {
                                let allowed_op = matches!(
                                    op,
                                    MirUnaryOp::Neg | MirUnaryOp::Not | MirUnaryOp::BitNot
                                );
                                if !allowed_op
                                    || scalar_operand_type(operand, &f.locals).is_none()
                                {
                                    return false;
                                }
                            }
                            Rvalue::Cast(operand, target_ty) => {
                                let Some(src_ty) = scalar_operand_type(operand, &f.locals) else {
                                    return false;
                                };
                                if !matches!(
                                    (src_ty, target_ty),
                                    (MirType::Int, MirType::Float) | (MirType::Float, MirType::Int)
                                ) {
                                    return false;
                                }
                            }
                            Rvalue::Aggregate(_, _) | Rvalue::Closure { .. } => return false,
                        }
                    }
                    // Fail-closed on Place::Index, Place::Field, Statement::ResourceCheck, etc.
                    _ => return false,
                }
            }

            match &block.terminator {
                Some(Terminator::Goto(_, _)) => {}
                Some(Terminator::Cond { cond, .. }) => {
                    if scalar_operand_type(cond, &f.locals).is_none() {
                        return false;
                    }
                }
                Some(Terminator::Switch { discr, .. }) => {
                    if scalar_operand_type(discr, &f.locals).is_none() {
                        return false;
                    }
                }
                Some(Terminator::Return(Some(op), _)) => {
                    if scalar_operand_type(op, &f.locals).is_none() {
                        return false;
                    }
                }
                Some(Terminator::Call {
                    callee,
                    args,
                    destination: Place::Local(dst),
                    error: None,
                    ..
                }) => {
                    if is_builtin_or_special_callee(callee)
                        || !known_fns.contains(callee.as_str())
                        || !f
                            .locals
                            .get(dst.0)
                            .is_some_and(|d| is_strict_scalar_type(&d.ty))
                        || args
                            .iter()
                            .any(|a| scalar_operand_type(a, &f.locals).is_none())
                    {
                        return false;
                    }
                }
                // Fail-closed on None, Return(None), Raise, Unreachable, Call with unwind/projected dst
                _ => return false,
            }
        }
        true
    }

    fn analyze_refined_module_call_effects(
        refined_functions: &[MirFunction],
        gas_budget_active: bool,
    ) -> hashbrown::HashMap<String, (bool, bool)> {
        let known_fns: hashbrown::HashSet<&str> = refined_functions
            .iter()
            .map(|f| f.signature.name.as_str())
            .collect();

        let mut can_deopt: hashbrown::HashMap<String, bool> =
            hashbrown::HashMap::with_capacity(refined_functions.len());
        let mut local_callees: hashbrown::HashMap<String, Vec<String>> =
            hashbrown::HashMap::with_capacity(refined_functions.len());

        // 1. Direct seed for can_deopt_transitively(F)
        for f in refined_functions {
            let mut deopt = !crate::mir::opt::Safepoints::find_loop_headers_in_fn(f).is_empty();
            let mut callees = Vec::new();
            for b in &f.blocks {
                if let Some(Terminator::Call { callee, .. }) = &b.terminator {
                    if is_effect_or_handler_callee(callee)
                        || is_builtin_or_special_callee(callee)
                        || !known_fns.contains(callee.as_str())
                    {
                        deopt = true;
                    } else {
                        callees.push(callee.clone());
                    }
                }
            }
            can_deopt.insert(f.signature.name.clone(), deopt);
            local_callees.insert(f.signature.name.clone(), callees);
        }

        // 2. Fixed-point propagation for can_deopt_transitively
        loop {
            let mut changed = false;
            for f in refined_functions {
                let name = &f.signature.name;
                if *can_deopt.get(name).unwrap_or(&true) {
                    continue;
                }
                if let Some(callees) = local_callees.get(name) {
                    if callees.iter().any(|c| *can_deopt.get(c).unwrap_or(&true)) {
                        can_deopt.insert(name.clone(), true);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // 3. Direct seed for can_fail_transitively(F) (strict closed-world allowlist, fail-closed)
        let mut can_fail: hashbrown::HashMap<String, bool> =
            hashbrown::HashMap::with_capacity(refined_functions.len());
        for f in refined_functions {
            let name = &f.signature.name;
            let deopt = *can_deopt.get(name).unwrap_or(&true);
            let fail = gas_budget_active
                || deopt
                || !local_body_is_whitelisted_non_failing(f, &known_fns);
            can_fail.insert(name.clone(), fail);
        }

        // 4. Fixed-point propagation for can_fail_transitively
        loop {
            let mut changed = false;
            for f in refined_functions {
                let name = &f.signature.name;
                if *can_fail.get(name).unwrap_or(&true) {
                    continue;
                }
                if let Some(callees) = local_callees.get(name) {
                    if callees.iter().any(|c| {
                        *can_fail.get(c).unwrap_or(&true) || *can_deopt.get(c).unwrap_or(&true)
                    }) {
                        can_fail.insert(name.clone(), true);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        let mut out = hashbrown::HashMap::with_capacity(refined_functions.len());
        for f in refined_functions {
            let name = &f.signature.name;
            let d = *can_deopt.get(name).unwrap_or(&true);
            let fl = *can_fail.get(name).unwrap_or(&true);
            out.insert(name.clone(), (d, fl));
        }
        out
    }

    /// Compile-time closed-world call effect analysis over a `MirModule`.
    /// Returns `map[func_name] = (can_deopt_transitively, can_fail_transitively)`.
    pub fn analyze_module_call_effects(
        module: &MirModule,
        gas_budget_active: bool,
    ) -> hashbrown::HashMap<String, (bool, bool)> {
        let mut struct_layouts = hashbrown::HashMap::new();
        for layout in &module.struct_layouts {
            struct_layouts.insert(layout.type_name.clone(), layout.clone());
        }
        let callee_rets: hashbrown::HashMap<String, MirType> = module
            .functions
            .iter()
            .map(|f| (f.signature.name.clone(), f.signature.return_type.clone()))
            .collect();
        let mut refined_functions = Vec::with_capacity(module.functions.len());
        for f in &module.functions {
            let mut f_with_calls = f.clone();
            apply_module_callee_return_types(&mut f_with_calls, &callee_rets);
            refined_functions.push(refine_mir_function_with_layouts(&f_with_calls, &struct_layouts));
        }
        analyze_refined_module_call_effects(&refined_functions, gas_budget_active)
    }

    pub(crate) fn mir_type_contains_unknown(ty: &MirType) -> bool {
        match ty {
            MirType::Unknown => true,
            MirType::Array(inner) => mir_type_contains_unknown(inner),
            MirType::Option(inner) => mir_type_contains_unknown(inner),
            MirType::Tuple(elems) => elems.iter().any(mir_type_contains_unknown),
            MirType::Map(k, v) => mir_type_contains_unknown(k) || mir_type_contains_unknown(v),
            MirType::Function { params, return_type } => {
                params.iter().any(mir_type_contains_unknown)
                    || mir_type_contains_unknown(return_type)
            }
            _ => false,
        }
    }

    pub(crate) fn unify_mir_types(a: &MirType, b: &MirType) -> Option<MirType> {
        if a == b {
            return Some(a.clone());
        }
        match (a, b) {
            (MirType::Unknown, other) | (other, MirType::Unknown) => Some(other.clone()),
            (MirType::Array(ea), MirType::Array(eb)) => {
                unify_mir_types(ea, eb).map(|u| MirType::Array(Box::new(u)))
            }
            (MirType::Option(oa), MirType::Option(ob)) => {
                unify_mir_types(oa, ob).map(|u| MirType::Option(Box::new(u)))
            }
            (MirType::Tuple(ta), MirType::Tuple(tb)) if ta.len() == tb.len() => {
                let mut out = Vec::with_capacity(ta.len());
                for (ea, eb) in ta.iter().zip(tb.iter()) {
                    out.push(unify_mir_types(ea, eb)?);
                }
                Some(MirType::Tuple(out))
            }
            (MirType::Map(ka, va), MirType::Map(kb, vb)) => {
                let k = unify_mir_types(ka, kb)?;
                let v = unify_mir_types(va, vb)?;
                Some(MirType::Map(Box::new(k), Box::new(v)))
            }
            (MirType::Enum(n1), MirType::Enum(n2)) if n1 == "Enum" => {
                Some(MirType::Enum(n2.clone()))
            }
            (MirType::Enum(n1), MirType::Enum(n2)) if n2 == "Enum" => {
                Some(MirType::Enum(n1.clone()))
            }
            (MirType::Option(inner), MirType::Enum(n))
            | (MirType::Enum(n), MirType::Option(inner))
                if n == "Option" =>
            {
                Some(MirType::Option(inner.clone()))
            }
            (MirType::Int, MirType::Enum(n)) | (MirType::Enum(n), MirType::Int)
                if n == "Option" || n == "Result" =>
            {
                Some(MirType::Enum(n.clone()))
            }
            _ => None,
        }
    }

    pub fn apply_module_callee_return_types(
        mir_fn: &mut MirFunction,
        callee_rets: &hashbrown::HashMap<String, MirType>,
    ) {
        for block in &mut mir_fn.blocks {
            if let Some(Terminator::Call {
                callee,
                destination: Place::Local(dst),
                ..
            }) = &block.terminator
            {
                if let Some(ret_ty) = callee_rets.get(callee) {
                    if !matches!(ret_ty, MirType::Unit | MirType::Unknown)
                        && !mir_fn.params.contains(dst)
                    {
                        if let Some(decl) = mir_fn.locals.get_mut(dst.0) {
                            if !decl.is_param
                                && (decl.ty == MirType::Unknown
                                    || (decl.ty == MirType::Int && decl.name.is_none()))
                            {
                                decl.ty = ret_ty.clone();
                            }
                        }
                    }
                }
            }
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    enum TypeLattice {
        Unconstrained,
        Concrete(MirType),
        Conflict(String),
    }

    impl TypeLattice {
        fn merge_evidence(&mut self, incoming: &MirType, reason: &str) -> bool {
            if *incoming == MirType::Unknown {
                return false;
            }
            match self {
                TypeLattice::Unconstrained => {
                    *self = TypeLattice::Concrete(incoming.clone());
                    true
                }
                TypeLattice::Concrete(existing) => {
                    if existing == incoming {
                        false
                    } else if let Some(unified) = unify_mir_types(existing, incoming) {
                        if unified != *existing {
                            *existing = unified;
                            true
                        } else {
                            false
                        }
                    } else {
                        *self = TypeLattice::Conflict(format!(
                            "conflict between '{existing}' and '{incoming}' ({reason})"
                        ));
                        true
                    }
                }
                TypeLattice::Conflict(_) => false,
            }
        }
    }

    pub fn refine_locals(
        locals: &[LocalDecl],
        blocks: &[BasicBlock],
        return_type: &MirType,
    ) -> Vec<LocalDecl> {
        refine_locals_with_params(locals, blocks, return_type, &[])
    }

    pub fn refine_locals_with_params(
        locals: &[LocalDecl],
        blocks: &[BasicBlock],
        return_type: &MirType,
        params: &[LocalId],
    ) -> Vec<LocalDecl> {
        let dummy_layouts = hashbrown::HashMap::new();
        refine_locals_with_context(locals, blocks, return_type, params, &dummy_layouts)
    }

    pub fn refine_locals_with_context(
        locals: &[LocalDecl],
        blocks: &[BasicBlock],
        return_type: &MirType,
        params: &[LocalId],
        struct_layouts: &hashbrown::HashMap<String, MirStructLayout>,
    ) -> Vec<LocalDecl> {
        refine_locals_internal(locals, blocks, return_type, params, struct_layouts, None)
    }

    /// Explicit test seam for testing local variable convergence boundaries.
    /// Available only under the `test-deopt` feature.
    #[cfg(feature = "test-deopt")]
    pub fn refine_locals_with_seam(
        locals: &[LocalDecl],
        blocks: &[BasicBlock],
        return_type: &MirType,
        params: &[LocalId],
        struct_layouts: &hashbrown::HashMap<String, MirStructLayout>,
        max_iterations_override: Option<usize>,
    ) -> Vec<LocalDecl> {
        refine_locals_internal(locals, blocks, return_type, params, struct_layouts, max_iterations_override)
    }

    fn refine_locals_internal(
        locals: &[LocalDecl],
        blocks: &[BasicBlock],
        return_type: &MirType,
        params: &[LocalId],
        struct_layouts: &hashbrown::HashMap<String, MirStructLayout>,
        max_iterations_override: Option<usize>,
    ) -> Vec<LocalDecl> {
        let mut inferred = locals.to_vec();

        // 1. Compute reachable basic blocks from CFG entry to ignore dead unreachable code.
        // If blocks lack terminators (e.g. synthetic test blocks), treat all blocks as reachable.
        let all_have_terminators = !blocks.is_empty() && blocks.iter().all(|b| b.terminator.is_some());
        let mut reachable_blocks = hashbrown::HashSet::new();
        if all_have_terminators {
            let mut queue = std::collections::VecDeque::new();
            let entry_id = blocks[0].id;
            reachable_blocks.insert(entry_id);
            queue.push_back(entry_id);
            while let Some(bid) = queue.pop_front() {
                if let Some(b) = blocks.iter().find(|b| b.id == bid) {
                    if let Some(term) = &b.terminator {
                        for succ_idx in term.successors() {
                            let target = BlockId(succ_idx);
                            if reachable_blocks.insert(target) {
                                queue.push_back(target);
                            }
                        }
                    }
                }
            }
        } else {
            for b in blocks {
                reachable_blocks.insert(b.id);
            }
        }

        // 2. Compute definition counts across reachable blocks to detect multiply-defined mutable locals.
        // Projected writes (Place::Field, Place::Index) mutate the root local and MUST increment def_count.
        let mut def_count = hashbrown::HashMap::<LocalId, usize>::new();
        for block in blocks {
            if !reachable_blocks.contains(&block.id) {
                continue;
            }
            for stmt in &block.statements {
                if let Statement::Assign(place, _, _) = stmt {
                    *def_count.entry(place.root_local()).or_insert(0) += 1;
                }
            }
            if let Some(Terminator::Call { destination, .. }) = &block.terminator {
                *def_count.entry(destination.root_local()).or_insert(0) += 1;
            }
        }

        // 3. Identify formal parameters using explicit parameter metadata (l.is_param or explicit params slice).
        // Never rely on unstable local variable names or "argN" prefix heuristics.
        let is_param_set: hashbrown::HashSet<LocalId> = locals
            .iter()
            .enumerate()
            .filter(|(i, l)| l.is_param || params.contains(&LocalId(*i)))
            .map(|(i, _)| LocalId(i))
            .collect();

        let mut lattice: Vec<TypeLattice> = locals
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let is_param = is_param_set.contains(&LocalId(i));
                if l.name.as_deref() == Some("_return") {
                    if *return_type != MirType::Unknown {
                        TypeLattice::Concrete(return_type.clone())
                    } else {
                        TypeLattice::Unconstrained
                    }
                } else if is_param {
                    if l.ty != MirType::Unknown {
                        TypeLattice::Concrete(l.ty.clone())
                    } else {
                        TypeLattice::Unconstrained
                    }
                } else if l.ty == MirType::Unknown || l.ty == MirType::Int {
                    TypeLattice::Unconstrained
                } else {
                    TypeLattice::Concrete(l.ty.clone())
                }
            })
            .collect();

        // Safe finite monotone fixed-point loop.
        // Height of lattice is 2 (Unconstrained -> Concrete -> Conflict).
        // Max iterations dynamically sized based on locals count, with explicit test seam override support.
        let max_iterations = max_iterations_override
            .unwrap_or_else(|| (locals.len() * 4).max(64));
        let mut converged = false;
        for _ in 0..max_iterations {
            let mut changed = false;

            // Reflect current proven concrete types into inferred working slice for try_operand_type / try_place_type
            for (i, lat) in lattice.iter().enumerate() {
                match lat {
                    TypeLattice::Concrete(ty) => inferred[i].ty = ty.clone(),
                    TypeLattice::Conflict(_) | TypeLattice::Unconstrained => {
                        inferred[i].ty = MirType::Unknown;
                    }
                }
            }

            for block in blocks {
                if !reachable_blocks.contains(&block.id) {
                    continue;
                }
                for stmt in &block.statements {
                    if let Statement::Assign(Place::Local(loc), rval, _) = stmt {
                        if loc.0 >= lattice.len() {
                            continue;
                        }
                        if matches!(lattice[loc.0], TypeLattice::Conflict(_)) {
                            continue;
                        }

                        match rval {
                            Rvalue::Use(Operand::Constant(MirConstant::Int(_))) => {
                                if lattice[loc.0].merge_evidence(&MirType::Int, "int const") {
                                    changed = true;
                                }
                            }
                            Rvalue::Use(Operand::Constant(MirConstant::Float(_))) => {
                                if lattice[loc.0].merge_evidence(&MirType::Float, "float const") {
                                    changed = true;
                                }
                            }
                            Rvalue::Use(Operand::Constant(MirConstant::String(_))) => {
                                if lattice[loc.0].merge_evidence(&MirType::String, "string const") {
                                    changed = true;
                                }
                            }
                            Rvalue::Use(Operand::Constant(MirConstant::Bool(_))) => {
                                if lattice[loc.0].merge_evidence(&MirType::Bool, "bool const") {
                                    changed = true;
                                }
                            }
                            Rvalue::Use(Operand::Constant(MirConstant::Unit)) => {
                                if lattice[loc.0].merge_evidence(&MirType::Unit, "unit const") {
                                    changed = true;
                                }
                            }
                            Rvalue::Use(Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src))) => {
                                if src.0 < lattice.len() {
                                    // Forward copy: src -> loc
                                    let src_state = lattice[src.0].clone();
                                    match src_state {
                                        TypeLattice::Concrete(ref src_ty) => {
                                            if lattice[loc.0].merge_evidence(src_ty, "local copy") {
                                                changed = true;
                                            }
                                        }
                                        TypeLattice::Conflict(reason) => {
                                            let conf_msg = format!("copied conflict from local {src:?}: {reason}");
                                            if !matches!(&lattice[loc.0], TypeLattice::Conflict(_)) {
                                                lattice[loc.0] = TypeLattice::Conflict(conf_msg);
                                                changed = true;
                                            }
                                        }
                                        TypeLattice::Unconstrained => {}
                                    }

                                    // Backward copy: ONLY sound if `loc` is uniquely defined by this single definition
                                    // and is not a multiply-defined mutable local or parameter.
                                    let is_singly_defined = def_count.get(loc).copied().unwrap_or(0) == 1 && !is_param_set.contains(loc);
                                    if is_singly_defined {
                                        let loc_state = lattice[loc.0].clone();
                                        match loc_state {
                                            TypeLattice::Concrete(ref loc_ty) => {
                                                if lattice[src.0].merge_evidence(loc_ty, "copy destination constraint") {
                                                    changed = true;
                                                }
                                            }
                                            TypeLattice::Conflict(reason) => {
                                                let conf_msg = format!("propagated conflict from unique copy destination {loc:?}: {reason}");
                                                if !matches!(&lattice[src.0], TypeLattice::Conflict(_)) {
                                                    lattice[src.0] = TypeLattice::Conflict(conf_msg);
                                                    changed = true;
                                                }
                                            }
                                            TypeLattice::Unconstrained => {}
                                        }
                                    }
                                }
                            }
                            Rvalue::Use(Operand::Copy(p) | Operand::Move(p)) => {
                                if let Ok(p_ty) = try_place_type(p, &inferred, struct_layouts) {
                                    if p_ty != MirType::Unknown {
                                        if lattice[loc.0].merge_evidence(&p_ty, "place copy") {
                                            changed = true;
                                        }
                                    }
                                }
                            }
                            Rvalue::Aggregate(ty, elems) => {
                                match ty {
                                    MirType::Array(decl_elem) => {
                                        let mut has_conflict = false;
                                        let mut any_unknown = false;
                                        let mut unified_elem: Option<MirType> = None;

                                        for elem in elems {
                                            if let Operand::Copy(Place::Local(src))
                                            | Operand::Move(Place::Local(src)) = elem
                                            {
                                                if src.0 < lattice.len() {
                                                    if let TypeLattice::Conflict(reason) = &lattice[src.0] {
                                                        lattice[loc.0] = TypeLattice::Conflict(format!(
                                                            "aggregate element conflict from {src:?}: {reason}"
                                                        ));
                                                        changed = true;
                                                        has_conflict = true;
                                                        break;
                                                    }
                                                }
                                            }
                                            match try_operand_type(elem, &inferred, struct_layouts) {
                                                Ok(op_ty) if op_ty != MirType::Unknown => {
                                                    match &unified_elem {
                                                        None => unified_elem = Some(op_ty),
                                                        Some(prev) => {
                                                            if let Some(u) = unify_mir_types(prev, &op_ty) {
                                                                unified_elem = Some(u);
                                                            } else {
                                                                lattice[loc.0] = TypeLattice::Conflict(format!(
                                                                    "heterogeneous array elements: '{prev}' vs '{op_ty}'"
                                                                ));
                                                                changed = true;
                                                                has_conflict = true;
                                                                break;
                                                            }
                                                        }
                                                    }
                                                }
                                                _ => {
                                                    any_unknown = true;
                                                }
                                            }
                                        }

                                        if !has_conflict {
                                            if elems.is_empty() {
                                                if lattice[loc.0].merge_evidence(ty, "empty array aggregate") {
                                                    changed = true;
                                                }
                                            } else if !any_unknown {
                                                if let Some(el_ty) = unified_elem {
                                                    if let Some(final_elem) = unify_mir_types(decl_elem, &el_ty) {
                                                        let resolved_arr = MirType::Array(Box::new(final_elem));
                                                        if lattice[loc.0].merge_evidence(&resolved_arr, "array aggregate") {
                                                            changed = true;
                                                        }
                                                    } else {
                                                        lattice[loc.0] = TypeLattice::Conflict(format!(
                                                            "array aggregate element '{el_ty}' conflicts with declared '{decl_elem}'"
                                                        ));
                                                        changed = true;
                                                    }
                                                }
                                            } else if let Some(el_ty) = unified_elem {
                                                if unify_mir_types(decl_elem, &el_ty).is_none() {
                                                    lattice[loc.0] = TypeLattice::Conflict(format!(
                                                        "array aggregate element '{el_ty}' conflicts with declared '{decl_elem}'"
                                                    ));
                                                    changed = true;
                                                } else if **decl_elem != MirType::Unknown {
                                                    if lattice[loc.0].merge_evidence(ty, "partial array aggregate") {
                                                        changed = true;
                                                    }
                                                }
                                            }
                                        }

                                        if let TypeLattice::Concrete(MirType::Array(expected_el)) =
                                            lattice[loc.0].clone()
                                        {
                                            if *expected_el != MirType::Unknown {
                                                for elem in elems {
                                                    if let Operand::Copy(Place::Local(src))
                                                    | Operand::Move(Place::Local(src)) = elem
                                                    {
                                                        let is_singly_defined =
                                                            def_count.get(src).copied().unwrap_or(0) == 1
                                                                && !is_param_set.contains(src);
                                                        if is_singly_defined && src.0 < lattice.len() {
                                                            if lattice[src.0].merge_evidence(
                                                                &expected_el,
                                                                "array aggregate element constraint",
                                                            ) {
                                                                changed = true;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    MirType::Tuple(decl_elems) => {
                                        if decl_elems.len() != elems.len() {
                                            lattice[loc.0] = TypeLattice::Conflict(format!(
                                                "tuple arity mismatch: declared {}, got {}",
                                                decl_elems.len(),
                                                elems.len()
                                            ));
                                            changed = true;
                                        } else {
                                            let mut unified_elems = Vec::with_capacity(elems.len());
                                            let mut has_conflict = false;
                                            for (decl_el, elem) in decl_elems.iter().zip(elems.iter()) {
                                                if let Operand::Copy(Place::Local(src))
                                                | Operand::Move(Place::Local(src)) = elem
                                                {
                                                    if src.0 < lattice.len() {
                                                        if let TypeLattice::Conflict(reason) = &lattice[src.0] {
                                                            lattice[loc.0] = TypeLattice::Conflict(format!(
                                                                "tuple element conflict from {src:?}: {reason}"
                                                            ));
                                                            changed = true;
                                                            has_conflict = true;
                                                            break;
                                                        }
                                                    }
                                                }
                                                let op_ty = try_operand_type(elem, &inferred, struct_layouts)
                                                    .unwrap_or(MirType::Unknown);
                                                if let Some(u) = unify_mir_types(decl_el, &op_ty) {
                                                    unified_elems.push(u);
                                                } else {
                                                    lattice[loc.0] = TypeLattice::Conflict(format!(
                                                        "tuple element '{op_ty}' conflicts with declared '{decl_el}'"
                                                    ));
                                                    changed = true;
                                                    has_conflict = true;
                                                    break;
                                                }
                                            }
                                            if !has_conflict {
                                                let resolved_tup = MirType::Tuple(unified_elems);
                                                if lattice[loc.0].merge_evidence(&resolved_tup, "tuple aggregate") {
                                                    changed = true;
                                                }
                                            }
                                        }
                                    }
                                    _ => {
                                        if lattice[loc.0].merge_evidence(ty, "aggregate") {
                                            changed = true;
                                        }
                                    }
                                }
                            }
                            Rvalue::Cast(_, target_ty) => {
                                if lattice[loc.0].merge_evidence(target_ty, "cast") {
                                    changed = true;
                                }
                            }
                            Rvalue::BinaryOp(binop, lhs, rhs) => {
                                let l_ty_res = try_operand_type(lhs, &inferred, struct_layouts);
                                let r_ty_res = try_operand_type(rhs, &inferred, struct_layouts);
                                if let (Ok(l_ty), Ok(r_ty)) = (l_ty_res, r_ty_res) {
                                    if l_ty != MirType::Unknown && r_ty != MirType::Unknown {
                                        let out_ty = if matches!(binop, MirBinaryOp::Concat) {
                                            Some(MirType::String)
                                        } else if matches!(
                                            binop,
                                            MirBinaryOp::Eq
                                                | MirBinaryOp::NotEq
                                                | MirBinaryOp::Lt
                                                | MirBinaryOp::LtEq
                                                | MirBinaryOp::Gt
                                                | MirBinaryOp::GtEq
                                        ) {
                                            Some(MirType::Bool)
                                        } else if l_ty == MirType::Float || r_ty == MirType::Float {
                                            Some(MirType::Float)
                                        } else if l_ty == MirType::String && r_ty == MirType::String {
                                            if matches!(binop, MirBinaryOp::Add) {
                                                Some(MirType::String)
                                            } else {
                                                Some(MirType::Bool)
                                            }
                                        } else if l_ty == MirType::Int && r_ty == MirType::Int {
                                            Some(MirType::Int)
                                        } else {
                                            None
                                        };
                                        if let Some(t) = out_ty {
                                            if lattice[loc.0].merge_evidence(&t, "binop") {
                                                changed = true;
                                            }
                                        }
                                    }
                                }
                            }
                            Rvalue::UnaryOp(unop, op) => {
                                if let Ok(o_ty) = try_operand_type(op, &inferred, struct_layouts) {
                                    if o_ty != MirType::Unknown {
                                        let out_ty = match unop {
                                            MirUnaryOp::Neg if o_ty == MirType::Float => Some(MirType::Float),
                                            MirUnaryOp::Neg if o_ty == MirType::Int => Some(MirType::Int),
                                            MirUnaryOp::Not if o_ty == MirType::Bool => Some(MirType::Bool),
                                            MirUnaryOp::BitNot if o_ty == MirType::Int => Some(MirType::Int),
                                            MirUnaryOp::Len
                                                 if o_ty == MirType::String
                                                     || matches!(o_ty, MirType::Array(_)) =>
                                             {
                                                 Some(MirType::Int)
                                             }
                                             _ => None,
                                         };
                                         if let Some(t) = out_ty {
                                             if lattice[loc.0].merge_evidence(&t, "unaryop") {
                                                 changed = true;
                                             }
                                         }
                                     }
                                 }
                             }
                            Rvalue::Closure { .. } => {}
                         }
                     }
                    if let Statement::Assign(place @ (Place::Field(..) | Place::Index(..)), rval, _) = stmt {
                        let root = place.root_local();
                        if root.0 < lattice.len() && !matches!(lattice[root.0], TypeLattice::Conflict(_)) {
                            if let Ok(field_ty) = try_place_type(place, &inferred, struct_layouts) {
                                if field_ty != MirType::Unknown {
                                    match rval {
                                        Rvalue::Use(Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src))) => {
                                            if src.0 < lattice.len() {
                                                let is_singly_defined = def_count.get(src).copied().unwrap_or(0) == 1 && !is_param_set.contains(src);
                                                if is_singly_defined {
                                                    if lattice[src.0].merge_evidence(&field_ty, "projected write destination constraint") {
                                                        changed = true;
                                                    }
                                                } else if let TypeLattice::Concrete(ref src_ty) = lattice[src.0] {
                                                    if *src_ty != field_ty {
                                                        lattice[root.0] = TypeLattice::Conflict(format!(
                                                            "type conflict in projected write: cannot store {src_ty} into field of type {field_ty}"
                                                        ));
                                                        changed = true;
                                                    }
                                                }
                                            }
                                        }
                                        Rvalue::Use(op) => {
                                            if let Ok(rval_ty) = try_operand_type(op, &inferred, struct_layouts) {
                                                if rval_ty != MirType::Unknown && rval_ty != field_ty {
                                                    lattice[root.0] = TypeLattice::Conflict(format!(
                                                        "type conflict in projected write: cannot store {rval_ty} into field of type {field_ty}"
                                                    ));
                                                    changed = true;
                                                }
                                            }
                                        }
                                        Rvalue::Aggregate(agg_ty, _) => {
                                            if *agg_ty != field_ty {
                                                lattice[root.0] = TypeLattice::Conflict(format!(
                                                    "type conflict in projected write: cannot store aggregate {agg_ty} into field of type {field_ty}"
                                                ));
                                                changed = true;
                                            }
                                        }
                                        Rvalue::Cast(_, target_ty) => {
                                            if *target_ty != field_ty {
                                                lattice[root.0] = TypeLattice::Conflict(format!(
                                                    "type conflict in projected write: cannot store cast {target_ty} into field of type {field_ty}"
                                                ));
                                                changed = true;
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                 }

                 if let Some(Terminator::Call { callee, args, destination: Place::Local(dst), .. }) = &block.terminator {
                     if (callee == "array.push" || callee == "table.push") && args.len() == 2 {
                         if let Operand::Copy(Place::Local(arr_loc)) | Operand::Move(Place::Local(arr_loc)) = &args[0] {
                             if arr_loc.0 < lattice.len() && !matches!(lattice[arr_loc.0], TypeLattice::Conflict(_)) {
                                 if let Ok(val_ty) = try_operand_type(&args[1], &inferred, struct_layouts) {
                                     if val_ty != MirType::Unknown {
                                         let expected_arr = MirType::Array(Box::new(val_ty));
                                         if lattice[arr_loc.0].merge_evidence(&expected_arr, "array.push element type") {
                                             changed = true;
                                         }
                                     }
                                 }
                             }
                         }
                     }
                     if callee == "map.set" && args.len() == 3 {
                         if let Operand::Copy(Place::Local(map_loc)) | Operand::Move(Place::Local(map_loc)) = &args[0] {
                             if map_loc.0 < lattice.len() && !matches!(lattice[map_loc.0], TypeLattice::Conflict(_)) {
                                 let k_ty = try_operand_type(&args[1], &inferred, struct_layouts)
                                     .unwrap_or(MirType::Unknown);
                                 let v_ty = try_operand_type(&args[2], &inferred, struct_layouts)
                                     .unwrap_or(MirType::Unknown);
                                 if k_ty != MirType::Unknown || v_ty != MirType::Unknown {
                                     let expected_map = MirType::Map(Box::new(k_ty), Box::new(v_ty));
                                     if lattice[map_loc.0].merge_evidence(&expected_map, "map.set kv type") {
                                         changed = true;
                                     }
                                 }
                             }
                         }
                     }
                     if dst.0 < lattice.len() && !matches!(lattice[dst.0], TypeLattice::Conflict(_)) {
                         let call_ret_ty: Option<MirType> = match callee.as_str() {
                             "math.abs" | "abs" => {
                                 let is_float = args.first().and_then(|a| try_operand_type(a, &inferred, struct_layouts).ok()) == Some(MirType::Float);
                                 if is_float { Some(MirType::Float) } else { Some(MirType::Int) }
                             }
                             "math.min" | "min" | "math.max" | "max" | "math.clamp" | "clamp" => {
                                 let is_float = args.iter().any(|a| try_operand_type(a, &inferred, struct_layouts).ok() == Some(MirType::Float));
                                 if is_float { Some(MirType::Float) } else { Some(MirType::Int) }
                             }
                             "table.len"
                             | "array.len"
                             | "map.len"
                             | "string.len"
                             | "string.byte_at"
                             | "enum.tag"
                             | "enum.payload"
                             | "queue.new"
                             | "queue.offer_int"
                             | "queue.take_int"
                             | "__mir_unknown_unwrap_int" => {
                                 Some(MirType::Int)
                             }
                             "__mir_unknown_is_int" => Some(MirType::Bool),
                             "enum.payload_at" => {
                                 if locals.get(dst.0).is_some_and(|l| l.ty == MirType::Unknown) {
                                     Some(MirType::Int)
                                 } else {
                                     None
                                 }
                             }
                             "tostring" | "string.sub" => Some(MirType::String),
                             "string.gsub" | "string.split" => {
                                 if matches!(locals.get(dst.0).map(|l| &l.ty), Some(MirType::Array(_))) {
                                     None
                                 } else {
                                     Some(MirType::Array(Box::new(MirType::String)))
                                 }
                             }
                             "string.byte" | "string.find" => {
                                 if locals.get(dst.0).is_some_and(|l| l.ty == MirType::Unknown) {
                                     Some(MirType::Array(Box::new(MirType::Int)))
                                 } else {
                                     None
                                 }
                             }
                             "map.has" => Some(MirType::Bool),
                             "map.get" => {
                                 if let Some(Ok(MirType::Map(_, v_ty))) =
                                     args.first().map(|a| try_operand_type(a, &inferred, struct_layouts))
                                     && *v_ty != MirType::Unknown
                                 {
                                     Some(MirType::Option(v_ty))
                                 } else {
                                     Some(MirType::Enum("Option".to_string()))
                                 }
                             }
                             "map.get_unwrap" | "map.get_unwrap_or" => {
                                 if let Some(Ok(MirType::Map(_, v_ty))) =
                                     args.first().map(|a| try_operand_type(a, &inferred, struct_layouts))
                                     && *v_ty != MirType::Unknown
                                 {
                                     Some(*v_ty)
                                 } else if let Some(Ok(def_ty)) =
                                     args.get(2).map(|a| try_operand_type(a, &inferred, struct_layouts))
                                     && def_ty != MirType::Unknown
                                 {
                                     Some(def_ty)
                                 } else if locals.get(dst.0).is_some_and(|l| l.ty == MirType::Unknown) {
                                     Some(MirType::Int)
                                 } else {
                                     None
                                 }
                             }
                             "map.set" | "array.push" | "table.push" => {
                                 Some(MirType::Unit)
                             }
                             "map.new" => {
                                 Some(MirType::Map(Box::new(MirType::Unknown), Box::new(MirType::Unknown)))
                             }
                             "enum.new" | "enum.alloc" | "enum.new2" | "enum.alloc2" => {
                                 // Derive enum representation from destination or context without hardcoded "Option"
                                 let target_name = match &lattice[dst.0] {
                                     TypeLattice::Concrete(MirType::Enum(n)) if n != "Enum" => Some(n.clone()),
                                     _ => match locals.get(dst.0).map(|l| &l.ty) {
                                         Some(MirType::Enum(n)) if n != "Enum" => Some(n.clone()),
                                         _ => match return_type {
                                             MirType::Enum(n) if n != "Enum" && locals.get(dst.0).and_then(|l| l.name.as_deref()) == Some("_return") => Some(n.clone()),
                                             _ => None,
                                         },
                                     },
                                 };
                                 Some(MirType::Enum(target_name.unwrap_or_else(|| "Enum".to_string())))
                             }
                             _ => locals.iter().find_map(|l| {
                                 if l.name.as_deref() == Some(callee.as_str())
                                     && let MirType::Function { return_type, .. } = &l.ty
                                     && **return_type != MirType::Unknown
                                 {
                                     Some((**return_type).clone())
                                 } else {
                                     None
                                 }
                             }),
                         };
                         if let Some(ty) = call_ret_ty {
                             if lattice[dst.0].merge_evidence(&ty, "call return") {
                                 changed = true;
                             }
                         }
                     }
                 }

                 if let Some(Terminator::Return(
                     Some(Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc))),
                     _,
                 )) = &block.terminator
                 {
                     if loc.0 < lattice.len()
                         && *return_type != MirType::Unknown
                         && (locals[loc.0].name.is_none()
                             || locals[loc.0].ty == MirType::Unknown
                             || locals[loc.0].ty == MirType::Int)
                     {
                         if lattice[loc.0].merge_evidence(return_type, "return terminator constraint") {
                             changed = true;
                         }
                     }
                 }
             }

             if !changed {
                 converged = true;
                 break;
             }
         }

         // Fail closed if inference exceeded bound without reaching fixed point (no silent finalization while changing)
         if !converged {
             for lat in &mut lattice {
                 *lat = TypeLattice::Conflict(
                     "type inference exceeded convergence bound without reaching fixed point".into(),
                 );
             }
         }

         // Finalize inferred types from sound lattice states
         for (i, lat) in lattice.into_iter().enumerate() {
             match lat {
                 TypeLattice::Concrete(ty) => {
                     inferred[i].ty = ty;
                 }
                 TypeLattice::Conflict(_) => {
                     // Conflicting type evidence rejects single Cranelift unboxed type
                     inferred[i].ty = MirType::Unknown;
                 }
                 TypeLattice::Unconstrained => {
                     // Unconstrained: preserves declared type (remains Unknown if originally Unknown).
                     inferred[i].ty = locals[i].ty.clone();
                 }
             }
         }

         inferred
    }

    pub fn refine_mir_function(mir_fn: &MirFunction) -> MirFunction {
        let dummy_layouts = hashbrown::HashMap::new();
        refine_mir_function_with_layouts(mir_fn, &dummy_layouts)
    }

    pub fn refine_mir_function_with_layouts(
        mir_fn: &MirFunction,
        struct_layouts: &hashbrown::HashMap<String, MirStructLayout>,
    ) -> MirFunction {
        refine_mir_function_internal(mir_fn, struct_layouts, None)
    }

    pub(crate) fn refine_mir_function_internal(
        mir_fn: &MirFunction,
        struct_layouts: &hashbrown::HashMap<String, MirStructLayout>,
        max_iterations_override: Option<usize>,
    ) -> MirFunction {
        let mut cloned = mir_fn.clone();
        cloned.locals = refine_locals_internal(
            &cloned.locals,
            &cloned.blocks,
            &cloned.signature.return_type,
            &cloned.params,
            struct_layouts,
            max_iterations_override,
        );
        for block in &mut cloned.blocks {
            for stmt in &mut block.statements {
                if let Statement::Assign(
                    Place::Local(dst),
                    Rvalue::Aggregate(agg_ty, _),
                    _,
                ) = stmt
                {
                    if let Some(decl) = cloned.locals.get(dst.0) {
                        if mir_type_contains_unknown(agg_ty)
                            && !mir_type_contains_unknown(&decl.ty)
                            && unify_mir_types(agg_ty, &decl.ty).as_ref() == Some(&decl.ty)
                        {
                            *agg_ty = decl.ty.clone();
                        }
                    }
                }
            }
        }
        let any_promoted = cloned.locals.iter().any(|l| l.is_stack_promoted);
        if !any_promoted {
            crate::mir::opt::escape_analysis::EscapeAnalysis::run_on_function(&mut cloned);
        }
        cloned
    }

    /// Explicit test seam for testing function convergence boundaries.
    /// Available only under the `test-deopt` feature.
    #[cfg(feature = "test-deopt")]
    pub fn refine_mir_function_with_seam(
        mir_fn: &MirFunction,
        struct_layouts: &hashbrown::HashMap<String, MirStructLayout>,
        max_iterations_override: Option<usize>,
    ) -> MirFunction {
        refine_mir_function_internal(mir_fn, struct_layouts, max_iterations_override)
    }

    fn mir_to_clif_type(ty: &MirType) -> Result<types::Type, MirCraneliftBackendError> {
        match ty {
            MirType::Float => Ok(types::F64),
            MirType::Int => Ok(types::I64),
            MirType::Bool => Ok(types::I8),
            MirType::Unit => Ok(types::I8),
            MirType::String
            | MirType::Array(_)
            | MirType::Tuple(_)
            | MirType::Struct(_)
            | MirType::Map(_, _)
            | MirType::Enum(_)
            | MirType::Effect { .. } => Ok(types::I64),
            MirType::Option(inner)
                if matches!(
                    &**inner,
                    MirType::Int
                        | MirType::Bool
                        | MirType::Float
                        | MirType::String
                        | MirType::Enum(_)
                ) || matches!(&**inner, MirType::Struct(name) if !name.is_empty()) =>
            {
                Ok(types::I64)
            }
            MirType::Option(_) => Err(MirCraneliftBackendError::UnsupportedType(ty.clone())),
            MirType::Unknown => Ok(types::I64),
            MirType::Function { .. } => Ok(types::I64),
        }
    }

    fn mir_type_to_map_operand_tag(ty: &MirType) -> i64 {
        match ty {
            MirType::Int => 0,
            MirType::Bool => 1,
            MirType::Float => 2,
            _ => 3,
        }
    }

    fn clif_val_to_raw_i64(builder: &mut FunctionBuilder, val: ClifValue) -> ClifValue {
        let ty = builder.func.dfg.value_type(val);
        if ty == types::F64 {
            builder.ins().bitcast(types::I64, MemFlagsData::new(), val)
        } else if ty == types::I8 {
            builder.ins().uextend(types::I64, val)
        } else {
            val
        }
    }

    fn cast_clif_type(
        builder: &mut FunctionBuilder,
        val: ClifValue,
        target_ty: types::Type,
    ) -> ClifValue {
        let cur_ty = builder.func.dfg.value_type(val);
        if cur_ty == target_ty {
            return val;
        }
        match (cur_ty, target_ty) {
            (types::I8, types::I64) => builder.ins().uextend(types::I64, val),
            (types::I64, types::I8) => builder.ins().ireduce(types::I8, val),
            (types::I32, types::I64) => builder.ins().sextend(types::I64, val),
            (types::I64, types::I32) => builder.ins().ireduce(types::I32, val),
            (types::I8, types::I32) => builder.ins().uextend(types::I32, val),
            (types::I32, types::I8) => builder.ins().ireduce(types::I8, val),
            (types::I64, types::F64) => builder.ins().fcvt_from_sint(types::F64, val),
            (types::F64, types::I64) => builder.ins().fcvt_to_sint_sat(types::I64, val),
            _ => val,
        }
    }

    fn try_place_type(
        place: &Place,
        locals: &[LocalDecl],
        struct_layouts: &hashbrown::HashMap<String, MirStructLayout>,
    ) -> Result<MirType, MirCraneliftBackendError> {
        match place {
            Place::Local(loc) => locals.get(loc.0).map(|l| l.ty.clone()).ok_or_else(|| {
                MirCraneliftBackendError::CodegenError(format!(
                    "Place local {loc:?} out of bounds (locals len {})",
                    locals.len()
                ))
            }),
            Place::Index(base, idx) => {
                let idx_decl = locals.get(idx.0).ok_or_else(|| {
                    MirCraneliftBackendError::CodegenError(format!(
                        "Index local {idx:?} out of bounds (locals len {})",
                        locals.len()
                    ))
                })?;
                if idx_decl.ty != MirType::Int {
                    return Err(MirCraneliftBackendError::CodegenError(format!(
                        "Index local {idx:?} must have Int type, got '{}'",
                        idx_decl.ty
                    )));
                }
                match try_place_type(base, locals, struct_layouts)? {
                    MirType::Array(inner) => Ok(*inner),
                    MirType::Tuple(elems) if !elems.is_empty() => Ok(elems[0].clone()),
                    other => Err(MirCraneliftBackendError::CodegenError(format!(
                        "Cannot index non-array place of type '{other}'"
                    ))),
                }
            }
            Place::Field(base, idx) => match try_place_type(base, locals, struct_layouts)? {
                MirType::Tuple(elems) => elems.get(*idx).cloned().ok_or_else(|| {
                    MirCraneliftBackendError::CodegenError(format!(
                        "Tuple field index {idx} out of bounds (len {})",
                        elems.len()
                    ))
                }),
                MirType::Struct(name) => {
                    if let Some(layout) = struct_layouts.get(&name) {
                        let f = layout.fields.get(*idx).ok_or_else(|| {
                            MirCraneliftBackendError::CodegenError(format!(
                                "Struct '{name}' field index {idx} out of bounds (len {})",
                                layout.fields.len()
                            ))
                        })?;
                        let resolved = match f.tag {
                            CanonicalTypeTag::Bool => Ok(MirType::Bool),
                            CanonicalTypeTag::Int64
                            | CanonicalTypeTag::UInt64
                            | CanonicalTypeTag::DateTimeUtc => Ok(MirType::Int),
                            CanonicalTypeTag::Float64 => Ok(MirType::Float),
                            CanonicalTypeTag::StringSlice => Ok(MirType::String),
                            other => with_current_mir_context(|ctx| {
                                ctx.get_struct_field_type(&name, *idx)
                            })
                            .ok_or_else(|| {
                                MirCraneliftBackendError::CodegenError(format!(
                                    "Unsupported struct field tag '{other:?}' at '{name}.{idx}'"
                                ))
                            }),
                        }?;
                        if matches!(
                            resolved,
                            MirType::Array(_)
                                | MirType::Tuple(_)
                                | MirType::Struct(_)
                                | MirType::Function { .. }
                                | MirType::Unknown
                        ) || matches!(
                            &resolved,
                            MirType::Option(inner)
                                if !matches!(&**inner, MirType::Struct(n) if !n.is_empty())
                        ) {
                            return Err(MirCraneliftBackendError::UnsupportedType(resolved));
                        }
                        return Ok(resolved);
                    }
                    let resolved = with_current_mir_context(|ctx| ctx.get_struct_field_type(&name, *idx))
                        .ok_or_else(|| {
                            MirCraneliftBackendError::CodegenError(format!(
                                "Unknown struct layout or out-of-bounds field {idx} for struct '{name}'"
                            ))
                        })?;
                    if matches!(
                        resolved,
                        MirType::Array(_)
                            | MirType::Tuple(_)
                            | MirType::Struct(_)
                            | MirType::Function { .. }
                            | MirType::Unknown
                    ) || matches!(
                        &resolved,
                        MirType::Option(inner)
                            if !matches!(&**inner, MirType::Struct(n) if !n.is_empty())
                    ) {
                        return Err(MirCraneliftBackendError::UnsupportedType(resolved));
                    }
                    Ok(resolved)
                }
                other => Err(MirCraneliftBackendError::CodegenError(format!(
                    "Cannot access field {idx} on non-aggregate type '{other}'"
                ))),
            },
        }
    }

    #[derive(Clone, Copy)]
    struct BoundsCheckCtx<'a> {
        status_set_fn: FuncRef,
        emergency_exit_bb: Block,
        hoisted_arrays: &'a hashbrown::HashMap<usize, HoistedArrayVars>,
        hoisted_structs: &'a hashbrown::HashMap<usize, Variable>,
        proven_in_bounds: &'a [(LocalId, LocalId)],
        const_int_locals: &'a hashbrown::HashMap<usize, i64>,
        pow2_mod_defs: &'a hashbrown::HashMap<usize, (ClifValue, i64)>,
        inline_status_ptr_var: Option<Variable>,
        closure_fn_ids: Option<&'a hashbrown::HashMap<String, i64>>,
    }

    fn is_bool_array_index_place(
        place: &Place,
        locals: &[LocalDecl],
        struct_layouts: &hashbrown::HashMap<String, MirStructLayout>,
    ) -> bool {
        matches!(
            place,
            Place::Index(base, _)
                if matches!(
                    try_place_type(base, locals, struct_layouts),
                    Ok(MirType::Array(elem_ty)) if *elem_ty == MirType::Bool
                )
        )
    }

    fn load_local_i64(
        loc: LocalId,
        builder: &mut FunctionBuilder,
        local_vars: &[Variable],
        stack_slots: &hashbrown::HashMap<usize, StackSlot>,
    ) -> Result<ClifValue, MirCraneliftBackendError> {
        if let Some(&slot) = stack_slots.get(&loc.0) {
            Ok(builder.ins().stack_load(types::I64, types::I64, slot, 0))
        } else {
            let var = local_vars.get(loc.0).copied().ok_or_else(|| {
                MirCraneliftBackendError::CodegenError(format!(
                    "Local {loc:?} out of bounds (local_vars len {})",
                    local_vars.len()
                ))
            })?;
            Ok(builder.use_var(var))
        }
    }

    fn store_local_val(
        loc: LocalId,
        val: ClifValue,
        builder: &mut FunctionBuilder,
        local_vars: &[Variable],
        stack_slots: &hashbrown::HashMap<usize, StackSlot>,
    ) -> Result<(), MirCraneliftBackendError> {
        let var = local_vars.get(loc.0).copied().ok_or_else(|| {
            MirCraneliftBackendError::CodegenError(format!(
                "Local {loc:?} out of bounds (local_vars len {})",
                local_vars.len()
            ))
        })?;
        builder.def_var(var, val);
        if let Some(&slot) = stack_slots.get(&loc.0) {
            let val_ty = builder.func.dfg.value_type(val);
            let store_val = if val_ty == types::I8 {
                builder.ins().uextend(types::I64, val)
            } else if val_ty == types::F64 {
                builder.ins().bitcast(types::I64, MemFlagsData::new(), val)
            } else {
                val
            };
            builder.ins().stack_store(types::I64, store_val, slot, 0);
        }
        Ok(())
    }

    fn emit_oob_branch_if_bad(
        is_bad: ClifValue,
        builder: &mut FunctionBuilder,
        bc_ctx: BoundsCheckCtx<'_>,
    ) {
        let err_bb = builder.create_block();
        let ok_bb = builder.create_block();
        builder.set_cold_block(err_bb);
        builder.ins().brif(is_bad, err_bb, &[], ok_bb, &[]);
        builder.switch_to_block(err_bb);
        let code = builder.ins().iconst(types::I64, MIR_STATUS_ERR_OUT_OF_BOUNDS);
        builder.ins().call(bc_ctx.status_set_fn, &[code]);
        builder.ins().jump(bc_ctx.emergency_exit_bb, &[]);
        builder.switch_to_block(ok_bb);
    }

    fn emit_vec_mul_i64x2(
        builder: &mut FunctionBuilder,
        va: ClifValue,
        vb: ClifValue,
    ) -> ClifValue {
        let l0_a = builder.ins().extractlane(va, 0);
        let l1_a = builder.ins().extractlane(va, 1);
        let l0_b = builder.ins().extractlane(vb, 0);
        let l1_b = builder.ins().extractlane(vb, 1);
        let p0 = builder.ins().imul(l0_a, l0_b);
        let p1 = builder.ins().imul(l1_a, l1_b);
        let v0 = builder.ins().splat(types::I64X2, p0);
        builder.ins().insertlane(v0, p1, 1)
    }

    fn lower_place_addr(
        place: &Place,
        builder: &mut FunctionBuilder,
        local_vars: &[Variable],
        stack_slots: &hashbrown::HashMap<usize, StackSlot>,
        locals: &[LocalDecl],
        struct_layouts: &hashbrown::HashMap<String, MirStructLayout>,
        bc_ctx: BoundsCheckCtx<'_>,
    ) -> Result<ClifValue, MirCraneliftBackendError> {
        match place {
            Place::Local(loc) => {
                if let Some(&slot) = stack_slots.get(&loc.0) {
                    Ok(builder.ins().stack_load(types::I64, types::I64, slot, 0))
                } else {
                    let var = local_vars.get(loc.0).copied().ok_or_else(|| {
                        MirCraneliftBackendError::CodegenError(format!(
                            "Place local {loc:?} out of bounds (local_vars len {})",
                            local_vars.len()
                        ))
                    })?;
                    Ok(builder.use_var(var))
                }
            }
            Place::Index(base, idx) => {
                let _ = try_place_type(place, locals, struct_layouts)?;
                let base_ty = try_place_type(base, locals, struct_layouts)?;
                let is_bool_array =
                    matches!(&base_ty, MirType::Array(elem_ty) if **elem_ty == MirType::Bool);

                // PERF-T4-2 + NG-W10-5 fast path: hoisted loop-invariant array (len >= 0, resolved data_ptr)
                if let Place::Local(arr_loc) = base.as_ref()
                    && let Some(&hoisted) = bc_ctx.hoisted_arrays.get(&arr_loc.0)
                {
                    let idx_val = load_local_i64(*idx, builder, local_vars, stack_slots)?;
                    let is_proven_in_bounds = bc_ctx.proven_in_bounds.contains(&(*arr_loc, *idx));
                    if !is_proven_in_bounds {
                        let arr_len = builder.use_var(hoisted.len_var);
                        // guarded_preheader_bb proved arr_len >= 0, so unsigned idx >= arr_len checks both idx < 0 and idx >= len.
                        let is_oob = builder
                            .ins()
                            .icmp(IntCC::UnsignedGreaterThanOrEqual, idx_val, arr_len);
                        emit_oob_branch_if_bad(is_oob, builder, bc_ctx);
                    }
                    let data_ptr = builder.use_var(hoisted.data_ptr_var);
                    let byte_idx = if is_bool_array {
                        idx_val
                    } else {
                        builder.ins().ishl_imm_s(idx_val, 3)
                    };
                    return Ok(builder.ins().iadd(data_ptr, byte_idx));
                }

                let base_ptr = lower_place_addr(
                    base,
                    builder,
                    local_vars,
                    stack_slots,
                    locals,
                    struct_layouts,
                    bc_ctx,
                )?;
                let idx_val = load_local_i64(*idx, builder, local_vars, stack_slots)?;

                let data_off = core::mem::size_of::<MirArrayHeader>() as i64;
                let data_ptr = if matches!(base_ty, MirType::Array(_) | MirType::Tuple(_)) {
                    let is_null = builder.ins().icmp_imm_s(IntCC::Equal, base_ptr, 0);
                    emit_oob_branch_if_bad(is_null, builder, bc_ctx);
                    let is_proven_in_bounds = matches!(
                        base.as_ref(),
                        Place::Local(arr_loc)
                            if bc_ctx.proven_in_bounds.contains(&(*arr_loc, *idx))
                    );
                    if !is_proven_in_bounds {
                        let len_off = core::mem::offset_of!(MirArrayHeader, len) as i32;
                        let arr_len = builder.ins().load(types::I64, MemFlagsData::trusted(), base_ptr, len_off);
                        let idx_neg = builder.ins().icmp_imm_s(IntCC::SignedLessThan, idx_val, 0);
                        let idx_ge = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, idx_val, arr_len);
                        let is_oob = builder.ins().bor(idx_neg, idx_ge);
                        emit_oob_branch_if_bad(is_oob, builder, bc_ctx);
                    }

                    if matches!(base_ty, MirType::Array(_)) {
                        let cap_off = core::mem::offset_of!(MirArrayHeader, cap) as i32;
                        let cap_raw = builder.ins().load(types::I64, MemFlagsData::trusted(), base_ptr, cap_off);
                        let is_redir = builder.ins().icmp_imm_s(IntCC::SignedLessThan, cap_raw, 0);
                        let redir_bb = builder.create_block();
                        let flat_bb = builder.create_block();
                        let data_merge_bb = builder.create_block();
                        let data_param = builder.append_block_param(data_merge_bb, types::I64);
                        builder.set_cold_block(redir_bb);
                        builder.ins().brif(is_redir, redir_bb, &[], flat_bb, &[]);

                        builder.switch_to_block(redir_bb);
                        let redir_ptr = builder.ins().load(
                            types::I64,
                            MemFlagsData::trusted(),
                            base_ptr,
                            data_off as i32,
                        );
                        builder.ins().jump(data_merge_bb, &[BlockArg::Value(redir_ptr)]);

                        builder.switch_to_block(flat_bb);
                        let flat_ptr = builder.ins().iadd_imm_s(base_ptr, data_off);
                        builder.ins().jump(data_merge_bb, &[BlockArg::Value(flat_ptr)]);

                        builder.switch_to_block(data_merge_bb);
                        data_param
                    } else {
                        builder.ins().iadd_imm_s(base_ptr, data_off)
                    }
                } else {
                    builder.ins().iadd_imm_s(base_ptr, data_off)
                };

                let byte_idx = if is_bool_array {
                    idx_val
                } else {
                    builder.ins().ishl_imm_s(idx_val, 3)
                };
                Ok(builder.ins().iadd(data_ptr, byte_idx))
            }
            Place::Field(base, field_idx) => {
                let _ = try_place_type(place, locals, struct_layouts)?;
                let base_ty = try_place_type(base, locals, struct_layouts)?;
                let (base_ptr, already_non_null) = if let Place::Local(st_loc) = base.as_ref()
                    && let Some(&hoisted_base) = bc_ctx.hoisted_structs.get(&st_loc.0)
                {
                    (builder.use_var(hoisted_base), true)
                } else {
                    let bp = lower_place_addr(
                        base,
                        builder,
                        local_vars,
                        stack_slots,
                        locals,
                        struct_layouts,
                        bc_ctx,
                    )?;
                    (bp, false)
                };
                if !already_non_null && matches!(base_ty, MirType::Tuple(_) | MirType::Struct(_)) {
                    let is_null = builder.ins().icmp_imm_s(IntCC::Equal, base_ptr, 0);
                    emit_oob_branch_if_bad(is_null, builder, bc_ctx);
                }
                let offset_bytes = match &base_ty {
                    MirType::Struct(name) => {
                        if let Some(layout) = struct_layouts.get(name) {
                            if let Some(f) = layout.fields.get(*field_idx) {
                                f.byte_offset as i64
                            } else {
                                return Err(MirCraneliftBackendError::CodegenError(format!(
                                    "Struct '{name}' field index {field_idx} out of bounds"
                                )));
                            }
                        } else {
                            16 + (*field_idx as i64) * 8
                        }
                    }
                    MirType::Tuple(_) => 16 + (*field_idx as i64) * 8,
                    other => {
                        return Err(MirCraneliftBackendError::CodegenError(format!(
                            "Cannot access field {field_idx} on non-aggregate type '{other}'"
                        )));
                    }
                };
                let offset = builder.ins().iconst(types::I64, offset_bytes);
                Ok(builder.ins().iadd(base_ptr, offset))
            }
        }
    }

    fn lower_place(
        place: &Place,
        builder: &mut FunctionBuilder,
        local_vars: &[Variable],
        stack_slots: &hashbrown::HashMap<usize, StackSlot>,
        locals: &[LocalDecl],
        struct_layouts: &hashbrown::HashMap<String, MirStructLayout>,
        bc_ctx: BoundsCheckCtx<'_>,
    ) -> Result<ClifValue, MirCraneliftBackendError> {
        match place {
            Place::Local(loc) => {
                let mir_ty = try_place_type(place, locals, struct_layouts)?;
                let clif_ty = mir_to_clif_type(&mir_ty)?;
                if let Some(&slot) = stack_slots.get(&loc.0) {
                    Ok(builder.ins().stack_load(types::I64, clif_ty, slot, 0))
                } else {
                    let var = local_vars.get(loc.0).copied().ok_or_else(|| {
                        MirCraneliftBackendError::CodegenError(format!(
                            "Place local {loc:?} out of bounds"
                        ))
                    })?;
                    Ok(builder.use_var(var))
                }
            }
            Place::Index(_, _) | Place::Field(_, _) => {
                let mir_ty = try_place_type(place, locals, struct_layouts)?;
                let addr = lower_place_addr(
                    place,
                    builder,
                    local_vars,
                    stack_slots,
                    locals,
                    struct_layouts,
                    bc_ctx,
                )?;
                let clif_ty = mir_to_clif_type(&mir_ty)?;
                Ok(builder.ins().load(clif_ty, MemFlagsData::trusted(), addr, 0))
            }
        }
    }

    fn try_operand_type(
        op: &Operand,
        locals: &[LocalDecl],
        struct_layouts: &hashbrown::HashMap<String, MirStructLayout>,
    ) -> Result<MirType, MirCraneliftBackendError> {
        match op {
            Operand::Copy(p) | Operand::Move(p) => try_place_type(p, locals, struct_layouts),
            Operand::Constant(c) => match c {
                MirConstant::Int(_) => Ok(MirType::Int),
                MirConstant::Float(_) => Ok(MirType::Float),
                MirConstant::Bool(_) => Ok(MirType::Bool),
                MirConstant::String(_) => Ok(MirType::String),
                MirConstant::Unit => Ok(MirType::Unit),
            },
        }
    }

    fn lower_operand(
        op: &Operand,
        builder: &mut FunctionBuilder,
        local_vars: &[Variable],
        stack_slots: &hashbrown::HashMap<usize, StackSlot>,
        locals: &[LocalDecl],
        struct_layouts: &hashbrown::HashMap<String, MirStructLayout>,
        bc_ctx: BoundsCheckCtx<'_>,
    ) -> Result<ClifValue, MirCraneliftBackendError> {
        match op {
            Operand::Copy(place) | Operand::Move(place) => {
                lower_place(place, builder, local_vars, stack_slots, locals, struct_layouts, bc_ctx)
            }
            Operand::Constant(c) => match c {
                MirConstant::Int(i) => Ok(builder.ins().iconst(types::I64, *i)),
                MirConstant::Float(f) => Ok(builder.ins().f64const(*f)),
                MirConstant::Bool(b) => Ok(builder.ins().iconst(types::I8, if *b { 1 } else { 0 })),
                MirConstant::String(s) => {
                    let handle = with_current_mir_context(|ctx| ctx.intern_string(s));
                    Ok(builder.ins().iconst(types::I64, handle))
                }
                MirConstant::Unit => Err(MirCraneliftBackendError::UnlowerableValue(
                    "failed or unlowerable value cannot be evaluated as Unit in Cranelift backend".to_string(),
                )),
            },
        }
    }

    fn emit_status_check(
        builder: &mut FunctionBuilder,
        status_fn: FuncRef,
        inline_status_ptr_var: Option<Variable>,
        emergency_exit_bb: Block,
    ) {
        let stat_val = if let Some(gas_ptr_var) = inline_status_ptr_var {
            const _: () = assert!(core::mem::offset_of!(MirContext, gas_fast) == 0);
            const OFF_LAST_STATUS: i32 = core::mem::offset_of!(MirContext, last_status) as i32;
            let gas_ptr = builder.use_var(gas_ptr_var);
            builder
                .ins()
                .load(types::I64, MemFlagsData::trusted(), gas_ptr, OFF_LAST_STATUS)
        } else {
            let stat_call = builder.ins().call(status_fn, &[]);
            builder.inst_results(stat_call)[0]
        };
        let is_err = builder.ins().icmp_imm_s(IntCC::NotEqual, stat_val, 0);
        let cont_bb = builder.create_block();
        builder.ins().brif(is_err, emergency_exit_bb, &[], cont_bb, &[]);
        builder.switch_to_block(cont_bb);
    }

    fn emit_aggregate_word_stores(
        base_ptr: ClifValue,
        elements: &[Operand],
        builder: &mut FunctionBuilder,
        local_vars: &[Variable],
        stack_slots: &hashbrown::HashMap<usize, StackSlot>,
        locals: &[LocalDecl],
        struct_layouts: &hashbrown::HashMap<String, MirStructLayout>,
        bc_ctx: BoundsCheckCtx<'_>,
    ) -> Result<(), MirCraneliftBackendError> {
        for (i, elem) in elements.iter().enumerate() {
            let el_val = lower_operand(elem, builder, local_vars, stack_slots, locals, struct_layouts, bc_ctx)?;
            let el_ty = builder.func.dfg.value_type(el_val);
            let el_to_store = if el_ty == types::I8 {
                builder.ins().uextend(types::I64, el_val)
            } else {
                el_val
            };
            builder.ins().store(MemFlagsData::trusted(), el_to_store, base_ptr, 16 + (i as i32) * 8);
        }
        Ok(())
    }

    fn intern_static_type_name(name: &str) -> (*const u8, usize) {
        use std::sync::Mutex;
        static NAME_CACHE: Mutex<Option<hashbrown::HashSet<&'static str>>> = Mutex::new(None);
        let mut guard = NAME_CACHE.lock().unwrap();
        let set = guard.get_or_insert_with(hashbrown::HashSet::new);
        if let Some(&existing) = set.get(name) {
            return (existing.as_ptr(), existing.len());
        }
        let leaked: &'static str = alloc::boxed::Box::leak(name.to_string().into_boxed_str());
        set.insert(leaked);
        (leaked.as_ptr(), leaked.len())
    }

    fn emit_inline_int_div_or_rem(
        is_div: bool,
        l: ClifValue,
        r: ClifValue,
        rhs_operand: &Operand,
        builder: &mut FunctionBuilder,
        bc_ctx: BoundsCheckCtx<'_>,
        pow2_mod_out: Option<&mut Option<(ClifValue, i64)>>,
    ) -> ClifValue {
        let emergency_exit_bb = bc_ctx.emergency_exit_bb;
        if let Some(c) = resolve_const_int_operand(rhs_operand, bc_ctx.const_int_locals) {
            if c == 0 {
                let code = builder.ins().iconst(types::I64, MIR_STATUS_ERR_DIV_BY_ZERO);
                builder.ins().call(bc_ctx.status_set_fn, &[code]);
                builder.ins().jump(emergency_exit_bb, &[]);
                let dead_bb = builder.create_block();
                builder.switch_to_block(dead_bb);
                builder.seal_block(dead_bb);
                return builder.ins().iconst(types::I64, 0);
            }
            if c == 1 {
                return if is_div {
                    l
                } else {
                    builder.ins().iconst(types::I64, 0)
                };
            }
            if c == -1 {
                return if is_div {
                    builder.ins().ineg(l)
                } else {
                    builder.ins().iconst(types::I64, 0)
                };
            }
            if c == i64::MIN {
                // Truncated-toward-zero semantics for divisor i64::MIN (-2^63):
                //   l / i64::MIN == 1 iff l == i64::MIN else 0
                //   l % i64::MIN == 0 iff l == i64::MIN else l
                //   l % i64::MIN == 0 iff (l & i64::MAX) == 0
                let is_min = builder.ins().icmp_imm_s(IntCC::Equal, l, i64::MIN);
                if is_div {
                    return builder.ins().uextend(types::I64, is_min);
                } else {
                    if let Some(slot) = pow2_mod_out {
                        *slot = Some((l, i64::MAX));
                    }
                    let zero = builder.ins().iconst(types::I64, 0);
                    return builder.ins().select(is_min, zero, l);
                }
            }

            let abs_c = c.unsigned_abs();
            if abs_c.is_power_of_two() {
                // NG-W14-5: Exact branchless signed power-of-two truncated-toward-zero
                // division / remainder for c = ±2^k (1 <= k <= 62).
                let k = abs_c.trailing_zeros() as i64;
                let bias = if k == 1 {
                    // For ±2: bias = 1 if l < 0 else 0 (single logical right shift).
                    builder.ins().ushr_imm_s(l, 63)
                } else {
                    // For ±2^k (2 <= k <= 62): bias = (2^k - 1) if l < 0 else 0.
                    let sign = builder.ins().sshr_imm_s(l, k - 1);
                    builder.ins().ushr_imm_s(sign, 64 - k)
                };
                let biased = builder.ins().iadd(l, bias);
                if is_div {
                    let q = builder.ins().sshr_imm_s(biased, k);
                    return if c < 0 { builder.ins().ineg(q) } else { q };
                } else {
                    if let Some(slot) = pow2_mod_out {
                        *slot = Some((l, (abs_c - 1) as i64));
                    }
                    // Truncated quotient * 2^k = biased & -(2^k); remainder = l - (q * 2^k).
                    // Identical for +2^k and -2^k because l % (-d) == l % d.
                    let mask = builder.ins().iconst(types::I64, abs_c.wrapping_neg() as i64);
                    let trunc_mul = builder.ins().band(biased, mask);
                    return builder.ins().isub(l, trunc_mul);
                }
            }

            // General non-zero, non-power-of-two constant `c` (e.g. 3, -3, 7, -7, 1_000_000_007):
            // Both `rhs == 0` and `(lhs == i64::MIN && rhs == -1)` runtime trap guards are
            // completely elided. Materialize `iconst c` directly so Cranelift's egraph
            // (`simplify_skeleton` -> `apply_div_const_magic_s64`) unconditionally lowers
            // `sdiv`/`srem` to Hacker's Delight magic-multiplier `smulhi` + shifts even
            // when `rhs_operand` originated from a cross-block constant local.
            let c_val = builder.ins().iconst(types::I64, c);
            if !is_div && c >= 1_000_000 {
                // Fast path for large positive moduli (e.g. `% 1_000_000_007`):
                // `(l as u64) < (c as u64)` holds iff `0 <= l < c`, in which case `l % c == l`
                // with zero `smulhi` latency; out-of-range inputs fall through to exact `srem`.
                let in_range = builder.ins().icmp(IntCC::UnsignedLessThan, l, c_val);
                let slow_rem_bb = builder.create_block();
                let done_bb = builder.create_block();
                builder.set_cold_block(slow_rem_bb);
                let rem_param = builder.append_block_param(done_bb, types::I64);
                builder.ins().brif(
                    in_range,
                    done_bb,
                    &[BlockArg::Value(l)],
                    slow_rem_bb,
                    &[],
                );
                builder.switch_to_block(slow_rem_bb);
                let rem = builder.ins().srem(l, c_val);
                builder.ins().jump(done_bb, &[BlockArg::Value(rem)]);
                builder.switch_to_block(done_bb);
                return rem_param;
            }
            return if is_div {
                builder.ins().sdiv(l, c_val)
            } else {
                builder.ins().srem(l, c_val)
            };
        }

        let div_zero_bb = builder.create_block();
        let non_zero_bb = builder.create_block();
        let neg_one_bb = builder.create_block();
        let safe_div_bb = builder.create_block();
        let merge_bb = builder.create_block();
        let res_param = builder.append_block_param(merge_bb, types::I64);
        builder.set_cold_block(div_zero_bb);
        builder.set_cold_block(neg_one_bb);

        let is_zero = builder.ins().icmp_imm_s(IntCC::Equal, r, 0);
        builder.ins().brif(is_zero, div_zero_bb, &[], non_zero_bb, &[]);

        builder.switch_to_block(div_zero_bb);
        builder.seal_block(div_zero_bb);
        let code = builder.ins().iconst(types::I64, MIR_STATUS_ERR_DIV_BY_ZERO);
        builder.ins().call(bc_ctx.status_set_fn, &[code]);
        builder.ins().jump(emergency_exit_bb, &[]);

        builder.switch_to_block(non_zero_bb);
        builder.seal_block(non_zero_bb);
        let is_neg_one = builder.ins().icmp_imm_s(IntCC::Equal, r, -1);
        builder.ins().brif(is_neg_one, neg_one_bb, &[], safe_div_bb, &[]);

        builder.switch_to_block(neg_one_bb);
        builder.seal_block(neg_one_bb);
        let neg_one_res = if is_div {
            builder.ins().ineg(l)
        } else {
            builder.ins().iconst(types::I64, 0)
        };
        builder.ins().jump(merge_bb, &[BlockArg::Value(neg_one_res)]);

        builder.switch_to_block(safe_div_bb);
        builder.seal_block(safe_div_bb);
        let div_res = if is_div {
            builder.ins().sdiv(l, r)
        } else {
            builder.ins().srem(l, r)
        };
        builder.ins().jump(merge_bb, &[BlockArg::Value(div_res)]);

        builder.switch_to_block(merge_bb);
        builder.seal_block(merge_bb);
        res_param
    }

    struct ArrayLenProvenancePlan {
        validated_in: Vec<hashbrown::HashSet<usize>>,
        hoist_at_header_entry: hashbrown::HashMap<usize, Vec<LocalId>>,
        loop_bodies: hashbrown::HashMap<usize, hashbrown::HashSet<usize>>,
    }

    fn compute_array_len_provenance_plan(
        mir_fn: &MirFunction,
        loop_headers: &hashbrown::HashSet<usize>,
        osr_info: Option<(BlockId, &[LocalId])>,
    ) -> ArrayLenProvenancePlan {
        let num_blocks = mir_fn.blocks.len();
        if num_blocks == 0 {
            return ArrayLenProvenancePlan {
                validated_in: Vec::new(),
                hoist_at_header_entry: hashbrown::HashMap::new(),
                loop_bodies: hashbrown::HashMap::new(),
            };
        }

        let entry_idx = osr_info
            .map(|(tb, _)| tb.0)
            .unwrap_or(mir_fn.entry.0)
            .min(num_blocks - 1);

        // 1. Reachable blocks & predecessors from entry_idx
        let mut reachable = hashbrown::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        reachable.insert(entry_idx);
        queue.push_back(entry_idx);
        while let Some(b) = queue.pop_front() {
            if let Some(term) = &mir_fn.blocks[b].terminator {
                for succ in term.successors() {
                    if succ < num_blocks && reachable.insert(succ) {
                        queue.push_back(succ);
                    }
                }
            }
        }

        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];
        for &b in &reachable {
            if let Some(term) = &mir_fn.blocks[b].terminator {
                for succ in term.successors() {
                    if succ < num_blocks && reachable.contains(&succ) && !preds[succ].contains(&b) {
                        preds[succ].push(b);
                    }
                }
            }
        }

        // 2. Natural loop bodies per reachable loop header
        let backedges: Vec<(usize, usize)> =
            crate::mir::opt::Safepoints::find_loop_backedges_in_fn(mir_fn)
                .into_iter()
                .map(|(f, t)| (f.0, t.0))
                .collect();
        let mut loop_bodies: hashbrown::HashMap<usize, hashbrown::HashSet<usize>> =
            hashbrown::HashMap::new();
        for &(tail, hdr) in &backedges {
            if !loop_headers.contains(&hdr) || !reachable.contains(&hdr) || !reachable.contains(&tail)
            {
                continue;
            }
            let body = loop_bodies.entry(hdr).or_default();
            body.insert(hdr);
            if body.insert(tail) {
                let mut stack = vec![tail];
                while let Some(cur) = stack.pop() {
                    for &p in &preds[cur] {
                        if body.insert(p) {
                            stack.push(p);
                        }
                    }
                }
            }
        }

        // 3. Per-block assigned locals and definite-initialization must-analysis
        let mut assigned_in_block: Vec<hashbrown::HashSet<usize>> =
            vec![hashbrown::HashSet::new(); num_blocks];
        for (b_idx, block) in mir_fn.blocks.iter().enumerate() {
            for stmt in &block.statements {
                if let Statement::Assign(Place::Local(dst), _, _) = stmt {
                    assigned_in_block[b_idx].insert(dst.0);
                }
            }
            if let Some(Terminator::Call {
                destination: Place::Local(dst),
                ..
            }) = &block.terminator
            {
                assigned_in_block[b_idx].insert(dst.0);
            }
        }

        let init_at_entry: hashbrown::HashSet<usize> = if let Some((_, live)) = osr_info {
            live.iter().map(|l| l.0).collect()
        } else {
            mir_fn.params.iter().map(|l| l.0).collect()
        };
        let all_locals: hashbrown::HashSet<usize> = (0..mir_fn.locals.len()).collect();
        let mut def_in: Vec<hashbrown::HashSet<usize>> = (0..num_blocks)
            .map(|b| {
                if b == entry_idx {
                    init_at_entry.clone()
                } else {
                    all_locals.clone()
                }
            })
            .collect();
        let mut def_out: Vec<hashbrown::HashSet<usize>> = (0..num_blocks)
            .map(|b| {
                if b == entry_idx {
                    let mut s = init_at_entry.clone();
                    s.extend(assigned_in_block[b].iter().copied());
                    s
                } else {
                    all_locals.clone()
                }
            })
            .collect();

        loop {
            let mut changed = false;
            for b in 0..num_blocks {
                if !reachable.contains(&b) {
                    continue;
                }
                let new_in = if b == entry_idx {
                    let mut s = init_at_entry.clone();
                    for &p in &preds[b] {
                        s.retain(|x| def_out[p].contains(x));
                    }
                    s
                } else if preds[b].is_empty() {
                    hashbrown::HashSet::new()
                } else {
                    let mut iter = preds[b].iter();
                    let first = *iter.next().unwrap();
                    let mut s = def_out[first].clone();
                    for &p in iter {
                        s.retain(|x| def_out[p].contains(x));
                    }
                    s
                };
                let mut new_out = new_in.clone();
                new_out.extend(assigned_in_block[b].iter().copied());
                if new_in != def_in[b] || new_out != def_out[b] {
                    def_in[b] = new_in;
                    def_out[b] = new_out;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        // 4. Iterative must-validated provenance dataflow
        let run_dataflow =
            |hoist_map: &hashbrown::HashMap<usize, Vec<LocalId>>| -> Vec<hashbrown::HashSet<usize>> {
                let mut val_in: Vec<hashbrown::HashSet<usize>> = (0..num_blocks)
                    .map(|b| {
                        if b == entry_idx {
                            let mut s = hashbrown::HashSet::new();
                            if let Some(hoisted) = hoist_map.get(&b) {
                                for loc in hoisted {
                                    s.insert(loc.0);
                                }
                            }
                            s
                        } else {
                            all_locals.clone()
                        }
                    })
                    .collect();
                let mut val_out: Vec<hashbrown::HashSet<usize>> = vec![all_locals.clone(); num_blocks];

                let transfer = |b: usize, in_set: &hashbrown::HashSet<usize>| -> hashbrown::HashSet<usize> {
                    let mut cur = in_set.clone();
                    let block = &mir_fn.blocks[b];
                    for stmt in &block.statements {
                        if let Statement::Assign(Place::Local(dst), rval, _) = stmt {
                            if let Rvalue::UnaryOp(
                                MirUnaryOp::Len,
                                Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src)),
                            ) = rval
                            {
                                if matches!(
                                    mir_fn.locals.get(src.0).map(|l| &l.ty),
                                    Some(MirType::Array(_))
                                ) {
                                    cur.insert(src.0);
                                }
                            }
                            match rval {
                                Rvalue::Aggregate(MirType::Array(_), _) => {
                                    cur.insert(dst.0);
                                }
                                Rvalue::Use(
                                    Operand::Copy(Place::Local(src))
                                    | Operand::Move(Place::Local(src)),
                                ) if cur.contains(&src.0) => {
                                    cur.insert(dst.0);
                                }
                                _ => {
                                    cur.remove(&dst.0);
                                }
                            }
                        }
                    }
                    if let Some(Terminator::Call {
                        callee,
                        args,
                        destination,
                        ..
                    }) = &block.terminator
                    {
                        if callee == "array.len" || callee == "table.len" {
                            if let Some(
                                Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src)),
                            ) = args.first()
                            {
                                if matches!(
                                    mir_fn.locals.get(src.0).map(|l| &l.ty),
                                    Some(MirType::Array(_))
                                ) {
                                    cur.insert(src.0);
                                }
                            }
                        }
                        if let Place::Local(dst) = destination {
                            if (callee == "string.split" || callee == "string.gsub")
                                && matches!(
                                    mir_fn.locals.get(dst.0).map(|l| &l.ty),
                                    Some(MirType::Array(_))
                                )
                            {
                                cur.insert(dst.0);
                            } else {
                                cur.remove(&dst.0);
                            }
                        }
                    }
                    cur
                };

                val_out[entry_idx] = transfer(entry_idx, &val_in[entry_idx]);

                loop {
                    let mut changed = false;
                    for b in 0..num_blocks {
                        if !reachable.contains(&b) {
                            continue;
                        }
                        let new_in = if b == entry_idx {
                            let mut s = hashbrown::HashSet::new();
                            if let Some(hoisted) = hoist_map.get(&b) {
                                for loc in hoisted {
                                    s.insert(loc.0);
                                }
                            }
                            for &p in &preds[b] {
                                s.retain(|x| val_out[p].contains(x));
                            }
                            s
                        } else if preds[b].is_empty() {
                            hashbrown::HashSet::new()
                        } else {
                            let mut first_set: Option<hashbrown::HashSet<usize>> = None;
                            for &p in &preds[b] {
                                let mut edge_out = val_out[p].clone();
                                if let Some(hoisted) = hoist_map.get(&b) {
                                    let is_backedge_or_internal = loop_bodies
                                        .get(&b)
                                        .is_some_and(|body| body.contains(&p));
                                    if !is_backedge_or_internal {
                                        for loc in hoisted {
                                            edge_out.insert(loc.0);
                                        }
                                    }
                                }
                                match &mut first_set {
                                    None => first_set = Some(edge_out),
                                    Some(acc) => acc.retain(|x| edge_out.contains(x)),
                                }
                            }
                            first_set.unwrap_or_default()
                        };
                        let new_out = transfer(b, &new_in);
                        if new_in != val_in[b] || new_out != val_out[b] {
                            val_in[b] = new_in;
                            val_out[b] = new_out;
                            changed = true;
                        }
                    }
                    if !changed {
                        break;
                    }
                }

                for b in 0..num_blocks {
                    if !reachable.contains(&b) {
                        val_in[b].clear();
                    }
                }
                val_in
            };

        let mut hoist_at_header_entry: hashbrown::HashMap<usize, Vec<LocalId>> =
            hashbrown::HashMap::new();
        let mut validated_in = run_dataflow(&hoist_at_header_entry);

        // 5. Hoist loop-invariant array provenance checks to outermost loop preheaders
        let mut ordered_hdrs: Vec<usize> = loop_bodies.keys().copied().collect();
        ordered_hdrs.sort_by(|a, b| {
            loop_bodies[b]
                .len()
                .cmp(&loop_bodies[a].len())
                .then_with(|| a.cmp(b))
        });

        for hdr in ordered_hdrs {
            let body = &loop_bodies[&hdr];
            let mut len_used_locals: Vec<usize> = Vec::new();
            for &b in body {
                let block = &mir_fn.blocks[b];
                for stmt in &block.statements {
                    if let Statement::Assign(
                        _,
                        Rvalue::UnaryOp(
                            MirUnaryOp::Len,
                            Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src)),
                        ),
                        _,
                    ) = stmt
                    {
                        if !len_used_locals.contains(&src.0) {
                            len_used_locals.push(src.0);
                        }
                    }
                }
                if let Some(Terminator::Call { callee, args, .. }) = &block.terminator {
                    if callee == "array.len" || callee == "table.len" {
                        if let Some(
                            Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src)),
                        ) = args.first()
                        {
                            if !len_used_locals.contains(&src.0) {
                                len_used_locals.push(src.0);
                            }
                        }
                    }
                }
            }
            len_used_locals.sort_unstable();

            let mut to_hoist: Vec<LocalId> = Vec::new();
            for loc_idx in len_used_locals {
                let is_array = matches!(
                    mir_fn.locals.get(loc_idx).map(|l| &l.ty),
                    Some(MirType::Array(_))
                );
                if !is_array {
                    continue;
                }
                if !def_in[hdr].contains(&loc_idx) {
                    continue;
                }
                let reassigned_in_loop = body
                    .iter()
                    .any(|&b| assigned_in_block[b].contains(&loc_idx));
                if reassigned_in_loop {
                    continue;
                }
                if validated_in[hdr].contains(&loc_idx) {
                    continue;
                }
                to_hoist.push(LocalId(loc_idx));
            }

            if !to_hoist.is_empty() {
                hoist_at_header_entry.insert(hdr, to_hoist);
                validated_in = run_dataflow(&hoist_at_header_entry);
            }
        }

        ArrayLenProvenancePlan {
            validated_in,
            hoist_at_header_entry,
            loop_bodies,
        }
    }

    fn emit_array_len(
        op: &Operand,
        arr_val: ClifValue,
        builder: &mut FunctionBuilder,
        len_fn: FuncRef,
        emergency_exit_bb: Block,
        validated_arrays: &mut hashbrown::HashSet<usize>,
    ) -> ClifValue {
        let local_idx = match op {
            Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => Some(loc.0),
            _ => None,
        };
        if let Some(idx) = local_idx {
            if validated_arrays.contains(&idx) {
                let len_off = core::mem::offset_of!(MirArrayHeader, len) as i32;
                return builder
                    .ins()
                    .load(types::I64, MemFlagsData::trusted(), arr_val, len_off);
            }
        }
        let call = builder.ins().call(len_fn, &[arr_val]);
        let res = builder.inst_results(call)[0];
        let is_err = builder.ins().icmp_imm_s(IntCC::SignedLessThan, res, 0);
        let len_ok_bb = builder.create_block();
        builder.ins().brif(is_err, emergency_exit_bb, &[], len_ok_bb, &[]);
        builder.switch_to_block(len_ok_bb);
        if let Some(idx) = local_idx {
            validated_arrays.insert(idx);
        }
        res
    }

    #[derive(Clone, Copy)]
    struct StringHelperFns {
        concat_fn: FuncRef,
        append_inplace_fn: FuncRef,
        eq_fn: FuncRef,
        len_fn: FuncRef,
        tostring_int_fn: FuncRef,
        tostring_float_fn: FuncRef,
        tostring_bool_fn: FuncRef,
    }

    fn operand_mentions_local(op: &Operand, target: LocalId) -> bool {
        let mut uses = hashbrown::HashSet::new();
        op.collect_uses(&mut uses);
        uses.contains(&target.0)
    }

    /// Computes the set of `MirType::String` locals in `mir_fn` that are proven locally
    /// unaliased across the entire function, making in-place COW append (`dst = dst .. rhs`)
    /// sound against MIR `i64`-handle aliasing.
    fn compute_unaliased_string_locals(mir_fn: &MirFunction) -> hashbrown::HashSet<usize> {
        let n_locals = mir_fn.locals.len();
        let mut aliased = hashbrown::HashSet::<usize>::new();

        // Parameters may alias caller locals or each other.
        for &p in &mir_fn.params {
            aliased.insert(p.0);
        }

        // Track definition/use counts to allow a single-def, single-read, same-block fresh
        // Concat temporary to transfer unique ownership (`tmp = a .. b; s = move tmp`)
        // without marking `s` as aliased.
        let mut def_counts = vec![0usize; n_locals];
        let mut read_counts = vec![0usize; n_locals];
        let mut single_concat_def: Vec<Option<(usize, usize)>> = vec![None; n_locals];
        let mut used_in_term_or_proj = vec![false; n_locals];

        for (b_idx, block) in mir_fn.blocks.iter().enumerate() {
            for (s_idx, stmt) in block.statements.iter().enumerate() {
                if let Statement::Assign(place, rval, _) = stmt {
                    match place {
                        Place::Local(dst) => {
                            if dst.0 < n_locals {
                                def_counts[dst.0] += 1;
                                if matches!(rval, Rvalue::BinaryOp(MirBinaryOp::Concat, _, _)) {
                                    single_concat_def[dst.0] = Some((b_idx, s_idx));
                                }
                            }
                        }
                        Place::Field(..) | Place::Index(..) => {
                            let mut p_uses = hashbrown::HashSet::new();
                            p_uses.insert(place.root_local().0);
                            place.collect_uses(&mut p_uses);
                            for u in p_uses {
                                if u < n_locals {
                                    read_counts[u] += 1;
                                    used_in_term_or_proj[u] = true;
                                }
                            }
                        }
                    }
                    let mut r_uses = hashbrown::HashSet::new();
                    rval.collect_uses(&mut r_uses);
                    for u in r_uses {
                        if u < n_locals {
                            read_counts[u] += 1;
                        }
                    }
                }
            }
            if let Some(term) = &block.terminator {
                match term {
                    Terminator::Call {
                        args, destination, ..
                    } => {
                        match destination {
                            Place::Local(dst) => {
                                if dst.0 < n_locals {
                                    def_counts[dst.0] += 1;
                                    used_in_term_or_proj[dst.0] = true;
                                }
                            }
                            Place::Field(..) | Place::Index(..) => {
                                let mut p_uses = hashbrown::HashSet::new();
                                p_uses.insert(destination.root_local().0);
                                destination.collect_uses(&mut p_uses);
                                for u in p_uses {
                                    if u < n_locals {
                                        read_counts[u] += 1;
                                        used_in_term_or_proj[u] = true;
                                    }
                                }
                            }
                        }
                        for arg in args {
                            let mut a_uses = hashbrown::HashSet::new();
                            arg.collect_uses(&mut a_uses);
                            for u in a_uses {
                                if u < n_locals {
                                    read_counts[u] += 1;
                                    used_in_term_or_proj[u] = true;
                                }
                            }
                        }
                    }
                    Terminator::Cond { cond, .. } => {
                        let mut uses = hashbrown::HashSet::new();
                        cond.collect_uses(&mut uses);
                        for u in uses {
                            if u < n_locals {
                                read_counts[u] += 1;
                                used_in_term_or_proj[u] = true;
                            }
                        }
                    }
                    Terminator::Switch { discr, .. } => {
                        let mut uses = hashbrown::HashSet::new();
                        discr.collect_uses(&mut uses);
                        for u in uses {
                            if u < n_locals {
                                read_counts[u] += 1;
                                used_in_term_or_proj[u] = true;
                            }
                        }
                    }
                    Terminator::Return(Some(op), _) | Terminator::Raise(op, _) => {
                        let mut uses = hashbrown::HashSet::new();
                        op.collect_uses(&mut uses);
                        for u in uses {
                            if u < n_locals {
                                read_counts[u] += 1;
                                used_in_term_or_proj[u] = true;
                            }
                        }
                    }
                    Terminator::Return(None, _)
                    | Terminator::Goto(..)
                    | Terminator::Unreachable(_) => {}
                }
            }
        }

        let mark_rval_uses = |rval: &Rvalue, aliased: &mut hashbrown::HashSet<usize>| {
            let mut uses = hashbrown::HashSet::new();
            rval.collect_uses(&mut uses);
            aliased.extend(uses);
        };
        let mark_op_uses = |op: &Operand, aliased: &mut hashbrown::HashSet<usize>| {
            let mut uses = hashbrown::HashSet::new();
            op.collect_uses(&mut uses);
            aliased.extend(uses);
        };

        for (b_idx, block) in mir_fn.blocks.iter().enumerate() {
            for (s_idx, stmt) in block.statements.iter().enumerate() {
                if let Statement::Assign(place, rval, _) = stmt {
                    match place {
                        Place::Local(dst) => match rval {
                            Rvalue::Use(
                                Operand::Copy(Place::Local(src))
                                | Operand::Move(Place::Local(src)),
                            ) => {
                                if dst != src {
                                    let unique_same_block_concat_transfer = src.0 < n_locals
                                        && !aliased.contains(&src.0)
                                        && def_counts[src.0] == 1
                                        && read_counts[src.0] == 1
                                        && !used_in_term_or_proj[src.0]
                                        && matches!(
                                            single_concat_def[src.0],
                                            Some((def_b, def_s)) if def_b == b_idx && def_s < s_idx
                                        );
                                    if !unique_same_block_concat_transfer {
                                        aliased.insert(src.0);
                                        aliased.insert(dst.0);
                                    }
                                }
                            }
                            Rvalue::Use(
                                Operand::Copy(Place::Field(..) | Place::Index(..))
                                | Operand::Move(Place::Field(..) | Place::Index(..)),
                            ) => {
                                aliased.insert(dst.0);
                            }
                            Rvalue::Use(Operand::Constant(_)) => {}
                            Rvalue::BinaryOp(
                                MirBinaryOp::Concat
                                | MirBinaryOp::Add
                                | MirBinaryOp::Eq
                                | MirBinaryOp::NotEq,
                                _,
                                _,
                            ) => {}
                            Rvalue::UnaryOp(MirUnaryOp::Len, _) => {}
                            Rvalue::Cast(op, _) => {
                                aliased.insert(dst.0);
                                mark_op_uses(op, &mut aliased);
                            }
                            Rvalue::Aggregate(_, elems) => {
                                aliased.insert(dst.0);
                                for e in elems {
                                    mark_op_uses(e, &mut aliased);
                                }
                            }
                            Rvalue::Closure { captures, .. } => {
                                aliased.insert(dst.0);
                                for (_, op, _) in captures {
                                    mark_op_uses(op, &mut aliased);
                                }
                            }
                            Rvalue::BinaryOp(..) | Rvalue::UnaryOp(..) => {}
                        },
                        Place::Field(..) | Place::Index(..) => {
                            mark_rval_uses(rval, &mut aliased);
                        }
                    }
                }
            }
            if let Some(term) = &block.terminator {
                match term {
                    Terminator::Call {
                        callee,
                        args,
                        destination,
                        ..
                    } => {
                        if matches!(
                            callee.as_str(),
                            "string.len"
                                | "string.sub"
                                | "string.byte"
                                | "string.byte_at"
                                | "string.find"
                                | "string.gsub"
                                | "string.split"
                                | "print"
                                | "println"
                        ) {
                            // Pure non-escaping string readers.
                        } else if callee == "tostring" {
                            // `tostring(string)` returns the same handle verbatim; mark any
                            // argument local aliased, and mark destination aliased when input
                            // is a string (or unknown).
                            let mut input_may_be_string = false;
                            for arg in args {
                                mark_op_uses(arg, &mut aliased);
                                if let Operand::Copy(Place::Local(loc))
                                | Operand::Move(Place::Local(loc)) = arg
                                {
                                    if matches!(
                                        mir_fn.locals.get(loc.0).map(|d| &d.ty),
                                        Some(MirType::String | MirType::Unknown) | None
                                    ) {
                                        input_may_be_string = true;
                                    }
                                } else if !matches!(
                                    arg,
                                    Operand::Constant(
                                        MirConstant::Int(_)
                                            | MirConstant::Float(_)
                                            | MirConstant::Bool(_)
                                    )
                                ) {
                                    input_may_be_string = true;
                                }
                            }
                            if input_may_be_string
                                && let Place::Local(dst) = destination
                            {
                                aliased.insert(dst.0);
                            }
                        } else {
                            for arg in args {
                                mark_op_uses(arg, &mut aliased);
                            }
                            if let Place::Local(dst) = destination {
                                aliased.insert(dst.0);
                            }
                        }
                    }
                    Terminator::Raise(op, _) => {
                        mark_op_uses(op, &mut aliased);
                    }
                    Terminator::Goto(..)
                    | Terminator::Cond { .. }
                    | Terminator::Switch { .. }
                    | Terminator::Return(..)
                    | Terminator::Unreachable(_) => {}
                }
            }
        }

        let mut unaliased = hashbrown::HashSet::new();
        for (idx, decl) in mir_fn.locals.iter().enumerate() {
            if decl.ty == MirType::String && !aliased.contains(&idx) {
                unaliased.insert(idx);
            }
        }
        unaliased
    }

    fn lower_to_string_handle(
        mir_ty: &MirType,
        val: ClifValue,
        builder: &mut FunctionBuilder,
        string_helpers: StringHelperFns,
        status_fn: FuncRef,
        inline_status_ptr_var: Option<Variable>,
        emergency_exit_bb: Block,
    ) -> Result<ClifValue, MirCraneliftBackendError> {
        match mir_ty {
            MirType::String => Ok(cast_clif_type(builder, val, types::I64)),
            MirType::Int => {
                let v = cast_clif_type(builder, val, types::I64);
                let call = builder.ins().call(string_helpers.tostring_int_fn, &[v]);
                let res = builder.inst_results(call)[0];
                emit_status_check(builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                Ok(res)
            }
            MirType::Float => {
                let v = cast_clif_type(builder, val, types::F64);
                let call = builder.ins().call(string_helpers.tostring_float_fn, &[v]);
                let res = builder.inst_results(call)[0];
                emit_status_check(builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                Ok(res)
            }
            MirType::Bool => {
                let v_ty = builder.func.dfg.value_type(val);
                let v = if v_ty == types::I8 {
                    builder.ins().uextend(types::I64, val)
                } else {
                    cast_clif_type(builder, val, types::I64)
                };
                let call = builder.ins().call(string_helpers.tostring_bool_fn, &[v]);
                let res = builder.inst_results(call)[0];
                emit_status_check(builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                Ok(res)
            }
            other => Err(MirCraneliftBackendError::CodegenError(format!(
                "Unsupported type '{other}' for string conversion in Cranelift backend"
            ))),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_rvalue(
        rval: &Rvalue,
        expected_dst_ty: Option<&MirType>,
        inplace_concat_dst: Option<LocalId>,
        validated_arrays: &mut hashbrown::HashSet<usize>,
        builder: &mut FunctionBuilder,
        local_vars: &[Variable],
        stack_slots: &hashbrown::HashMap<usize, StackSlot>,
        locals: &[LocalDecl],
        struct_layouts: &hashbrown::HashMap<String, MirStructLayout>,
        alloc_array_fn: FuncRef,
        alloc_array_typed_fn: FuncRef,
        alloc_tuple_typed_fn: FuncRef,
        alloc_struct_named_fn: FuncRef,
        alloc_cflat_struct_fn: FuncRef,
        len_fn: FuncRef,
        floordiv_fn: FuncRef,
        pow_fn: FuncRef,
        floordiv_float_fn: FuncRef,
        pow_float_fn: FuncRef,
        rem_float_fn: FuncRef,
        string_helpers: StringHelperFns,
        status_fn: FuncRef,
        bc_ctx: BoundsCheckCtx<'_>,
        pow2_mod_out: Option<&mut Option<(ClifValue, i64)>>,
    ) -> Result<ClifValue, MirCraneliftBackendError> {
        let emergency_exit_bb = bc_ctx.emergency_exit_bb;
        let inline_status_ptr_var = bc_ctx.inline_status_ptr_var;
        match rval {
            Rvalue::Use(op) => lower_operand(op, builder, local_vars, stack_slots, locals, struct_layouts, bc_ctx),
            Rvalue::Cast(op, target_ty) => {
                let val = lower_operand(op, builder, local_vars, stack_slots, locals, struct_layouts, bc_ctx)?;
                let target_clif_ty = mir_to_clif_type(target_ty)?;
                Ok(cast_clif_type(builder, val, target_clif_ty))
            }
            Rvalue::BinaryOp(binop, lhs, rhs) => {
                let l_ty_mir = try_operand_type(lhs, locals, struct_layouts)?;
                let r_ty_mir = try_operand_type(rhs, locals, struct_layouts)?;

                if matches!(binop, MirBinaryOp::Concat) {
                    let l_raw = lower_operand(lhs, builder, local_vars, stack_slots, locals, struct_layouts, bc_ctx)?;
                    let r_raw = lower_operand(rhs, builder, local_vars, stack_slots, locals, struct_layouts, bc_ctx)?;
                    let l_str = lower_to_string_handle(
                        &l_ty_mir,
                        l_raw,
                        builder,
                        string_helpers,
                        status_fn,
                        inline_status_ptr_var,
                        emergency_exit_bb,
                    )?;
                    let r_str = lower_to_string_handle(
                        &r_ty_mir,
                        r_raw,
                        builder,
                        string_helpers,
                        status_fn,
                        inline_status_ptr_var,
                        emergency_exit_bb,
                    )?;
                    let use_inplace = l_ty_mir == MirType::String
                        && inplace_concat_dst.is_some_and(|dst_id| {
                            matches!(
                                lhs,
                                Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l))
                                    if *l == dst_id
                            ) && !operand_mentions_local(rhs, dst_id)
                        });
                    let target_fn = if use_inplace {
                        string_helpers.append_inplace_fn
                    } else {
                        string_helpers.concat_fn
                    };
                    let call = builder.ins().call(target_fn, &[l_str, r_str]);
                    let res = builder.inst_results(call)[0];
                    emit_status_check(builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                    return Ok(res);
                }

                let is_float = l_ty_mir == MirType::Float || r_ty_mir == MirType::Float;
                let is_string = l_ty_mir == MirType::String && r_ty_mir == MirType::String;

                if is_float {
                    let mut l = lower_operand(lhs, builder, local_vars, stack_slots, locals, struct_layouts, bc_ctx)?;
                    let mut r = lower_operand(rhs, builder, local_vars, stack_slots, locals, struct_layouts, bc_ctx)?;
                    let l_clif_ty = builder.func.dfg.value_type(l);
                    let r_clif_ty = builder.func.dfg.value_type(r);
                    if l_clif_ty != types::F64 {
                        l = cast_clif_type(builder, l, types::F64);
                    }
                    if r_clif_ty != types::F64 {
                        r = cast_clif_type(builder, r, types::F64);
                    }

                    let val = match binop {
                        MirBinaryOp::Add => builder.ins().fadd(l, r),
                        MirBinaryOp::Sub => builder.ins().fsub(l, r),
                        MirBinaryOp::Mul => builder.ins().fmul(l, r),
                        MirBinaryOp::Div => builder.ins().fdiv(l, r),
                        MirBinaryOp::Mod => {
                            let call = builder.ins().call(rem_float_fn, &[l, r]);
                            let res = builder.inst_results(call)[0];
                            emit_status_check(builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                            res
                        }
                        MirBinaryOp::FloorDiv => {
                            let call = builder.ins().call(floordiv_float_fn, &[l, r]);
                            let res = builder.inst_results(call)[0];
                            emit_status_check(builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                            res
                        }
                        MirBinaryOp::Pow => {
                            let call = builder.ins().call(pow_float_fn, &[l, r]);
                            let res = builder.inst_results(call)[0];
                            emit_status_check(builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                            res
                        }
                        MirBinaryOp::Eq => builder.ins().fcmp(FloatCC::Equal, l, r),
                        MirBinaryOp::NotEq => builder.ins().fcmp(FloatCC::NotEqual, l, r),
                        MirBinaryOp::Lt => builder.ins().fcmp(FloatCC::LessThan, l, r),
                        MirBinaryOp::LtEq => builder.ins().fcmp(FloatCC::LessThanOrEqual, l, r),
                        MirBinaryOp::Gt => builder.ins().fcmp(FloatCC::GreaterThan, l, r),
                        MirBinaryOp::GtEq => builder.ins().fcmp(FloatCC::GreaterThanOrEqual, l, r),
                        _ => {
                            return Err(MirCraneliftBackendError::UnsupportedBinaryOp(binop.clone()));
                        }
                    };
                    return Ok(val);
                }

                if is_string {
                    let l = lower_operand(lhs, builder, local_vars, stack_slots, locals, struct_layouts, bc_ctx)?;
                    let r = lower_operand(rhs, builder, local_vars, stack_slots, locals, struct_layouts, bc_ctx)?;
                    match binop {
                        MirBinaryOp::Add => {
                            let call = builder.ins().call(string_helpers.concat_fn, &[l, r]);
                            let res = builder.inst_results(call)[0];
                            emit_status_check(builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                            return Ok(res);
                        }
                        MirBinaryOp::Eq | MirBinaryOp::NotEq => {
                            let is_eq = matches!(binop, MirBinaryOp::Eq);
                            let is_static_const = |op: &Operand| -> bool {
                                matches!(
                                    op,
                                    Operand::Constant(MirConstant::String(s))
                                        if s.is_empty() || (s.len() == 1 && s.as_bytes()[0] < 128)
                                )
                            };
                            let true_i8 = builder.ins().iconst(types::I8, i64::from(is_eq));
                            let false_i8 = builder.ins().iconst(types::I8, i64::from(!is_eq));
                            let done_bb = builder.create_block();
                            builder.append_block_param(done_bb, types::I8);
                            let slow_bb = builder.create_block();

                            if is_static_const(rhs) || is_static_const(lhs) {
                                let other = if is_static_const(rhs) { l } else { r };
                                let neq_bb = builder.create_block();
                                let same = builder.ins().icmp(IntCC::Equal, l, r);
                                builder.ins().brif(
                                    same,
                                    done_bb,
                                    &[BlockArg::Value(true_i8)],
                                    neq_bb,
                                    &[],
                                );
                                builder.switch_to_block(neq_bb);
                                let other_is_static = builder.ins().icmp_imm_s(
                                    IntCC::SignedLessThanOrEqual,
                                    other,
                                    crate::mir::runtime_helpers::ASCII_HANDLE_BASE,
                                );
                                builder.ins().brif(
                                    other_is_static,
                                    done_bb,
                                    &[BlockArg::Value(false_i8)],
                                    slow_bb,
                                    &[],
                                );
                            } else {
                                let neq_bb = builder.create_block();
                                let same = builder.ins().icmp(IntCC::Equal, l, r);
                                let l_neg = builder.ins().icmp_imm_s(IntCC::SignedLessThan, l, 0);
                                let fast_eq = builder.ins().band(same, l_neg);
                                builder.ins().brif(
                                    fast_eq,
                                    done_bb,
                                    &[BlockArg::Value(true_i8)],
                                    neq_bb,
                                    &[],
                                );
                                builder.switch_to_block(neq_bb);
                                let l_static = builder.ins().icmp_imm_s(
                                    IntCC::SignedLessThanOrEqual,
                                    l,
                                    crate::mir::runtime_helpers::ASCII_HANDLE_BASE,
                                );
                                let r_static = builder.ins().icmp_imm_s(
                                    IntCC::SignedLessThanOrEqual,
                                    r,
                                    crate::mir::runtime_helpers::ASCII_HANDLE_BASE,
                                );
                                let both_static = builder.ins().band(l_static, r_static);
                                builder.ins().brif(
                                    both_static,
                                    done_bb,
                                    &[BlockArg::Value(false_i8)],
                                    slow_bb,
                                    &[],
                                );
                            }

                            builder.switch_to_block(slow_bb);
                            let call = builder.ins().call(string_helpers.eq_fn, &[l, r]);
                            let raw_eq = builder.inst_results(call)[0];
                            let slow_res = if is_eq {
                                raw_eq
                            } else {
                                let one = builder.ins().iconst(types::I8, 1);
                                builder.ins().bxor(raw_eq, one)
                            };
                            builder.ins().jump(done_bb, &[BlockArg::Value(slow_res)]);
                            builder.switch_to_block(done_bb);
                            return Ok(builder.block_params(done_bb)[0]);
                        }
                        _ => {
                            return Err(MirCraneliftBackendError::UnsupportedBinaryOp(binop.clone()));
                        }
                    }
                }

                let mut l = lower_operand(lhs, builder, local_vars, stack_slots, locals, struct_layouts, bc_ctx)?;
                let mut r = lower_operand(rhs, builder, local_vars, stack_slots, locals, struct_layouts, bc_ctx)?;
                let l_clif_ty = builder.func.dfg.value_type(l);
                let r_clif_ty = builder.func.dfg.value_type(r);
                if l_clif_ty != types::I64 {
                    l = cast_clif_type(builder, l, types::I64);
                }
                if r_clif_ty != types::I64 {
                    r = cast_clif_type(builder, r, types::I64);
                }

                let val = match binop {
                    MirBinaryOp::Add => builder.ins().iadd(l, r),
                    MirBinaryOp::Sub => builder.ins().isub(l, r),
                    MirBinaryOp::Mul => builder.ins().imul(l, r),
                    MirBinaryOp::Div => {
                        emit_inline_int_div_or_rem(true, l, r, rhs, builder, bc_ctx, None)
                    }
                    MirBinaryOp::Mod => {
                        emit_inline_int_div_or_rem(false, l, r, rhs, builder, bc_ctx, pow2_mod_out)
                    }
                    MirBinaryOp::FloorDiv => {
                        let call = builder.ins().call(floordiv_fn, &[l, r]);
                        let res = builder.inst_results(call)[0];
                        emit_status_check(builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                        res
                    }
                    MirBinaryOp::Pow => {
                        let call = builder.ins().call(pow_fn, &[l, r]);
                        let res = builder.inst_results(call)[0];
                        emit_status_check(builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                        res
                    }
                    MirBinaryOp::BitAnd => builder.ins().band(l, r),
                    MirBinaryOp::BitOr => builder.ins().bor(l, r),
                    MirBinaryOp::BitXor => builder.ins().bxor(l, r),
                    MirBinaryOp::Shl => {
                        let mask = builder.ins().iconst(types::I64, 63);
                        let sh = builder.ins().band(r, mask);
                        builder.ins().ishl(l, sh)
                    }
                    MirBinaryOp::Shr => {
                        let mask = builder.ins().iconst(types::I64, 63);
                        let sh = builder.ins().band(r, mask);
                        builder.ins().sshr(l, sh)
                    }
                    MirBinaryOp::Eq | MirBinaryOp::NotEq => {
                        let cc = if matches!(binop, MirBinaryOp::Eq) {
                            IntCC::Equal
                        } else {
                            IntCC::NotEqual
                        };
                        // NG-W14-5 peephole: `(v % ±2^k) == 0` / `!= 0` (and `v % i64::MIN == 0`)
                        // holds iff `(v & mask) == 0` / `!= 0` across all signed 64-bit integers.
                        let pow2_zero_check = if resolve_const_int_operand(rhs, bc_ctx.const_int_locals) == Some(0) {
                            match lhs {
                                Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => {
                                    bc_ctx.pow2_mod_defs.get(&loc.0).copied()
                                }
                                _ => None,
                            }
                        } else if resolve_const_int_operand(lhs, bc_ctx.const_int_locals) == Some(0) {
                            match rhs {
                                Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => {
                                    bc_ctx.pow2_mod_defs.get(&loc.0).copied()
                                }
                                _ => None,
                            }
                        } else {
                            None
                        };
                        if let Some((orig_lhs_val, mask)) = pow2_zero_check {
                            let mask_val = builder.ins().iconst(types::I64, mask);
                            let masked = builder.ins().band(orig_lhs_val, mask_val);
                            builder.ins().icmp_imm_s(cc, masked, 0)
                        } else {
                            builder.ins().icmp(cc, l, r)
                        }
                    }
                    MirBinaryOp::Lt => builder.ins().icmp(IntCC::SignedLessThan, l, r),
                    MirBinaryOp::LtEq => builder.ins().icmp(IntCC::SignedLessThanOrEqual, l, r),
                    MirBinaryOp::Gt => builder.ins().icmp(IntCC::SignedGreaterThan, l, r),
                    MirBinaryOp::GtEq => builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, l, r),
                    MirBinaryOp::And => builder.ins().band(l, r),
                    MirBinaryOp::Or => builder.ins().bor(l, r),
                    MirBinaryOp::Concat => {
                        unreachable!("MirBinaryOp::Concat handled above")
                    }
                };
                Ok(val)
            }
            Rvalue::UnaryOp(unop, op) => {
                let op_ty = try_operand_type(op, locals, struct_layouts)?;
                let v = lower_operand(op, builder, local_vars, stack_slots, locals, struct_layouts, bc_ctx)?;
                if op_ty == MirType::Float && matches!(unop, MirUnaryOp::Neg) {
                    return Ok(builder.ins().fneg(v));
                }
                if matches!(unop, MirUnaryOp::Len) {
                    match op_ty {
                        MirType::String => {
                            let call = builder.ins().call(string_helpers.len_fn, &[v]);
                            return Ok(builder.inst_results(call)[0]);
                        }
                        MirType::Array(_) => {
                            if let Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) = op
                                && let Some(&hoisted) = bc_ctx.hoisted_arrays.get(&loc.0)
                            {
                                return Ok(builder.use_var(hoisted.len_var));
                            }
                            return Ok(emit_array_len(
                                op,
                                v,
                                builder,
                                len_fn,
                                emergency_exit_bb,
                                validated_arrays,
                            ));
                        }
                        other => {
                            return Err(MirCraneliftBackendError::CodegenError(format!(
                                "MirUnaryOp::Len requires Array or String operand, got '{other}'"
                            )));
                        }
                    }
                }
                let val = match unop {
                    MirUnaryOp::Neg => builder.ins().ineg(v),
                    MirUnaryOp::Not => {
                        if builder.func.dfg.value_type(v).is_int() {
                            builder.ins().icmp_imm_s(IntCC::Equal, v, 0)
                        } else {
                            return Err(MirCraneliftBackendError::CodegenError(
                                "MirUnaryOp::Not requires bool/int operand".into(),
                            ));
                        }
                    }
                    MirUnaryOp::BitNot => builder.ins().bnot(v),
                    MirUnaryOp::Len => unreachable!("handled above"),
                };
                Ok(val)
            }
            Rvalue::Aggregate(MirType::Struct(name), elements) if struct_layouts.contains_key(name) => {
                if name.is_empty() {
                    return Err(MirCraneliftBackendError::CodegenError(
                        "Unnamed C-flat struct aggregate is not supported".into(),
                    ));
                }
                let layout = &struct_layouts[name];
                if elements.len() != layout.fields.len() {
                    return Err(MirCraneliftBackendError::CodegenError(format!(
                        "Struct '{}' field count mismatch: expected {}, got {}",
                        name,
                        layout.fields.len(),
                        elements.len()
                    )));
                }
                let size_val = builder.ins().iconst(types::I64, layout.size_bytes as i64);
                let (name_ptr, name_len) = intern_static_type_name(name);
                let name_ptr_val = builder.ins().iconst(types::I64, name_ptr as i64);
                let name_len_val = builder.ins().iconst(types::I64, name_len as i64);
                let call_inst = builder.ins().call(
                    alloc_cflat_struct_fn,
                    &[size_val, name_ptr_val, name_len_val],
                );
                let struct_ptr = builder.inst_results(call_inst)[0];
                emit_status_check(builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                for (i, elem) in elements.iter().enumerate() {
                    let el_val = lower_operand(elem, builder, local_vars, stack_slots, locals, struct_layouts, bc_ctx)?;
                    let offset = layout.fields[i].byte_offset as i32;
                    builder.ins().store(MemFlagsData::trusted(), el_val, struct_ptr, offset);
                }
                Ok(struct_ptr)
            }
            Rvalue::Aggregate(MirType::Struct(name), elements) => {
                if name.is_empty() {
                    return Err(MirCraneliftBackendError::CodegenError(
                        "Unnamed Struct aggregate is not supported by Cranelift".into(),
                    ));
                }
                let len_val = builder.ins().iconst(types::I64, elements.len() as i64);
                let (name_ptr, name_len) = intern_static_type_name(name);
                let name_ptr_val = builder.ins().iconst(types::I64, name_ptr as i64);
                let name_len_val = builder.ins().iconst(types::I64, name_len as i64);
                let call_inst = builder.ins().call(
                    alloc_struct_named_fn,
                    &[len_val, name_ptr_val, name_len_val],
                );
                let struct_ptr = builder.inst_results(call_inst)[0];
                emit_status_check(builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                emit_aggregate_word_stores(
                    struct_ptr,
                    elements,
                    builder,
                    local_vars,
                    stack_slots,
                    locals,
                    struct_layouts,
                    bc_ctx,
                )?;
                Ok(struct_ptr)
            }
            Rvalue::Aggregate(MirType::Tuple(elem_types), elements) => {
                if elem_types.len() != elements.len() {
                    return Err(MirCraneliftBackendError::CodegenError(format!(
                        "Tuple arity mismatch: type has {}, elements has {}",
                        elem_types.len(),
                        elements.len()
                    )));
                }
                let mut tags = Vec::with_capacity(elem_types.len());
                for (elem, expected_ty) in elements.iter().zip(elem_types.iter()) {
                    let Some(tag) = mir_array_elem_tag(expected_ty) else {
                        return Err(MirCraneliftBackendError::UnsupportedType(expected_ty.clone()));
                    };
                    let actual_ty = try_operand_type(elem, locals, struct_layouts)?;
                    if actual_ty != *expected_ty {
                        return Err(MirCraneliftBackendError::CodegenError(format!(
                            "Tuple element type mismatch: expected '{expected_ty}', got '{actual_ty}'"
                        )));
                    }
                    tags.push(tag);
                }
                let len_val = builder.ins().iconst(types::I64, elements.len() as i64);
                let tags_addr = if tags.is_empty() {
                    builder.ins().iconst(types::I64, 0)
                } else {
                    let slot = builder.create_sized_stack_slot(StackSlotData::new(
                        StackSlotKind::ExplicitSlot,
                        (tags.len() * 8) as u32,
                        3,
                    ));
                    for (i, tag) in tags.iter().enumerate() {
                        let tag_val = builder.ins().iconst(types::I64, *tag);
                        builder.ins().stack_store(types::I64, tag_val, slot, (i * 8) as i32);
                    }
                    builder.ins().stack_addr(types::I64, slot, 0)
                };
                let call_inst = builder.ins().call(alloc_tuple_typed_fn, &[len_val, tags_addr]);
                let tuple_ptr = builder.inst_results(call_inst)[0];
                emit_status_check(builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                emit_aggregate_word_stores(
                    tuple_ptr,
                    elements,
                    builder,
                    local_vars,
                    stack_slots,
                    locals,
                    struct_layouts,
                    bc_ctx,
                )?;
                Ok(tuple_ptr)
            }
            Rvalue::Aggregate(MirType::Array(elem_ty), elements) => {
                let effective_elem_ty: &MirType = if **elem_ty == MirType::Unknown {
                    match expected_dst_ty {
                        Some(MirType::Array(dst_elem)) if **dst_elem != MirType::Unknown => {
                            dst_elem.as_ref()
                        }
                        _ => elem_ty.as_ref(),
                    }
                } else {
                    elem_ty.as_ref()
                };
                let elem_tag_opt = mir_array_elem_tag(effective_elem_ty);
                let is_struct_elem =
                    matches!(effective_elem_ty, MirType::Struct(name) if !name.is_empty());
                if elem_tag_opt.is_none() && !is_struct_elem {
                    return Err(MirCraneliftBackendError::UnsupportedType(
                        effective_elem_ty.clone(),
                    ));
                }
                for elem in elements {
                    let actual_ty = try_operand_type(elem, locals, struct_layouts)?;
                    if actual_ty != *effective_elem_ty {
                        return Err(MirCraneliftBackendError::CodegenError(format!(
                            "Array element type mismatch: expected '{effective_elem_ty}', got '{actual_ty}'"
                        )));
                    }
                }
                let len_val = builder.ins().iconst(types::I64, elements.len() as i64);
                let call_inst = if let Some(elem_tag) = elem_tag_opt {
                    let tag_val = builder.ins().iconst(types::I64, elem_tag);
                    builder.ins().call(alloc_array_typed_fn, &[len_val, tag_val])
                } else {
                    builder.ins().call(alloc_array_fn, &[len_val])
                };
                let array_ptr = builder.inst_results(call_inst)[0];
                emit_status_check(builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                if *effective_elem_ty == MirType::Bool {
                    for (i, elem) in elements.iter().enumerate() {
                        let el_val = lower_operand(
                            elem,
                            builder,
                            local_vars,
                            stack_slots,
                            locals,
                            struct_layouts,
                            bc_ctx,
                        )?;
                        let el_ty = builder.func.dfg.value_type(el_val);
                        let byte_val = if el_ty == types::I8 {
                            el_val
                        } else {
                            builder.ins().icmp_imm_s(IntCC::NotEqual, el_val, 0)
                        };
                        builder.ins().store(
                            MemFlagsData::trusted(),
                            byte_val,
                            array_ptr,
                            16 + (i as i32),
                        );
                    }
                } else {
                    emit_aggregate_word_stores(
                        array_ptr,
                        elements,
                        builder,
                        local_vars,
                        stack_slots,
                        locals,
                        struct_layouts,
                        bc_ctx,
                    )?;
                }
                Ok(array_ptr)
            }
            Rvalue::Closure { func_name, captures } => {
                let Some(fn_id) = bc_ctx.closure_fn_ids.and_then(|m| m.get(func_name).copied()) else {
                    return Err(MirCraneliftBackendError::UnlowerableValue(format!(
                        "Rvalue::Closure target '{func_name}' not found in compiled module"
                    )));
                };
                let word_count = (1 + captures.len()) as i64;
                let len_val = builder.ins().iconst(types::I64, word_count);
                let call_inst = builder.ins().call(alloc_array_fn, &[len_val]);
                let closure_ptr = builder.inst_results(call_inst)[0];
                emit_status_check(builder, status_fn, inline_status_ptr_var, emergency_exit_bb);
                let fn_id_val = builder.ins().iconst(types::I64, fn_id);
                builder
                    .ins()
                    .store(MemFlagsData::trusted(), fn_id_val, closure_ptr, 16);
                for (i, (_, cap_op, _)) in captures.iter().enumerate() {
                    let cap_val = lower_operand(
                        cap_op,
                        builder,
                        local_vars,
                        stack_slots,
                        locals,
                        struct_layouts,
                        bc_ctx,
                    )?;
                    let cap_i64 = clif_val_to_raw_i64(builder, cap_val);
                    builder.ins().store(
                        MemFlagsData::trusted(),
                        cap_i64,
                        closure_ptr,
                        16 + ((1 + i) as i32) * 8,
                    );
                }
                Ok(closure_ptr)
            }
            _ => Err(MirCraneliftBackendError::UnsupportedConstant(MirConstant::Unit)),
        }
    }
}
