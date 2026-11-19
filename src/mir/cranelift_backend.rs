#[cfg(feature = "std")]
pub use jit_engine::{
    clear_registered_struct_layouts, find_loop_backedges, find_loop_headers,
    get_registered_struct_layout, get_registered_struct_layout_by_name, ic_dispatch,
    ic_invalidate, ic_invalidate_all, ic_probe, ic_register_site, ic_reset, ic_site_stats,
    ic_update, is_loop_backedge, lust_mir_abs_int, lust_mir_alloc_array,
    lust_mir_alloc_array_typed, lust_mir_alloc_cflat_struct, lust_mir_alloc_enum,
    lust_mir_alloc_map, lust_mir_alloc_struct, lust_mir_alloc_struct_named,
    lust_mir_alloc_tuple, lust_mir_alloc_tuple_typed, lust_mir_array_get, lust_mir_array_len,
    lust_mir_array_len_checked, lust_mir_array_set,
    mir_array_elem_tag, MIR_ARRAY_ELEM_TAG_BOOL, MIR_ARRAY_ELEM_TAG_FLOAT,
    MIR_ARRAY_ELEM_TAG_INT, MIR_ARRAY_ELEM_TAG_STRING,
    lust_mir_charge_gas, lust_mir_clamp_int, lust_mir_div_int, lust_mir_enum_payload,
    lust_mir_enum_tag, lust_mir_floordiv_float, lust_mir_floordiv_int, lust_mir_get_gas,
    lust_mir_get_memory_used, lust_mir_get_status, lust_mir_ic_dispatch, lust_mir_ic_miss,
    lust_mir_ic_probe, lust_mir_init_deopt, lust_mir_init_deopt_at_terminator, lust_mir_map_get, lust_mir_map_len,
    lust_mir_map_set, lust_mir_max_int, lust_mir_min_int, lust_mir_pow_float,
    lust_mir_pow_int, lust_mir_print_int, lust_mir_record_deopt, lust_mir_record_deopt_local,
    lust_mir_record_native_entry, lust_mir_rem_float, lust_mir_rem_int, lust_mir_safepoint,
    lust_mir_set_gas, lust_mir_set_max_memory, lust_mir_set_memory_limit, lust_mir_set_status,
    lust_mir_string_concat, lust_mir_string_eq, lust_mir_string_len,
    lust_mir_struct_decode_binary, lust_mir_struct_decode_contiguous, publish_deopt_payload,
    publish_deopt_payload_with_func, publish_deopt_payload_name_only,
    publish_deopt_payload_at_terminator, publish_deopt_payload_with_site_kind,
    refine_locals, refine_locals_with_params, refine_locals_with_context, refine_mir_function, refine_mir_function_with_layouts, apply_module_callee_return_types, register_struct_layout, reset_native_instrumentation_counters, safepoint_enter_mask,
    safepoint_exit_mask, safepoint_register_finalizer, safepoint_request_deopt,
    safepoint_request_interrupt, safepoint_reset, safepoint_stats, take_deopt_payload,
    set_current_deopt_payload, with_current_deopt_payload_mut,
    take_current_sticky_error, get_current_sticky_error, set_current_sticky_error,
    take_last_schema_decode_error, set_current_native_function, get_current_native_function,
    push_native_invocation, push_native_invocation_with_generation, get_current_native_invocation, native_invocation_depth,
    is_vm_effect_intrinsic_callee, is_effect_or_handler_callee,
    lust_mir_enter_native_callee, lust_mir_exit_native_callee,
    NativeInvocationIdentity, NativeInvocationGuard, DeoptSiteKind,
    CraneliftJitCompiler, DeoptPayload, IcEntry, IcSiteStats,
    IcStateKind, InlineCacheSite, MirCraneliftBackendError, OsrPromotionTracker,
    SafepointRuntimeState, SafepointStatsSnapshot, SchemaDecodeError,
    CRANELIFT_NATIVE_ENTRY_COUNT, CRANELIFT_SIMD_SCALAR_COUNT, CRANELIFT_SIMD_VECTOR_COUNT,
    DIRECT_DECODE_COUNT, DYNAMIC_LOOKUP_COUNT,
    lust_mir_is_simd_scalar_forced, lust_mir_record_simd_scalar_op,
    lust_mir_record_simd_vector_op, MIR_STATUS_DEOPT, MIR_STATUS_ERR_DIV_BY_ZERO,
    MIR_STATUS_ERR_INTERRUPTED, MIR_STATUS_ERR_NEG_EXPONENT, MIR_STATUS_ERR_OOM,
    MIR_STATUS_ERR_OUT_OF_BOUNDS, MIR_STATUS_ERR_OUT_OF_GAS, MIR_STATUS_OK,
};

#[cfg(all(feature = "std", feature = "test-deopt"))]
pub use jit_engine::{
    refine_locals_with_seam, refine_mir_function_with_seam, safepoint_request_deopt_on_poll,
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
        types, AbiParam, Block, FuncRef, InstBuilder, MemFlagsData, Signature, StackSlot, StackSlotData,
        StackSlotKind, Value as ClifValue,
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

    #[derive(Default)]
    pub struct SafepointRuntimeState {
        pub mask_depth: usize,
        pub pending_interrupt: Option<crate::effect_abi::InterruptReason>,
        pub safepoints_checked: u64,
        pub safepoints_masked_deferred: u64,
        pub safepoints_delivered: u64,
        pub deopt_requested: bool,
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
        static STANDALONE_DEOPT_PAYLOAD: core::cell::RefCell<Option<DeoptPayload>> = const { core::cell::RefCell::new(None) };
        static STANDALONE_STICKY_ERROR: core::cell::RefCell<Option<String>> = const { core::cell::RefCell::new(None) };
        static CURRENT_NATIVE_INVOCATION_STACK: core::cell::RefCell<Vec<NativeInvocationEntry>> = const { core::cell::RefCell::new(Vec::new()) };
        static STRUCT_LAYOUT_REGISTRY: core::cell::RefCell<hashbrown::HashMap<u64, MirStructLayout>> = core::cell::RefCell::new(hashbrown::HashMap::new());
        static STRUCT_NAME_MAP: core::cell::RefCell<hashbrown::HashMap<String, u64>> = core::cell::RefCell::new(hashbrown::HashMap::new());
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
    pub extern "C" fn lust_mir_safepoint() -> i64 {
        SAFEPOINT_STATE.with(|state_cell| {
            let mut st = state_cell.borrow_mut();
            st.safepoints_checked += 1;
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
                        (target_func.is_none(), st.safepoints_checked)
                    }
                });
                if matches_fn && current_poll == target_poll {
                    st.deopt_at_check_ordinal = None;
                    st.deopt_target_func = None;
                    if is_nested_native_callee() {
                        return fail_nested_native_deopt();
                    }
                    with_current_mir_context(|ctx| {
                        ctx.last_status.store(MIR_STATUS_DEOPT, core::sync::atomic::Ordering::SeqCst);
                    });
                    return MIR_STATUS_DEOPT;
                }
            }
            if st.deopt_requested {
                if is_nested_native_callee() {
                    st.deopt_requested = false;
                    return fail_nested_native_deopt();
                }
                with_current_mir_context(|ctx| {
                    ctx.last_status.store(MIR_STATUS_DEOPT, core::sync::atomic::Ordering::SeqCst);
                });
                return MIR_STATUS_DEOPT;
            }
            if st.mask_depth > 0 {
                if st.pending_interrupt.is_some() {
                    st.safepoints_masked_deferred += 1;
                }
                return 0;
            }
            if let Some(_reason) = st.pending_interrupt.take() {
                st.safepoints_delivered += 1;
                let finalizers = core::mem::take(&mut st.finalizers);
                for f in finalizers {
                    f();
                    st.cleanups_run += 1;
                }
                with_current_mir_context(|ctx| {
                    ctx.last_status.store(MIR_STATUS_ERR_INTERRUPTED, core::sync::atomic::Ordering::SeqCst);
                });
                return MIR_STATUS_ERR_INTERRUPTED;
            }
            0
        })
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
            st.borrow_mut().mask_depth += 1;
        });
    }

    pub fn safepoint_exit_mask() -> Option<crate::effect_abi::InterruptReason> {
        SAFEPOINT_STATE.with(|st| {
            let mut state = st.borrow_mut();
            state.mask_depth = state.mask_depth.saturating_sub(1);
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
        });
    }

    pub fn safepoint_register_finalizer<F: FnOnce() + Send + 'static>(f: F) {
        SAFEPOINT_STATE.with(|st| {
            st.borrow_mut().finalizers.push(alloc::boxed::Box::new(f));
        });
    }

    pub fn safepoint_request_deopt(deopt: bool) {
        SAFEPOINT_STATE.with(|st| {
            st.borrow_mut().deopt_requested = deopt;
        });
    }

    #[cfg(feature = "test-deopt")]
    pub fn safepoint_request_deopt_on_poll(func_name: Option<String>, poll_ordinal: u64) {
        SAFEPOINT_STATE.with(|st| {
            let mut state = st.borrow_mut();
            state.deopt_target_func = func_name;
            state.deopt_at_check_ordinal = Some(poll_ordinal);
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
        SAFEPOINT_STATE.with(|st| {
            let s = st.borrow();
            SafepointStatsSnapshot {
                mask_depth: s.mask_depth,
                safepoints_checked: s.safepoints_checked,
                safepoints_masked_deferred: s.safepoints_masked_deferred,
                safepoints_delivered: s.safepoints_delivered,
                cleanups_run: s.cleanups_run,
            }
        })
    }

    pub fn safepoint_reset() {
        SAFEPOINT_STATE.with(|st| {
            *st.borrow_mut() = SafepointRuntimeState::default();
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

    pub fn find_loop_headers(blocks: &[BasicBlock]) -> hashbrown::HashSet<usize> {
        let mut loop_headers = hashbrown::HashSet::new();
        for (src_idx, block) in blocks.iter().enumerate() {
            if let Some(term) = &block.terminator {
                match term {
                    Terminator::Goto(target, _) => {
                        if target.0 <= src_idx {
                            loop_headers.insert(target.0);
                        }
                    }
                    Terminator::Cond { then_block, else_block, .. } => {
                        if then_block.0 <= src_idx {
                            loop_headers.insert(then_block.0);
                        }
                        if else_block.0 <= src_idx {
                            loop_headers.insert(else_block.0);
                        }
                    }
                    Terminator::Switch { cases, default, .. } => {
                        for (_, target) in cases {
                            if target.0 <= src_idx {
                                loop_headers.insert(target.0);
                            }
                        }
                        if default.0 <= src_idx {
                            loop_headers.insert(default.0);
                        }
                    }
                    _ => {}
                }
            }
        }
        loop_headers
    }

    pub fn find_loop_backedges(blocks: &[BasicBlock]) -> Vec<(usize, usize)> {
        let mut backedges = Vec::new();
        for (src_idx, block) in blocks.iter().enumerate() {
            if let Some(term) = &block.terminator {
                match term {
                    Terminator::Goto(target, _) => {
                        if target.0 <= src_idx {
                            backedges.push((src_idx, target.0));
                        }
                    }
                    Terminator::Cond { then_block, else_block, .. } => {
                        if then_block.0 <= src_idx {
                            backedges.push((src_idx, then_block.0));
                        }
                        if else_block.0 <= src_idx {
                            backedges.push((src_idx, else_block.0));
                        }
                    }
                    Terminator::Switch { cases, default, .. } => {
                        for (_, target) in cases {
                            if target.0 <= src_idx {
                                backedges.push((src_idx, target.0));
                            }
                        }
                        if default.0 <= src_idx {
                            backedges.push((src_idx, default.0));
                        }
                    }
                    _ => {}
                }
            }
        }
        backedges
    }

    pub fn is_loop_backedge(blocks: &[BasicBlock], from: usize, to: usize) -> bool {
        if to > from {
            return false;
        }
        if let Some(block) = blocks.get(from) {
            if let Some(term) = &block.terminator {
                return match term {
                    Terminator::Goto(target, _) => target.0 == to,
                    Terminator::Cond { then_block, else_block, .. } => {
                        then_block.0 == to || else_block.0 == to
                    }
                    Terminator::Switch { cases, default, .. } => {
                        cases.iter().any(|(_, target)| target.0 == to) || default.0 == to
                    }
                    _ => false,
                };
            }
        }
        false
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

    fn emit_safepoint_check(
        builder: &mut FunctionBuilder,
        safepoint_fn: FuncRef,
        init_deopt_fn: FuncRef,
        record_deopt_local_fn: FuncRef,
        target_block_idx: usize,
        live_locals: &[LocalId],
        local_vars: &[Variable],
        stack_slots: &hashbrown::HashMap<usize, StackSlot>,
        locals: &[LocalDecl],
        emergency_exit_bb: Block,
        pending_vec_reduction: Option<(LocalId, Variable, Variable)>,
    ) {
        let sp_call = builder.ins().call(safepoint_fn, &[]);
        let sp_res = builder.inst_results(sp_call)[0];
        let is_err = builder.ins().icmp_imm_s(IntCC::NotEqual, sp_res, 0);

        let cont_bb = builder.create_block();
        let err_dispatch_bb = builder.create_block();
        builder.ins().brif(is_err, err_dispatch_bb, &[], cont_bb, &[]);

        builder.switch_to_block(err_dispatch_bb);
        let is_deopt = builder.ins().icmp_imm_s(IntCC::Equal, sp_res, MIR_STATUS_DEOPT);
        let deopt_capture_bb = builder.create_block();
        builder.ins().brif(is_deopt, deopt_capture_bb, &[], emergency_exit_bb, &[]);

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
        builder.ins().jump(emergency_exit_bb, &[]);

        builder.switch_to_block(cont_bb);
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
        #[cfg(feature = "test-deopt")]
        refine_max_iterations_override: Option<usize>,
    }

    impl CraneliftJitCompiler {
        pub fn new() -> Result<Self, String> {
            let mut flag_builder = settings::builder();
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
            builder.symbol("lust_mir_set_gas", lust_mir_set_gas as *const u8);
            builder.symbol("lust_mir_get_gas", lust_mir_get_gas as *const u8);
            builder.symbol("lust_mir_charge_gas", lust_mir_charge_gas as *const u8);
            builder.symbol("lust_mir_alloc_array", lust_mir_alloc_array as *const u8);
            builder.symbol("lust_mir_alloc_array_typed", lust_mir_alloc_array_typed as *const u8);
            builder.symbol("lust_mir_alloc_tuple", lust_mir_alloc_tuple as *const u8);
            builder.symbol("lust_mir_alloc_tuple_typed", lust_mir_alloc_tuple_typed as *const u8);
            builder.symbol("lust_mir_alloc_struct", lust_mir_alloc_struct as *const u8);
            builder.symbol("lust_mir_alloc_struct_named", lust_mir_alloc_struct_named as *const u8);
            builder.symbol("lust_mir_alloc_cflat_struct", lust_mir_alloc_cflat_struct as *const u8);
            builder.symbol("lust_mir_array_len", lust_mir_array_len as *const u8);
            builder.symbol("lust_mir_array_len_checked", lust_mir_array_len_checked as *const u8);
            builder.symbol("lust_mir_array_get", lust_mir_array_get as *const u8);
            builder.symbol("lust_mir_array_set", lust_mir_array_set as *const u8);
            builder.symbol("lust_mir_floordiv_int", lust_mir_floordiv_int as *const u8);
            builder.symbol("lust_mir_pow_int", lust_mir_pow_int as *const u8);
            builder.symbol("lust_mir_abs_int", lust_mir_abs_int as *const u8);
            builder.symbol("lust_mir_min_int", lust_mir_min_int as *const u8);
            builder.symbol("lust_mir_max_int", lust_mir_max_int as *const u8);
            builder.symbol("lust_mir_clamp_int", lust_mir_clamp_int as *const u8);
            builder.symbol("lust_mir_get_status", lust_mir_get_status as *const u8);
            builder.symbol("lust_mir_set_status", lust_mir_set_status as *const u8);
            builder.symbol("lust_mir_print_int", lust_mir_print_int as *const u8);
            builder.symbol("lust_mir_div_int", lust_mir_div_int as *const u8);
            builder.symbol("lust_mir_rem_int", lust_mir_rem_int as *const u8);

            // Float helpers
            builder.symbol("lust_mir_floordiv_float", lust_mir_floordiv_float as *const u8);
            builder.symbol("lust_mir_pow_float", lust_mir_pow_float as *const u8);
            builder.symbol("lust_mir_rem_float", lust_mir_rem_float as *const u8);

            // String helpers
            builder.symbol("lust_mir_string_concat", lust_mir_string_concat as *const u8);
            builder.symbol("lust_mir_string_eq", lust_mir_string_eq as *const u8);
            builder.symbol("lust_mir_string_len", lust_mir_string_len as *const u8);

            // Map helpers
            builder.symbol("lust_mir_alloc_map", lust_mir_alloc_map as *const u8);
            builder.symbol("lust_mir_map_get", lust_mir_map_get as *const u8);
            builder.symbol("lust_mir_map_set", lust_mir_map_set as *const u8);
            builder.symbol("lust_mir_map_len", lust_mir_map_len as *const u8);

            // Enum helpers
            builder.symbol("lust_mir_alloc_enum", lust_mir_alloc_enum as *const u8);
            builder.symbol("lust_mir_enum_tag", lust_mir_enum_tag as *const u8);
            builder.symbol("lust_mir_enum_payload", lust_mir_enum_payload as *const u8);

            // Safepoint and deopt helpers
            builder.symbol("lust_mir_safepoint", lust_mir_safepoint as *const u8);
            builder.symbol("lust_mir_init_deopt", lust_mir_init_deopt as *const u8);
            builder.symbol("lust_mir_init_deopt_at_terminator", lust_mir_init_deopt_at_terminator as *const u8);
            builder.symbol("lust_mir_record_deopt_local", lust_mir_record_deopt_local as *const u8);
            builder.symbol("lust_mir_record_deopt", lust_mir_record_deopt as *const u8);
            builder.symbol("lust_mir_enter_native_callee", lust_mir_enter_native_callee as *const u8);
            builder.symbol("lust_mir_exit_native_callee", lust_mir_exit_native_callee as *const u8);

            // J-IC helpers
            builder.symbol("lust_mir_ic_probe", lust_mir_ic_probe as *const u8);
            builder.symbol("lust_mir_ic_miss", lust_mir_ic_miss as *const u8);
            builder.symbol("lust_mir_ic_dispatch", lust_mir_ic_dispatch as *const u8);

            // Gate-SCHEMA-09/13 struct decoder & native entry instrumentation helpers
            builder.symbol("lust_mir_struct_decode_contiguous", lust_mir_struct_decode_contiguous as *const u8);
            builder.symbol("lust_mir_struct_decode_binary", lust_mir_struct_decode_binary as *const u8);
            builder.symbol("lust_mir_record_native_entry", lust_mir_record_native_entry as *const u8);
            builder.symbol("lust_mir_record_simd_vector_op", lust_mir_record_simd_vector_op as *const u8);
            builder.symbol("lust_mir_record_simd_scalar_op", lust_mir_record_simd_scalar_op as *const u8);
            builder.symbol("lust_mir_is_simd_scalar_forced", lust_mir_is_simd_scalar_forced as *const u8);

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
                #[cfg(feature = "test-deopt")]
                refine_max_iterations_override: None,
            })
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
            self.register_struct_layouts(&module.struct_layouts);

            let mut module_funcs = hashbrown::HashMap::new();
            let callee_rets: hashbrown::HashMap<String, MirType> = module
                .functions
                .iter()
                .map(|f| (f.signature.name.clone(), f.signature.return_type.clone()))
                .collect();
            let mut refined_functions = Vec::with_capacity(module.functions.len());
            for f in &module.functions {
                let mut f_with_calls = f.clone();
                apply_module_callee_return_types(&mut f_with_calls, &callee_rets);
                refined_functions.push(self.refine_function_for_compiler(&f_with_calls));
            }

            // 1. Declare all functions first so they can reference each other
            for mir_fn in &refined_functions {
                let mut sig = self.module.make_signature();
                for param_id in &mir_fn.params {
                    let p_ty = &mir_fn.locals[param_id.0].ty;
                    sig.params.push(AbiParam::new(mir_to_clif_type(p_ty)?));
                }

                let ret_clif_ty = mir_to_clif_type(&mir_fn.signature.return_type)?;
                if ret_clif_ty != types::I8 || mir_fn.signature.return_type != MirType::Unit {
                    sig.returns.push(AbiParam::new(ret_clif_ty));
                }

                let func_id = self
                    .module
                    .declare_function(&mir_fn.signature.name, Linkage::Export, &sig)
                    .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;
                module_funcs.insert(mir_fn.signature.name.clone(), func_id);
            }

            // 2. Compile each function definition
            for mir_fn in &refined_functions {
                let func_id = module_funcs[&mir_fn.signature.name];
                self.compile_function_internal(mir_fn, func_id, &module_funcs)?;
            }

            // 3. Finalize definitions
            self.module
                .finalize_definitions()
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            // 4. Collect finalized code pointers and code ranges
            let mut res = hashbrown::HashMap::new();
            for mir_fn in &refined_functions {
                let func_id = module_funcs[&mir_fn.signature.name];
                let code_ptr = self.module.get_finalized_function(func_id);
                let size = self.compiled_code_sizes.get(&func_id).copied().unwrap_or(1);
                let start = code_ptr as usize;
                self.compiled_code_ranges.insert(
                    mir_fn.signature.name.clone(),
                    (start, start.saturating_add(size.max(1))),
                );
                res.insert(mir_fn.signature.name.clone(), code_ptr);
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
                            for (i, arg) in args.iter().enumerate() {
                                let arg_ty = try_operand_type(arg, &f.locals, &self.struct_layouts)?;
                                let expected_ty = target_fn
                                    .locals
                                    .get(target_fn.params[i].0)
                                    .map(|l| &l.ty)
                                    .unwrap_or_else(|| &target_fn.signature.param_types[i]);
                                if !Self::mir_types_compatible_for_call(&arg_ty, expected_ty) {
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
            self.compile_function_with_osr(mir_fn, func_id, module_funcs, None)
        }

        fn compile_function_with_osr(
            &mut self,
            mir_fn: &MirFunction,
            func_id: FuncId,
            module_funcs: &hashbrown::HashMap<String, FuncId>,
            osr_info: Option<(BlockId, &[LocalId])>,
        ) -> Result<(), MirCraneliftBackendError> {
            self.module.clear_context(&mut self.ctx);
            self.builder_ctx = FunctionBuilderContext::new();
            let res = self.compile_function_with_osr_inner(mir_fn, func_id, module_funcs, osr_info);
            if res.is_err() {
                self.module.clear_context(&mut self.ctx);
                self.builder_ctx = FunctionBuilderContext::new();
            }
            res
        }

        fn compile_function_with_osr_inner(
            &mut self,
            mir_fn: &MirFunction,
            func_id: FuncId,
            module_funcs: &hashbrown::HashMap<String, FuncId>,
            osr_info: Option<(BlockId, &[LocalId])>,
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

            // Fail-closed compile boundary: ensure all locals have proven concrete types.
            // Any unresolved or conflicting Unknown type is rejected before native mutation.
            for (idx, local) in mir_fn.locals.iter().enumerate() {
                if local.ty == MirType::Unknown {
                    return Err(MirCraneliftBackendError::CodegenError(format!(
                        "Function '{}' local {} ('{}') has unproven or conflicting type 'unknown'",
                        mir_fn.signature.name,
                        idx,
                        local.name.as_deref().unwrap_or("<anonymous>")
                    )));
                }
            }

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

            let gas_callee = self
                .module
                .declare_function("lust_mir_charge_gas", Linkage::Import, &sig1)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let mut alloc_typed_sig = self.module.make_signature();
            alloc_typed_sig.params.push(AbiParam::new(types::I64));
            alloc_typed_sig.params.push(AbiParam::new(types::I64));
            alloc_typed_sig.returns.push(AbiParam::new(types::I64));

            let alloc_array_typed_callee = self
                .module
                .declare_function("lust_mir_alloc_array_typed", Linkage::Import, &alloc_typed_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let alloc_tuple_callee = self
                .module
                .declare_function("lust_mir_alloc_tuple", Linkage::Import, &sig1)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let alloc_tuple_typed_callee = self
                .module
                .declare_function("lust_mir_alloc_tuple_typed", Linkage::Import, &alloc_typed_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let alloc_struct_callee = self
                .module
                .declare_function("lust_mir_alloc_struct", Linkage::Import, &sig1)
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

            let mut print_sig = self.module.make_signature();
            print_sig.params.push(AbiParam::new(types::I64));
            let print_callee = self
                .module
                .declare_function("lust_mir_print_int", Linkage::Import, &print_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            // Declare safepoint and deopt imports
            let mut sp_sig = self.module.make_signature();
            sp_sig.returns.push(AbiParam::new(types::I64));
            let safepoint_callee = self
                .module
                .declare_function("lust_mir_safepoint", Linkage::Import, &sp_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let mut init_deopt_sig = self.module.make_signature();
            init_deopt_sig.params.push(AbiParam::new(types::I64));
            let init_deopt_callee = self
                .module
                .declare_function("lust_mir_init_deopt", Linkage::Import, &init_deopt_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let init_deopt_at_terminator_callee = self
                .module
                .declare_function("lust_mir_init_deopt_at_terminator", Linkage::Import, &init_deopt_sig)
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

            let div_callee = self
                .module
                .declare_function("lust_mir_div_int", Linkage::Import, &binop_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let rem_callee = self
                .module
                .declare_function("lust_mir_rem_int", Linkage::Import, &binop_sig)
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

            let abs_callee = self
                .module
                .declare_function("lust_mir_abs_int", Linkage::Import, &sig1)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let len_callee = self
                .module
                .declare_function("lust_mir_array_len", Linkage::Import, &sig1)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;
            let len_checked_callee = self
                .module
                .declare_function("lust_mir_array_len_checked", Linkage::Import, &sig1)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let mut sig2 = self.module.make_signature();
            sig2.params.push(AbiParam::new(types::I64));
            sig2.params.push(AbiParam::new(types::I64));
            sig2.returns.push(AbiParam::new(types::I64));

            let min_callee = self
                .module
                .declare_function("lust_mir_min_int", Linkage::Import, &sig2)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let max_callee = self
                .module
                .declare_function("lust_mir_max_int", Linkage::Import, &sig2)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let mut sig3 = self.module.make_signature();
            sig3.params.push(AbiParam::new(types::I64));
            sig3.params.push(AbiParam::new(types::I64));
            sig3.params.push(AbiParam::new(types::I64));
            sig3.returns.push(AbiParam::new(types::I64));

            let clamp_callee = self
                .module
                .declare_function("lust_mir_clamp_int", Linkage::Import, &sig3)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let map_get_callee = self
                .module
                .declare_function("lust_mir_map_get", Linkage::Import, &binop_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let mut sig_void3 = self.module.make_signature();
            sig_void3.params.push(AbiParam::new(types::I64));
            sig_void3.params.push(AbiParam::new(types::I64));
            sig_void3.params.push(AbiParam::new(types::I64));

            let map_set_callee = self
                .module
                .declare_function("lust_mir_map_set", Linkage::Import, &sig_void3)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let map_len_callee = self
                .module
                .declare_function("lust_mir_map_len", Linkage::Import, &sig1)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let mut sig0_ret = self.module.make_signature();
            sig0_ret.returns.push(AbiParam::new(types::I64));

            let alloc_map_callee = self
                .module
                .declare_function("lust_mir_alloc_map", Linkage::Import, &sig0_ret)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let alloc_enum_callee = self
                .module
                .declare_function("lust_mir_alloc_enum", Linkage::Import, &binop_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let enum_tag_callee = self
                .module
                .declare_function("lust_mir_enum_tag", Linkage::Import, &sig1)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let enum_payload_callee = self
                .module
                .declare_function("lust_mir_enum_payload", Linkage::Import, &sig1)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            // Declare J-IC dispatch callee
            let ic_dispatch_callee = self
                .module
                .declare_function("lust_mir_ic_dispatch", Linkage::Import, &sig3)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            // Declare struct decode & native entry callees
            let struct_decode_callee = self
                .module
                .declare_function("lust_mir_struct_decode_contiguous", Linkage::Import, &binop_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let sig_void0 = self.module.make_signature();
            let enter_native_callee_id = self
                .module
                .declare_function("lust_mir_enter_native_callee", Linkage::Import, &sig_void0)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;
            let exit_native_callee_id = self
                .module
                .declare_function("lust_mir_exit_native_callee", Linkage::Import, &sig_void0)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;
            let native_entry_callee = self
                .module
                .declare_function("lust_mir_record_native_entry", Linkage::Import, &sig_void0)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;
            let simd_vector_callee = self
                .module
                .declare_function("lust_mir_record_simd_vector_op", Linkage::Import, &sig_void0)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;
            let simd_scalar_callee = self
                .module
                .declare_function("lust_mir_record_simd_scalar_op", Linkage::Import, &sig_void0)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;
            let simd_scalar_forced_callee = self
                .module
                .declare_function("lust_mir_is_simd_scalar_forced", Linkage::Import, &sig0_i64)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let vec_loops = if osr_info.is_none() {
                crate::mir::opt::simd::recognize_vectorized_loop_blocks(mir_fn)
            } else {
                Vec::new()
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
                let alloc_array_typed_fn = self.module.declare_func_in_func(alloc_array_typed_callee, &mut builder.func);
                let _alloc_tuple_fn = self.module.declare_func_in_func(alloc_tuple_callee, &mut builder.func);
                let alloc_tuple_typed_fn = self.module.declare_func_in_func(alloc_tuple_typed_callee, &mut builder.func);
                let _alloc_struct_fn = self.module.declare_func_in_func(alloc_struct_callee, &mut builder.func);
                let alloc_struct_named_fn = self.module.declare_func_in_func(alloc_struct_named_callee, &mut builder.func);
                let alloc_cflat_struct_fn = self.module.declare_func_in_func(alloc_cflat_struct_callee, &mut builder.func);
                let len_fn = self.module.declare_func_in_func(len_checked_callee, &mut builder.func);
                let gas_fn = self.module.declare_func_in_func(gas_callee, &mut builder.func);
                let status_fn = self.module.declare_func_in_func(status_callee, &mut builder.func);
                let status_set_fn = self.module.declare_func_in_func(status_set_callee, &mut builder.func);
                let print_fn = self.module.declare_func_in_func(print_callee, &mut builder.func);
                let floordiv_fn = self.module.declare_func_in_func(floordiv_callee, &mut builder.func);
                let pow_fn = self.module.declare_func_in_func(pow_callee, &mut builder.func);
                let div_fn = self.module.declare_func_in_func(div_callee, &mut builder.func);
                let rem_fn = self.module.declare_func_in_func(rem_callee, &mut builder.func);
                let floordiv_float_fn = self.module.declare_func_in_func(floordiv_float_callee, &mut builder.func);
                let pow_float_fn = self.module.declare_func_in_func(pow_float_callee, &mut builder.func);
                let rem_float_fn = self.module.declare_func_in_func(rem_float_callee, &mut builder.func);
                let string_concat_fn = self.module.declare_func_in_func(string_concat_callee, &mut builder.func);
                let string_eq_fn = self.module.declare_func_in_func(string_eq_callee, &mut builder.func);
                let string_len_fn = self.module.declare_func_in_func(string_len_callee, &mut builder.func);
                let safepoint_fn = self.module.declare_func_in_func(safepoint_callee, &mut builder.func);
                let init_deopt_fn = self.module.declare_func_in_func(init_deopt_callee, &mut builder.func);
                let init_deopt_at_terminator_fn = self.module.declare_func_in_func(init_deopt_at_terminator_callee, &mut builder.func);
                let record_deopt_local_fn = self.module.declare_func_in_func(rec_deopt_callee, &mut builder.func);
                let ic_dispatch_fn = self.module.declare_func_in_func(ic_dispatch_callee, &mut builder.func);
                let struct_decode_fn = self.module.declare_func_in_func(struct_decode_callee, &mut builder.func);
                let native_entry_fn = self.module.declare_func_in_func(native_entry_callee, &mut builder.func);
                let enter_native_callee_fn = self.module.declare_func_in_func(enter_native_callee_id, &mut builder.func);
                let exit_native_callee_fn = self.module.declare_func_in_func(exit_native_callee_id, &mut builder.func);
                let simd_vector_fn = self.module.declare_func_in_func(simd_vector_callee, &mut builder.func);
                let simd_scalar_fn = self.module.declare_func_in_func(simd_scalar_callee, &mut builder.func);
                let simd_scalar_forced_fn = self.module.declare_func_in_func(simd_scalar_forced_callee, &mut builder.func);

                let emergency_exit_bb = builder.create_block();
                let bc_ctx = BoundsCheckCtx {
                    status_set_fn,
                    emergency_exit_bb,
                };

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

                // Declare SSA variables and Cranelift StackSlots for is_stack_promoted locals
                let mut local_vars = Vec::with_capacity(mir_fn.locals.len());
                let mut stack_slots: hashbrown::HashMap<usize, StackSlot> = hashbrown::HashMap::new();
                for local in &mir_fn.locals {
                    let clif_ty = mir_to_clif_type(&local.ty)?;
                    let var = builder.declare_var(clif_ty);
                    local_vars.push(var);
                    if local.is_stack_promoted {
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

                let loop_headers = find_loop_headers(&mir_fn.blocks);

                if let Some((target_block, live_locals)) = osr_info {
                    let osr_entry_bb = builder.create_block();
                    builder.append_block_params_for_function_params(osr_entry_bb);
                    builder.switch_to_block(osr_entry_bb);
                    builder.ins().call(native_entry_fn, &[]);

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
                        safepoint_fn,
                        init_deopt_fn,
                        record_deopt_local_fn,
                        target_block.0,
                        live_locals,
                        &local_vars,
                        &stack_slots,
                        &mir_fn.locals,
                        emergency_exit_bb,
                        None,
                    );

                    builder.ins().jump(clif_blocks[target_block.0], &[]);
                } else {
                    let fn_entry_bb = builder.create_block();
                    builder.append_block_params_for_function_params(fn_entry_bb);
                    builder.switch_to_block(fn_entry_bb);
                    builder.ins().call(native_entry_fn, &[]);

                    for (idx, p_id) in mir_fn.params.iter().enumerate() {
                        let param_val = builder.block_params(fn_entry_bb)[idx];
                        builder.def_var(local_vars[p_id.0], param_val);
                        if let Some(&slot) = stack_slots.get(&p_id.0) {
                            builder.ins().stack_store(types::I64, param_val, slot, 0);
                        }
                    }
                    for acc_pair in vec_acc_vars.iter().flatten() {
                        let zero_s = builder.ins().iconst(types::I64, 0);
                        let vzero = builder.ins().splat(types::I64X2, zero_s);
                        builder.def_var(acc_pair.0, vzero);
                        builder.def_var(acc_pair.1, vzero);
                    }
                    builder.ins().jump(clif_blocks[mir_fn.entry.0], &[]);
                }

                // Lower each basic block
                for (b_idx, mir_block) in mir_fn.blocks.iter().enumerate() {
                    let clif_bb = clif_blocks[b_idx];
                    builder.switch_to_block(clif_bb);

                    let should_safepoint = loop_headers.contains(&b_idx);

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
                        emit_safepoint_check(
                            &mut builder,
                            safepoint_fn,
                            init_deopt_fn,
                            record_deopt_local_fn,
                            b_idx,
                            locals_to_record,
                            &local_vars,
                            &stack_slots,
                            &mir_fn.locals,
                            emergency_exit_bb,
                            pending_vec_reduction,
                        );
                    }

                    // Native Cranelift hardware SIMD lowering for recognized vectorized loop header
                    if let Some(&vl_idx) = vec_hdr_map.get(&b_idx) {
                        let vl = &vec_loops[vl_idx];
                        if let Some(amt) = vl.gas_check {
                            let amt_val = builder.ins().iconst(types::I64, amt as i64);
                            let call_inst = builder.ins().call(gas_fn, &[amt_val]);
                            let rem = builder.inst_results(call_inst)[0];
                            let is_neg = builder.ins().icmp_imm_s(IntCC::SignedLessThan, rem, 0);
                            let cont_bb = builder.create_block();
                            builder.ins().brif(is_neg, emergency_exit_bb, &[], cont_bb, &[]);
                            builder.switch_to_block(cont_bb);
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
                            builder.ins().jump(clif_blocks[vl.tail_block.0], &[]);
                        } else {
                            builder.ins().brif(cond, clif_blocks[vl.body_block.0], &[], clif_blocks[vl.tail_block.0], &[]);
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
                        continue;
                    }

                    for stmt in &mir_block.statements {
                        match stmt {
                            Statement::Assign(Place::Local(dst), rvalue, _) => {
                                let val = lower_rvalue(
                                    rvalue,
                                    &mut builder,
                                    &local_vars,
                                    &stack_slots,
                                    &mir_fn.locals,
                                    &self.struct_layouts,
                                    alloc_array_typed_fn,
                                    alloc_tuple_typed_fn,
                                    alloc_struct_named_fn,
                                    alloc_cflat_struct_fn,
                                    len_fn,
                                    floordiv_fn,
                                    pow_fn,
                                    div_fn,
                                    rem_fn,
                                    floordiv_float_fn,
                                    pow_float_fn,
                                    rem_float_fn,
                                    string_concat_fn,
                                    string_eq_fn,
                                    string_len_fn,
                                    status_fn,
                                    bc_ctx,
                                )?;
                                let dst_decl = mir_fn.locals.get(dst.0).ok_or_else(|| {
                                    MirCraneliftBackendError::CodegenError(format!(
                                        "Destination local {dst:?} out of bounds"
                                    ))
                                })?;
                                let target_ty = mir_to_clif_type(&dst_decl.ty)?;
                                let val = cast_clif_type(&mut builder, val, target_ty);
                                store_local_val(*dst, val, &mut builder, &local_vars, &stack_slots)?;
                            }
                            Statement::Assign(lhs_place @ Place::Index(..), rvalue, _) => {
                                let val = lower_rvalue(
                                    rvalue,
                                    &mut builder,
                                    &local_vars,
                                    &stack_slots,
                                    &mir_fn.locals,
                                    &self.struct_layouts,
                                    alloc_array_typed_fn,
                                    alloc_tuple_typed_fn,
                                    alloc_struct_named_fn,
                                    alloc_cflat_struct_fn,
                                    len_fn,
                                    floordiv_fn,
                                    pow_fn,
                                    div_fn,
                                    rem_fn,
                                    floordiv_float_fn,
                                    pow_float_fn,
                                    rem_float_fn,
                                    string_concat_fn,
                                    string_eq_fn,
                                    string_len_fn,
                                    status_fn,
                                    bc_ctx,
                                )?;
                                let elem_addr = lower_place_addr(
                                    lhs_place,
                                    &mut builder,
                                    &local_vars,
                                    &stack_slots,
                                    &mir_fn.locals,
                                    &self.struct_layouts,
                                    bc_ctx,
                                )?;
                                let val_ty = builder.func.dfg.value_type(val);
                                let val_to_store = if val_ty == types::I8 {
                                    builder.ins().uextend(types::I64, val)
                                } else {
                                    val
                                };
                                builder.ins().store(MemFlagsData::trusted(), val_to_store, elem_addr, 0);
                            }
                            Statement::Assign(lhs_place @ Place::Field(..), rvalue, _) => {
                                let val = lower_rvalue(
                                    rvalue,
                                    &mut builder,
                                    &local_vars,
                                    &stack_slots,
                                    &mir_fn.locals,
                                    &self.struct_layouts,
                                    alloc_array_typed_fn,
                                    alloc_tuple_typed_fn,
                                    alloc_struct_named_fn,
                                    alloc_cflat_struct_fn,
                                    len_fn,
                                    floordiv_fn,
                                    pow_fn,
                                    div_fn,
                                    rem_fn,
                                    floordiv_float_fn,
                                    pow_float_fn,
                                    rem_float_fn,
                                    string_concat_fn,
                                    string_eq_fn,
                                    string_len_fn,
                                    status_fn,
                                    bc_ctx,
                                )?;
                                let field_addr = lower_place_addr(
                                    lhs_place,
                                    &mut builder,
                                    &local_vars,
                                    &stack_slots,
                                    &mir_fn.locals,
                                    &self.struct_layouts,
                                    bc_ctx,
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
                                let amt_val = builder.ins().iconst(types::I64, *amt as i64);
                                let call_inst = builder.ins().call(gas_fn, &[amt_val]);
                                let rem = builder.inst_results(call_inst)[0];
                                let is_neg = builder.ins().icmp_imm_s(IntCC::SignedLessThan, rem, 0);
                                let cont_bb = builder.create_block();
                                builder.ins().brif(is_neg, emergency_exit_bb, &[], cont_bb, &[]);
                                builder.switch_to_block(cont_bb);
                            }
                            _ => {}
                        }
                    }

                    // If this block is the pre-entry to a vectorized loop, insert runtime scalar-forced
                    // and shifted-alias guards before entering the SIMD loop header
                    if let Some(&vl_idx) = vec_pre_map.get(&b_idx) {
                        let vl = &vec_loops[vl_idx];
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

                            builder.ins().brif(any_shifted_ov, scalar_fallback_bb, &[], enter_vec_bb, &[]);
                        } else {
                            builder.ins().jump(enter_vec_bb, &[]);
                        }

                        builder.switch_to_block(scalar_fallback_bb);
                        builder.ins().call(simd_scalar_fn, &[]);
                        builder.ins().jump(clif_blocks[vl.tail_block.0], &[]);

                        builder.switch_to_block(enter_vec_bb);
                        builder.ins().call(simd_vector_fn, &[]);
                        if let Some((vacc0, vacc1)) = vec_acc_vars[vl_idx] {
                            let zero_s = builder.ins().iconst(types::I64, 0);
                            let vzero = builder.ins().splat(types::I64X2, zero_s);
                            builder.def_var(vacc0, vzero);
                            builder.def_var(vacc1, vzero);
                        }
                        builder.ins().jump(clif_blocks[vl.hdr_block.0], &[]);
                        continue;
                    }

                    if let Some(term) = &mir_block.terminator {
                        match term {
                            Terminator::Goto(target, _) => {
                                let target_bb = clif_blocks.get(target.0).copied().ok_or_else(|| {
                                    MirCraneliftBackendError::CodegenError(format!(
                                        "Goto target block {target:?} out of bounds"
                                    ))
                                })?;
                                builder.ins().jump(target_bb, &[]);
                            }
                            Terminator::Cond { cond, then_block, else_block, .. } => {
                                let c_val = lower_operand(cond, &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                let then_bb = clif_blocks.get(then_block.0).copied().ok_or_else(|| {
                                    MirCraneliftBackendError::CodegenError(format!(
                                        "Cond then block {then_block:?} out of bounds"
                                    ))
                                })?;
                                let else_bb = clif_blocks.get(else_block.0).copied().ok_or_else(|| {
                                    MirCraneliftBackendError::CodegenError(format!(
                                        "Cond else block {else_block:?} out of bounds"
                                    ))
                                })?;
                                builder.ins().brif(c_val, then_bb, &[], else_bb, &[]);
                            }
                            Terminator::Switch { discr, cases, default, .. } => {
                                let d_val = lower_operand(discr, &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                for (val, target_bb) in cases {
                                    let is_eq = builder.ins().icmp_imm_s(IntCC::Equal, d_val, *val);
                                    let next_check = builder.create_block();
                                    let t_bb = clif_blocks.get(target_bb.0).copied().ok_or_else(|| {
                                        MirCraneliftBackendError::CodegenError(format!(
                                            "Switch target block {target_bb:?} out of bounds"
                                        ))
                                    })?;
                                    builder.ins().brif(is_eq, t_bb, &[], next_check, &[]);
                                    builder.switch_to_block(next_check);
                                }
                                let def_bb = clif_blocks.get(default.0).copied().ok_or_else(|| {
                                    MirCraneliftBackendError::CodegenError(format!(
                                        "Switch default block {default:?} out of bounds"
                                    ))
                                })?;
                                builder.ins().jump(def_bb, &[]);
                            }
                            Terminator::Call { callee, args, destination, success, .. } => {
                                if callee == "print" || callee == "println" {
                                    let arg_val = if let Some(first) = args.first() {
                                        lower_operand(first, &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?
                                    } else {
                                        builder.ins().iconst(types::I64, 0)
                                    };
                                    builder.ins().call(print_fn, &[arg_val]);
                                    if let Place::Local(loc) = destination {
                                        let zero = builder.ins().iconst(types::I64, 0);
                                        store_local_val(*loc, zero, &mut builder, &local_vars, &stack_slots)?;
                                    }
                                    builder.ins().jump(clif_blocks[success.0], &[]);
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
                                    builder.ins().jump(clif_blocks[success.0], &[]);
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
                                    let call_inst = builder.ins().call(ic_dispatch_fn, &[site_id_val, tag_val, def_val]);
                                    let res_val = builder.inst_results(call_inst)[0];
                                    if let Place::Local(loc) = destination {
                                        let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                        let res_cast = cast_clif_type(&mut builder, res_val, target_ty);
                                        store_local_val(*loc, res_cast, &mut builder, &local_vars, &stack_slots)?;
                                    }
                                    builder.ins().jump(clif_blocks[success.0], &[]);
                                } else if callee == "simd.add" || callee == "simd_add" || callee == "vector.add" {
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
                                    let next_mir_bb = clif_blocks[success.0];

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
                                    let next_mir_bb = clif_blocks[success.0];

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
                                    let next_mir_bb = clif_blocks[success.0];

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
                                    let next_mir_bb = clif_blocks[success.0];

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
                                    builder.ins().jump(next_mir_bb, &[]);
                                } else {
                                    let mut is_native_mir_call = false;
                                    let (callee_func_id, param_count) = match callee.as_str() {
                                        "math.abs" | "abs" => (abs_callee, 1),
                                        "math.min" | "min" => (min_callee, 2),
                                        "math.max" | "max" => (max_callee, 2),
                                        "math.clamp" | "clamp" => (clamp_callee, 3),
                                        "table.len" | "array.len" => {
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
                                            (len_callee, 1)
                                        }
                                        "map.len" => (map_len_callee, 1),
                                        "map.get" => (map_get_callee, 2),
                                        "map.set" => (map_set_callee, 3),
                                        "map.new" => (alloc_map_callee, 0),
                                        "enum.tag" => (enum_tag_callee, 1),
                                        "enum.payload" => (enum_payload_callee, 1),
                                        "enum.new" | "enum.alloc" => (alloc_enum_callee, 2),
                                        _ => {
                                            let Some(&fid) = module_funcs.get(callee) else {
                                                return Err(MirCraneliftBackendError::UnlowerableValue(format!(
                                                    "External or unknown callee '{callee}' is not supported by Cranelift backend"
                                                )));
                                            };
                                            is_native_mir_call = true;
                                            (fid, args.len())
                                        }
                                    };
                                    let callee_ref = self.module.declare_func_in_func(callee_func_id, &mut builder.func);
                                    let mut arg_vals = Vec::with_capacity(param_count);
                                    for a in args {
                                        arg_vals.push(lower_operand(a, &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?);
                                    }
                                    while arg_vals.len() < param_count {
                                        arg_vals.push(builder.ins().iconst(types::I64, 0));
                                    }
                                    if is_native_mir_call {
                                        builder.ins().call(enter_native_callee_fn, &[]);
                                    }
                                    let call_inst = builder.ins().call(callee_ref, &arg_vals);
                                    if is_native_mir_call {
                                        builder.ins().call(exit_native_callee_fn, &[]);
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
                                                let val_to_store = if val_ty == types::I8 {
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
                                    let stat_call = builder.ins().call(status_fn, &[]);
                                    let stat_val = builder.inst_results(stat_call)[0];
                                    let is_err = builder.ins().icmp_imm_s(IntCC::NotEqual, stat_val, 0);
                                    let call_cont_bb = builder.create_block();
                                    builder.ins().brif(is_err, emergency_exit_bb, &[], call_cont_bb, &[]);
                                    builder.switch_to_block(call_cont_bb);

                                    let succ_bb = clif_blocks.get(success.0).copied().ok_or_else(|| {
                                        MirCraneliftBackendError::CodegenError(format!(
                                            "Call success block {success:?} out of bounds"
                                        ))
                                    })?;
                                    builder.ins().jump(succ_bb, &[]);
                                }
                            }
                            Terminator::Return(opt_op, _) => {
                                if ret_clif_ty != types::I8 || mir_fn.signature.return_type != MirType::Unit {
                                    let ret_val = if let Some(op) = opt_op {
                                        let v = lower_operand(op, &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts, bc_ctx)?;
                                        cast_clif_type(&mut builder, v, ret_clif_ty)
                                    } else if ret_clif_ty == types::F64 {
                                        builder.ins().f64const(0.0)
                                    } else {
                                        builder.ins().iconst(ret_clif_ty, 0)
                                    };
                                    builder.ins().return_(&[ret_val]);
                                } else {
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
                builder.finalize(self.module.target_config());
            }

            self.module
                .define_function(func_id, &mut self.ctx)
                .map_err(|e| {
                    eprintln!("Verifier error in define_function: {:?}", e);
                    MirCraneliftBackendError::CodegenError(format!("{e:?}"))
                })?;
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

    pub(crate) fn mir_type_contains_unknown(ty: &MirType) -> bool {
        match ty {
            MirType::Unknown => true,
            MirType::Array(inner) => mir_type_contains_unknown(inner),
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
                                        let out_ty = if matches!(
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
                             "table.len" | "array.len" | "map.len" | "string.len" | "enum.tag" | "enum.payload" => {
                                 Some(MirType::Int)
                             }
                             "map.get" => {
                                 Some(MirType::Enum("Option".to_string()))
                             }
                             "map.set" => {
                                 Some(MirType::Unit)
                             }
                             "map.new" => {
                                 Some(MirType::Map(Box::new(MirType::Unknown), Box::new(MirType::Unknown)))
                             }
                             "enum.new" | "enum.alloc" => {
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
                             _ => None,
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
            MirType::Unknown => Err(MirCraneliftBackendError::UnsupportedType(MirType::Unknown)),
            MirType::Function { .. } => Err(MirCraneliftBackendError::UnsupportedType(ty.clone())),
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
    struct BoundsCheckCtx {
        status_set_fn: FuncRef,
        emergency_exit_bb: Block,
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
            builder.ins().stack_store(types::I64, val, slot, 0);
        }
        Ok(())
    }

    fn emit_oob_branch_if_bad(
        is_bad: ClifValue,
        builder: &mut FunctionBuilder,
        bc_ctx: BoundsCheckCtx,
    ) {
        let err_bb = builder.create_block();
        let ok_bb = builder.create_block();
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
        bc_ctx: BoundsCheckCtx,
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

                if matches!(base_ty, MirType::Array(_) | MirType::Tuple(_)) {
                    let is_null = builder.ins().icmp_imm_s(IntCC::Equal, base_ptr, 0);
                    emit_oob_branch_if_bad(is_null, builder, bc_ctx);
                    let arr_len = builder.ins().load(types::I64, MemFlagsData::trusted(), base_ptr, 0);
                    let idx_neg = builder.ins().icmp_imm_s(IntCC::SignedLessThan, idx_val, 0);
                    let idx_ge = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, idx_val, arr_len);
                    let is_oob = builder.ins().bor(idx_neg, idx_ge);
                    emit_oob_branch_if_bad(is_oob, builder, bc_ctx);
                }

                let eight = builder.ins().iconst(types::I64, 8);
                let byte_idx = builder.ins().imul(idx_val, eight);
                let sixteen = builder.ins().iconst(types::I64, 16);
                let offset = builder.ins().iadd(byte_idx, sixteen);
                Ok(builder.ins().iadd(base_ptr, offset))
            }
            Place::Field(base, field_idx) => {
                let _ = try_place_type(place, locals, struct_layouts)?;
                let base_ptr = lower_place_addr(
                    base,
                    builder,
                    local_vars,
                    stack_slots,
                    locals,
                    struct_layouts,
                    bc_ctx,
                )?;
                let base_ty = try_place_type(base, locals, struct_layouts)?;
                if matches!(base_ty, MirType::Tuple(_) | MirType::Struct(_)) {
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
        bc_ctx: BoundsCheckCtx,
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
        bc_ctx: BoundsCheckCtx,
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
        emergency_exit_bb: Block,
    ) {
        let stat_call = builder.ins().call(status_fn, &[]);
        let stat_val = builder.inst_results(stat_call)[0];
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
        bc_ctx: BoundsCheckCtx,
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

    fn lower_rvalue(
        rval: &Rvalue,
        builder: &mut FunctionBuilder,
        local_vars: &[Variable],
        stack_slots: &hashbrown::HashMap<usize, StackSlot>,
        locals: &[LocalDecl],
        struct_layouts: &hashbrown::HashMap<String, MirStructLayout>,
        alloc_array_typed_fn: FuncRef,
        alloc_tuple_typed_fn: FuncRef,
        alloc_struct_named_fn: FuncRef,
        alloc_cflat_struct_fn: FuncRef,
        len_fn: FuncRef,
        floordiv_fn: FuncRef,
        pow_fn: FuncRef,
        div_fn: FuncRef,
        rem_fn: FuncRef,
        floordiv_float_fn: FuncRef,
        pow_float_fn: FuncRef,
        rem_float_fn: FuncRef,
        string_concat_fn: FuncRef,
        string_eq_fn: FuncRef,
        string_len_fn: FuncRef,
        status_fn: FuncRef,
        bc_ctx: BoundsCheckCtx,
    ) -> Result<ClifValue, MirCraneliftBackendError> {
        let emergency_exit_bb = bc_ctx.emergency_exit_bb;
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
                            emit_status_check(builder, status_fn, emergency_exit_bb);
                            res
                        }
                        MirBinaryOp::FloorDiv => {
                            let call = builder.ins().call(floordiv_float_fn, &[l, r]);
                            let res = builder.inst_results(call)[0];
                            emit_status_check(builder, status_fn, emergency_exit_bb);
                            res
                        }
                        MirBinaryOp::Pow => {
                            let call = builder.ins().call(pow_float_fn, &[l, r]);
                            let res = builder.inst_results(call)[0];
                            emit_status_check(builder, status_fn, emergency_exit_bb);
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
                            let call = builder.ins().call(string_concat_fn, &[l, r]);
                            let res = builder.inst_results(call)[0];
                            emit_status_check(builder, status_fn, emergency_exit_bb);
                            return Ok(res);
                        }
                        MirBinaryOp::Eq => {
                            let call = builder.ins().call(string_eq_fn, &[l, r]);
                            let res = builder.inst_results(call)[0];
                            return Ok(res);
                        }
                        MirBinaryOp::NotEq => {
                            let call = builder.ins().call(string_eq_fn, &[l, r]);
                            let eq_res = builder.inst_results(call)[0];
                            let one = builder.ins().iconst(types::I8, 1);
                            return Ok(builder.ins().bxor(eq_res, one));
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
                        let call = builder.ins().call(div_fn, &[l, r]);
                        let res = builder.inst_results(call)[0];
                        emit_status_check(builder, status_fn, emergency_exit_bb);
                        res
                    }
                    MirBinaryOp::Mod => {
                        let call = builder.ins().call(rem_fn, &[l, r]);
                        let res = builder.inst_results(call)[0];
                        emit_status_check(builder, status_fn, emergency_exit_bb);
                        res
                    }
                    MirBinaryOp::FloorDiv => {
                        let call = builder.ins().call(floordiv_fn, &[l, r]);
                        let res = builder.inst_results(call)[0];
                        emit_status_check(builder, status_fn, emergency_exit_bb);
                        res
                    }
                    MirBinaryOp::Pow => {
                        let call = builder.ins().call(pow_fn, &[l, r]);
                        let res = builder.inst_results(call)[0];
                        emit_status_check(builder, status_fn, emergency_exit_bb);
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
                    MirBinaryOp::Eq => builder.ins().icmp(IntCC::Equal, l, r),
                    MirBinaryOp::NotEq => builder.ins().icmp(IntCC::NotEqual, l, r),
                    MirBinaryOp::Lt => builder.ins().icmp(IntCC::SignedLessThan, l, r),
                    MirBinaryOp::LtEq => builder.ins().icmp(IntCC::SignedLessThanOrEqual, l, r),
                    MirBinaryOp::Gt => builder.ins().icmp(IntCC::SignedGreaterThan, l, r),
                    MirBinaryOp::GtEq => builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, l, r),
                    MirBinaryOp::And => builder.ins().band(l, r),
                    MirBinaryOp::Or => builder.ins().bor(l, r),
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
                            let call = builder.ins().call(string_len_fn, &[v]);
                            return Ok(builder.inst_results(call)[0]);
                        }
                        MirType::Array(_) => {
                            let call = builder.ins().call(len_fn, &[v]);
                            let res = builder.inst_results(call)[0];
                            let is_err = builder.ins().icmp_imm_s(IntCC::SignedLessThan, res, 0);
                            let len_ok_bb = builder.create_block();
                            builder.ins().brif(is_err, emergency_exit_bb, &[], len_ok_bb, &[]);
                            builder.switch_to_block(len_ok_bb);
                            return Ok(res);
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
                    MirUnaryOp::Not => builder.ins().bnot(v),
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
                emit_status_check(builder, status_fn, emergency_exit_bb);
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
                emit_status_check(builder, status_fn, emergency_exit_bb);
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
                emit_status_check(builder, status_fn, emergency_exit_bb);
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
                let Some(elem_tag) = mir_array_elem_tag(elem_ty) else {
                    return Err(MirCraneliftBackendError::UnsupportedType((**elem_ty).clone()));
                };
                for elem in elements {
                    let actual_ty = try_operand_type(elem, locals, struct_layouts)?;
                    if actual_ty != **elem_ty {
                        return Err(MirCraneliftBackendError::CodegenError(format!(
                            "Array element type mismatch: expected '{elem_ty}', got '{actual_ty}'"
                        )));
                    }
                }
                let len_val = builder.ins().iconst(types::I64, elements.len() as i64);
                let tag_val = builder.ins().iconst(types::I64, elem_tag);
                let call_inst = builder.ins().call(alloc_array_typed_fn, &[len_val, tag_val]);
                let array_ptr = builder.inst_results(call_inst)[0];
                emit_status_check(builder, status_fn, emergency_exit_bb);
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
                Ok(array_ptr)
            }
            _ => Err(MirCraneliftBackendError::UnsupportedConstant(MirConstant::Unit)),
        }
    }
}
