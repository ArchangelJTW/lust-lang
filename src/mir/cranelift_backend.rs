#[cfg(feature = "std")]
pub use jit_engine::{
    clear_registered_struct_layouts, find_loop_backedges, find_loop_headers,
    get_registered_struct_layout, get_registered_struct_layout_by_name, ic_dispatch,
    ic_invalidate, ic_invalidate_all, ic_probe, ic_register_site, ic_reset, ic_site_stats,
    ic_update, is_loop_backedge, lust_mir_abs_int, lust_mir_alloc_array, lust_mir_alloc_enum,
    lust_mir_alloc_map, lust_mir_array_get, lust_mir_array_len, lust_mir_array_set,
    lust_mir_charge_gas, lust_mir_clamp_int, lust_mir_div_int, lust_mir_enum_payload,
    lust_mir_enum_tag, lust_mir_floordiv_float, lust_mir_floordiv_int, lust_mir_get_gas,
    lust_mir_get_memory_used, lust_mir_get_status, lust_mir_ic_dispatch, lust_mir_ic_miss,
    lust_mir_ic_probe, lust_mir_init_deopt, lust_mir_map_get, lust_mir_map_len,
    lust_mir_map_set, lust_mir_max_int, lust_mir_min_int, lust_mir_pow_float,
    lust_mir_pow_int, lust_mir_print_int, lust_mir_record_deopt, lust_mir_record_deopt_local,
    lust_mir_record_native_entry, lust_mir_rem_float, lust_mir_rem_int, lust_mir_safepoint,
    lust_mir_set_gas, lust_mir_set_max_memory, lust_mir_set_memory_limit, lust_mir_set_status,
    lust_mir_string_concat, lust_mir_string_eq, lust_mir_string_len,
    lust_mir_struct_decode_binary, lust_mir_struct_decode_contiguous, publish_deopt_payload,
    register_struct_layout, reset_native_instrumentation_counters, safepoint_enter_mask,
    safepoint_exit_mask, safepoint_register_finalizer, safepoint_request_deopt,
    safepoint_request_interrupt, safepoint_reset, safepoint_stats, take_deopt_payload,
    take_last_schema_decode_error, CraneliftJitCompiler, DeoptPayload, IcEntry, IcSiteStats,
    IcStateKind, InlineCacheSite, MirCraneliftBackendError, OsrPromotionTracker,
    SafepointRuntimeState, SafepointStatsSnapshot, SchemaDecodeError,
    CRANELIFT_NATIVE_ENTRY_COUNT, DIRECT_DECODE_COUNT, DYNAMIC_LOOKUP_COUNT,
    MIR_STATUS_DEOPT, MIR_STATUS_ERR_DIV_BY_ZERO, MIR_STATUS_ERR_INTERRUPTED,
    MIR_STATUS_ERR_NEG_EXPONENT, MIR_STATUS_ERR_OOM, MIR_STATUS_ERR_OUT_OF_GAS,
    MIR_STATUS_OK,
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
        types, AbiParam, Block, FuncRef, InstBuilder, MemFlagsData, StackSlot, StackSlotData,
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
    }

    pub const MIR_STATUS_ERR_INTERRUPTED: i64 = crate::effect_abi::LUST_STATUS_ERR_INTERRUPTED as i64;
    pub const MIR_STATUS_DEOPT: i64 = -100;

    #[derive(Default)]
    pub struct SafepointRuntimeState {
        pub mask_depth: usize,
        pub pending_interrupt: Option<crate::effect_abi::InterruptReason>,
        pub safepoints_checked: u64,
        pub safepoints_masked_deferred: u64,
        pub safepoints_delivered: u64,
        pub deopt_requested: bool,
        pub cleanups_run: usize,
        pub finalizers: Vec<alloc::boxed::Box<dyn FnOnce() + Send + 'static>>,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct DeoptPayload {
        pub target_pc: usize,
        pub live_locals: Vec<(usize, i64)>,
    }

    impl DeoptPayload {
        pub fn new(target_pc: usize, live_locals: Vec<(usize, i64)>) -> Self {
            Self {
                target_pc,
                live_locals,
            }
        }

        pub fn to_vm_values_with_types(
            &self,
            types: &[(usize, MirType)],
        ) -> Vec<(usize, crate::bytecode::Value)> {
            let mut res = Vec::with_capacity(self.live_locals.len());
            for &(loc_idx, raw_val) in &self.live_locals {
                let mir_ty = types
                    .iter()
                    .find(|(i, _)| *i == loc_idx)
                    .map(|(_, ty)| ty)
                    .unwrap_or(&MirType::Int);
                let val = match mir_ty {
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
                        handle_val.unwrap_or(crate::bytecode::Value::Int(raw_val))
                    }
                    _ => crate::bytecode::Value::Int(raw_val),
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
        DEOPT_STATE.with(|d| {
            *d.borrow_mut() = Some(DeoptPayload {
                target_pc,
                live_locals,
            });
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

    thread_local! {
        static IC_REGISTRY: core::cell::RefCell<hashbrown::HashMap<u32, InlineCacheSite>> = core::cell::RefCell::new(hashbrown::HashMap::new());
        static SAFEPOINT_STATE: core::cell::RefCell<SafepointRuntimeState> = core::cell::RefCell::new(SafepointRuntimeState::default());
        static DEOPT_STATE: core::cell::RefCell<Option<DeoptPayload>> = const { core::cell::RefCell::new(None) };
        static STRUCT_LAYOUT_REGISTRY: core::cell::RefCell<hashbrown::HashMap<u64, MirStructLayout>> = core::cell::RefCell::new(hashbrown::HashMap::new());
        static STRUCT_NAME_MAP: core::cell::RefCell<hashbrown::HashMap<String, u64>> = core::cell::RefCell::new(hashbrown::HashMap::new());
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
    pub static DIRECT_DECODE_COUNT: core::sync::atomic::AtomicUsize =
        core::sync::atomic::AtomicUsize::new(0);
    pub static DYNAMIC_LOOKUP_COUNT: core::sync::atomic::AtomicUsize =
        core::sync::atomic::AtomicUsize::new(0);

    pub fn reset_native_instrumentation_counters() {
        CRANELIFT_NATIVE_ENTRY_COUNT.store(0, core::sync::atomic::Ordering::SeqCst);
        DIRECT_DECODE_COUNT.store(0, core::sync::atomic::Ordering::SeqCst);
        DYNAMIC_LOOKUP_COUNT.store(0, core::sync::atomic::Ordering::SeqCst);
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn lust_mir_record_native_entry() {
        CRANELIFT_NATIVE_ENTRY_COUNT.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
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
            ctx.allocated_buffers
                .lock()
                .unwrap()
                .push((raw_box as *mut i64, count));
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
                    ctx.allocated_buffers.lock().unwrap().push((raw_box as *mut i64, count));
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

    #[inline(always)]
    fn is_effect_or_handler_callee(callee: &str) -> bool {
        callee == "perform"
            || callee.starts_with("perform.")
            || callee.starts_with("effect.")
            || callee == "resume"
            || callee.starts_with("resume.")
            || callee == "handle"
            || callee.starts_with("handle.")
            || callee.starts_with("handler.")
            || callee.contains("effect")
            || callee.contains("handler")
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn lust_mir_safepoint() -> i64 {
        SAFEPOINT_STATE.with(|state_cell| {
            let mut st = state_cell.borrow_mut();
            st.safepoints_checked += 1;
            if st.deopt_requested {
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

    #[unsafe(no_mangle)]
    pub extern "C" fn lust_mir_init_deopt(target_pc: i64) {
        DEOPT_STATE.with(|d| {
            *d.borrow_mut() = Some(DeoptPayload {
                target_pc: target_pc as usize,
                live_locals: Vec::new(),
            });
        });
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn lust_mir_record_deopt_local(_target_pc: i64, local_idx: i64, val: i64) {
        DEOPT_STATE.with(|d| {
            let mut opt = d.borrow_mut();
            if let Some(payload) = opt.as_mut() {
                payload.live_locals.push((local_idx as usize, val));
            }
        });
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn lust_mir_record_deopt(target_pc: i64, num_locals: i64, locals_ptr: *const i64) -> i64 {
        let mut locals = Vec::new();
        if !locals_ptr.is_null() && num_locals > 0 {
            for i in 0..num_locals as usize {
                let val = unsafe { *locals_ptr.add(i) };
                locals.push((i, val));
            }
        }
        DEOPT_STATE.with(|d| {
            *d.borrow_mut() = Some(DeoptPayload {
                target_pc: target_pc as usize,
                live_locals: locals,
            });
        });
        MIR_STATUS_DEOPT
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
        DEOPT_STATE.with(|d| {
            *d.borrow_mut() = None;
        });
    }

    pub fn take_deopt_payload() -> Option<DeoptPayload> {
        DEOPT_STATE.with(|d| d.borrow_mut().take())
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
            builder.symbol("lust_mir_array_len", lust_mir_array_len as *const u8);
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
            builder.symbol("lust_mir_record_deopt_local", lust_mir_record_deopt_local as *const u8);
            builder.symbol("lust_mir_record_deopt", lust_mir_record_deopt as *const u8);

            // J-IC helpers
            builder.symbol("lust_mir_ic_probe", lust_mir_ic_probe as *const u8);
            builder.symbol("lust_mir_ic_miss", lust_mir_ic_miss as *const u8);
            builder.symbol("lust_mir_ic_dispatch", lust_mir_ic_dispatch as *const u8);

            // Gate-SCHEMA-09/13 struct decoder & native entry instrumentation helpers
            builder.symbol("lust_mir_struct_decode_contiguous", lust_mir_struct_decode_contiguous as *const u8);
            builder.symbol("lust_mir_struct_decode_binary", lust_mir_struct_decode_binary as *const u8);
            builder.symbol("lust_mir_record_native_entry", lust_mir_record_native_entry as *const u8);

            let module = JITModule::new(builder);

            Ok(Self {
                builder_ctx: FunctionBuilderContext::new(),
                ctx: module.make_context(),
                module,
                struct_layouts: hashbrown::HashMap::new(),
                compiled_code_sizes: hashbrown::HashMap::new(),
                compiled_code_ranges: hashbrown::HashMap::new(),
            })
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
            let mut refined_functions = Vec::with_capacity(module.functions.len());
            for f in &module.functions {
                refined_functions.push(refine_mir_function(f));
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
            let refined = refine_mir_function(mir_fn);
            let osr_name = format!("{}_osr_{}", func_name, target_block.0);

            // 1. Declare all module functions so OSR entry can call them
            let mut module_funcs = hashbrown::HashMap::new();
            for f in &module.functions {
                let mut sig = self.module.make_signature();
                for param_id in &f.params {
                    let p_ty = &f.locals[param_id.0].ty;
                    sig.params.push(AbiParam::new(mir_to_clif_type(p_ty)?));
                }
                let ret_clif_ty = mir_to_clif_type(&f.signature.return_type)?;
                if ret_clif_ty != types::I8 || f.signature.return_type != MirType::Unit {
                    sig.returns.push(AbiParam::new(ret_clif_ty));
                }
                let func_id = self
                    .module
                    .declare_function(&f.signature.name, Linkage::Export, &sig)
                    .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;
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

            // 3. Compile other module functions first
            for other_fn in &module.functions {
                if other_fn.signature.name != func_name {
                    let fid = module_funcs[&other_fn.signature.name];
                    let refined_other = refine_mir_function(other_fn);
                    self.compile_function_internal(&refined_other, fid, &module_funcs)?;
                }
            }

            // 4. Compile OSR entry
            self.compile_function_with_osr(
                &refined,
                osr_func_id,
                &module_funcs,
                Some((target_block, live_locals)),
            )?;

            // 5. Finalize definitions
            self.module
                .finalize_definitions()
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            Ok(self.module.get_finalized_function(osr_func_id))
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
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;
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
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;
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
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;
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
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;
            self.module.clear_context(&mut self.ctx);
            self.module
                .finalize_definitions()
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;
            Ok(self.module.get_finalized_function(func_id))
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

            let alloc_callee = self
                .module
                .declare_function("lust_mir_alloc_array", Linkage::Import, &sig1)
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
            let native_entry_callee = self
                .module
                .declare_function("lust_mir_record_native_entry", Linkage::Import, &sig_void0)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            {
                let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);
                let alloc_fn = self.module.declare_func_in_func(alloc_callee, &mut builder.func);
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
                let record_deopt_local_fn = self.module.declare_func_in_func(rec_deopt_callee, &mut builder.func);
                let ic_dispatch_fn = self.module.declare_func_in_func(ic_dispatch_callee, &mut builder.func);
                let struct_decode_fn = self.module.declare_func_in_func(struct_decode_callee, &mut builder.func);
                let native_entry_fn = self.module.declare_func_in_func(native_entry_callee, &mut builder.func);

                let emergency_exit_bb = builder.create_block();

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

                let all_locals: Vec<LocalId> = (0..mir_fn.locals.len()).map(LocalId).collect();
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
                    builder.ins().jump(clif_blocks[mir_fn.entry.0], &[]);
                }

                // Lower each basic block
                for (b_idx, mir_block) in mir_fn.blocks.iter().enumerate() {
                    let clif_bb = clif_blocks[b_idx];
                    builder.switch_to_block(clif_bb);

                    let should_safepoint = loop_headers.contains(&b_idx);

                    if should_safepoint {
                        let locals_to_record = if let Some((_, live_locals)) = osr_info {
                            live_locals
                        } else {
                            &all_locals
                        };
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
                        );
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
                                    alloc_fn,
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
                                    emergency_exit_bb,
                                )?;
                                let target_ty = mir_to_clif_type(&mir_fn.locals[dst.0].ty)?;
                                let val = cast_clif_type(&mut builder, val, target_ty);
                                builder.def_var(local_vars[dst.0], val);
                                if let Some(&slot) = stack_slots.get(&dst.0) {
                                    builder.ins().stack_store(types::I64, val, slot, 0);
                                }
                            }
                            Statement::Assign(Place::Index(base, idx), rvalue, _) => {
                                let val = lower_rvalue(
                                    rvalue,
                                    &mut builder,
                                    &local_vars,
                                    &stack_slots,
                                    &mir_fn.locals,
                                    &self.struct_layouts,
                                    alloc_fn,
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
                                    emergency_exit_bb,
                                )?;
                                let base_ptr = lower_place_addr(base, &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts);
                                let idx_val = if let Some(&slot) = stack_slots.get(&idx.0) {
                                    builder.ins().stack_load(types::I64, types::I64, slot, 0)
                                } else {
                                    builder.use_var(local_vars[idx.0])
                                };
                                let eight = builder.ins().iconst(types::I64, 8);
                                let byte_idx = builder.ins().imul(idx_val, eight);
                                let sixteen = builder.ins().iconst(types::I64, 16);
                                let offset = builder.ins().iadd(byte_idx, sixteen);
                                let elem_addr = builder.ins().iadd(base_ptr, offset);
                                let val_ty = builder.func.dfg.value_type(val);
                                let val_to_store = if val_ty == types::I8 {
                                    builder.ins().uextend(types::I64, val)
                                } else {
                                    val
                                };
                                builder.ins().store(MemFlagsData::trusted(), val_to_store, elem_addr, 0);
                            }
                            Statement::Assign(Place::Field(base, field_idx), rvalue, _) => {
                                let val = lower_rvalue(
                                    rvalue,
                                    &mut builder,
                                    &local_vars,
                                    &stack_slots,
                                    &mir_fn.locals,
                                    &self.struct_layouts,
                                    alloc_fn,
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
                                    emergency_exit_bb,
                                )?;
                                let base_ptr = lower_place_addr(base, &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts);
                                let base_ty = place_type(base, &mir_fn.locals, &self.struct_layouts);
                                let offset_bytes = match &base_ty {
                                    MirType::Struct(name) => {
                                        if let Some(layout) = self.struct_layouts.get(name) {
                                            if let Some(f) = layout.fields.get(*field_idx) {
                                                f.byte_offset as i32
                                            } else {
                                                16 + (*field_idx as i32) * 8
                                            }
                                        } else {
                                            16 + (*field_idx as i32) * 8
                                        }
                                    }
                                    _ => 16 + (*field_idx as i32) * 8,
                                };
                                let val_ty = builder.func.dfg.value_type(val);
                                let val_to_store = if val_ty == types::I8 {
                                    builder.ins().uextend(types::I64, val)
                                } else {
                                    val
                                };
                                builder.ins().store(MemFlagsData::trusted(), val_to_store, base_ptr, offset_bytes);
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

                    if let Some(term) = &mir_block.terminator {
                        match term {
                            Terminator::Goto(target, _) => {
                                builder.ins().jump(clif_blocks[target.0], &[]);
                            }
                            Terminator::Cond { cond, then_block, else_block, .. } => {
                                let c_val = lower_operand(cond, &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts)?;
                                builder.ins().brif(c_val, clif_blocks[then_block.0], &[], clif_blocks[else_block.0], &[]);
                            }
                            Terminator::Switch { discr, cases, default, .. } => {
                                let d_val = lower_operand(discr, &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts)?;
                                for (val, target_bb) in cases {
                                    let is_eq = builder.ins().icmp_imm_s(IntCC::Equal, d_val, *val);
                                    let next_check = builder.create_block();
                                    builder.ins().brif(is_eq, clif_blocks[target_bb.0], &[], next_check, &[]);
                                    builder.switch_to_block(next_check);
                                }
                                builder.ins().jump(clif_blocks[default.0], &[]);
                            }
                            Terminator::Call { callee, args, destination, success, .. } => {
                                if callee == "print" || callee == "println" {
                                    let arg_val = if let Some(first) = args.first() {
                                        lower_operand(first, &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts)?
                                    } else {
                                        builder.ins().iconst(types::I64, 0)
                                    };
                                    builder.ins().call(print_fn, &[arg_val]);
                                    if let Place::Local(loc) = destination {
                                        let zero = builder.ins().iconst(types::I64, 0);
                                        builder.def_var(local_vars[loc.0], zero);
                                        if let Some(&slot) = stack_slots.get(&loc.0) {
                                            builder.ins().stack_store(types::I64, zero, slot, 0);
                                        }
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
                                        lower_operand(first, &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts)?
                                    } else {
                                        builder.ins().iconst(types::I64, 0)
                                    };
                                    let call_inst = builder.ins().call(struct_decode_fn, &[layout_id_val, arg_val]);
                                    let res_val = builder.inst_results(call_inst)[0];
                                    if let Place::Local(loc) = destination {
                                        let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                        let res_cast = cast_clif_type(&mut builder, res_val, target_ty);
                                        builder.def_var(local_vars[loc.0], res_cast);
                                        if let Some(&slot) = stack_slots.get(&loc.0) {
                                            builder.ins().stack_store(types::I64, res_cast, slot, 0);
                                        }
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
                                    let locals_to_record = if let Some((_, live_locals)) = osr_info {
                                        live_locals
                                    } else {
                                        &all_locals
                                    };
                                    let target_pc_val = builder.ins().iconst(types::I64, b_idx as i64);
                                    builder.ins().call(init_deopt_fn, &[target_pc_val]);
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
                                        lower_operand(first, &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts)?
                                    } else {
                                        builder.ins().iconst(types::I64, 0)
                                    };
                                    let def_val = if args.len() > 1 {
                                        lower_operand(&args[1], &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts)?
                                    } else {
                                        builder.ins().iconst(types::I64, 0)
                                    };
                                    let call_inst = builder.ins().call(ic_dispatch_fn, &[site_id_val, tag_val, def_val]);
                                    let res_val = builder.inst_results(call_inst)[0];
                                    if let Place::Local(loc) = destination {
                                        let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                        let res_cast = cast_clif_type(&mut builder, res_val, target_ty);
                                        builder.def_var(local_vars[loc.0], res_cast);
                                        if let Some(&slot) = stack_slots.get(&loc.0) {
                                            builder.ins().stack_store(types::I64, res_cast, slot, 0);
                                        }
                                    }
                                    builder.ins().jump(clif_blocks[success.0], &[]);
                                } else {
                                    let (callee_func_id, param_count) = match callee.as_str() {
                                        "math.abs" | "abs" => (abs_callee, 1),
                                        "math.min" | "min" => (min_callee, 2),
                                        "math.max" | "max" => (max_callee, 2),
                                        "math.clamp" | "clamp" => (clamp_callee, 3),
                                        "table.len" | "array.len" => (len_callee, 1),
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
                                            (fid, args.len())
                                        }
                                    };
                                    let callee_ref = self.module.declare_func_in_func(callee_func_id, &mut builder.func);
                                    let mut arg_vals = Vec::with_capacity(param_count);
                                    for a in args {
                                        arg_vals.push(lower_operand(a, &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts)?);
                                    }
                                    while arg_vals.len() < param_count {
                                        arg_vals.push(builder.ins().iconst(types::I64, 0));
                                    }
                                    let call_inst = builder.ins().call(callee_ref, &arg_vals);
                                    let ret_vals = builder.inst_results(call_inst);
                                    if !ret_vals.is_empty() {
                                        let res_val = ret_vals[0];
                                        match destination {
                                            Place::Local(loc) => {
                                                let target_ty = mir_to_clif_type(&mir_fn.locals[loc.0].ty)?;
                                                let res_val = cast_clif_type(&mut builder, res_val, target_ty);
                                                builder.def_var(local_vars[loc.0], res_val);
                                                if let Some(&slot) = stack_slots.get(&loc.0) {
                                                    builder.ins().stack_store(types::I64, res_val, slot, 0);
                                                }
                                            }
                                            Place::Index(base, idx) => {
                                                let base_ptr = lower_place_addr(base, &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts);
                                                let idx_val = if let Some(&slot) = stack_slots.get(&idx.0) {
                                                    builder.ins().stack_load(types::I64, types::I64, slot, 0)
                                                } else {
                                                    builder.use_var(local_vars[idx.0])
                                                };
                                                let eight = builder.ins().iconst(types::I64, 8);
                                                let byte_idx = builder.ins().imul(idx_val, eight);
                                                let sixteen = builder.ins().iconst(types::I64, 16);
                                                let offset = builder.ins().iadd(byte_idx, sixteen);
                                                let elem_addr = builder.ins().iadd(base_ptr, offset);
                                                builder.ins().store(MemFlagsData::trusted(), res_val, elem_addr, 0);
                                            }
                                            Place::Field(base, field_idx) => {
                                                let base_ptr = lower_place_addr(base, &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts);
                                                let base_ty = place_type(base, &mir_fn.locals, &self.struct_layouts);
                                                let offset_bytes = match &base_ty {
                                                    MirType::Struct(name) => {
                                                        if let Some(layout) = self.struct_layouts.get(name) {
                                                            if let Some(f) = layout.fields.get(*field_idx) {
                                                                f.byte_offset as i32
                                                            } else {
                                                                16 + (*field_idx as i32) * 8
                                                            }
                                                        } else {
                                                            16 + (*field_idx as i32) * 8
                                                        }
                                                    }
                                                    _ => 16 + (*field_idx as i32) * 8,
                                                };
                                                builder.ins().store(MemFlagsData::trusted(), res_val, base_ptr, offset_bytes);
                                            }
                                        }
                                    } else if let Place::Local(loc) = destination {
                                        let zero = builder.ins().iconst(types::I64, 0);
                                        builder.def_var(local_vars[loc.0], zero);
                                        if let Some(&slot) = stack_slots.get(&loc.0) {
                                            builder.ins().stack_store(types::I64, zero, slot, 0);
                                        }
                                    }
                                    let stat_call = builder.ins().call(status_fn, &[]);
                                    let stat_val = builder.inst_results(stat_call)[0];
                                    let is_err = builder.ins().icmp_imm_s(IntCC::NotEqual, stat_val, 0);
                                    let call_cont_bb = builder.create_block();
                                    builder.ins().brif(is_err, emergency_exit_bb, &[], call_cont_bb, &[]);
                                    builder.switch_to_block(call_cont_bb);

                                    builder.ins().jump(clif_blocks[success.0], &[]);
                                }
                            }
                            Terminator::Return(opt_op, _) => {
                                if ret_clif_ty != types::I8 || mir_fn.signature.return_type != MirType::Unit {
                                    let ret_val = if let Some(op) = opt_op {
                                        let v = lower_operand(op, &mut builder, &local_vars, &stack_slots, &mir_fn.locals, &self.struct_layouts)?;
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
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;
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

    pub fn refine_locals(
        locals: &[LocalDecl],
        blocks: &[BasicBlock],
        return_type: &MirType,
    ) -> Vec<LocalDecl> {
        let mut inferred = locals.to_vec();
        let dummy_layouts = hashbrown::HashMap::new();
        for _ in 0..10 {
            let mut changed = false;
            for block in blocks {
                for stmt in &block.statements {
                    if let Statement::Assign(Place::Local(loc), rval, _) = stmt {
                        if inferred[loc.0].name.is_none() || inferred[loc.0].ty == MirType::Int {
                            let old_ty = inferred[loc.0].ty.clone();
                            match rval {
                                Rvalue::Use(Operand::Constant(MirConstant::Float(_))) => {
                                    inferred[loc.0].ty = MirType::Float;
                                }
                                Rvalue::Use(Operand::Constant(MirConstant::String(_))) => {
                                    inferred[loc.0].ty = MirType::String;
                                }
                                Rvalue::Use(Operand::Constant(MirConstant::Bool(_))) => {
                                    inferred[loc.0].ty = MirType::Bool;
                                }
                                Rvalue::Use(Operand::Copy(p) | Operand::Move(p)) => {
                                    let p_ty = place_type(p, &inferred, &dummy_layouts);
                                    if p_ty != MirType::Int {
                                        inferred[loc.0].ty = p_ty;
                                    }
                                }
                                Rvalue::Aggregate(ty, _) => {
                                    inferred[loc.0].ty = ty.clone();
                                }
                                Rvalue::BinaryOp(binop, lhs, rhs) => {
                                    let l_ty = operand_type(lhs, &inferred, &dummy_layouts);
                                    let r_ty = operand_type(rhs, &inferred, &dummy_layouts);
                                    if l_ty == MirType::Float || r_ty == MirType::Float {
                                        if matches!(
                                            binop,
                                            MirBinaryOp::Eq
                                                | MirBinaryOp::NotEq
                                                | MirBinaryOp::Lt
                                                | MirBinaryOp::LtEq
                                                | MirBinaryOp::Gt
                                                | MirBinaryOp::GtEq
                                        ) {
                                            inferred[loc.0].ty = MirType::Bool;
                                        } else {
                                            inferred[loc.0].ty = MirType::Float;
                                        }
                                    } else if l_ty == MirType::String && r_ty == MirType::String {
                                        if matches!(binop, MirBinaryOp::Add) {
                                            inferred[loc.0].ty = MirType::String;
                                        } else {
                                            inferred[loc.0].ty = MirType::Bool;
                                        }
                                    }
                                }
                                Rvalue::UnaryOp(unop, op) => {
                                    let o_ty = operand_type(op, &inferred, &dummy_layouts);
                                    if o_ty == MirType::Float && matches!(unop, MirUnaryOp::Neg) {
                                        inferred[loc.0].ty = MirType::Float;
                                    }
                                }
                                Rvalue::Cast(_, target_ty) => {
                                    inferred[loc.0].ty = target_ty.clone();
                                }
                                _ => {}
                            }
                            if inferred[loc.0].ty != old_ty {
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
                    if inferred[loc.0].name.is_none() && inferred[loc.0].ty == MirType::Int && *return_type != MirType::Int {
                        inferred[loc.0].ty = return_type.clone();
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        inferred
    }

    pub fn refine_mir_function(mir_fn: &MirFunction) -> MirFunction {
        let mut cloned = mir_fn.clone();
        cloned.locals = refine_locals(&cloned.locals, &cloned.blocks, &cloned.signature.return_type);
        let any_promoted = cloned.locals.iter().any(|l| l.is_stack_promoted);
        if !any_promoted {
            crate::mir::opt::escape_analysis::EscapeAnalysis::run_on_function(&mut cloned);
        }
        cloned
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
            | MirType::Effect { .. }
            | MirType::Unknown => Ok(types::I64),
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

    fn place_type(
        place: &Place,
        locals: &[LocalDecl],
        struct_layouts: &hashbrown::HashMap<String, MirStructLayout>,
    ) -> MirType {
        match place {
            Place::Local(loc) => locals.get(loc.0).map(|l| l.ty.clone()).unwrap_or(MirType::Int),
            Place::Index(base, _) => match place_type(base, locals, struct_layouts) {
                MirType::Array(inner) => *inner,
                MirType::Tuple(elems) => elems.first().cloned().unwrap_or(MirType::Int),
                MirType::Struct(name) => {
                    with_current_mir_context(|ctx| ctx.get_struct_field_type(&name, 0)).unwrap_or(MirType::Int)
                }
                _ => MirType::Int,
            },
            Place::Field(base, idx) => match place_type(base, locals, struct_layouts) {
                MirType::Tuple(elems) => elems.get(*idx).cloned().unwrap_or(MirType::Int),
                MirType::Struct(name) => {
                    if let Some(layout) = struct_layouts.get(&name) {
                        if let Some(f) = layout.fields.get(*idx) {
                            return match f.tag {
                                CanonicalTypeTag::Bool => MirType::Bool,
                                CanonicalTypeTag::Int64
                                | CanonicalTypeTag::UInt64
                                | CanonicalTypeTag::DateTimeUtc => MirType::Int,
                                CanonicalTypeTag::Float64 => MirType::Float,
                                CanonicalTypeTag::StringSlice => MirType::String,
                                _ => MirType::Int,
                            };
                        }
                    }
                    with_current_mir_context(|ctx| ctx.get_struct_field_type(&name, *idx)).unwrap_or(MirType::Int)
                }
                _ => MirType::Int,
            },
        }
    }

    fn lower_place_addr(
        place: &Place,
        builder: &mut FunctionBuilder,
        local_vars: &[Variable],
        stack_slots: &hashbrown::HashMap<usize, StackSlot>,
        locals: &[LocalDecl],
        struct_layouts: &hashbrown::HashMap<String, MirStructLayout>,
    ) -> ClifValue {
        match place {
            Place::Local(loc) => {
                if let Some(&slot) = stack_slots.get(&loc.0) {
                    builder.ins().stack_addr(types::I64, slot, 0)
                } else {
                    builder.use_var(local_vars[loc.0])
                }
            }
            Place::Index(base, idx) => {
                let base_ptr = lower_place_addr(base, builder, local_vars, stack_slots, locals, struct_layouts);
                let idx_val = if let Some(&slot) = stack_slots.get(&idx.0) {
                    builder.ins().stack_load(types::I64, types::I64, slot, 0)
                } else {
                    builder.use_var(local_vars[idx.0])
                };
                let eight = builder.ins().iconst(types::I64, 8);
                let byte_idx = builder.ins().imul(idx_val, eight);
                let sixteen = builder.ins().iconst(types::I64, 16);
                let offset = builder.ins().iadd(byte_idx, sixteen);
                builder.ins().iadd(base_ptr, offset)
            }
            Place::Field(base, field_idx) => {
                let base_ptr = lower_place_addr(base, builder, local_vars, stack_slots, locals, struct_layouts);
                let base_ty = place_type(base, locals, struct_layouts);
                let offset_bytes = match &base_ty {
                    MirType::Struct(name) => {
                        if let Some(layout) = struct_layouts.get(name) {
                            if let Some(f) = layout.fields.get(*field_idx) {
                                f.byte_offset as i64
                            } else {
                                16 + (*field_idx as i64) * 8
                            }
                        } else {
                            16 + (*field_idx as i64) * 8
                        }
                    }
                    _ => 16 + (*field_idx as i64) * 8,
                };
                let offset = builder.ins().iconst(types::I64, offset_bytes);
                builder.ins().iadd(base_ptr, offset)
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
    ) -> ClifValue {
        match place {
            Place::Local(loc) => {
                let mir_ty = locals.get(loc.0).map(|l| l.ty.clone()).unwrap_or(MirType::Int);
                let clif_ty = mir_to_clif_type(&mir_ty).unwrap_or(types::I64);
                if let Some(&slot) = stack_slots.get(&loc.0) {
                    builder.ins().stack_load(types::I64, clif_ty, slot, 0)
                } else {
                    builder.use_var(local_vars[loc.0])
                }
            }
            Place::Index(_, _) | Place::Field(_, _) => {
                let addr = lower_place_addr(place, builder, local_vars, stack_slots, locals, struct_layouts);
                let mir_ty = place_type(place, locals, struct_layouts);
                let clif_ty = mir_to_clif_type(&mir_ty).unwrap_or(types::I64);
                builder.ins().load(clif_ty, MemFlagsData::trusted(), addr, 0)
            }
        }
    }

    fn operand_type(
        op: &Operand,
        locals: &[LocalDecl],
        struct_layouts: &hashbrown::HashMap<String, MirStructLayout>,
    ) -> MirType {
        match op {
            Operand::Copy(p) | Operand::Move(p) => place_type(p, locals, struct_layouts),
            Operand::Constant(c) => match c {
                MirConstant::Int(_) => MirType::Int,
                MirConstant::Float(_) => MirType::Float,
                MirConstant::Bool(_) => MirType::Bool,
                MirConstant::String(_) => MirType::String,
                MirConstant::Unit => MirType::Unit,
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
    ) -> Result<ClifValue, MirCraneliftBackendError> {
        match op {
            Operand::Copy(place) | Operand::Move(place) => {
                Ok(lower_place(place, builder, local_vars, stack_slots, locals, struct_layouts))
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

    fn lower_rvalue(
        rval: &Rvalue,
        builder: &mut FunctionBuilder,
        local_vars: &[Variable],
        stack_slots: &hashbrown::HashMap<usize, StackSlot>,
        locals: &[LocalDecl],
        struct_layouts: &hashbrown::HashMap<String, MirStructLayout>,
        alloc_fn: FuncRef,
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
        emergency_exit_bb: Block,
    ) -> Result<ClifValue, MirCraneliftBackendError> {
        match rval {
            Rvalue::Use(op) => lower_operand(op, builder, local_vars, stack_slots, locals, struct_layouts),
            Rvalue::Cast(op, target_ty) => {
                let val = lower_operand(op, builder, local_vars, stack_slots, locals, struct_layouts)?;
                let target_clif_ty = mir_to_clif_type(target_ty)?;
                Ok(cast_clif_type(builder, val, target_clif_ty))
            }
            Rvalue::BinaryOp(binop, lhs, rhs) => {
                let l_ty_mir = operand_type(lhs, locals, struct_layouts);
                let r_ty_mir = operand_type(rhs, locals, struct_layouts);
                let is_float = l_ty_mir == MirType::Float || r_ty_mir == MirType::Float;
                let is_string = l_ty_mir == MirType::String && r_ty_mir == MirType::String;

                if is_float {
                    let mut l = lower_operand(lhs, builder, local_vars, stack_slots, locals, struct_layouts)?;
                    let mut r = lower_operand(rhs, builder, local_vars, stack_slots, locals, struct_layouts)?;
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
                    let l = lower_operand(lhs, builder, local_vars, stack_slots, locals, struct_layouts)?;
                    let r = lower_operand(rhs, builder, local_vars, stack_slots, locals, struct_layouts)?;
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

                let mut l = lower_operand(lhs, builder, local_vars, stack_slots, locals, struct_layouts)?;
                let mut r = lower_operand(rhs, builder, local_vars, stack_slots, locals, struct_layouts)?;
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
                let op_ty = operand_type(op, locals, struct_layouts);
                let v = lower_operand(op, builder, local_vars, stack_slots, locals, struct_layouts)?;
                if op_ty == MirType::Float && matches!(unop, MirUnaryOp::Neg) {
                    return Ok(builder.ins().fneg(v));
                }
                if op_ty == MirType::String && matches!(unop, MirUnaryOp::Len) {
                    let call = builder.ins().call(string_len_fn, &[v]);
                    return Ok(builder.inst_results(call)[0]);
                }
                let val = match unop {
                    MirUnaryOp::Neg => builder.ins().ineg(v),
                    MirUnaryOp::Not => builder.ins().bnot(v),
                    MirUnaryOp::BitNot => builder.ins().bnot(v),
                    MirUnaryOp::Len => {
                        builder.ins().load(types::I64, MemFlagsData::trusted(), v, 0)
                    }
                };
                Ok(val)
            }
            Rvalue::Aggregate(MirType::Struct(name), elements) if struct_layouts.contains_key(name) => {
                let layout = &struct_layouts[name];
                let size_val = builder.ins().iconst(types::I64, layout.size_bytes as i64);
                let call_inst = builder.ins().call(alloc_fn, &[size_val]);
                let struct_ptr = builder.inst_results(call_inst)[0];
                emit_status_check(builder, status_fn, emergency_exit_bb);
                for (i, elem) in elements.iter().enumerate() {
                    let el_val = lower_operand(elem, builder, local_vars, stack_slots, locals, struct_layouts)?;
                    let offset = if let Some(f) = layout.fields.get(i) {
                        f.byte_offset as i32
                    } else {
                        16 + (i as i32) * 8
                    };
                    builder.ins().store(MemFlagsData::trusted(), el_val, struct_ptr, offset);
                }
                Ok(struct_ptr)
            }
            Rvalue::Aggregate(MirType::Array(_) | MirType::Tuple(_) | MirType::Struct(_), elements) => {
                let len = elements.len() as i64;
                let len_val = builder.ins().iconst(types::I64, len);
                let call_inst = builder.ins().call(alloc_fn, &[len_val]);
                let array_ptr = builder.inst_results(call_inst)[0];
                emit_status_check(builder, status_fn, emergency_exit_bb);
                for (i, elem) in elements.iter().enumerate() {
                    let el_val = lower_operand(elem, builder, local_vars, stack_slots, locals, struct_layouts)?;
                    let el_ty = builder.func.dfg.value_type(el_val);
                    let el_to_store = if el_ty == types::I8 {
                        builder.ins().uextend(types::I64, el_val)
                    } else {
                        el_val
                    };
                    builder.ins().store(MemFlagsData::trusted(), el_to_store, array_ptr, 16 + (i as i32) * 8);
                }
                Ok(array_ptr)
            }
            _ => Err(MirCraneliftBackendError::UnsupportedConstant(MirConstant::Unit)),
        }
    }
}
