pub mod budget;
pub mod channel;
mod corelib;
pub(crate) mod cycle;
#[cfg(feature = "std")]
pub mod osr;
#[cfg(feature = "std")]
pub mod stdlib;
mod task;
pub mod cranelift_dispatch;
#[path = "../telemetry/mod.rs"]
pub mod telemetry;
pub use self::budget::BudgetState;
pub use self::task::{FiberContinuation, TaskId, TaskInstance, TaskManager, TaskState, WaitKey, YieldReason};
pub(super) use crate::ast::{FieldOwnership, StructDef, Type, TypeKind};
pub(super) use crate::bytecode::{
    FieldStorage, Function, Instruction, NativeCallResult, Register, StructLayout, TaskHandle,
    Upvalue, Value,
};
#[cfg(feature = "std")]
pub(super) use crate::embed::native_types::ModuleStub;
pub(super) use crate::error::StackFrame;
pub(super) use crate::jit::{
    JitCompiler, JitState, MAX_TRACE_LENGTH, TraceOptimizer, TraceRecorder,
};
pub(super) use crate::number::{LustFloat, LustInt, float_from_int, int_from_usize};
pub(super) use crate::{LustError, Result};
pub(super) use alloc::{
    boxed::Box,
    format,
    rc::Rc,
    string::{String, ToString},
    vec,
    vec::Vec,
};
#[cfg(feature = "std")]
use core::cell::RefCell;
// FixedState (not hashbrown's DefaultHashBuilder/foldhash RandomState): RandomState
// resolves its global seed from a per-crate-copy static, so maps created by the host
// binary are unreadable by extension cdylibs that statically link their own copy.
use foldhash::fast::FixedState as DefaultHashBuilder;
use hashbrown::HashMap;
mod api;
mod execution;
mod tasks;
mod tracing;
// Used by the std-only extern stub writers (`externs`, `packages`).
#[cfg(feature = "std")]
pub(crate) use self::api::format_doc_comment;
pub use self::api::{
    NativeEffectSpec, NativeExport, NativeExportParam, NativeGasPolicy, NativeMemoryPolicy,
    NativePurity,
};
pub use self::execution::{
    register_effect_finalizer, reset_effect_handler_state, MUTATION_M1_EXECUTION_COUNT,
    MUTATION_M1_SKIP_GAS, MutationM1Guard,
};
pub use self::tasks::{
    MUTATION_M2_EXECUTION_COUNT, MUTATION_M2_SKIP_PANIC_FINALIZER, MutationM2Guard,
};
pub(crate) use self::tracing::JitCallSite;
#[cfg(feature = "std")]
thread_local! {
    static CURRENT_VM_STACK: RefCell<Vec<*mut VM>> = const { RefCell::new(Vec::new()) };
}

#[cfg(not(feature = "std"))]
struct VmStack {
    inner: core::cell::UnsafeCell<Option<Vec<*mut VM>>>,
}

#[cfg(not(feature = "std"))]
impl VmStack {
    const fn new() -> Self {
        Self {
            inner: core::cell::UnsafeCell::new(None),
        }
    }

    fn with_mut<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut Vec<*mut VM>) -> R,
    {
        let vec = self.ensure_vec();
        f(vec)
    }

    fn with_ref<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Vec<*mut VM>) -> R,
    {
        let vec = self.ensure_vec();
        f(vec)
    }

    fn ensure_vec(&self) -> &mut Vec<*mut VM> {
        unsafe {
            let slot = &mut *self.inner.get();
            if slot.is_none() {
                *slot = Some(Vec::new());
            }
            slot.as_mut().unwrap()
        }
    }
}

#[cfg(not(feature = "std"))]
unsafe impl Sync for VmStack {}

#[cfg(not(feature = "std"))]
static VM_STACK: VmStack = VmStack::new();

#[cfg(feature = "std")]
fn with_vm_stack_ref<F, R>(f: F) -> R
where
    F: FnOnce(&Vec<*mut VM>) -> R,
{
    CURRENT_VM_STACK.with(|stack| {
        let stack = stack.borrow();
        f(&stack)
    })
}

#[cfg(feature = "std")]
fn with_vm_stack_mut<F, R>(f: F) -> R
where
    F: FnOnce(&mut Vec<*mut VM>) -> R,
{
    CURRENT_VM_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        f(&mut stack)
    })
}

#[cfg(not(feature = "std"))]
fn with_vm_stack_ref<F, R>(f: F) -> R
where
    F: FnOnce(&Vec<*mut VM>) -> R,
{
    VM_STACK.with_ref(f)
}

#[cfg(not(feature = "std"))]
fn with_vm_stack_mut<F, R>(f: F) -> R
where
    F: FnOnce(&mut Vec<*mut VM>) -> R,
{
    VM_STACK.with_mut(f)
}

pub fn push_vm_ptr(vm: *mut VM) {
    with_vm_stack_mut(|stack| stack.push(vm));
}

pub fn pop_vm_ptr() {
    with_vm_stack_mut(|stack| {
        stack.pop();
    });
}

pub fn current_vm_ptr() -> Option<*mut VM> {
    with_vm_stack_ref(|stack| stack.last().copied().filter(|p| !p.is_null()))
}

pub struct CurrentVmGuard {
    vm: *mut VM,
    _lua_guard: Option<crate::lua_compat::LuaCompatContextGuard>,
    _mir_guard: Option<crate::mir::runtime_helpers::MirContextGuard>,
}

impl CurrentVmGuard {
    /// Creates a guard from a mutable reference to VM (safe).
    pub fn from_vm(vm: &mut VM) -> Self {
        unsafe { Self::new(vm as *mut VM) }
    }

    /// Creates a guard from a raw VM pointer.
    ///
    /// # Safety
    /// Caller must ensure that `vm` is either null or points to a valid, live `VM` instance.
    pub unsafe fn new(vm: *mut VM) -> Self {
        push_vm_ptr(vm);
        let (lua_guard, mir_guard) = if !vm.is_null() {
            let vm_ref = unsafe { &*vm };
            (
                Some(vm_ref.lua_context.enter()),
                Some(vm_ref.mir_context.enter()),
            )
        } else {
            (None, None)
        };
        Self {
            vm,
            _lua_guard: lua_guard,
            _mir_guard: mir_guard,
        }
    }
}

impl Drop for CurrentVmGuard {
    fn drop(&mut self) {
        with_vm_stack_mut(|stack| {
            if let Some(top) = stack.last() {
                if *top == self.vm {
                    stack.pop();
                    return;
                }
            }
            if let Some(pos) = stack.iter().rposition(|&p| p == self.vm) {
                stack.remove(pos);
            }
        });
    }
}

pub(crate) struct ActiveNativeGuard {
    vm: *mut VM,
}

impl Drop for ActiveNativeGuard {
    fn drop(&mut self) {
        if !self.vm.is_null() {
            unsafe {
                (*self.vm).active_native_depth = (*self.vm).active_native_depth.saturating_sub(1);
            }
        }
    }
}

pub(crate) struct ActiveJitGuard {
    vm: *mut VM,
}

impl Drop for ActiveJitGuard {
    fn drop(&mut self) {
        if !self.vm.is_null() {
            unsafe {
                (*self.vm).active_jit_depth = (*self.vm).active_jit_depth.saturating_sub(1);
            }
        }
    }
}

pub(super) const TO_STRING_TRAIT: &str = "ToString";
pub(super) const TO_STRING_METHOD: &str = "to_string";
pub(super) const HASH_KEY_TRAIT: &str = "HashKey";
pub(super) const HASH_KEY_METHOD: &str = "to_hashkey";

/// Execution backend selection for VM runtime dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VmBackend {
    /// Baseline Bytecode VM interpreter (with Dynasm trace JIT where supported).
    #[default]
    Bytecode,
    /// Native Cranelift JIT compilation backend via Lust MIR.
    Cranelift,
}

pub struct VM {
    // Retain the host runtime's lookup function across dynamic-library boundaries.
    // Each extension can have its own copy of CURRENT_VM_STACK.
    current_vm_lookup: fn() -> Option<*mut VM>,
    pub backend: VmBackend,
    pub cranelift_entry_count: u64,
    pub cranelift_compilation_count: u64,
    #[cfg(feature = "std")]
    pub(crate) cranelift_jit: Option<crate::mir::cranelift_backend::jit_engine::CraneliftJitCompiler>,
    pub mir_functions: HashMap<String, crate::mir::ir::MirFunction>,
    pub cranelift_compiled: HashMap<String, *const u8>,
    pub(super) jit: JitState,
    /// Cells compiled code reads and writes through the VM pointer.
    pub(crate) jit_cells: crate::jit::JitCells,
    /// One shared object per unit enum value (`Option.None`), keyed by
    /// the interned names: constructing one is a count bump.
    pub(crate) unit_enums: HashMap<(usize, usize), Rc<crate::bytecode::EnumObject>>,
    pub(super) budgets: BudgetState,
    pub(super) functions: Vec<Function>,
    /// Per-function facts the call path needs, indexed like `functions`
    /// (see `CallMeta`).
    pub(super) call_meta: Vec<CallMeta>,
    pub(super) natives: HashMap<String, Value>,
    pub(super) globals: HashMap<String, Value>,
    /// Bumped on every change to `globals` or `natives`. Traces snapshot
    /// the globals they read and guard on this counter (see
    /// `TraceOp::GuardGlobals`).
    pub(crate) globals_version: u64,
    pub(super) map_hasher: DefaultHashBuilder,
    /// Frames are boxed: a `CallFrame` carries 12 KB of inline registers,
    /// and moving that into and out of the stack on every call dominated
    /// call cost. Returned frames go to `frame_pool` for reuse.
    pub(super) call_stack: Vec<Box<CallFrame>>,
    /// Recycled frames with their registers already reset to Nil.
    pub(super) frame_pool: Vec<Box<CallFrame>>,
    /// Resolved user-defined struct methods per call site:
    /// (struct layout, calling function, method-name constant) -> function.
    /// Resolution otherwise formats a mangled name and scans every function
    /// by string on each call.
    pub(super) method_cache: hashbrown::HashMap<(usize, usize, u16), usize>,
    /// Reused buffer for call arguments, so a call allocates nothing.
    pub(super) arg_scratch: Vec<Value>,
    pub(super) max_stack_depth: usize,
    pub(super) pending_return_value: Option<Value>,
    pub(super) pending_return_dest: Option<Register>,
    pub(super) pending_jit_error: Option<LustError>,
    pub(super) trace_recorder: Option<TraceRecorder>,
    /// Set by `jit_run_nested_loop` when the inner loop's trace bailed out
    /// somewhere other than its normal exit: the ip the interpreter resumes
    /// at instead of the outer guard's own bailout ip.
    pub(super) nested_loop_exit_ip: Option<usize>,
    pub(super) skip_next_trace_record: bool,
    pub(super) trait_impls: HashMap<(String, String), bool>,
    pub(super) struct_tostring_cache: HashMap<usize, Rc<String>>,
    pub(super) struct_metadata: HashMap<String, RuntimeStructInfo>,
    pub(super) call_until_depth: Option<usize>,
    pub(super) task_manager: TaskManager,
    pub(super) current_task: Option<TaskId>,
    pub(super) active_scope_tasks: Vec<TaskId>,
    pub(super) pending_task_signal: Option<TaskSignal>,
    pub(super) last_task_signal: Option<TaskSignal>,
    pub(super) cycle_collector: cycle::CycleCollector,
    pub(super) exported_natives: Vec<NativeExport>,
    pub(super) export_prefix_stack: Vec<String>,
    #[cfg(feature = "std")]
    pub(super) exported_type_stubs: Vec<ModuleStub>,
    pub(super) last_dispatched_target: Option<(usize, String)>,
    pub(super) root_frame: Option<Box<CallFrame>>,
    pub(super) active_native_depth: usize,
    pub(super) active_jit_depth: usize,
    pub host_ledger: TrustedHostResourceLedger,
    #[cfg(feature = "std")]
    pub rng: core::cell::RefCell<Box<dyn rand::Rng + Send>>,
    pub id: crate::bytecode::value::VmId,
    pub next_array_sequence_id: core::sync::atomic::AtomicU64,
    pub array_id_map: core::cell::RefCell<HashMap<usize, crate::bytecode::value::ArraySequenceId>>,
    pub lua_context: Rc<crate::lua_compat::LuaCompatContext>,
    pub mir_context: alloc::sync::Arc<crate::mir::runtime_helpers::MirContext>,
    pub clock: alloc::sync::Arc<crate::sim::clock::LustClock>,
    #[cfg(feature = "std")]
    pub vfs: alloc::sync::Arc<dyn crate::vfs::VirtualFileSystem>,
    #[cfg(feature = "std")]
    pub env: crate::vfs::VirtualEnv,
    #[cfg(feature = "std")]
    pub console: crate::vfs::VirtualConsole,
    pub quantum_fuel: Option<u64>,
    pub config: crate::config::LustConfig,
}

pub type HostResourceCleanupFn = fn(usize);

#[derive(Clone, Copy)]
pub struct HostResourceSlot {
    pub handle: usize,
    pub cleanup: HostResourceCleanupFn,
    pub generation: u32,
    pub scope_id: u64,
    pub active: bool,
}

pub struct TrustedHostResourceLedger {
    pub slots: [HostResourceSlot; 256],
    pub count: usize,
    pub next_generation: u32,
}

impl Default for TrustedHostResourceLedger {
    fn default() -> Self {
        Self::new()
    }
}

impl TrustedHostResourceLedger {
    pub fn new() -> Self {
        Self {
            slots: [HostResourceSlot {
                handle: 0,
                cleanup: |_| {},
                generation: 1,
                scope_id: 0,
                active: false,
            }; 256],
            count: 0,
            next_generation: 1,
        }
    }

    pub fn acquire(
        &mut self,
        handle: usize,
        cleanup: HostResourceCleanupFn,
    ) -> crate::error::Result<usize> {
        self.acquire_scoped(handle, cleanup, 0)
    }

    pub fn acquire_scoped(
        &mut self,
        handle: usize,
        cleanup: HostResourceCleanupFn,
        scope_id: u64,
    ) -> crate::error::Result<usize> {
        for (idx, slot) in self.slots.iter_mut().enumerate() {
            if !slot.active {
                slot.handle = handle;
                slot.cleanup = cleanup;
                slot.generation = self.next_generation;
                self.next_generation = self.next_generation.wrapping_add(1);
                if self.next_generation == 0 {
                    self.next_generation = 1;
                }
                slot.scope_id = scope_id;
                slot.active = true;
                self.count += 1;
                return Ok(idx);
            }
        }
        Err(crate::error::LustError::RuntimeError {
            message: "TooManyOpenResources: Host resource ledger full (capacity 256 slots)".to_string(),
        })
    }

    pub fn release(&mut self, slot_id: usize) {
        if slot_id < self.slots.len() && self.slots[slot_id].active {
            let slot = &mut self.slots[slot_id];
            slot.active = false;
            (slot.cleanup)(slot.handle);
            if self.count > 0 {
                self.count -= 1;
            }
        }
    }

    pub fn cleanup_scope(&mut self, scope_id: u64) {
        for slot in self.slots.iter_mut().rev() {
            if slot.active && slot.scope_id == scope_id {
                slot.active = false;
                (slot.cleanup)(slot.handle);
                if self.count > 0 {
                    self.count -= 1;
                }
            }
        }
    }

    pub fn cleanup_all(&mut self) {
        for slot in self.slots.iter_mut() {
            if slot.active {
                slot.active = false;
                (slot.cleanup)(slot.handle);
            }
        }
        self.count = 0;
    }
}

impl VM {
    #[inline]
    pub fn config(&self) -> &crate::config::LustConfig {
        &self.config
    }

    #[inline]
    pub fn charge_gas(&mut self, amount: u64) -> crate::error::Result<()> {
        let res = self.budgets.charge_gas(amount);
        #[cfg(any(
            feature = "telemetry",
            feature = "telemetry-all",
            feature = "telemetry-vm"
        ))]
        {
            if res.is_ok() {
                self.flush_gas_telemetry();
            } else {
                self.flush_gas_telemetry();
                self.record_gas_exhausted("vm");
            }
        }
        res
    }

    #[cfg(any(
        feature = "telemetry",
        feature = "telemetry-all",
        feature = "telemetry-vm",
        feature = "telemetry-jit",
        feature = "telemetry-effects",
        feature = "telemetry-user"
    ))]
    pub fn set_telemetry_sink(
        &mut self,
        sink: alloc::sync::Arc<dyn crate::vm::telemetry::TelemetrySink>,
        mask: crate::vm::telemetry::TelemetrySignalMask,
    ) {
        self.budgets.telemetry_sink = Some(sink);
        self.budgets.telemetry_mask = mask;
    }

    #[cfg(not(any(
        feature = "telemetry",
        feature = "telemetry-all",
        feature = "telemetry-vm",
        feature = "telemetry-jit",
        feature = "telemetry-effects",
        feature = "telemetry-user"
    )))]
    #[inline(always)]
    pub fn set_telemetry_sink(
        &mut self,
        _sink: alloc::sync::Arc<dyn crate::vm::telemetry::TelemetrySink>,
        _mask: crate::vm::telemetry::TelemetrySignalMask,
    ) {
    }

    #[cfg(any(
        feature = "telemetry",
        feature = "telemetry-all",
        feature = "telemetry-vm",
        feature = "telemetry-jit",
        feature = "telemetry-effects",
        feature = "telemetry-user"
    ))]
    pub fn clear_telemetry_sink(&mut self) {
        self.budgets.telemetry_sink = None;
        self.budgets.telemetry_mask = crate::vm::telemetry::TelemetrySignalMask::NONE;
    }

    #[cfg(not(any(
        feature = "telemetry",
        feature = "telemetry-all",
        feature = "telemetry-vm",
        feature = "telemetry-jit",
        feature = "telemetry-effects",
        feature = "telemetry-user"
    )))]
    #[inline(always)]
    pub fn clear_telemetry_sink(&mut self) {}

    #[cfg(any(
        feature = "telemetry",
        feature = "telemetry-all",
        feature = "telemetry-vm",
        feature = "telemetry-jit",
        feature = "telemetry-effects",
        feature = "telemetry-user"
    ))]
    pub fn telemetry_sink(&self) -> Option<&alloc::sync::Arc<dyn crate::vm::telemetry::TelemetrySink>> {
        self.budgets.telemetry_sink.as_ref()
    }

    #[cfg(not(any(
        feature = "telemetry",
        feature = "telemetry-all",
        feature = "telemetry-vm",
        feature = "telemetry-jit",
        feature = "telemetry-effects",
        feature = "telemetry-user"
    )))]
    #[inline(always)]
    pub fn telemetry_sink(&self) -> Option<&alloc::sync::Arc<dyn crate::vm::telemetry::TelemetrySink>> {
        None
    }

    #[cfg(any(
        feature = "telemetry",
        feature = "telemetry-all",
        feature = "telemetry-vm",
        feature = "telemetry-jit",
        feature = "telemetry-effects",
        feature = "telemetry-user"
    ))]
    pub fn telemetry_mask(&self) -> crate::vm::telemetry::TelemetrySignalMask {
        self.budgets.telemetry_mask
    }

    #[cfg(not(any(
        feature = "telemetry",
        feature = "telemetry-all",
        feature = "telemetry-vm",
        feature = "telemetry-jit",
        feature = "telemetry-effects",
        feature = "telemetry-user"
    )))]
    #[inline(always)]
    pub fn telemetry_mask(&self) -> crate::vm::telemetry::TelemetrySignalMask {
        crate::vm::telemetry::TelemetrySignalMask::NONE
    }

    pub fn telemetry_snapshot(&self) -> crate::vm::telemetry::RuntimeTelemetrySnapshot {
        let stats = self.jit_stats();
        crate::vm::telemetry::RuntimeTelemetrySnapshot {
            gas_used: self.gas_used(),
            gas_remaining: self.gas_remaining(),
            memory_used_bytes: self.memory_used(),
            active_host_slots: self.host_ledger.count as u16,
            jit_traces_compiled: stats.root_traces_compiled + stats.functions_compiled,
            jit_guard_exits: stats.guard_exits,
        }
    }

    #[inline]
    pub fn flush_gas_telemetry(&mut self) {
        #[cfg(any(
            feature = "telemetry",
            feature = "telemetry-all",
            feature = "telemetry-vm"
        ))]
        {
            if let Some(ref sink) = self.budgets.telemetry_sink {
                if self.budgets.telemetry_mask.contains(crate::vm::telemetry::TelemetrySignalMask::GAS) {
                    let current = self.budgets.gas.used;
                    if current > self.budgets.last_gas_consumed_flushed {
                        let delta = current - self.budgets.last_gas_consumed_flushed;
                        self.budgets.last_gas_consumed_flushed = current;
                        sink.counter_add("lust.vm.gas.consumed_total", delta, &[]);
                    }
                }
            }
        }
    }

    #[inline]
    pub fn record_gas_exhausted(&self, source: &str) {
        #[cfg(any(
            feature = "telemetry",
            feature = "telemetry-all",
            feature = "telemetry-vm"
        ))]
        {
            if let Some(ref sink) = self.budgets.telemetry_sink {
                if self.budgets.telemetry_mask.contains(crate::vm::telemetry::TelemetrySignalMask::GAS) {
                    sink.counter_add("lust.vm.gas.exhausted_total", 1, &[("source", source)]);
                }
            }
        }
        let _ = source;
    }

    #[inline]
    pub fn record_memory_used(&self, bytes: usize) {
        #[cfg(any(
            feature = "telemetry",
            feature = "telemetry-all",
            feature = "telemetry-vm"
        ))]
        {
            if let Some(ref sink) = self.budgets.telemetry_sink {
                if self.budgets.telemetry_mask.contains(crate::vm::telemetry::TelemetrySignalMask::MEMORY) {
                    sink.gauge_set("lust.vm.memory.bytes_used", bytes as f64, &[]);
                }
            }
        }
        let _ = bytes;
    }

    #[inline]
    pub fn record_memory_limit_rejection(&self) {
        #[cfg(any(
            feature = "telemetry",
            feature = "telemetry-all",
            feature = "telemetry-vm"
        ))]
        {
            if let Some(ref sink) = self.budgets.telemetry_sink {
                if self.budgets.telemetry_mask.contains(crate::vm::telemetry::TelemetrySignalMask::MEMORY) {
                    sink.counter_add("lust.vm.memory.limit_rejections_total", 1, &[]);
                }
            }
        }
    }

    #[inline]
    pub fn record_host_ledger_active(&self, count: u16) {
        #[cfg(any(
            feature = "telemetry",
            feature = "telemetry-all",
            feature = "telemetry-vm"
        ))]
        {
            if let Some(ref sink) = self.budgets.telemetry_sink {
                if self.budgets.telemetry_mask.contains(crate::vm::telemetry::TelemetrySignalMask::HOST_LEDGER) {
                    sink.gauge_set("lust.vm.host_ledger.active_slots", count as f64, &[]);
                }
            }
        }
        let _ = count;
    }

    #[inline]
    pub fn record_host_ledger_overflow(&self) {
        #[cfg(any(
            feature = "telemetry",
            feature = "telemetry-all",
            feature = "telemetry-vm"
        ))]
        {
            if let Some(ref sink) = self.budgets.telemetry_sink {
                if self.budgets.telemetry_mask.contains(crate::vm::telemetry::TelemetrySignalMask::HOST_LEDGER) {
                    sink.counter_add("lust.vm.host_ledger.overflow_total", 1, &[]);
                }
            }
        }
    }

    #[inline]
    pub fn notify_host_resource_acquired(&self) {
        self.record_host_ledger_active(self.host_ledger.count as u16);
    }

    #[inline]
    pub fn notify_host_resource_released(&self) {
        self.record_host_ledger_active(self.host_ledger.count as u16);
    }

    #[inline]
    pub fn record_jit_trace_compiled(&self, count: u64) {
        #[cfg(any(
            feature = "telemetry",
            feature = "telemetry-all",
            feature = "telemetry-jit"
        ))]
        {
            if let Some(ref sink) = self.budgets.telemetry_sink {
                if self.budgets.telemetry_mask.contains(crate::vm::telemetry::TelemetrySignalMask::JIT) {
                    sink.counter_add("lust.jit.traces_compiled_total", count, &[]);
                }
            }
        }
        let _ = count;
    }

    #[inline]
    pub fn record_jit_bailout(&self, guard_kind: &str) {
        #[cfg(any(
            feature = "telemetry",
            feature = "telemetry-all",
            feature = "telemetry-jit"
        ))]
        {
            if let Some(ref sink) = self.budgets.telemetry_sink {
                if self.budgets.telemetry_mask.contains(crate::vm::telemetry::TelemetrySignalMask::JIT) {
                    sink.counter_add("lust.jit.bailouts_total", 1, &[("guard_kind", guard_kind)]);
                }
            }
        }
        let _ = guard_kind;
    }

    #[inline]
    pub fn record_fiber_lifecycle(&self, event: &str) {
        #[cfg(any(
            feature = "telemetry",
            feature = "telemetry-all",
            feature = "telemetry-effects"
        ))]
        {
            if let Some(ref sink) = self.budgets.telemetry_sink {
                if self.budgets.telemetry_mask.contains(crate::vm::telemetry::TelemetrySignalMask::EFFECTS) {
                    sink.counter_add("lust.fiber.lifecycle_total", 1, &[("event", event)]);
                }
            }
        }
        let _ = event;
    }

    #[inline]
    pub fn span_enter_effect(&self, name: &str, labels: &[(&str, &str)]) -> u64 {
        #[cfg(any(
            feature = "telemetry",
            feature = "telemetry-all",
            feature = "telemetry-effects"
        ))]
        {
            if let Some(ref sink) = self.budgets.telemetry_sink {
                if self.budgets.telemetry_mask.contains(crate::vm::telemetry::TelemetrySignalMask::EFFECTS) {
                    return sink.span_enter(name, labels);
                }
            }
        }
        let _ = (name, labels);
        0
    }

    #[inline]
    pub fn span_exit_effect(&self, span_id: u64, status: crate::vm::telemetry::SpanStatus) {
        #[cfg(any(
            feature = "telemetry",
            feature = "telemetry-all",
            feature = "telemetry-effects"
        ))]
        {
            if let Some(ref sink) = self.budgets.telemetry_sink {
                if self.budgets.telemetry_mask.contains(crate::vm::telemetry::TelemetrySignalMask::EFFECTS) {
                    sink.span_exit(span_id, status);
                }
            }
        }
        let _ = (span_id, status);
    }

    #[inline]
    pub fn record_telemetry_counter(&self, name: &str, delta: u64, labels: &[(&str, &str)]) {
        #[cfg(any(
            feature = "telemetry",
            feature = "telemetry-all",
            feature = "telemetry-user"
        ))]
        {
            if let Some(ref sink) = self.budgets.telemetry_sink {
                if self.budgets.telemetry_mask.contains(crate::vm::telemetry::TelemetrySignalMask::USER) {
                    sink.counter_add(name, delta, labels);
                }
            }
        }
        let _ = (name, delta, labels);
    }

    #[inline]
    pub fn record_telemetry_gauge(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        #[cfg(any(
            feature = "telemetry",
            feature = "telemetry-all",
            feature = "telemetry-user"
        ))]
        {
            if let Some(ref sink) = self.budgets.telemetry_sink {
                if self.budgets.telemetry_mask.contains(crate::vm::telemetry::TelemetrySignalMask::USER) {
                    sink.gauge_set(name, value, labels);
                }
            }
        }
        let _ = (name, value, labels);
    }

    #[inline]
    pub fn record_telemetry_histogram(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        #[cfg(any(
            feature = "telemetry",
            feature = "telemetry-all",
            feature = "telemetry-user"
        ))]
        {
            if let Some(ref sink) = self.budgets.telemetry_sink {
                if self.budgets.telemetry_mask.contains(crate::vm::telemetry::TelemetrySignalMask::USER) {
                    sink.histogram_record(name, value, labels);
                }
            }
        }
        let _ = (name, value, labels);
    }

    #[inline]
    pub fn telemetry_span_enter(&self, name: &str, labels: &[(&str, &str)]) -> u64 {
        #[cfg(any(
            feature = "telemetry",
            feature = "telemetry-all",
            feature = "telemetry-user"
        ))]
        {
            if let Some(ref sink) = self.budgets.telemetry_sink {
                if self.budgets.telemetry_mask.contains(crate::vm::telemetry::TelemetrySignalMask::USER) {
                    return sink.span_enter(name, labels);
                }
            }
        }
        let _ = (name, labels);
        0
    }

    #[inline]
    pub fn telemetry_span_exit(&self, span_id: u64, status: crate::vm::telemetry::SpanStatus) {
        #[cfg(any(
            feature = "telemetry",
            feature = "telemetry-all",
            feature = "telemetry-user"
        ))]
        {
            if let Some(ref sink) = self.budgets.telemetry_sink {
                if self.budgets.telemetry_mask.contains(crate::vm::telemetry::TelemetrySignalMask::USER) {
                    sink.span_exit(span_id, status);
                }
            }
        }
        let _ = (span_id, status);
    }

    /// Whether the VM is currently idle (no executing call frame, native callback, or JIT code).
    #[inline]
    pub fn is_idle(&self) -> bool {
        self.call_stack.is_empty() && self.active_native_depth == 0 && self.active_jit_depth == 0
    }

    /// Whether JIT compilation and native execution are configured to be enabled.
    #[inline]
    pub fn jit_configured_enabled(&self) -> bool {
        self.jit.enabled
    }

    /// Whether JIT execution is effectively active.
    ///
    /// When a memory budget is active or Cranelift backend is selected, effective
    /// Dynasm JIT execution is bypassed.
    #[inline]
    pub fn effective_jit_enabled(&self) -> bool {
        self.jit.enabled
            && self.backend != VmBackend::Cranelift
            && !self.budgets.mem_budget_enabled()
            && self.quantum_fuel.is_none()
    }

    /// Sets the execution backend for VM runtime dispatch.
    #[inline]
    pub fn set_backend(&mut self, backend: VmBackend) {
        self.backend = backend;
    }

    /// Returns the current execution backend.
    #[inline]
    pub fn backend(&self) -> VmBackend {
        self.backend
    }

    /// Sets JIT backend (alias for `set_backend`).
    #[inline]
    pub fn set_jit_backend(&mut self, backend: VmBackend) {
        self.backend = backend;
    }

    /// Builder-style backend configuration.
    #[inline]
    pub fn with_backend(mut self, backend: VmBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Loads a MIR function into the VM, automatically generating its bytecode representation
    /// for interpreter oracle parity while registering it for native Cranelift JIT compilation.
    pub fn load_mir_function(&mut self, mir_fn: crate::mir::ir::MirFunction) -> Result<()> {
        let mut direct_callees: hashbrown::HashSet<String> =
            self.functions.iter().map(|f| f.name.clone()).collect();
        direct_callees.extend(self.mir_functions.keys().cloned());
        direct_callees.insert(mir_fn.signature.name.clone());
        self.load_mir_function_with_callees(mir_fn, &direct_callees)
    }

    fn load_mir_function_with_callees(
        &mut self,
        mir_fn: crate::mir::ir::MirFunction,
        direct_callees: &hashbrown::HashSet<String>,
    ) -> Result<()> {
        let name = mir_fn.signature.name.clone();
        let bytecode_fn = crate::mir::vm_backend::MirToBytecodeCompiler::compile_with_direct_callees(
            &mir_fn,
            direct_callees,
        )
        .map_err(|e| LustError::RuntimeError {
            message: format!("Failed to lower MIR function '{}' to bytecode: {:?}", name, e),
        })?;
        let meta = CallMeta::of(&bytecode_fn);
        let func_idx = if let Some(pos) = self.functions.iter().position(|f| f.name == name) {
            let mut bytecode_fn = bytecode_fn;
            bytecode_fn.is_method = self.functions[pos].is_method;
            self.functions[pos] = bytecode_fn;
            if pos < self.call_meta.len() {
                self.call_meta[pos] = meta;
            } else {
                self.call_meta.push(meta);
            }
            pos
        } else {
            let idx = self.functions.len();
            self.functions.push(bytecode_fn);
            self.call_meta.push(meta);
            idx
        };
        self.globals.insert(name.clone(), Value::Function(func_idx));
        self.globals_version = self.globals_version.wrapping_add(1);
        self.jit.reset_function_tables(self.functions.len());
        self.jit_cells.entry_table = self.jit.function_entry_table();
        self.mir_functions.insert(name, mir_fn);
        #[cfg(feature = "std")]
        {
            self.cranelift_jit = None;
            self.cranelift_compiled.clear();
        }
        Ok(())
    }

    /// Loads all functions from a MIR module into the VM.
    pub fn load_mir_module(&mut self, module: &crate::mir::ir::MirModule) -> Result<()> {
        let mut direct_callees: hashbrown::HashSet<String> =
            self.functions.iter().map(|f| f.name.clone()).collect();
        direct_callees.extend(self.mir_functions.keys().cloned());
        for mir_fn in &module.functions {
            direct_callees.insert(mir_fn.signature.name.clone());
        }
        for mir_fn in &module.functions {
            self.load_mir_function_with_callees(mir_fn.clone(), &direct_callees)?;
        }
        Ok(())
    }


    #[inline]
    pub(crate) fn enter_native_scope(&mut self) -> ActiveNativeGuard {
        self.active_native_depth = self.active_native_depth.saturating_add(1);
        ActiveNativeGuard {
            vm: self as *mut VM,
        }
    }

    #[inline]
    pub(crate) fn enter_jit_scope(&mut self) -> ActiveJitGuard {
        self.active_jit_depth = self.active_jit_depth.saturating_add(1);
        ActiveJitGuard {
            vm: self as *mut VM,
        }
    }

    pub(crate) fn invalidate_jit_code(&mut self) {
        self.trace_recorder = None;
        self.skip_next_trace_record = false;
        self.jit.invalidate_compiled_code();
        self.jit_cells.entry_table = self.jit.function_entry_table();
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CallFrame {
    pub function_idx: usize,
    pub ip: usize,
    /// In std mode this is a fixed [Value; 256] stored inline in the frame (no extra
    /// indirection, JIT-compatible raw pointer layout). In no_std mode (no JIT) it is a
    /// Vec sized to the function's actual register_count, saving the unused slots.
    #[cfg(feature = "std")]
    pub registers: [Value; 256],
    #[cfg(not(feature = "std"))]
    pub registers: Vec<Value>,
    #[allow(dead_code)]
    pub base_register: usize,
    pub return_dest: Option<Register>,
    pub upvalues: Vec<Value>,
    pub shared_upvalues: Vec<Upvalue>,
}

/// What a bytecode-to-bytecode call checks about its callee, computed
/// once per function instead of re-derived from the signature on every
/// call.
#[derive(Debug, Clone, Default)]
pub(super) struct CallMeta {
    /// Lua-compat function (`LuaValue` params or multi-return): arguments
    /// are padded/truncated rather than counted. Takes the general path.
    pub(super) lua_function: bool,
    /// Shallow kind check per parameter, when the signature is known and
    /// matches the parameter count.
    pub(super) params: Vec<ShallowKind>,
    /// Shallow kind check for the return value.
    pub(super) return_kind: ShallowKind,
    /// A Lua multi-return function may return Nil for "nothing".
    pub(super) lua_multi_return: bool,
}

/// The O(1) argument/return check for calls the typechecker validated:
/// scalar and container kinds are verified, contents are not. Anything the
/// checker treats dynamically (`unknown`, generics, unions, function types,
/// Lua values) is `Any`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum ShallowKind {
    #[default]
    Any,
    Int,
    Float,
    String,
    Bool,
    Unit,
    Array,
    Map,
    Tuple,
}

impl ShallowKind {
    pub(super) fn of(ty: &crate::ast::Type) -> Self {
        use crate::ast::TypeKind;
        match &ty.kind {
            TypeKind::Int => Self::Int,
            TypeKind::Float => Self::Float,
            TypeKind::String => Self::String,
            TypeKind::Bool => Self::Bool,
            TypeKind::Unit => Self::Unit,
            TypeKind::Array(_) => Self::Array,
            TypeKind::Map(..) => Self::Map,
            TypeKind::Tuple(_) => Self::Tuple,
            _ => Self::Any,
        }
    }

    #[inline]
    pub(super) fn matches(self, value: &Value) -> bool {
        match self {
            Self::Any => true,
            Self::Int => matches!(value, Value::Int(_)),
            Self::Float => matches!(value, Value::Float(_)),
            Self::String => matches!(value, Value::String(_)),
            Self::Bool => matches!(value, Value::Bool(_)),
            Self::Unit => matches!(value, Value::Nil),
            Self::Array => matches!(value, Value::Array(_)),
            Self::Map => matches!(value, Value::Map(_)),
            Self::Tuple => matches!(value, Value::Tuple(_)),
        }
    }
}

impl ShallowKind {
    /// The JIT's scalar type for this kind, when it is one.
    #[cfg_attr(
        not(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64"))),
        allow(dead_code)
    )]
    pub(super) fn value_type(self) -> Option<crate::jit::trace::ValueType> {
        use crate::jit::trace::ValueType;
        match self {
            Self::Int => Some(ValueType::Int),
            Self::Float => Some(ValueType::Float),
            Self::Bool => Some(ValueType::Bool),
            _ => None,
        }
    }
}

impl CallMeta {
    pub(super) fn of(function: &Function) -> Self {
        use crate::ast::TypeKind;
        let Some(signature) = &function.signature else {
            return Self {
                params: alloc::vec![ShallowKind::Any; function.param_count as usize],
                ..Self::default()
            };
        };
        let lua_params = signature
            .params
            .iter()
            .all(|ty| matches!(&ty.kind, TypeKind::Named(name) if name == "LuaValue"));
        let lua_multi_return = matches!(
            &signature.return_type.kind,
            TypeKind::Array(inner)
                if matches!(&inner.kind, TypeKind::Named(name) if name == "LuaValue")
        );
        let lua_function = lua_params && (!signature.params.is_empty() || lua_multi_return);
        let params = if signature.params.len() == function.param_count as usize {
            signature.params.iter().map(ShallowKind::of).collect()
        } else {
            alloc::vec![ShallowKind::Any; function.param_count as usize]
        };
        Self {
            lua_function,
            params,
            return_kind: ShallowKind::of(&signature.return_type),
            lua_multi_return,
        }
    }
}

/// Upper bound on recycled frames kept around (each holds 12 KB).
pub(super) const FRAME_POOL_LIMIT: usize = 64;

impl CallFrame {
    #[allow(unused_variables)]
    pub(super) fn new(
        function_idx: usize,
        return_dest: Option<Register>,
        register_count: u8,
    ) -> Box<Self> {
        Box::new(Self {
            function_idx,
            ip: 0,
            #[cfg(feature = "std")]
            registers: [const { Value::Nil }; 256],
            #[cfg(not(feature = "std"))]
            registers: alloc::vec![Value::Nil; register_count as usize],
            base_register: 0,
            return_dest,
            upvalues: Vec::new(),
            shared_upvalues: Vec::new(),
        })
    }

    /// Reset a frame that ran `function_idx` (which used the first
    /// `register_count` registers) so it can be handed out again.
    pub(super) fn reset(
        &mut self,
        function_idx: usize,
        return_dest: Option<Register>,
        register_count: u8,
    ) {
        for value in &mut self.registers[..register_count as usize] {
            if value.is_plain() {
                // SAFETY: nothing to drop.
                unsafe { core::ptr::write(value, Value::Nil) };
            } else {
                *value = Value::Nil;
            }
        }
        self.function_idx = function_idx;
        self.ip = 0;
        self.base_register = 0;
        self.return_dest = return_dest;
        self.upvalues.clear();
        self.shared_upvalues.clear();
    }
}

#[derive(Debug, Clone)]
pub(super) struct RuntimeStructInfo {
    pub layout: Rc<StructLayout>,
}

#[derive(Debug, Clone)]
pub(super) enum TaskSignal {
    Yield { dest: Register, value: Value },
    YieldTimer {
        dest: Register,
        value: Value,
        timer_id: crate::sim::clock::TimerId,
    },
    QuantumExhausted,
    Stop { value: Value },
}

impl Default for VM {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for VM {
    fn drop(&mut self) {
        // Track A: Immediately release all native host resources with 0 gas and 0 alloc
        self.host_ledger.cleanup_all();

        let mut collector = core::mem::take(&mut self.cycle_collector);

        // Drop every VM-owned root before the collector itself disappears so
        // closed Rc cycles are broken during runtime teardown. (Collecting
        // with the roots still in place would walk the whole live heap for
        // nothing: everything reachable from a root is kept anyway.)
        self.jit.traces.clear();
        self.functions.clear();
        self.natives.clear();
        self.globals.clear();
        self.globals_version = self.globals_version.wrapping_add(1);
        self.call_stack.clear();
        self.pending_return_value = None;
        self.pending_task_signal = None;
        self.last_task_signal = None;
        self.trace_recorder = None;
        self.nested_loop_exit_ip = None;
        self.task_manager = TaskManager::new();

        if collector.is_enabled() && collector.has_live_containers() {
            collector.collect(self);
        }
    }
}
