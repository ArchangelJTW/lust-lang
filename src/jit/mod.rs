#[cfg(all(feature = "std", target_arch = "x86_64"))]
pub mod codegen;
#[cfg(all(feature = "std", target_arch = "aarch64"))]
pub mod codegen_aarch64;
#[cfg(all(feature = "rv32", target_arch = "riscv32"))]
pub mod codegen_rv32;
pub mod function;
pub mod layout;
pub mod optimizer;
pub mod profiler;
pub mod specialization;
pub mod trace;
use crate::VM;
use crate::bytecode::Value;
#[cfg(all(feature = "std", target_arch = "x86_64"))]
pub use codegen::JitCompiler;
#[cfg(all(feature = "std", target_arch = "aarch64"))]
pub use codegen_aarch64::JitCompiler;
#[cfg(all(feature = "rv32", target_arch = "riscv32"))]
pub use codegen_rv32::JitCompiler;
#[cfg(not(any(
    all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")),
    all(feature = "rv32", target_arch = "riscv32")
)))]
pub struct JitCompiler;
use alloc::{boxed::Box, rc::Rc, string::String, vec::Vec};
use hashbrown::{HashMap, HashSet};
pub use optimizer::TraceOptimizer;
pub use profiler::{HotSpot, Profiler};
pub use trace::{Trace, TraceOp, TraceRecorder};
#[cfg(not(any(
    all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")),
    all(feature = "rv32", target_arch = "riscv32")
)))]
impl JitCompiler {
    pub fn new() -> Self {
        Self
    }

    /// Nothing is compiled here, so there are no loops to charge.
    pub fn with_gas_checks(self, _checked: bool) -> Self {
        self
    }

    pub fn compile_trace(
        &mut self,
        _trace: &Trace,
        _trace_id: TraceId,
        _hoisted_constants: Vec<(u8, Value)>,
    ) -> crate::Result<CompiledTrace> {
        Err(crate::LustError::RuntimeError {
            message: "JIT is unavailable: enable `std` (x86_64/aarch64) or `rv32` (riscv32)".into(),
        })
    }
}
#[cfg(all(debug_assertions, feature = "std"))]
#[inline]
pub(crate) fn log<F>(message: F)
where
    F: FnOnce() -> String,
{
    // Debug builds narrate the JIT; `LUST_JIT_QUIET=1` silences it (the
    // fuzzer's debug runs would otherwise drown in it).
    static QUIET: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *QUIET.get_or_init(|| std::env::var_os("LUST_JIT_QUIET").is_some()) {
        return;
    }
    println!("{}", message());
}

#[cfg(not(all(debug_assertions, feature = "std")))]
#[inline]
pub(crate) fn log<F>(_message: F)
where
    F: FnOnce() -> String,
{
}

pub const HOT_THRESHOLD: u32 = 5;
/// Calls to a bytecode function before its whole body is compiled.
pub const FUNCTION_HOT_THRESHOLD: u32 = 30;
/// Result of compiled function code entered from the interpreter whose
/// `Return` reached: `FUNCTION_RETURN_BASE + register` (255 = Nil).
pub const FUNCTION_RETURN_BASE: i32 = 1 << 20;
/// Result of compiled function code called natively by other compiled code
/// when it returned normally (anything else is an exit to propagate). Above
/// every `FUNCTION_RETURN_BASE + register`, and an aarch64 12-bit
/// immediate shifted by 12, so a caller compares against it in one
/// instruction.
pub const NATIVE_RETURNED: i32 = 1 << 23;
/// Runtime state compiled code reads and writes directly, addressed
/// through the VM pointer it holds (`VM::jit_cells`): a native call checks
/// the stack limit and depth budget, an exit records where it left, and a
/// helper call publishes its ip. One block per VM, so VMs on other
/// threads (the fuzzer's workers) never share it.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct JitCells {
    /// Lowest machine stack address native-to-native calls may grow to.
    /// Set by the VM before entering function code; a call that would go
    /// below it is handed to the interpreter instead. It is the lower of
    /// the native stack reserve and the interpreter's remaining frame
    /// depth times `MIN_NATIVE_FRAME`: a native call takes at least that
    /// much stack, so the depth limit cannot be passed natively (the
    /// interpreter, handed the call, counts the rest and raises the
    /// overflow).
    pub stack_limit: usize,
    /// Where the interpreter resumes after function code exits (any depth
    /// of native calls in): the exiting site's bytecode ip in the low 48
    /// bits, its kind in the high bits (`EXIT_KIND_*`). Set by the exit
    /// stubs of function code just before they leave.
    pub exit_info: usize,
    /// Bytecode ip of the call a trace is making through a runtime helper
    /// (`jit_call_function_safe` / `jit_call_native_safe`), stored just
    /// before the call so the caller's frame can show the right line in a
    /// stack trace; `usize::MAX` when unknown (a call from an inlined
    /// body, whose frame is not on the interpreter's stack). The helper
    /// resets it.
    pub call_ip: usize,
    /// The function entry table (`JitState::function_entries`): compiled
    /// code loads a callee's entry through it rather than embedding the
    /// table's address.
    pub entry_table: usize,
    /// Gas left before the budget is exhausted, while a gas budget is set:
    /// the VM computes it from the budget before entering native code and
    /// reads it back after. Code compiled under a budget charges each loop
    /// back-edge here and exits to the interpreter when it runs out, which
    /// then raises the error (code compiled without a budget does not
    /// check; setting a budget discards it).
    pub gas_left: u64,
}

impl Default for JitCells {
    fn default() -> Self {
        Self {
            stack_limit: 0,
            exit_info: usize::MAX,
            call_ip: usize::MAX,
            entry_table: 0,
            gas_left: u64::MAX,
        }
    }
}

/// Byte offsets of the cells from the VM pointer, for compiled code.
pub const CELLS_OFFSET: usize = core::mem::offset_of!(VM, jit_cells);
pub const STACK_LIMIT_OFFSET: usize = CELLS_OFFSET + core::mem::offset_of!(JitCells, stack_limit);
pub const EXIT_INFO_OFFSET: usize = CELLS_OFFSET + core::mem::offset_of!(JitCells, exit_info);
pub const CALL_IP_OFFSET: usize = CELLS_OFFSET + core::mem::offset_of!(JitCells, call_ip);
pub const ENTRY_TABLE_OFFSET: usize = CELLS_OFFSET + core::mem::offset_of!(JitCells, entry_table);
pub const GAS_LEFT_OFFSET: usize = CELLS_OFFSET + core::mem::offset_of!(JitCells, gas_left);
// Compiled code addresses the cells with 12-bit scaled immediates.
const _: () = assert!(GAS_LEFT_OFFSET < 32760);

impl VM {
    /// Take the call ip a trace stored for the helper call in progress.
    pub(crate) fn take_call_ip(&mut self) -> Option<usize> {
        let ip = core::mem::replace(&mut self.jit_cells.call_ip, usize::MAX);
        (ip != usize::MAX).then_some(ip)
    }
}

/// Machine stack kept free below the interpreter's entry into function
/// code before native calls hand over to the interpreter.
pub const NATIVE_STACK_RESERVE: usize = 1 << 20;
/// The least machine stack one native-to-native call level takes (its
/// record, a frame of at least one register, the callee's saved
/// registers), on either backend. Used to turn the interpreter's frame
/// depth limit into a stack address.
pub const MIN_NATIVE_FRAME: usize = 96;
pub const EXIT_KIND_SHIFT: u32 = 48;
/// A guard on something the code assumed: the function's code is evicted.
pub const EXIT_KIND_GUARD: usize = 0;
/// A call handed to the interpreter (no compiled callee / stack limit).
pub const EXIT_KIND_HANDOFF: usize = 1;
/// A failing op: the interpreter re-executes it and raises its error.
pub const EXIT_KIND_FAIL: usize = 2;
pub const MAX_TRACE_LENGTH: usize = 2000; // Increased to allow more loop unrolling
pub const UNROLL_FACTOR: usize = 32;
/// How many times to unroll a loop during trace recording
pub const LOOP_UNROLL_COUNT: usize = 32;
/// Cap on the recording-abort backoff, same shape as the eviction one
/// below. A loop the recorder can never trace (a global store, say) used to
/// be re-recorded every 32 back-edges for the life of the program, which
/// made recording overhead the main cost of running it.
const MAX_ROOT_RETRY_SHIFT: u32 = 16;
/// Cap on the eviction backoff: a root trace that keeps exiting is retried
/// after 1, 2, 4, ... 2^16 backedges, so a site that never stabilises costs
/// O(log n) compiles rather than one every few iterations.
const MAX_ROOT_EVICTION_SHIFT: u32 = 16;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraceId(pub usize);
pub struct CompiledTrace {
    pub id: TraceId,
    /// `(registers, vm, record, result)`: the record and the result slot
    /// are what a native caller passes (see `compile_call_direct`); the
    /// interpreter passes null for both.
    entry: extern "C" fn(*mut Value, *mut VM, *const crate::bytecode::Function, *mut Value) -> i32,
    #[cfg(any(
        all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")),
        all(feature = "rv32", target_arch = "riscv32")
    ))]
    _executable: dynasmrt::ExecutableBuffer,
    _data: Vec<JitData>,
    pub trace: Trace,
    pub guards: Vec<Guard>,
    /// Resume ips for failure exits: a trace result of `-(k + 2)` means the
    /// op recorded from bytecode ip `fail_sites[k]` failed; the interpreter
    /// re-executes that instruction. `-1` is a failure without a known site.
    pub fail_sites: Vec<usize>,
    pub hoisted_constants: Vec<(u8, Value)>,
}

impl CompiledTrace {
    pub fn execute(
        &self,
        registers: *mut Value,
        vm: *mut VM,
        function: *const crate::bytecode::Function,
    ) -> i32 {
        (self.entry)(registers, vm, function, core::ptr::null_mut())
    }
}

// Generated code embeds pointers into these allocations. Keeping each payload
// boxed makes those pointers independent of compiler and CompiledTrace moves.
#[allow(dead_code)]
pub(super) enum JitData {
    Value(Box<Value>),
    Name(crate::bytecode::value::Name),
    Key(Box<crate::bytecode::ValueKey>),
    CallSite(Box<crate::vm::JitCallSite>),
    String(Box<str>),
    StringPointers(Box<[*const u8]>),
    StringLengths(Box<[usize]>),
}

#[derive(Debug, Clone)]
pub struct Guard {
    pub index: usize,
    pub bailout_ip: usize,
    pub kind: GuardKind,
    pub fail_count: u32,
}

#[derive(Debug, Clone)]
pub enum GuardKind {
    IntType {
        register: u8,
    },
    FloatType {
        register: u8,
    },
    BoolType {
        register: u8,
    },
    Truthy {
        register: u8,
    },
    Falsy {
        register: u8,
    },
    /// The register holds nothing owned at trace entry (a preamble guard
    /// the optimizer adds for a register the loop overwrites with a
    /// scalar). Failing it means one interpreted iteration writes the
    /// scalar; the trace stays.
    Plain {
        register: u8,
    },
    ArrayBoundsCheck {
        array_register: u8,
        index_register: u8,
    },
    NestedLoop {
        function_idx: usize,
        loop_start_ip: usize,
    },
    NativeFunction {
        register: u8,
        expected: *const (),
    },
    /// `VM::globals_version` no longer matches the snapshots the trace took.
    Globals {
        version: u64,
    },
    StructLayout {
        register: u8,
        layout: *const (),
    },
    Function {
        register: u8,
        function_idx: usize,
    },
    /// Compiled function code handing a call to the interpreter (no compiled
    /// callee, or the native stack limit reached): resumes at the call.
    Call {
        function_idx: usize,
    },
    Closure {
        register: u8,
        function_idx: usize,
        upvalues_ptr: *const (),
    },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JitStats {
    pub recordings_started: u64,
    pub recordings_aborted: u64,
    pub root_traces_compiled: u64,
    pub functions_compiled: u64,
    pub native_trace_entries: u64,
    pub guard_exits: u64,
    pub execution_failures: u64,
    pub function_calls: u64,
    pub recursive_calls: u64,
}

/// A stdlib native the recorder knows the meaning of, so a call to it on a
/// specialized array becomes the matching `SpecializedOpKind` instead of
/// a native call the array would have to escape to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intrinsic {
    ArrayPush,
    ArrayLen,
}

pub struct JitState {
    pub profiler: Profiler,
    pub traces: HashMap<TraceId, Rc<CompiledTrace>>,
    pub root_traces: HashMap<(usize, usize), TraceId>,
    next_root_recording: HashMap<(usize, usize), u32>,
    root_recording_failures: HashMap<(usize, usize), u32>,
    /// Loop sites whose trace input arrays must not be specialized: a
    /// recording there mutated a specialized array and then let it escape
    /// (to a native, a call, a store), which the unboxed copy cannot follow.
    pub no_specialize_sites: HashSet<(usize, usize)>,
    /// Native functions the recorder may specialize, by `Rc` pointer.
    pub intrinsics: HashMap<usize, Intrinsic>,
    /// Times a compiled root trace at this site was evicted after a guard
    /// exit. Unlike recording failures this is never reset by a successful
    /// compile, so the retry delay keeps growing for a site that thrashes.
    root_evictions: HashMap<(usize, usize), u32>,
    next_trace_id: usize,
    pub enabled: bool,
    stats: JitStats,
    /// Whole-function code by function index (see `function`).
    function_code: Vec<Option<Rc<CompiledTrace>>>,
    /// Entry points of `function_code`, read by compiled callers at each
    /// `CallDirect` (0 = none). Allocated once per function table so the
    /// addresses compiled code embeds stay valid until the table is
    /// replaced, which invalidates all code.
    function_entries: Box<[core::sync::atomic::AtomicUsize]>,
    /// Calls seen per function, for the compile threshold.
    function_calls: Vec<u32>,
    /// Per function: `u32::MAX` = not compilable; otherwise the call count
    /// at which compiling may next be attempted.
    function_next_compile: Vec<u32>,
    function_evictions: Vec<u32>,
}

impl JitState {
    pub fn new() -> Self {
        let enabled = cfg!(all(
            feature = "std",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )) || cfg!(all(feature = "rv32", target_arch = "riscv32"));
        // `LUST_JIT=0` (or `off`) is a kill switch for differential testing
        // against the interpreter; it overrides every configuration path.
        #[cfg(feature = "std")]
        let enabled = enabled
            && !matches!(
                std::env::var("LUST_JIT").as_deref(),
                Ok("0") | Ok("off") | Ok("false")
            );
        Self {
            profiler: Profiler::new(),
            traces: HashMap::new(),
            root_traces: HashMap::new(),
            next_root_recording: HashMap::new(),
            root_recording_failures: HashMap::new(),
            no_specialize_sites: HashSet::new(),
            intrinsics: HashMap::new(),
            root_evictions: HashMap::new(),
            next_trace_id: 0,
            enabled,
            stats: JitStats::default(),
            function_code: Vec::new(),
            function_entries: Box::new([]),
            function_calls: Vec::new(),
            function_next_compile: Vec::new(),
            function_evictions: Vec::new(),
        }
    }

    /// Size the per-function tables for a new function table. Called by
    /// `VM::load_functions` after invalidation.
    pub(crate) fn reset_function_tables(&mut self, count: usize) {
        self.function_code = alloc::vec![None; count];
        self.function_entries = (0..count)
            .map(|_| core::sync::atomic::AtomicUsize::new(0))
            .collect();
        self.function_calls = alloc::vec![0; count];
        self.function_next_compile = alloc::vec![FUNCTION_HOT_THRESHOLD; count];
        self.function_evictions = alloc::vec![0; count];
    }

    /// One more function appended to the table (task wrappers).
    pub(crate) fn push_function_slot(&mut self) {
        let count = self.function_code.len() + 1;
        // Entry addresses embedded so far stay valid only if the table is
        // not moved; a grown table invalidates function code.
        self.function_code = alloc::vec![None; count];
        self.function_entries = (0..count)
            .map(|_| core::sync::atomic::AtomicUsize::new(0))
            .collect();
        self.function_calls = alloc::vec![0; count];
        self.function_next_compile = alloc::vec![FUNCTION_HOT_THRESHOLD; count];
        self.function_evictions = alloc::vec![0; count];
    }

    // Function code is compiled for std on x86_64 / aarch64 only.
    #[cfg_attr(
        not(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64"))),
        allow(dead_code)
    )]
    pub(crate) fn function_code(&self, func_idx: usize) -> Option<Rc<CompiledTrace>> {
        self.function_code.get(func_idx).and_then(|c| c.clone())
    }

    pub(crate) fn function_entry_table(&self) -> usize {
        self.function_entries.as_ptr() as usize
    }

    /// Count a call; true when the function should be compiled now.
    /// `LUST_JIT_NOFN=1` disables whole-function compilation (loop traces
    /// stay on), for bisecting.
    #[cfg_attr(
        not(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64"))),
        allow(dead_code)
    )]
    pub(crate) fn record_function_entry(&mut self, func_idx: usize) -> bool {
        #[cfg(feature = "std")]
        {
            static NOFN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            if *NOFN.get_or_init(|| std::env::var_os("LUST_JIT_NOFN").is_some()) {
                return false;
            }
        }
        let Some(count) = self.function_calls.get_mut(func_idx) else {
            return false;
        };
        *count = count.saturating_add(1);
        self.function_next_compile
            .get(func_idx)
            .is_some_and(|next| *next != u32::MAX && *count >= *next)
    }

    #[cfg_attr(
        not(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64"))),
        allow(dead_code)
    )]
    pub(crate) fn store_function_code(&mut self, func_idx: usize, code: CompiledTrace) {
        let entry = code.entry as usize;
        self.function_code[func_idx] = Some(Rc::new(code));
        self.function_entries[func_idx].store(entry, core::sync::atomic::Ordering::Release);
        self.stats.functions_compiled = self.stats.functions_compiled.saturating_add(1);
    }

    /// The function cannot be compiled: never try again.
    #[cfg_attr(
        not(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64"))),
        allow(dead_code)
    )]
    pub(crate) fn function_not_compilable(&mut self, func_idx: usize) {
        if let Some(next) = self.function_next_compile.get_mut(func_idx) {
            *next = u32::MAX;
        }
    }

    /// Forget a function's code after an unexpected guard failure; retried
    /// after a delay that doubles per eviction.
    #[cfg_attr(
        not(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64"))),
        allow(dead_code)
    )]
    pub(crate) fn evict_function_code(&mut self, func_idx: usize) {
        if let Some(slot) = self.function_code.get_mut(func_idx) {
            *slot = None;
        }
        if let Some(entry) = self.function_entries.get(func_idx) {
            entry.store(0, core::sync::atomic::Ordering::Release);
        }
        let evictions = &mut self.function_evictions[func_idx];
        *evictions = evictions.saturating_add(1);
        let delay = 1u32 << evictions.saturating_sub(1).min(MAX_ROOT_EVICTION_SHIFT);
        let count = self.function_calls[func_idx];
        self.function_next_compile[func_idx] =
            count.saturating_add(delay.max(FUNCTION_HOT_THRESHOLD));
    }

    pub fn alloc_trace_id(&mut self) -> TraceId {
        let id = TraceId(self.next_trace_id);
        self.next_trace_id += 1;
        id
    }

    pub fn check_hot(&mut self, func_idx: usize, ip: usize) -> bool {
        if !self.enabled {
            return false;
        }

        self.profiler.record_backedge(func_idx, ip) >= HOT_THRESHOLD
    }

    pub fn get_root_trace(&self, func_idx: usize, ip: usize) -> Option<&CompiledTrace> {
        self.root_traces
            .get(&(func_idx, ip))
            .and_then(|id| self.traces.get(id).map(Rc::as_ref))
    }

    pub fn get_trace(&self, id: TraceId) -> Option<&CompiledTrace> {
        self.traces.get(&id).map(Rc::as_ref)
    }

    pub fn trace_handle(&self, id: TraceId) -> Option<Rc<CompiledTrace>> {
        self.traces.get(&id).cloned()
    }

    pub fn get_trace_mut(&mut self, id: TraceId) -> Option<&mut CompiledTrace> {
        self.traces.get_mut(&id).and_then(Rc::get_mut)
    }

    pub fn store_root_trace(&mut self, func_idx: usize, ip: usize, trace: CompiledTrace) {
        let id = trace.id;
        if let Some(previous) = self.root_traces.insert((func_idx, ip), id) {
            self.traces.remove(&previous);
        }
        self.traces.insert(id, Rc::new(trace));
        self.next_root_recording.remove(&(func_idx, ip));
        self.root_recording_failures.remove(&(func_idx, ip));
        self.stats.root_traces_compiled = self.stats.root_traces_compiled.saturating_add(1);
    }

    /// Forget the root trace at a site because its guards keep failing.
    /// The compiled code is freed; the site is retried with a delay that
    /// doubles on each eviction.
    pub(crate) fn evict_root_trace(&mut self, func_idx: usize, ip: usize) {
        if let Some(id) = self.root_traces.remove(&(func_idx, ip)) {
            self.traces.remove(&id);
        }
        let count = self.profiler.get_count(func_idx, ip);
        let evictions = self
            .root_evictions
            .entry((func_idx, ip))
            .and_modify(|value| *value = value.saturating_add(1))
            .or_insert(1);
        let retry_delay = 1u32 << evictions.saturating_sub(1).min(MAX_ROOT_EVICTION_SHIFT);
        let next = count.saturating_add(retry_delay);
        let entry = self.next_root_recording.entry((func_idx, ip)).or_insert(0);
        *entry = (*entry).max(next);
    }

    pub fn stats(&self) -> JitStats {
        self.stats
    }

    pub(crate) fn record_function_call(&mut self, recursive: bool) {
        if !self.enabled {
            return;
        }
        self.stats.function_calls = self.stats.function_calls.saturating_add(1);
        if recursive {
            self.stats.recursive_calls = self.stats.recursive_calls.saturating_add(1);
        }
    }

    pub(crate) fn should_record_root(
        &self,
        func_idx: usize,
        ip: usize,
        count: u32,
        initial_threshold: u32,
    ) -> bool {
        count
            >= self
                .next_root_recording
                .get(&(func_idx, ip))
                .copied()
                .unwrap_or(initial_threshold)
    }

    pub(crate) fn recording_started(&mut self) {
        self.stats.recordings_started = self.stats.recordings_started.saturating_add(1);
    }

    pub(crate) fn recording_aborted(&mut self, func_idx: usize, ip: usize) {
        self.stats.recordings_aborted = self.stats.recordings_aborted.saturating_add(1);
        self.schedule_root_retry(func_idx, ip);
    }

    pub(crate) fn schedule_root_retry(&mut self, func_idx: usize, ip: usize) {
        let count = self.profiler.get_count(func_idx, ip);
        let failures = self
            .root_recording_failures
            .entry((func_idx, ip))
            .and_modify(|value| *value = value.saturating_add(1))
            .or_insert(1);
        let retry_delay = 1u32 << failures.saturating_sub(1).min(MAX_ROOT_RETRY_SHIFT);
        self.next_root_recording
            .insert((func_idx, ip), count.saturating_add(retry_delay));
    }

    pub(crate) fn record_native_entry(&mut self) {
        self.stats.native_trace_entries = self.stats.native_trace_entries.saturating_add(1);
    }

    pub(crate) fn record_guard_exit(&mut self) {
        self.stats.guard_exits = self.stats.guard_exits.saturating_add(1);
    }

    pub(crate) fn record_execution_failure(&mut self) {
        self.stats.execution_failures = self.stats.execution_failures.saturating_add(1);
    }

    pub(crate) fn invalidate_compiled_code(&mut self) {
        self.profiler.reset();
        self.traces.clear();
        self.root_traces.clear();
        self.next_root_recording.clear();
        self.root_recording_failures.clear();
        self.root_evictions.clear();
        self.next_trace_id = 0;
        self.stats = JitStats::default();
        let count = self.function_code.len();
        self.reset_function_tables(count);
    }
}

impl Default for JitState {
    fn default() -> Self {
        Self::new()
    }
}
