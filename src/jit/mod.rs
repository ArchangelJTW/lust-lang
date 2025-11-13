#[cfg(feature = "std")]
pub mod codegen;
pub mod optimizer;
pub mod profiler;
pub mod specialization;
pub mod trace;
use crate::bytecode::Value;
use crate::VM;
#[cfg(feature = "std")]
pub use codegen::JitCompiler;
#[cfg(not(feature = "std"))]
pub struct JitCompiler;
use alloc::{string::String, vec::Vec};
use hashbrown::HashMap;
pub use optimizer::TraceOptimizer;
pub use profiler::{HotSpot, Profiler};
pub use trace::{Trace, TraceOp, TraceRecorder};
#[cfg(not(feature = "std"))]
impl JitCompiler {
    pub fn new() -> Self {
        Self
    }

    pub fn compile_trace(
        &mut self,
        _trace: &Trace,
        _trace_id: TraceId,
        _parent: Option<TraceId>,
        _hoisted_constants: Vec<(u8, Value)>,
    ) -> crate::Result<CompiledTrace> {
        Err(crate::LustError::RuntimeError {
            message: "JIT is unavailable without the `std` feature".into(),
        })
    }
}
#[cfg(all(debug_assertions, feature = "std"))]
#[inline]
pub(crate) fn log<F>(message: F)
where
    F: FnOnce() -> String,
{
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
pub const MAX_TRACE_LENGTH: usize = 2000; // Increased to allow more loop unrolling
pub const SIDE_EXIT_THRESHOLD: u32 = 10;
pub const UNROLL_FACTOR: usize = 32;
/// How many times to unroll a loop during trace recording
pub const LOOP_UNROLL_COUNT: usize = 32;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraceId(pub usize);
pub struct CompiledTrace {
    pub id: TraceId,
    pub entry: extern "C" fn(*mut Value, *mut VM, *const crate::bytecode::Function) -> i32,
    pub trace: Trace,
    pub guards: Vec<Guard>,
    pub parent: Option<TraceId>,
    pub side_traces: Vec<TraceId>,
    pub leaked_constants: Vec<*const Value>,
    pub hoisted_constants: Vec<(u8, Value)>,
}

#[derive(Debug, Clone)]
pub struct Guard {
    pub index: usize,
    pub bailout_ip: usize,
    pub kind: GuardKind,
    pub fail_count: u32,
    pub side_trace: Option<TraceId>,
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
    Function {
        register: u8,
        function_idx: usize,
    },
    Closure {
        register: u8,
        function_idx: usize,
        upvalues_ptr: *const (),
    },
}

pub struct JitState {
    pub profiler: Profiler,
    pub traces: HashMap<TraceId, CompiledTrace>,
    pub root_traces: HashMap<(usize, usize), TraceId>,
    next_trace_id: usize,
    pub enabled: bool,
}

impl JitState {
    pub fn new() -> Self {
        let enabled = cfg!(all(feature = "std", target_arch = "x86_64"));
        Self {
            profiler: Profiler::new(),
            traces: HashMap::new(),
            root_traces: HashMap::new(),
            next_trace_id: 0,
            enabled,
        }
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
            .and_then(|id| self.traces.get(id))
    }

    pub fn get_trace(&self, id: TraceId) -> Option<&CompiledTrace> {
        self.traces.get(&id)
    }

    pub fn get_trace_mut(&mut self, id: TraceId) -> Option<&mut CompiledTrace> {
        self.traces.get_mut(&id)
    }

    pub fn store_root_trace(&mut self, func_idx: usize, ip: usize, trace: CompiledTrace) {
        let id = trace.id;
        self.root_traces.insert((func_idx, ip), id);
        self.traces.insert(id, trace);
    }

    pub fn store_side_trace(&mut self, trace: CompiledTrace) {
        let id = trace.id;
        self.traces.insert(id, trace);
    }
}

impl Default for JitState {
    fn default() -> Self {
        Self::new()
    }
}
