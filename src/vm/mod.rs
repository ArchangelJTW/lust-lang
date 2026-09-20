mod budget;
mod corelib;
mod cycle;
#[cfg(feature = "std")]
pub mod stdlib;
mod task;
use self::budget::BudgetState;
pub(super) use self::task::{TaskId, TaskInstance, TaskManager, TaskState};
pub(super) use crate::ast::{FieldOwnership, StructDef, Type, TypeKind};
pub(super) use crate::bytecode::{
    FieldStorage, Function, Instruction, NativeCallResult, Register, StructLayout, TaskHandle,
    Value,
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
pub(crate) use self::api::format_doc_comment;
pub use self::api::{NativeExport, NativeExportParam};
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

pub(crate) fn push_vm_ptr(vm: *mut VM) {
    with_vm_stack_mut(|stack| stack.push(vm));
}

pub(crate) fn pop_vm_ptr() {
    with_vm_stack_mut(|stack| {
        stack.pop();
    });
}

fn current_vm_ptr() -> Option<*mut VM> {
    with_vm_stack_ref(|stack| stack.last().copied())
}

struct CurrentVmGuard;

impl CurrentVmGuard {
    fn new(vm: *mut VM) -> Self {
        push_vm_ptr(vm);
        Self
    }
}

impl Drop for CurrentVmGuard {
    fn drop(&mut self) {
        pop_vm_ptr();
    }
}

pub(super) const TO_STRING_TRAIT: &str = "ToString";
pub(super) const TO_STRING_METHOD: &str = "to_string";
pub(super) const HASH_KEY_TRAIT: &str = "HashKey";
pub(super) const HASH_KEY_METHOD: &str = "to_hashkey";
pub struct VM {
    // Retain the host runtime's lookup function across dynamic-library boundaries.
    // Each extension can have its own copy of CURRENT_VM_STACK.
    current_vm_lookup: fn() -> Option<*mut VM>,
    pub(super) jit: JitState,
    /// Cells compiled code reads and writes through the VM pointer.
    pub(crate) jit_cells: crate::jit::JitCells,
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
    pub(super) pending_task_signal: Option<TaskSignal>,
    pub(super) last_task_signal: Option<TaskSignal>,
    pub(super) cycle_collector: cycle::CycleCollector,
    pub(super) exported_natives: Vec<NativeExport>,
    pub(super) export_prefix_stack: Vec<String>,
    #[cfg(feature = "std")]
    pub(super) exported_type_stubs: Vec<ModuleStub>,
}

#[derive(Debug, Clone)]
pub(super) struct CallFrame {
    pub(super) function_idx: usize,
    pub(super) ip: usize,
    /// In std mode this is a fixed [Value; 256] stored inline in the frame (no extra
    /// indirection, JIT-compatible raw pointer layout). In no_std mode (no JIT) it is a
    /// Vec sized to the function's actual register_count, saving the unused slots.
    #[cfg(feature = "std")]
    pub(super) registers: [Value; 256],
    #[cfg(not(feature = "std"))]
    pub(super) registers: Vec<Value>,
    #[allow(dead_code)]
    pub(super) base_register: usize,
    pub(super) return_dest: Option<Register>,
    pub(super) upvalues: Vec<Value>,
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
            registers: core::array::from_fn(|_| Value::Nil),
            #[cfg(not(feature = "std"))]
            registers: alloc::vec![Value::Nil; register_count as usize],
            base_register: 0,
            return_dest,
            upvalues: Vec::new(),
        })
    }

    /// Reset a frame that ran `function_idx` (which used the first
    /// `register_count` registers) so it can be handed out again.
    pub(super) fn reset(&mut self, function_idx: usize, return_dest: Option<Register>, register_count: u8) {
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
    }
}

#[derive(Debug, Clone)]
pub(super) struct RuntimeStructInfo {
    pub layout: Rc<StructLayout>,
}

#[derive(Debug, Clone)]
pub(super) enum TaskSignal {
    Yield { dest: Register, value: Value },
    Stop { value: Value },
}

impl Default for VM {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for VM {
    fn drop(&mut self) {
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

        collector.collect(self);
    }
}
