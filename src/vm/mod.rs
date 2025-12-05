mod corelib;
mod cycle;
#[cfg(feature = "std")]
pub mod stdlib;
mod task;
pub(super) use self::task::{TaskId, TaskInstance, TaskManager, TaskState};
pub(super) use crate::ast::{FieldOwnership, StructDef};
pub(super) use crate::bytecode::{
    FieldStorage, Function, Instruction, NativeCallResult, Register, StructLayout, TaskHandle,
    Value,
};
pub(super) use crate::embed::native_types::ModuleStub;
pub(super) use crate::error::StackFrame;
pub(super) use crate::jit::{
    JitCompiler, JitState, TraceOptimizer, TraceRecorder, MAX_TRACE_LENGTH,
};
pub(super) use crate::number::{
    float_abs, float_ceil, float_clamp, float_floor, float_from_int, float_round, float_sqrt,
    int_from_float, int_from_usize, LustFloat,
};
pub(super) use crate::{LustError, Result};
pub(super) use alloc::{
    format,
    rc::Rc,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::cell::RefCell;
use hashbrown::{DefaultHashBuilder, HashMap};
mod api;
mod execution;
mod tasks;
mod tracing;
pub use self::api::{NativeExport, NativeExportParam};
#[cfg(feature = "std")]
thread_local! {
    static CURRENT_VM_STACK: RefCell<Vec<*mut VM>> = RefCell::new(Vec::new());
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

#[cfg(feature = "std")]
pub(super) fn with_vm_stack<F, R>(f: F) -> R
where
    F: FnOnce(&Vec<*mut VM>) -> R,
{
    with_vm_stack_ref(f)
}

#[cfg(not(feature = "std"))]
pub(super) fn with_vm_stack<F, R>(f: F) -> R
where
    F: FnOnce(&Vec<*mut VM>) -> R,
{
    with_vm_stack_ref(f)
}

pub(super) const TO_STRING_TRAIT: &str = "ToString";
pub(super) const TO_STRING_METHOD: &str = "to_string";
pub(super) const HASH_KEY_TRAIT: &str = "HashKey";
pub(super) const HASH_KEY_METHOD: &str = "to_hashkey";
pub struct VM {
    pub(super) jit: JitState,
    pub(super) functions: Vec<Function>,
    pub(super) natives: HashMap<String, Value>,
    pub(super) globals: HashMap<String, Value>,
    pub(super) map_hasher: DefaultHashBuilder,
    pub(super) call_stack: Vec<CallFrame>,
    pub(super) max_stack_depth: usize,
    pub(super) pending_return_value: Option<Value>,
    pub(super) pending_return_dest: Option<Register>,
    pub(super) trace_recorder: Option<TraceRecorder>,
    pub(super) side_trace_context: Option<(crate::jit::TraceId, usize)>,
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
    pub(super) exported_type_stubs: Vec<ModuleStub>,
}

#[derive(Debug, Clone)]
pub(super) struct CallFrame {
    pub(super) function_idx: usize,
    pub(super) ip: usize,
    pub(super) registers: [Value; 256],
    #[allow(dead_code)]
    pub(super) base_register: usize,
    pub(super) return_dest: Option<Register>,
    pub(super) upvalues: Vec<Value>,
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
