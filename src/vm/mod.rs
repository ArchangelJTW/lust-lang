mod cycle;
pub mod stdlib;
mod task;
pub(super) use self::task::{TaskId, TaskInstance, TaskManager, TaskState};
pub(super) use crate::ast::{FieldOwnership, StructDef};
pub(super) use crate::bytecode::{
    FieldStorage, Function, Instruction, NativeCallResult, Register, StructLayout, TaskHandle,
    Value,
};
pub(super) use crate::error::StackFrame;
pub(super) use crate::jit::{
    JitCompiler, JitState, TraceOptimizer, TraceRecorder, MAX_TRACE_LENGTH,
};
pub(super) use crate::{LustError, Result};
pub(super) use std::cell::RefCell;
pub(super) use std::collections::HashMap;
pub(super) use std::rc::Rc;
mod api;
mod execution;
mod tasks;
mod tracing;
pub use self::api::{NativeExport, NativeExportParam};
thread_local! {
    static CURRENT_VM_STACK: RefCell<Vec<*mut VM>> = RefCell::new(Vec::new());
}

pub(crate) fn push_vm_ptr(vm: *mut VM) {
    CURRENT_VM_STACK.with(|stack| {
        stack.borrow_mut().push(vm);
    });
}

pub(crate) fn pop_vm_ptr() {
    CURRENT_VM_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.pop();
    });
}

pub(super) const TO_STRING_TRAIT: &str = "ToString";
pub(super) const TO_STRING_METHOD: &str = "to_string";
pub struct VM {
    pub(super) functions: Vec<Function>,
    pub(super) natives: HashMap<String, Value>,
    pub(super) globals: HashMap<String, Value>,
    pub(super) call_stack: Vec<CallFrame>,
    pub(super) max_stack_depth: usize,
    pub(super) pending_return_value: Option<Value>,
    pub(super) pending_return_dest: Option<Register>,
    pub(super) jit: JitState,
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
