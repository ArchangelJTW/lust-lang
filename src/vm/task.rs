use super::CallFrame;
use crate::bytecode::{Register, TaskHandle, Value};
use crate::effect_abi::{FiberContext, FiberExecutionState, InterruptReason};
use crate::error::LustError;
use alloc::{boxed::Box, vec, vec::Vec};
use hashbrown::HashMap;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TaskId(pub u64);
impl TaskId {
    pub fn as_u64(self) -> u64 {
        self.0
    }

    pub fn to_handle(self) -> TaskHandle {
        TaskHandle(self.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TaskState {
    Ready,
    Running,
    Yielded,
    Completed,
    Failed,
    Stopped,
}

impl TaskState {
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskState::Ready => "ready",
            TaskState::Running => "running",
            TaskState::Yielded => "yielded",
            TaskState::Completed => "completed",
            TaskState::Failed => "failed",
            TaskState::Stopped => "stopped",
        }
    }
}

pub struct TaskInstance {
    pub id: TaskId,
    pub state: TaskState,
    pub timer_id: Option<crate::sim::clock::TimerId>,
    pub fiber_context: FiberContext,
    pub scope_id: Option<u64>,
    pub(super) call_stack: Vec<Box<CallFrame>>,
    pub pending_return_value: Option<Value>,
    pub pending_return_dest: Option<Register>,
    pub yield_dest: Option<Register>,
    pub last_yield: Option<Value>,
    pub last_result: Option<Value>,
    pub error: Option<LustError>,
    kind: TaskKind,
    initial_frame: Option<Box<CallFrame>>,
}

#[derive(Clone, Debug)]
pub enum TaskKind {
    Script,
    NativeFuture { future_id: Option<u64> },
}

impl TaskInstance {
    pub(super) fn new(id: TaskId, initial_frame: Box<CallFrame>) -> Self {
        Self {
            id,
            timer_id: None,
            fiber_context: FiberContext::new(id.0, 0),
            scope_id: None,
            state: TaskState::Ready,
            call_stack: vec![initial_frame.clone()],
            pending_return_value: None,
            pending_return_dest: None,
            yield_dest: None,
            last_yield: None,
            last_result: None,
            error: None,
            kind: TaskKind::Script,
            initial_frame: Some(initial_frame),
        }
    }

    pub(super) fn new_native_future(id: TaskId) -> Self {
        Self {
            id,
            timer_id: None,
            fiber_context: FiberContext::new(id.0, 0),
            scope_id: None,
            state: TaskState::Yielded,
            call_stack: Vec::new(),
            pending_return_value: None,
            pending_return_dest: None,
            yield_dest: None,
            last_yield: None,
            last_result: None,
            error: None,
            kind: TaskKind::NativeFuture { future_id: None },
            initial_frame: None,
        }
    }

    /// First interrupt wins; the context owns the pending reason, including while masked.
    pub fn request_interrupt(&mut self, reason: InterruptReason) {
        if matches!(
            self.state,
            TaskState::Completed | TaskState::Failed | TaskState::Stopped
        ) {
            return;
        }
        self.fiber_context.pending_interrupt.get_or_insert(reason);
        // Running tasks must unwind through the dispatch safe point before being joined.
        if self.state != TaskState::Running {
            self.poll_interrupt();
        }
    }

    pub fn poll_interrupt(&mut self) -> Option<InterruptReason> {
        if self.fiber_context.is_masked() {
            return None;
        }
        let reason = self.fiber_context.pending_interrupt.take()?;
        self.state = TaskState::Stopped;
        self.fiber_context.state = FiberExecutionState::Interrupted;
        self.error = Some(LustError::RuntimeError {
            message: alloc::format!("Task interrupted: {:?}", reason),
        });
        Some(reason)
    }

    pub fn reset(&mut self) {
        self.fiber_context = FiberContext::new(self.id.0, self.scope_id.unwrap_or(0));
        self.timer_id = None;
        match self.kind {
            TaskKind::Script => {
                self.state = TaskState::Ready;
                if let Some(frame) = self.initial_frame.clone() {
                    self.call_stack = vec![frame];
                }
                self.pending_return_value = None;
                self.pending_return_dest = None;
                self.yield_dest = None;
                self.last_yield = None;
                self.last_result = None;
                self.error = None;
            }
            TaskKind::NativeFuture { .. } => {
                self.state = TaskState::Yielded;
                self.call_stack.clear();
                self.pending_return_value = None;
                self.pending_return_dest = None;
                self.yield_dest = None;
                self.last_yield = None;
                self.last_result = None;
                self.error = None;
            }
        }
    }

    pub fn handle(&self) -> TaskHandle {
        self.id.to_handle()
    }

    pub(super) fn initial_frame(&self) -> Option<&CallFrame> {
        self.initial_frame.as_deref()
    }

    pub fn kind(&self) -> &TaskKind {
        &self.kind
    }

    pub fn kind_mut(&mut self) -> &mut TaskKind {
        &mut self.kind
    }
}

pub struct TaskManager {
    tasks: HashMap<TaskId, TaskInstance>,
    next_id: u64,
}

impl TaskManager {
    pub fn new() -> Self {
        Self {
            tasks: HashMap::new(),
            next_id: 1,
        }
    }

    pub fn insert(&mut self, task: TaskInstance) {
        self.tasks.insert(task.id, task);
    }

    pub fn detach(&mut self, id: TaskId) -> Option<TaskInstance> {
        self.tasks.remove(&id)
    }

    pub fn attach(&mut self, task: TaskInstance) {
        self.tasks.insert(task.id, task);
    }

    pub fn get(&self, id: TaskId) -> Option<&TaskInstance> {
        self.tasks.get(&id)
    }

    pub fn get_mut(&mut self, id: TaskId) -> Option<&mut TaskInstance> {
        self.tasks.get_mut(&id)
    }

    pub fn next_id(&mut self) -> TaskId {
        let id = TaskId(self.next_id);
        self.next_id += 1;
        id
    }

    pub fn contains(&self, id: TaskId) -> bool {
        self.tasks.contains_key(&id)
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = &TaskInstance> {
        self.tasks.values()
    }
}
