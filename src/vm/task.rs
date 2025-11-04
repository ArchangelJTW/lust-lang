use super::CallFrame;
use crate::bytecode::{Register, TaskHandle, Value};
use crate::error::LustError;
use alloc::{vec, vec::Vec};
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
    pub(super) call_stack: Vec<CallFrame>,
    pub pending_return_value: Option<Value>,
    pub pending_return_dest: Option<Register>,
    pub yield_dest: Option<Register>,
    pub last_yield: Option<Value>,
    pub last_result: Option<Value>,
    pub error: Option<LustError>,
    kind: TaskKind,
    initial_frame: Option<CallFrame>,
}

#[derive(Clone, Debug)]
pub enum TaskKind {
    Script,
    NativeFuture { future_id: Option<u64> },
}

impl TaskInstance {
    pub(super) fn new(id: TaskId, initial_frame: CallFrame) -> Self {
        Self {
            id,
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

    pub fn reset(&mut self) {
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
        self.initial_frame.as_ref()
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
