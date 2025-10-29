use super::CallFrame;
use crate::bytecode::{Register, TaskHandle, Value};
use crate::error::LustError;
use std::collections::HashMap;
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
    pub call_stack: Vec<CallFrame>,
    pub pending_return_value: Option<Value>,
    pub pending_return_dest: Option<Register>,
    pub yield_dest: Option<Register>,
    pub last_yield: Option<Value>,
    pub last_result: Option<Value>,
    pub error: Option<LustError>,
    initial_frame: CallFrame,
}

impl TaskInstance {
    pub fn new(id: TaskId, initial_frame: CallFrame) -> Self {
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
            initial_frame,
        }
    }

    pub fn reset(&mut self) {
        self.state = TaskState::Ready;
        self.call_stack = vec![self.initial_frame.clone()];
        self.pending_return_value = None;
        self.pending_return_dest = None;
        self.yield_dest = None;
        self.last_yield = None;
        self.last_result = None;
        self.error = None;
    }

    pub fn handle(&self) -> TaskHandle {
        self.id.to_handle()
    }

    pub(super) fn initial_frame(&self) -> &CallFrame {
        &self.initial_frame
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
