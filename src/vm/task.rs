use super::CallFrame;
use crate::bytecode::{Register, TaskHandle, Upvalue, Value};
use crate::effect_abi::{FiberContext, FiberExecutionState, InterruptReason};
use crate::error::LustError;
use alloc::{boxed::Box, rc::Rc, vec, vec::Vec};
use core::cell::RefCell;
use hashbrown::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TaskId(pub u64);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WaitKey(pub u64);

impl WaitKey {
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl From<u64> for WaitKey {
    fn from(val: u64) -> Self {
        WaitKey(val)
    }
}

impl TaskId {
    pub fn as_u64(self) -> u64 {
        self.0
    }

    pub fn to_handle(self) -> TaskHandle {
        TaskHandle(self.0)
    }
}

/// Reason for a fiber yielding execution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum YieldReason {
    /// Explicit cooperative yield via task.yield().
    Explicit,
    /// Yield waiting on virtual monotonic clock timer.
    Timer(crate::sim::clock::TimerId),
    /// Preempted by quantum instruction fuel exhaustion (DES).
    QuantumExhausted,
    Parked(WaitKey),
}

impl YieldReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            YieldReason::Explicit => "explicit",
            YieldReason::Timer(_) => "timer",
            YieldReason::QuantumExhausted => "quantum_exhausted",
            YieldReason::Parked(_) => "parked",
        }
    }
}

impl From<crate::sim::clock::TimerId> for YieldReason {
    fn from(id: crate::sim::clock::TimerId) -> Self {
        YieldReason::Timer(id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TaskState {
    Ready,
    Running,
    Yielded(YieldReason),
    Completed,
    Failed,
    Stopped,
}

impl TaskState {
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskState::Ready => "ready",
            TaskState::Running => "running",
            TaskState::Yielded(_) => "yielded",
            TaskState::Completed => "completed",
            TaskState::Failed => "failed",
            TaskState::Stopped => "stopped",
        }
    }

    pub fn is_yielded(&self) -> bool {
        matches!(self, TaskState::Yielded(_))
    }

    pub fn yield_reason(&self) -> Option<&YieldReason> {
        match self {
            TaskState::Yielded(r) => Some(r),
            _ => None,
        }
    }
}

/// Full resumable fiber continuation holding complete VM execution context.
#[derive(Clone, Debug, PartialEq)]
pub struct FiberContinuation {
    pub call_stack: Vec<Box<CallFrame>>,
    pub ip: usize,
    pub pending_return_value: Option<Value>,
    pub pending_return_dest: Option<Register>,
    pub yield_dest: Option<Register>,
    pub fiber_context: FiberContext,
    pub upvalues: Vec<Rc<RefCell<Upvalue>>>,
}

pub struct TaskInstance {
    pub id: TaskId,
    pub state: TaskState,
    pub timer_id: Option<crate::sim::clock::TimerId>,
    pub fiber_context: FiberContext,
    pub scope_id: Option<u64>,
    pub call_stack: Vec<Box<CallFrame>>,
    pub pending_return_value: Option<Value>,
    pub pending_return_dest: Option<Register>,
    pub yield_dest: Option<Register>,
    pub last_yield: Option<Value>,
    pub last_result: Option<Value>,
    pub error: Option<LustError>,
    pub continuation: Option<FiberContinuation>,
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
            continuation: None,
            kind: TaskKind::Script,
            initial_frame: Some(initial_frame),
        }
    }

    pub fn new_native_future(id: TaskId) -> Self {
        Self {
            id,
            timer_id: None,
            fiber_context: FiberContext::new(id.0, 0),
            scope_id: None,
            state: TaskState::Yielded(YieldReason::Explicit),
            call_stack: Vec::new(),
            pending_return_value: None,
            pending_return_dest: None,
            yield_dest: None,
            last_yield: None,
            last_result: None,
            error: None,
            continuation: None,
            kind: TaskKind::NativeFuture { future_id: None },
            initial_frame: None,
        }
    }

    /// Signals an interrupt to the task; non-maskable (OutOfGas/OutOfMemory) preempts immediately.
    pub fn request_interrupt(&mut self, reason: InterruptReason) {
        if matches!(
            self.state,
            TaskState::Completed | TaskState::Failed | TaskState::Stopped
        ) {
            return;
        }
        self.fiber_context.request_interrupt(reason);
        if self.fiber_context.pending_interrupt.is_none()
            || matches!(reason, InterruptReason::OutOfGas | InterruptReason::OutOfMemory)
        {
            self.fiber_context.pending_interrupt = Some(reason);
        }
        // Running tasks must unwind through the dispatch safe point before being joined.
        if self.state != TaskState::Running {
            self.poll_interrupt();
        }
    }

    pub fn poll_interrupt(&mut self) -> Option<InterruptReason> {
        if self.fiber_context.is_masked() {
            // INV-MASK-02: OutOfGas and OutOfMemory are non-maskable budget interrupts.
            if let Some(reason) = self.fiber_context.pending_interrupt {
                if matches!(reason, InterruptReason::OutOfGas | InterruptReason::OutOfMemory) {
                    self.fiber_context.pending_interrupt = None;
                    self.state = TaskState::Stopped;
                    self.fiber_context.state = FiberExecutionState::Interrupted;
                    self.error = Some(LustError::RuntimeError {
                        message: alloc::format!("Task interrupted: {:?}", reason),
                    });
                    return Some(reason);
                }
            }
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

    pub fn enter_mask(&mut self) {
        self.fiber_context.enter_mask();
    }

    pub fn exit_mask(&mut self) -> Option<InterruptReason> {
        if self.fiber_context.interrupt_mask_depth == 0 {
            return None;
        }
        let reason = self.fiber_context.exit_mask();
        if let Some(r) = reason {
            self.fiber_context.pending_interrupt = Some(r);
            if self.state != TaskState::Running {
                self.poll_interrupt();
            }
        }
        reason
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
                self.continuation = None;
            }
            TaskKind::NativeFuture { .. } => {
                self.state = TaskState::Yielded(YieldReason::Explicit);
                self.call_stack.clear();
                self.pending_return_value = None;
                self.pending_return_dest = None;
                self.yield_dest = None;
                self.last_yield = None;
                self.last_result = None;
                self.error = None;
                self.continuation = None;
            }
        }
    }

    pub fn continuation(&self) -> Option<&FiberContinuation> {
        self.continuation.as_ref()
    }

    pub fn take_continuation(&mut self) -> Option<FiberContinuation> {
        self.continuation.take()
    }

    pub fn capture_continuation(&mut self) -> FiberContinuation {
        let ip = self.call_stack.last().map(|f| f.ip).unwrap_or(0);
        let upvalues = self
            .call_stack
            .last()
            .map(|f| {
                f.shared_upvalues
                    .iter()
                    .cloned()
                    .map(|u| Rc::new(RefCell::new(u)))
                    .collect()
            })
            .unwrap_or_default();

        let cont = FiberContinuation {
            call_stack: self.call_stack.clone(),
            ip,
            pending_return_value: self.pending_return_value.clone(),
            pending_return_dest: self.pending_return_dest,
            yield_dest: self.yield_dest,
            fiber_context: self.fiber_context.clone(),
            upvalues,
        };
        self.continuation = Some(cont.clone());
        cont
    }

    pub fn restore_continuation(&mut self, continuation: FiberContinuation) {
        self.call_stack = continuation.call_stack;
        if let Some(top) = self.call_stack.last_mut() {
            top.ip = continuation.ip;
            if !continuation.upvalues.is_empty() {
                top.shared_upvalues = continuation
                    .upvalues
                    .into_iter()
                    .map(|u| u.borrow().clone())
                    .collect();
                top.upvalues = top.shared_upvalues.iter().map(|u| u.get()).collect();
            }
        }
        self.pending_return_value = continuation.pending_return_value;
        self.pending_return_dest = continuation.pending_return_dest;
        self.yield_dest = continuation.yield_dest;
        self.fiber_context = continuation.fiber_context;
        self.continuation = None;
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
    waiters: HashMap<WaitKey, Vec<TaskId>>,
    next_id: u64,
    next_wait_key: u64,
}

impl TaskManager {
    pub fn new() -> Self {
        Self {
            tasks: HashMap::new(),
            waiters: HashMap::new(),
            next_id: 1,
            next_wait_key: 1,
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

    pub fn next_wait_key(&mut self) -> WaitKey {
        let key = WaitKey(self.next_wait_key);
        self.next_wait_key += 1;
        key
    }

    pub fn contains(&self, id: TaskId) -> bool {
        self.tasks.contains_key(&id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &TaskInstance> {
        self.tasks.values()
    }

    pub fn park(&mut self, id: TaskId, key: WaitKey) -> bool {
        let Some(task) = self.tasks.get_mut(&id) else {
            return false;
        };
        if matches!(
            task.state,
            TaskState::Stopped | TaskState::Completed | TaskState::Failed
        ) {
            return false;
        }
        if task.fiber_context.state == FiberExecutionState::Interrupted {
            task.state = TaskState::Stopped;
            return false;
        }
        if !task.fiber_context.is_masked() && task.fiber_context.pending_interrupt.is_some() {
            task.poll_interrupt();
            return false;
        }
        if let Some(reason) = task.fiber_context.pending_interrupt {
            if matches!(reason, InterruptReason::OutOfGas | InterruptReason::OutOfMemory) {
                task.poll_interrupt();
                return false;
            }
        }
        task.continuation = Some(task.capture_continuation());
        task.state = TaskState::Yielded(YieldReason::Parked(key));
        let q = self.waiters.entry(key).or_default();
        if !q.contains(&id) {
            q.push(id);
        }
        true
    }

    pub fn wake(&mut self, key: WaitKey, all: bool) -> Vec<TaskId> {
        let mut woken = Vec::new();
        while let Some(id) = self.waiters.get_mut(&key).and_then(|q| {
            if q.is_empty() {
                None
            } else {
                Some(q.remove(0))
            }
        }) {
            let valid = self.tasks.get_mut(&id).is_some_and(|t| {
                if matches!(t.state, TaskState::Yielded(YieldReason::Parked(k)) if k == key)
                    && t.fiber_context.state != FiberExecutionState::Interrupted
                {
                    t.state = TaskState::Ready;
                    true
                } else {
                    false
                }
            });
            if valid {
                woken.push(id);
                if !all {
                    break;
                }
            }
        }
        if self.waiters.get(&key).is_some_and(Vec::is_empty) {
            self.waiters.remove(&key);
        }
        woken
    }

    pub fn unregister_task_from_waiters(&mut self, id: TaskId) {
        for q in self.waiters.values_mut() {
            q.retain(|waiter_id| *waiter_id != id);
        }
        self.waiters.retain(|_, q| !q.is_empty());
    }

    pub fn is_task_waiting(&self, id: TaskId) -> bool {
        self.waiters.values().any(|q| q.contains(&id))
    }

    pub fn cancel_task(&mut self, id: TaskId, reason: InterruptReason) {
        if let Some(task) = self.tasks.get_mut(&id) {
            task.request_interrupt(reason);
        }
        self.unregister_task_from_waiters(id);
    }
}

impl Default for TaskManager {
    fn default() -> Self {
        Self::new()
    }
}
