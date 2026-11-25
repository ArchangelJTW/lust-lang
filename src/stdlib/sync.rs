use alloc::{collections::VecDeque, vec::Vec};
use crate::{
    effect_abi::{Exit, FiberExecutionState},
    vm::{TaskId, TaskManager, TaskState, VM, WaitKey, YieldReason},
};

/// Deterministic, zero-OS-thread atomic-like reference cell.
#[derive(Clone, Debug)]
pub struct Ref<T>(T);

impl<T: Clone> Ref<T> {
    pub fn new(initial: T) -> Self {
        Self(initial)
    }

    pub fn get(&self) -> T {
        self.0.clone()
    }

    pub fn set(&mut self, val: T) {
        self.0 = val;
    }

    pub fn update<F: FnOnce(T) -> T>(&mut self, f: F) -> T {
        self.0 = f(self.0.clone());
        self.get()
    }

    pub fn modify<B, F: FnOnce(T) -> (B, T)>(&mut self, f: F) -> (B, T) {
        let (b, v) = f(self.0.clone());
        self.0 = v.clone();
        (b, v)
    }
}

/// Single-assignment promise backed by a stable WaitKey allocated/supplied by VM/task layer.
pub struct Deferred<T, E> {
    result: Option<Exit<T, E>>,
    key: WaitKey,
}

impl<T, E> Deferred<T, E> {
    pub fn new(key: WaitKey) -> Self {
        Self { result: None, key }
    }

    pub fn with_key(key: WaitKey) -> Self {
        Self { result: None, key }
    }

    pub fn new_allocated(tm: &mut TaskManager) -> Self {
        Self {
            result: None,
            key: tm.next_wait_key(),
        }
    }

    pub fn new_allocated_vm(vm: &mut VM) -> Self {
        Self {
            result: None,
            key: vm.next_wait_key(),
        }
    }

    pub fn key(&self) -> WaitKey {
        self.key
    }

    pub fn poll(&self) -> Option<&Exit<T, E>> {
        self.result.as_ref()
    }

    pub fn is_completed(&self) -> bool {
        self.result.is_some()
    }

    pub fn complete(&mut self, value: Exit<T, E>) -> bool {
        if self.result.is_some() {
            false
        } else {
            self.result = Some(value);
            true
        }
    }

    pub fn complete_and_wake(&mut self, value: Exit<T, E>, vm: &mut VM) -> (bool, Vec<TaskId>) {
        if self.complete(value) {
            let woken = vm.wake_all(self.key);
            (true, woken)
        } else {
            (false, Vec::new())
        }
    }

    pub fn complete_and_wake_tm(
        &mut self,
        value: Exit<T, E>,
        tm: &mut TaskManager,
    ) -> (bool, Vec<TaskId>) {
        if self.complete(value) {
            let woken = tm.wake(self.key, true);
            (true, woken)
        } else {
            (false, Vec::new())
        }
    }

    pub fn await_or_park(&self, _task_id: TaskId) -> Result<Option<&Exit<T, E>>, WaitKey> {
        match self.result.as_ref() {
            Some(v) => Ok(Some(v)),
            None => Err(self.key),
        }
    }

    pub fn await_or_park_vm<'a>(
        &'a self,
        task_id: TaskId,
        vm: &mut VM,
    ) -> Result<Option<&'a Exit<T, E>>, WaitKey> {
        if let Some(res) = self.result.as_ref() {
            return Ok(Some(res));
        }
        if vm.park_task(task_id, self.key) {
            // Race-safe re-check after parking: if completed in race before park, wake immediately.
            if let Some(res) = self.result.as_ref() {
                vm.wake_one(self.key);
                return Ok(Some(res));
            }
            Err(self.key)
        } else if let Some(res) = self.result.as_ref() {
            Ok(Some(res))
        } else {
            Err(self.key)
        }
    }

    pub fn await_or_park_tm<'a>(
        &'a self,
        task_id: TaskId,
        tm: &mut TaskManager,
    ) -> Result<Option<&'a Exit<T, E>>, WaitKey> {
        if let Some(res) = self.result.as_ref() {
            return Ok(Some(res));
        }
        if tm.park(task_id, self.key) {
            if let Some(res) = self.result.as_ref() {
                tm.wake(self.key, false);
                return Ok(Some(res));
            }
            Err(self.key)
        } else if let Some(res) = self.result.as_ref() {
            Ok(Some(res))
        } else {
            Err(self.key)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueueError {
    Closed,
}

pub struct BoundedQueue<T> {
    values: VecDeque<T>,
    capacity: usize,
    closed: bool,
    pub offer_key: WaitKey,
    pub take_key: WaitKey,
    pub offer_waiters: Vec<TaskId>,
    pub take_waiters: Vec<TaskId>,
}

impl<T> BoundedQueue<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            values: VecDeque::new(),
            capacity,
            closed: false,
            offer_key: WaitKey(1),
            take_key: WaitKey(2),
            offer_waiters: Vec::new(),
            take_waiters: Vec::new(),
        }
    }

    pub fn with_keys(capacity: usize, offer_key: WaitKey, take_key: WaitKey) -> Self {
        Self {
            values: VecDeque::new(),
            capacity,
            closed: false,
            offer_key,
            take_key,
            offer_waiters: Vec::new(),
            take_waiters: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.values.len() >= self.capacity
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }

    pub fn offer(&mut self, value: T) -> Result<bool, QueueError> {
        if self.closed {
            return Err(QueueError::Closed);
        }
        if self.values.len() >= self.capacity {
            return Ok(false);
        }
        self.values.push_back(value);
        Ok(true)
    }

    pub fn offer_tm(&mut self, value: T, tm: &mut TaskManager) -> Result<bool, QueueError> {
        if self.closed {
            return Err(QueueError::Closed);
        }
        if self.values.len() >= self.capacity {
            return Ok(false);
        }
        self.values.push_back(value);
        self.cleanup_dead_take_waiters(tm);
        while let Some(waiter_id) = self.take_waiters.first().copied() {
            self.take_waiters.remove(0);
            let woken = tm.wake(self.take_key, false);
            if woken.contains(&waiter_id) {
                break;
            }
        }
        Ok(true)
    }

    pub fn offer_vm(&mut self, value: T, vm: &mut VM) -> Result<bool, QueueError> {
        self.offer_tm(value, &mut vm.task_manager)
    }

    pub fn offer_or_park_tm(
        &mut self,
        task_id: TaskId,
        value: T,
        tm: &mut TaskManager,
    ) -> Result<bool, QueueError> {
        if self.closed {
            return Err(QueueError::Closed);
        }
        if self.offer_tm(value, tm)? {
            return Ok(true);
        }
        if !self.offer_waiters.contains(&task_id) {
            self.offer_waiters.push(task_id);
        }
        tm.park(task_id, self.offer_key);
        if self.closed {
            self.cancel_waiter(task_id);
            tm.wake(self.offer_key, false);
            return Err(QueueError::Closed);
        }
        Ok(false)
    }

    pub fn offer_or_park_vm(
        &mut self,
        task_id: TaskId,
        value: T,
        vm: &mut VM,
    ) -> Result<bool, QueueError> {
        self.offer_or_park_tm(task_id, value, &mut vm.task_manager)
    }

    pub fn take(&mut self) -> Result<Option<T>, QueueError> {
        if let Some(v) = self.values.pop_front() {
            Ok(Some(v))
        } else if self.closed {
            Err(QueueError::Closed)
        } else {
            Ok(None)
        }
    }

    pub fn take_tm(&mut self, tm: &mut TaskManager) -> Result<Option<T>, QueueError> {
        if let Some(v) = self.values.pop_front() {
            self.cleanup_dead_offer_waiters(tm);
            while let Some(waiter_id) = self.offer_waiters.first().copied() {
                self.offer_waiters.remove(0);
                let woken = tm.wake(self.offer_key, false);
                if woken.contains(&waiter_id) {
                    break;
                }
            }
            Ok(Some(v))
        } else if self.closed {
            Err(QueueError::Closed)
        } else {
            Ok(None)
        }
    }

    pub fn take_vm(&mut self, vm: &mut VM) -> Result<Option<T>, QueueError> {
        self.take_tm(&mut vm.task_manager)
    }

    pub fn take_or_park_tm(
        &mut self,
        task_id: TaskId,
        tm: &mut TaskManager,
    ) -> Result<Option<T>, QueueError> {
        if let Some(v) = self.take_tm(tm)? {
            return Ok(Some(v));
        }
        if self.closed {
            return Err(QueueError::Closed);
        }
        if !self.take_waiters.contains(&task_id) {
            self.take_waiters.push(task_id);
        }
        tm.park(task_id, self.take_key);
        if self.closed {
            self.cancel_waiter(task_id);
            tm.wake(self.take_key, false);
            return Err(QueueError::Closed);
        }
        if let Some(v) = self.take_tm(tm)? {
            self.cancel_waiter(task_id);
            tm.wake(self.take_key, false);
            return Ok(Some(v));
        }
        Ok(None)
    }

    pub fn take_or_park_vm(
        &mut self,
        task_id: TaskId,
        vm: &mut VM,
    ) -> Result<Option<T>, QueueError> {
        self.take_or_park_tm(task_id, &mut vm.task_manager)
    }

    pub fn cancel_waiter(&mut self, id: TaskId) {
        self.offer_waiters.retain(|x| *x != id);
        self.take_waiters.retain(|x| *x != id);
    }

    pub fn cancel_waiter_tm(&mut self, id: TaskId, tm: &mut TaskManager) {
        self.cancel_waiter(id);
        tm.unregister_task_from_waiters(id);
    }

    pub fn cancel_waiter_vm(&mut self, id: TaskId, vm: &mut VM) {
        self.cancel_waiter_tm(id, &mut vm.task_manager);
    }

    pub fn close(&mut self) -> Vec<TaskId> {
        self.closed = true;
        let mut out = self.offer_waiters.drain(..).collect::<Vec<_>>();
        out.extend(self.take_waiters.drain(..));
        out
    }

    pub fn close_tm(&mut self, tm: &mut TaskManager) -> Vec<TaskId> {
        self.closed = true;
        self.offer_waiters.clear();
        self.take_waiters.clear();
        let mut woken = tm.wake(self.offer_key, true);
        woken.extend(tm.wake(self.take_key, true));
        woken
    }

    pub fn close_vm(&mut self, vm: &mut VM) -> Vec<TaskId> {
        self.close_tm(&mut vm.task_manager)
    }

    fn cleanup_dead_offer_waiters(&mut self, tm: &TaskManager) {
        let key = self.offer_key;
        self.offer_waiters.retain(|&id| {
            tm.get(id).is_some_and(|t| {
                matches!(t.state, TaskState::Yielded(YieldReason::Parked(k)) if k == key)
                    && t.fiber_context.state != FiberExecutionState::Interrupted
            })
        });
    }

    fn cleanup_dead_take_waiters(&mut self, tm: &TaskManager) {
        let key = self.take_key;
        self.take_waiters.retain(|&id| {
            tm.get(id).is_some_and(|t| {
                matches!(t.state, TaskState::Yielded(YieldReason::Parked(k)) if k == key)
                    && t.fiber_context.state != FiberExecutionState::Interrupted
            })
        });
    }
}

pub struct Semaphore {
    permits: usize,
    key: WaitKey,
    waiters: Vec<(TaskId, usize)>,
}

impl Semaphore {
    pub fn new(permits: usize) -> Self {
        Self {
            permits,
            key: WaitKey(4),
            waiters: Vec::new(),
        }
    }

    pub fn with_key(permits: usize, key: WaitKey) -> Self {
        Self {
            permits,
            key,
            waiters: Vec::new(),
        }
    }

    pub fn key(&self) -> WaitKey {
        self.key
    }

    pub fn available_permits(&self) -> usize {
        self.permits
    }

    pub fn try_acquire(&mut self, n: usize) -> bool {
        if n <= self.permits {
            self.permits -= n;
            true
        } else {
            false
        }
    }

    pub fn acquire_or_park(&mut self, id: TaskId, n: usize) -> bool {
        if self.try_acquire(n) {
            true
        } else {
            self.waiters.push((id, n));
            false
        }
    }

    pub fn acquire_or_park_tm(&mut self, id: TaskId, n: usize, tm: &mut TaskManager) -> bool {
        self.cleanup_dead_waiters_tm(tm);
        if self.try_acquire(n) {
            return true;
        }
        self.waiters.push((id, n));
        tm.park(id, self.key);
        // Race-safe re-check after parking:
        if self.permits >= n && self.waiters.first().is_some_and(|(front_id, _)| *front_id == id) {
            self.waiters.remove(0);
            self.permits -= n;
            tm.wake(self.key, false);
            return true;
        }
        false
    }

    pub fn acquire_or_park_vm(&mut self, id: TaskId, n: usize, vm: &mut VM) -> bool {
        self.acquire_or_park_tm(id, n, &mut vm.task_manager)
    }

    pub fn release(&mut self, n: usize) -> Vec<TaskId> {
        self.permits = self.permits.saturating_add(n);
        let mut out = Vec::new();
        while self.waiters.first().is_some_and(|(_, need)| *need <= self.permits) {
            let (id, need) = self.waiters.remove(0);
            self.permits -= need;
            out.push(id);
        }
        out
    }

    pub fn release_tm(&mut self, n: usize, tm: &mut TaskManager) -> Vec<TaskId> {
        self.permits = self.permits.saturating_add(n);
        let mut woken = Vec::new();
        while let Some(&(id, need)) = self.waiters.first() {
            let is_alive = tm.get(id).is_some_and(|t| {
                matches!(t.state, TaskState::Yielded(YieldReason::Parked(k)) if k == self.key)
                    && t.fiber_context.state != FiberExecutionState::Interrupted
            });
            if !is_alive {
                // Cancelled or stopped waiter: remove WITHOUT consuming permits!
                self.waiters.remove(0);
                tm.unregister_task_from_waiters(id);
                continue;
            }
            if need <= self.permits {
                self.waiters.remove(0);
                self.permits -= need;
                if tm.wake(self.key, false).contains(&id) {
                    woken.push(id);
                }
            } else {
                break;
            }
        }
        woken
    }

    pub fn release_vm(&mut self, n: usize, vm: &mut VM) -> Vec<TaskId> {
        self.release_tm(n, &mut vm.task_manager)
    }

    pub fn cancel_waiter(&mut self, id: TaskId) {
        self.waiters.retain(|(x, _)| *x != id);
    }

    pub fn cancel_waiter_tm(&mut self, id: TaskId, tm: &mut TaskManager) {
        self.cancel_waiter(id);
        tm.unregister_task_from_waiters(id);
    }

    pub fn cancel_waiter_vm(&mut self, id: TaskId, vm: &mut VM) {
        self.cancel_waiter_tm(id, &mut vm.task_manager);
    }

    pub fn with_permit<R, F: FnOnce(&Self) -> R>(&mut self, n: usize, f: F) -> Option<R> {
        if !self.try_acquire(n) {
            return None;
        }
        let guard = core::mem::DropGuard::new(self, move |sem| {
            sem.release(n);
        });
        let res = f(&**guard);
        drop(guard);
        Some(res)
    }

    fn cleanup_dead_waiters_tm(&mut self, tm: &TaskManager) {
        let key = self.key;
        self.waiters.retain(|&(id, _)| {
            tm.get(id).is_some_and(|t| {
                matches!(t.state, TaskState::Yielded(YieldReason::Parked(k)) if k == key)
                    && t.fiber_context.state != FiberExecutionState::Interrupted
            })
        });
    }
}

pub struct Latch {
    count: usize,
    key: WaitKey,
    waiters: Vec<TaskId>,
}

impl Latch {
    pub fn new(count: usize) -> Self {
        Self {
            count,
            key: WaitKey(5),
            waiters: Vec::new(),
        }
    }

    pub fn with_key(count: usize, key: WaitKey) -> Self {
        Self {
            count,
            key,
            waiters: Vec::new(),
        }
    }

    pub fn is_open(&self) -> bool {
        self.count == 0
    }

    pub fn count_down(&mut self) -> Vec<TaskId> {
        self.count = self.count.saturating_sub(1);
        if self.count == 0 {
            core::mem::take(&mut self.waiters)
        } else {
            Vec::new()
        }
    }

    pub fn count_down_tm(&mut self, tm: &mut TaskManager) -> Vec<TaskId> {
        self.count = self.count.saturating_sub(1);
        if self.count == 0 {
            self.waiters.clear();
            tm.wake(self.key, true)
        } else {
            Vec::new()
        }
    }

    pub fn count_down_vm(&mut self, vm: &mut VM) -> Vec<TaskId> {
        self.count_down_tm(&mut vm.task_manager)
    }

    pub fn wait_or_park(&mut self, id: TaskId) -> bool {
        if self.is_open() {
            true
        } else {
            self.waiters.push(id);
            false
        }
    }

    pub fn wait_or_park_tm(&mut self, id: TaskId, tm: &mut TaskManager) -> bool {
        if self.is_open() {
            return true;
        }
        if !self.waiters.contains(&id) {
            self.waiters.push(id);
        }
        tm.park(id, self.key);
        if self.is_open() {
            tm.wake(self.key, false);
            return true;
        }
        false
    }

    pub fn wait_or_park_vm(&mut self, id: TaskId, vm: &mut VM) -> bool {
        self.wait_or_park_tm(id, &mut vm.task_manager)
    }
}
