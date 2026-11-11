//! RxScheduler coordinating virtual/physical execution, task queues, and participant join tracking.
//! Implements DST Phase 2 (D4) ScopeJoinCoordinator quiescence protocols (Case A & Case B).

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use super::clock::{PhysicalClock, SharedClock, VirtualClock};
use super::timer::RxTimerService;
use super::RxError;

/// Identifier for tracking active participants of a subscription scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParticipantId(pub u64);

/// Waiter lifecycle states for quiescence coordination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaiterState {
    Registered,
    Woken,
    TimedOut,
}

/// Outcome of polling a waiter during scheduler yield or quiescence check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaiterPollResult {
    /// Waiter completed with success (true) or timeout/failure (false).
    Completed(bool),
    /// Waiter remains registered and waiting.
    Pending,
    /// Waiter was not found in registry.
    NotFound,
}

/// Waiter entry in ScopeRegistry.
#[derive(Clone)]
pub struct WaiterRecord {
    pub waiter_id: u64,
    pub scope_id: u64,
    pub generation: u64,
    pub state: WaiterState,
    pub deadline_nanos: Option<u64>,
    pub timer_id: Option<super::timer::TimerId>,
    pub is_closed_scope: bool,
}

#[derive(Default)]
struct ScopeRegistry {
    participants: HashMap<u64, Vec<ParticipantId>>,
    closed_scopes: HashSet<u64>,
    tombstones: HashSet<u64>,
    waiters: HashMap<u64, WaiterRecord>,
}

/// Tracks active participants per scope to guarantee zero lingering callbacks on join.
pub struct ScopeJoinCoordinator {
    next_id: AtomicU64,
    next_waiter_id: AtomicU64,
    registry: Arc<Mutex<ScopeRegistry>>,
    condvar: Arc<Condvar>,
    pub clock: SharedClock,
    pub timers: Arc<RxTimerService>,
}

impl Default for ScopeJoinCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl ScopeJoinCoordinator {
    pub fn new() -> Self {
        let clock: SharedClock = Arc::new(PhysicalClock::new());
        let timers = Arc::new(RxTimerService::new());
        Self::new_with_clock_and_timers(clock, timers)
    }

    pub fn new_with_clock_and_timers(clock: SharedClock, timers: Arc<RxTimerService>) -> Self {
        Self {
            next_id: AtomicU64::new(1),
            next_waiter_id: AtomicU64::new(1),
            registry: Arc::new(Mutex::new(ScopeRegistry::default())),
            condvar: Arc::new(Condvar::new()),
            clock,
            timers,
        }
    }

    /// Register a new active participant for a given scope.
    /// Fails closed atomically if admission has already been closed or retired (tombstone) for this scope.
    pub fn register_participant(&self, scope_id: u64) -> Result<ParticipantId, RxError> {
        let mut reg = self.registry.lock().unwrap();
        if reg.closed_scopes.contains(&scope_id) || reg.tombstones.contains(&scope_id) {
            return Err(RxError::Cancelled);
        }
        let id = ParticipantId(self.next_id.fetch_add(1, Ordering::SeqCst));
        reg.participants.entry(scope_id).or_default().push(id);
        Ok(id)
    }

    /// Close admission for a scope atomically so that no new participants can register.
    pub fn close_admission(&self, scope_id: u64) {
        let mut reg = self.registry.lock().unwrap();
        reg.closed_scopes.insert(scope_id);
    }

    /// Arm/schedule a timer for the given waiter and generation.
    fn arm_timer(&self, waiter_id: u64, generation: u64, deadline_nanos: Option<u64>) {
        let dl = match deadline_nanos {
            Some(d) => d,
            None => return,
        };

        let reg = Arc::clone(&self.registry);
        let cv = Arc::clone(&self.condvar);

        let tid = self.timers.schedule(dl, generation as u32, move || {
            let mut should_notify = false;
            {
                let mut r = reg.lock().unwrap();
                if let Some(w) = r.waiters.get_mut(&waiter_id) {
                    if w.generation == generation && w.state == WaiterState::Registered {
                        w.state = WaiterState::TimedOut;
                        w.timer_id = None;
                        should_notify = true;
                    }
                }
            }
            if should_notify {
                cv.notify_all();
            }
        });

        let mut reg_lock = self.registry.lock().unwrap();
        if let Some(w) = reg_lock.waiters.get_mut(&waiter_id) {
            if w.generation == generation && w.state == WaiterState::Registered {
                w.timer_id = Some(tid);
            } else {
                // If waiter already woke up before timer was saved, cancel it immediately
                self.timers.cancel(tid, generation as u32);
            }
        }
    }

    /// Atomically close admission and wait until all registered participants have exited.
    /// Case A (Permanent Quiescence):
    /// 1. Under registry lock: closed_scopes.insert(scope_id) atomically.
    /// 2. If active_participants == 0 -> return true immediately.
    /// 3. Register WaiterEntry in state Registered with deadline.
    /// 4. Yield/wait until woken or timed out (retained wake checked before parking).
    /// 5. On participant exit: state -> Woken, cancel timer, notify condvar.
    /// 6. On timeout: state -> TimedOut, final check active == 0.
    /// 7. Closed tombstone preserved upon retirement.
    pub fn close_and_wait_quiescent(&self, scope_id: u64, timeout: Option<Duration>) -> bool {
        let waiter_id = self.next_waiter_id.fetch_add(1, Ordering::SeqCst);
        let generation = 1;
        let start_instant = Instant::now();

        let (immediate_return, deadline_nanos) = {
            let mut reg = self.registry.lock().unwrap();
            reg.closed_scopes.insert(scope_id);

            let active = reg.participants.get(&scope_id).map_or(0, |l| l.len());
            if active == 0 {
                return true;
            }

            let now = self.clock.now_nanos();
            let dl = timeout.map(|d| now.saturating_add(d.as_nanos().min(u64::MAX as u128) as u64));

            reg.waiters.insert(
                waiter_id,
                WaiterRecord {
                    waiter_id,
                    scope_id,
                    generation,
                    state: WaiterState::Registered,
                    deadline_nanos: dl,
                    timer_id: None,
                    is_closed_scope: true,
                },
            );
            self.condvar.notify_all();
            (false, dl)
        };

        if immediate_return {
            return true;
        }

        self.arm_timer(waiter_id, generation, deadline_nanos);

        loop {
            let mut reg = self.registry.lock().unwrap();
            let waiter = match reg.waiters.get(&waiter_id) {
                Some(w) => w.clone(),
                None => {
                    let active = reg.participants.get(&scope_id).map_or(0, |l| l.len());
                    return active == 0;
                }
            };

            match waiter.state {
                WaiterState::Woken => {
                    reg.waiters.remove(&waiter_id);
                    return true;
                }
                WaiterState::TimedOut => {
                    let active = reg.participants.get(&scope_id).map_or(0, |l| l.len());
                    reg.waiters.remove(&waiter_id);
                    return active == 0;
                }
                WaiterState::Registered => {
                    let clock_timed_out = if let Some(dl) = waiter.deadline_nanos {
                        self.clock.now_nanos() >= dl
                    } else {
                        false
                    };

                    let physical_timed_out = if let Some(limit) = timeout {
                        start_instant.elapsed() >= limit
                    } else {
                        false
                    };

                    if clock_timed_out || (!self.clock.is_virtual() && physical_timed_out) {
                        let active = reg.participants.get(&scope_id).map_or(0, |l| l.len());
                        if let Some(w) = reg.waiters.get_mut(&waiter_id) {
                            if let Some(tid) = w.timer_id.take() {
                                self.timers.cancel(tid, waiter.generation as u32);
                            }
                        }
                        reg.waiters.remove(&waiter_id);
                        return active == 0;
                    }

                    let wait_timeout = if let Some(limit) = timeout {
                        let elapsed = start_instant.elapsed();
                        if elapsed >= limit {
                            let active = reg.participants.get(&scope_id).map_or(0, |l| l.len());
                            reg.waiters.remove(&waiter_id);
                            return active == 0;
                        }
                        limit - elapsed
                    } else {
                        Duration::from_millis(50)
                    };

                    let (new_reg, _) = self
                        .condvar
                        .wait_timeout(reg, wait_timeout.min(Duration::from_millis(50)))
                        .unwrap();
                    drop(new_reg);
                }
            }
        }
    }

    /// Block or wait until all participants for the given open scope have exited.
    /// Case B (Transient Quiescence):
    /// 1. Registers waiter without closing admission.
    /// 2. Upon waking by Woken: MANDATORY re-check that active_participants == 0.
    ///    - If active == 0 -> returns true.
    ///    - If a new participant registered before resumption (active > 0):
    ///      IMMEDIATE RETURN true IS FORBIDDEN!
    ///      If deadline has expired -> returns false.
    ///      Otherwise: transitions back to Registered with incremented generation,
    ///      restores timer on the ORIGINAL ABSOLUTE DEADLINE (without extending duration!),
    ///      and re-parks atomically.
    pub fn wait_for_quiescence(&self, scope_id: u64, timeout: Option<Duration>) -> bool {
        let waiter_id = self.next_waiter_id.fetch_add(1, Ordering::SeqCst);
        let mut generation = 1;
        let start_instant = Instant::now();

        let (immediate_return, deadline_nanos) = {
            let mut reg = self.registry.lock().unwrap();
            let active = reg.participants.get(&scope_id).map_or(0, |l| l.len());
            if active == 0 {
                return true;
            }

            let now = self.clock.now_nanos();
            let dl = timeout.map(|d| now.saturating_add(d.as_nanos().min(u64::MAX as u128) as u64));

            reg.waiters.insert(
                waiter_id,
                WaiterRecord {
                    waiter_id,
                    scope_id,
                    generation,
                    state: WaiterState::Registered,
                    deadline_nanos: dl,
                    timer_id: None,
                    is_closed_scope: false,
                },
            );
            self.condvar.notify_all();
            (false, dl)
        };

        if immediate_return {
            return true;
        }

        self.arm_timer(waiter_id, generation, deadline_nanos);

        loop {
            let mut reg = self.registry.lock().unwrap();
            let waiter = match reg.waiters.get(&waiter_id) {
                Some(w) => w.clone(),
                None => {
                    let active = reg.participants.get(&scope_id).map_or(0, |l| l.len());
                    return active == 0;
                }
            };

            match waiter.state {
                WaiterState::Woken => {
                    // MANDATORY RE-CHECK: active_participants == 0
                    let active = reg.participants.get(&scope_id).map_or(0, |l| l.len());
                    if active == 0 {
                        reg.waiters.remove(&waiter_id);
                        return true;
                    }

                    // active > 0: a new participant registered before waiter resumed!
                    // IMMEDIATE RETURN true IS FORBIDDEN!
                    let clock_timed_out = if let Some(dl) = waiter.deadline_nanos {
                        self.clock.now_nanos() >= dl
                    } else {
                        false
                    };

                    let physical_timed_out = if let Some(limit) = timeout {
                        start_instant.elapsed() >= limit
                    } else {
                        false
                    };

                    if clock_timed_out || (!self.clock.is_virtual() && physical_timed_out) {
                        reg.waiters.remove(&waiter_id);
                        return false;
                    }

                    // Deadline has NOT expired: transition waiter back to Registered
                    generation += 1;
                    if let Some(w) = reg.waiters.get_mut(&waiter_id) {
                        w.state = WaiterState::Registered;
                        w.generation = generation;
                        w.timer_id = None;
                    }
                    drop(reg);

                    // Restore timer on the ORIGINAL ABSOLUTE DEADLINE (no extension!)
                    self.arm_timer(waiter_id, generation, deadline_nanos);
                    continue;
                }

                WaiterState::TimedOut => {
                    let active = reg.participants.get(&scope_id).map_or(0, |l| l.len());
                    reg.waiters.remove(&waiter_id);
                    return active == 0;
                }

                WaiterState::Registered => {
                    let clock_timed_out = if let Some(dl) = waiter.deadline_nanos {
                        self.clock.now_nanos() >= dl
                    } else {
                        false
                    };

                    let physical_timed_out = if let Some(limit) = timeout {
                        start_instant.elapsed() >= limit
                    } else {
                        false
                    };

                    if clock_timed_out || (!self.clock.is_virtual() && physical_timed_out) {
                        let active = reg.participants.get(&scope_id).map_or(0, |l| l.len());
                        if let Some(w) = reg.waiters.get_mut(&waiter_id) {
                            if let Some(tid) = w.timer_id.take() {
                                self.timers.cancel(tid, waiter.generation as u32);
                            }
                        }
                        reg.waiters.remove(&waiter_id);
                        return active == 0;
                    }

                    let wait_timeout = if let Some(limit) = timeout {
                        let elapsed = start_instant.elapsed();
                        if elapsed >= limit {
                            let active = reg.participants.get(&scope_id).map_or(0, |l| l.len());
                            reg.waiters.remove(&waiter_id);
                            return active == 0;
                        }
                        limit - elapsed
                    } else {
                        Duration::from_millis(50)
                    };

                    let (new_reg, _) = self
                        .condvar
                        .wait_timeout(reg, wait_timeout.min(Duration::from_millis(50)))
                        .unwrap();
                    drop(new_reg);
                }
            }
        }
    }

    /// Acknowledge that a participant has completely exited.
    pub fn acknowledge_exit(&self, scope_id: u64, participant_id: ParticipantId) {
        let mut timers_to_cancel = Vec::new();
        {
            let mut reg = self.registry.lock().unwrap();
            if let Some(list) = reg.participants.get_mut(&scope_id) {
                list.retain(|&p| p != participant_id);
                if list.is_empty() {
                    reg.participants.remove(&scope_id);
                }
            }
            let active = reg.participants.get(&scope_id).map_or(0, |l| l.len());
            if active == 0 {
                for waiter in reg.waiters.values_mut() {
                    if waiter.scope_id == scope_id && waiter.state == WaiterState::Registered {
                        waiter.state = WaiterState::Woken;
                        if let Some(tid) = waiter.timer_id.take() {
                            timers_to_cancel.push((tid, waiter.generation as u32));
                        }
                    }
                }
            }
        }
        for (tid, generation) in timers_to_cancel {
            self.timers.cancel(tid, generation);
        }
        self.condvar.notify_all();
    }

    /// Return count of currently active participants for a scope.
    pub fn active_participants(&self, scope_id: u64) -> usize {
        let reg = self.registry.lock().unwrap();
        reg.participants.get(&scope_id).map_or(0, |list| list.len())
    }

    /// Check if admission has been closed for a scope.
    pub fn is_admission_closed(&self, scope_id: u64) -> bool {
        let reg = self.registry.lock().unwrap();
        reg.closed_scopes.contains(&scope_id) || reg.tombstones.contains(&scope_id)
    }

    /// Retire scope from registry to bound historical records.
    /// Preserves closed tombstone to permanently reject subsequent registrations.
    pub fn retire_scope(&self, scope_id: u64) {
        let mut reg = self.registry.lock().unwrap();
        if reg.closed_scopes.contains(&scope_id) {
            reg.tombstones.insert(scope_id);
        }
        reg.closed_scopes.remove(&scope_id);
        reg.participants.remove(&scope_id);
    }

    /// Inspect waiter state for testing or scheduler integration.
    pub fn waiter_state(&self, waiter_id: u64) -> Option<WaiterState> {
        let reg = self.registry.lock().unwrap();
        reg.waiters.get(&waiter_id).map(|w| w.state)
    }

    /// Inspect waiter generation for testing.
    pub fn waiter_generation(&self, waiter_id: u64) -> Option<u64> {
        let reg = self.registry.lock().unwrap();
        reg.waiters.get(&waiter_id).map(|w| w.generation)
    }

    /// Register a waiter without blocking (returns waiter_id).
    pub fn register_waiter_for_test(
        &self,
        scope_id: u64,
        timeout: Option<Duration>,
        is_closed_scope: bool,
    ) -> (u64, bool) {
        let mut reg = self.registry.lock().unwrap();
        if is_closed_scope {
            reg.closed_scopes.insert(scope_id);
        }
        let active = reg.participants.get(&scope_id).map_or(0, |l| l.len());
        if active == 0 {
            return (0, true);
        }
        let waiter_id = self.next_waiter_id.fetch_add(1, Ordering::SeqCst);
        let now = self.clock.now_nanos();
        let dl = timeout.map(|d| now.saturating_add(d.as_nanos().min(u64::MAX as u128) as u64));

        reg.waiters.insert(
            waiter_id,
            WaiterRecord {
                waiter_id,
                scope_id,
                generation: 1,
                state: WaiterState::Registered,
                deadline_nanos: dl,
                timer_id: None,
                is_closed_scope,
            },
        );
        self.condvar.notify_all();
        drop(reg);

        self.arm_timer(waiter_id, 1, dl);
        (waiter_id, false)
    }

    /// Wait deterministically until a waiter for the specified scope is registered.
    pub fn wait_until_waiter_registered(&self, scope_id: u64) {
        let mut reg = self.registry.lock().unwrap();
        while !reg.waiters.values().any(|w| w.scope_id == scope_id) {
            reg = self.condvar.wait(reg).unwrap();
        }
    }

    /// Check and transition a waiter according to Quiescence rules (Case A & Case B).
    pub fn poll_waiter(&self, waiter_id: u64) -> WaiterPollResult {
        let mut reg = self.registry.lock().unwrap();
        let waiter = match reg.waiters.get(&waiter_id) {
            Some(w) => w.clone(),
            None => return WaiterPollResult::NotFound,
        };

        match waiter.state {
            WaiterState::Woken => {
                let active = reg.participants.get(&waiter.scope_id).map_or(0, |l| l.len());
                if active == 0 {
                    reg.waiters.remove(&waiter_id);
                    WaiterPollResult::Completed(true)
                } else {
                    // Case B Re-admission! active > 0: a new participant registered before resumption!
                    // IMMEDIATE RETURN true IS FORBIDDEN!
                    let now = self.clock.now_nanos();
                    let timed_out = waiter.deadline_nanos.map_or(false, |dl| now >= dl);
                    if timed_out {
                        reg.waiters.remove(&waiter_id);
                        WaiterPollResult::Completed(false)
                    } else {
                        // Restore timer on the ORIGINAL ABSOLUTE DEADLINE (no duration extension!)
                        let new_gen = waiter.generation + 1;
                        if let Some(w) = reg.waiters.get_mut(&waiter_id) {
                            w.state = WaiterState::Registered;
                            w.generation = new_gen;
                            w.timer_id = None;
                        }
                        let dl = waiter.deadline_nanos;
                        drop(reg);
                        self.arm_timer(waiter_id, new_gen, dl);
                        WaiterPollResult::Pending
                    }
                }
            }
            WaiterState::TimedOut => {
                let active = reg.participants.get(&waiter.scope_id).map_or(0, |l| l.len());
                reg.waiters.remove(&waiter_id);
                WaiterPollResult::Completed(active == 0)
            }
            WaiterState::Registered => {
                let now = self.clock.now_nanos();
                let timed_out = waiter.deadline_nanos.map_or(false, |dl| now >= dl);
                if timed_out {
                    let active = reg.participants.get(&waiter.scope_id).map_or(0, |l| l.len());
                    if let Some(w) = reg.waiters.get_mut(&waiter_id) {
                        if let Some(tid) = w.timer_id.take() {
                            let gen_u32 = w.generation as u32;
                            drop(reg);
                            self.timers.cancel(tid, gen_u32);
                            let mut reg2 = self.registry.lock().unwrap();
                            reg2.waiters.remove(&waiter_id);
                            return WaiterPollResult::Completed(active == 0);
                        }
                    }
                    reg.waiters.remove(&waiter_id);
                    WaiterPollResult::Completed(active == 0)
                } else {
                    WaiterPollResult::Pending
                }
            }
        }
    }
}

/// A scheduled task entry in the ready queue.
pub struct ReadyTask {
    pub sequence: u64,
    pub priority: u8,
    pub task: Box<dyn FnOnce() + Send + 'static>,
}

/// Core Scheduler coordinating clocks, timers, and cooperative task loops.
pub struct RxScheduler {
    pub clock: SharedClock,
    pub timers: Arc<RxTimerService>,
    pub coordinator: Arc<ScopeJoinCoordinator>,
    ready_tasks: Mutex<VecDeque<ReadyTask>>,
    max_ready_tasks: usize,
    next_task_seq: AtomicU64,
    stopped: AtomicBool,
}

impl RxScheduler {
    pub fn new_virtual() -> Self {
        Self::new_virtual_with_capacity(1024)
    }

    pub fn new_virtual_with_capacity(max_ready_tasks: usize) -> Self {
        let clock: SharedClock = Arc::new(VirtualClock::new(0));
        let timers = Arc::new(RxTimerService::new());
        let coordinator = Arc::new(ScopeJoinCoordinator::new_with_clock_and_timers(
            clock.clone(),
            timers.clone(),
        ));
        Self {
            clock,
            timers,
            coordinator,
            ready_tasks: Mutex::new(VecDeque::with_capacity(max_ready_tasks.min(4096))),
            max_ready_tasks,
            next_task_seq: AtomicU64::new(1),
            stopped: AtomicBool::new(false),
        }
    }

    pub fn new_with_clock(clock: SharedClock, max_ready_tasks: usize) -> Self {
        let timers = Arc::new(RxTimerService::new());
        let coordinator = Arc::new(ScopeJoinCoordinator::new_with_clock_and_timers(
            clock.clone(),
            timers.clone(),
        ));
        Self {
            clock,
            timers,
            coordinator,
            ready_tasks: Mutex::new(VecDeque::with_capacity(max_ready_tasks.min(4096))),
            max_ready_tasks,
            next_task_seq: AtomicU64::new(1),
            stopped: AtomicBool::new(false),
        }
    }

    /// Enqueue a cooperative task to run in the scheduler.
    /// Returns Err(RxError::BackpressureOverflow) if ready queue capacity is exceeded.
    pub fn schedule_task<F>(&self, priority: u8, task: F) -> Result<(), RxError>
    where
        F: FnOnce() + Send + 'static,
    {
        if self.stopped.load(Ordering::SeqCst) {
            return Err(RxError::Cancelled);
        }
        let sequence = self.next_task_seq.fetch_add(1, Ordering::SeqCst);
        let entry = ReadyTask {
            sequence,
            priority,
            task: Box::new(task),
        };
        let mut queue = self.ready_tasks.lock().unwrap();
        if queue.len() >= self.max_ready_tasks {
            return Err(RxError::BackpressureOverflow);
        }
        queue.push_back(entry);
        Ok(())
    }

    /// Execute a single pending task if available.
    /// Also drains and enqueues any expired timers.
    pub fn step(&self) -> bool {
        // Poll expired timers
        let now = self.clock.now_nanos();
        let expired = self.timers.poll_expired(now);
        for action in expired {
            let _ = self.schedule_task(10, action);
        }

        // Take the next ready task
        let next = {
            let mut queue = self.ready_tasks.lock().unwrap();
            queue.pop_front()
        };

        if let Some(task_entry) = next {
            (task_entry.task)();
            true
        } else {
            false
        }
    }

    /// Yield/wait until the specified waiter has been woken or timed out.
    /// Retained wake check: if already Woken or TimedOut, returns immediately without parking.
    /// Case B Re-check: if Woken but scope has active participants, re-arms waiter on original deadline.
    pub fn yield_until_woken(&self, waiter_id: u64) -> bool {
        loop {
            match self.coordinator.poll_waiter(waiter_id) {
                WaiterPollResult::Completed(success) => return success,
                WaiterPollResult::NotFound => return false,
                WaiterPollResult::Pending => {
                    if !self.step() {
                        if self.clock.is_virtual() {
                            if let Some(next_dl) = self.timers.next_deadline() {
                                let now = self.clock.now_nanos();
                                if next_dl > now {
                                    self.clock.advance(Duration::from_nanos(next_dl - now));
                                    continue;
                                } else if self.step() {
                                    continue;
                                }
                            }
                        }
                        break;
                    }
                }
            }
        }
        false
    }

    /// Drain all currently ready tasks and expired timers at the current clock time.
    pub fn drain_ready(&self) -> usize {
        let mut steps = 0;
        while self.step() {
            steps += 1;
        }
        steps
    }

    /// Advance virtual clock by the given duration and drain all expired timers and tasks.
    pub fn advance_and_drain(&self, duration: Duration) -> usize {
        self.clock.advance(duration);
        self.drain_ready()
    }

    /// Run the scheduler until there are no remaining ready tasks and no timers can be triggered.
    /// In a virtual clock environment, if there are pending timers, advances time to the next deadline.
    pub fn run_until_quiescent(&self, max_steps: usize) -> usize {
        let mut steps = 0;
        while steps < max_steps {
            if self.step() {
                steps += 1;
                continue;
            }

            // If no ready tasks, check if virtual clock can advance to next deadline
            if self.clock.is_virtual() {
                if let Some(next_deadline) = self.timers.next_deadline() {
                    let now = self.clock.now_nanos();
                    if next_deadline > now {
                        let advance_duration = Duration::from_nanos(next_deadline - now);
                        self.clock.advance(advance_duration);
                        if self.step() {
                            steps += 1;
                            continue;
                        }
                    }
                }
            }

            // Quiescent
            break;
        }
        steps
    }

    /// Stop accepting new work.
    pub fn shutdown(&self) {
        self.stopped.store(true, Ordering::SeqCst);
    }
}
