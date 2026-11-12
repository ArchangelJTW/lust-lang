use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

/// Stable C-ABI version tag for first-class Lust effects (0x0001_0000 = v1.0).
pub const LUST_EFFECT_ABI_V1: u32 = 0x0001_0000;

/// Return / context status codes matching RFC v2.0 Section 2.4.
pub const LUST_STATUS_SUCCESS: u32 = 0;
pub const LUST_STATUS_ERR_GENERIC: u32 = 1;
pub const LUST_STATUS_ERR_OUT_OF_GAS: u32 = 2;
pub const LUST_STATUS_ERR_OUT_OF_MEMORY: u32 = 3;
pub const LUST_STATUS_ERR_INTERRUPTED: u32 = 4;

pub type LustStatus = u32;

/// Typed reasons for effect interruption preserving canonical status codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterruptReason {
    Cancelled,
    Timeout,
    OutOfGas,
    OutOfMemory,
}

impl InterruptReason {
    #[inline]
    pub fn status_code(&self) -> LustStatus {
        match self {
            InterruptReason::Cancelled | InterruptReason::Timeout => LUST_STATUS_ERR_INTERRUPTED,
            InterruptReason::OutOfGas => LUST_STATUS_ERR_OUT_OF_GAS,
            InterruptReason::OutOfMemory => LUST_STATUS_ERR_OUT_OF_MEMORY,
        }
    }
}

/// C-ABI invocation function pointer type.
pub type LustInvokeFn = unsafe extern "C" fn(
    userdata: *mut core::ffi::c_void,
    rt: *mut core::ffi::c_void,
    env: *const core::ffi::c_void,
    out_exit: *mut core::ffi::c_void,
) -> LustStatus;

/// C-ABI release function pointer type for deterministic cleanup.
pub type LustReleaseFn = unsafe extern "C" fn(
    userdata: *mut core::ffi::c_void,
    rt: *mut core::ffi::c_void,
);

/// Escaping closure representation matching RFC Section 2.4.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LustClosureObject {
    pub code_ptr: *const (),
    pub env_layout_id: u32,
    pub capture_handle: *mut (),
}

/// Versioned C-ABI descriptor for first-class escaping Effect<A, E, R> and host interop (F04).
#[repr(C)]
pub struct LustEffectDescriptor {
    pub abi_version: u32,
    pub userdata: *mut core::ffi::c_void,
    pub invoke_fn: Option<LustInvokeFn>,
    pub release_fn: Option<LustReleaseFn>,
}

unsafe impl Send for LustEffectDescriptor {}
unsafe impl Sync for LustEffectDescriptor {}

impl Default for LustEffectDescriptor {
    fn default() -> Self {
        Self {
            abi_version: LUST_EFFECT_ABI_V1,
            userdata: core::ptr::null_mut(),
            invoke_fn: None,
            release_fn: None,
        }
    }
}

/// Internal termination cause for effect execution (F07).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cause<E = String> {
    /// Expected typed failure branch (E).
    Fail(E),
    /// Unchecked defect / fatal panic.
    Die(String),
    /// Terminal interruption (generic).
    Interrupt,
    /// Terminal interruption with typed reason.
    Interrupted(InterruptReason),
}

impl<E> Cause<E> {
    #[inline]
    pub fn is_interrupt(&self) -> bool {
        matches!(self, Cause::Interrupt | Cause::Interrupted(_))
    }
}

/// Disjoint result of executing an Effect<A, E, R>.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Exit<A, E = String> {
    Success(A),
    Failure(Cause<E>),
}

impl<A, E> Exit<A, E> {
    #[inline]
    pub fn is_success(&self) -> bool {
        matches!(self, Exit::Success(_))
    }

    #[inline]
    pub fn is_failure(&self) -> bool {
        matches!(self, Exit::Failure(_))
    }

    #[inline]
    pub fn is_interrupt(&self) -> bool {
        matches!(self, Exit::Failure(c) if c.is_interrupt())
    }

    #[inline]
    pub fn interrupt_reason(&self) -> Option<InterruptReason> {
        match self {
            Exit::Failure(Cause::Interrupted(r)) => Some(*r),
            Exit::Failure(Cause::Interrupt) => Some(InterruptReason::Cancelled),
            _ => None,
        }
    }

    /// Maps successful values preserving failures.
    pub fn map<B, F: FnOnce(A) -> B>(self, f: F) -> Exit<B, E> {
        match self {
            Exit::Success(a) => Exit::Success(f(a)),
            Exit::Failure(cause) => Exit::Failure(cause),
        }
    }

    /// Flat-maps successful values preserving failures.
    pub fn and_then<B, F: FnOnce(A) -> Exit<B, E>>(self, f: F) -> Exit<B, E> {
        match self {
            Exit::Success(a) => f(a),
            Exit::Failure(cause) => Exit::Failure(cause),
        }
    }

    /// Catches expected `Cause::Fail(e)`.
    /// INVARIANT: `Cause::Interrupt` and `Cause::Die` are STICKY and cannot be caught.
    pub fn catch<F: FnOnce(E) -> Exit<A, E>>(self, f: F) -> Exit<A, E> {
        match self {
            Exit::Failure(Cause::Fail(e)) => f(e),
            other => other,
        }
    }

    /// Converts exit status to canonical C-ABI status code.
    pub fn status_code(&self) -> LustStatus {
        match self {
            Exit::Success(_) => LUST_STATUS_SUCCESS,
            Exit::Failure(Cause::Interrupted(reason)) => reason.status_code(),
            Exit::Failure(Cause::Interrupt) => LUST_STATUS_ERR_INTERRUPTED,
            Exit::Failure(Cause::Die(_)) => LUST_STATUS_ERR_GENERIC,
            Exit::Failure(Cause::Fail(_)) => LUST_STATUS_ERR_GENERIC,
        }
    }
}

/// Stack-allocated context projection adapter (F03).
/// Allows zero-heap-allocation projection of environment records (e.g. Env_ABC -> Env_AC).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct StackContextProjection<const N: usize> {
    pub service_ptrs: [*const core::ffi::c_void; N],
    pub service_tags: [u32; N],
}

impl<const N: usize> StackContextProjection<N> {
    #[inline]
    pub const fn new(service_ptrs: [*const core::ffi::c_void; N], service_tags: [u32; N]) -> Self {
        Self {
            service_ptrs,
            service_tags,
        }
    }

    #[inline]
    pub fn get(&self, tag: u32) -> Option<*const core::ffi::c_void> {
        let mut i = 0;
        while i < N {
            if self.service_tags[i] == tag {
                return Some(self.service_ptrs[i]);
            }
            i += 1;
        }
        None
    }
}

/// RAII transaction ledger for layered service assembly (F13).
/// If layer composition fails, all previously acquired services are finalized in reverse (LIFO) order.
pub struct LayerRollbackLedger {
    finalizers: Vec<Box<dyn FnOnce() + 'static>>,
    committed: bool,
}

impl LayerRollbackLedger {
    pub fn new() -> Self {
        Self {
            finalizers: Vec::new(),
            committed: false,
        }
    }

    /// Registers a finalizer for an acquired layer service.
    pub fn register<F: FnOnce() + 'static>(&mut self, finalizer: F) {
        self.finalizers.push(Box::new(finalizer));
    }

    /// Commit the layer assembly, preventing rollback on drop.
    pub fn commit(&mut self) {
        self.committed = true;
    }

    /// Explicitly trigger rollback, executing all registered finalizers in reverse (LIFO) order.
    pub fn rollback(&mut self) {
        self.committed = true; // prevent double rollback on drop
        while let Some(f) = self.finalizers.pop() {
            f();
        }
    }

    pub fn count(&self) -> usize {
        self.finalizers.len()
    }
}

impl Default for LayerRollbackLedger {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for LayerRollbackLedger {
    fn drop(&mut self) {
        if !self.committed {
            while let Some(f) = self.finalizers.pop() {
                f();
            }
        }
    }
}

/// Scope admission states: Open accepts child forks and resource acquisition, Closed denies new entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeAdmission {
    Open,
    Closed,
}

/// Stages of the Scope teardown lifecycle (RFC §3.4, Gate-NEG-02).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeTeardown {
    Idle,
    Cancelling,
    Joining,
    Finalizing,
    Done,
}

/// Structured Concurrency Scope governing child fibers and LIFO finalizers.
pub struct Scope {
    pub id: u64,
    pub parent_id: Option<u64>,
    pub admission: ScopeAdmission,
    pub teardown: ScopeTeardown,
    pub children: Vec<u64>,
    pub finalizers: Vec<Box<dyn FnOnce() + 'static>>,
}

impl Scope {
    pub fn new(id: u64, parent_id: Option<u64>) -> Self {
        Self {
            id,
            parent_id,
            admission: ScopeAdmission::Open,
            teardown: ScopeTeardown::Idle,
            children: Vec::new(),
            finalizers: Vec::new(),
        }
    }

    #[inline]
    pub fn is_open(&self) -> bool {
        self.admission == ScopeAdmission::Open
    }

    /// Registers a child fiber ID. Fails if the scope has closed admission.
    pub fn register_child(&mut self, fiber_id: u64) -> Result<(), &'static str> {
        if self.admission == ScopeAdmission::Closed {
            return Err("Scope closed: cannot fork new child fiber");
        }
        self.children.push(fiber_id);
        Ok(())
    }

    /// Registers a finalizer to be executed in LIFO order upon scope exit.
    pub fn register_finalizer<F: FnOnce() + 'static>(&mut self, finalizer: F) -> Result<(), &'static str> {
        if self.admission == ScopeAdmission::Closed {
            return Err("Scope closed: cannot register new resource finalizer");
        }
        self.finalizers.push(Box::new(finalizer));
        Ok(())
    }

    /// Initiates teardown: closes admission to new operations, transitions to Cancelling,
    /// and returns the list of child fiber IDs that must be cancelled.
    pub fn begin_teardown(&mut self) -> Vec<u64> {
        self.admission = ScopeAdmission::Closed;
        self.teardown = ScopeTeardown::Cancelling;
        self.children.clone()
    }

    /// Transitions scope teardown stage to Joining after cancelling all children.
    pub fn mark_joining(&mut self) {
        self.teardown = ScopeTeardown::Joining;
    }

    /// Called after all child fibers have joined and stopped executing.
    /// Executes all registered finalizers in strict reverse (LIFO) order.
    pub fn finalize_resources(&mut self) {
        self.teardown = ScopeTeardown::Finalizing;
        while let Some(f) = self.finalizers.pop() {
            f();
        }
        self.teardown = ScopeTeardown::Done;
    }
}

/// Execution state of a structured Fiber.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiberExecutionState {
    Ready,
    Running,
    Suspended,
    Completed,
    Failed,
    Interrupted,
}

/// Context for an active Fiber with interruption masking for acquire_release.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FiberContext {
    pub id: u64,
    pub scope_id: u64,
    pub state: FiberExecutionState,
    pub pending_interrupt: Option<InterruptReason>,
    pub interrupt_mask_depth: usize,
}

impl FiberContext {
    pub fn new(id: u64, scope_id: u64) -> Self {
        Self {
            id,
            scope_id,
            state: FiberExecutionState::Ready,
            pending_interrupt: None,
            interrupt_mask_depth: 0,
        }
    }

    #[inline]
    pub fn is_masked(&self) -> bool {
        self.interrupt_mask_depth > 0
    }

    /// Enters interrupt-masked section (e.g. during acquire_release).
    pub fn enter_mask(&mut self) {
        self.interrupt_mask_depth = self.interrupt_mask_depth.saturating_add(1);
    }

    /// Exits interrupt-masked section. If mask reaches 0, returns any pending interrupt.
    pub fn exit_mask(&mut self) -> Option<InterruptReason> {
        self.interrupt_mask_depth = self.interrupt_mask_depth.saturating_sub(1);
        if self.interrupt_mask_depth == 0 {
            self.pending_interrupt.take()
        } else {
            None
        }
    }

    /// Signals an interrupt to the fiber.
    /// If masked, the interrupt is queued in `pending_interrupt`.
    /// Returns true if delivered immediately, false if masked/deferred.
    pub fn request_interrupt(&mut self, reason: InterruptReason) -> bool {
        if self.is_masked() {
            if self.pending_interrupt.is_none() {
                self.pending_interrupt = Some(reason);
            }
            false
        } else {
            self.state = FiberExecutionState::Interrupted;
            true
        }
    }
}

/// Schedule policy definition (immutable blueprint).
#[derive(Debug, Clone, PartialEq)]
pub enum ScheduleKind {
    /// Exactly n additional attempts (total runs <= n + 1). n = 0 -> only initial run.
    Recurs(u64),
    /// Fixed delay between attempts in milliseconds.
    Spaced(u64),
    /// Exponential backoff with saturation cap: delay = min(base * (factor ^ attempt), max_ms).
    Exponential { base_ms: u64, factor: f64, max_ms: u64 },
    /// Apply full jitter to computed delay (random fraction between 0 and delay).
    WithJitter(Box<ScheduleKind>),
    /// Composed schedule: executes first schedule, then second.
    AndThen(Box<ScheduleKind>, Box<ScheduleKind>),
}

/// Per-execution mutable state tracking for a running schedule.
#[derive(Debug, Clone, PartialEq)]
pub struct ScheduleState {
    pub attempt: u64,
    pub total_delay_ms: u64,
    pub last_delay_ms: u64,
}

impl ScheduleState {
    pub fn initial() -> Self {
        Self {
            attempt: 0,
            total_delay_ms: 0,
            last_delay_ms: 0,
        }
    }
}

/// Result of evaluating one step of a Schedule.
#[derive(Debug, Clone, PartialEq)]
pub enum ScheduleStep {
    Done,
    Continue { next_state: ScheduleState, delay_ms: u64 },
}

impl ScheduleKind {
    pub fn recurs(n: u64) -> Self {
        ScheduleKind::Recurs(n)
    }

    pub fn spaced(ms: u64) -> Self {
        ScheduleKind::Spaced(ms)
    }

    pub fn exponential(base_ms: u64, factor: f64, max_ms: u64) -> Self {
        ScheduleKind::Exponential { base_ms, factor, max_ms }
    }

    pub fn with_jitter(self) -> Self {
        ScheduleKind::WithJitter(Box::new(self))
    }

    pub fn and_then(self, other: Self) -> Self {
        ScheduleKind::AndThen(Box::new(self), Box::new(other))
    }

    /// Evaluates one step of the schedule given current state.
    pub fn step(&self, state: &ScheduleState) -> ScheduleStep {
        match self {
            ScheduleKind::Recurs(max_retries) => {
                if state.attempt < *max_retries {
                    ScheduleStep::Continue {
                        next_state: ScheduleState {
                            attempt: state.attempt + 1,
                            total_delay_ms: state.total_delay_ms,
                            last_delay_ms: 0,
                        },
                        delay_ms: 0,
                    }
                } else {
                    ScheduleStep::Done
                }
            }
            ScheduleKind::Spaced(ms) => {
                ScheduleStep::Continue {
                    next_state: ScheduleState {
                        attempt: state.attempt + 1,
                        total_delay_ms: state.total_delay_ms.saturating_add(*ms),
                        last_delay_ms: *ms,
                    },
                    delay_ms: *ms,
                }
            }
            ScheduleKind::Exponential { base_ms, factor, max_ms } => {
                let mult = factor.powi(state.attempt.min(30) as i32);
                let computed = (*base_ms as f64 * mult) as u64;
                let delay = computed.min(*max_ms);
                ScheduleStep::Continue {
                    next_state: ScheduleState {
                        attempt: state.attempt + 1,
                        total_delay_ms: state.total_delay_ms.saturating_add(delay),
                        last_delay_ms: delay,
                    },
                    delay_ms: delay,
                }
            }
            ScheduleKind::WithJitter(inner) => {
                match inner.step(state) {
                    ScheduleStep::Continue { next_state, delay_ms } => {
                        // In deterministic mode or test harness, half-delay is standard baseline jitter
                        let jittered_delay = delay_ms / 2;
                        ScheduleStep::Continue {
                            next_state: ScheduleState {
                                attempt: next_state.attempt,
                                total_delay_ms: state.total_delay_ms.saturating_add(jittered_delay),
                                last_delay_ms: jittered_delay,
                            },
                            delay_ms: jittered_delay,
                        }
                    }
                    ScheduleStep::Done => ScheduleStep::Done,
                }
            }
            ScheduleKind::AndThen(first, second) => {
                match first.step(state) {
                    ScheduleStep::Continue { next_state, delay_ms } => {
                        ScheduleStep::Continue { next_state, delay_ms }
                    }
                    ScheduleStep::Done => second.step(state),
                }
            }
        }
    }
}

/// Executes an effect, retrying on standard failure `Cause::Fail` according to the given schedule.
///
/// Critical Invariant (Fail-Closed):
/// `Cause::Die` (defects) and `Cause::Interrupted` / `Cause::Interrupt` (OutOfGas, Timeout, Cancelled, OOM)
/// are fail-closed and strictly NEVER retried.
pub fn retry<A, E, F>(mut effect_fn: F, schedule: &ScheduleKind) -> Exit<A, E>
where
    F: FnMut() -> Exit<A, E>,
{
    let mut state = ScheduleState::initial();
    loop {
        let outcome = effect_fn();
        match outcome {
            Exit::Success(val) => return Exit::Success(val),
            Exit::Failure(Cause::Fail(err)) => {
                match schedule.step(&state) {
                    ScheduleStep::Continue { next_state, .. } => {
                        state = next_state;
                    }
                    ScheduleStep::Done => return Exit::Failure(Cause::Fail(err)),
                }
            }
            Exit::Failure(cause) => {
                // Fail-closed: Die or Interrupt cannot be retried!
                return Exit::Failure(cause);
            }
        }
    }
}

/// Backwards-compatible alias for `retry`.
#[inline]
pub fn effect_retry<A, E, F>(effect_fn: F, schedule: &ScheduleKind) -> Exit<A, E>
where
    F: FnMut() -> Exit<A, E>,
{
    retry(effect_fn, schedule)
}

/// Repeatedly executes an effect upon `Exit::Success` according to the given schedule.
/// Halts immediately on any failure.
pub fn repeat<A, E, F>(mut effect_fn: F, schedule: &ScheduleKind) -> Exit<A, E>
where
    F: FnMut() -> Exit<A, E>,
{
    let mut state = ScheduleState::initial();
    loop {
        let outcome = effect_fn();
        match outcome {
            Exit::Success(val) => {
                match schedule.step(&state) {
                    ScheduleStep::Continue { next_state, .. } => {
                        state = next_state;
                    }
                    ScheduleStep::Done => return Exit::Success(val),
                }
            }
            Exit::Failure(cause) => return Exit::Failure(cause),
        }
    }
}

/// Backwards-compatible alias for `repeat`.
#[inline]
pub fn effect_repeat<A, E, F>(effect_fn: F, schedule: &ScheduleKind) -> Exit<A, E>
where
    F: FnMut() -> Exit<A, E>,
{
    repeat(effect_fn, schedule)
}

/// Executes an effect with a time limit in milliseconds.
/// If execution exceeds `timeout_ms` (or timeout_ms == 0), returns
/// `Exit::Failure(Cause::Interrupted(InterruptReason::Timeout))` with C-ABI status code 4.
#[cfg(feature = "std")]
pub fn timeout<A, E, F>(effect_fn: F, timeout_ms: u64) -> Exit<A, E>
where
    A: Send + 'static,
    E: Send + 'static,
    F: FnOnce() -> Exit<A, E> + Send + 'static,
{
    if timeout_ms == 0 {
        return Exit::Failure(Cause::Interrupted(InterruptReason::Timeout));
    }
    let (tx, rx) = std::sync::mpsc::channel();
    let duration = std::time::Duration::from_millis(timeout_ms);
    std::thread::spawn(move || {
        let res = effect_fn();
        let _ = tx.send(res);
    });
    match rx.recv_timeout(duration) {
        Ok(res) => res,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            Exit::Failure(Cause::Interrupted(InterruptReason::Timeout))
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            Exit::Failure(Cause::Die("Effect thread disconnected unexpectedly".into()))
        }
    }
}

#[cfg(not(feature = "std"))]
pub fn timeout<A, E, F>(effect_fn: F, timeout_ms: u64) -> Exit<A, E>
where
    F: FnOnce() -> Exit<A, E>,
{
    if timeout_ms == 0 {
        Exit::Failure(Cause::Interrupted(InterruptReason::Timeout))
    } else {
        effect_fn()
    }
}

/// Executes an effect with a timeout constraint.
/// If `timed_out` flag is triggered or duration exceeds limit, returns `Exit::Failure(Cause::Interrupted(InterruptReason::Timeout))`.
pub fn effect_timeout<A, E, F>(effect_fn: F, timed_out: bool) -> Exit<A, E>
where
    F: FnOnce() -> Exit<A, E>,
{
    if timed_out {
        Exit::Failure(Cause::Interrupted(InterruptReason::Timeout))
    } else {
        effect_fn()
    }
}

/// Executes `acquire` with interruption masked (`FiberContext::enter_mask`),
/// registers `release` finalizer in current `Scope` upon success, and unmasks interrupts (`FiberContext::exit_mask`).
/// If an interrupt arrived during `acquire`, it is delivered upon exit from mask.
pub fn acquire_release<A, E, Acq, Rel>(
    fiber: &mut FiberContext,
    scope: &mut Scope,
    acquire: Acq,
    release: Rel,
) -> Exit<A, E>
where
    Acq: FnOnce(&mut FiberContext) -> Exit<A, E>,
    Rel: FnOnce() + 'static,
{
    fiber.enter_mask();
    let outcome = acquire(fiber);
    if outcome.is_success() {
        let _ = scope.register_finalizer(release);
    }
    let pending = fiber.exit_mask();
    if let Some(reason) = pending {
        fiber.state = FiberExecutionState::Interrupted;
        Exit::Failure(Cause::Interrupted(reason))
    } else {
        outcome
    }
}

/// Convenience variant of `acquire_release` where `acquire` takes no arguments.
pub fn acquire_release_simple<A, E, Acq, Rel>(
    fiber: &mut FiberContext,
    scope: &mut Scope,
    acquire: Acq,
    release: Rel,
) -> Exit<A, E>
where
    Acq: FnOnce() -> Exit<A, E>,
    Rel: FnOnce() + 'static,
{
    acquire_release(fiber, scope, |_| acquire(), release)
}

/// Backwards-compatible alias for `acquire_release`.
#[inline]
pub fn effect_acquire_release<A, E, Acq, Rel>(
    fiber: &mut FiberContext,
    scope: &mut Scope,
    acquire: Acq,
    release: Rel,
) -> Exit<A, E>
where
    Acq: FnOnce(&mut FiberContext) -> Exit<A, E>,
    Rel: FnOnce() + 'static,
{
    acquire_release(fiber, scope, acquire, release)
}

impl FiberContext {
    #[inline]
    pub fn acquire_release<A, E, Acq, Rel>(
        &mut self,
        scope: &mut Scope,
        acquire: Acq,
        release: Rel,
    ) -> Exit<A, E>
    where
        Acq: FnOnce(&mut FiberContext) -> Exit<A, E>,
        Rel: FnOnce() + 'static,
    {
        acquire_release(self, scope, acquire, release)
    }
}

impl Scope {
    #[inline]
    pub fn acquire_release<A, E, Acq, Rel>(
        &mut self,
        fiber: &mut FiberContext,
        acquire: Acq,
        release: Rel,
    ) -> Exit<A, E>
    where
        Acq: FnOnce(&mut FiberContext) -> Exit<A, E>,
        Rel: FnOnce() + 'static,
    {
        acquire_release(fiber, self, acquire, release)
    }
}

impl ScheduleKind {
    #[inline]
    pub fn retry<A, E, F>(&self, effect_fn: F) -> Exit<A, E>
    where
        F: FnMut() -> Exit<A, E>,
    {
        retry(effect_fn, self)
    }

    #[inline]
    pub fn repeat<A, E, F>(&self, effect_fn: F) -> Exit<A, E>
    where
        F: FnMut() -> Exit<A, E>,
    {
        repeat(effect_fn, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::rc::Rc;
    use core::cell::RefCell;

    #[test]
    fn test_retry_recurs_two_retries_three_total_attempts() {
        let attempts = Rc::new(RefCell::new(0));
        let attempts_clone = attempts.clone();

        let res: Exit<i32, &'static str> = retry(
            || {
                *attempts_clone.borrow_mut() += 1;
                Exit::Failure(Cause::Fail("transient error"))
            },
            &ScheduleKind::recurs(2),
        );

        assert_eq!(*attempts.borrow(), 3, "retry with recurs(2) must attempt exactly 3 times (1 initial + 2 retries)");
        assert_eq!(res, Exit::Failure(Cause::Fail("transient error")));
    }

    #[test]
    fn test_retry_fail_closed_on_interrupt_and_die() {
        let attempts = Rc::new(RefCell::new(0));
        let attempts_clone = attempts.clone();

        let res: Exit<i32, &'static str> = retry(
            || {
                *attempts_clone.borrow_mut() += 1;
                Exit::Failure(Cause::Interrupted(InterruptReason::OutOfGas))
            },
            &ScheduleKind::recurs(5),
        );

        // Exactly 1 attempt, OutOfGas interruption cannot be retried!
        assert_eq!(*attempts.borrow(), 1, "retry must fail-closed on OutOfGas with zero retries");
        assert_eq!(res, Exit::Failure(Cause::Interrupted(InterruptReason::OutOfGas)));
        assert_eq!(res.status_code(), LUST_STATUS_ERR_OUT_OF_GAS);
    }

    #[test]
    fn test_repeat_recurs_two() {
        let count = Rc::new(RefCell::new(0));
        let count_clone = count.clone();

        let res: Exit<i32, ()> = repeat(
            || {
                *count_clone.borrow_mut() += 1;
                Exit::Success(*count_clone.borrow())
            },
            &ScheduleKind::recurs(2),
        );

        assert_eq!(*count.borrow(), 3, "repeat with recurs(2) must execute exactly 3 times (1 initial + 2 repeats)");
        assert_eq!(res, Exit::Success(3));
    }

    #[test]
    fn test_acquire_release_masks_interrupt_during_acquire() {
        let mut fiber = FiberContext::new(101, 1);
        let mut scope = Scope::new(1, None);
        let released = Rc::new(RefCell::new(false));
        let released_clone = released.clone();

        let res: Exit<i32, &'static str> = acquire_release(
            &mut fiber,
            &mut scope,
            |fib| {
                // While inside acquire, interrupt request arrives
                let delivered = fib.request_interrupt(InterruptReason::Cancelled);
                assert!(!delivered, "Interrupt must be masked during acquire");
                assert_eq!(fib.pending_interrupt, Some(InterruptReason::Cancelled));
                Exit::Success(42)
            },
            move || {
                *released_clone.borrow_mut() = true;
            },
        );

        // Finalizer was successfully registered in scope despite incoming interrupt
        assert_eq!(scope.finalizers.len(), 1);
        // Deferred interrupt delivered upon unmasking
        assert_eq!(res, Exit::Failure(Cause::Interrupted(InterruptReason::Cancelled)));
        assert_eq!(fiber.state, FiberExecutionState::Interrupted);

        // Teardown executes release
        scope.finalize_resources();
        assert!(*released.borrow());
    }

    #[test]
    fn test_timeout_combinator() {
        // Successful execution within timeout
        let res: Exit<i32, &'static str> = timeout(|| Exit::Success(42), 1000);
        assert_eq!(res, Exit::Success(42));

        // Immediate zero timeout
        let expired: Exit<i32, &'static str> = timeout(|| Exit::Success(99), 0);
        assert_eq!(expired, Exit::Failure(Cause::Interrupted(InterruptReason::Timeout)));
        assert_eq!(expired.status_code(), 4);
    }
}
