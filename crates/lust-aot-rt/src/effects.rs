//! Stack-scoped effect handler runtime, nominal service providers (`NG-W13-1A`),
//! one-shot non-tail `EffectYield<OpArg, ResumeVal, BodyRet>` state-machine support,
//! and 3-phase LIFO structured concurrency `AotScope` (`NG-W13-1B`).
//!
//! 100% `#![no_std] + alloc`, zero `unsafe` (`TASK_ALLOWS_UNSAFE=0`), and panic-free (Hazard 9).
//!
//! - Tail-resumptive and abortive `handle .. with` blocks push an [`EffectHandlerFrame`] onto
//!   [`AotCtx::effects`] protected by [`core::mem::DropGuard`], guaranteeing frame removal on
//!   normal completion, abort unwind, `?` error propagation, or early `return`.
//! - One-shot non-tail `handle .. with` blocks use the fully parameterized
//!   [`EffectYield<OpArg, ResumeVal, BodyRet>`] two-phase continuation protocol
//!   ([`handle_non_tail_oneshot`] + [`OneShotResume`]), supporting arbitrary types (`i64`, `f64`,
//!   `bool`, `String`, `()`, [`AotValue`], and user aggregates).
//! - Structured concurrency (`StmtKind::Scoped` / `ExprKind::EffectBlock`) runs under
//!   [`AotScope`] with strict 3-phase teardown ordering:
//!   `Cancelling -> Joining -> Finalizing` (LIFO finalizers with first-defect error aggregation).
//! - Stateless nominal services instantiate via safe `Default::default()` ZST providers
//!   ([`default_zst_provider`]), never `unsafe fn core::mem::conjure_zst`.

use crate::alloc_crate::borrow::Cow;
use crate::alloc_crate::boxed::Box;
use crate::alloc_crate::collections::BTreeMap;
use crate::alloc_crate::rc::Rc;
use crate::alloc_crate::string::String;
use crate::alloc_crate::vec::Vec;
use crate::value::{AotClosure, AotValue};
use crate::{AotCtx, AotError};
use core::cell::{Cell, RefCell};
use core::fmt;
use core::future::Future;
use core::mem::DropGuard;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};

/// Safe zero-cost instantiation helper for stateless ZST nominal service providers (`F-11`).
///
/// Replaces `unsafe fn core::mem::conjure_zst::<T>()` with 100% safe `Default::default()`.
#[inline(always)]
pub fn default_zst_provider<P: Default + Copy>() -> P {
    debug_assert_eq!(core::mem::size_of::<P>(), 0);
    P::default()
}

/// A single `with Op(args) => ...` handler arm bound inside an [`EffectHandlerFrame`].
#[derive(Clone)]
pub struct EffectHandlerArm {
    pub effect_name: Cow<'static, str>,
    pub handler: AotClosure,
}

impl EffectHandlerArm {
    /// Creates an arm from a pre-wrapped [`AotClosure`].
    #[inline]
    pub fn new(effect_name: impl Into<Cow<'static, str>>, handler: AotClosure) -> Self {
        Self {
            effect_name: effect_name.into(),
            handler,
        }
    }

    /// Creates an arm from any `'static` closure without panicking.
    #[inline]
    pub fn from_fn<F>(effect_name: impl Into<Cow<'static, str>>, f: F) -> Self
    where
        F: Fn(&mut AotCtx, &[AotValue]) -> Result<AotValue, AotError> + 'static,
    {
        Self {
            effect_name: effect_name.into(),
            handler: Rc::new(f),
        }
    }
}

impl fmt::Debug for EffectHandlerArm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EffectHandlerArm")
            .field("effect_name", &self.effect_name)
            .finish()
    }
}

/// A stack-scoped handler frame stored in [`AotCtx::effects`] while a `handle .. with` body runs.
#[derive(Debug)]
pub struct EffectHandlerFrame {
    pub id: u64,
    pub arms: Vec<EffectHandlerArm>,
    pub active: bool,
}

#[derive(Debug)]
struct ResumptionState {
    #[allow(dead_code)]
    handler_id: u64,
    resumed: bool,
    resume_value: Option<AotValue>,
}

#[derive(Default)]
struct EffectState {
    next_handler_id: u64,
    next_scope_id: u64,
    frames: Vec<EffectHandlerFrame>,
    resumptions: Vec<ResumptionState>,
    abort_payload: Option<(u64, AotValue)>,
    provided_services: BTreeMap<Cow<'static, str>, AotValue>,
    scopes: Vec<Rc<RefCell<AotScope>>>,
}

impl fmt::Debug for EffectState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EffectState")
            .field("next_handler_id", &self.next_handler_id)
            .field("next_scope_id", &self.next_scope_id)
            .field("frames", &self.frames)
            .field("resumptions", &self.resumptions)
            .field("abort_payload", &self.abort_payload)
            .field("provided_services", &self.provided_services)
            .field("active_scopes", &self.scopes.len())
            .finish()
    }
}

/// Lazy stack-scoped effect & nominal service provider runtime embedded inside [`AotCtx`].
///
/// Uses `RefCell<Option<Rc<RefCell<EffectState>>>>` so [`AotCtx::new`] remains `const fn`
/// with zero heap allocation when effects and dynamic service bindings are unused.
#[derive(Debug)]
pub struct EffectRuntime {
    state: RefCell<Option<Rc<RefCell<EffectState>>>>,
}

impl EffectRuntime {
    /// Creates an empty, zero-allocation [`EffectRuntime`] in `const` context.
    #[inline]
    pub const fn new() -> Self {
        Self {
            state: RefCell::new(None),
        }
    }

    fn state_rc(&self) -> Result<Rc<RefCell<EffectState>>, AotError> {
        let mut slot = self
            .state
            .try_borrow_mut()
            .map_err(|_| AotError::UserError(String::from("re-entrant effect runtime borrow")))?;
        if let Some(rc) = slot.as_ref() {
            return Ok(Rc::clone(rc));
        }
        let rc = Rc::new(RefCell::new(EffectState::default()));
        *slot = Some(Rc::clone(&rc));
        Ok(rc)
    }

    fn existing_state_rc(&self) -> Result<Option<Rc<RefCell<EffectState>>>, AotError> {
        let slot = self
            .state
            .try_borrow()
            .map_err(|_| AotError::UserError(String::from("re-entrant effect runtime borrow")))?;
        Ok(slot.as_ref().map(Rc::clone))
    }

    /// Returns the number of currently installed [`EffectHandlerFrame`]s on the stack.
    #[inline]
    pub fn active_frame_depth(&self) -> usize {
        let Ok(slot) = self.state.try_borrow() else {
            return 0;
        };
        let Some(rc) = slot.as_ref() else {
            return 0;
        };
        let Ok(st) = rc.try_borrow() else {
            return 0;
        };
        st.frames.len()
    }

    /// Returns `true` when executing inside a `with Op(...) => ...` handler arm.
    #[inline]
    pub fn is_in_handler_arm(&self) -> bool {
        let Ok(slot) = self.state.try_borrow() else {
            return false;
        };
        let Some(rc) = slot.as_ref() else {
            return false;
        };
        let Ok(st) = rc.try_borrow() else {
            return false;
        };
        !st.resumptions.is_empty()
    }
}

impl Default for EffectRuntime {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Executes `body` under a stack-scoped [`EffectHandlerFrame`] backed by [`core::mem::DropGuard`].
///
/// - Tail-resumptive arms (`resume(val)`) return `val` directly on the current stack with zero
///   continuation allocation.
/// - Abortive arms (returning without `resume`) unwind `body` (running any inner `defer`/
///   `errdefer` [`DropGuard`]s) and deliver the converted abort value as `Ok(R)` to the matching
///   `handle` boundary.
/// - Aborts targeting an outer handler pop this frame via [`DropGuard`] and propagate upward.
pub fn handle_tail_resumptive_typed<R, F, Conv>(
    ctx: &mut AotCtx,
    arms: Vec<EffectHandlerArm>,
    body: F,
    from_abort: Conv,
) -> Result<R, AotError>
where
    F: FnOnce(&mut AotCtx) -> Result<R, AotError>,
    Conv: FnOnce(AotValue) -> Result<R, AotError>,
{
    let state_rc = ctx.effects.state_rc()?;
    let handler_id = {
        let mut st = state_rc
            .try_borrow_mut()
            .map_err(|_| AotError::UserError(String::from("re-entrant effect stack borrow")))?;
        st.next_handler_id = st.next_handler_id.wrapping_add(1);
        let id = st.next_handler_id;
        st.frames.push(EffectHandlerFrame {
            id,
            arms,
            active: true,
        });
        id
    };

    // Automatically removes this handler frame on normal return, abort unwind, or error propagation.
    let _frame_guard = DropGuard::new((Rc::clone(&state_rc), handler_id), |(st_rc, hid)| {
        if let Ok(mut st) = st_rc.try_borrow_mut()
            && let Some(pos) = st.frames.iter().rposition(|f| f.id == hid)
        {
            st.frames.remove(pos);
        }
    });

    let body_res = body(ctx);
    drop(_frame_guard);

    match body_res {
        Ok(val) => Ok(val),
        Err(AotError::EffectAbort {
            handler_id: abort_hid,
        }) if abort_hid == handler_id => {
            let payload = {
                let mut st = state_rc.try_borrow_mut().map_err(|_| {
                    AotError::UserError(String::from("re-entrant effect abort borrow"))
                })?;
                match st.abort_payload.take() {
                    Some((pid, val)) if pid == handler_id => val,
                    other => {
                        st.abort_payload = other;
                        AotValue::Nil
                    }
                }
            };
            from_abort(payload)
        }
        Err(other) => Err(other),
    }
}

/// Dynamic [`AotValue`] convenience wrapper around [`handle_tail_resumptive_typed`].
#[inline]
pub fn handle_tail_resumptive<F>(
    ctx: &mut AotCtx,
    arms: Vec<EffectHandlerArm>,
    body: F,
) -> Result<AotValue, AotError>
where
    F: FnOnce(&mut AotCtx) -> Result<AotValue, AotError>,
{
    handle_tail_resumptive_typed(ctx, arms, body, Ok)
}

/// Dispatches `perform(effect_name, args)` to the innermost active [`EffectHandlerFrame`]
/// on the current call stack without allocating a continuation.
pub fn perform_effect(
    ctx: &mut AotCtx,
    effect_name: &str,
    args: &[AotValue],
) -> Result<AotValue, AotError> {
    let Some(state_rc) = ctx.effects.existing_state_rc()? else {
        return Err(AotError::UnhandledEffect(String::from(effect_name)));
    };

    // Strip optional outer quotes if passed dynamically (matches VM parity).
    let canonical_name = if effect_name.len() >= 2
        && effect_name.starts_with('"')
        && effect_name.ends_with('"')
    {
        &effect_name[1..effect_name.len() - 1]
    } else {
        effect_name
    };

    let matched = {
        let mut st = state_rc
            .try_borrow_mut()
            .map_err(|_| AotError::UserError(String::from("re-entrant effect perform borrow")))?;
        let mut found: Option<(usize, u64, AotClosure)> = None;
        for (idx, frame) in st.frames.iter().enumerate().rev() {
            if !frame.active {
                continue;
            }
            for arm in &frame.arms {
                if arm.effect_name.as_ref() == canonical_name {
                    found = Some((idx, frame.id, Rc::clone(&arm.handler)));
                    break;
                }
            }
            if found.is_some() {
                break;
            }
        }
        if let Some((frame_idx, handler_id, handler_fn)) = found {
            if let Some(frame) = st.frames.get_mut(frame_idx) {
                frame.active = false;
            }
            st.resumptions.push(ResumptionState {
                handler_id,
                resumed: false,
                resume_value: None,
            });
            Some((handler_id, handler_fn))
        } else {
            None
        }
    };

    let Some((handler_id, handler_fn)) = matched else {
        return Err(AotError::UnhandledEffect(String::from(canonical_name)));
    };

    // Guarantee frame reactivation and resumption-stack pop even if the handler arm returns Err.
    let _reactivate_guard = DropGuard::new((Rc::clone(&state_rc), handler_id), |(st_rc, hid)| {
        if let Ok(mut st) = st_rc.try_borrow_mut()
            && let Some(frame) = st.frames.iter_mut().rev().find(|f| f.id == hid)
        {
            frame.active = true;
        }
    });
    let resumption_guard = DropGuard::new(Rc::clone(&state_rc), |st_rc| {
        if let Ok(mut st) = st_rc.try_borrow_mut() {
            let _ = st.resumptions.pop();
        }
    });

    // Execute handler arm closure directly on the current stack (zero continuation allocation).
    let arm_res = handler_fn(ctx, args);

    let popped = {
        let mut st = state_rc
            .try_borrow_mut()
            .map_err(|_| AotError::UserError(String::from("re-entrant effect resume pop")))?;
        st.resumptions.pop()
    };
    DropGuard::dismiss(resumption_guard);
    drop(_reactivate_guard);

    match arm_res {
        Ok(arm_val) => {
            let Some(resumption) = popped else {
                return Err(AotError::UserError(String::from(
                    "Corrupted effect resumption stack",
                )));
            };
            if resumption.resumed {
                Ok(resumption.resume_value.unwrap_or(AotValue::Nil))
            } else {
                let mut st = state_rc.try_borrow_mut().map_err(|_| {
                    AotError::UserError(String::from("re-entrant effect abort store"))
                })?;
                st.abort_payload = Some((handler_id, arm_val));
                Err(AotError::EffectAbort { handler_id })
            }
        }
        Err(err) => Err(err),
    }
}

/// Resumes the innermost active handler arm with `val` (at most once per `perform`).
pub fn resume_effect(ctx: &mut AotCtx, val: AotValue) -> Result<AotValue, AotError> {
    let Some(state_rc) = ctx.effects.existing_state_rc()? else {
        return Err(AotError::UserError(String::from(
            "Cannot resume outside of an effect handler arm",
        )));
    };
    let mut st = state_rc
        .try_borrow_mut()
        .map_err(|_| AotError::UserError(String::from("re-entrant effect resume borrow")))?;
    let Some(resumption) = st.resumptions.last_mut() else {
        return Err(AotError::UserError(String::from(
            "Cannot resume outside of an effect handler arm",
        )));
    };
    if resumption.resumed {
        return Err(AotError::UserError(String::from(
            "Effect already resumed: resumption can only be called once",
        )));
    }
    resumption.resumed = true;
    resumption.resume_value = Some(val.clone());
    Ok(val)
}

/// Binds a nominal service provider into [`AotCtx::effects`] for the duration of `body`,
/// restoring any prior binding via [`core::mem::DropGuard`] on exit or error.
pub fn with_provided_service<R, F>(
    ctx: &mut AotCtx,
    service_name: impl Into<Cow<'static, str>>,
    provider: AotValue,
    body: F,
) -> Result<R, AotError>
where
    F: FnOnce(&mut AotCtx) -> Result<R, AotError>,
{
    let state_rc = ctx.effects.state_rc()?;
    let key: Cow<'static, str> = service_name.into();
    let prev = {
        let mut st = state_rc
            .try_borrow_mut()
            .map_err(|_| AotError::UserError(String::from("re-entrant service provide borrow")))?;
        st.provided_services.insert(key.clone(), provider)
    };

    let _restore = DropGuard::new((Rc::clone(&state_rc), key, prev), |(st_rc, k, old)| {
        if let Ok(mut st) = st_rc.try_borrow_mut() {
            match old {
                Some(v) => {
                    st.provided_services.insert(k, v);
                }
                None => {
                    st.provided_services.remove(&k);
                }
            }
        }
    });

    body(ctx)
}

/// Looks up the currently provided nominal service binding from [`AotCtx::effects`], if any.
pub fn get_provided_service(ctx: &AotCtx, service_name: &str) -> Option<AotValue> {
    let state_rc = ctx.effects.existing_state_rc().ok()??;
    let st = state_rc.try_borrow().ok()?;
    st.provided_services.get(service_name).cloned()
}

// ===========================================================================
// NG-W13-1B Part 1: One-Shot Non-Tail `handle..with` via `EffectYield`
// ===========================================================================

/// Fully parameterized two-phase continuation protocol for one-shot non-tail `handle..with`
/// (`NG-W13-1B`, `F-12`, `F-19`).
///
/// Generic over all three types (`OpArg`, `ResumeVal`, `BodyRet`) so callers and generated
/// AOT state machines work uniformly with `i64`, `f64`, `bool`, `String`, `()`, [`AotValue`],
/// and user structs/enums without `i64` hardcoding.
pub enum EffectYield<OpArg, ResumeVal, BodyRet> {
    Completed(BodyRet),
    Yielded {
        arg: OpArg,
        resume_fn: Box<dyn FnOnce(&mut crate::AotCtx, ResumeVal) -> Result<BodyRet, crate::AotError>>,
    },
}

impl<OpArg, ResumeVal, BodyRet> EffectYield<OpArg, ResumeVal, BodyRet> {
    /// Constructs a completed yield carrying the final body value.
    #[inline]
    pub const fn completed(value: BodyRet) -> Self {
        Self::Completed(value)
    }

    /// Constructs a suspended yield carrying the operation argument and one-shot continuation.
    #[inline]
    pub fn yielded<F>(arg: OpArg, resume_fn: F) -> Self
    where
        F: FnOnce(&mut AotCtx, ResumeVal) -> Result<BodyRet, AotError> + 'static,
    {
        Self::Yielded {
            arg,
            resume_fn: Box::new(resume_fn),
        }
    }

    /// Returns `true` when the computation completed without yielding an effect operation.
    #[inline]
    pub const fn is_completed(&self) -> bool {
        matches!(self, Self::Completed(_))
    }

    /// Returns `true` when the computation suspended at a `perform` point.
    #[inline]
    pub const fn is_yielded(&self) -> bool {
        matches!(self, Self::Yielded { .. })
    }
}

impl<OpArg: fmt::Debug, ResumeVal, BodyRet: fmt::Debug> fmt::Debug
    for EffectYield<OpArg, ResumeVal, BodyRet>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Completed(v) => f.debug_tuple("Completed").field(v).finish(),
            Self::Yielded { arg, .. } => f
                .debug_struct("Yielded")
                .field("arg", arg)
                .field("resume_fn", &"<FnOnce>")
                .finish(),
        }
    }
}

/// Panic-free one-shot resumption cell wrapping an [`EffectYield::Yielded`] continuation.
///
/// Enforces at-most-once resumption at runtime when a handler arm invokes `resume(...)`
/// across branches or erroneously attempts double resumption, returning the canonical
/// `"Effect already resumed: resumption can only be called once"` error instead of panicking.
pub struct OneShotResume<ResumeVal, BodyRet> {
    inner: Option<Box<dyn FnOnce(&mut AotCtx, ResumeVal) -> Result<BodyRet, AotError>>>,
}

impl<ResumeVal, BodyRet> OneShotResume<ResumeVal, BodyRet> {
    /// Wraps a one-shot continuation closure from [`EffectYield::Yielded`].
    #[inline]
    pub fn new(
        resume_fn: Box<dyn FnOnce(&mut AotCtx, ResumeVal) -> Result<BodyRet, AotError>>,
    ) -> Self {
        Self {
            inner: Some(resume_fn),
        }
    }

    /// Invokes the suspended continuation at most once, returning `BodyRet` so post-resume code
    /// in the handler arm can inspect and transform the body's result.
    #[inline]
    pub fn resume(&mut self, ctx: &mut AotCtx, val: ResumeVal) -> Result<BodyRet, AotError> {
        match self.inner.take() {
            Some(f) => f(ctx, val),
            None => Err(AotError::UserError(String::from(
                "Effect already resumed: resumption can only be called once",
            ))),
        }
    }

    /// Returns `true` once the continuation has been consumed by [`Self::resume`].
    #[inline]
    pub fn is_consumed(&self) -> bool {
        self.inner.is_none()
    }
}

impl<ResumeVal, BodyRet> fmt::Debug for OneShotResume<ResumeVal, BodyRet> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OneShotResume")
            .field("consumed", &self.inner.is_none())
            .finish()
    }
}

/// Executes a one-shot non-tail `handle { body } with { Op(arg) => { pre; let r = resume(v); post } }`
/// using the parameterized [`EffectYield<OpArg, ResumeVal, BodyRet>`] protocol.
#[inline]
pub fn handle_non_tail_oneshot<OpArg, ResumeVal, BodyRet, BodyFn, HandlerFn>(
    ctx: &mut AotCtx,
    body: BodyFn,
    handler: HandlerFn,
) -> Result<BodyRet, AotError>
where
    BodyFn: FnOnce(&mut AotCtx) -> Result<EffectYield<OpArg, ResumeVal, BodyRet>, AotError>,
    HandlerFn: FnOnce(
        &mut AotCtx,
        OpArg,
        Box<dyn FnOnce(&mut AotCtx, ResumeVal) -> Result<BodyRet, AotError>>,
    ) -> Result<BodyRet, AotError>,
{
    match body(ctx)? {
        EffectYield::Completed(val) => Ok(val),
        EffectYield::Yielded { arg, resume_fn } => handler(ctx, arg, resume_fn),
    }
}

// ===========================================================================
// NG-W13-1B Part 2: 3-Phase LIFO Structured Concurrency `AotScope`
// ===========================================================================

/// Teardown lifecycle phases of [`AotScope`], strictly ordered
/// `Open -> Cancelling -> Joining -> Finalizing -> Done`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AotScopePhase {
    /// Scope is open and accepts new child fibers and finalizer registrations.
    Open,
    /// Phase 1: admission closed; cancellation flag broadcast to all unfinished child fibers.
    Cancelling,
    /// Phase 2: waiting for / driving all child fibers to terminal completion before finalizing.
    Joining,
    /// Phase 3: running all registered finalizers in strict reverse (LIFO) order.
    Finalizing,
    /// Scope teardown finished; subsequent `close` calls are idempotent no-ops.
    Done,
}

/// Lifecycle state of an [`AotScope`] child fiber.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AotFiberState {
    Pending,
    Running,
    Completed,
    Cancelled,
    Failed,
}

impl AotFiberState {
    #[inline]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Cancelled | Self::Failed)
    }
}

/// Shared, `unsafe`-free handle to a child fiber registered inside an [`AotScope`].
#[derive(Clone, Debug)]
pub struct AotFiberHandle {
    id: u64,
    state: Rc<Cell<AotFiberState>>,
    cancel_requested: Rc<Cell<bool>>,
    masked_depth: Rc<Cell<usize>>,
}

impl AotFiberHandle {
    /// Stable per-scope fiber identifier.
    #[inline]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Current execution state of the child fiber.
    #[inline]
    pub fn state(&self) -> AotFiberState {
        self.state.get()
    }

    /// Returns `true` once Phase 1 (`Cancelling`) has flagged this fiber for cancellation.
    #[inline]
    pub fn is_cancel_requested(&self) -> bool {
        self.cancel_requested.get()
    }

    /// Returns `true` when the fiber has finished (`Completed`, `Cancelled`, or `Failed`).
    #[inline]
    pub fn is_terminal(&self) -> bool {
        self.state.get().is_terminal()
    }

    /// Returns `true` while inside an interrupt-masked critical section (Gate-NEG-02).
    #[inline]
    pub fn is_masked(&self) -> bool {
        self.masked_depth.get() > 0
    }

    /// Enters an interrupt-masked section; cancellation is deferred until [`Self::exit_mask`].
    #[inline]
    pub fn enter_mask(&self) {
        self.masked_depth
            .set(self.masked_depth.get().saturating_add(1));
    }

    /// Exits an interrupt-masked section. Returns `true` if a cancellation was deferred while masked.
    #[inline]
    pub fn exit_mask(&self) -> bool {
        let next = self.masked_depth.get().saturating_sub(1);
        self.masked_depth.set(next);
        next == 0 && self.cancel_requested.get()
    }

    /// Marks the fiber completed (useful for externally driven fibers / tests).
    #[inline]
    pub fn mark_completed(&self) {
        if !self.is_terminal() {
            self.state.set(AotFiberState::Completed);
        }
    }

    /// Marks the fiber cancelled (useful after delivering a deferred masked cancellation).
    #[inline]
    pub fn mark_cancelled(&self) {
        if !self.is_terminal() {
            self.state.set(AotFiberState::Cancelled);
        }
    }
}

type ScopeStepFn = Box<dyn FnMut(&mut AotCtx, bool) -> Result<bool, AotError>>;
type ScopeCancelHook = Box<dyn FnOnce(&mut AotCtx)>;
type ScopeFinalizerFn = Box<dyn FnOnce(&mut AotCtx) -> Result<(), AotError>>;

struct AotChildFiber {
    handle: AotFiberHandle,
    on_cancel: Option<ScopeCancelHook>,
    on_step: Option<ScopeStepFn>,
}

const MAX_SCOPE_JOIN_ROUNDS: usize = 10_000;

/// 3-phase LIFO structured concurrency scope for `StmtKind::Scoped` and `ExprKind::EffectBlock`
/// (`NG-W13-1B`).
///
/// [`AotScope::close`] strictly enforces the 3-phase order:
/// 1. **`Cancelling`**: closes admission and sets the cancel flag on every unfinished child fiber.
/// 2. **`Joining`**: drives/waits for every child fiber to reach a terminal state before touching
///    any scope finalizers (Gate-NEG-02).
/// 3. **`Finalizing`**: pops and executes every registered finalizer in strict reverse (LIFO)
///    order, aggregating errors while preserving the first defect.
pub struct AotScope {
    id: u64,
    next_child_id: u64,
    phase: AotScopePhase,
    phase_transitions: Vec<AotScopePhase>,
    children: Vec<AotChildFiber>,
    finalizers: Vec<ScopeFinalizerFn>,
    first_error: Option<AotError>,
    suppressed_errors: Vec<AotError>,
}

impl AotScope {
    /// Creates a new open [`AotScope`] with identifier `id`.
    pub fn new(id: u64) -> Self {
        let mut transitions = Vec::with_capacity(5);
        transitions.push(AotScopePhase::Open);
        Self {
            id,
            next_child_id: 0,
            phase: AotScopePhase::Open,
            phase_transitions: transitions,
            children: Vec::new(),
            finalizers: Vec::new(),
            first_error: None,
            suppressed_errors: Vec::new(),
        }
    }

    #[inline]
    pub const fn id(&self) -> u64 {
        self.id
    }

    #[inline]
    pub const fn phase(&self) -> AotScopePhase {
        self.phase
    }

    /// Ordered history of phase transitions observed by this scope (starts with `[Open]`).
    #[inline]
    pub fn phase_transitions(&self) -> &[AotScopePhase] {
        &self.phase_transitions
    }

    #[inline]
    pub const fn is_open(&self) -> bool {
        matches!(self.phase, AotScopePhase::Open)
    }

    #[inline]
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    #[inline]
    pub fn finalizer_count(&self) -> usize {
        self.finalizers.len()
    }

    /// Secondary errors aggregated after the first defect during `Joining`/`Finalizing`.
    #[inline]
    pub fn suppressed_errors(&self) -> &[AotError] {
        &self.suppressed_errors
    }

    fn transition_to(&mut self, next: AotScopePhase) {
        if self.phase != next {
            self.phase = next;
            self.phase_transitions.push(next);
        }
    }

    fn record_error(&mut self, err: AotError) {
        if self.first_error.is_none() {
            self.first_error = Some(err);
        } else {
            self.suppressed_errors.push(err);
        }
    }

    fn alloc_child_handle(&mut self) -> Result<AotFiberHandle, AotError> {
        if !self.is_open() {
            return Err(AotError::UserError(String::from(
                "Scope closed: cannot spawn new child fiber",
            )));
        }
        self.next_child_id = self.next_child_id.wrapping_add(1);
        Ok(AotFiberHandle {
            id: self.next_child_id,
            state: Rc::new(Cell::new(AotFiberState::Pending)),
            cancel_requested: Rc::new(Cell::new(false)),
            masked_depth: Rc::new(Cell::new(0)),
        })
    }

    /// Registers a manually driven child fiber (useful for masked Gate-NEG-02 host integration).
    pub fn register_external_child(&mut self) -> Result<AotFiberHandle, AotError> {
        let handle = self.alloc_child_handle()?;
        handle.state.set(AotFiberState::Running);
        self.children.push(AotChildFiber {
            handle: handle.clone(),
            on_cancel: None,
            on_step: None,
        });
        Ok(handle)
    }

    /// Spawns a cooperative child fiber polled with `(ctx, cancel_requested) -> Result<done, AotError>`.
    pub fn spawn_child<F>(&mut self, step_fn: F) -> Result<AotFiberHandle, AotError>
    where
        F: FnMut(&mut AotCtx, bool) -> Result<bool, AotError> + 'static,
    {
        let handle = self.alloc_child_handle()?;
        handle.state.set(AotFiberState::Running);
        self.children.push(AotChildFiber {
            handle: handle.clone(),
            on_cancel: None,
            on_step: Some(Box::new(step_fn)),
        });
        Ok(handle)
    }

    /// Spawns a cooperative child fiber with an explicit Phase 1 (`Cancelling`) callback hook.
    pub fn spawn_child_with_cancel<C, F>(
        &mut self,
        on_cancel: C,
        step_fn: F,
    ) -> Result<AotFiberHandle, AotError>
    where
        C: FnOnce(&mut AotCtx) + 'static,
        F: FnMut(&mut AotCtx, bool) -> Result<bool, AotError> + 'static,
    {
        let handle = self.alloc_child_handle()?;
        handle.state.set(AotFiberState::Running);
        self.children.push(AotChildFiber {
            handle: handle.clone(),
            on_cancel: Some(Box::new(on_cancel)),
            on_step: Some(Box::new(step_fn)),
        });
        Ok(handle)
    }

    /// Spawns a Rust `core::future::Future` as a cooperative `#![no_std]` child fiber using
    /// `core::task::Waker::noop()` (100% safe, zero `unsafe`).
    ///
    /// On unmasked cancellation during `Joining`, the pinned future is dropped immediately so
    /// its internal RAII guards unwind in LIFO order before parent scope `Finalizing` runs.
    pub fn spawn_future<Fut>(&mut self, fut: Fut) -> Result<AotFiberHandle, AotError>
    where
        Fut: Future<Output = Result<AotValue, AotError>> + 'static,
    {
        let handle = self.alloc_child_handle()?;
        let masked_ref = Rc::clone(&handle.masked_depth);
        let mut pinned: Pin<Box<Fut>> = Box::pin(fut);
        let mut init_cx = Context::from_waker(Waker::noop());
        let mut pinned_opt: Option<Pin<Box<Fut>>> = match pinned.as_mut().poll(&mut init_cx) {
            Poll::Ready(Ok(_)) => {
                handle.state.set(AotFiberState::Completed);
                None
            }
            Poll::Ready(Err(err)) => {
                handle.state.set(AotFiberState::Failed);
                self.record_error(err);
                None
            }
            Poll::Pending => {
                handle.state.set(AotFiberState::Running);
                Some(pinned)
            }
        };
        let on_step: Option<ScopeStepFn> = pinned_opt.take().map(|p| {
            let mut active = Some(p);
            let step: ScopeStepFn = Box::new(move |_ctx, cancelled| {
                if cancelled && masked_ref.get() == 0 {
                    // Drop future now so child RAII guards unwind before parent finalizers.
                    let _ = active.take();
                    return Ok(true);
                }
                let Some(pinned_ref) = active.as_mut() else {
                    return Ok(true);
                };
                let mut cx = Context::from_waker(Waker::noop());
                match pinned_ref.as_mut().poll(&mut cx) {
                    Poll::Ready(Ok(_)) => {
                        let _ = active.take();
                        Ok(true)
                    }
                    Poll::Ready(Err(err)) => {
                        let _ = active.take();
                        Err(err)
                    }
                    Poll::Pending => Ok(false),
                }
            });
            step
        });
        self.children.push(AotChildFiber {
            handle: handle.clone(),
            on_cancel: None,
            on_step,
        });
        Ok(handle)
    }

    /// Registers a fallible finalizer to run in strict reverse (LIFO) order during Phase 3 (`Finalizing`).
    pub fn register_finalizer<F>(&mut self, finalizer: F) -> Result<(), AotError>
    where
        F: FnOnce(&mut AotCtx) -> Result<(), AotError> + 'static,
    {
        if !self.is_open() {
            return Err(AotError::UserError(String::from(
                "Scope closed: cannot register new resource finalizer",
            )));
        }
        self.finalizers.push(Box::new(finalizer));
        Ok(())
    }

    /// Phase 1 helper: closes admission and broadcasts cancellation to all unfinished children.
    fn run_phase1_cancelling(&mut self, ctx: &mut AotCtx) {
        if self.phase != AotScopePhase::Open {
            return;
        }
        self.transition_to(AotScopePhase::Cancelling);
        for child in &mut self.children {
            if !child.handle.is_terminal() {
                child.handle.cancel_requested.set(true);
                if let Some(on_cancel) = child.on_cancel.take() {
                    on_cancel(ctx);
                }
                if child.on_step.is_none() && !child.handle.is_masked() {
                    child.handle.state.set(AotFiberState::Cancelled);
                }
            }
        }
    }

    /// Single cooperative poll pass of Phase 2 (`Joining`). Returns `true` when all children are terminal.
    fn poll_phase2_joining_once(&mut self, ctx: &mut AotCtx) -> bool {
        if self.phase == AotScopePhase::Cancelling {
            self.transition_to(AotScopePhase::Joining);
        }
        let mut all_done = true;
        let mut step_errors: Vec<AotError> = Vec::new();
        for child in &mut self.children {
            if child.handle.is_terminal() {
                child.on_step = None;
                continue;
            }
            let cancelled = child.handle.cancel_requested.get();
            if let Some(mut step_fn) = child.on_step.take() {
                match step_fn(ctx, cancelled) {
                    Ok(true) => {
                        if !child.handle.is_terminal() {
                            let final_st = if cancelled {
                                AotFiberState::Cancelled
                            } else {
                                AotFiberState::Completed
                            };
                            child.handle.state.set(final_st);
                        }
                    }
                    Ok(false) => {
                        child.on_step = Some(step_fn);
                        all_done = false;
                    }
                    Err(err) => {
                        child.handle.state.set(AotFiberState::Failed);
                        step_errors.push(err);
                    }
                }
            } else if !child.handle.is_masked() {
                child.handle.state.set(AotFiberState::Cancelled);
            } else {
                all_done = false;
            }
        }
        for err in step_errors {
            self.record_error(err);
        }
        all_done
    }

    /// Phase 3 helper: runs every registered finalizer in strict reverse (LIFO) order,
    /// aggregating errors without losing the first defect.
    fn run_phase3_finalizing(&mut self, ctx: &mut AotCtx) -> Result<(), AotError> {
        self.transition_to(AotScopePhase::Finalizing);
        while let Some(finalizer) = self.finalizers.pop() {
            if let Err(err) = finalizer(ctx) {
                self.record_error(err);
            }
        }
        self.transition_to(AotScopePhase::Done);
        match self.first_error.take() {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    /// Non-blocking 3-phase teardown step (Gate-NEG-02):
    /// - Runs Phase 1 (`Cancelling`) on first call.
    /// - Polls Phase 2 (`Joining`) once. If any child fiber is still running or masked, stops in
    ///   [`AotScopePhase::Joining`] and returns `Ok(false)` without running any finalizers.
    /// - Once all children are terminal, runs Phase 3 (`Finalizing`) in strict LIFO order and
    ///   returns `Ok(true)` (or `Err(first_defect)`).
    pub fn try_close(&mut self, ctx: &mut AotCtx) -> Result<bool, AotError> {
        if self.phase == AotScopePhase::Done {
            return Ok(true);
        }
        if self.phase == AotScopePhase::Open {
            self.run_phase1_cancelling(ctx);
        }
        if !self.poll_phase2_joining_once(ctx) {
            return Ok(false);
        }
        self.run_phase3_finalizing(ctx)?;
        Ok(true)
    }

    /// Full 3-phase LIFO teardown (`Cancelling -> Joining -> Finalizing -> Done`).
    ///
    /// Strictly enforces:
    /// 1. `Cancelling`: sets the cancel flag on every unfinished child fiber and closes admission.
    /// 2. `Joining`: drives all child fibers to terminal completion before touching any finalizers.
    /// 3. `Finalizing`: executes all registered finalizers in strict reverse (LIFO) order,
    ///    aggregating errors without losing the first defect.
    pub fn close(&mut self, ctx: &mut AotCtx) -> Result<(), AotError> {
        if self.phase == AotScopePhase::Done {
            return Ok(());
        }
        if self.phase == AotScopePhase::Open {
            self.run_phase1_cancelling(ctx);
        }
        let mut joined = false;
        for _ in 0..MAX_SCOPE_JOIN_ROUNDS {
            if self.poll_phase2_joining_once(ctx) {
                joined = true;
                break;
            }
            let any_steppable = self
                .children
                .iter()
                .any(|c| !c.handle.is_terminal() && c.on_step.is_some());
            if !any_steppable {
                break;
            }
        }
        if !joined {
            return Err(AotError::UserError(String::from(
                "Scope join stalled: child fiber did not terminate before finalization",
            )));
        }
        self.run_phase3_finalizing(ctx)
    }
}

impl fmt::Debug for AotScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AotScope")
            .field("id", &self.id)
            .field("phase", &self.phase)
            .field("phase_transitions", &self.phase_transitions)
            .field("children", &self.children.len())
            .field("finalizers", &self.finalizers.len())
            .field("suppressed_errors", &self.suppressed_errors.len())
            .finish()
    }
}

/// Executes `body` inside a fresh 3-phase LIFO [`AotScope`] pushed onto [`AotCtx::effects`].
///
/// On normal completion OR `Err(...)` return from `body`, removes the scope and runs
/// [`AotScope::close`] (`Cancelling -> Joining -> Finalizing`), preserving the first defect
/// without losing `body`'s error if a finalizer also fails.
pub fn with_aot_scope<R, F>(ctx: &mut AotCtx, body: F) -> Result<R, AotError>
where
    F: FnOnce(&mut AotCtx) -> Result<R, AotError>,
{
    let state_rc = ctx.effects.state_rc()?;
    let (scope_id, scope_rc) = {
        let mut st = state_rc
            .try_borrow_mut()
            .map_err(|_| AotError::UserError(String::from("re-entrant scope push borrow")))?;
        st.next_scope_id = st.next_scope_id.wrapping_add(1);
        let id = st.next_scope_id;
        let sc = Rc::new(RefCell::new(AotScope::new(id)));
        st.scopes.push(Rc::clone(&sc));
        (id, sc)
    };

    // Safety guard ensuring the scope entry is popped from the stack even on unexpected exits.
    let scope_guard = DropGuard::new((Rc::clone(&state_rc), scope_id), |(st_rc, sid)| {
        if let Ok(mut st) = st_rc.try_borrow_mut()
            && let Some(pos) = st
                .scopes
                .iter()
                .rposition(|s| s.try_borrow().is_ok_and(|b| b.id() == sid))
        {
            st.scopes.remove(pos);
        }
    });

    let body_res = body(ctx);

    // Pop scope from active stack before executing close so nested finalizers observe parent scope.
    drop(scope_guard);

    let close_res = {
        let mut scope = scope_rc
            .try_borrow_mut()
            .map_err(|_| AotError::UserError(String::from("re-entrant scope close borrow")))?;
        scope.close(ctx)
    };

    // Aggregate errors without losing the first defect.
    match (body_res, close_res) {
        (Err(body_err), _) => Err(body_err),
        (Ok(_), Err(close_err)) => Err(close_err),
        (Ok(val), Ok(())) => Ok(val),
    }
}

/// Registers a LIFO finalizer onto the innermost active [`AotScope`] in `ctx`.
pub fn scope_register_finalizer<F>(ctx: &mut AotCtx, finalizer: F) -> Result<(), AotError>
where
    F: FnOnce(&mut AotCtx) -> Result<(), AotError> + 'static,
{
    let Some(state_rc) = ctx.effects.existing_state_rc()? else {
        return Err(AotError::UserError(String::from(
            "No active AotScope for finalizer registration",
        )));
    };
    let scope_rc = {
        let st = state_rc
            .try_borrow()
            .map_err(|_| AotError::UserError(String::from("re-entrant scope borrow")))?;
        st.scopes.last().cloned().ok_or_else(|| {
            AotError::UserError(String::from(
                "No active AotScope for finalizer registration",
            ))
        })?
    };
    let mut scope = scope_rc
        .try_borrow_mut()
        .map_err(|_| AotError::UserError(String::from("re-entrant scope borrow")))?;
    scope.register_finalizer(finalizer)
}

/// Spawns a cooperative child fiber inside the innermost active [`AotScope`] in `ctx`.
pub fn scope_spawn_child<F>(ctx: &mut AotCtx, step_fn: F) -> Result<AotFiberHandle, AotError>
where
    F: FnMut(&mut AotCtx, bool) -> Result<bool, AotError> + 'static,
{
    let Some(state_rc) = ctx.effects.existing_state_rc()? else {
        return Err(AotError::UserError(String::from(
            "No active AotScope for child fiber spawn",
        )));
    };
    let scope_rc = {
        let st = state_rc
            .try_borrow()
            .map_err(|_| AotError::UserError(String::from("re-entrant scope borrow")))?;
        st.scopes.last().cloned().ok_or_else(|| {
            AotError::UserError(String::from("No active AotScope for child fiber spawn"))
        })?
    };
    let mut scope = scope_rc
        .try_borrow_mut()
        .map_err(|_| AotError::UserError(String::from("re-entrant scope borrow")))?;
    scope.spawn_child(step_fn)
}

/// Spawns a `core::future::Future` child fiber inside the innermost active [`AotScope`] in `ctx`.
pub fn scope_spawn_future<Fut>(ctx: &mut AotCtx, fut: Fut) -> Result<AotFiberHandle, AotError>
where
    Fut: Future<Output = Result<AotValue, AotError>> + 'static,
{
    let Some(state_rc) = ctx.effects.existing_state_rc()? else {
        return Err(AotError::UserError(String::from(
            "No active AotScope for future spawn",
        )));
    };
    let scope_rc = {
        let st = state_rc
            .try_borrow()
            .map_err(|_| AotError::UserError(String::from("re-entrant scope borrow")))?;
        st.scopes.last().cloned().ok_or_else(|| {
            AotError::UserError(String::from("No active AotScope for future spawn"))
        })?
    };
    let mut scope = scope_rc
        .try_borrow_mut()
        .map_err(|_| AotError::UserError(String::from("re-entrant scope borrow")))?;
    scope.spawn_future(fut)
}

/// Returns the number of currently active [`AotScope`] frames on `ctx`.
#[inline]
pub fn active_scope_depth(ctx: &AotCtx) -> usize {
    let Ok(Some(state_rc)) = ctx.effects.existing_state_rc() else {
        return 0;
    };
    let Ok(st) = state_rc.try_borrow() else {
        return 0;
    };
    st.scopes.len()
}
