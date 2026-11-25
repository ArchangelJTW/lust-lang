use crate::ast::Type;
use crate::bytecode::{TaskHandle, Value};
use hashbrown::HashMap;
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

pub(crate) type AsyncValueFuture =
    Pin<Box<dyn Future<Output = std::result::Result<Value, String>>>>;

/// Trait for notifying an external async runtime / executor (such as Tokio `Notify`)
/// that a pending async native future has been awakened and the registry needs polling.
pub trait AsyncWakeNotifier: Send + Sync + 'static {
    fn notify(&self);
}

impl<F: Fn() + Send + Sync + 'static> AsyncWakeNotifier for F {
    fn notify(&self) {
        (self)();
    }
}

/// Trait for observing cancellation of an async task.
pub trait AsyncCancelObserver: Send + Sync + 'static {
    fn cancelled(&self);
}

impl<F: Fn() + Send + Sync + 'static> AsyncCancelObserver for F {
    fn cancelled(&self) {
        (self)();
    }
}

/// Synchronized, thread-safe slot holding an optional [`AsyncWakeNotifier`].
pub struct AtomicWakeNotifierSlot {
    inner: RwLock<Option<Arc<dyn AsyncWakeNotifier>>>,
}

impl AtomicWakeNotifierSlot {
    pub fn new(notifier: Option<Arc<dyn AsyncWakeNotifier>>) -> Self {
        Self {
            inner: RwLock::new(notifier),
        }
    }

    pub fn load(&self) -> Option<Arc<dyn AsyncWakeNotifier>> {
        self.inner.read().unwrap().clone()
    }

    pub fn set(
        &self,
        notifier: Option<Arc<dyn AsyncWakeNotifier>>,
    ) -> Option<Arc<dyn AsyncWakeNotifier>> {
        let mut guard = self.inner.write().unwrap();
        let prev = guard.clone();
        *guard = notifier;
        prev
    }

    pub fn notify(&self) {
        if let Some(notifier) = self.load() {
            notifier.notify();
        }
    }
}

thread_local! {
    static NOTIFIER_STACK: std::cell::RefCell<Vec<(u64, Arc<dyn AsyncWakeNotifier>)>> = const { std::cell::RefCell::new(Vec::new()) };
    static STACK_GENERATION: std::cell::Cell<u64> = const { std::cell::Cell::new(1) };
    static CURRENT_THREAD_PENDING_TASKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Returns the number of currently pending async tasks on the current thread.
#[allow(dead_code)]
pub fn current_thread_pending_task_count() -> usize {
    CURRENT_THREAD_PENDING_TASKS.with(|c| c.get())
}

/// Sets the wake notifier for the current thread and returns the previous notifier.
pub fn set_thread_wake_notifier(
    notifier: Option<Arc<dyn AsyncWakeNotifier>>,
) -> Option<Arc<dyn AsyncWakeNotifier>> {
    NOTIFIER_STACK.with(|stack| {
        let mut list = stack.borrow_mut();
        let prev = list.last().map(|(_, n)| Arc::clone(n));
        list.clear();
        if let Some(n) = notifier {
            list.push((0, n));
        }
        prev
    })
}

/// Gets the current thread wake notifier, if any.
pub fn current_thread_wake_notifier() -> Option<Arc<dyn AsyncWakeNotifier>> {
    NOTIFIER_STACK.with(|stack| {
        stack.borrow().last().map(|(_, n)| Arc::clone(n))
    })
}

/// RAII scope guard for scoped thread wake notifier installation.
///
/// Uses an internal generation-tagged stack ensuring strict LIFO semantics and safe
/// restoration even if guards are dropped out-of-order.
/// Contains `PhantomData<std::rc::Rc<()>>` to ensure it cannot be moved across threads.
pub struct WakeNotifierScope {
    generation: u64,
    owner_thread: std::thread::ThreadId,
    _not_send: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl WakeNotifierScope {
    pub fn install(notifier: Arc<dyn AsyncWakeNotifier>) -> Self {
        let gen_id = STACK_GENERATION.with(|g| {
            let next = g.get() + 1;
            g.set(next);
            next
        });

        NOTIFIER_STACK.with(|stack| {
            stack.borrow_mut().push((gen_id, notifier));
        });

        Self {
            generation: gen_id,
            owner_thread: std::thread::current().id(),
            _not_send: std::marker::PhantomData,
        }
    }
}

impl Drop for WakeNotifierScope {
    fn drop(&mut self) {
        if std::thread::current().id() == self.owner_thread {
            NOTIFIER_STACK.with(|stack| {
                let mut list = stack.borrow_mut();
                if let Some(pos) = list.iter().rposition(|(g, _)| *g == self.generation) {
                    list.remove(pos);
                }
            });
        }
    }
}

pub(crate) struct AsyncRegistry {
    pub(crate) next_id: u64,
    pub(crate) pending: HashMap<u64, AsyncTaskEntry>,
    wake_notifier: Arc<AtomicWakeNotifierSlot>,
}

impl AsyncRegistry {
    pub(crate) fn new() -> Self {
        let inherited = current_thread_wake_notifier();
        Self {
            next_id: 1,
            pending: HashMap::new(),
            wake_notifier: Arc::new(AtomicWakeNotifierSlot::new(inherited)),
        }
    }

    #[allow(dead_code)]
    pub fn set_wake_notifier(
        &mut self,
        notifier: Option<Arc<dyn AsyncWakeNotifier>>,
    ) -> Option<Arc<dyn AsyncWakeNotifier>> {
        let prev = self.wake_notifier.set(notifier);
        // Wake upon replacement so any waiters during handoff are notified
        self.wake_notifier.notify();
        prev
    }

    #[allow(dead_code)]
    pub fn wake_notifier(&self) -> Option<Arc<dyn AsyncWakeNotifier>> {
        self.wake_notifier.load()
    }

    pub(crate) fn register(&mut self, entry: AsyncTaskEntry) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        CURRENT_THREAD_PENDING_TASKS.with(|c| {
            c.set(c.get() + 1);
        });
        entry.wake_flag.set_notifier(Arc::clone(&self.wake_notifier));
        self.pending.insert(id, entry);
        id
    }

    pub(crate) fn has_pending_for(&self, handle: TaskHandle) -> bool {
        self.pending.values().any(|entry| match entry.target {
            AsyncTaskTarget::ScriptTask(existing) | AsyncTaskTarget::NativeTask(existing) => {
                existing == handle
            }
        })
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.pending.len()
    }
}

pub(crate) struct AsyncTaskEntry {
    pub(crate) target: AsyncTaskTarget,
    pub(crate) future: AsyncValueFuture,
    pub(crate) expected_type: Option<Type>,
    wake_flag: Arc<WakeFlag>,
    immediate_poll: bool,
    #[allow(dead_code)]
    cancel_observer: Option<Arc<dyn AsyncCancelObserver>>,
}

#[derive(Clone, Copy)]
pub(crate) enum AsyncTaskTarget {
    ScriptTask(TaskHandle),
    NativeTask(TaskHandle),
}

impl AsyncTaskEntry {
    pub(crate) fn new(target: AsyncTaskTarget, future: AsyncValueFuture) -> Self {
        Self {
            target,
            future,
            expected_type: None,
            wake_flag: Arc::new(WakeFlag::new(None)),
            immediate_poll: true,
            cancel_observer: None,
        }
    }

    #[allow(dead_code)]
    pub fn with_cancel_observer(mut self, observer: Arc<dyn AsyncCancelObserver>) -> Self {
        self.cancel_observer = Some(observer);
        self
    }

    #[allow(dead_code)]
    pub(crate) fn cancel(&mut self) {
        if let Some(obs) = &self.cancel_observer {
            obs.cancelled();
        }
    }

    pub(crate) fn with_expected_type(mut self, expected_type: Type) -> Self {
        self.expected_type = Some(expected_type);
        self
    }

    pub(crate) fn take_should_poll(&mut self) -> bool {
        if self.immediate_poll {
            self.immediate_poll = false;
            true
        } else {
            self.wake_flag.take()
        }
    }

    pub(crate) fn make_waker(&self) -> Waker {
        make_async_waker(&self.wake_flag)
    }
}

impl Drop for AsyncTaskEntry {
    fn drop(&mut self) {
        CURRENT_THREAD_PENDING_TASKS.with(|c| {
            c.set(c.get().saturating_sub(1));
        });
        if let Some(obs) = &self.cancel_observer {
            obs.cancelled();
        }
    }
}

struct WakeFlag {
    pending: AtomicBool,
    notifier: RwLock<Option<Arc<AtomicWakeNotifierSlot>>>,
}

impl WakeFlag {
    fn new(notifier: Option<Arc<AtomicWakeNotifierSlot>>) -> Self {
        Self {
            pending: AtomicBool::new(true),
            notifier: RwLock::new(notifier),
        }
    }

    fn set_notifier(&self, slot: Arc<AtomicWakeNotifierSlot>) {
        *self.notifier.write().unwrap() = Some(slot);
    }

    fn take(&self) -> bool {
        self.pending.swap(false, Ordering::SeqCst)
    }

    fn wake(&self) {
        self.pending.store(true, Ordering::SeqCst);
        let slot = self.notifier.read().unwrap().clone();
        if let Some(s) = slot {
            s.notify();
        }
    }
}

fn make_async_waker(flag: &Arc<WakeFlag>) -> Waker {
    unsafe {
        Waker::from_raw(RawWaker::new(
            Arc::into_raw(flag.clone()) as *const (),
            &ASYNC_WAKER_VTABLE,
        ))
    }
}

unsafe fn async_waker_clone(ptr: *const ()) -> RawWaker {
    unsafe {
        let arc = Arc::<WakeFlag>::from_raw(ptr as *const WakeFlag);
        let cloned = arc.clone();
        std::mem::forget(arc);
        RawWaker::new(Arc::into_raw(cloned) as *const (), &ASYNC_WAKER_VTABLE)
    }
}

unsafe fn async_waker_wake(ptr: *const ()) {
    unsafe {
        let arc = Arc::<WakeFlag>::from_raw(ptr as *const WakeFlag);
        arc.wake();
    }
}

unsafe fn async_waker_wake_by_ref(ptr: *const ()) {
    unsafe {
        let arc = Arc::<WakeFlag>::from_raw(ptr as *const WakeFlag);
        arc.wake();
        std::mem::forget(arc);
    }
}

unsafe fn async_waker_drop(ptr: *const ()) {
    unsafe {
        let _ = Arc::<WakeFlag>::from_raw(ptr as *const WakeFlag);
    }
}

static ASYNC_WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(
    async_waker_clone,
    async_waker_wake,
    async_waker_wake_by_ref,
    async_waker_drop,
);

struct AsyncTaskQueueInner<Args, R> {
    queue: Mutex<VecDeque<PendingAsyncTask<Args, R>>>,
    condvar: Condvar,
}

pub struct AsyncTaskQueue<Args, R> {
    inner: Arc<AsyncTaskQueueInner<Args, R>>,
}

impl<Args, R> Clone for AsyncTaskQueue<Args, R> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<Args, R> Default for AsyncTaskQueue<Args, R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Args, R> AsyncTaskQueue<Args, R> {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(AsyncTaskQueueInner {
                queue: Mutex::new(VecDeque::new()),
                condvar: Condvar::new(),
            }),
        }
    }

    pub fn push(&self, task: PendingAsyncTask<Args, R>) {
        let mut guard = self.inner.queue.lock().unwrap();
        guard.push_back(task);
        self.inner.condvar.notify_one();
    }

    pub fn pop(&self) -> Option<PendingAsyncTask<Args, R>> {
        let mut guard = self.inner.queue.lock().unwrap();
        guard.pop_front()
    }

    pub fn pop_blocking(&self) -> PendingAsyncTask<Args, R> {
        let mut guard = self.inner.queue.lock().unwrap();
        loop {
            if let Some(task) = guard.pop_front() {
                return task;
            }
            guard = self.inner.condvar.wait(guard).unwrap();
        }
    }

    pub fn len(&self) -> usize {
        let guard = self.inner.queue.lock().unwrap();
        guard.len()
    }

    pub fn is_empty(&self) -> bool {
        let guard = self.inner.queue.lock().unwrap();
        guard.is_empty()
    }
}

pub struct PendingAsyncTask<Args, R> {
    task: TaskHandle,
    args: Args,
    completer: AsyncCompleter<R>,
}

impl<Args, R> PendingAsyncTask<Args, R> {
    pub fn new(task: TaskHandle, args: Args, completer: AsyncCompleter<R>) -> Self {
        Self {
            task,
            args,
            completer,
        }
    }

    pub fn task(&self) -> TaskHandle {
        self.task
    }

    pub fn args(&self) -> &Args {
        &self.args
    }

    pub fn complete_ok(self, value: R) {
        self.completer.complete_ok(value);
    }

    pub fn complete_err(self, message: impl Into<String>) {
        self.completer.complete_err(message);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalState {
    Pending,
    Ready,
    Consumed,
}

struct SignalInner<T> {
    state: SignalState,
    result: Option<std::result::Result<T, String>>,
    waker: Option<Waker>,
}

pub struct AsyncCompleter<T> {
    inner: Arc<Mutex<SignalInner<T>>>,
}

impl<T> Clone for AsyncCompleter<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> AsyncCompleter<T> {
    pub fn complete_ok(&self, value: T) {
        self.set_result(Ok(value));
    }

    pub fn complete_err(&self, message: impl Into<String>) {
        self.set_result(Err(message.into()));
    }

    fn set_result(&self, value: std::result::Result<T, String>) {
        let waker = {
            let mut guard = self.inner.lock().unwrap();
            if guard.state != SignalState::Pending {
                // Duplicate or late completion: leave original result intact
                return;
            }
            guard.state = SignalState::Ready;
            guard.result = Some(value);
            guard.waker.take()
        };

        if let Some(waker) = waker {
            waker.wake();
        }
    }

    pub fn is_completed(&self) -> bool {
        let guard = self.inner.lock().unwrap();
        guard.state != SignalState::Pending
    }

    pub fn state(&self) -> SignalState {
        let guard = self.inner.lock().unwrap();
        guard.state
    }
}

pub struct AsyncSignalFuture<T> {
    inner: Arc<Mutex<SignalInner<T>>>,
}

impl<T> Future for AsyncSignalFuture<T> {
    type Output = std::result::Result<T, String>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut guard = self.inner.lock().unwrap();
        match guard.state {
            SignalState::Ready => {
                guard.state = SignalState::Consumed;
                let res = guard
                    .result
                    .take()
                    .expect("AsyncSignalFuture: result must be present in Ready state");
                Poll::Ready(res)
            }
            SignalState::Consumed => {
                Poll::Ready(Err("AsyncSignalFuture: signal already consumed".to_string()))
            }
            SignalState::Pending => {
                guard.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}

pub fn signal_pair<T>() -> (AsyncCompleter<T>, AsyncSignalFuture<T>) {
    let inner = Arc::new(Mutex::new(SignalInner {
        state: SignalState::Pending,
        result: None,
        waker: None,
    }));
    (
        AsyncCompleter {
            inner: Arc::clone(&inner),
        },
        AsyncSignalFuture { inner },
    )
}
