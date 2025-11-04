use crate::bytecode::{TaskHandle, Value};
use hashbrown::HashMap;
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

pub(crate) type AsyncValueFuture =
    Pin<Box<dyn Future<Output = std::result::Result<Value, String>>>>;

pub(crate) struct AsyncRegistry {
    pub(crate) next_id: u64,
    pub(crate) pending: HashMap<u64, AsyncTaskEntry>,
}

impl AsyncRegistry {
    pub(crate) fn new() -> Self {
        Self {
            next_id: 1,
            pending: HashMap::new(),
        }
    }

    pub(crate) fn register(&mut self, entry: AsyncTaskEntry) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
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
}

pub(crate) struct AsyncTaskEntry {
    pub(crate) target: AsyncTaskTarget,
    pub(crate) future: AsyncValueFuture,
    wake_flag: Arc<WakeFlag>,
    immediate_poll: bool,
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
            wake_flag: Arc::new(WakeFlag::new()),
            immediate_poll: true,
        }
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

struct WakeFlag {
    pending: AtomicBool,
}

impl WakeFlag {
    fn new() -> Self {
        Self {
            pending: AtomicBool::new(true),
        }
    }

    fn take(&self) -> bool {
        self.pending.swap(false, Ordering::SeqCst)
    }

    fn wake(&self) {
        self.pending.store(true, Ordering::SeqCst);
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
    let arc = Arc::<WakeFlag>::from_raw(ptr as *const WakeFlag);
    let cloned = arc.clone();
    std::mem::forget(arc);
    RawWaker::new(Arc::into_raw(cloned) as *const (), &ASYNC_WAKER_VTABLE)
}

unsafe fn async_waker_wake(ptr: *const ()) {
    let arc = Arc::<WakeFlag>::from_raw(ptr as *const WakeFlag);
    arc.wake();
}

unsafe fn async_waker_wake_by_ref(ptr: *const ()) {
    let arc = Arc::<WakeFlag>::from_raw(ptr as *const WakeFlag);
    arc.wake();
    std::mem::forget(arc);
}

unsafe fn async_waker_drop(ptr: *const ()) {
    let _ = Arc::<WakeFlag>::from_raw(ptr as *const WakeFlag);
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
    pub(crate) fn new(task: TaskHandle, args: Args, completer: AsyncCompleter<R>) -> Self {
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

struct AsyncSignalState<T> {
    result: Mutex<Option<std::result::Result<T, String>>>,
    waker: Mutex<Option<Waker>>,
}

pub(crate) struct AsyncCompleter<T> {
    inner: Arc<AsyncSignalState<T>>,
}

impl<T> Clone for AsyncCompleter<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> AsyncCompleter<T> {
    fn complete_ok(&self, value: T) {
        self.set_result(Ok(value));
    }

    fn complete_err(&self, message: impl Into<String>) {
        self.set_result(Err(message.into()));
    }

    fn set_result(&self, value: std::result::Result<T, String>) {
        {
            let mut guard = self.inner.result.lock().unwrap();
            if guard.is_some() {
                return;
            }
            *guard = Some(value);
        }

        if let Some(waker) = self.inner.waker.lock().unwrap().take() {
            waker.wake();
        }
    }
}

pub(crate) struct AsyncSignalFuture<T> {
    inner: Arc<AsyncSignalState<T>>,
}

impl<T> Future for AsyncSignalFuture<T> {
    type Output = std::result::Result<T, String>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        {
            let mut guard = self.inner.result.lock().unwrap();
            if let Some(result) = guard.take() {
                return Poll::Ready(result);
            }
        }

        let mut waker_slot = self.inner.waker.lock().unwrap();
        *waker_slot = Some(cx.waker().clone());
        Poll::Pending
    }
}

pub(crate) fn signal_pair<T>() -> (AsyncCompleter<T>, AsyncSignalFuture<T>) {
    let inner = Arc::new(AsyncSignalState {
        result: Mutex::new(None),
        waker: Mutex::new(None),
    });
    (
        AsyncCompleter {
            inner: Arc::clone(&inner),
        },
        AsyncSignalFuture { inner },
    )
}
