//! Coordination Primitives Foundation for Lust (`R-COORD`).
//!
//! Complies with `RFC-STDLIB-ASYNC-COORDINATION` (§3.4) and `WAVE3_EXECUTION_PLAN.md` (3C):
//! - `Stream` & `Sink`: chunk-oriented lazy pull streams and composable consumers.
//! - `Queue`: multi-fiber FIFO queue with bounded backpressure policies.
//! - `Deferred`: single-assignment promise with synchronous multi-waiter wakeup.
//! - `Semaphore`: fair counting semaphore with head-of-line ordering and zero-leak RAII guard.
//! - `Ref`: deterministic shared mutable state cell for fibers.
//! - `Schedule`: composable recurrence, backoff, and timer-safe timeout policies.
//! - `builtins`: full native registration for `@std/coord`, `@std/sync`, `@std/stream`, `@std/schedule`.

pub mod builtins;
pub mod deferred;
pub mod queue;
pub mod ref_cell;
pub mod schedule;
pub mod semaphore;
pub mod sink;
pub mod stream;

pub use builtins::{register_builtins, reset_global_coord_registries};
pub use deferred::Deferred;
pub use queue::{OverflowStrategy, Queue, QueueError};
pub use ref_cell::Ref;
pub use schedule::{Schedule, ScheduleKind, ScheduleState, ScheduleStep};
pub use semaphore::{Semaphore, SemaphoreError};
pub use sink::{Sink, SinkControl};
pub use stream::{Stream, StreamStep};
