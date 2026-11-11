//! Deterministic Clock & Timer abstractions for Lust runtime (DST Phase 2, D1).
//!
//! Provides:
//! - `WallClock`: Calendar/epoch time (`now_epoch_nanos()`) for `os.time()` / `os.date()`.
//! - `MonotonicClock`: Strictly monotonic nanosecond counter (`now_nanos()`) for timer deadlines.
//! - `TimerControl`: Virtual time advancement interface for simulation testing harness.
//! - `TimerQueue`: Deterministic min-heap timer queue with generation tokens and cancellation.
//! - `LustClock`: Unified clock struct supporting physical system time and simulated time.
//! - Civil date calculations (`epoch_secs_to_civil`) for deterministic `os.date()`.

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering as AtomicOrdering};
use core::time::Duration;

#[cfg(feature = "std")]
use std::collections::BinaryHeap;
#[cfg(feature = "std")]
use std::sync::Mutex;
#[cfg(feature = "std")]
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Unique identifier for a registered timer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TimerId(pub u64);

impl TimerId {
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

/// Abstract Wall Clock providing epoch time in nanoseconds since UNIX_EPOCH.
pub trait WallClock: Send + Sync {
    fn now_epoch_nanos(&self) -> u64;
}

/// Abstract Monotonic Clock providing monotonic time in nanoseconds for timer deadlines.
pub trait MonotonicClock: Send + Sync {
    fn now_nanos(&self) -> u64;
}

/// Control interface for virtual/simulated time, accessible to simulation harnesses and tests.
pub trait TimerControl: Send + Sync {
    /// Advance simulated monotonic and wall time by the given duration.
    fn advance(&self, duration: Duration);
    /// Return the current monotonic nanoseconds.
    fn now_nanos(&self) -> u64;
    /// Explicitly set the monotonic nanoseconds.
    fn set_now_nanos(&self, nanos: u64);
    /// Advance wall clock epoch time independently if needed.
    fn advance_wall(&self, duration: Duration);
    /// Returns true if this clock is simulated/virtual.
    fn is_simulated(&self) -> bool;
}

/// Real physical wall clock based on `std::time::SystemTime`.
pub struct SystemWallClock;

impl WallClock for SystemWallClock {
    fn now_epoch_nanos(&self) -> u64 {
        #[cfg(feature = "std")]
        {
            match SystemTime::now().duration_since(UNIX_EPOCH) {
                Ok(dur) => dur.as_nanos().min(u64::MAX as u128) as u64,
                Err(_) => 0,
            }
        }
        #[cfg(not(feature = "std"))]
        {
            0
        }
    }
}

/// Real physical monotonic clock based on `std::time::Instant`.
pub struct SystemMonotonicClock {
    #[cfg(feature = "std")]
    start: Instant,
}

impl Default for SystemMonotonicClock {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemMonotonicClock {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "std")]
            start: Instant::now(),
        }
    }
}

impl MonotonicClock for SystemMonotonicClock {
    fn now_nanos(&self) -> u64 {
        #[cfg(feature = "std")]
        {
            self.start.elapsed().as_nanos().min(u64::MAX as u128) as u64
        }
        #[cfg(not(feature = "std"))]
        {
            0
        }
    }
}

/// Deterministic Simulated Clock for reproducible testing.
pub struct SimClock {
    epoch_nanos: AtomicU64,
    monotonic_nanos: AtomicU64,
}

impl SimClock {
    pub fn new(initial_epoch_nanos: u64, initial_monotonic_nanos: u64) -> Self {
        Self {
            epoch_nanos: AtomicU64::new(initial_epoch_nanos),
            monotonic_nanos: AtomicU64::new(initial_monotonic_nanos),
        }
    }
}

impl WallClock for SimClock {
    fn now_epoch_nanos(&self) -> u64 {
        self.epoch_nanos.load(AtomicOrdering::SeqCst)
    }
}

impl MonotonicClock for SimClock {
    fn now_nanos(&self) -> u64 {
        self.monotonic_nanos.load(AtomicOrdering::SeqCst)
    }
}

impl TimerControl for SimClock {
    fn advance(&self, duration: Duration) {
        let nanos = duration.as_nanos().min(u64::MAX as u128) as u64;
        self.monotonic_nanos.fetch_add(nanos, AtomicOrdering::SeqCst);
        self.epoch_nanos.fetch_add(nanos, AtomicOrdering::SeqCst);
    }

    fn now_nanos(&self) -> u64 {
        self.monotonic_nanos.load(AtomicOrdering::SeqCst)
    }

    fn set_now_nanos(&self, nanos: u64) {
        self.monotonic_nanos.store(nanos, AtomicOrdering::SeqCst);
    }

    fn advance_wall(&self, duration: Duration) {
        let nanos = duration.as_nanos().min(u64::MAX as u128) as u64;
        self.epoch_nanos.fetch_add(nanos, AtomicOrdering::SeqCst);
    }

    fn is_simulated(&self) -> bool {
        true
    }
}

/// Internal scheduled timer item in min-heap.
#[cfg(feature = "std")]
struct ScheduledTimer {
    id: TimerId,
    generation: u32,
    deadline_nanos: u64,
    sequence: u64,
    cancelled: AtomicBool,
    action: Mutex<Option<Box<dyn FnOnce() + Send + 'static>>>,
}

#[cfg(feature = "std")]
impl PartialEq for ScheduledTimer {
    fn eq(&self, other: &Self) -> bool {
        self.deadline_nanos == other.deadline_nanos && self.sequence == other.sequence
    }
}

#[cfg(feature = "std")]
impl Eq for ScheduledTimer {}

#[cfg(feature = "std")]
impl PartialOrd for ScheduledTimer {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(feature = "std")]
impl Ord for ScheduledTimer {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: earliest deadline has highest priority
        other
            .deadline_nanos
            .cmp(&self.deadline_nanos)
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

/// Thread-safe deterministic timer queue managing deadlines, generations, and cancellations.
pub struct TimerQueue {
    next_timer_id: AtomicU64,
    next_sequence: AtomicU64,
    #[cfg(feature = "std")]
    heap: Mutex<BinaryHeap<ScheduledTimer>>,
    active_count: AtomicU32,
}

impl Default for TimerQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl TimerQueue {
    pub fn new() -> Self {
        Self {
            next_timer_id: AtomicU64::new(1),
            next_sequence: AtomicU64::new(1),
            #[cfg(feature = "std")]
            heap: Mutex::new(BinaryHeap::new()),
            active_count: AtomicU32::new(0),
        }
    }

    /// Schedule an action to be executed when monotonic now >= deadline_nanos.
    pub fn schedule<F>(&self, deadline_nanos: u64, generation: u32, action: F) -> TimerId
    where
        F: FnOnce() + Send + 'static,
    {
        let id = TimerId(self.next_timer_id.fetch_add(1, AtomicOrdering::SeqCst));
        let sequence = self.next_sequence.fetch_add(1, AtomicOrdering::SeqCst);

        #[cfg(feature = "std")]
        {
            let entry = ScheduledTimer {
                id,
                generation,
                deadline_nanos,
                sequence,
                cancelled: AtomicBool::new(false),
                action: Mutex::new(Some(Box::new(action))),
            };
            let mut heap = self.heap.lock().unwrap();
            heap.push(entry);
            self.active_count.fetch_add(1, AtomicOrdering::SeqCst);
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = (deadline_nanos, generation, action, sequence);
        }

        id
    }

    /// Cancel a timer by ID and generation token.
    /// Returns true if the timer was found, active, and successfully cancelled.
    pub fn cancel(&self, id: TimerId, generation: u32) -> bool {
        #[cfg(feature = "std")]
        {
            let heap = self.heap.lock().unwrap();
            for entry in heap.iter() {
                if entry.id == id && entry.generation == generation {
                    if !entry.cancelled.swap(true, AtomicOrdering::SeqCst) {
                        let mut action_lock = entry.action.lock().unwrap();
                        let was_active = action_lock.take().is_some();
                        if was_active {
                            self.active_count.fetch_sub(1, AtomicOrdering::SeqCst);
                        }
                        return was_active;
                    }
                    return false;
                }
            }
            false
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = (id, generation);
            false
        }
    }

    /// Poll all expired, uncancelled actions whose deadline <= now_nanos.
    pub fn poll_expired(&self, now_nanos: u64) -> Vec<Box<dyn FnOnce() + Send + 'static>> {
        #[cfg(feature = "std")]
        {
            let mut expired_actions = Vec::new();
            let mut heap = self.heap.lock().unwrap();

            while let Some(top) = heap.peek() {
                if top.deadline_nanos <= now_nanos {
                    let entry = heap.pop().unwrap();
                    if !entry.cancelled.load(AtomicOrdering::SeqCst) {
                        let mut action_lock = entry.action.lock().unwrap();
                        if let Some(action) = action_lock.take() {
                            self.active_count.fetch_sub(1, AtomicOrdering::SeqCst);
                            expired_actions.push(action);
                        }
                    }
                } else {
                    break;
                }
            }

            expired_actions
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = now_nanos;
            Vec::new()
        }
    }

    /// Peek the earliest deadline of an uncancelled timer.
    pub fn next_deadline(&self) -> Option<u64> {
        #[cfg(feature = "std")]
        {
            let mut heap = self.heap.lock().unwrap();
            while let Some(top) = heap.peek() {
                if top.cancelled.load(AtomicOrdering::SeqCst) {
                    heap.pop();
                } else {
                    return Some(top.deadline_nanos);
                }
            }
            None
        }
        #[cfg(not(feature = "std"))]
        {
            None
        }
    }

    pub fn active_count(&self) -> u32 {
        self.active_count.load(AtomicOrdering::SeqCst)
    }

    pub fn is_empty(&self) -> bool {
        self.active_count() == 0
    }
}

/// LustClock unifies WallClock, MonotonicClock, and TimerQueue.
///
/// Supports instant switching between production host time (`LustClock::system()`)
/// and deterministic simulated time (`LustClock::simulated(epoch_nanos, monotonic_nanos)`).
pub struct LustClock {
    wall: Arc<dyn WallClock>,
    monotonic: Arc<dyn MonotonicClock>,
    control: Option<Arc<dyn TimerControl>>,
    timers: Arc<TimerQueue>,
}

impl Default for LustClock {
    fn default() -> Self {
        Self::system()
    }
}

impl LustClock {
    /// Create real production system clock backed by host OS time.
    pub fn system() -> Self {
        Self {
            wall: Arc::new(SystemWallClock),
            monotonic: Arc::new(SystemMonotonicClock::new()),
            control: None,
            timers: Arc::new(TimerQueue::new()),
        }
    }

    /// Create deterministic simulated clock with starting epoch and monotonic nanoseconds.
    pub fn simulated(initial_epoch_nanos: u64, initial_monotonic_nanos: u64) -> Self {
        let sim = Arc::new(SimClock::new(initial_epoch_nanos, initial_monotonic_nanos));
        Self {
            wall: sim.clone(),
            monotonic: sim.clone(),
            control: Some(sim),
            timers: Arc::new(TimerQueue::new()),
        }
    }

    pub fn new(
        wall: Arc<dyn WallClock>,
        monotonic: Arc<dyn MonotonicClock>,
        control: Option<Arc<dyn TimerControl>>,
        timers: Arc<TimerQueue>,
    ) -> Self {
        Self {
            wall,
            monotonic,
            control,
            timers,
        }
    }

    /// Current epoch timestamp in nanoseconds since UNIX_EPOCH.
    pub fn now_epoch_nanos(&self) -> u64 {
        self.wall.now_epoch_nanos()
    }

    /// Current epoch timestamp in floating-point seconds.
    pub fn now_epoch_secs_f64(&self) -> f64 {
        (self.now_epoch_nanos() as f64) / 1_000_000_000.0
    }

    /// Current strictly monotonic timestamp in nanoseconds.
    pub fn now_nanos(&self) -> u64 {
        self.monotonic.now_nanos()
    }

    /// Check if virtual/simulated time is active.
    pub fn is_simulated(&self) -> bool {
        self.control.is_some()
    }

    /// Access timer control interface if running under simulation.
    pub fn control(&self) -> Option<&Arc<dyn TimerControl>> {
        self.control.as_ref()
    }

    /// Access the underlying timer queue.
    pub fn timer_queue(&self) -> &Arc<TimerQueue> {
        &self.timers
    }

    /// Advance simulated time. If running on system clock, this is a no-op.
    pub fn advance(&self, duration: Duration) {
        if let Some(ctrl) = &self.control {
            ctrl.advance(duration);
        }
    }

    /// Schedule an action at `deadline_nanos`.
    pub fn schedule_timer<F>(&self, deadline_nanos: u64, generation: u32, action: F) -> TimerId
    where
        F: FnOnce() + Send + 'static,
    {
        self.timers.schedule(deadline_nanos, generation, action)
    }

    /// Cancel a scheduled timer.
    pub fn cancel_timer(&self, timer_id: TimerId, generation: u32) -> bool {
        self.timers.cancel(timer_id, generation)
    }

    /// Poll expired timers up to `now_nanos`.
    pub fn poll_expired_timers(&self, now_nanos: u64) -> Vec<Box<dyn FnOnce() + Send + 'static>> {
        self.timers.poll_expired(now_nanos)
    }

    /// Earliest deadline among active timers.
    pub fn next_deadline(&self) -> Option<u64> {
        self.timers.next_deadline()
    }
}

/// Civil calendar date representation for `os.date` formatting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CivilDateTime {
    pub year: i32,
    pub month: u32,  // 1..=12
    pub day: u32,    // 1..=31
    pub hour: u32,   // 0..=23
    pub min: u32,    // 0..=59
    pub sec: u32,    // 0..=59
    pub wday: u32,   // 1..=7 (1 = Sunday, 7 = Saturday, Lua convention)
    pub yday: u32,   // 1..=366
    pub isdst: bool, // always false for UTC
}

/// Convert seconds since UNIX epoch (1970-01-01 00:00:00 UTC) to civil date components.
/// Uses Howard Hinnant's algorithm for exact calendar math across full signed range.
pub fn epoch_secs_to_civil(epoch_secs: i64) -> CivilDateTime {
    let days = epoch_secs.div_euclid(86400);
    let rem = epoch_secs.rem_euclid(86400);
    let hour = (rem / 3600) as u32;
    let min = ((rem % 3600) / 60) as u32;
    let sec = (rem % 60) as u32;

    // 1970-01-01 was Thursday.
    // Sunday is 1, Thursday is 5.
    let wday = (((days + 4).rem_euclid(7)) + 1) as u32;

    // Howard Hinnant algorithm
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1020 + doe / 1461 - doe / 36524) / 365;
    let y = (yoe as i32) + (era as i32) * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };

    let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    let month_days = [
        31,
        if is_leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut yday = d;
    for i in 0..(m.saturating_sub(1) as usize) {
        yday += month_days[i];
    }

    CivilDateTime {
        year,
        month: m,
        day: d,
        hour,
        min,
        sec,
        wday,
        yday,
        isdst: false,
    }
}

impl CivilDateTime {
    pub fn format_default(&self) -> String {
        self.format("%Y-%m-%d %H:%M:%S")
    }

    pub fn format(&self, fmt: &str) -> String {
        let mut result = String::new();
        let mut chars = fmt.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '%' {
                if let Some(spec) = chars.next() {
                    match spec {
                        'Y' => result.push_str(&format!("{:04}", self.year)),
                        'y' => result.push_str(&format!("{:02}", (self.year % 100).abs())),
                        'm' => result.push_str(&format!("{:02}", self.month)),
                        'd' => result.push_str(&format!("{:02}", self.day)),
                        'H' => result.push_str(&format!("{:02}", self.hour)),
                        'M' => result.push_str(&format!("{:02}", self.min)),
                        'S' => result.push_str(&format!("{:02}", self.sec)),
                        'w' => result.push_str(&format!("{}", (self.wday + 6) % 7)), // 0 = Sunday .. 6 = Saturday
                        'j' => result.push_str(&format!("{:03}", self.yday)),
                        'c' => result.push_str(&format!(
                            "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
                            self.year, self.month, self.day, self.hour, self.min, self.sec
                        )),
                        '%' => result.push('%'),
                        other => {
                            result.push('%');
                            result.push(other);
                        }
                    }
                } else {
                    result.push('%');
                }
            } else {
                result.push(ch);
            }
        }

        result
    }
}
