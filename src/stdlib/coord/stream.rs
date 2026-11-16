//! Coordination Stream: chunk-oriented lazy pull streams with backpressure.
//!
//! Complies with `RFC-STDLIB-ASYNC-COORDINATION` (§3.4):
//! - Chunk-oriented pull execution (minimizing VM context switches; default chunk size 64).
//! - Composable operators: `map`, `filter`, `take`, `drop`, `take_while`, `chunks`, `concat`, `buffer`.
//! - Seamless execution into `Sink` targets via `run(sink)`.
//! - Integrates with `Queue` and `Channel` for bounded pipeline backpressure.

use alloc::{boxed::Box, string::String, vec, vec::Vec};
use crate::rx::buffer::OverflowStrategy;
use super::queue::{Queue, QueueError};
use super::sink::{Sink, SinkControl};
use crate::vm::channel::Channel;

/// Result of pulling the next step of a `Stream`.
pub enum StreamStep<T> {
    /// A chunk of available items.
    Chunk(Vec<T>),
    /// Stream is exhausted and closed.
    Done,
    /// An error occurred in the upstream producer.
    Error(String),
}

/// A lazy, chunk-oriented pull stream of elements of type `T`.
pub struct Stream<T> {
    pull: Box<dyn FnMut() -> StreamStep<T>>,
}

impl<T: 'static> Stream<T> {
    /// Creates a stream from a raw chunk pull closure.
    pub fn new<F: FnMut() -> StreamStep<T> + 'static>(pull: F) -> Self {
        Self {
            pull: Box::new(pull),
        }
    }

    /// Pulls the next chunk from this stream.
    pub fn pull_chunk(&mut self) -> StreamStep<T> {
        (self.pull)()
    }

    /// Creates an empty stream that immediately yields `Done`.
    pub fn empty() -> Self {
        Self::new(|| StreamStep::Done)
    }

    /// Creates a stream from an array / `Vec<T>` with default chunk size of 64.
    pub fn from_array(items: Vec<T>) -> Self {
        Self::from_array_chunked(items, 64)
    }

    /// Creates a stream from an array / `Vec<T>` with a custom chunk size.
    pub fn from_array_chunked(mut items: Vec<T>, chunk_size: usize) -> Self {
        let chunk_size = chunk_size.max(1);
        Self::new(move || {
            if items.is_empty() {
                StreamStep::Done
            } else {
                let take_count = items.len().min(chunk_size);
                let chunk: Vec<T> = items.drain(..take_count).collect();
                StreamStep::Chunk(chunk)
            }
        })
    }

    /// Creates a stream from any standard iterator.
    pub fn from_iter<I: IntoIterator<Item = T> + 'static>(iter: I) -> Self
    where
        I::IntoIter: 'static,
    {
        let mut it = iter.into_iter();
        Self::new(move || {
            let mut chunk = Vec::with_capacity(64);
            for _ in 0..64 {
                if let Some(item) = it.next() {
                    chunk.push(item);
                } else {
                    break;
                }
            }
            if chunk.is_empty() {
                StreamStep::Done
            } else {
                StreamStep::Chunk(chunk)
            }
        })
    }

    /// Maps each element of this stream using function `f`.
    pub fn map<U: 'static, F: FnMut(T) -> U + 'static>(mut self, mut f: F) -> Stream<U> {
        Stream::new(move || match self.pull_chunk() {
            StreamStep::Chunk(chunk) => {
                let mapped = chunk.into_iter().map(&mut f).collect();
                StreamStep::Chunk(mapped)
            }
            StreamStep::Done => StreamStep::Done,
            StreamStep::Error(e) => StreamStep::Error(e),
        })
    }

    /// Filters elements of this stream using predicate `pred`.
    pub fn filter<F: FnMut(&T) -> bool + 'static>(mut self, mut pred: F) -> Self {
        Stream::new(move || loop {
            match self.pull_chunk() {
                StreamStep::Chunk(chunk) => {
                    let filtered: Vec<T> = chunk.into_iter().filter(&mut pred).collect();
                    if !filtered.is_empty() {
                        return StreamStep::Chunk(filtered);
                    }
                    // If filtered chunk became empty, continue pulling
                }
                StreamStep::Done => return StreamStep::Done,
                StreamStep::Error(e) => return StreamStep::Error(e),
            }
        })
    }

    /// Takes at most `n` items from this stream and then closes immediately.
    pub fn take(mut self, n: usize) -> Self {
        let mut remaining = n;
        Stream::new(move || {
            if remaining == 0 {
                return StreamStep::Done;
            }
            match self.pull_chunk() {
                StreamStep::Chunk(mut chunk) => {
                    if chunk.len() > remaining {
                        chunk.truncate(remaining);
                        remaining = 0;
                        StreamStep::Chunk(chunk)
                    } else {
                        remaining -= chunk.len();
                        StreamStep::Chunk(chunk)
                    }
                }
                StreamStep::Done => StreamStep::Done,
                StreamStep::Error(e) => StreamStep::Error(e),
            }
        })
    }

    /// Drops the first `n` items from this stream.
    pub fn drop(mut self, n: usize) -> Self {
        let mut to_drop = n;
        Stream::new(move || loop {
            match self.pull_chunk() {
                StreamStep::Chunk(mut chunk) => {
                    if to_drop == 0 {
                        return StreamStep::Chunk(chunk);
                    }
                    if chunk.len() <= to_drop {
                        to_drop -= chunk.len();
                        // entirely dropped, continue pulling
                    } else {
                        chunk.drain(..to_drop);
                        to_drop = 0;
                        return StreamStep::Chunk(chunk);
                    }
                }
                StreamStep::Done => return StreamStep::Done,
                StreamStep::Error(e) => return StreamStep::Error(e),
            }
        })
    }

    /// Concatenates this stream with `other`: plays all items of `self`, then all items of `other`.
    pub fn concat(mut self, mut other: Stream<T>) -> Self {
        let mut first_done = false;
        Stream::new(move || {
            if !first_done {
                match self.pull_chunk() {
                    StreamStep::Chunk(chunk) => return StreamStep::Chunk(chunk),
                    StreamStep::Done => {
                        first_done = true;
                    }
                    StreamStep::Error(e) => return StreamStep::Error(e),
                }
            }
            other.pull_chunk()
        })
    }

    /// Groups stream elements into fixed-size chunks of size `chunk_size`.
    pub fn chunks(mut self, chunk_size: usize) -> Stream<Vec<T>> {
        let chunk_size = chunk_size.max(1);
        let mut pending = Vec::new();
        Stream::new(move || loop {
            if pending.len() >= chunk_size {
                let chunk: Vec<T> = pending.drain(..chunk_size).collect();
                return StreamStep::Chunk(vec![chunk]);
            }
            match self.pull_chunk() {
                StreamStep::Chunk(items) => {
                    pending.extend(items);
                    if pending.len() >= chunk_size {
                        let chunk: Vec<T> = pending.drain(..chunk_size).collect();
                        return StreamStep::Chunk(vec![chunk]);
                    }
                }
                StreamStep::Done => {
                    if !pending.is_empty() {
                        let rem = core::mem::take(&mut pending);
                        return StreamStep::Chunk(vec![rem]);
                    }
                    return StreamStep::Done;
                }
                StreamStep::Error(e) => return StreamStep::Error(e),
            }
        })
    }

    /// Executes this stream into a `Sink` until the stream is exhausted or the sink stops.
    pub fn run<Res: 'static>(mut self, mut sink: Sink<T, Res>) -> Result<Res, String> {
        loop {
            match self.pull_chunk() {
                StreamStep::Chunk(chunk) => {
                    let control = sink.step(chunk)?;
                    if control == SinkControl::Stop {
                        break;
                    }
                }
                StreamStep::Done => break,
                StreamStep::Error(err) => return Err(err),
            }
        }
        sink.finish()
    }

    /// Collects all stream elements into a `Vec<T>`.
    pub fn collect(self) -> Result<Vec<T>, String>
    where
        T: Clone,
    {
        self.run(Sink::collect())
    }

    /// Drains and discards all stream elements, returning the total count.
    pub fn drain(self) -> Result<usize, String> {
        self.run(Sink::drain())
    }

    /// Folds all elements of this stream with initial accumulator `init` and folder `f`.
    pub fn fold<Res: Clone + 'static, F: FnMut(Res, T) -> Res + 'static>(
        self,
        init: Res,
        f: F,
    ) -> Result<Res, String> {
        self.run(Sink::fold(init, f))
    }

    /// Iterates through all stream elements, applying `f` to each element.
    pub fn for_each<F: FnMut(T) + 'static>(self, f: F) -> Result<(), String> {
        self.run(Sink::for_each(f))
    }
}

impl Stream<i64> {
    /// Creates a numeric range stream `[start, end)` advancing by `step`.
    pub fn range(start: i64, end: i64, step: i64) -> Self {
        let step = if step == 0 { 1 } else { step };
        let mut curr = start;
        Self::new(move || {
            let mut chunk = Vec::with_capacity(64);
            while (step > 0 && curr < end) || (step < 0 && curr > end) {
                chunk.push(curr);
                curr += step;
                if chunk.len() >= 64 {
                    break;
                }
            }
            if chunk.is_empty() {
                StreamStep::Done
            } else {
                StreamStep::Chunk(chunk)
            }
        })
    }
}

impl<T: Clone + 'static> Stream<T> {
    /// Creates a stream that repeats `item` indefinitely or at most `count` times.
    pub fn repeat(item: T, count: Option<usize>) -> Self {
        let mut remaining = count;
        Self::new(move || {
            if let Some(0) = remaining {
                return StreamStep::Done;
            }
            let batch = match remaining {
                Some(rem) => {
                    let take_count = rem.min(64);
                    remaining = Some(rem - take_count);
                    take_count
                }
                None => 64,
            };
            StreamStep::Chunk(vec![item.clone(); batch])
        })
    }

    /// Wraps this stream with a bounded backpressure queue buffer.
    pub fn buffer(self, capacity: usize, overflow: OverflowStrategy) -> Stream<T> {
        let mut q = Queue::make(capacity, overflow);
        let mut upstream = self;
        let mut upstream_exhausted = false;
        let mut pending: std::collections::VecDeque<T> = std::collections::VecDeque::new();

        Stream::new(move || {
            loop {
                // First feed pending into queue
                while let Some(item) = pending.pop_front() {
                    match q.try_offer(item.clone()) {
                        Ok(true) => {}
                        Ok(false) | Err(QueueError::Full) => {
                            // Queue is full: keep item and stop offering until drained
                            pending.push_front(item);
                            break;
                        }
                        Err(e) => return StreamStep::Error(e.to_string()),
                    }
                }

                // If pending is empty and upstream is not exhausted and queue is not full, pull more
                if pending.is_empty() && !upstream_exhausted && !q.is_full() {
                    match upstream.pull_chunk() {
                        StreamStep::Chunk(items) => {
                            pending.extend(items);
                            continue;
                        }
                        StreamStep::Done => {
                            upstream_exhausted = true;
                        }
                        StreamStep::Error(e) => return StreamStep::Error(e),
                    }
                }

                break;
            }

            // Drain batch from queue
            let mut chunk = Vec::with_capacity(64);
            while let Ok(Some(item)) = q.try_take() {
                chunk.push(item);
                if chunk.len() >= 64 {
                    break;
                }
            }

            if chunk.is_empty() {
                if upstream_exhausted && pending.is_empty() && q.is_empty() {
                    StreamStep::Done
                } else {
                    StreamStep::Chunk(Vec::new())
                }
            } else {
                StreamStep::Chunk(chunk)
            }
        })
    }

    /// Creates a stream that continuously drains items from a `Queue<T>` until closed.
    pub fn from_queue(mut queue: Queue<T>) -> Self {
        Self::new(move || {
            let mut chunk = Vec::with_capacity(64);
            while let Ok(Some(item)) = queue.try_take() {
                chunk.push(item);
                if chunk.len() >= 64 {
                    break;
                }
            }
            if chunk.is_empty() {
                if queue.is_closed() {
                    StreamStep::Done
                } else {
                    StreamStep::Chunk(Vec::new())
                }
            } else {
                StreamStep::Chunk(chunk)
            }
        })
    }

    /// Creates a stream that continuously drains items from a `Channel<T>` until closed.
    pub fn from_channel(ch: Channel<T>) -> Self {
        Self::new(move || {
            let mut chunk = Vec::with_capacity(64);
            while let Ok(Some(item)) = ch.try_recv() {
                chunk.push(item);
                if chunk.len() >= 64 {
                    break;
                }
            }
            if chunk.is_empty() {
                if ch.is_closed() {
                    StreamStep::Done
                } else {
                    StreamStep::Chunk(Vec::new())
                }
            } else {
                StreamStep::Chunk(chunk)
            }
        })
    }
}
