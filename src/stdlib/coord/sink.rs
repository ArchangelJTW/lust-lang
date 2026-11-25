//! Coordination Sink: stream consumers, reducers, and channel/queue sinks.
//!
//! Complies with `RFC-STDLIB-ASYNC-COORDINATION` (§3.4):
//! - Stream consumer (`collect`, `drain`, `fold`, `for_each`, `head`, `last`, `queue`, `channel`).
//! - Early-stop control (`SinkControl::Stop`) for bounded downstream consumption.
//! - Safe composition with `Stream::run`.

use alloc::{boxed::Box, rc::Rc, string::String, vec::Vec};
use core::cell::RefCell;
use super::queue::Queue;
use crate::vm::channel::Channel;

/// Control flow signal emitted by a sink after processing a chunk of items.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SinkControl {
    /// Continue pulling chunks from upstream.
    Continue,
    /// Downstream consumption is complete; stop pulling upstream immediately.
    Stop,
}

/// A consumer of stream chunks that computes a final result `Res`.
pub struct Sink<In, Res = ()> {
    consume_chunk: Box<dyn FnMut(Vec<In>) -> Result<SinkControl, String>>,
    finish: Box<dyn FnOnce() -> Result<Res, String>>,
}

impl<In: 'static, Res: 'static> Sink<In, Res> {
    /// Creates a custom sink from chunk processor and finisher closures.
    pub fn new<
        C: FnMut(Vec<In>) -> Result<SinkControl, String> + 'static,
        F: FnOnce() -> Result<Res, String> + 'static,
    >(
        consume_chunk: C,
        finish: F,
    ) -> Self {
        Self {
            consume_chunk: Box::new(consume_chunk),
            finish: Box::new(finish),
        }
    }

    /// Feeds a chunk of items into the sink.
    pub fn step(&mut self, chunk: Vec<In>) -> Result<SinkControl, String> {
        (self.consume_chunk)(chunk)
    }

    /// Finishes consumption and produces the final result.
    pub fn finish(self) -> Result<Res, String> {
        (self.finish)()
    }
}

impl<In: Clone + 'static> Sink<In, Vec<In>> {
    /// Creates a sink that collects all streamed elements into a single `Vec<In>`.
    pub fn collect() -> Self {
        let accumulated = Rc::new(RefCell::new(Vec::new()));
        let acc_clone = accumulated.clone();
        Self::new(
            move |chunk| {
                accumulated.borrow_mut().extend(chunk);
                Ok(SinkControl::Continue)
            },
            move || Ok(core::mem::take(&mut *acc_clone.borrow_mut())),
        )
    }
}

impl<In: 'static> Sink<In, usize> {
    /// Creates a sink that drains and discards all items, returning the total count.
    pub fn drain() -> Self {
        let count = Rc::new(RefCell::new(0usize));
        let count_clone = count.clone();
        Self::new(
            move |chunk| {
                *count.borrow_mut() += chunk.len();
                Ok(SinkControl::Continue)
            },
            move || {
                let res = *count_clone.borrow();
                Ok(res)
            },
        )
    }
}

impl<In: 'static> Sink<In, ()> {
    /// Creates a sink that invokes `f` for each individual element.
    pub fn for_each<F: FnMut(In) + 'static>(mut f: F) -> Self {
        Self::new(
            move |chunk| {
                for item in chunk {
                    f(item);
                }
                Ok(SinkControl::Continue)
            },
            || Ok(()),
        )
    }
}

impl<In: Clone + 'static> Sink<In, Option<In>> {
    /// Creates a sink that takes the first element and immediately signals `Stop`.
    pub fn head() -> Self {
        let first = Rc::new(RefCell::new(None));
        let first_clone = first.clone();
        Self::new(
            move |chunk| {
                if first.borrow().is_none() {
                    *first.borrow_mut() = chunk.into_iter().next();
                }
                Ok(SinkControl::Stop)
            },
            move || {
                let res = first_clone.borrow_mut().take();
                Ok(res)
            },
        )
    }

    /// Creates a sink that captures the last element seen before completion.
    pub fn last() -> Self {
        let last_val = Rc::new(RefCell::new(None));
        let last_clone = last_val.clone();
        Self::new(
            move |chunk| {
                if let Some(item) = chunk.into_iter().last() {
                    *last_val.borrow_mut() = Some(item);
                }
                Ok(SinkControl::Continue)
            },
            move || {
                let res = last_clone.borrow_mut().take();
                Ok(res)
            },
        )
    }
}

impl<In: 'static, Res: Clone + 'static> Sink<In, Res> {
    /// Creates a sink that folds/reduces all stream items into an accumulated result.
    pub fn fold<F: FnMut(Res, In) -> Res + 'static>(init: Res, mut f: F) -> Self {
        let acc = Rc::new(RefCell::new(init));
        let acc_clone = acc.clone();
        Self::new(
            move |chunk| {
                for item in chunk {
                    let old = acc.borrow().clone();
                    *acc.borrow_mut() = f(old, item);
                }
                Ok(SinkControl::Continue)
            },
            move || {
                let res = acc_clone.borrow().clone();
                Ok(res)
            },
        )
    }
}

impl<In: Clone + 'static> Sink<In, usize> {
    /// Creates a sink that pushes all elements into a `Queue<In>`.
    pub fn queue(mut q: Queue<In>) -> Self {
        let count = Rc::new(RefCell::new(0usize));
        let count_clone = count.clone();
        Self::new(
            move |chunk| {
                for item in chunk {
                    let _ = q.try_offer(item);
                    *count.borrow_mut() += 1;
                }
                Ok(SinkControl::Continue)
            },
            move || {
                let res = *count_clone.borrow();
                Ok(res)
            },
        )
    }

    /// Creates a sink that pushes all elements into a `Channel<In>`.
    pub fn channel(ch: Channel<In>) -> Self {
        let count = Rc::new(RefCell::new(0usize));
        let count_clone = count.clone();
        Self::new(
            move |chunk| {
                for item in chunk {
                    let _ = ch.try_send(item);
                    *count.borrow_mut() += 1;
                }
                Ok(SinkControl::Continue)
            },
            move || {
                let res = *count_clone.borrow();
                Ok(res)
            },
        )
    }
}
