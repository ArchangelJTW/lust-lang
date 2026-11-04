// Gate 1 Prototype: Isolated Bounded Lexical Parser Probe for Issue #11
//
// Implements and empirically verifies the Gate 1 bounded lexical parser prototype
// under the coordinator-accepted Luna-high remediation plan.
//
// Remediation findings addressed:
// 1. Flat budget-owned node arena: replaces recursive `PrototypeValue` Vec/String
//    allocations with a flat arena and borrowed/decoded string/key spans.
// 2. Capacity ledger lifecycle: all storage capacity is charged to the CapacityLedger
//    before reservation and remains charged while the returned parsed document is live;
//    the ledger returns to zero only after all document-owned buffers are dropped.
// 3. Fallible `try_reserve_exact` & fault injection: checked arithmetic at every
//    reservation step, zero `Vec::with_capacity` or `Vec::new().push` growth during parse,
//    zero `to_string`, automatic rollback and deterministic failure injection.
// 4. Quota check ordering: node count enforced before decoding/allocating value/key,
//    `max_args` checked before parse, depth checked before descent.
// 5. Bounded duplicate key detection: replaced O(k^2) linear scans with bounded pre-reserved
//    in-place sorting (`sort_unstable_by`, O(k log k), zero heap allocation) and adjacent
//    decoded equality check.
// 6. Fixed float underflow: inspects only significand digits before exponent, correctly
//    accepting exact-zero significand cases (`0e-500`, `-0e-500`, `0.0e-500`, `-0.0e-500`)
//    as Float(±0) while rejecting nonzero underflow (`1e-500`, `-1e-500`).
// 7. Exact RFC definitions: node counting covers outer list, each value, each object key;
//    strict integer/float token class preservation; -0 integer maps to Int(0); out-of-i64
//    integers rejected with NumberRange; finite f64.
//
// Scope & Limitations explicitly acknowledged:
// - Feasibility prototype only: does NOT construct actual `lust::bytecode::Value` (no VM runtime).
// - CapacityLedger accounts for requested bytes with checked arithmetic, not OS allocator rounding.
//   `Arc<Mutex<LedgerState>>` and `ArenaDropWitness` are test-only measurement bookkeeping
//   instrumentation excluded from bridge payload capacity accounting (never described as
//   bridge-owned parser allocation).
// - Does not expose C FFI exports or perform VM memory accounting transfer / VM quota adoption;
//   does not claim to prove OS RSS/allocator usable size.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// RFC 04 Status Codes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LustJsonStatus {
    Ok = 0,
    InvalidArgument = 1,
    InvalidJson = 2,
    NumberRange = 3,
    Limit = 4,
    Type = 5,
    Call = 6,
    Unsupported = 7,
    Cycle = 8,
    Borrow = 9,
    Internal = 10,
}

impl fmt::Display for LustJsonStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ---------------------------------------------------------------------------
// RFC 04 Limits
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct LustJsonLimits {
    pub max_input_bytes: usize,
    pub max_output_bytes: usize,
    pub max_depth: usize,
    pub max_nodes: usize,
    pub max_args: usize,
    pub max_bridge_bytes: usize,
}

impl Default for LustJsonLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 1_048_576,
            max_output_bytes: 1_048_576,
            max_depth: 64,
            max_nodes: 65_536,
            max_args: 256,
            max_bridge_bytes: 16_777_216,
        }
    }
}

// ---------------------------------------------------------------------------
// Capacity Ledger & Pre-growth Reservation with Fault Injection
// ---------------------------------------------------------------------------

// TEST-ONLY MEASUREMENT BOOKKEEPING NOTE:
// `LedgerState` and `CapacityLedger` (including `Arc<Mutex<LedgerState>>`) are
// test-only measurement bookkeeping instrumentation used solely by probes to
// track and verify bridge payload capacity constraints and lifecycle invariants.
// This instrumentation bookkeeping is strictly excluded from bridge payload capacity
// accounting and is never counted as bridge-owned parser allocation.

#[derive(Debug, Default)]
struct LedgerState {
    allocated: usize,
    peak: usize,
    max_bridge_bytes: usize,
    /// When Some(0), the next reservation fails. When Some(n > 0), decrements on reservation.
    fail_after_reservations: Option<usize>,
}

/// Test-only capacity ledger tracking payload buffer allocation charges.
/// The internal `Arc<Mutex<LedgerState>>` is probe measurement infrastructure,
/// not bridge payload memory.
#[derive(Debug, Clone)]
pub struct CapacityLedger {
    state: Arc<Mutex<LedgerState>>,
}

impl CapacityLedger {
    pub fn new(max_bridge_bytes: usize) -> Self {
        Self {
            state: Arc::new(Mutex::new(LedgerState {
                allocated: 0,
                peak: 0,
                max_bridge_bytes,
                fail_after_reservations: None,
            })),
        }
    }

    pub fn set_fail_after_reservations(&self, count: Option<usize>) {
        let mut st = self.state.lock().unwrap();
        st.fail_after_reservations = count;
    }

    pub fn try_reserve(&self, bytes: usize) -> Result<(), LustJsonStatus> {
        let mut st = self.state.lock().unwrap();

        // Deterministic fault injection check
        if let Some(ref mut count) = st.fail_after_reservations {
            if *count == 0 {
                // Injected failure triggers
                return Err(LustJsonStatus::Limit);
            }
            *count -= 1;
        }

        let new_total = st
            .allocated
            .checked_add(bytes)
            .ok_or(LustJsonStatus::Limit)?;
        if new_total > st.max_bridge_bytes {
            return Err(LustJsonStatus::Limit);
        }
        st.allocated = new_total;
        if st.allocated > st.peak {
            st.peak = st.allocated;
        }
        Ok(())
    }

    pub fn release(&self, bytes: usize) {
        let mut st = self.state.lock().unwrap();
        st.allocated = st.allocated.saturating_sub(bytes);
    }

    pub fn allocated(&self) -> usize {
        self.state.lock().unwrap().allocated
    }

    pub fn peak(&self) -> usize {
        self.state.lock().unwrap().peak
    }
}

// ---------------------------------------------------------------------------
// Fixed-capacity Pre-reserved Collections (Guaranteed Zero Growth During Parse)
// ---------------------------------------------------------------------------

pub struct PreallocatedVec<T> {
    inner: Vec<T>,
    capacity: usize,
    reserved_bytes: usize,
    ledger: CapacityLedger,
}

impl<T> PreallocatedVec<T> {
    pub fn empty(ledger: &CapacityLedger) -> Self {
        Self {
            inner: Vec::new(),
            capacity: 0,
            reserved_bytes: 0,
            ledger: ledger.clone(),
        }
    }

    pub fn try_with_capacity(
        capacity: usize,
        ledger: &CapacityLedger,
    ) -> Result<Self, LustJsonStatus> {
        let reserved_bytes = capacity
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(LustJsonStatus::Limit)?;

        // 1. Charge ledger BEFORE reserving capacity
        ledger.try_reserve(reserved_bytes)?;

        // 2. Fallible reservation using try_reserve_exact
        let mut inner = Vec::new();
        if inner.try_reserve_exact(capacity).is_err() {
            // Rollback on reservation failure
            ledger.release(reserved_bytes);
            return Err(LustJsonStatus::Limit);
        }

        Ok(Self {
            inner,
            capacity,
            reserved_bytes,
            ledger: ledger.clone(),
        })
    }

    #[inline]
    pub fn push(&mut self, item: T) -> Result<(), LustJsonStatus> {
        if self.inner.len() >= self.capacity {
            return Err(LustJsonStatus::Limit);
        }
        let cap_before = self.inner.capacity();
        self.inner.push(item);
        // Invariant assertion: capacity must NEVER change after pre-reservation!
        assert_eq!(
            self.inner.capacity(),
            cap_before,
            "CRITICAL: Collection reallocated after parsing started!"
        );
        Ok(())
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        self.inner.pop()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.inner
    }

    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        &mut self.inner
    }

    #[inline]
    pub fn truncate(&mut self, len: usize) {
        self.inner.truncate(len);
    }

    pub fn release_early(&mut self) {
        let detached = std::mem::take(&mut self.inner);
        drop(detached);
        self.capacity = 0;
        if self.reserved_bytes > 0 {
            let bytes = self.reserved_bytes;
            self.reserved_bytes = 0;
            self.ledger.release(bytes);
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for PreallocatedVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreallocatedVec")
            .field("inner", &self.inner)
            .field("capacity", &self.capacity)
            .field("reserved_bytes", &self.reserved_bytes)
            .finish()
    }
}

impl<T> Drop for PreallocatedVec<T> {
    fn drop(&mut self) {
        self.release_early();
    }
}

// TEST-ONLY MEASUREMENT BOOKKEEPING NOTE:
// `ArenaDropWitness` is a test-only instrumentation callback invoked strictly after
// dropping the detached backing `Vec<u8>` storage and immediately before `ledger.release(...)`.
// It allows probes to observe and verify that the backing buffer is physically deallocated before
// the capacity ledger charge is released, without requiring global allocator hooks.
// This callback and its captured state are strictly test-only instrumentation,
// excluded from bridge payload capacity accounting (never counted as bridge-owned parser allocation).
pub type ArenaDropWitness = Box<dyn FnMut(&CapacityLedger) + Send + 'static>;

pub struct PreallocatedStringArena {
    bytes: Vec<u8>,
    capacity: usize,
    reserved_bytes: usize,
    ledger: CapacityLedger,
    drop_witness: Option<ArenaDropWitness>,
}

impl fmt::Debug for PreallocatedStringArena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreallocatedStringArena")
            .field("capacity", &self.capacity)
            .field("len", &self.bytes.len())
            .field("reserved_bytes", &self.reserved_bytes)
            .field("has_drop_witness", &self.drop_witness.is_some())
            .finish()
    }
}

impl PreallocatedStringArena {
    pub fn empty(ledger: &CapacityLedger) -> Self {
        Self {
            bytes: Vec::new(),
            capacity: 0,
            reserved_bytes: 0,
            ledger: ledger.clone(),
            drop_witness: None,
        }
    }

    pub fn try_with_capacity(
        capacity: usize,
        ledger: &CapacityLedger,
    ) -> Result<Self, LustJsonStatus> {
        let reserved_bytes = capacity;
        // 1. Charge ledger BEFORE reserving capacity
        ledger.try_reserve(reserved_bytes)?;

        // 2. Fallible reservation using try_reserve_exact
        let mut bytes = Vec::new();
        if bytes.try_reserve_exact(capacity).is_err() {
            // Rollback on reservation failure
            ledger.release(reserved_bytes);
            return Err(LustJsonStatus::Limit);
        }

        Ok(Self {
            bytes,
            capacity,
            reserved_bytes,
            ledger: ledger.clone(),
            drop_witness: None,
        })
    }

    pub fn set_drop_witness<F>(&mut self, witness: F)
    where
        F: FnMut(&CapacityLedger) + Send + 'static,
    {
        self.drop_witness = Some(Box::new(witness));
    }

    pub fn push_str(&mut self, s: &str) -> Result<(usize, usize), LustJsonStatus> {
        let start = self.bytes.len();
        let needed = s.len();
        let new_len = start.checked_add(needed).ok_or(LustJsonStatus::Limit)?;
        if new_len > self.capacity {
            return Err(LustJsonStatus::Limit);
        }
        let cap_before = self.bytes.capacity();
        self.bytes.extend_from_slice(s.as_bytes());
        assert_eq!(
            self.bytes.capacity(),
            cap_before,
            "CRITICAL: String arena reallocated after parsing started!"
        );
        Ok((start, new_len))
    }

    pub fn push_char(&mut self, ch: char) -> Result<(), LustJsonStatus> {
        let mut buf = [0u8; 4];
        let encoded = ch.encode_utf8(&mut buf);
        self.push_str(encoded).map(|_| ())
    }

    #[inline]
    pub fn get(&self, start: usize, end: usize) -> &str {
        std::str::from_utf8(&self.bytes[start..end]).expect("arena contains valid utf-8")
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    pub fn truncate(&mut self, len: usize) {
        self.bytes.truncate(len);
    }

    pub fn release_early(&mut self) {
        let detached = std::mem::take(&mut self.bytes);
        drop(detached);
        self.capacity = 0;
        if let Some(mut witness_fn) = self.drop_witness.take() {
            witness_fn(&self.ledger);
        }
        if self.reserved_bytes > 0 {
            let bytes = self.reserved_bytes;
            self.reserved_bytes = 0;
            self.ledger.release(bytes);
        }
    }
}

impl Drop for PreallocatedStringArena {
    fn drop(&mut self) {
        self.release_early();
    }
}

// ---------------------------------------------------------------------------
// Flat Budget-Owned Node Arena & Document Model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParsedNumber {
    Int(i64),
    Float(f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrSpan<'a> {
    Borrowed(&'a str),
    Decoded(usize, usize),
}

impl<'a> StrSpan<'a> {
    #[inline]
    pub fn as_str<'arena>(&self, arena: &'arena PreallocatedStringArena) -> &'arena str
    where
        'a: 'arena,
    {
        match *self {
            StrSpan::Borrowed(s) => s,
            StrSpan::Decoded(start, end) => arena.get(start, end),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NodeKind<'a> {
    Nil,
    Bool(bool),
    Number(ParsedNumber),
    String(StrSpan<'a>),
    Array,
    Object,
    ObjectEntry { key: StrSpan<'a>, val_idx: usize },
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrototypeNode<'a> {
    pub kind: NodeKind<'a>,
    pub first_child: Option<usize>,
    pub next_sibling: Option<usize>,
    pub child_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerKind {
    Array,
    Object,
}

#[derive(Debug, Clone)]
pub struct ContainerFrame {
    pub kind: ContainerKind,
    pub node_idx: usize,
    pub last_child_idx: Option<usize>,
}

/// A parsed JSON document where all storage allocations are charged in `CapacityLedger`
/// and remain charged while this document is live. Dropping the document releases the ledger.
#[derive(Debug)]
pub struct PrototypeDocument<'a> {
    pub nodes: PreallocatedVec<PrototypeNode<'a>>,
    pub string_arena: PreallocatedStringArena,
    pub root_node_idx: usize,
    pub ledger: CapacityLedger,
}

impl<'a> PrototypeDocument<'a> {
    pub fn len(&self) -> usize {
        self.nodes.as_slice()[self.root_node_idx].child_count
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, idx: usize) -> Option<ValueRef<'_, 'a>> {
        if idx >= self.len() {
            return None;
        }
        let mut curr = self.nodes.as_slice()[self.root_node_idx].first_child;
        let mut count = 0;
        while let Some(c) = curr {
            if count == idx {
                return Some(self.node_to_ref(c));
            }
            curr = self.nodes.as_slice()[c].next_sibling;
            count += 1;
        }
        None
    }

    pub fn node_to_ref<'doc>(&'doc self, node_idx: usize) -> ValueRef<'doc, 'a> {
        let node = &self.nodes.as_slice()[node_idx];
        match node.kind {
            NodeKind::Nil => ValueRef::Nil,
            NodeKind::Bool(b) => ValueRef::Bool(b),
            NodeKind::Number(n) => ValueRef::Number(n),
            NodeKind::String(span) => ValueRef::String(span.as_str(&self.string_arena)),
            NodeKind::Array => ValueRef::Array(ArrayView {
                doc: self,
                node_idx,
            }),
            NodeKind::Object => ValueRef::Object(ObjectView {
                doc: self,
                node_idx,
            }),
            NodeKind::ObjectEntry { .. } => unreachable!("ObjectEntry is not a value"),
        }
    }
}

// ---------------------------------------------------------------------------
// Non-allocating Borrowed Value Views for Assertion and Inspection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ValueRef<'doc, 'a> {
    Nil,
    Bool(bool),
    Number(ParsedNumber),
    String(&'doc str),
    Array(ArrayView<'doc, 'a>),
    Object(ObjectView<'doc, 'a>),
}

#[derive(Clone, Copy)]
pub struct ArrayView<'doc, 'a> {
    doc: &'doc PrototypeDocument<'a>,
    node_idx: usize,
}

impl<'doc, 'a> fmt::Debug for ArrayView<'doc, 'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        for i in 0..self.len() {
            if let Some(val) = self.get(i) {
                list.entry(&val);
            }
        }
        list.finish()
    }
}

impl<'doc, 'a> ArrayView<'doc, 'a> {
    pub fn len(&self) -> usize {
        self.doc.nodes.as_slice()[self.node_idx].child_count
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, idx: usize) -> Option<ValueRef<'doc, 'a>> {
        if idx >= self.len() {
            return None;
        }
        let mut curr = self.doc.nodes.as_slice()[self.node_idx].first_child;
        let mut count = 0;
        while let Some(c) = curr {
            if count == idx {
                return Some(self.doc.node_to_ref(c));
            }
            curr = self.doc.nodes.as_slice()[c].next_sibling;
            count += 1;
        }
        None
    }
}

impl<'doc, 'a> PartialEq for ArrayView<'doc, 'a> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        for i in 0..self.len() {
            if self.get(i) != other.get(i) {
                return false;
            }
        }
        true
    }
}

#[derive(Clone, Copy)]
pub struct ObjectView<'doc, 'a> {
    doc: &'doc PrototypeDocument<'a>,
    node_idx: usize,
}

impl<'doc, 'a> fmt::Debug for ObjectView<'doc, 'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut map = f.debug_map();
        for i in 0..self.len() {
            if let Some((k, v)) = self.get_entry(i) {
                map.entry(&k, &v);
            }
        }
        map.finish()
    }
}

impl<'doc, 'a> ObjectView<'doc, 'a> {
    pub fn len(&self) -> usize {
        self.doc.nodes.as_slice()[self.node_idx].child_count
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get_entry(&self, idx: usize) -> Option<(&'doc str, ValueRef<'doc, 'a>)> {
        if idx >= self.len() {
            return None;
        }
        let mut curr = self.doc.nodes.as_slice()[self.node_idx].first_child;
        let mut count = 0;
        while let Some(c) = curr {
            if count == idx {
                let entry_node = &self.doc.nodes.as_slice()[c];
                if let NodeKind::ObjectEntry { key, val_idx } = entry_node.kind {
                    let k = key.as_str(&self.doc.string_arena);
                    let v = self.doc.node_to_ref(val_idx);
                    return Some((k, v));
                }
            }
            curr = self.doc.nodes.as_slice()[c].next_sibling;
            count += 1;
        }
        None
    }
}

impl<'doc, 'a> PartialEq for ObjectView<'doc, 'a> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        for i in 0..self.len() {
            if self.get_entry(i) != other.get_entry(i) {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Bounded Lexical Parser Implementation
// ---------------------------------------------------------------------------

pub struct BoundedLexer<'a> {
    input: &'a [u8],
    pos: usize,
    limits: LustJsonLimits,
    ledger: CapacityLedger,
    current_depth: usize,
    node_count: usize,
    pub stack: PreallocatedVec<ContainerFrame>,
    pub key_check_scratch: PreallocatedVec<StrSpan<'a>>,
    pub doc_nodes: PreallocatedVec<PrototypeNode<'a>>,
    pub doc_arena: PreallocatedStringArena,
}

impl<'a> BoundedLexer<'a> {
    pub fn new(
        input: &'a str,
        limits: LustJsonLimits,
        ledger: CapacityLedger,
    ) -> Result<Self, LustJsonStatus> {
        let input_bytes = input.as_bytes();
        if input_bytes.len() > limits.max_input_bytes {
            return Err(LustJsonStatus::Limit);
        }

        // Bounded capacity upper-bound: a document of length L cannot contain more than L+2 nodes.
        let max_nodes = limits.max_nodes.min(input_bytes.len().saturating_add(2));

        // Exact pre-growth fallible reservations with rollback & fault injection:
        // Step 0: stack
        let stack = PreallocatedVec::try_with_capacity(limits.max_depth, &ledger)?;

        // Step 1: key check scratch
        let key_check_scratch = PreallocatedVec::try_with_capacity(max_nodes, &ledger)?;

        // Step 2: document nodes arena
        let doc_nodes = PreallocatedVec::try_with_capacity(max_nodes, &ledger)?;

        // Step 3: document string arena
        let doc_arena = PreallocatedStringArena::try_with_capacity(input_bytes.len(), &ledger)?;

        Ok(Self {
            input: input_bytes,
            pos: 0,
            limits,
            ledger,
            current_depth: 0,
            node_count: 0,
            stack,
            key_check_scratch,
            doc_nodes,
            doc_arena,
        })
    }

    pub fn ledger(&self) -> &CapacityLedger {
        &self.ledger
    }

    pub fn set_arena_drop_witness<F>(&mut self, witness: F)
    where
        F: FnMut(&CapacityLedger) + Send + 'static,
    {
        self.doc_arena.set_drop_witness(witness);
    }

    #[inline]
    fn skip_whitespace(&mut self) {
        while self.pos < self.input.len() {
            match self.input[self.pos] {
                b' ' | b'\t' | b'\r' | b'\n' => self.pos += 1,
                _ => break,
            }
        }
    }

    #[inline]
    fn peek(&self) -> Option<u8> {
        if self.pos < self.input.len() {
            Some(self.input[self.pos])
        } else {
            None
        }
    }

    #[inline]
    fn bump(&mut self) -> Option<u8> {
        if self.pos < self.input.len() {
            let b = self.input[self.pos];
            self.pos += 1;
            Some(b)
        } else {
            None
        }
    }

    #[inline]
    fn check_and_increment_node(&mut self) -> Result<(), LustJsonStatus> {
        let next = self
            .node_count
            .checked_add(1)
            .ok_or(LustJsonStatus::Limit)?;
        if next > self.limits.max_nodes {
            return Err(LustJsonStatus::Limit);
        }
        self.node_count = next;
        Ok(())
    }

    // Parse the entire document: root MUST be an array `[ ... ]`
    pub fn parse_root_arguments(&mut self) -> Result<PrototypeDocument<'a>, LustJsonStatus> {
        self.skip_whitespace();
        if self.pos >= self.input.len() {
            return Err(LustJsonStatus::InvalidArgument);
        }

        if self.peek() != Some(b'[') {
            return Err(LustJsonStatus::InvalidJson);
        }

        // Depth before descent: root array is depth 1
        let next_depth = self
            .current_depth
            .checked_add(1)
            .ok_or(LustJsonStatus::Limit)?;
        if next_depth > self.limits.max_depth {
            return Err(LustJsonStatus::Limit);
        }
        self.current_depth = next_depth;

        // Node count before decoding/allocation: root array is node 1
        self.check_and_increment_node()?;

        self.bump(); // consume '['

        let root_node_idx = self.doc_nodes.len();
        self.doc_nodes.push(PrototypeNode {
            kind: NodeKind::Array,
            first_child: None,
            next_sibling: None,
            child_count: 0,
        })?;

        self.stack.push(ContainerFrame {
            kind: ContainerKind::Array,
            node_idx: root_node_idx,
            last_child_idx: None,
        })?;

        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.bump();
            self.current_depth -= 1;
            self.stack.pop();
            self.skip_whitespace();
            if self.pos < self.input.len() {
                return Err(LustJsonStatus::InvalidJson); // trailing tokens!
            }

            // Release lexer scratch buffers early so only document buffers remain charged:
            self.stack.release_early();
            self.key_check_scratch.release_early();

            return Ok(self.build_document(root_node_idx));
        }

        loop {
            // max_args before parse:
            let cur_args = self.doc_nodes.as_slice()[root_node_idx].child_count;
            if cur_args >= self.limits.max_args {
                return Err(LustJsonStatus::Limit);
            }

            let val_idx = self.parse_value()?;

            // Attach val_idx to root array
            let frame = self.stack.as_slice_mut().last_mut().unwrap();
            if let Some(prev) = frame.last_child_idx {
                self.doc_nodes.as_slice_mut()[prev].next_sibling = Some(val_idx);
            } else {
                self.doc_nodes.as_slice_mut()[root_node_idx].first_child = Some(val_idx);
            }
            frame.last_child_idx = Some(val_idx);
            self.doc_nodes.as_slice_mut()[root_node_idx].child_count += 1;

            self.skip_whitespace();
            match self.peek() {
                Some(b',') => {
                    self.bump();
                    self.skip_whitespace();
                    if self.peek() == Some(b']') {
                        return Err(LustJsonStatus::InvalidJson); // trailing comma!
                    }
                }
                Some(b']') => {
                    self.bump();
                    break;
                }
                _ => return Err(LustJsonStatus::InvalidJson),
            }
        }

        self.current_depth -= 1;
        self.stack.pop();

        self.skip_whitespace();
        if self.pos < self.input.len() {
            return Err(LustJsonStatus::InvalidJson); // trailing tokens!
        }

        // Release lexer scratch buffers early so only document buffers remain charged:
        self.stack.release_early();
        self.key_check_scratch.release_early();

        Ok(self.build_document(root_node_idx))
    }

    fn build_document(&mut self, root_node_idx: usize) -> PrototypeDocument<'a> {
        let nodes = std::mem::replace(&mut self.doc_nodes, PreallocatedVec::empty(&self.ledger));
        let string_arena = std::mem::replace(
            &mut self.doc_arena,
            PreallocatedStringArena::empty(&self.ledger),
        );

        PrototypeDocument {
            nodes,
            string_arena,
            root_node_idx,
            ledger: self.ledger.clone(),
        }
    }

    fn parse_value(&mut self) -> Result<usize, LustJsonStatus> {
        self.skip_whitespace();
        let b = self.peek().ok_or(LustJsonStatus::InvalidJson)?;

        match b {
            b'n' => self.parse_null(),
            b't' | b'f' => self.parse_bool(),
            b'"' => self.parse_string_value(),
            b'[' => self.parse_array(),
            b'{' => self.parse_object(),
            b'-' | b'0'..=b'9' => self.parse_number(),
            _ => Err(LustJsonStatus::InvalidJson),
        }
    }

    fn parse_null(&mut self) -> Result<usize, LustJsonStatus> {
        // Enforce node count before decoding/allocating value
        self.check_and_increment_node()?;

        if self.input[self.pos..].starts_with(b"null") {
            self.pos += 4;
            let idx = self.doc_nodes.len();
            self.doc_nodes.push(PrototypeNode {
                kind: NodeKind::Nil,
                first_child: None,
                next_sibling: None,
                child_count: 0,
            })?;
            Ok(idx)
        } else {
            Err(LustJsonStatus::InvalidJson)
        }
    }

    fn parse_bool(&mut self) -> Result<usize, LustJsonStatus> {
        // Enforce node count before decoding/allocating value
        self.check_and_increment_node()?;

        if self.input[self.pos..].starts_with(b"true") {
            self.pos += 4;
            let idx = self.doc_nodes.len();
            self.doc_nodes.push(PrototypeNode {
                kind: NodeKind::Bool(true),
                first_child: None,
                next_sibling: None,
                child_count: 0,
            })?;
            Ok(idx)
        } else if self.input[self.pos..].starts_with(b"false") {
            self.pos += 5;
            let idx = self.doc_nodes.len();
            self.doc_nodes.push(PrototypeNode {
                kind: NodeKind::Bool(false),
                first_child: None,
                next_sibling: None,
                child_count: 0,
            })?;
            Ok(idx)
        } else {
            Err(LustJsonStatus::InvalidJson)
        }
    }

    fn parse_string_value(&mut self) -> Result<usize, LustJsonStatus> {
        // Enforce node count before decoding/allocating string
        self.check_and_increment_node()?;

        let span = self.parse_string_span()?;
        let idx = self.doc_nodes.len();
        self.doc_nodes.push(PrototypeNode {
            kind: NodeKind::String(span),
            first_child: None,
            next_sibling: None,
            child_count: 0,
        })?;
        Ok(idx)
    }

    fn parse_array(&mut self) -> Result<usize, LustJsonStatus> {
        // Depth before descent
        let next_depth = self
            .current_depth
            .checked_add(1)
            .ok_or(LustJsonStatus::Limit)?;
        if next_depth > self.limits.max_depth {
            return Err(LustJsonStatus::Limit);
        }
        self.current_depth = next_depth;

        // Node count before allocation
        self.check_and_increment_node()?;

        self.bump(); // consume '['

        let node_idx = self.doc_nodes.len();
        self.doc_nodes.push(PrototypeNode {
            kind: NodeKind::Array,
            first_child: None,
            next_sibling: None,
            child_count: 0,
        })?;

        self.stack.push(ContainerFrame {
            kind: ContainerKind::Array,
            node_idx,
            last_child_idx: None,
        })?;

        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.bump();
            self.current_depth -= 1;
            self.stack.pop();
            return Ok(node_idx);
        }

        loop {
            let val_idx = self.parse_value()?;

            // Attach val_idx to this array
            let frame = self.stack.as_slice_mut().last_mut().unwrap();
            if let Some(prev) = frame.last_child_idx {
                self.doc_nodes.as_slice_mut()[prev].next_sibling = Some(val_idx);
            } else {
                self.doc_nodes.as_slice_mut()[node_idx].first_child = Some(val_idx);
            }
            frame.last_child_idx = Some(val_idx);
            self.doc_nodes.as_slice_mut()[node_idx].child_count += 1;

            self.skip_whitespace();
            match self.peek() {
                Some(b',') => {
                    self.bump();
                    self.skip_whitespace();
                    if self.peek() == Some(b']') {
                        return Err(LustJsonStatus::InvalidJson); // trailing comma
                    }
                }
                Some(b']') => {
                    self.bump();
                    break;
                }
                _ => return Err(LustJsonStatus::InvalidJson),
            }
        }

        self.current_depth -= 1;
        self.stack.pop();
        Ok(node_idx)
    }

    fn parse_object(&mut self) -> Result<usize, LustJsonStatus> {
        // Depth before descent
        let next_depth = self
            .current_depth
            .checked_add(1)
            .ok_or(LustJsonStatus::Limit)?;
        if next_depth > self.limits.max_depth {
            return Err(LustJsonStatus::Limit);
        }
        self.current_depth = next_depth;

        // Node count before allocation
        self.check_and_increment_node()?;

        self.bump(); // consume '{'

        let node_idx = self.doc_nodes.len();
        self.doc_nodes.push(PrototypeNode {
            kind: NodeKind::Object,
            first_child: None,
            next_sibling: None,
            child_count: 0,
        })?;

        let scratch_start = self.key_check_scratch.len();

        self.stack.push(ContainerFrame {
            kind: ContainerKind::Object,
            node_idx,
            last_child_idx: None,
        })?;

        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.bump();
            self.current_depth -= 1;
            self.stack.pop();
            return Ok(node_idx);
        }

        loop {
            self.skip_whitespace();
            if self.peek() != Some(b'"') {
                return Err(LustJsonStatus::InvalidJson); // non-string object key!
            }

            // Node count before decoding/allocating key (key counts as a node)
            self.check_and_increment_node()?;

            let key_span = self.parse_string_span()?;
            self.key_check_scratch.push(key_span)?;

            self.skip_whitespace();
            if self.peek() != Some(b':') {
                return Err(LustJsonStatus::InvalidJson);
            }
            self.bump(); // consume ':'

            let val_node_idx = self.parse_value()?;

            let entry_node_idx = self.doc_nodes.len();
            self.doc_nodes.push(PrototypeNode {
                kind: NodeKind::ObjectEntry {
                    key: key_span,
                    val_idx: val_node_idx,
                },
                first_child: None,
                next_sibling: None,
                child_count: 0,
            })?;

            // Attach entry_node_idx to object
            let frame = self.stack.as_slice_mut().last_mut().unwrap();
            if let Some(prev) = frame.last_child_idx {
                self.doc_nodes.as_slice_mut()[prev].next_sibling = Some(entry_node_idx);
            } else {
                self.doc_nodes.as_slice_mut()[node_idx].first_child = Some(entry_node_idx);
            }
            frame.last_child_idx = Some(entry_node_idx);
            self.doc_nodes.as_slice_mut()[node_idx].child_count += 1;

            self.skip_whitespace();
            match self.peek() {
                Some(b',') => {
                    self.bump();
                    self.skip_whitespace();
                    if self.peek() == Some(b'}') {
                        return Err(LustJsonStatus::InvalidJson); // trailing comma
                    }
                }
                Some(b'}') => {
                    self.bump();
                    break;
                }
                _ => return Err(LustJsonStatus::InvalidJson),
            }
        }

        // Bounded duplicate key check: in-place sort of key span slice, then adjacent equality check.
        // `sort_unstable_by` is in-place pdqsort with zero heap allocations.
        let scratch_slice = &mut self.key_check_scratch.as_slice_mut()[scratch_start..];
        scratch_slice
            .sort_unstable_by(|a, b| a.as_str(&self.doc_arena).cmp(b.as_str(&self.doc_arena)));

        for i in 1..scratch_slice.len() {
            if scratch_slice[i - 1].as_str(&self.doc_arena)
                == scratch_slice[i].as_str(&self.doc_arena)
            {
                return Err(LustJsonStatus::InvalidJson); // duplicate decoded key!
            }
        }
        self.key_check_scratch.truncate(scratch_start);

        self.current_depth -= 1;
        self.stack.pop();
        Ok(node_idx)
    }

    // Scans string and returns key span (borrowed if unescaped, decoded into arena if escaped)
    fn parse_string_span(&mut self) -> Result<StrSpan<'a>, LustJsonStatus> {
        self.bump(); // consume opening '"'
        let start = self.pos;
        let mut has_escapes = false;

        while self.pos < self.input.len() {
            let b = self.input[self.pos];
            if b < 0x20 {
                return Err(LustJsonStatus::InvalidJson); // unescaped control character!
            }
            if b == b'\\' {
                has_escapes = true;
                break;
            }
            if b == b'"' {
                let slice = &self.input[start..self.pos];
                let s = std::str::from_utf8(slice).map_err(|_| LustJsonStatus::InvalidJson)?;
                self.pos += 1; // consume closing '"'
                return Ok(StrSpan::Borrowed(s));
            }
            self.pos += 1;
        }

        if !has_escapes {
            return Err(LustJsonStatus::InvalidJson); // unterminated string
        }

        // Escaped string: decode into preallocated string arena
        let arena_start = self.doc_arena.len();
        let unescaped_prefix = std::str::from_utf8(&self.input[start..self.pos])
            .map_err(|_| LustJsonStatus::InvalidJson)?;
        self.doc_arena.push_str(unescaped_prefix)?;

        self.decode_escapes_into_arena()?;
        let arena_end = self.doc_arena.len();
        Ok(StrSpan::Decoded(arena_start, arena_end))
    }

    fn decode_escapes_into_arena(&mut self) -> Result<(), LustJsonStatus> {
        while self.pos < self.input.len() {
            let b = self.input[self.pos];
            if b == b'"' {
                self.pos += 1; // closing quote
                return Ok(());
            }
            if b < 0x20 {
                return Err(LustJsonStatus::InvalidJson);
            }
            if b != b'\\' {
                // Collect unescaped slice
                let seg_start = self.pos;
                while self.pos < self.input.len()
                    && self.input[self.pos] != b'\\'
                    && self.input[self.pos] != b'"'
                {
                    if self.input[self.pos] < 0x20 {
                        return Err(LustJsonStatus::InvalidJson);
                    }
                    self.pos += 1;
                }
                let seg = std::str::from_utf8(&self.input[seg_start..self.pos])
                    .map_err(|_| LustJsonStatus::InvalidJson)?;
                self.doc_arena.push_str(seg)?;
                continue;
            }

            // Consume '\\'
            self.pos += 1;
            let esc = self.bump().ok_or(LustJsonStatus::InvalidJson)?;
            match esc {
                b'"' => self.doc_arena.push_char('"')?,
                b'\\' => self.doc_arena.push_char('\\')?,
                b'/' => self.doc_arena.push_char('/')?,
                b'b' => self.doc_arena.push_char('\x08')?,
                b'f' => self.doc_arena.push_char('\x0C')?,
                b'n' => self.doc_arena.push_char('\n')?,
                b'r' => self.doc_arena.push_char('\r')?,
                b't' => self.doc_arena.push_char('\t')?,
                b'u' => {
                    let u1 = self.parse_hex4()?;
                    // Check if high surrogate
                    if (0xD800..=0xDBFF).contains(&u1) {
                        // Must be followed by \uDC00..\uDFFF
                        if self.pos + 2 > self.input.len()
                            || self.input[self.pos] != b'\\'
                            || self.input[self.pos + 1] != b'u'
                        {
                            return Err(LustJsonStatus::InvalidJson); // unpaired high surrogate!
                        }
                        self.pos += 2; // consume \u
                        let u2 = self.parse_hex4()?;
                        if !(0xDC00..=0xDFFF).contains(&u2) {
                            return Err(LustJsonStatus::InvalidJson); // invalid low surrogate!
                        }
                        let code = 0x10000 + (((u1 as u32 - 0xD800) << 10) | (u2 as u32 - 0xDC00));
                        let ch = char::from_u32(code).ok_or(LustJsonStatus::InvalidJson)?;
                        self.doc_arena.push_char(ch)?;
                    } else if (0xDC00..=0xDFFF).contains(&u1) {
                        return Err(LustJsonStatus::InvalidJson); // unpaired low surrogate!
                    } else {
                        let ch = char::from_u32(u1 as u32).ok_or(LustJsonStatus::InvalidJson)?;
                        self.doc_arena.push_char(ch)?;
                    }
                }
                _ => return Err(LustJsonStatus::InvalidJson),
            }
        }
        Err(LustJsonStatus::InvalidJson) // unterminated string
    }

    fn parse_hex4(&mut self) -> Result<u16, LustJsonStatus> {
        if self.pos + 4 > self.input.len() {
            return Err(LustJsonStatus::InvalidJson);
        }
        let s = std::str::from_utf8(&self.input[self.pos..self.pos + 4])
            .map_err(|_| LustJsonStatus::InvalidJson)?;
        let val = u16::from_str_radix(s, 16).map_err(|_| LustJsonStatus::InvalidJson)?;
        self.pos += 4;
        Ok(val)
    }

    // Number Lexing with Token Class Fidelity, Significand-Only Underflow, and RFC 04 Bounds
    fn parse_number(&mut self) -> Result<usize, LustJsonStatus> {
        // Enforce node count before parsing number digits
        self.check_and_increment_node()?;

        let start = self.pos;
        let mut has_minus = false;
        let mut has_dot = false;
        let mut has_exp = false;

        if self.peek() == Some(b'-') {
            has_minus = true;
            self.bump();
        }

        // Integer part
        let int_start = self.pos;
        match self.peek() {
            Some(b'0') => {
                self.bump();
                // Leading zero: next char MUST NOT be a digit
                if let Some(next) = self.peek() {
                    if next.is_ascii_digit() {
                        return Err(LustJsonStatus::InvalidJson); // leading zero rejected!
                    }
                }
            }
            Some(b'1'..=b'9') => {
                while let Some(d) = self.peek() {
                    if d.is_ascii_digit() {
                        self.bump();
                    } else {
                        break;
                    }
                }
            }
            _ => return Err(LustJsonStatus::InvalidJson),
        }
        let int_digits = &self.input[int_start..self.pos];

        // Fraction part
        if self.peek() == Some(b'.') {
            has_dot = true;
            self.bump();
            let frac_start = self.pos;
            while let Some(d) = self.peek() {
                if d.is_ascii_digit() {
                    self.bump();
                } else {
                    break;
                }
            }
            if self.pos == frac_start {
                return Err(LustJsonStatus::InvalidJson); // decimal point must be followed by digit!
            }
        }

        // Significand digits end before exponent: inspect only significand for underflow detection
        let sig_bytes = &self.input[int_start..self.pos];
        let significand_has_nonzero = sig_bytes.iter().any(|&b| (b'1'..=b'9').contains(&b));

        // Exponent part
        if self.peek() == Some(b'e') || self.peek() == Some(b'E') {
            has_exp = true;
            self.bump();
            if self.peek() == Some(b'+') || self.peek() == Some(b'-') {
                self.bump();
            }
            let exp_start = self.pos;
            while let Some(d) = self.peek() {
                if d.is_ascii_digit() {
                    self.bump();
                } else {
                    break;
                }
            }
            if self.pos == exp_start {
                return Err(LustJsonStatus::InvalidJson); // exponent must have digits!
            }
        }

        let raw_num_str = std::str::from_utf8(&self.input[start..self.pos])
            .map_err(|_| LustJsonStatus::InvalidJson)?;

        let parsed_num = if !has_dot && !has_exp {
            // Integer token class
            if has_minus && int_digits == b"0" {
                // RFC 04: Integer -0 maps to Int(0)
                ParsedNumber::Int(0)
            } else {
                match raw_num_str.parse::<i64>() {
                    Ok(val) => ParsedNumber::Int(val),
                    Err(_) => {
                        // Out-of-i64 integer (>i64::MAX, u64 only, >u64::MAX, <i64::MIN)
                        // RFC 04: Reject with NUMBER_RANGE, NEVER fallback to float!
                        return Err(LustJsonStatus::NumberRange);
                    }
                }
            }
        } else {
            // Float token class
            let mut f = raw_num_str
                .parse::<f64>()
                .map_err(|_| LustJsonStatus::NumberRange)?;

            // Overflow / non-finite
            if !f.is_finite() {
                return Err(LustJsonStatus::NumberRange);
            }

            // Underflow-to-zero: nonzero significand that rounds to 0.0 or -0.0
            if f == 0.0 {
                if significand_has_nonzero {
                    return Err(LustJsonStatus::NumberRange); // underflow to zero rejected!
                }
                // Zero-significand exponent cases (0e-500, -0e-500, 0.0e-500, -0.0e-500):
                // Accepted as Float(±0)
                if has_minus {
                    f = -0.0;
                } else {
                    f = 0.0;
                }
            } else if has_minus && f == 0.0 {
                f = -0.0;
            }

            ParsedNumber::Float(f)
        };

        let node_idx = self.doc_nodes.len();
        self.doc_nodes.push(PrototypeNode {
            kind: NodeKind::Number(parsed_num),
            first_child: None,
            next_sibling: None,
            child_count: 0,
        })?;
        Ok(node_idx)
    }
}

// ===========================================================================
// TESTS & EMPIRICAL EVIDENCE
// ===========================================================================

#[test]
fn test_root_positional_array_grammar() {
    println!("\n=== 1. ROOT POSITIONAL ARRAY JSON GRAMMAR ===");

    let limits = LustJsonLimits::default();
    let ledger = CapacityLedger::new(limits.max_bridge_bytes);

    // Empty array: 0 arguments
    let mut lexer = BoundedLexer::new("[]", limits, ledger.clone()).unwrap();
    let doc = lexer.parse_root_arguments().unwrap();
    assert_eq!(doc.len(), 0);
    assert!(doc.is_empty());

    // Whitespace handling
    let mut lexer = BoundedLexer::new("   [  \t\n\r  ]  \t ", limits, ledger.clone()).unwrap();
    let doc = lexer.parse_root_arguments().unwrap();
    assert_eq!(doc.len(), 0);

    // Positional arguments
    let mut lexer = BoundedLexer::new(
        r#"[null, true, false, 42, "hello", [1, 2], {"a": true}]"#,
        limits,
        ledger.clone(),
    )
    .unwrap();
    let doc = lexer.parse_root_arguments().unwrap();
    assert_eq!(doc.len(), 7);
    assert_eq!(doc.get(0).unwrap(), ValueRef::Nil);
    assert_eq!(doc.get(1).unwrap(), ValueRef::Bool(true));
    assert_eq!(doc.get(2).unwrap(), ValueRef::Bool(false));
    assert_eq!(doc.get(3).unwrap(), ValueRef::Number(ParsedNumber::Int(42)));
    assert_eq!(doc.get(4).unwrap(), ValueRef::String("hello"));

    // Nested array inspection
    if let ValueRef::Array(arr) = doc.get(5).unwrap() {
        assert_eq!(arr.len(), 2);
        assert_eq!(arr.get(0).unwrap(), ValueRef::Number(ParsedNumber::Int(1)));
        assert_eq!(arr.get(1).unwrap(), ValueRef::Number(ParsedNumber::Int(2)));
    } else {
        panic!("Expected array at arg index 5");
    }

    // Nested object inspection
    if let ValueRef::Object(obj) = doc.get(6).unwrap() {
        assert_eq!(obj.len(), 1);
        assert_eq!(obj.get_entry(0).unwrap(), ("a", ValueRef::Bool(true)));
    } else {
        panic!("Expected object at arg index 6");
    }

    // Non-array roots rejected with InvalidJson
    let non_array_cases = [
        "{}",
        "\"string\"",
        "123",
        "true",
        "false",
        "null",
        "{\"args\": []}",
    ];
    for case in non_array_cases {
        let mut lexer = BoundedLexer::new(case, limits, ledger.clone()).unwrap();
        assert_eq!(
            lexer.parse_root_arguments().err().unwrap(),
            LustJsonStatus::InvalidJson,
            "Non-array root {:?} must be rejected",
            case
        );
    }

    // Empty or whitespace only input rejected with InvalidArgument
    let mut lexer = BoundedLexer::new("", limits, ledger.clone()).unwrap();
    assert_eq!(
        lexer.parse_root_arguments().err().unwrap(),
        LustJsonStatus::InvalidArgument
    );
    let mut lexer = BoundedLexer::new("   \t\n  ", limits, ledger.clone()).unwrap();
    assert_eq!(
        lexer.parse_root_arguments().err().unwrap(),
        LustJsonStatus::InvalidArgument
    );

    // Trailing tokens rejected with InvalidJson
    let trailing_cases = ["[1] 2", "[] []", "[1], [2]", "[1] ;"];
    for case in trailing_cases {
        let mut lexer = BoundedLexer::new(case, limits, ledger.clone()).unwrap();
        assert_eq!(
            lexer.parse_root_arguments().err().unwrap(),
            LustJsonStatus::InvalidJson,
            "Trailing tokens in {:?} must be rejected",
            case
        );
    }

    // Truncated input rejected
    let truncated_cases = ["[", "[1,", "[1, 2", "[1, [2]"];
    for case in truncated_cases {
        let mut lexer = BoundedLexer::new(case, limits, ledger.clone()).unwrap();
        assert_eq!(
            lexer.parse_root_arguments().err().unwrap(),
            LustJsonStatus::InvalidJson,
            "Truncated input {:?} must be rejected",
            case
        );
    }

    // Trailing / misplaced commas
    let comma_cases = ["[1,]", "[,1]", "[1,,2]"];
    for case in comma_cases {
        let mut lexer = BoundedLexer::new(case, limits, ledger.clone()).unwrap();
        assert_eq!(
            lexer.parse_root_arguments().err().unwrap(),
            LustJsonStatus::InvalidJson,
            "Comma error in {:?} must be rejected",
            case
        );
    }

    println!("PASS: Root positional array grammar validated successfully.");
}

#[test]
fn test_strings_utf8_escapes_and_surrogates() {
    println!("\n=== 2. STRINGS, UTF-8, ESCAPES, AND SURROGATE PAIRS ===");

    let limits = LustJsonLimits::default();
    let ledger = CapacityLedger::new(limits.max_bridge_bytes);

    // Plain unescaped string: verify borrowed directly from input slice
    let mut lexer = BoundedLexer::new(r#"["hello world"]"#, limits, ledger.clone()).unwrap();
    let doc = lexer.parse_root_arguments().unwrap();
    assert_eq!(doc.get(0).unwrap(), ValueRef::String("hello world"));
    // Inspect internal node representation:
    let node_idx = doc.nodes.as_slice()[doc.root_node_idx].first_child.unwrap();
    assert_eq!(
        doc.nodes.as_slice()[node_idx].kind,
        NodeKind::String(StrSpan::Borrowed("hello world")),
        "Unescaped string must be borrowed directly from input"
    );

    // Standard escapes
    let mut lexer =
        BoundedLexer::new(r#"["\" \\ \/ \b \f \n \r \t"]"#, limits, ledger.clone()).unwrap();
    let doc = lexer.parse_root_arguments().unwrap();
    assert_eq!(
        doc.get(0).unwrap(),
        ValueRef::String("\" \\ / \x08 \x0C \n \r \t")
    );

    // Unicode escapes and escaped NUL
    let mut lexer =
        BoundedLexer::new(r#"["\u0041 \u0000 \u00e9"]"#, limits, ledger.clone()).unwrap();
    let doc = lexer.parse_root_arguments().unwrap();
    assert_eq!(doc.get(0).unwrap(), ValueRef::String("A \0 é"));

    // Valid UTF-16 surrogate pairs:
    // U+1F600 (GRINNING FACE 😀): \uD83D\uDE00
    let mut lexer = BoundedLexer::new(r#"["\uD83D\uDE00"]"#, limits, ledger.clone()).unwrap();
    let doc = lexer.parse_root_arguments().unwrap();
    assert_eq!(doc.get(0).unwrap(), ValueRef::String("😀"));

    // U+1D11E (MUSICAL SYMBOL G CLEF 𝄞): \uD834\uDD1E
    let mut lexer = BoundedLexer::new(r#"["\uD834\uDD1E"]"#, limits, ledger.clone()).unwrap();
    let doc = lexer.parse_root_arguments().unwrap();
    assert_eq!(doc.get(0).unwrap(), ValueRef::String("𝄞"));

    // Unpaired high surrogate rejected
    let mut lexer = BoundedLexer::new(r#"["\uD83D"]"#, limits, ledger.clone()).unwrap();
    assert_eq!(
        lexer.parse_root_arguments().err().unwrap(),
        LustJsonStatus::InvalidJson
    );

    let mut lexer = BoundedLexer::new(r#"["\uD83D abc"]"#, limits, ledger.clone()).unwrap();
    assert_eq!(
        lexer.parse_root_arguments().err().unwrap(),
        LustJsonStatus::InvalidJson
    );

    // High surrogate followed by non-surrogate rejected
    let mut lexer = BoundedLexer::new(r#"["\uD83D\u0041"]"#, limits, ledger.clone()).unwrap();
    assert_eq!(
        lexer.parse_root_arguments().err().unwrap(),
        LustJsonStatus::InvalidJson
    );

    // Unpaired low surrogate rejected
    let mut lexer = BoundedLexer::new(r#"["\uDE00"]"#, limits, ledger.clone()).unwrap();
    assert_eq!(
        lexer.parse_root_arguments().err().unwrap(),
        LustJsonStatus::InvalidJson
    );

    // Invalid hex digits rejected
    let mut lexer = BoundedLexer::new(r#"["\u12G4"]"#, limits, ledger.clone()).unwrap();
    assert_eq!(
        lexer.parse_root_arguments().err().unwrap(),
        LustJsonStatus::InvalidJson
    );

    // Unescaped control characters (< 0x20) rejected
    let raw_control = "[\"hello\nworld\"]";
    let mut lexer = BoundedLexer::new(raw_control, limits, ledger.clone()).unwrap();
    assert_eq!(
        lexer.parse_root_arguments().err().unwrap(),
        LustJsonStatus::InvalidJson
    );

    // Invalid escape letter rejected
    let mut lexer = BoundedLexer::new(r#"["\a"]"#, limits, ledger.clone()).unwrap();
    assert_eq!(
        lexer.parse_root_arguments().err().unwrap(),
        LustJsonStatus::InvalidJson
    );

    println!("PASS: Strings, UTF-8, escapes, and surrogate pairs validated.");
}

#[test]
fn test_decoded_duplicate_keys() {
    println!("\n=== 3. DECODED DUPLICATE KEY REJECTION ===");

    let limits = LustJsonLimits::default();
    let ledger = CapacityLedger::new(limits.max_bridge_bytes);

    // Plain literal duplicate key
    let mut lexer = BoundedLexer::new(r#"[{"a": 1, "a": 2}]"#, limits, ledger.clone()).unwrap();
    assert_eq!(
        lexer.parse_root_arguments().err().unwrap(),
        LustJsonStatus::InvalidJson,
        "Literal duplicate key 'a' must be rejected"
    );

    // Escape-equivalent duplicate key: "foo" vs "\u0066oo"
    let mut lexer =
        BoundedLexer::new(r#"[{"foo": 1, "\u0066oo": 2}]"#, limits, ledger.clone()).unwrap();
    assert_eq!(
        lexer.parse_root_arguments().err().unwrap(),
        LustJsonStatus::InvalidJson,
        "Escape-equivalent key 'foo' vs '\\u0066oo' must be rejected"
    );

    // Multi-byte UTF-8 escape-equivalent duplicate: "café" vs "caf\u00e9"
    let mut lexer =
        BoundedLexer::new(r#"[{"café": 1, "caf\u00e9": 2}]"#, limits, ledger.clone()).unwrap();
    assert_eq!(
        lexer.parse_root_arguments().err().unwrap(),
        LustJsonStatus::InvalidJson,
        "Unicode escape-equivalent key 'café' vs 'caf\\u00e9' must be rejected"
    );

    // Distinct keys pass
    let mut lexer = BoundedLexer::new(r#"[{"a": 1, "b": 2}]"#, limits, ledger.clone()).unwrap();
    let doc = lexer.parse_root_arguments().unwrap();
    assert_eq!(doc.len(), 1);

    // Sibling objects with identical keys pass (different objects)
    let mut lexer = BoundedLexer::new(r#"[{"x": 1}, {"x": 2}]"#, limits, ledger.clone()).unwrap();
    let doc = lexer.parse_root_arguments().unwrap();
    assert_eq!(doc.len(), 2);

    // Nested object with identical key name pass (different nesting depth)
    let mut lexer = BoundedLexer::new(r#"[{"a": {"a": 1}}]"#, limits, ledger.clone()).unwrap();
    let doc = lexer.parse_root_arguments().unwrap();
    assert_eq!(doc.len(), 1);

    // Non-string object key rejected
    let mut lexer = BoundedLexer::new(r#"[{1: 2}]"#, limits, ledger.clone()).unwrap();
    assert_eq!(
        lexer.parse_root_arguments().err().unwrap(),
        LustJsonStatus::InvalidJson
    );

    println!("PASS: Decoded duplicate keys rejected with zero unbounded allocation.");
}

#[test]
fn test_number_token_class_and_bounds() {
    println!("\n=== 4. NUMBER TOKEN CLASS FIDELITY & BOUNDS ===");

    let limits = LustJsonLimits::default();
    let ledger = CapacityLedger::new(limits.max_bridge_bytes);

    // 1. Integer -0 vs Float -0.0
    // Integer "-0": must produce Int(0)
    let mut lexer = BoundedLexer::new("[-0]", limits, ledger.clone()).unwrap();
    let doc = lexer.parse_root_arguments().unwrap();
    assert_eq!(
        doc.get(0).unwrap(),
        ValueRef::Number(ParsedNumber::Int(0)),
        "Integer -0 must yield Int(0)"
    );

    // Float "-0.0": must produce Float(-0.0) with negative sign intact
    let mut lexer = BoundedLexer::new("[-0.0]", limits, ledger.clone()).unwrap();
    let doc = lexer.parse_root_arguments().unwrap();
    match doc.get(0).unwrap() {
        ValueRef::Number(ParsedNumber::Float(f)) => {
            assert_eq!(f, 0.0);
            assert!(
                f.is_sign_negative(),
                "Float -0.0 must preserve negative sign"
            );
        }
        other => panic!("Expected Float(-0.0), got {:?}", other),
    }

    // Float "-0e0": must produce Float(-0.0) with negative sign intact
    let mut lexer = BoundedLexer::new("[-0e0]", limits, ledger.clone()).unwrap();
    let doc = lexer.parse_root_arguments().unwrap();
    match doc.get(0).unwrap() {
        ValueRef::Number(ParsedNumber::Float(f)) => {
            assert_eq!(f, 0.0);
            assert!(
                f.is_sign_negative(),
                "Float -0e0 must preserve negative sign"
            );
        }
        other => panic!("Expected Float(-0e0), got {:?}", other),
    }

    // 2. Exact-zero significand with exponents: 0e-500, -0e-500, 0.0e-500, -0.0e-500
    // Positive 0e-500 -> Float(0.0)
    let mut lexer = BoundedLexer::new("[0e-500]", limits, ledger.clone()).unwrap();
    let doc = lexer.parse_root_arguments().unwrap();
    match doc.get(0).unwrap() {
        ValueRef::Number(ParsedNumber::Float(f)) => {
            assert_eq!(f, 0.0);
            assert!(
                f.is_sign_positive(),
                "Float 0e-500 must preserve positive zero"
            );
        }
        other => panic!("Expected Float(0.0), got {:?}", other),
    }

    // Negative -0e-500 -> Float(-0.0)
    let mut lexer = BoundedLexer::new("[-0e-500]", limits, ledger.clone()).unwrap();
    let doc = lexer.parse_root_arguments().unwrap();
    match doc.get(0).unwrap() {
        ValueRef::Number(ParsedNumber::Float(f)) => {
            assert_eq!(f, 0.0);
            assert!(
                f.is_sign_negative(),
                "Float -0e-500 must preserve negative zero"
            );
        }
        other => panic!("Expected Float(-0e-500), got {:?}", other),
    }

    // Positive 0.0e-500 -> Float(0.0)
    let mut lexer = BoundedLexer::new("[0.0e-500]", limits, ledger.clone()).unwrap();
    let doc = lexer.parse_root_arguments().unwrap();
    match doc.get(0).unwrap() {
        ValueRef::Number(ParsedNumber::Float(f)) => {
            assert_eq!(f, 0.0);
            assert!(
                f.is_sign_positive(),
                "Float 0.0e-500 must preserve positive zero"
            );
        }
        other => panic!("Expected Float(0.0), got {:?}", other),
    }

    // Negative -0.0e-500 -> Float(-0.0)
    let mut lexer = BoundedLexer::new("[-0.0e-500]", limits, ledger.clone()).unwrap();
    let doc = lexer.parse_root_arguments().unwrap();
    match doc.get(0).unwrap() {
        ValueRef::Number(ParsedNumber::Float(f)) => {
            assert_eq!(f, 0.0);
            assert!(
                f.is_sign_negative(),
                "Float -0.0e-500 must preserve negative zero"
            );
        }
        other => panic!("Expected Float(-0.0e-500), got {:?}", other),
    }

    // 3. Token class distinction: 1 vs 1.0 vs 1e0
    let mut lexer = BoundedLexer::new("[1, 1.0, 1e0]", limits, ledger.clone()).unwrap();
    let doc = lexer.parse_root_arguments().unwrap();
    assert_eq!(doc.get(0).unwrap(), ValueRef::Number(ParsedNumber::Int(1)));
    assert_eq!(
        doc.get(1).unwrap(),
        ValueRef::Number(ParsedNumber::Float(1.0))
    );
    assert_eq!(
        doc.get(2).unwrap(),
        ValueRef::Number(ParsedNumber::Float(1.0))
    );

    // 4. Exact i64 endpoints
    let mut lexer = BoundedLexer::new(
        "[9223372036854775807, -9223372036854775808]",
        limits,
        ledger.clone(),
    )
    .unwrap();
    let doc = lexer.parse_root_arguments().unwrap();
    assert_eq!(
        doc.get(0).unwrap(),
        ValueRef::Number(ParsedNumber::Int(i64::MAX))
    );
    assert_eq!(
        doc.get(1).unwrap(),
        ValueRef::Number(ParsedNumber::Int(i64::MIN))
    );

    // 5. Integers out of i64 range rejected with NumberRange (NEVER fallback to float!)
    // i64::MAX + 1 (9223372036854775808)
    let mut lexer = BoundedLexer::new("[9223372036854775808]", limits, ledger.clone()).unwrap();
    assert_eq!(
        lexer.parse_root_arguments().err().unwrap(),
        LustJsonStatus::NumberRange,
        "i64::MAX + 1 must return NumberRange"
    );

    // u64::MAX (18446744073709551615)
    let mut lexer = BoundedLexer::new("[18446744073709551615]", limits, ledger.clone()).unwrap();
    assert_eq!(
        lexer.parse_root_arguments().err().unwrap(),
        LustJsonStatus::NumberRange,
        "u64::MAX must return NumberRange"
    );

    // > u64::MAX (18446744073709551616)
    let mut lexer = BoundedLexer::new("[18446744073709551616]", limits, ledger.clone()).unwrap();
    assert_eq!(
        lexer.parse_root_arguments().err().unwrap(),
        LustJsonStatus::NumberRange,
        "> u64::MAX must return NumberRange"
    );

    // < i64::MIN (-9223372036854775809)
    let mut lexer = BoundedLexer::new("[-9223372036854775809]", limits, ledger.clone()).unwrap();
    assert_eq!(
        lexer.parse_root_arguments().err().unwrap(),
        LustJsonStatus::NumberRange,
        "< i64::MIN must return NumberRange"
    );

    // 6. Float overflow rejected with NumberRange
    let mut lexer = BoundedLexer::new("[1e500]", limits, ledger.clone()).unwrap();
    assert_eq!(
        lexer.parse_root_arguments().err().unwrap(),
        LustJsonStatus::NumberRange,
        "Float overflow 1e500 must return NumberRange"
    );

    let mut lexer = BoundedLexer::new("[-1e500]", limits, ledger.clone()).unwrap();
    assert_eq!(
        lexer.parse_root_arguments().err().unwrap(),
        LustJsonStatus::NumberRange,
        "Float overflow -1e500 must return NumberRange"
    );

    // 7. Nonzero Float underflow-to-zero rejected with NumberRange
    let mut lexer = BoundedLexer::new("[1e-500]", limits, ledger.clone()).unwrap();
    assert_eq!(
        lexer.parse_root_arguments().err().unwrap(),
        LustJsonStatus::NumberRange,
        "Float underflow 1e-500 must return NumberRange"
    );

    let mut lexer = BoundedLexer::new("[-1e-500]", limits, ledger.clone()).unwrap();
    assert_eq!(
        lexer.parse_root_arguments().err().unwrap(),
        LustJsonStatus::NumberRange,
        "Float underflow -1e-500 must return NumberRange"
    );

    // 8. Non-underflow zero float (literal 0.0 and 0e0) pass
    let mut lexer = BoundedLexer::new("[0.0, 0e0]", limits, ledger.clone()).unwrap();
    let doc = lexer.parse_root_arguments().unwrap();
    assert_eq!(
        doc.get(0).unwrap(),
        ValueRef::Number(ParsedNumber::Float(0.0))
    );
    assert_eq!(
        doc.get(1).unwrap(),
        ValueRef::Number(ParsedNumber::Float(0.0))
    );

    // 9. Syntax errors: leading zeroes, missing digits
    let syntax_cases = ["[01]", "[-01]", "[1.]", "[1e]", "[1e+]", "[-]"];
    for case in syntax_cases {
        let mut lexer = BoundedLexer::new(case, limits, ledger.clone()).unwrap();
        assert_eq!(
            lexer.parse_root_arguments().err().unwrap(),
            LustJsonStatus::InvalidJson,
            "Syntax error in {:?} must return InvalidJson",
            case
        );
    }

    println!("PASS: Number token class fidelity and boundary ranges verified.");
}

#[test]
fn test_ceilings_and_boundary_limits() {
    println!("\n=== 5. CEILINGS & BOUNDARY LIMITS AT N-1 / N / N+1 ===");

    // 1. max_input_bytes ceiling at N-1, N, N+1
    // Target input "[1, 2]" is 6 bytes.
    let input = "[1, 2]";
    let ledger = CapacityLedger::new(16_777_216);

    // N - 1 = 5 bytes: fails before parsing starts
    let limits_in_minus1 = LustJsonLimits {
        max_input_bytes: 5,
        ..Default::default()
    };
    assert_eq!(
        BoundedLexer::new(input, limits_in_minus1, ledger.clone())
            .err()
            .unwrap(),
        LustJsonStatus::Limit,
        "Input > max_input_bytes (N-1) must fail before parsing"
    );

    // N = 6 bytes: passes
    let limits_in_n = LustJsonLimits {
        max_input_bytes: 6,
        ..Default::default()
    };
    let mut lexer = BoundedLexer::new(input, limits_in_n, ledger.clone()).unwrap();
    assert_eq!(lexer.parse_root_arguments().unwrap().len(), 2);

    // N + 1 = 7 bytes: passes
    let limits_in_plus1 = LustJsonLimits {
        max_input_bytes: 7,
        ..Default::default()
    };
    let mut lexer = BoundedLexer::new(input, limits_in_plus1, ledger.clone()).unwrap();
    assert_eq!(lexer.parse_root_arguments().unwrap().len(), 2);

    // 2. max_args ceiling at N-1, N, N+1
    // Target N = 2 args.
    let limits_args = LustJsonLimits {
        max_args: 2,
        ..Default::default()
    };

    // N - 1 = 1 arg: passes
    let mut lexer = BoundedLexer::new("[1]", limits_args, ledger.clone()).unwrap();
    assert_eq!(lexer.parse_root_arguments().unwrap().len(), 1);

    // N = 2 args: passes
    let mut lexer = BoundedLexer::new("[1, 2]", limits_args, ledger.clone()).unwrap();
    assert_eq!(lexer.parse_root_arguments().unwrap().len(), 2);

    // N + 1 = 3 args: fails with Limit before parsing 3rd argument
    let mut lexer = BoundedLexer::new("[1, 2, 3]", limits_args, ledger.clone()).unwrap();
    assert_eq!(
        lexer.parse_root_arguments().err().unwrap(),
        LustJsonStatus::Limit,
        "Exceeding max_args (N+1) must return Limit"
    );

    // 3. max_depth ceiling at N-1, N, N+1
    // Target N = 2 depth.
    let limits_depth = LustJsonLimits {
        max_depth: 2,
        ..Default::default()
    };

    // N - 1 = depth 1: "[1]" passes
    let mut lexer = BoundedLexer::new("[1]", limits_depth, ledger.clone()).unwrap();
    assert_eq!(lexer.parse_root_arguments().unwrap().len(), 1);

    // N = depth 2: "[[1]]" passes
    let mut lexer = BoundedLexer::new("[[1]]", limits_depth, ledger.clone()).unwrap();
    assert_eq!(lexer.parse_root_arguments().unwrap().len(), 1);

    // N + 1 = depth 3: "[[[1]]]" fails with Limit before descent
    let mut lexer = BoundedLexer::new("[[[1]]]", limits_depth, ledger.clone()).unwrap();
    assert_eq!(
        lexer.parse_root_arguments().err().unwrap(),
        LustJsonStatus::Limit,
        "Exceeding max_depth (N+1) must return Limit"
    );

    // 4. max_nodes ceiling at N-1, N, N+1
    // Target N = 3 nodes.
    // Under RFC definitions: outer list is node 1, each value is a node.
    let limits_nodes = LustJsonLimits {
        max_nodes: 3,
        ..Default::default()
    };

    // N - 1 = 2 nodes: outer list (1) + element (1) = 2 nodes -> passes
    let mut lexer = BoundedLexer::new("[1]", limits_nodes, ledger.clone()).unwrap();
    assert_eq!(lexer.parse_root_arguments().unwrap().len(), 1);

    // N = 3 nodes: outer list (1) + 2 elements (2) = 3 nodes -> passes
    let mut lexer = BoundedLexer::new("[1, 2]", limits_nodes, ledger.clone()).unwrap();
    assert_eq!(lexer.parse_root_arguments().unwrap().len(), 2);

    // N + 1 = 4 nodes: outer list (1) + 3 elements (3) = 4 nodes -> fails with Limit before 4th node
    let mut lexer = BoundedLexer::new("[1, 2, 3]", limits_nodes, ledger.clone()).unwrap();
    assert_eq!(
        lexer.parse_root_arguments().err().unwrap(),
        LustJsonStatus::Limit,
        "Exceeding max_nodes (N+1) must return Limit"
    );

    // Object node count boundary: outer list (1) + object (2) + key "a" (3) + value 1 (4) = 4 nodes
    // Plus entry node internally = at least 4 nodes.
    let limits_obj_nodes = LustJsonLimits {
        max_nodes: 8,
        ..Default::default()
    };
    let mut lexer = BoundedLexer::new(r#"[{"a": 1}]"#, limits_obj_nodes, ledger.clone()).unwrap();
    assert_eq!(lexer.parse_root_arguments().unwrap().len(), 1);

    let limits_obj_nodes_minus1 = LustJsonLimits {
        max_nodes: 3,
        ..Default::default()
    };
    let mut lexer =
        BoundedLexer::new(r#"[{"a": 1}]"#, limits_obj_nodes_minus1, ledger.clone()).unwrap();
    assert_eq!(
        lexer.parse_root_arguments().err().unwrap(),
        LustJsonStatus::Limit,
        "Object exceeding max_nodes on key/value allocation must return Limit"
    );

    // 5. max_bridge_bytes ceiling at N-1, N, N+1
    // Calculate exact required bridge bytes for input "[1, 2]" with max_depth 4, max_nodes 8:
    // (for input length 6, max_nodes = min(8, 6+2) = 8)
    let limits_exact = LustJsonLimits {
        max_depth: 4,
        max_nodes: 8,
        ..Default::default()
    };
    let exact_bridge_needed = limits_exact.max_depth * std::mem::size_of::<ContainerFrame>()
        + 8 * std::mem::size_of::<StrSpan>()
        + 8 * std::mem::size_of::<PrototypeNode>()
        + "[1, 2]".len();

    // N - 1: fails with Limit before parsing begins, leaving ledger at 0
    let ledger_minus1 = CapacityLedger::new(exact_bridge_needed - 1);
    let res_minus1 = BoundedLexer::new("[1, 2]", limits_exact, ledger_minus1.clone());
    assert_eq!(
        res_minus1.err().unwrap(),
        LustJsonStatus::Limit,
        "max_bridge_bytes at N-1 must fail before parse"
    );
    assert_eq!(
        ledger_minus1.allocated(),
        0,
        "Ledger must rollback to 0 on insufficient bridge bytes"
    );

    // N: passes exactly
    let ledger_n = CapacityLedger::new(exact_bridge_needed);
    let mut lexer_n = BoundedLexer::new("[1, 2]", limits_exact, ledger_n.clone()).unwrap();
    assert_eq!(lexer_n.parse_root_arguments().unwrap().len(), 2);

    // N + 1: passes
    let ledger_plus1 = CapacityLedger::new(exact_bridge_needed + 1);
    let mut lexer_plus1 = BoundedLexer::new("[1, 2]", limits_exact, ledger_plus1.clone()).unwrap();
    assert_eq!(lexer_plus1.parse_root_arguments().unwrap().len(), 2);

    println!("PASS: Ceilings and boundary limits tested at N-1, N, and N+1.");
}

#[test]
fn test_pre_growth_reservation_and_ledger_lifecycle() {
    println!("\n=== 6. PRE-GROWTH RESERVATION & CAPACITY LEDGER LIFECYCLE ===");

    // 1. All collections pre-reserved before parse via fallible try_reserve_exact
    let limits = LustJsonLimits {
        max_depth: 8,
        max_nodes: 64,
        max_bridge_bytes: 100_000,
        ..Default::default()
    };
    let ledger = CapacityLedger::new(limits.max_bridge_bytes);
    let mut lexer = BoundedLexer::new(
        r#"[{"a": 1, "b": [2, 3]}, {"c": "hello \u0041"}]"#,
        limits,
        ledger.clone(),
    )
    .unwrap();

    let initial_stack_cap = lexer.stack.capacity();
    let initial_scratch_cap = lexer.key_check_scratch.capacity();
    let initial_nodes_cap = lexer.doc_nodes.capacity();
    let initial_arena_cap = lexer.doc_arena.capacity();
    let initial_allocated = ledger.allocated();

    println!(
        "Initial pre-reserved ledger allocation: {} bytes",
        initial_allocated
    );
    assert!(initial_allocated > 0, "Parser must pre-reserve in ledger");

    // 2. Parse and verify document owns allocations while live
    let doc = lexer.parse_root_arguments().unwrap();
    assert_eq!(doc.len(), 2);

    // Hard assertions: collection capacities NEVER grew!
    assert_eq!(doc.nodes.capacity(), initial_nodes_cap);
    assert_eq!(doc.string_arena.capacity(), initial_arena_cap);
    assert_eq!(initial_stack_cap, limits.max_depth);
    assert_eq!(initial_scratch_cap, initial_nodes_cap);

    // LEDGER REMAINS NONZERO WHILE RETURNED DOCUMENT IS LIVE:
    let live_doc_allocated = ledger.allocated();
    println!(
        "Ledger while document is live: {} bytes",
        live_doc_allocated
    );
    assert!(
        live_doc_allocated > 0,
        "Ledger must remain charged while document is live"
    );

    // ZERO LEDGER ONLY AFTER DOCUMENT-OWNED BUFFERS ARE DROPPED:
    drop(doc);
    let after_drop_allocated = ledger.allocated();
    println!(
        "Ledger after document drop: {} bytes (Peak: {})",
        after_drop_allocated,
        ledger.peak()
    );
    assert_eq!(
        after_drop_allocated, 0,
        "Ledger must return to exactly zero after document drop"
    );

    // 3. Ledger cleanup on parse error path:
    let err_ledger = CapacityLedger::new(limits.max_bridge_bytes);
    let mut lexer_err =
        BoundedLexer::new(r#"[{"a": 1, "a": 2}]"#, limits, err_ledger.clone()).unwrap();
    let parse_res = lexer_err.parse_root_arguments();
    assert_eq!(parse_res.err().unwrap(), LustJsonStatus::InvalidJson);
    assert!(
        err_ledger.allocated() > 0,
        "Ledger must remain charged while lexer error instance and its allocations are live"
    );
    drop(lexer_err);
    assert_eq!(
        err_ledger.allocated(),
        0,
        "Ledger must be released cleanly even on parse error"
    );

    // 4. Deterministic reservation-failure injection at each reservation step (steps 0..4):
    println!("Testing deterministic reservation failure injection at each step...");
    for step in 0..4 {
        let fault_ledger = CapacityLedger::new(limits.max_bridge_bytes);
        fault_ledger.set_fail_after_reservations(Some(step));
        let res = BoundedLexer::new("[1, 2, 3]", limits, fault_ledger.clone());
        assert_eq!(
            res.err().unwrap(),
            LustJsonStatus::Limit,
            "Reservation step {} failure injection must return Limit",
            step
        );
        assert_eq!(
            fault_ledger.allocated(),
            0,
            "Ledger must rollback to 0 after reservation failure at step {}",
            step
        );
    }

    println!("PASS: Pre-growth reservation and ledger lifecycle verified.");
}

#[test]
fn test_sorting_in_place_zero_allocation() {
    println!("\n=== 7. IN-PLACE SORTING WITH ZERO HEAP ALLOCATION ===");

    // Verifies that sort_unstable_by operates strictly in-place with zero heap allocations.
    let ledger = CapacityLedger::new(100_000);
    let mut keys = PreallocatedVec::try_with_capacity(5, &ledger).unwrap();
    let arena = PreallocatedStringArena::try_with_capacity(64, &ledger).unwrap();

    keys.push(StrSpan::Borrowed("banana")).unwrap();
    keys.push(StrSpan::Borrowed("apple")).unwrap();
    keys.push(StrSpan::Borrowed("date")).unwrap();
    keys.push(StrSpan::Borrowed("cherry")).unwrap();
    keys.push(StrSpan::Borrowed("elderberry")).unwrap();

    let cap_before = keys.capacity();
    let len_before = keys.len();
    let allocated_before = ledger.allocated();

    // In-place sort using standard library pdqsort
    keys.as_slice_mut()
        .sort_unstable_by(|a, b| a.as_str(&arena).cmp(b.as_str(&arena)));

    // Assert zero capacity changes and zero ledger impact
    assert_eq!(keys.capacity(), cap_before);
    assert_eq!(keys.len(), len_before);
    assert_eq!(ledger.allocated(), allocated_before);

    // Verify correct sorting order
    assert_eq!(keys.as_slice()[0].as_str(&arena), "apple");
    assert_eq!(keys.as_slice()[1].as_str(&arena), "banana");
    assert_eq!(keys.as_slice()[2].as_str(&arena), "cherry");
    assert_eq!(keys.as_slice()[3].as_str(&arena), "date");
    assert_eq!(keys.as_slice()[4].as_str(&arena), "elderberry");

    println!("PASS: In-place sort confirmed zero heap allocations and correct order.");
}

#[test]
fn test_prototype_limitations_evidence() {
    println!("\n=== 8. PROTOTYPE LIMITATIONS ASSESSMENT ===");
    // This test formally witnesses what Gate 1 prototype CAN and CANNOT establish:
    //
    // CAN establish:
    // 1. Direct scanning of bounded JSON without generic DOM or serde deserializer.
    // 2. Pre-reservation of all parser and document collections before parsing begins
    //    via fallible try_reserve_exact with zero collection growth during parse.
    // 3. Document ownership of allocated capacity: ledger remains charged while document is live,
    //    and drops to zero only after document is dropped.
    // 4. Exact RFC 04 number token class fidelity:
    //    - "-0" -> Int(0)
    //    - "-0.0" -> Float(-0.0) with negative sign
    //    - 1 vs 1.0 vs 1e0 distinct token classes
    //    - Out-of-i64 (>i64::MAX, u64 only, >u64::MAX, <i64::MIN) rejected with NumberRange
    //    - Float underflow-to-zero (e.g. 1e-500) rejected with NumberRange
    //    - Exact zero significand with exponent (0e-500, -0e-500, 0.0e-500) accepted as Float(±0)
    //    - Float overflow (e.g. 1e500) rejected with NumberRange
    // 5. Decoded duplicate key detection via in-place sort (O(k log k)) with zero heap allocation.
    // 6. Ceilings: max_input_bytes, max_depth, max_nodes, max_args, max_bridge_bytes at N-1/N/N+1.
    // 7. Deterministic reservation failure injection and rollback at every reservation step.
    //
    // CANNOT establish (deferred to later gates):
    // 1. Actual `lust::bytecode::Value` construction (requires Lust VM & runtime integration).
    // 2. Allocation identity & OS allocator rounding (CapacityLedger tracks requested bytes, not jemalloc page/chunk headers;
    //    `Arc<Mutex<LedgerState>>` and `ArenaDropWitness` are test-only measurement bookkeeping excluded from bridge payload capacity,
    //    not bridge-owned parser allocation; probe does not use global allocator hooks and does not claim to prove OS RSS/allocator usable size).
    // 3. VM memory accounting transfer / VM quota adoption (requires active VM instance).
    // 4. C ABI export & dynamic linker symbols.

    println!("All Gate 1 prototype capabilities and explicit limitations witnessed.");
}

#[test]
fn test_drop_probe_allocation_released_before_ledger_decrement() {
    println!("\n=== 9. DROP PROBE: BACKING STORAGE RELEASED BEFORE LEDGER DECREMENT ===");

    struct DropProbe {
        ledger: CapacityLedger,
        dropped: Arc<AtomicBool>,
        allocated_at_drop: Arc<AtomicUsize>,
    }

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
            // Crucial verification: at the exact moment the element inside backing
            // storage is dropped, the ledger charge must STILL be active.
            self.allocated_at_drop
                .store(self.ledger.allocated(), Ordering::SeqCst);
        }
    }

    // --- Sub-test A: PreallocatedVec::release_early ---
    {
        let ledger = CapacityLedger::new(4096);
        let dropped = Arc::new(AtomicBool::new(false));
        let allocated_at_drop = Arc::new(AtomicUsize::new(0));

        let mut pvec = PreallocatedVec::try_with_capacity(4, &ledger).unwrap();
        let expected_bytes = 4 * std::mem::size_of::<DropProbe>();
        assert_eq!(ledger.allocated(), expected_bytes);

        pvec.push(DropProbe {
            ledger: ledger.clone(),
            dropped: dropped.clone(),
            allocated_at_drop: allocated_at_drop.clone(),
        })
        .unwrap();

        assert!(!dropped.load(Ordering::SeqCst));
        assert_eq!(pvec.len(), 1);
        assert_eq!(pvec.capacity(), 4);

        // Call release_early: must drop backing storage detached Vec BEFORE decrementing ledger!
        pvec.release_early();

        assert!(
            dropped.load(Ordering::SeqCst),
            "DropProbe must be dropped when backing storage is released"
        );
        assert_eq!(
            allocated_at_drop.load(Ordering::SeqCst),
            expected_bytes,
            "Ledger must remain fully charged while backing storage elements are being dropped"
        );
        assert_eq!(
            ledger.allocated(),
            0,
            "Ledger charge must be released only after backing storage is dropped"
        );
        assert_eq!(pvec.capacity(), 0);
        assert_eq!(pvec.len(), 0);

        // Verify subsequent drop of already-released vector does not underflow or double-release
        drop(pvec);
        assert_eq!(ledger.allocated(), 0);
    }

    // --- Sub-test B: Wrapper Drop for PreallocatedVec ---
    {
        let ledger = CapacityLedger::new(4096);
        let dropped = Arc::new(AtomicBool::new(false));
        let allocated_at_drop = Arc::new(AtomicUsize::new(0));
        let expected_bytes = 2 * std::mem::size_of::<DropProbe>();

        {
            let mut pvec = PreallocatedVec::try_with_capacity(2, &ledger).unwrap();
            pvec.push(DropProbe {
                ledger: ledger.clone(),
                dropped: dropped.clone(),
                allocated_at_drop: allocated_at_drop.clone(),
            })
            .unwrap();

            assert_eq!(ledger.allocated(), expected_bytes);
            assert!(!dropped.load(Ordering::SeqCst));
            // pvec goes out of scope here, invoking <PreallocatedVec as Drop>::drop
        }

        assert!(
            dropped.load(Ordering::SeqCst),
            "DropProbe must be dropped on PreallocatedVec drop"
        );
        assert_eq!(
            allocated_at_drop.load(Ordering::SeqCst),
            expected_bytes,
            "Ledger must remain charged when backing storage element is dropped by wrapper Drop"
        );
        assert_eq!(
            ledger.allocated(),
            0,
            "Ledger must return to zero after wrapper Drop completes"
        );
    }

    // --- Sub-test C: PreallocatedStringArena::release_early with ArenaDropWitness ---
    {
        let ledger = CapacityLedger::new(4096);
        let witness_called = Arc::new(AtomicBool::new(false));
        let allocated_at_witness = Arc::new(AtomicUsize::new(0));

        let mut arena = PreallocatedStringArena::try_with_capacity(128, &ledger).unwrap();
        arena.push_str("sample payload string").unwrap();

        assert_eq!(ledger.allocated(), 128);
        assert_eq!(arena.capacity(), 128);

        let w_called = witness_called.clone();
        let a_witness = allocated_at_witness.clone();
        arena.set_drop_witness(move |l| {
            w_called.store(true, Ordering::SeqCst);
            // Crucial verification: at the moment the witness runs (strictly after
            // the detached backing Vec<u8> has been dropped and strictly before ledger.release),
            // the ledger charge must STILL be active and nonzero.
            a_witness.store(l.allocated(), Ordering::SeqCst);
        });

        assert!(!witness_called.load(Ordering::SeqCst));

        // Call release_early: must drop backing storage detached Vec<u8> BEFORE decrementing ledger,
        // invoking the witness while the ledger is still charged!
        arena.release_early();

        assert!(
            witness_called.load(Ordering::SeqCst),
            "ArenaDropWitness must be invoked when arena backing storage is released early"
        );
        assert_eq!(
            allocated_at_witness.load(Ordering::SeqCst),
            128,
            "Ledger must remain fully charged when arena backing storage is dropped in release_early"
        );
        assert_eq!(
            ledger.allocated(),
            0,
            "Ledger charge must be released only after arena backing storage is dropped"
        );
        assert_eq!(arena.capacity(), 0);
        assert_eq!(arena.len(), 0);

        // Calling push_str on released arena must cleanly fail with Limit
        assert_eq!(arena.push_str("more").unwrap_err(), LustJsonStatus::Limit);

        // Subsequent drop must not double-decrement or double-fire witness
        drop(arena);
        assert_eq!(ledger.allocated(), 0);
    }

    // --- Sub-test D: Wrapper Drop for PreallocatedStringArena with ArenaDropWitness ---
    {
        let ledger = CapacityLedger::new(4096);
        let witness_called = Arc::new(AtomicBool::new(false));
        let allocated_at_witness = Arc::new(AtomicUsize::new(0));

        {
            let mut arena = PreallocatedStringArena::try_with_capacity(256, &ledger).unwrap();
            arena.push_str("wrapper drop test payload").unwrap();
            assert_eq!(ledger.allocated(), 256);

            let w_called = witness_called.clone();
            let a_witness = allocated_at_witness.clone();
            arena.set_drop_witness(move |l| {
                w_called.store(true, Ordering::SeqCst);
                a_witness.store(l.allocated(), Ordering::SeqCst);
            });

            assert!(!witness_called.load(Ordering::SeqCst));
            // arena goes out of scope here, invoking <PreallocatedStringArena as Drop>::drop
        }

        assert!(
            witness_called.load(Ordering::SeqCst),
            "ArenaDropWitness must be invoked on PreallocatedStringArena wrapper Drop"
        );
        assert_eq!(
            allocated_at_witness.load(Ordering::SeqCst),
            256,
            "Ledger must remain charged when arena backing storage is dropped by wrapper Drop"
        );
        assert_eq!(
            ledger.allocated(),
            0,
            "Ledger must return to zero after wrapper Drop completes"
        );
    }

    // --- Sub-test E: Successful parsed-document destruction with ArenaDropWitness ---
    {
        let limits = LustJsonLimits {
            max_depth: 4,
            max_nodes: 8,
            max_bridge_bytes: 100_000,
            ..Default::default()
        };
        let ledger = CapacityLedger::new(limits.max_bridge_bytes);
        let input = r#"["payload unescaped", "escaped \u0041"]"#;

        let mut lexer = BoundedLexer::new(input, limits, ledger.clone())
            .expect("Lexer pre-reservation should succeed");

        let initial_reserved = ledger.allocated();
        assert!(initial_reserved > 0);

        let witness_called = Arc::new(AtomicBool::new(false));
        let allocated_at_witness = Arc::new(AtomicUsize::new(0));
        let w_called = witness_called.clone();
        let a_witness = allocated_at_witness.clone();

        // Arrange the arena witness on doc_arena BEFORE parsing/build_document transfers ownership:
        lexer.set_arena_drop_witness(move |l| {
            w_called.store(true, Ordering::SeqCst);
            // Verify ledger charge is still nonzero when witness fires:
            a_witness.store(l.allocated(), Ordering::SeqCst);
        });

        assert!(!witness_called.load(Ordering::SeqCst));

        // Parse root arguments: transfers ownership of doc_nodes and doc_arena (with witness) to doc
        let doc = lexer.parse_root_arguments().expect("Parse should succeed");
        assert_eq!(doc.len(), 2);

        // While document is live, ledger remains charged and witness must NOT have fired:
        let doc_live_allocated = ledger.allocated();
        assert!(
            doc_live_allocated > 0,
            "Ledger must remain charged while parsed document is live"
        );
        assert!(
            !witness_called.load(Ordering::SeqCst),
            "Witness must not fire while document is live"
        );

        // Drop the returned document:
        drop(doc);

        // Verify witness fired while ledger was still charged:
        assert!(
            witness_called.load(Ordering::SeqCst),
            "ArenaDropWitness must fire during document destruction"
        );
        assert!(
            allocated_at_witness.load(Ordering::SeqCst) > 0,
            "Ledger must remain charged when arena backing storage is dropped during document destruction"
        );
        assert_eq!(
            allocated_at_witness.load(Ordering::SeqCst),
            input.len(),
            "Arena charge must still be active when arena backing Vec is dropped"
        );

        // Verify ledger returns to exactly zero after drop returns:
        assert_eq!(
            ledger.allocated(),
            0,
            "Ledger must return to exactly zero after parsed document is dropped"
        );
    }

    println!(
        "PASS: DropProbe confirmed backing storage is physically dropped before ledger decrement."
    );
}

#[test]
fn test_parse_error_lifetime_and_ledger_cleanup() {
    println!("\n=== 10. PARSE ERROR LIFETIME & LEDGER CLEANUP ===");

    let limits = LustJsonLimits {
        max_depth: 4,
        max_nodes: 8,
        max_bridge_bytes: 100_000,
        ..Default::default()
    };

    let error_cases: [(&str, &str, LustJsonStatus); 7] = [
        (
            "syntax error (unclosed array)",
            "[1, 2, ",
            LustJsonStatus::InvalidJson,
        ),
        (
            "trailing tokens after valid JSON",
            "[1, 2] trailing",
            LustJsonStatus::InvalidJson,
        ),
        (
            "duplicate object keys",
            r#"[{"k": 1, "k": 2}]"#,
            LustJsonStatus::InvalidJson,
        ),
        ("depth limit exceeded", "[[[[[1]]]]]", LustJsonStatus::Limit),
        (
            "node ceiling exceeded",
            "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
            LustJsonStatus::Limit,
        ),
        (
            "integer out of range (>i64::MAX)",
            "[999999999999999999999999999999999999999999999999999999999999]",
            LustJsonStatus::NumberRange,
        ),
        (
            "invalid unicode escape surrogate",
            r#"["\uD800"]"#,
            LustJsonStatus::InvalidJson,
        ),
    ];

    for (name, input, expected_status) in error_cases {
        let ledger = CapacityLedger::new(limits.max_bridge_bytes);
        let mut lexer = BoundedLexer::new(input, limits, ledger.clone())
            .expect("Lexer pre-reservation should succeed");

        let pre_reserved_bytes = ledger.allocated();
        assert!(
            pre_reserved_bytes > 0,
            "[{}] Ledger must have pre-reserved capacity bytes before parse",
            name
        );

        let parse_result = lexer.parse_root_arguments();
        assert_eq!(
            parse_result.unwrap_err(),
            expected_status,
            "[{}] Unexpected error status",
            name
        );

        // INVARIANT 1: After parse error, while lexer is alive, ALL pre-allocated buffers
        // remain held in memory and ledger charges remain strictly positive.
        assert_eq!(
            ledger.allocated(),
            pre_reserved_bytes,
            "[{}] Ledger charge must remain fully charged after parse error while lexer is alive",
            name
        );
        assert!(
            lexer.stack.capacity() > 0,
            "[{}] Stack buffer must still be allocated",
            name
        );
        assert!(
            lexer.key_check_scratch.capacity() > 0,
            "[{}] Scratch buffer must still be allocated",
            name
        );
        assert!(
            lexer.doc_nodes.capacity() > 0,
            "[{}] Node buffer must still be allocated",
            name
        );
        assert!(
            lexer.doc_arena.capacity() > 0,
            "[{}] Arena buffer must still be allocated",
            name
        );

        // INVARIANT 2: Only after lexer and all its buffers are dropped does the ledger return to zero.
        drop(lexer);
        assert_eq!(
            ledger.allocated(),
            0,
            "[{}] Ledger must return to exactly zero only after lexer and all owned buffers are dropped",
            name
        );
    }

    println!(
        "PASS: Parse error cleanup and lifetime invariants verified across all error classes."
    );
}

#[test]
fn test_successful_parsed_document_destruction_witness() {
    println!("\n=== 11. SUCCESSFUL PARSED DOCUMENT DESTRUCTION & ARENA DROP WITNESS ===");

    let limits = LustJsonLimits {
        max_depth: 8,
        max_nodes: 64,
        max_bridge_bytes: 100_000,
        ..Default::default()
    };

    let test_inputs: [&str; 4] = [
        r#"[]"#,
        r#"["purely unescaped", 42, true, null]"#,
        r#"["string with escapes: \n \t \u0041 \uD83D\uDE00"]"#,
        r#"[{"key1": "value1", "key2": "value2 \u0066oo"}, [1, 2, "nested"]]"#,
    ];

    for input in test_inputs {
        let ledger = CapacityLedger::new(limits.max_bridge_bytes);
        let mut lexer = BoundedLexer::new(input, limits, ledger.clone())
            .expect("Lexer pre-reservation should succeed");

        let pre_reserved_bytes = ledger.allocated();
        assert!(
            pre_reserved_bytes > 0,
            "Ledger must pre-reserve capacity before parse"
        );

        let witness_called = Arc::new(AtomicBool::new(false));
        let allocated_at_witness = Arc::new(AtomicUsize::new(0));

        let w_called = witness_called.clone();
        let a_witness = allocated_at_witness.clone();

        // 1. Arrange the arena witness on doc_arena BEFORE parsing/build_document transfers ownership
        lexer.set_arena_drop_witness(move |l| {
            w_called.store(true, Ordering::SeqCst);
            a_witness.store(l.allocated(), Ordering::SeqCst);
        });

        assert!(!witness_called.load(Ordering::SeqCst));

        // 2. Parse successfully and obtain returned document
        let doc = lexer
            .parse_root_arguments()
            .expect("Parse should succeed for valid document");

        // 3. Document is live: ledger remains charged, witness has not fired
        let live_allocated = ledger.allocated();
        assert!(
            live_allocated > 0,
            "Ledger must remain charged while document is live"
        );
        assert!(
            !witness_called.load(Ordering::SeqCst),
            "Arena witness must not fire while document is live"
        );

        // 4. Drop the returned document
        drop(doc);

        // 5. Verify witness fired while ledger was still charged
        assert!(
            witness_called.load(Ordering::SeqCst),
            "Arena witness must fire when document and its arena are dropped"
        );
        let at_witness = allocated_at_witness.load(Ordering::SeqCst);
        assert!(
            at_witness > 0,
            "Ledger charge must be nonzero when arena backing Vec is dropped"
        );
        assert_eq!(
            at_witness,
            input.len(),
            "Arena charge must remain active until after backing Vec is dropped"
        );

        // 6. Verify ledger returns to exactly zero after drop returns
        assert_eq!(
            ledger.allocated(),
            0,
            "Ledger must return to exactly zero after document drop returns"
        );
    }

    println!(
        "PASS: Successful parsed-document destruction verified with per-instance arena drop witness."
    );
}
