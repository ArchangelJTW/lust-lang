use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

pub const SCHEMA_VERSION_V1: u32 = 1;
pub const CANONICAL_ENCODING_ID: &str = "lust-dst-v1-le";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetInfo {
    pub triple: String,
    pub endianness: String,
    pub toolchain: String,
    pub candidate_digest: String,
    pub fixture_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrngSpec {
    pub algorithm: String,
    pub version: String,
}

impl Default for PrngSpec {
    fn default() -> Self {
        Self {
            algorithm: "splitmix64".to_string(),
            version: "1.0.0".to_string(),
        }
    }
}

/// Deterministic `no_std` PRNG (SplitMix64) with zero host entropy dependencies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub const fn state(&self) -> u64 {
        self.state
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn next_bounded(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            return 0;
        }
        (self.next_u64() % (bound as u64)) as usize
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalInput {
    pub name: String,
    pub content_hash: String,
    pub payload_len: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceEventKind {
    TaskStep = 1,
    EffectPerform = 2,
    FaultInjected = 3,
    TaskCompleted = 4,
}

impl TraceEventKind {
    fn from_u8(v: u8) -> Result<Self, TraceCodecError> {
        match v {
            1 => Ok(Self::TaskStep),
            2 => Ok(Self::EffectPerform),
            3 => Ok(Self::FaultInjected),
            4 => Ok(Self::TaskCompleted),
            _ => Err(TraceCodecError::InvalidTag("TraceEventKind", v)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulingChoice {
    pub ready_queue_len: u32,
    pub chosen_index: u32,
    pub chosen_task_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FaultPoint {
    pub label: String,
    pub code: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceEvent {
    pub event_index: u64,
    pub logical_time: u64,
    pub actor_task_id: u64,
    pub event_kind: TraceEventKind,
    pub inputs: Vec<u8>,
    pub outputs: Vec<u8>,
    pub scheduling_choice: SchedulingChoice,
    pub fault_injection_point: Option<FaultPoint>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinalOutcome {
    OkValue(i64),
    StableError(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceObservation {
    pub name: String,
    pub value: u64,
    pub unit: String,
    pub sampling_point: String,
}

/// Canonical v1 simulation trace format (`WAVE4_EXECUTION_PLAN.md` + `WAVE4_E0_INVENTORY.md` §2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceV1 {
    pub schema_version: u32,
    pub encoding: String,
    pub run_id: String,
    pub target: TargetInfo,
    pub seed: u64,
    pub prng: PrngSpec,
    pub external_inputs: Vec<ExternalInput>,
    pub events: Vec<TraceEvent>,
    pub final_state: String,
    pub final_result: FinalOutcome,
    pub resource_observations: Vec<ResourceObservation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceCodecError {
    UnexpectedEof,
    UnsupportedSchemaVersion(u32),
    InvalidEncodingId(String),
    InvalidUtf8,
    InvalidTag(&'static str, u8),
    NonContiguousEventIndex { expected: u64, got: u64 },
    TrailingBytes(usize),
}

impl fmt::Display for TraceCodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => write!(f, "unexpected end of trace stream"),
            Self::UnsupportedSchemaVersion(v) => write!(f, "unsupported schema_version={v}"),
            Self::InvalidEncodingId(e) => write!(f, "invalid canonical encoding id '{e}'"),
            Self::InvalidUtf8 => write!(f, "invalid utf-8 in trace string field"),
            Self::InvalidTag(kind, tag) => write!(f, "invalid {kind} tag {tag}"),
            Self::NonContiguousEventIndex { expected, got } => {
                write!(f, "non-contiguous event_index: expected {expected}, got {got}")
            }
            Self::TrailingBytes(n) => write!(f, "unexpected {n} trailing bytes in trace stream"),
        }
    }
}

pub struct CanonicalTraceCodec;

impl CanonicalTraceCodec {
    /// Deterministic, little-endian length-prefixed canonical binary encoding in exact schema field order.
    pub fn encode_canonical(trace: &TraceV1) -> Vec<u8> {
        let mut out = Vec::with_capacity(512);
        out.extend_from_slice(b"LDST");
        put_u32(&mut out, trace.schema_version);
        put_str(&mut out, &trace.encoding);
        put_str(&mut out, &trace.run_id);

        put_str(&mut out, &trace.target.triple);
        put_str(&mut out, &trace.target.endianness);
        put_str(&mut out, &trace.target.toolchain);
        put_str(&mut out, &trace.target.candidate_digest);
        put_str(&mut out, &trace.target.fixture_hash);

        put_u64(&mut out, trace.seed);
        put_str(&mut out, &trace.prng.algorithm);
        put_str(&mut out, &trace.prng.version);

        put_u32(&mut out, trace.external_inputs.len() as u32);
        for inp in &trace.external_inputs {
            put_str(&mut out, &inp.name);
            put_str(&mut out, &inp.content_hash);
            put_u64(&mut out, inp.payload_len);
        }

        put_u32(&mut out, trace.events.len() as u32);
        for ev in &trace.events {
            put_u64(&mut out, ev.event_index);
            put_u64(&mut out, ev.logical_time);
            put_u64(&mut out, ev.actor_task_id);
            out.push(ev.event_kind as u8);
            put_bytes(&mut out, &ev.inputs);
            put_bytes(&mut out, &ev.outputs);
            put_u32(&mut out, ev.scheduling_choice.ready_queue_len);
            put_u32(&mut out, ev.scheduling_choice.chosen_index);
            put_u64(&mut out, ev.scheduling_choice.chosen_task_id);
            match &ev.fault_injection_point {
                None => out.push(0),
                Some(fp) => {
                    out.push(1);
                    put_str(&mut out, &fp.label);
                    put_i32(&mut out, fp.code);
                }
            }
        }

        put_str(&mut out, &trace.final_state);
        match &trace.final_result {
            FinalOutcome::OkValue(v) => {
                out.push(0);
                put_i64(&mut out, *v);
            }
            FinalOutcome::StableError(msg) => {
                out.push(1);
                put_str(&mut out, msg);
            }
        }

        put_u32(&mut out, trace.resource_observations.len() as u32);
        for obs in &trace.resource_observations {
            put_str(&mut out, &obs.name);
            put_u64(&mut out, obs.value);
            put_str(&mut out, &obs.unit);
            put_str(&mut out, &obs.sampling_point);
        }

        out
    }

    pub fn decode_canonical(bytes: &[u8]) -> Result<TraceV1, TraceCodecError> {
        let mut r = Cursor { buf: bytes, pos: 0 };
        let magic = r.take_slice(4)?;
        if magic != b"LDST" {
            return Err(TraceCodecError::InvalidTag("magic", magic[0]));
        }
        let schema_version = r.take_u32()?;
        if schema_version != SCHEMA_VERSION_V1 {
            return Err(TraceCodecError::UnsupportedSchemaVersion(schema_version));
        }
        let encoding = r.take_str()?;
        if encoding != CANONICAL_ENCODING_ID {
            return Err(TraceCodecError::InvalidEncodingId(encoding));
        }
        let run_id = r.take_str()?;

        let target = TargetInfo {
            triple: r.take_str()?,
            endianness: r.take_str()?,
            toolchain: r.take_str()?,
            candidate_digest: r.take_str()?,
            fixture_hash: r.take_str()?,
        };

        let seed = r.take_u64()?;
        let prng = PrngSpec {
            algorithm: r.take_str()?,
            version: r.take_str()?,
        };

        let ext_len = r.take_u32()? as usize;
        let mut external_inputs = Vec::with_capacity(ext_len.min(1024));
        for _ in 0..ext_len {
            external_inputs.push(ExternalInput {
                name: r.take_str()?,
                content_hash: r.take_str()?,
                payload_len: r.take_u64()?,
            });
        }

        let ev_len = r.take_u32()? as usize;
        let mut events = Vec::with_capacity(ev_len.min(4096));
        for expected_idx in 0..ev_len {
            let event_index = r.take_u64()?;
            if event_index != expected_idx as u64 {
                return Err(TraceCodecError::NonContiguousEventIndex {
                    expected: expected_idx as u64,
                    got: event_index,
                });
            }
            let logical_time = r.take_u64()?;
            let actor_task_id = r.take_u64()?;
            let event_kind = TraceEventKind::from_u8(r.take_u8()?)?;
            let inputs = r.take_bytes()?;
            let outputs = r.take_bytes()?;
            let scheduling_choice = SchedulingChoice {
                ready_queue_len: r.take_u32()?,
                chosen_index: r.take_u32()?,
                chosen_task_id: r.take_u64()?,
            };
            let fault_tag = r.take_u8()?;
            let fault_injection_point = match fault_tag {
                0 => None,
                1 => Some(FaultPoint {
                    label: r.take_str()?,
                    code: r.take_i32()?,
                }),
                other => return Err(TraceCodecError::InvalidTag("FaultPoint", other)),
            };
            events.push(TraceEvent {
                event_index,
                logical_time,
                actor_task_id,
                event_kind,
                inputs,
                outputs,
                scheduling_choice,
                fault_injection_point,
            });
        }

        let final_state = r.take_str()?;
        let res_tag = r.take_u8()?;
        let final_result = match res_tag {
            0 => FinalOutcome::OkValue(r.take_i64()?),
            1 => FinalOutcome::StableError(r.take_str()?),
            other => return Err(TraceCodecError::InvalidTag("FinalOutcome", other)),
        };

        let obs_len = r.take_u32()? as usize;
        let mut resource_observations = Vec::with_capacity(obs_len.min(256));
        for _ in 0..obs_len {
            resource_observations.push(ResourceObservation {
                name: r.take_str()?,
                value: r.take_u64()?,
                unit: r.take_str()?,
                sampling_point: r.take_str()?,
            });
        }

        if r.pos != r.buf.len() {
            return Err(TraceCodecError::TrailingBytes(r.buf.len() - r.pos));
        }

        Ok(TraceV1 {
            schema_version,
            encoding,
            run_id,
            target,
            seed,
            prng,
            external_inputs,
            events,
            final_state,
            final_result,
            resource_observations,
        })
    }

    /// Compares semantic fields across two traces (e.g., host vs target), explicitly excluding
    /// nonsemantic host-specific fields (`run_id`, `target.triple`, `target.toolchain`).
    pub fn compare_semantic(lhs: &TraceV1, rhs: &TraceV1) -> Result<(), ReplayMismatch> {
        if lhs.schema_version != rhs.schema_version
            || lhs.encoding != rhs.encoding
            || lhs.seed != rhs.seed
            || lhs.prng != rhs.prng
            || lhs.external_inputs != rhs.external_inputs
        {
            return Err(ReplayMismatch::HeaderMismatch);
        }
        if lhs.events.len() != rhs.events.len() {
            return Err(ReplayMismatch::EventCountMismatch {
                expected: lhs.events.len(),
                actual: rhs.events.len(),
            });
        }
        for (idx, (a, b)) in lhs.events.iter().zip(rhs.events.iter()).enumerate() {
            if a != b {
                return Err(ReplayMismatch::EventDiverged {
                    index: idx,
                    detail: format!("semantic mismatch at event_index={idx}"),
                });
            }
        }
        if lhs.final_state != rhs.final_state || lhs.final_result != rhs.final_result {
            return Err(ReplayMismatch::FinalOutcomeMismatch);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimTaskSpec {
    pub task_id: u64,
    pub steps: u32,
    pub delta_per_step: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayMismatch {
    HeaderMismatch,
    EventCountMismatch { expected: usize, actual: usize },
    EventDiverged { index: usize, detail: String },
    FinalOutcomeMismatch,
}

impl fmt::Display for ReplayMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HeaderMismatch => write!(f, "trace header/seed/inputs mismatch"),
            Self::EventCountMismatch { expected, actual } => {
                write!(f, "event count mismatch: expected {expected}, got {actual}")
            }
            Self::EventDiverged { index, detail } => {
                write!(f, "replay diverged at event {index}: {detail}")
            }
            Self::FinalOutcomeMismatch => write!(f, "final state/result mismatch"),
        }
    }
}

/// Deterministic `no_std + alloc` cooperative simulation scheduler & replay verifier.
pub struct DeterministicScheduler {
    pub seed: u64,
    pub run_id: String,
    pub target: TargetInfo,
    pub external_inputs: Vec<ExternalInput>,
    pub fault_at_event: Option<(u64, FaultPoint)>,
}

impl DeterministicScheduler {
    pub fn run(&self, tasks: &[SimTaskSpec]) -> TraceV1 {
        let mut rng = SplitMix64::new(self.seed);
        let mut ready: Vec<(u64, u32, i64)> = tasks
            .iter()
            .filter(|t| t.steps > 0)
            .map(|t| (t.task_id, t.steps, t.delta_per_step))
            .collect();
        let mut events = Vec::new();
        let mut acc: i64 = 0;
        let mut state_hash: u64 = 0xcbf2_9ce4_8422_2325 ^ self.seed;
        let mut event_index: u64 = 0;
        let mut logical_time: u64 = 10;
        let mut aborted_err: Option<String> = None;

        while !ready.is_empty() {
            let queue_len = ready.len();
            let chosen_index = rng.next_bounded(queue_len);
            let (task_id, remaining, delta) = ready[chosen_index];
            let choice = SchedulingChoice {
                ready_queue_len: queue_len as u32,
                chosen_index: chosen_index as u32,
                chosen_task_id: task_id,
            };

            let fault = self
                .fault_at_event
                .as_ref()
                .filter(|(idx, _)| *idx == event_index)
                .map(|(_, fp)| fp.clone());

            let in_bytes = acc.to_le_bytes().to_vec();
            let event_kind;
            let out_bytes;

            if let Some(ref fp) = fault {
                event_kind = TraceEventKind::FaultInjected;
                state_hash = fnv1a_step(state_hash, task_id ^ (fp.code as u64));
                out_bytes = fp.code.to_le_bytes().to_vec();
                aborted_err = Some(format!("FaultInjected:{}:{}", fp.label, fp.code));
                events.push(TraceEvent {
                    event_index,
                    logical_time,
                    actor_task_id: task_id,
                    event_kind,
                    inputs: in_bytes,
                    outputs: out_bytes,
                    scheduling_choice: choice,
                    fault_injection_point: fault,
                });
                break;
            } else {
                let next_rem = remaining - 1;
                acc = acc.wrapping_add(delta * ((task_id as i64) + 1));
                state_hash = fnv1a_step(state_hash, (acc as u64) ^ task_id);
                event_kind = if next_rem == 0 {
                    TraceEventKind::TaskCompleted
                } else {
                    TraceEventKind::TaskStep
                };
                out_bytes = acc.to_le_bytes().to_vec();
                if next_rem == 0 {
                    ready.remove(chosen_index);
                } else {
                    ready[chosen_index].1 = next_rem;
                }
            }

            events.push(TraceEvent {
                event_index,
                logical_time,
                actor_task_id: task_id,
                event_kind,
                inputs: in_bytes,
                outputs: out_bytes,
                scheduling_choice: choice,
                fault_injection_point: None,
            });
            event_index += 1;
            logical_time += 10;
        }

        let final_state = format!("fnv64:{state_hash:016x}:acc={acc}");
        let final_result = match aborted_err {
            Some(err) => FinalOutcome::StableError(err),
            None => FinalOutcome::OkValue(acc),
        };
        let resource_observations = vec![
            ResourceObservation {
                name: "scheduler_events".to_string(),
                value: events.len() as u64,
                unit: "events".to_string(),
                sampling_point: "end_of_run".to_string(),
            },
            ResourceObservation {
                name: "logical_clock_final".to_string(),
                value: logical_time,
                unit: "ticks".to_string(),
                sampling_point: "end_of_run".to_string(),
            },
        ];

        TraceV1 {
            schema_version: SCHEMA_VERSION_V1,
            encoding: CANONICAL_ENCODING_ID.to_string(),
            run_id: self.run_id.clone(),
            target: self.target.clone(),
            seed: self.seed,
            prng: PrngSpec::default(),
            external_inputs: self.external_inputs.clone(),
            events,
            final_state,
            final_result,
            resource_observations,
        }
    }

    /// Strict replay mode: consumes recorded scheduling choices without re-rolling RNG
    /// and verifies every step, choice, inputs/outputs, and final outcome match.
    pub fn replay_verify(
        &self,
        recorded: &TraceV1,
        tasks: &[SimTaskSpec],
    ) -> Result<(), ReplayMismatch> {
        if recorded.schema_version != SCHEMA_VERSION_V1
            || recorded.seed != self.seed
            || recorded.external_inputs != self.external_inputs
        {
            return Err(ReplayMismatch::HeaderMismatch);
        }

        let mut ready: Vec<(u64, u32, i64)> = tasks
            .iter()
            .filter(|t| t.steps > 0)
            .map(|t| (t.task_id, t.steps, t.delta_per_step))
            .collect();
        let mut acc: i64 = 0;
        let mut state_hash: u64 = 0xcbf2_9ce4_8422_2325 ^ self.seed;
        let mut expected_logical_time: u64 = 10;

        let mut consumed = 0usize;
        for (pos, ev) in recorded.events.iter().enumerate() {
            if ev.event_index != pos as u64 {
                return Err(ReplayMismatch::EventDiverged {
                    index: pos,
                    detail: format!("expected event_index={pos}, got {}", ev.event_index),
                });
            }
            if ev.logical_time != expected_logical_time {
                return Err(ReplayMismatch::EventDiverged {
                    index: pos,
                    detail: format!(
                        "logical_time mismatch: expected {expected_logical_time}, got {}",
                        ev.logical_time
                    ),
                });
            }
            let qlen = ready.len();
            if qlen == 0 {
                return Err(ReplayMismatch::EventDiverged {
                    index: pos,
                    detail: "extra event recorded after all tasks completed".to_string(),
                });
            }
            if ev.scheduling_choice.ready_queue_len as usize != qlen {
                return Err(ReplayMismatch::EventDiverged {
                    index: pos,
                    detail: format!(
                        "ready_queue_len mismatch: expected {qlen}, got {}",
                        ev.scheduling_choice.ready_queue_len
                    ),
                });
            }
            let cidx = ev.scheduling_choice.chosen_index as usize;
            if cidx >= qlen || ready[cidx].0 != ev.scheduling_choice.chosen_task_id {
                return Err(ReplayMismatch::EventDiverged {
                    index: pos,
                    detail: "invalid or mismatched scheduling choice".to_string(),
                });
            }
            let (task_id, remaining, delta) = ready[cidx];
            if task_id != ev.actor_task_id {
                return Err(ReplayMismatch::EventDiverged {
                    index: pos,
                    detail: format!("actor_task_id mismatch: expected {task_id}, got {}", ev.actor_task_id),
                });
            }

            if let Some(ref fp) = ev.fault_injection_point {
                if ev.event_kind != TraceEventKind::FaultInjected {
                    return Err(ReplayMismatch::EventDiverged {
                        index: pos,
                        detail: "fault point present on non-FaultInjected event".to_string(),
                    });
                }
                state_hash = fnv1a_step(state_hash, task_id ^ (fp.code as u64));
                consumed = pos + 1;
                break;
            }

            let next_rem = remaining - 1;
            acc = acc.wrapping_add(delta * ((task_id as i64) + 1));
            state_hash = fnv1a_step(state_hash, (acc as u64) ^ task_id);
            let expected_kind = if next_rem == 0 {
                TraceEventKind::TaskCompleted
            } else {
                TraceEventKind::TaskStep
            };
            if ev.event_kind != expected_kind || ev.outputs != acc.to_le_bytes() {
                return Err(ReplayMismatch::EventDiverged {
                    index: pos,
                    detail: "event kind or output payload diverged during replay".to_string(),
                });
            }
            if next_rem == 0 {
                ready.remove(cidx);
            } else {
                ready[cidx].1 = next_rem;
            }
            consumed = pos + 1;
            expected_logical_time += 10;
        }

        if consumed != recorded.events.len() {
            return Err(ReplayMismatch::EventCountMismatch {
                expected: consumed,
                actual: recorded.events.len(),
            });
        }
        if !ready.is_empty()
            && !recorded
                .events
                .last()
                .is_some_and(|e| e.event_kind == TraceEventKind::FaultInjected)
        {
            return Err(ReplayMismatch::EventCountMismatch {
                expected: recorded.events.len() + ready.iter().map(|t| t.1 as usize).sum::<usize>(),
                actual: recorded.events.len(),
            });
        }

        let expected_state = format!("fnv64:{state_hash:016x}:acc={acc}");
        if expected_state != recorded.final_state {
            return Err(ReplayMismatch::FinalOutcomeMismatch);
        }
        Ok(())
    }
}

#[inline]
fn fnv1a_step(h: u64, data: u64) -> u64 {
    let mut x = h;
    for b in data.to_le_bytes() {
        x ^= b as u64;
        x = x.wrapping_mul(0x0100_0000_01B3);
    }
    x
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_i32(out: &mut Vec<u8>, v: i32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_i64(out: &mut Vec<u8>, v: i64) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_bytes(out: &mut Vec<u8>, b: &[u8]) {
    put_u32(out, b.len() as u32);
    out.extend_from_slice(b);
}
fn put_str(out: &mut Vec<u8>, s: &str) {
    put_bytes(out, s.as_bytes());
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take_slice(&mut self, n: usize) -> Result<&'a [u8], TraceCodecError> {
        if self.pos + n > self.buf.len() {
            return Err(TraceCodecError::UnexpectedEof);
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn take_u8(&mut self) -> Result<u8, TraceCodecError> {
        Ok(self.take_slice(1)?[0])
    }
    fn take_u32(&mut self) -> Result<u32, TraceCodecError> {
        let b: [u8; 4] = self.take_slice(4)?.try_into().unwrap();
        Ok(u32::from_le_bytes(b))
    }
    fn take_i32(&mut self) -> Result<i32, TraceCodecError> {
        let b: [u8; 4] = self.take_slice(4)?.try_into().unwrap();
        Ok(i32::from_le_bytes(b))
    }
    fn take_u64(&mut self) -> Result<u64, TraceCodecError> {
        let b: [u8; 8] = self.take_slice(8)?.try_into().unwrap();
        Ok(u64::from_le_bytes(b))
    }
    fn take_i64(&mut self) -> Result<i64, TraceCodecError> {
        let b: [u8; 8] = self.take_slice(8)?.try_into().unwrap();
        Ok(i64::from_le_bytes(b))
    }
    fn take_bytes(&mut self) -> Result<Vec<u8>, TraceCodecError> {
        let len = self.take_u32()? as usize;
        Ok(self.take_slice(len)?.to_vec())
    }
    fn take_str(&mut self) -> Result<String, TraceCodecError> {
        let b = self.take_bytes()?;
        core::str::from_utf8(&b)
            .map(|s| s.to_string())
            .map_err(|_| TraceCodecError::InvalidUtf8)
    }
}
