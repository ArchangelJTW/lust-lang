//! `no_std + alloc` deterministic runtime core and canonical v1 simulation trace codec (`DST-T3`).
//!
//! Strictly depends only on `core` and `alloc` so embedded/bare-metal targets can adopt
//! the deterministic state machine, SplitMix64 PRNG, scheduler, and trace verifier cleanly
//! without host OS clocks, filesystem, sockets, or `std::sync`.

pub mod trace;

pub use trace::{
    CanonicalTraceCodec, DeterministicScheduler, ExternalInput, FaultPoint, FinalOutcome,
    PrngSpec, ReplayMismatch, ResourceObservation, SchedulingChoice, SimTaskSpec, SplitMix64,
    TargetInfo, TraceCodecError, TraceEvent, TraceEventKind, TraceV1, CANONICAL_ENCODING_ID,
    SCHEMA_VERSION_V1,
};
