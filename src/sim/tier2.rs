//! Tier-2 Deterministic Simulation Testing (DST-T2) Engine.
//!
//! Provides simulation capture, interrupt injection, nested masking, and deterministic
//! replay for Tier-2 Cranelift JIT compiled functions and machine loops.
//!
//! Enforces:
//! 1. INV-FIBER-01: LIFO finalization and exactly-once execution.
//! 2. INV-FIBER-02: Preserved delivery during nested masks, bounded delivery upon unmasking (N=1).
//! 3. Bit-for-bit replay determinism between captured simulation bundles and replay runs.
//! 4. Fail-safe containment of malformed bundles and unwritable capture destinations.

use crate::effect_abi::InterruptReason;
use crate::mir::cranelift_backend::{
    safepoint_reset, MIR_STATUS_ERR_INTERRUPTED,
};
use crate::sim::invariants::{InvFiber01Oracle, InvFiber02Oracle, InvariantViolation};
use crate::sim::replay::{
    capture_failure_to_bundle, write_bundle_and_reproduction_task, CaptureConfig, CaptureError,
    EventLog, FailureKind, ReplayBundle, ReplayEngine, ReplayError, ReplayEventKind,
    ReplayResult,
};

use alloc::boxed::Box;
use alloc::format;
use alloc::string::ToString;
use alloc::vec::Vec;
use std::path::{Path, PathBuf};

/// Individual action scheduled to occur at a specific safepoint check index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tier2SimAction {
    EnterMask { at_safepoint: u64 },
    ExitMask { at_safepoint: u64 },
    InjectInterrupt { at_safepoint: u64, reason: InterruptReason },
    RegisterFinalizer { at_safepoint: u64, finalizer_id: u64 },
}

impl Tier2SimAction {
    pub fn at_safepoint(&self) -> u64 {
        match self {
            Self::EnterMask { at_safepoint } => *at_safepoint,
            Self::ExitMask { at_safepoint } => *at_safepoint,
            Self::InjectInterrupt { at_safepoint, .. } => *at_safepoint,
            Self::RegisterFinalizer { at_safepoint, .. } => *at_safepoint,
        }
    }
}

/// Deterministic schedule of simulated interrupts, masking scopes, and finalizer hooks.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Tier2InterruptSchedule {
    pub actions: Vec<Tier2SimAction>,
}

impl Tier2InterruptSchedule {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn inject_at(mut self, safepoint: u64, reason: InterruptReason) -> Self {
        self.actions.push(Tier2SimAction::InjectInterrupt {
            at_safepoint: safepoint,
            reason,
        });
        self
    }

    pub fn with_mask(mut self, enter_at: u64, exit_at: u64) -> Self {
        self.actions.push(Tier2SimAction::EnterMask {
            at_safepoint: enter_at,
        });
        self.actions.push(Tier2SimAction::ExitMask {
            at_safepoint: exit_at,
        });
        self
    }

    pub fn with_nested_masks(mut self, windows: &[(u64, u64)]) -> Self {
        for &(enter_at, exit_at) in windows {
            self = self.with_mask(enter_at, exit_at);
        }
        self
    }

    pub fn with_finalizer(mut self, at_safepoint: u64, finalizer_id: u64) -> Self {
        self.actions.push(Tier2SimAction::RegisterFinalizer {
            at_safepoint,
            finalizer_id,
        });
        self
    }

    pub fn with_action(mut self, action: Tier2SimAction) -> Self {
        self.actions.push(action);
        self
    }
}

/// Active controller coordinating Tier-2 safepoint simulation, logging, and oracle validation.
pub struct Tier2SafepointController {
    pub seed: u64,
    pub safepoint_counter: u64,
    pub mask_depth: usize,
    pub pending_interrupt: Option<InterruptReason>,
    pub deferred_count: u64,
    pub delivered_count: u64,
    pub cleanups_run: usize,
    pub registered_finalizers: Vec<u64>,
    pub executed_finalizers: Vec<u64>,
    pub event_log: EventLog,
    pub schedule: Tier2InterruptSchedule,
    pub oracle_fiber01: InvFiber01Oracle,
    pub oracle_fiber02: InvFiber02Oracle,
    finalizers: Vec<(u64, Box<dyn FnOnce() + Send + 'static>)>,
}

impl Tier2SafepointController {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            safepoint_counter: 0,
            mask_depth: 0,
            pending_interrupt: None,
            deferred_count: 0,
            delivered_count: 0,
            cleanups_run: 0,
            registered_finalizers: Vec::new(),
            executed_finalizers: Vec::new(),
            event_log: EventLog::new(),
            schedule: Tier2InterruptSchedule::new(),
            oracle_fiber01: InvFiber01Oracle::new(),
            oracle_fiber02: InvFiber02Oracle::new(),
            finalizers: Vec::new(),
        }
    }

    pub fn with_schedule(mut self, schedule: Tier2InterruptSchedule) -> Self {
        self.schedule = schedule;
        self
    }

    pub fn enter_mask(&mut self) {
        self.mask_depth += 1;
        self.oracle_fiber02.on_enter_mask();
        self.event_log.record(
            self.safepoint_counter,
            self.safepoint_counter * 100,
            ReplayEventKind::MaskEntered {
                new_depth: self.mask_depth,
            },
        );
    }

    pub fn exit_mask(&mut self) -> Option<InterruptReason> {
        self.mask_depth = self.mask_depth.saturating_sub(1);
        self.oracle_fiber02.on_exit_mask();
        let unmasked = if self.mask_depth == 0 {
            self.pending_interrupt.map(|r| format!("{r:?}"))
        } else {
            None
        };
        self.event_log.record(
            self.safepoint_counter,
            self.safepoint_counter * 100,
            ReplayEventKind::MaskExited {
                new_depth: self.mask_depth,
                unmasked_interrupt: unmasked,
            },
        );
        if self.mask_depth == 0 {
            self.pending_interrupt
        } else {
            None
        }
    }

    pub fn inject_interrupt(&mut self, reason: InterruptReason) {
        if let Some(existing) = self.pending_interrupt {
            if (reason as u32) > (existing as u32) {
                self.pending_interrupt = Some(reason);
            }
        } else {
            self.pending_interrupt = Some(reason);
        }
        self.oracle_fiber02.on_request_interrupt(reason);
        let deferred = self.mask_depth > 0;
        self.event_log.record(
            self.safepoint_counter,
            self.safepoint_counter * 100,
            ReplayEventKind::InterruptInjected {
                reason: format!("{reason:?}"),
                deferred,
            },
        );
    }

    pub fn register_finalizer<F: FnOnce() + Send + 'static>(&mut self, id: u64, f: F) {
        self.registered_finalizers.push(id);
        self.oracle_fiber01.record_register(id);
        self.finalizers.push((id, Box::new(f)));
        self.event_log.record(
            self.safepoint_counter,
            self.safepoint_counter * 100,
            ReplayEventKind::FinalizerRegistered { finalizer_id: id },
        );
    }

    pub fn record_compiled_entry(&mut self, func_name: &str, target_block: usize) {
        self.event_log.record(
            self.safepoint_counter,
            self.safepoint_counter * 100,
            ReplayEventKind::Tier2CompiledEntry {
                func_name: func_name.to_string(),
                target_block,
            },
        );
    }

    /// Advances the simulation by one safepoint check, executing any scheduled actions
    /// and delivering pending interrupts if unmasked.
    ///
    /// Returns `0` on normal continuation, or `MIR_STATUS_ERR_INTERRUPTED` (-2) if interrupted.
    pub fn step_safepoint(&mut self) -> Result<i64, InvariantViolation> {
        let current_step = self.safepoint_counter + 1;

        // Apply actions scheduled for this safepoint
        let actions = self.schedule.actions.clone();
        for action in &actions {
            if action.at_safepoint() == current_step {
                match action {
                    Tier2SimAction::EnterMask { .. } => self.enter_mask(),
                    Tier2SimAction::ExitMask { .. } => {
                        self.exit_mask();
                    }
                    Tier2SimAction::InjectInterrupt { reason, .. } => {
                        self.inject_interrupt(*reason);
                    }
                    Tier2SimAction::RegisterFinalizer { finalizer_id, .. } => {
                        let fid = *finalizer_id;
                        self.register_finalizer(fid, move || {});
                    }
                }
            }
        }

        self.safepoint_counter = current_step;

        self.event_log.record(
            self.safepoint_counter,
            self.safepoint_counter * 100,
            ReplayEventKind::SafepointChecked {
                safepoint_id: self.safepoint_counter,
                mask_depth: self.mask_depth,
            },
        );

        if self.mask_depth > 0 {
            if self.pending_interrupt.is_some() {
                self.deferred_count += 1;
            }
            Ok(0)
        } else if let Some(reason) = self.pending_interrupt.take() {
            self.delivered_count += 1;
            self.oracle_fiber02.on_deliver_interrupt()?;

            self.event_log.record(
                self.safepoint_counter,
                self.safepoint_counter * 100,
                ReplayEventKind::InterruptDelivered {
                    reason: format!("{reason:?}"),
                    safepoint_id: self.safepoint_counter,
                },
            );

            // Execute finalizers strictly in LIFO order
            let mut finalizers = core::mem::take(&mut self.finalizers);
            while let Some((id, f)) = finalizers.pop() {
                self.oracle_fiber01.record_execution(id)?;
                self.executed_finalizers.push(id);
                f();
                self.cleanups_run += 1;
                self.event_log.record(
                    self.safepoint_counter,
                    self.safepoint_counter * 100,
                    ReplayEventKind::FinalizerExecuted { finalizer_id: id },
                );
            }

            Ok(MIR_STATUS_ERR_INTERRUPTED)
        } else {
            Ok(0)
        }
    }

    /// Verifies both INV-FIBER-01 and INV-FIBER-02 invariants against the simulation run.
    pub fn verify_invariants(&self) -> Result<(), InvariantViolation> {
        self.oracle_fiber01.verify_completion()?;
        self.oracle_fiber02.verify_delivery()?;
        Ok(())
    }

    /// Captures the execution trace into a self-contained ReplayBundle.
    pub fn capture_bundle(
        &self,
        config: &CaptureConfig,
        failure: &FailureKind,
    ) -> Result<ReplayBundle, CaptureError> {
        capture_failure_to_bundle(config, failure, self.event_log.clone(), Vec::new())
    }

    /// Writes the bundle to disk as a TAR archive, guaranteeing temporary file cleanup on failure.
    pub fn export_bundle_tar(
        &self,
        config: &CaptureConfig,
        failure: &FailureKind,
        output_path: &Path,
    ) -> Result<PathBuf, CaptureError> {
        let bundle = self.capture_bundle(config, failure)?;
        write_bundle_and_reproduction_task(&bundle, output_path)
    }
}

/// Verification and replay engine for Tier-2 simulation bundles.
pub struct Tier2ReplayEngine;

impl Tier2ReplayEngine {
    /// Replays a simulation run from the provided bundle and verifies bit-for-bit trace determinism.
    pub fn replay_bundle(
        bundle: &ReplayBundle,
        schedule: &Tier2InterruptSchedule,
    ) -> Result<ReplayResult, ReplayError> {
        let mut controller =
            Tier2SafepointController::new(bundle.manifest.seed).with_schedule(schedule.clone());

        for _ in 0..bundle.manifest.step_count {
            match controller.step_safepoint() {
                Ok(status) => {
                    if status == MIR_STATUS_ERR_INTERRUPTED {
                        break;
                    }
                }
                Err(inv) => {
                    return Err(ReplayError::ExecutionFailed(format!(
                        "Invariant violation during replay: {inv}"
                    )));
                }
            }
        }

        let result = ReplayEngine::verify_trace(&bundle.events, &controller.event_log);
        Ok(result)
    }

    /// Imports a bundle from a TAR file and verifies trace determinism.
    pub fn replay_from_tar(
        tar_path: &Path,
        schedule: &Tier2InterruptSchedule,
    ) -> Result<ReplayResult, ReplayError> {
        let bundle = ReplayBundle::import_from_host_tar(tar_path)?;
        Self::replay_bundle(&bundle, schedule)
    }
}

/// High-level session harness binding Tier-2 simulation with the Cranelift JIT safepoint runtime.
pub struct Tier2SimSession {
    pub seed: u64,
    pub schedule: Tier2InterruptSchedule,
    pub controller: Tier2SafepointController,
}

impl Tier2SimSession {
    pub fn new(seed: u64, schedule: Tier2InterruptSchedule) -> Self {
        safepoint_reset();
        Self {
            seed,
            controller: Tier2SafepointController::new(seed).with_schedule(schedule.clone()),
            schedule,
        }
    }

    /// Advances the simulation controller by one safepoint.
    pub fn run_step(&mut self) -> Result<i64, InvariantViolation> {
        self.controller.step_safepoint()
    }

    /// Captures the completed session into a ReplayBundle.
    pub fn capture(
        &self,
        config: &CaptureConfig,
        failure: &FailureKind,
    ) -> Result<ReplayBundle, CaptureError> {
        self.controller.capture_bundle(config, failure)
    }
}
