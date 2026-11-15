#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RuntimeTelemetrySnapshot {
    pub gas_used: u64,
    pub gas_remaining: Option<u64>,
    pub memory_used_bytes: usize,
    pub active_host_slots: u16,
    pub jit_traces_compiled: u64,
    pub jit_guard_exits: u64,
}
