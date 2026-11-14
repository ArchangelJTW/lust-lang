pub const MAX_PATTERN_BYTES: usize = 65_536;
pub const MAX_NESTING_DEPTH: usize = 128;
pub const MAX_CAPTURE_GROUPS: usize = 32;
pub const MAX_DFA_STATES: usize = 65_536;
pub const MAX_COMPILED_PLAN_BYTES: usize = 16_777_216;
pub const MAX_COMPILE_WORK_UNITS: u64 = 10_000_000;
pub const POLL_INTERVAL_WORK_UNITS: u64 = 4096;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegexError {
    UnsupportedPattern(String),
    UnsupportedBackendFeature(String),
    PatternTooLarge(usize),
    NestingTooDeep(usize),
    TooManyCaptureGroups(usize),
    BudgetExceeded { limit: u64, consumed: u64 },
    SyntaxError(String),
    Interrupted,
}

impl std::fmt::Display for RegexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegexError::UnsupportedPattern(p) => {
                write!(f, "Unsupported regex pattern (requires backtracking/NFA fallback): {}", p)
            }
            RegexError::UnsupportedBackendFeature(msg) => {
                write!(f, "Unsupported regex backend feature: {}", msg)
            }
            RegexError::PatternTooLarge(size) => {
                write!(f, "Pattern size {} bytes exceeds limit of {} bytes", size, MAX_PATTERN_BYTES)
            }
            RegexError::NestingTooDeep(depth) => {
                write!(f, "Pattern nesting depth {} exceeds maximum {}", depth, MAX_NESTING_DEPTH)
            }
            RegexError::TooManyCaptureGroups(g) => {
                write!(f, "Capture groups {} exceeds maximum {}", g, MAX_CAPTURE_GROUPS)
            }
            RegexError::BudgetExceeded { limit, consumed } => {
                write!(f, "Work budget exceeded: limit={}, consumed={}", limit, consumed)
            }
            RegexError::SyntaxError(msg) => write!(f, "Regex syntax error: {}", msg),
            RegexError::Interrupted => write!(f, "Regex execution interrupted"),
        }
    }
}

impl std::error::Error for RegexError {}

#[derive(Debug, Clone)]
pub struct RegexWorkTracker {
    pub transitions: u64,
    pub capture_updates: u64,
    pub bytes_scanned: u64,
    pub bytes_copied: u64,
    pub max_budget: u64,
    pub units_since_poll: u64,
    pub poll_count: u64,
    pub cancellation: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl RegexWorkTracker {
    pub fn new(n_bytes: usize, g_groups: usize, o_output_bytes: usize) -> Result<Self, RegexError> {
        let n = n_bytes as u64;
        let g = g_groups as u64;
        let o = o_output_bytes as u64;

        // Formula: W = (8 + 4*g) * (n + 1) + o
        let factor = 8u64
            .checked_add(4u64.checked_mul(g).ok_or(RegexError::BudgetExceeded { limit: 0, consumed: 0 })?)
            .ok_or(RegexError::BudgetExceeded { limit: 0, consumed: 0 })?;

        let n_plus_1 = n.checked_add(1).ok_or(RegexError::BudgetExceeded { limit: 0, consumed: 0 })?;
        let work = factor
            .checked_mul(n_plus_1)
            .ok_or(RegexError::BudgetExceeded { limit: 0, consumed: 0 })?
            .checked_add(o)
            .ok_or(RegexError::BudgetExceeded { limit: 0, consumed: 0 })?;

        Ok(Self {
            transitions: 0,
            capture_updates: 0,
            bytes_scanned: 0,
            bytes_copied: 0,
            max_budget: work,
            units_since_poll: 0,
            poll_count: 0,
            cancellation: None,
        })
    }

    pub fn with_cancellation(mut self, token: std::sync::Arc<std::sync::atomic::AtomicBool>) -> Self {
        self.cancellation = Some(token);
        self
    }

    pub fn total_work(&self) -> u64 {
        self.transitions
            .saturating_add(self.capture_updates)
            .saturating_add(self.bytes_scanned)
            .saturating_add(self.bytes_copied)
    }

    pub fn charge_transitions(&mut self, count: u64) -> Result<(), RegexError> {
        self.transitions = self.transitions.saturating_add(count);
        self.on_work_charged(count)
    }

    pub fn charge_capture_updates(&mut self, count: u64) -> Result<(), RegexError> {
        self.capture_updates = self.capture_updates.saturating_add(count);
        self.on_work_charged(count)
    }

    pub fn charge_scanned(&mut self, count: u64) -> Result<(), RegexError> {
        self.bytes_scanned = self.bytes_scanned.saturating_add(count);
        self.on_work_charged(count)
    }

    pub fn charge_copied(&mut self, count: u64) -> Result<(), RegexError> {
        self.bytes_copied = self.bytes_copied.saturating_add(count);
        self.on_work_charged(count)
    }

    pub fn poll(&mut self) -> Result<(), RegexError> {
        self.poll_count = self.poll_count.saturating_add(1);
        self.units_since_poll = 0;
        if let Some(token) = &self.cancellation {
            if token.load(std::sync::atomic::Ordering::Relaxed) {
                return Err(RegexError::Interrupted);
            }
        }
        Ok(())
    }

    fn on_work_charged(&mut self, count: u64) -> Result<(), RegexError> {
        let total = self.total_work();
        if total > self.max_budget {
            return Err(RegexError::BudgetExceeded {
                limit: self.max_budget,
                consumed: total,
            });
        }
        self.units_since_poll = self.units_since_poll.saturating_add(count);
        if self.units_since_poll >= POLL_INTERVAL_WORK_UNITS {
            self.poll_count = self.poll_count.saturating_add(self.units_since_poll / POLL_INTERVAL_WORK_UNITS);
            self.units_since_poll %= POLL_INTERVAL_WORK_UNITS;
            // Interruption hook point every <= 4096 work units
            if let Some(token) = &self.cancellation {
                if token.load(std::sync::atomic::Ordering::Relaxed) {
                    return Err(RegexError::Interrupted);
                }
            }
        }
        Ok(())
    }
}
