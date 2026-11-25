//! Bounded DFA regex engine with linear work envelope guarantees.
//!
//! # Examples
//!
//! ```rust
//! use lust::regex::BoundedRegex;
//!
//! let re = BoundedRegex::new("hello [a-z]+").unwrap();
//! assert!(re.is_match("hello world").unwrap());
//! assert!(!re.is_match("hello 123").unwrap());
//! ```

pub mod compile;
pub mod execute;
pub mod plan;
pub mod policy;

pub use compile::compile_bounded;
pub use execute::{
    find, find_with_stats, find_with_tracker, is_match, is_match_with_tracker, replace_all,
    replace_all_with_tracker, RegexFindStats, RegexMatch,
};
pub use plan::RegexPlan;
pub use policy::{
    MAX_CAPTURE_GROUPS, MAX_COMPILED_PLAN_BYTES, MAX_DFA_STATES, MAX_NESTING_DEPTH,
    MAX_PATTERN_BYTES, POLL_INTERVAL_WORK_UNITS, RegexError, RegexWorkTracker,
};

/// Bounded regex wrapper implementing safe DFA execution.
#[derive(Debug, Clone)]
pub struct BoundedRegex {
    plan: RegexPlan,
    cancellation: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl BoundedRegex {
    pub fn new(pattern: &str) -> Result<Self, RegexError> {
        let plan = compile_bounded(pattern)?;
        Ok(Self { plan, cancellation: None })
    }

    pub fn with_cancellation(mut self, token: std::sync::Arc<std::sync::atomic::AtomicBool>) -> Self {
        self.cancellation = Some(token);
        self
    }

    pub fn plan(&self) -> &RegexPlan {
        &self.plan
    }

    pub fn is_match(&self, text: &str) -> Result<bool, RegexError> {
        let mut tracker = RegexWorkTracker::new(text.len(), self.plan.group_count, 0)?;
        if let Some(token) = &self.cancellation {
            tracker = tracker.with_cancellation(token.clone());
        }
        is_match_with_tracker(&self.plan, text, &mut tracker)
    }

    pub fn find(&self, text: &str) -> Result<Option<RegexMatch>, RegexError> {
        Ok(self.find_with_stats(text)?.0)
    }

    /// Same match as [`Self::find`] plus the number of DFA searches performed.
    pub fn find_with_stats(
        &self,
        text: &str,
    ) -> Result<(Option<RegexMatch>, RegexFindStats), RegexError> {
        let mut tracker = RegexWorkTracker::new(text.len(), self.plan.group_count, 0)?;
        if let Some(token) = &self.cancellation {
            tracker = tracker.with_cancellation(token.clone());
        }
        find_with_tracker(&self.plan, text, &mut tracker)
    }

    pub fn replace(&self, text: &str, replacement: &str) -> Result<String, RegexError> {
        self.replace_all(text, replacement)
    }

    pub fn replace_all(&self, text: &str, replacement: &str) -> Result<String, RegexError> {
        let max_matches = text.len().saturating_add(1);
        let max_output_estimate = text.len().saturating_add(max_matches.saturating_mul(replacement.len()));
        let mut tracker = RegexWorkTracker::new(text.len(), self.plan.group_count, max_output_estimate)?;
        if let Some(token) = &self.cancellation {
            tracker = tracker.with_cancellation(token.clone());
        }
        replace_all_with_tracker(&self.plan, text, replacement, &mut tracker)
    }
}
