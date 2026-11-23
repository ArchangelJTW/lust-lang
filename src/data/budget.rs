use core::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BudgetError {
    BytesExceeded { limit: usize, requested: usize },
    WorkUnitsExceeded { limit: usize, current: usize },
    DepthExceeded { limit: usize, current: usize },
    Interrupted,
}

impl From<crate::alloc_budget::BudgetAllocError> for BudgetError {
    fn from(err: crate::alloc_budget::BudgetAllocError) -> Self {
        BudgetError::BytesExceeded {
            limit: err.limit,
            requested: err.requested,
        }
    }
}

impl fmt::Display for BudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BudgetError::BytesExceeded { limit, requested } => {
                write!(
                    f,
                    "Memory budget exceeded: limit {} bytes, requested {}",
                    limit, requested
                )
            }
            BudgetError::WorkUnitsExceeded { limit, current } => {
                write!(
                    f,
                    "Work units budget exceeded: limit {}, consumed {}",
                    limit, current
                )
            }
            BudgetError::DepthExceeded { limit, current } => {
                write!(
                    f,
                    "Nesting depth limit exceeded: limit {}, current {}",
                    limit, current
                )
            }
            BudgetError::Interrupted => write!(f, "Execution interrupted"),
        }
    }
}

/// Resource budget tracker enforcing deterministic limits and interruption (Gate-SCHEMA-02)
#[derive(Debug, Clone)]
pub struct BudgetTracker {
    max_bytes: usize,
    allocated_bytes: usize,
    max_work_units: usize,
    consumed_work_units: usize,
    max_depth: usize,
    current_depth: usize,
    poll_interval: usize,
    units_since_poll: usize,
}

impl BudgetTracker {
    pub const fn new(max_bytes: usize, max_work_units: usize, max_depth: usize) -> Self {
        Self {
            max_bytes,
            allocated_bytes: 0,
            max_work_units,
            consumed_work_units: 0,
            max_depth,
            current_depth: 0,
            poll_interval: 4096,
            units_since_poll: 0,
        }
    }

    pub fn with_poll_interval(mut self, interval: usize) -> Self {
        self.poll_interval = interval.max(1);
        self
    }

    pub fn charge_bytes(&mut self, bytes: usize) -> Result<(), BudgetError> {
        let new_total = self
            .allocated_bytes
            .checked_add(bytes)
            .ok_or(BudgetError::BytesExceeded {
                limit: self.max_bytes,
                requested: usize::MAX,
            })?;

        if new_total > self.max_bytes {
            return Err(BudgetError::BytesExceeded {
                limit: self.max_bytes,
                requested: new_total,
            });
        }

        self.allocated_bytes = new_total;
        Ok(())
    }

    pub fn release_bytes(&mut self, bytes: usize) {
        self.allocated_bytes = self.allocated_bytes.saturating_sub(bytes);
    }

    /// Synchronously charges `bytes` in both `BudgetTracker` and `LustBudgetAlloc`.
    /// Rolls back `BudgetTracker` if `alloc.try_charge(bytes)` fails so both stay in sync.
    pub fn charge_bytes_with_alloc(
        &mut self,
        alloc: &crate::alloc_budget::LustBudgetAlloc,
        bytes: usize,
    ) -> Result<(), BudgetError> {
        self.charge_bytes(bytes)?;
        if let Err(err) = alloc.try_charge(bytes) {
            self.release_bytes(bytes);
            return Err(err.into());
        }
        Ok(())
    }

    /// Synchronously releases `bytes` in both `BudgetTracker` and `LustBudgetAlloc`.
    pub fn release_bytes_with_alloc(
        &mut self,
        alloc: &crate::alloc_budget::LustBudgetAlloc,
        bytes: usize,
    ) {
        self.release_bytes(bytes);
        alloc.refund(bytes);
    }

    pub fn charge_work(&mut self, units: usize) -> Result<bool, BudgetError> {
        let new_work = self
            .consumed_work_units
            .checked_add(units)
            .ok_or(BudgetError::WorkUnitsExceeded {
                limit: self.max_work_units,
                current: usize::MAX,
            })?;

        if new_work > self.max_work_units {
            return Err(BudgetError::WorkUnitsExceeded {
                limit: self.max_work_units,
                current: new_work,
            });
        }

        self.consumed_work_units = new_work;
        self.units_since_poll += units;

        if self.units_since_poll >= self.poll_interval {
            self.units_since_poll %= self.poll_interval;
            Ok(true) // Caller should poll for interruption
        } else {
            Ok(false)
        }
    }

    pub fn enter_depth(&mut self) -> Result<(), BudgetError> {
        if self.current_depth >= self.max_depth {
            return Err(BudgetError::DepthExceeded {
                limit: self.max_depth,
                current: self.current_depth + 1,
            });
        }
        self.current_depth += 1;
        Ok(())
    }

    pub fn exit_depth(&mut self) {
        self.current_depth = self.current_depth.saturating_sub(1);
    }

    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    pub fn consumed_work_units(&self) -> usize {
        self.consumed_work_units
    }

    pub fn current_depth(&self) -> usize {
        self.current_depth
    }
}
