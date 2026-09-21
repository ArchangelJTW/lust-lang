use super::*;
use core::mem::size_of;

const MAP_ENTRY_BYTES_ESTIMATE: usize = 64;
const UPVALUE_BYTES_ESTIMATE: usize = 64;

#[derive(Debug, Clone, Default)]
pub(crate) struct BudgetState {
    gas: GasBudget,
    memory: MemoryBudget,
}

#[derive(Debug, Clone, Default)]
struct GasBudget {
    limit: Option<u64>,
    used: u64,
}

#[derive(Debug, Clone, Default)]
struct MemoryBudget {
    limit_bytes: Option<usize>,
    used_bytes: usize,
}

impl BudgetState {
    #[inline]
    pub(super) fn charge_gas(&mut self, amount: u64) -> Result<()> {
        self.gas.used = self.gas.used.saturating_add(amount);
        if let Some(limit) = self.gas.limit
            && self.gas.used > limit
        {
            return Err(LustError::RuntimeError {
                message: format!("Out of gas (limit: {}, used: {})", limit, self.gas.used),
            });
        }
        Ok(())
    }

    #[inline]
    pub(super) fn charge_mem_bytes(&mut self, bytes: usize) -> Result<()> {
        let Some(limit) = self.memory.limit_bytes else {
            return Ok(());
        };
        self.memory.used_bytes = self.memory.used_bytes.saturating_add(bytes);
        if self.memory.used_bytes > limit {
            return Err(LustError::RuntimeError {
                message: format!(
                    "Out of memory budget (limit: {} bytes, used since reset: {} bytes)",
                    limit, self.memory.used_bytes
                ),
            });
        }
        Ok(())
    }

    #[inline]
    pub(super) fn try_charge_mem_bytes(&mut self, bytes: usize) -> bool {
        let Some(limit) = self.memory.limit_bytes else {
            return true;
        };
        let Some(next) = self.memory.used_bytes.checked_add(bytes) else {
            return false;
        };
        if next > limit {
            return false;
        }
        self.memory.used_bytes = next;
        true
    }

    #[inline]
    pub(super) fn mem_budget_enabled(&self) -> bool {
        self.memory.limit_bytes.is_some()
    }

    #[inline]
    pub(super) fn charge_value_vec(&mut self, element_count: usize) -> Result<()> {
        if element_count == 0 {
            return Ok(());
        }
        self.charge_mem_bytes(element_count.saturating_mul(size_of::<Value>()))
    }

    #[inline]
    pub(super) fn try_charge_value_vec(&mut self, element_count: usize) -> bool {
        if element_count == 0 {
            return true;
        }
        self.try_charge_mem_bytes(element_count.saturating_mul(size_of::<Value>()))
    }

    #[inline]
    #[allow(dead_code)]
    pub(super) fn charge_vec_growth<T>(&mut self, old_cap: usize, new_cap: usize) -> Result<()> {
        if new_cap <= old_cap {
            return Ok(());
        }
        let delta = new_cap - old_cap;
        self.charge_mem_bytes(delta.saturating_mul(size_of::<T>()))
    }

    #[inline]
    #[allow(dead_code)]
    pub(super) fn try_charge_vec_growth<T>(&mut self, old_cap: usize, new_cap: usize) -> bool {
        if new_cap <= old_cap {
            return true;
        }
        let delta = new_cap - old_cap;
        self.try_charge_mem_bytes(delta.saturating_mul(size_of::<T>()))
    }

    #[inline]
    pub(super) fn charge_map_entry_estimate(&mut self) -> Result<()> {
        self.charge_mem_bytes(MAP_ENTRY_BYTES_ESTIMATE)
    }

    #[inline]
    pub(super) fn charge_upvalues_estimate(&mut self, upvalue_count: usize) -> Result<()> {
        if upvalue_count == 0 {
            return Ok(());
        }
        self.charge_mem_bytes(upvalue_count.saturating_mul(UPVALUE_BYTES_ESTIMATE))
    }
}

impl VM {
    pub fn set_gas_budget(&mut self, limit: u64) {
        // Code compiled without a budget has no gas checks in its loops.
        if self.budgets.gas.limit.is_none() {
            self.jit.invalidate_compiled_code();
        }
        self.budgets.gas.limit = Some(limit);
    }

    pub fn clear_gas_budget(&mut self) {
        self.budgets.gas.limit = None;
    }

    /// Whether compiled code must charge gas at loop back-edges.
    pub(crate) fn gas_checked(&self) -> bool {
        self.budgets.gas.limit.is_some()
    }

    /// Hand the gas budget to native code (`JitCells::gas_left`) before
    /// entering it.
    pub(crate) fn publish_gas_to_cells(&mut self) {
        self.jit_cells.gas_left = match self.budgets.gas.limit {
            Some(limit) => limit.saturating_sub(self.budgets.gas.used),
            None => u64::MAX,
        };
    }

    /// Take back what native code charged.
    pub(crate) fn sync_gas_from_cells(&mut self) {
        if let Some(limit) = self.budgets.gas.limit {
            self.budgets.gas.used = limit.saturating_sub(self.jit_cells.gas_left);
        }
    }

    pub fn reset_gas_counter(&mut self) {
        self.budgets.gas.used = 0;
    }

    pub fn gas_used(&self) -> u64 {
        self.budgets.gas.used
    }

    pub fn gas_remaining(&self) -> Option<u64> {
        self.budgets
            .gas
            .limit
            .map(|limit| limit.saturating_sub(self.budgets.gas.used))
    }

    pub fn set_memory_budget_bytes(&mut self, limit_bytes: usize) {
        self.budgets.memory.limit_bytes = Some(limit_bytes);
    }

    pub fn set_memory_budget_kb(&mut self, limit_kb: u64) {
        let bytes = limit_kb.saturating_mul(1024);
        let limit_bytes = usize::try_from(bytes).unwrap_or(usize::MAX);
        self.set_memory_budget_bytes(limit_bytes);
    }

    pub fn clear_memory_budget(&mut self) {
        self.budgets.memory.limit_bytes = None;
        self.budgets.memory.used_bytes = 0;
    }

    pub fn reset_memory_counter(&mut self) {
        self.budgets.memory.used_bytes = 0;
    }

    pub fn memory_used_bytes(&self) -> usize {
        self.budgets.memory.used_bytes
    }

    pub fn memory_remaining_bytes(&self) -> Option<usize> {
        self.budgets
            .memory
            .limit_bytes
            .map(|limit| limit.saturating_sub(self.budgets.memory.used_bytes))
    }

    pub(crate) fn try_charge_memory_bytes(&mut self, bytes: usize) -> bool {
        self.budgets.try_charge_mem_bytes(bytes)
    }

    pub(crate) fn try_charge_memory_value_vec(&mut self, element_count: usize) -> bool {
        self.budgets.try_charge_value_vec(element_count)
    }

    pub(crate) fn charge_vec_growth<T>(&mut self, old_cap: usize, new_cap: usize) -> Result<()> {
        self.budgets.charge_vec_growth::<T>(old_cap, new_cap)
    }

    #[allow(dead_code)]
    pub(crate) fn try_charge_memory_vec_growth<T>(
        &mut self,
        old_cap: usize,
        new_cap: usize,
    ) -> bool {
        self.budgets.try_charge_vec_growth::<T>(old_cap, new_cap)
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use crate::EmbeddedProgram;
    use crate::{LustError, Result};

    #[test]
    fn gas_budget_traps() -> Result<()> {
        let mut program = EmbeddedProgram::builder()
            .module(
                "main",
                r#"
                    function spin(): ()
                        while true do
                        end
                    end
                "#,
            )
            .entry_module("main")
            .compile()?;

        program.vm_mut().set_gas_budget(30);
        program.vm_mut().reset_gas_counter();
        let err = program.call_raw("main.spin", vec![]).unwrap_err();
        match err {
            LustError::RuntimeErrorWithTrace { message, .. }
            | LustError::RuntimeError { message } => {
                assert!(message.to_lowercase().contains("out of gas"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
        Ok(())
    }

    /// A loop the JIT compiles (a counted one, whose trace runs its
    /// iterations natively) still traps when the budget runs out, and a
    /// loop the budget covers completes with its gas accounted for.
    #[test]
    fn gas_budget_traps_inside_compiled_loops() -> Result<()> {
        let source = r#"
            function count(n: int): int
                local i: int = 0
                local total: int = 0
                while i < n do
                    total = total + i
                    i = i + 1
                end
                return total
            end
            function outer(n: int): int
                local total: int = 0
                local pass: int = 0
                while pass < 3 do
                    total = total + count(n)
                    pass = pass + 1
                end
                return total
            end
            function warm(): int
                local total: int = 0
                local pass: int = 0
                while pass < 500 do
                    total = total + count(4)
                    pass = pass + 1
                end
                return total
            end
        "#;
        // `warm` makes `count` hot enough to be compiled as a whole
        // function (its loop a `Jump` back to a label) before the
        // budgeted calls; the direct call exercises the loop trace.
        for (entry, warm) in [
            ("main.count", false),
            ("main.count", true),
            ("main.outer", false),
            ("main.outer", true),
        ] {
            let mut program = EmbeddedProgram::builder()
                .module("main", source)
                .entry_module("main")
                .compile()?;
            program.vm_mut().set_gas_budget(200_000);
            if warm {
                program.vm_mut().reset_gas_counter();
                assert_eq!(
                    program.call_raw("main.warm", vec![])?,
                    crate::Value::Int(3000)
                );
            }
            program.vm_mut().reset_gas_counter();
            let err = program
                .call_raw(entry, vec![crate::Value::Int(1_000_000_000)])
                .unwrap_err();
            match err {
                LustError::RuntimeErrorWithTrace { message, .. }
                | LustError::RuntimeError { message } => {
                    assert!(
                        message.to_lowercase().contains("out of gas"),
                        "{entry}: {message}"
                    );
                }
                other => panic!("unexpected error: {other:?}"),
            }
            // Budgets are approximate under the JIT, but not by orders of
            // magnitude: the loop was stopped near the limit.
            let used = program.vm_mut().gas_used();
            assert!((200_000..400_000).contains(&used), "{entry}: used {used}");

            program.vm_mut().set_gas_budget(50_000_000);
            program.vm_mut().reset_gas_counter();
            let value = program.call_raw(entry, vec![crate::Value::Int(100_000)])?;
            let expected = if entry == "main.count" {
                4_999_950_000
            } else {
                3 * 4_999_950_000
            };
            assert_eq!(value, crate::Value::Int(expected));
            let used = program.vm_mut().gas_used();
            assert!(used > 100_000 && used < 50_000_000, "{entry}: used {used}");
        }
        Ok(())
    }

    #[test]
    fn memory_budget_traps_on_growth() -> Result<()> {
        let mut program = EmbeddedProgram::builder()
            .module(
                "main",
                r#"
                    function grow(): ()
                        local arr: Array<int> = []
                        array.push(arr, 1)
                    end
                "#,
            )
            .entry_module("main")
            .compile()?;

        program.vm_mut().set_memory_budget_bytes(32);
        program.vm_mut().reset_memory_counter();
        let err = program.call_raw("main.grow", vec![]).unwrap_err();
        match err {
            LustError::RuntimeErrorWithTrace { message, .. }
            | LustError::RuntimeError { message } => {
                assert!(message.to_lowercase().contains("memory budget"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
        Ok(())
    }
}
