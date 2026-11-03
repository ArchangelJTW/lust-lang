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
            self.invalidate_jit_code();
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

    /// Set a memory budget in bytes.
    ///
    /// While a memory budget is active, effective execution safely switches to the
    /// interpreter (Stage A) so existing logical array and growth checks apply.
    /// This does not charge all allocations or guarantee RSS/physical memory containment;
    /// other allocation classes and complete process containment are out of scope.
    /// Configured JIT preferences are preserved and will resume if the budget is cleared.
    ///
    /// # Errors
    /// Returns `Err` if called while VM execution is active (e.g. inside a native callback,
    /// reentrant call, or executing JIT/bytecode frame). An error leaves the previous
    /// budget policy and counters completely unchanged.
    #[must_use = "memory budget mutations are fallible and must not be ignored; an ignored error leaves the previous budget policy and counters unchanged"]
    pub fn set_memory_budget_bytes(&mut self, limit_bytes: usize) -> Result<()> {
        if !self.is_idle() {
            return Err(LustError::RuntimeError {
                message: "Cannot modify memory budget while VM execution is active".to_string(),
            });
        }
        self.budgets.memory.limit_bytes = Some(limit_bytes);
        self.invalidate_jit_code();
        Ok(())
    }

    /// Set a memory budget in kilobytes (1 KB = 1024 bytes).
    ///
    /// While a memory budget is active, effective execution safely switches to the
    /// interpreter (Stage A) so existing logical array and growth checks apply.
    /// This does not charge all allocations or guarantee RSS/physical memory containment;
    /// other allocation classes and complete process containment are out of scope.
    ///
    /// # Errors
    /// Returns `Err` if called while VM execution is active (e.g. inside a native callback,
    /// reentrant call, or executing JIT/bytecode frame). An error leaves the previous
    /// budget policy and counters completely unchanged.
    #[must_use = "memory budget mutations are fallible and must not be ignored; an ignored error leaves the previous budget policy and counters unchanged"]
    pub fn set_memory_budget_kb(&mut self, limit_kb: u64) -> Result<()> {
        let bytes = limit_kb.saturating_mul(1024);
        let limit_bytes = usize::try_from(bytes).unwrap_or(usize::MAX);
        self.set_memory_budget_bytes(limit_bytes)
    }

    /// Clear the active memory budget and reset the memory counter.
    ///
    /// When cleared, execution transitions back to normal unmetered execution
    /// (with JIT enabled if configured), and previously compiled code is invalidated
    /// so stale entries are never reused.
    ///
    /// # Errors
    /// Returns `Err` if called while VM execution is active (e.g. inside a native callback,
    /// reentrant call, or executing JIT/bytecode frame). An error leaves the previous
    /// budget policy and counters completely unchanged.
    #[must_use = "memory budget mutations are fallible and must not be ignored; an ignored error leaves the previous budget policy and counters unchanged"]
    pub fn clear_memory_budget(&mut self) -> Result<()> {
        if !self.is_idle() {
            return Err(LustError::RuntimeError {
                message: "Cannot clear memory budget while VM execution is active".to_string(),
            });
        }
        self.budgets.memory.limit_bytes = None;
        self.budgets.memory.used_bytes = 0;
        self.invalidate_jit_code();
        Ok(())
    }

    /// Reset the cumulative memory usage counter to zero without clearing the limit.
    ///
    /// # Errors
    /// Returns `Err` if called while VM execution is active (e.g. inside a native callback,
    /// reentrant call, or executing JIT/bytecode frame). An error leaves the previous
    /// budget policy and counters completely unchanged.
    #[must_use = "memory budget mutations are fallible and must not be ignored; an ignored error leaves the previous budget policy and counters unchanged"]
    pub fn reset_memory_counter(&mut self) -> Result<()> {
        if !self.is_idle() {
            return Err(LustError::RuntimeError {
                message: "Cannot reset memory counter while VM execution is active".to_string(),
            });
        }
        self.budgets.memory.used_bytes = 0;
        Ok(())
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

        program.vm_mut().set_memory_budget_bytes(32)?;
        program.vm_mut().reset_memory_counter()?;
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

    #[test]
    fn warm_root_trace_before_budget_then_no_native_entry_and_budget_error() -> Result<()> {
        let source = r#"
            function loop_push(count: int): int
                local arr: Array<int> = []
                local i: int = 0
                while i < count do
                    array.push(arr, i)
                    i = i + 1
                end
                return array.len(arr)
            end
        "#;
        let mut program = EmbeddedProgram::builder()
            .module("main", source)
            .entry_module("main")
            .compile()?;

        assert!(program.jit_configured_enabled());
        assert!(program.effective_jit_enabled());

        // Warm up the loop so that root traces are compiled and native code is entered.
        for _ in 0..10 {
            let res = program.call_typed::<_, i64>("main.loop_push", 50i64)?;
            assert_eq!(res, 50);
        }
        let warm_stats = program.vm().jit_stats();
        #[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            assert!(
                warm_stats.native_trace_entries > 0,
                "expected native code to be warm and entered before enabling budget, got: {:?}",
                warm_stats
            );
        }

        // Now enable tight budget on the same VM.
        program.set_memory_budget_bytes(10 * 1024)?;
        assert!(!program.effective_jit_enabled());
        assert!(program.jit_configured_enabled());

        // Snapshot JIT stats after budget transition.
        let pre_call_stats = program.vm().jit_stats();

        // 100,000 pushes under 10KB quota must fail with controlled budget error.
        let err = program
            .call_typed::<_, i64>("main.loop_push", 100_000i64)
            .unwrap_err();
        match &err {
            LustError::RuntimeErrorWithTrace { message, .. }
            | LustError::RuntimeError { message } => {
                assert!(
                    message.to_lowercase().contains("memory budget"),
                    "expected memory budget error, got: {message}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }

        // Verify NO new native trace entries occurred during the budgeted run!
        let post_call_stats = program.vm().jit_stats();
        assert_eq!(
            post_call_stats.native_trace_entries, pre_call_stats.native_trace_entries,
            "native trace entries must not increase while memory budget is active"
        );
        assert_eq!(
            post_call_stats.root_traces_compiled, pre_call_stats.root_traces_compiled,
            "root traces must not be compiled while memory budget is active"
        );

        Ok(())
    }

    #[test]
    fn warm_whole_function_before_budget_then_interpreter_fallback() -> Result<()> {
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
        let mut program = EmbeddedProgram::builder()
            .module("main", source)
            .entry_module("main")
            .compile()?;

        // Warm up count to trigger whole-function compilation
        assert_eq!(
            program.call_raw("main.warm", vec![])?,
            crate::Value::Int(3000)
        );

        #[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let warm_stats = program.vm().jit_stats();
            assert!(
                warm_stats.functions_compiled > 0,
                "expected functions to be compiled before setting budget, got: {:?}",
                warm_stats
            );
            assert!(
                warm_stats.native_trace_entries > 0,
                "expected native entry before setting budget, got: {:?}",
                warm_stats
            );
        }

        // Set memory budget
        program.set_memory_budget_bytes(1024 * 1024)?;
        assert!(!program.effective_jit_enabled());

        let pre_call_stats = program.vm().jit_stats();
        // Call count(10) -> sum of 0..10 is 45
        let res = program.call_raw("main.count", vec![crate::Value::Int(10)])?;
        assert_eq!(res, crate::Value::Int(45));

        let post_call_stats = program.vm().jit_stats();
        assert_eq!(
            post_call_stats.native_trace_entries, pre_call_stats.native_trace_entries,
            "no native entry should occur for whole-function call when budget is active"
        );
        assert_eq!(
            post_call_stats.functions_compiled, pre_call_stats.functions_compiled,
            "no additional function compilation should occur when budget is active"
        );

        Ok(())
    }

    #[test]
    fn nested_loop_bridge_under_budget() -> Result<()> {
        let source = r#"
            function inner(n: int): int
                local i: int = 0
                local sum: int = 0
                while i < n do
                    sum = sum + i
                    i = i + 1
                end
                return sum
            end
            function outer(m: int, n: int): int
                local j: int = 0
                local total: int = 0
                while j < m do
                    total = total + inner(n)
                    j = j + 1
                end
                return total
            end
        "#;
        let mut program = EmbeddedProgram::builder()
            .module("main", source)
            .entry_module("main")
            .compile()?;

        // Run warmup
        for _ in 0..5 {
            let res = program.call_typed::<_, i64>("main.outer", (5i64, 5i64))?;
            assert_eq!(res, 50);
        }

        program.set_memory_budget_bytes(512 * 1024)?;
        assert!(!program.effective_jit_enabled());
        let pre_stats = program.vm().jit_stats();

        let res = program.call_typed::<_, i64>("main.outer", (10i64, 10i64))?;
        assert_eq!(res, 450);

        let post_stats = program.vm().jit_stats();
        assert_eq!(
            post_stats.native_trace_entries, pre_stats.native_trace_entries,
            "nested bridge must not enter native traces while budget is active"
        );

        Ok(())
    }

    #[test]
    fn callback_attempts_mutations_and_verifies_rejection_and_state_preservation() -> Result<()> {
        let source = r#"
            extern
                function host_callback(): int
            end
            function run(): int
                return host_callback()
            end
        "#;
        let mut program = EmbeddedProgram::builder()
            .module("main", source)
            .entry_module("main")
            .compile()?;

        // Set initial budget
        program.set_memory_budget_bytes(50_000)?;
        assert_eq!(program.memory_remaining_bytes(), Some(50_000));
        assert_eq!(program.memory_used_bytes(), 0);

        program.register_typed_native(
            "host_callback",
            |_: ()| -> std::result::Result<i64, String> {
                crate::VM::with_current(|vm| {
                    // VM must be detected as not idle during native callback
                    assert!(
                        !vm.is_idle(),
                        "VM must not be idle during callback execution"
                    );

                    // 1. Attempt set_memory_budget_bytes
                    let err_set = vm.set_memory_budget_bytes(100).unwrap_err();
                    assert!(
                        format!("{err_set}")
                            .contains("Cannot modify memory budget while VM execution is active")
                    );

                    // 2. Attempt set_memory_budget_kb
                    let err_kb = vm.set_memory_budget_kb(2).unwrap_err();
                    assert!(
                        format!("{err_kb}")
                            .contains("Cannot modify memory budget while VM execution is active")
                    );

                    // 3. Attempt clear_memory_budget
                    let err_clear = vm.clear_memory_budget().unwrap_err();
                    assert!(
                        format!("{err_clear}")
                            .contains("Cannot clear memory budget while VM execution is active")
                    );

                    // 4. Attempt reset_memory_counter
                    let err_reset = vm.reset_memory_counter().unwrap_err();
                    assert!(
                        format!("{err_reset}")
                            .contains("Cannot reset memory counter while VM execution is active")
                    );

                    // 5. Test ignored Result: must not alter state
                    let _ = vm.set_memory_budget_bytes(1);
                    let _ = vm.clear_memory_budget();
                    let _ = vm.reset_memory_counter();

                    // State must remain completely unchanged!
                    assert_eq!(vm.memory_remaining_bytes(), Some(50_000));
                    assert_eq!(vm.memory_used_bytes(), 0);

                    Ok(42i64)
                })
            },
        )?;

        let res = program.call_typed::<_, i64>("main.run", ())?;
        assert_eq!(res, 42);

        // State after returning to host must also remain exactly as before
        assert_eq!(program.memory_remaining_bytes(), Some(50_000));
        assert_eq!(program.memory_used_bytes(), 0);

        Ok(())
    }

    #[test]
    fn root_trace_invoking_callback_rejects_budget_mutations() -> Result<()> {
        let source = r#"
            extern
                function host_callback(step: int): int
            end
            function loop_with_callback(count: int): int
                local i: int = 0
                local sum: int = 0
                while i < count do
                    sum = sum + host_callback(i)
                    i = i + 1
                end
                return sum
            end
        "#;
        let mut program = EmbeddedProgram::builder()
            .module("main", source)
            .entry_module("main")
            .compile()?;

        assert!(program.jit_configured_enabled());
        assert!(program.effective_jit_enabled());

        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

        let call_count = Arc::new(AtomicU64::new(0));
        let saw_active_jit = Arc::new(AtomicBool::new(false));
        let saw_native_entry = Arc::new(AtomicBool::new(false));

        let call_count_cb = Arc::clone(&call_count);
        let saw_active_jit_cb = Arc::clone(&saw_active_jit);
        let saw_native_entry_cb = Arc::clone(&saw_native_entry);

        program.register_typed_native(
            "host_callback",
            move |step: i64| -> std::result::Result<i64, String> {
                let _count = call_count_cb.fetch_add(1, Ordering::SeqCst);
                crate::VM::with_current(|vm| {
                    assert!(
                        !vm.is_idle(),
                        "VM must not be idle during callback execution"
                    );

                    if vm.active_jit_depth > 0 {
                        saw_active_jit_cb.store(true, Ordering::SeqCst);
                    }
                    if vm.jit_stats().native_trace_entries > 0 {
                        saw_native_entry_cb.store(true, Ordering::SeqCst);
                    }

                    // 1. Attempt set_memory_budget_bytes
                    let err_set = vm.set_memory_budget_bytes(100).unwrap_err();
                    assert!(
                        format!("{err_set}")
                            .contains("Cannot modify memory budget while VM execution is active")
                    );

                    // 2. Attempt set_memory_budget_kb
                    let err_kb = vm.set_memory_budget_kb(2).unwrap_err();
                    assert!(
                        format!("{err_kb}")
                            .contains("Cannot modify memory budget while VM execution is active")
                    );

                    // 3. Attempt clear_memory_budget
                    let err_clear = vm.clear_memory_budget().unwrap_err();
                    assert!(
                        format!("{err_clear}")
                            .contains("Cannot clear memory budget while VM execution is active")
                    );

                    // 4. Attempt reset_memory_counter
                    let err_reset = vm.reset_memory_counter().unwrap_err();
                    assert!(
                        format!("{err_reset}")
                            .contains("Cannot reset memory counter while VM execution is active")
                    );

                    // 5. Test ignored Result: must not alter state
                    let _ = vm.set_memory_budget_bytes(1);
                    let _ = vm.clear_memory_budget();
                    let _ = vm.reset_memory_counter();

                    // State must remain completely unchanged (unmetered: None / 0 used)
                    assert_eq!(vm.memory_remaining_bytes(), None);
                    assert_eq!(vm.memory_used_bytes(), 0);
                    assert!(vm.effective_jit_enabled());

                    Ok(step)
                })
            },
        )?;

        let sum = program.call_typed::<_, i64>("main.loop_with_callback", 50i64)?;
        assert_eq!(sum, (0..50).sum::<i64>());
        assert_eq!(call_count.load(Ordering::SeqCst), 50);

        #[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let stats = program.vm().jit_stats();
            assert!(
                stats.root_traces_compiled > 0,
                "expected root trace to compile, got: {:?}",
                stats
            );
            assert!(
                stats.native_trace_entries > 0,
                "expected native trace entries, got: {:?}",
                stats
            );
            assert!(
                saw_active_jit.load(Ordering::SeqCst),
                "expected callback to be executed from active JIT trace"
            );
            assert!(
                saw_native_entry.load(Ordering::SeqCst),
                "expected native trace entry to have been witnessed during/before callback"
            );
        }

        // State after returning to host must remain unmetered and effective JIT enabled
        assert_eq!(program.memory_remaining_bytes(), None);
        assert_eq!(program.memory_used_bytes(), 0);
        assert!(program.effective_jit_enabled());

        Ok(())
    }

    #[test]
    fn reentrant_call_path_rejection() -> Result<()> {
        let source = r#"
            extern
                function outer_cb(): int
                function inner_cb(): int
            end
            function script_inner(): int
                return inner_cb()
            end
            function run(): int
                return outer_cb()
            end
        "#;
        let mut program = EmbeddedProgram::builder()
            .module("main", source)
            .entry_module("main")
            .compile()?;

        program.set_memory_budget_bytes(80_000)?;

        program.register_typed_native("inner_cb", |_: ()| -> std::result::Result<i64, String> {
            crate::VM::with_current(|vm| {
                assert!(!vm.is_idle());
                assert!(vm.set_memory_budget_bytes(10).is_err());
                assert!(vm.clear_memory_budget().is_err());
                assert!(vm.reset_memory_counter().is_err());
                assert_eq!(vm.memory_remaining_bytes(), Some(80_000));
                Ok(777i64)
            })
        })?;

        program.register_typed_native("outer_cb", |_: ()| -> std::result::Result<i64, String> {
            crate::VM::with_current(|vm| {
                assert!(!vm.is_idle());
                assert!(vm.set_memory_budget_bytes(20).is_err());
                // Reentrant call back into Lust script
                let func = vm
                    .function_value("main.script_inner")
                    .expect("script_inner");
                let val = vm.call_value(&func, vec![]).map_err(|e| format!("{e}"))?;
                let n = val.as_int().unwrap_or(0);
                assert_eq!(n, 777);
                // Still rejected after reentrant call returned
                assert!(!vm.is_idle());
                assert!(vm.clear_memory_budget().is_err());
                Ok(n)
            })
        })?;

        let res = program.call_typed::<_, i64>("main.run", ())?;
        assert_eq!(res, 777);
        assert_eq!(program.memory_remaining_bytes(), Some(80_000));

        Ok(())
    }

    #[test]
    fn idle_clear_invalidates_stale_code_and_resumes_jit() -> Result<()> {
        let source = r#"
            function loop_sum(n: int): int
                local i: int = 0
                local sum: int = 0
                while i < n do
                    sum = sum + i
                    i = i + 1
                end
                return sum
            end
        "#;
        let mut program = EmbeddedProgram::builder()
            .module("main", source)
            .entry_module("main")
            .compile()?;

        // 1. Warm up with no budget
        for _ in 0..10 {
            assert_eq!(program.call_typed::<_, i64>("main.loop_sum", 100i64)?, 4950);
        }
        let warm_stats = program.vm().jit_stats();
        #[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
        assert!(warm_stats.native_trace_entries > 0);

        // 2. Set memory budget at idle boundary
        program.set_memory_budget_bytes(1024 * 1024)?;
        assert!(!program.effective_jit_enabled());

        // 3. Clear memory budget at idle boundary
        program.clear_memory_budget()?;
        assert!(program.effective_jit_enabled());
        assert_eq!(program.memory_remaining_bytes(), None);
        assert_eq!(program.memory_used_bytes(), 0);

        // 4. Run loop again: fresh compilation and native execution resumes
        for _ in 0..10 {
            assert_eq!(program.call_typed::<_, i64>("main.loop_sum", 100i64)?, 4950);
        }
        let resumed_stats = program.vm().jit_stats();
        #[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
        assert!(
            resumed_stats.native_trace_entries > 0,
            "native execution must resume after clearing budget"
        );

        Ok(())
    }

    #[test]
    fn configured_jit_off_semantics() -> Result<()> {
        let source = r#"
            function loop_sum(n: int): int
                local i: int = 0
                local sum: int = 0
                while i < n do
                    sum = sum + i
                    i = i + 1
                end
                return sum
            end
        "#;
        let mut cfg = crate::LustConfig::default();
        cfg.set_jit_enabled(false);
        let mut program = EmbeddedProgram::builder()
            .set_config(cfg)
            .module("main", source)
            .entry_module("main")
            .compile()?;

        assert!(!program.jit_configured_enabled());
        assert!(!program.effective_jit_enabled());

        // Warm up - JIT remains completely inactive
        for _ in 0..10 {
            assert_eq!(program.call_typed::<_, i64>("main.loop_sum", 100i64)?, 4950);
        }
        assert_eq!(program.vm().jit_stats().native_trace_entries, 0);

        // Set budget
        program.set_memory_budget_bytes(1024)?;
        assert!(!program.jit_configured_enabled());
        assert!(!program.effective_jit_enabled());

        // Clear budget: configured JIT off must be preserved!
        program.clear_memory_budget()?;
        assert!(!program.jit_configured_enabled());
        assert!(!program.effective_jit_enabled());

        for _ in 0..10 {
            assert_eq!(program.call_typed::<_, i64>("main.loop_sum", 100i64)?, 4950);
        }
        assert_eq!(program.vm().jit_stats().native_trace_entries, 0);

        Ok(())
    }

    #[test]
    fn byte_kb_zero_counter_reset_semantics() -> Result<()> {
        let source = r#"
            function push_elements(): ()
                local arr: Array<int> = []
                array.push(arr, 1)
                array.push(arr, 2)
            end
        "#;
        let mut program = EmbeddedProgram::builder()
            .module("main", source)
            .entry_module("main")
            .compile()?;

        // 1. set_memory_budget_kb(2) -> 2048 bytes
        program.set_memory_budget_kb(2)?;
        assert_eq!(program.memory_remaining_bytes(), Some(2048));

        // 2. set_memory_budget_bytes(1024) -> overrides to 1024
        program.set_memory_budget_bytes(1024)?;
        assert_eq!(program.memory_remaining_bytes(), Some(1024));

        // 3. Execute bounded allocations
        program.call_raw("main.push_elements", vec![])?;
        let used = program.memory_used_bytes();
        assert!(used > 0, "used memory should be tracked");
        assert_eq!(program.memory_remaining_bytes(), Some(1024 - used));

        // 4. reset_memory_counter()
        program.reset_memory_counter()?;
        assert_eq!(program.memory_used_bytes(), 0);
        assert_eq!(program.memory_remaining_bytes(), Some(1024));

        // 5. Zero budget: Some(0) must immediately trap on any positive allocation
        program.set_memory_budget_bytes(0)?;
        assert_eq!(program.memory_remaining_bytes(), Some(0));
        assert!(program.vm().budgets.mem_budget_enabled());
        assert!(!program.effective_jit_enabled());

        let err = program.call_raw("main.push_elements", vec![]).unwrap_err();
        match err {
            LustError::RuntimeErrorWithTrace { message, .. }
            | LustError::RuntimeError { message } => {
                assert!(message.to_lowercase().contains("memory budget"));
            }
            other => panic!("unexpected error: {other:?}"),
        }

        // 6. Clear budget
        program.clear_memory_budget()?;
        assert_eq!(program.memory_remaining_bytes(), None);
        assert_eq!(program.memory_used_bytes(), 0);
        assert!(!program.vm().budgets.mem_budget_enabled());
        assert!(program.effective_jit_enabled());

        Ok(())
    }

    #[test]
    fn aliased_array_contents_preserved_on_quota_failure() -> Result<()> {
        let source = r#"
            function test_preserve(): Array<int>
                local arr: Array<int> = []
                array.push(arr, 10)
                array.push(arr, 20)
                array.push(arr, 30)
                local i: int = 0
                while i < 10000 do
                    array.push(arr, i)
                    i = i + 1
                end
                return arr
            end
        "#;
        let mut program = EmbeddedProgram::builder()
            .module("main", source)
            .entry_module("main")
            .compile()?;

        // Tight budget that allows initial pushes but fails later
        program.set_memory_budget_bytes(64)?;
        let err = program.call_raw("main.test_preserve", vec![]).unwrap_err();
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
