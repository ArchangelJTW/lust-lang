use super::*;
use crate::bytecode::chunk::ChunkIcSlot;
use crate::bytecode::OpCode;
use crate::bytecode::ValueKey;
use crate::bytecode::value::{ClosureObject, EnumObject, Name, StructObject};
use core::ptr;

const OP_LOAD_NIL: u8 = OpCode::LoadNil as u8;
const OP_LOAD_BOOL: u8 = OpCode::LoadBool as u8;
const OP_LOAD_CONST: u8 = OpCode::LoadConst as u8;
const OP_LOAD_GLOBAL: u8 = OpCode::LoadGlobal as u8;
const OP_MOVE: u8 = OpCode::Move as u8;
const OP_ADD_INT: u8 = OpCode::AddInt as u8;
const OP_ADD_INT_IMM: u8 = OpCode::AddIntImm as u8;
const OP_SUB_INT: u8 = OpCode::SubInt as u8;
const OP_MUL_INT: u8 = OpCode::MulInt as u8;
const OP_DIV_INT: u8 = OpCode::DivInt as u8;
const OP_MOD_INT: u8 = OpCode::ModInt as u8;
const OP_POW_INT: u8 = OpCode::PowInt as u8;
const OP_FLOOR_DIV_INT: u8 = OpCode::FloorDivInt as u8;
const OP_NEG_INT: u8 = OpCode::NegInt as u8;
const OP_EQ_INT: u8 = OpCode::EqInt as u8;
const OP_NE_INT: u8 = OpCode::NeInt as u8;
const OP_LT_INT: u8 = OpCode::LtInt as u8;
const OP_LE_INT: u8 = OpCode::LeInt as u8;
const OP_GT_INT: u8 = OpCode::GtInt as u8;
const OP_GE_INT: u8 = OpCode::GeInt as u8;
const OP_ADD_FLOAT: u8 = OpCode::AddFloat as u8;
const OP_SUB_FLOAT: u8 = OpCode::SubFloat as u8;
const OP_MUL_FLOAT: u8 = OpCode::MulFloat as u8;
const OP_DIV_FLOAT: u8 = OpCode::DivFloat as u8;
const OP_MOD_FLOAT: u8 = OpCode::ModFloat as u8;
const OP_POW_FLOAT: u8 = OpCode::PowFloat as u8;
const OP_FLOOR_DIV_FLOAT: u8 = OpCode::FloorDivFloat as u8;
const OP_NEG_FLOAT: u8 = OpCode::NegFloat as u8;
const OP_EQ_FLOAT: u8 = OpCode::EqFloat as u8;
const OP_NE_FLOAT: u8 = OpCode::NeFloat as u8;
const OP_LT_FLOAT: u8 = OpCode::LtFloat as u8;
const OP_LE_FLOAT: u8 = OpCode::LeFloat as u8;
const OP_GT_FLOAT: u8 = OpCode::GtFloat as u8;
const OP_GE_FLOAT: u8 = OpCode::GeFloat as u8;
const OP_ADD: u8 = OpCode::Add as u8;
const OP_SUB: u8 = OpCode::Sub as u8;
const OP_MUL: u8 = OpCode::Mul as u8;
const OP_DIV: u8 = OpCode::Div as u8;
const OP_MOD: u8 = OpCode::Mod as u8;
const OP_POW: u8 = OpCode::Pow as u8;
const OP_FLOOR_DIV: u8 = OpCode::FloorDiv as u8;
const OP_NEG: u8 = OpCode::Neg as u8;
const OP_EQ: u8 = OpCode::Eq as u8;
const OP_NE: u8 = OpCode::Ne as u8;
const OP_LT: u8 = OpCode::Lt as u8;
const OP_LE: u8 = OpCode::Le as u8;
const OP_GT: u8 = OpCode::Gt as u8;
const OP_GE: u8 = OpCode::Ge as u8;
const OP_AND: u8 = OpCode::And as u8;
const OP_OR: u8 = OpCode::Or as u8;
const OP_NOT: u8 = OpCode::Not as u8;
const OP_BIT_AND: u8 = OpCode::BitAnd as u8;
const OP_BIT_OR: u8 = OpCode::BitOr as u8;
const OP_BIT_XOR: u8 = OpCode::BitXor as u8;
const OP_SHL: u8 = OpCode::Shl as u8;
const OP_SHR: u8 = OpCode::Shr as u8;
const OP_BIT_NOT: u8 = OpCode::BitNot as u8;
const OP_JUMP: u8 = OpCode::Jump as u8;
const OP_JUMP_IF: u8 = OpCode::JumpIf as u8;
const OP_JUMP_IF_NOT: u8 = OpCode::JumpIfNot as u8;
const OP_BRANCH_LT_INT: u8 = OpCode::BranchLtInt as u8;
const OP_BRANCH_LE_INT: u8 = OpCode::BranchLeInt as u8;
const OP_NEW_ARRAY: u8 = OpCode::NewArray as u8;
const OP_TUPLE_NEW: u8 = OpCode::TupleNew as u8;
const OP_TUPLE_GET: u8 = OpCode::TupleGet as u8;
const OP_IS_ENUM_VARIANT: u8 = OpCode::IsEnumVariant as u8;
const OP_GET_ENUM_VALUE: u8 = OpCode::GetEnumValue as u8;
const OP_NEW_STRUCT: u8 = OpCode::NewStruct as u8;
const OP_NEW_ENUM_UNIT: u8 = OpCode::NewEnumUnit as u8;
const OP_NEW_ENUM_VARIANT: u8 = OpCode::NewEnumVariant as u8;
const OP_GET_FIELD_SLOT: u8 = OpCode::GetFieldSlot as u8;
const OP_SET_FIELD_SLOT: u8 = OpCode::SetFieldSlot as u8;
const OP_GET_FIELD_ADD_SET_INT: u8 = OpCode::GetFieldAddSetInt as u8;
const OP_GET_FIELD: u8 = OpCode::GetField as u8;
const OP_SET_FIELD: u8 = OpCode::SetField as u8;
const OP_GET_FIELD_COMPACT: u8 = OpCode::GetFieldCompact as u8;
const OP_SET_FIELD_COMPACT: u8 = OpCode::SetFieldCompact as u8;

const BYTECODE_REGISTER_WINDOW: usize = 256;

fn register_window_oom() -> LustError {
    LustError::RuntimeError {
        message: "Out of memory allocating VM register window".to_string(),
    }
}

#[inline]
fn register_window_end(base: usize, count: u16) -> Result<usize> {
    base.checked_add(usize::from(count).max(BYTECODE_REGISTER_WINDOW))
        .ok_or_else(register_window_oom)
}

#[inline]
fn reserve_register_window(
    registers: &mut Vec<Value, crate::alloc_budget::VmAllocHandle>,
    budgets: &mut budget::BudgetState,
    end: usize,
) -> Result<()> {
    if end > registers.len() {
        if end > registers.capacity() {
            let bytes = (end - registers.capacity())
                .checked_mul(core::mem::size_of::<Value>())
                .ok_or_else(register_window_oom)?;
            let old_usage = budgets.memory.used_bytes;
            if !budgets.try_charge_mem_bytes(bytes) {
                return Err(register_window_oom());
            }
            if registers.try_reserve_exact(end - registers.len()).is_err() {
                budgets.memory.used_bytes = old_usage;
                return Err(register_window_oom());
            }
        }
        registers.resize(end, Value::Nil);
    }
    Ok(())
}
const OP_CONCAT: u8 = OpCode::Concat as u8;
const OP_GET_INDEX: u8 = OpCode::GetIndex as u8;
const OP_ARRAY_GET_UNWRAP: u8 = OpCode::ArrayGetUnwrap as u8;
const OP_ARRAY_PUSH: u8 = OpCode::ArrayPush as u8;
const OP_SET_INDEX: u8 = OpCode::SetIndex as u8;
const OP_ARRAY_INDEX_OK: u8 = OpCode::ArrayIndexOk as u8;
const OP_MAP_GET: u8 = OpCode::MapGet as u8;
const OP_MAP_SET: u8 = OpCode::MapSet as u8;
const OP_MAP_GET_UNWRAP: u8 = OpCode::MapGetUnwrap as u8;
const OP_ARRAY_LEN: u8 = OpCode::ArrayLen as u8;
const OP_LOAD_UPVALUE: u8 = OpCode::LoadUpvalue as u8;
const OP_STORE_UPVALUE: u8 = OpCode::StoreUpvalue as u8;
const OP_CALL: u8 = OpCode::Call as u8;
const OP_CALL_METHOD: u8 = OpCode::CallMethod as u8;
const OP_RETURN: u8 = OpCode::Return as u8;

#[cfg(feature = "std")]
std::thread_local! {
    static OPTION_NONE_SINGLETON: Rc<EnumObject> = EnumObject::new("Option", "None", None);
}

#[cfg(feature = "std")]
#[inline(always)]
fn option_none_singleton() -> Rc<EnumObject> {
    OPTION_NONE_SINGLETON.with(Rc::clone)
}

pub static MUTATION_M1_SKIP_GAS: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
pub static MUTATION_M1_EXECUTION_COUNT: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

#[cfg(all(feature = "std", debug_assertions))]
std::thread_local! {
    static FRAME_HOIST_RELOAD_COUNT: core::cell::Cell<u64> = const { core::cell::Cell::new(0) };
    static CLOSURE_UPVALUE_BUFFER_GROWTHS: core::cell::Cell<u64> = const { core::cell::Cell::new(0) };
}

#[inline(always)]
fn record_frame_hoist_reload() {
    #[cfg(all(feature = "std", debug_assertions))]
    FRAME_HOIST_RELOAD_COUNT.with(|c| c.set(c.get().wrapping_add(1)));
}

#[inline]
fn charge_array_push_budget(
    budgets: &mut budget::BudgetState,
    arr: &crate::bytecode::LustArray,
    val: &Value,
) -> Result<()> {
    if !budgets.mem_budget_enabled() {
        return Ok(());
    }
    let bytes = arr.with_unboxed_data(|data| match data {
        crate::bytecode::value::UnboxedArrayData::Int(v) => {
            if matches!(val, Value::Int(_)) {
                if v.len() == v.capacity() {
                    let cap = v.capacity();
                    let new_cap = if cap == 0 { 4 } else { cap.saturating_mul(2) };
                    (new_cap - cap).saturating_mul(core::mem::size_of::<i64>())
                } else {
                    0
                }
            } else {
                v.len()
                    .saturating_add(1)
                    .saturating_mul(core::mem::size_of::<Value>())
            }
        }
        crate::bytecode::value::UnboxedArrayData::Float(v) => {
            if matches!(val, Value::Float(_)) {
                if v.len() == v.capacity() {
                    let cap = v.capacity();
                    let new_cap = if cap == 0 { 4 } else { cap.saturating_mul(2) };
                    (new_cap - cap).saturating_mul(core::mem::size_of::<f64>())
                } else {
                    0
                }
            } else {
                v.len()
                    .saturating_add(1)
                    .saturating_mul(core::mem::size_of::<Value>())
            }
        }
        crate::bytecode::value::UnboxedArrayData::Bool(v) => {
            if matches!(val, Value::Bool(_)) {
                if v.len() == v.capacity() {
                    let cap = v.capacity();
                    let new_cap = if cap == 0 { 4 } else { cap.saturating_mul(2) };
                    (new_cap - cap).saturating_mul(core::mem::size_of::<u8>())
                } else {
                    0
                }
            } else {
                v.len()
                    .saturating_add(1)
                    .saturating_mul(core::mem::size_of::<Value>())
            }
        }
        crate::bytecode::value::UnboxedArrayData::None => {
            let elems = arr.borrow_elements_without_sync();
            if elems.is_empty() {
                match val {
                    Value::Int(_) => core::mem::size_of::<i64>(),
                    Value::Float(_) => core::mem::size_of::<f64>(),
                    Value::Bool(_) => core::mem::size_of::<u8>(),
                    _ => 4usize.saturating_mul(core::mem::size_of::<Value>()),
                }
            } else if elems.len() == elems.capacity() {
                let cap = elems.capacity();
                let new_cap = if cap == 0 { 4 } else { cap.saturating_mul(2) };
                (new_cap - cap).saturating_mul(core::mem::size_of::<Value>())
            } else {
                0
            }
        }
    });
    if bytes > 0 {
        budgets.charge_mem_bytes(bytes)?;
    }
    Ok(())
}

#[inline(always)]
fn write_frame_reg_scalar(
    regs: &mut [Value; BYTECODE_REGISTER_WINDOW],
    reg: Register,
    value: Value,
) -> bool {
    regs[reg as usize] = value;
    false
}

#[inline(always)]
fn write_frame_reg_fast(
    frame: &mut CallFrame,
    regs: &mut [Value; BYTECODE_REGISTER_WINDOW],
    cycle_collector: &mut cycle::CycleCollector,
    reg: Register,
    value: Value,
) -> bool {
    let idx = reg as usize;
    if !value.is_plain() {
        frame.mark_non_plain(idx);
        if cycle_collector.is_enabled() {
            cycle_collector.register_value(&value);
        }
    }
    regs[idx] = value;
    cycle_collector.is_enabled() && cycle_collector.should_collect()
}

#[inline]
fn map_read(map: &Value, key: &Value, unwrap: bool) -> Result<Value> {
    let Value::Map(map) = map else {
        return Err(LustError::RuntimeError {
            message: format!("map.get expects a map, got {:?}", map.type_of()),
        });
    };
    let found = map
        .borrow()
        .get(&ValueKey::from_value(key))
        .map(Value::fast_clone);
    match (found, unwrap) {
        (Some(value), true) => Ok(value),
        (Some(value), false) => Ok(Value::some(value)),
        (None, false) => Ok(Value::none()),
        (None, true) => Err(LustError::RuntimeError {
            message: "Called unwrap() on Option::None".to_string(),
        }),
    }
}

#[inline]
fn map_write(
    map: &Value,
    key: &Value,
    value: &Value,
    budgets: &mut crate::vm::budget::BudgetState,
    cycle_collector: &mut cycle::CycleCollector,
) -> Result<bool> {
    let Value::Map(map) = map else {
        return Err(LustError::RuntimeError {
            message: format!("map.set expects a map, got {:?}", map.type_of()),
        });
    };
    let stores_container =
        !cycle::CycleCollector::is_leaf(key) || !cycle::CycleCollector::is_leaf(value);
    let key = ValueKey::from_value(key);
    let mut map = map.borrow_mut();
    if budgets.mem_budget_enabled() && !map.contains_key(&key) {
        budgets
            .charge_map_entry_estimate()
            .map_err(|err| LustError::RuntimeError {
                message: err.to_string(),
            })?;
    }
    map.insert(key, value.fast_clone());
    if stores_container {
        cycle_collector.record_mutation();
    }
    Ok(stores_container)
}

#[inline(always)]
fn frame_int_binary_op<F>(
    regs: &mut [Value; BYTECODE_REGISTER_WINDOW],
    dest: Register,
    lhs: Register,
    rhs: Register,
    op: F,
) -> Result<bool>
where
    F: FnOnce(LustInt, LustInt) -> Result<Value>,
{
    let left = &regs[lhs as usize];
    let right = &regs[rhs as usize];
    let (Value::Int(a), Value::Int(b)) = (left, right) else {
        return Err(LustError::RuntimeError {
            message: format!(
                "Typed numeric instruction expected int operands, got {:?} and {:?}",
                left, right
            ),
        });
    };
    let result = op(*a, *b)?;
    Ok(write_frame_reg_scalar(regs, dest, result))
}

#[inline(always)]
fn frame_float_binary_op<F>(
    regs: &mut [Value; BYTECODE_REGISTER_WINDOW],
    dest: Register,
    lhs: Register,
    rhs: Register,
    op: F,
) -> Result<bool>
where
    F: FnOnce(LustFloat, LustFloat) -> Result<Value>,
{
    let left = &regs[lhs as usize];
    let right = &regs[rhs as usize];
    let (Value::Float(a), Value::Float(b)) = (left, right) else {
        return Err(LustError::RuntimeError {
            message: format!(
                "Typed numeric instruction expected float operands, got {:?} and {:?}",
                left, right
            ),
        });
    };
    let result = op(*a, *b)?;
    Ok(write_frame_reg_scalar(regs, dest, result))
}

#[inline(always)]
fn frame_binary_op<F>(
    frame: &mut CallFrame,
    regs: &mut [Value; BYTECODE_REGISTER_WINDOW],
    cycle_collector: &mut cycle::CycleCollector,
    dest: Register,
    lhs: Register,
    rhs: Register,
    op: F,
) -> Result<bool>
where
    F: FnOnce(&Value, &Value) -> Result<Value>,
{
    let left = &regs[lhs as usize];
    let right = &regs[rhs as usize];
    let result = op(left, right)?;
    Ok(write_frame_reg_fast(frame, regs, cycle_collector, dest, result))
}

#[inline(always)]
fn frame_comparison_op<I, F>(
    regs: &mut [Value; BYTECODE_REGISTER_WINDOW],
    dest: Register,
    lhs: Register,
    rhs: Register,
    int_op: I,
    op: F,
) -> Result<bool>
where
    I: FnOnce(LustInt, LustInt) -> bool,
    F: FnOnce(LustFloat, LustFloat) -> bool,
{
    let left = &regs[lhs as usize];
    let right = &regs[rhs as usize];
    let result = match (left, right) {
        (Value::Int(a), Value::Int(b)) => int_op(*a, *b),
        (Value::Float(a), Value::Float(b)) => op(*a, *b),
        (Value::Int(a), Value::Float(b)) => op(float_from_int(*a), *b),
        (Value::Float(a), Value::Int(b)) => op(*a, float_from_int(*b)),
        _ => {
            return Err(LustError::RuntimeError {
                message: format!("Cannot compare {:?} and {:?}", left, right),
            });
        }
    };
    Ok(write_frame_reg_scalar(regs, dest, Value::Bool(result)))
}

enum FrameExit {
    EndOfChunk,
    CollectCycles,
    OutOfGas(u64),
    FallbackToOuter,
    #[cfg(feature = "std")]
    TryOsr(usize),
    Slow(Instruction, usize),
}

enum FastStep {
    InLoopCall {
        target_func_idx: usize,
        closure_reg: Option<Register>,
        first_arg: Register,
        arg_count: u8,
        dest: Register,
        ip_before_execution: usize,
    },
    InLoopCallMethod {
        target_func_idx: usize,
        obj_reg: Register,
        first_arg: Register,
        arg_count: u8,
        dest: Register,
        ip_before_execution: usize,
    },
    InLoopReturn {
        ret_reg: Register,
        ip_before_execution: usize,
    },
}

pub struct MutationM1Guard;

impl MutationM1Guard {
    pub fn enable() -> Self {
        MUTATION_M1_SKIP_GAS.store(true, core::sync::atomic::Ordering::SeqCst);
        Self
    }
}

impl Drop for MutationM1Guard {
    fn drop(&mut self) {
        MUTATION_M1_SKIP_GAS.store(false, core::sync::atomic::Ordering::SeqCst);
    }
}

/// How thoroughly a call's arguments are checked against the signature.
#[derive(Clone, Copy)]
pub(super) enum ArgCheck {
    /// Full validation, including container contents (host and dynamic calls).
    Deep,
    /// Kinds only; the typechecker proved the rest (bytecode calls).
    Shallow,
}

#[inline]
fn wrapping_pow_i64(mut base: i64, mut exp: u64) -> i64 {
    let mut acc: i64 = 1;
    while exp > 0 {
        if (exp & 1) == 1 {
            acc = acc.wrapping_mul(base);
        }
        base = base.wrapping_mul(base);
        exp >>= 1;
    }
    acc
}

#[inline]
fn int_pow(a: i64, b: i64) -> Result<Value> {
    if b < 0 {
        return Err(LustError::RuntimeError {
            message: "Negative exponent in integer power; cast base to float first".to_string(),
        });
    }
    Ok(Value::Int(wrapping_pow_i64(a, b as u64)))
}

#[inline]
fn floor_div_i64(a: i64, b: i64) -> Result<i64> {
    if b == 0 {
        return Err(LustError::RuntimeError {
            message: "Division by zero".to_string(),
        });
    }
    if a == i64::MIN && b == -1 {
        return Ok(i64::MIN);
    }
    let d = a / b;
    let r = a % b;
    let floor_d = if (r != 0) && ((r < 0) != (b < 0)) { d - 1 } else { d };
    Ok(floor_d)
}

impl VM {
    /// Reports whether PERF-T3-2 frame/chunk cache hoisting with reload barriers is active.
    #[inline]
    pub const fn frame_cache_hoisting_enabled() -> bool {
        true
    }

    /// Returns the per-thread count of frame-cache reloads performed in `run_inner`
    /// (tracked in debug/test builds; returns 0 in release builds for zero overhead).
    #[inline]
    pub fn frame_hoist_reload_count() -> u64 {
        #[cfg(all(feature = "std", debug_assertions))]
        {
            FRAME_HOIST_RELOAD_COUNT.with(|c| c.get())
        }
        #[cfg(not(all(feature = "std", debug_assertions)))]
        {
            0
        }
    }

    /// Counts capacity growths of in-loop closure upvalue buffers in debug builds.
    #[cfg(all(feature = "std", debug_assertions))]
    pub fn closure_upvalue_buffer_growths() -> u64 {
        CLOSURE_UPVALUE_BUFFER_GROWTHS.with(core::cell::Cell::get)
    }

    fn is_loop_in_hierarchy(
        &self,
        function_idx: usize,
        loop_start_ip: usize,
        backedge_ip: usize,
    ) -> bool {
        let instructions = &self.functions[function_idx].chunk.instructions;
        let jump_target =
            |ip: usize, offset: i16| (ip as isize + 1 + offset as isize).max(0) as usize;
        let contains_nested_loop = instructions
            .iter()
            .enumerate()
            .skip(loop_start_ip)
            .take(backedge_ip.saturating_sub(loop_start_ip))
            .any(|(ip, instruction)| {
                let Instruction::Jump(offset) = instruction else {
                    return false;
                };
                *offset < 0 && jump_target(ip, *offset) > loop_start_ip
            });
        let is_nested_loop = instructions
            .iter()
            .enumerate()
            .skip(backedge_ip.saturating_add(1))
            .any(|(ip, instruction)| {
                let Instruction::Jump(offset) = instruction else {
                    return false;
                };
                *offset < 0 && jump_target(ip, *offset) < loop_start_ip
            });
        contains_nested_loop || is_nested_loop
    }

    fn lua_table_map(value: &Value) -> Option<Value> {
        if let Value::Enum(object) = value
            && object.enum_name == "LuaValue"
            && object.variant == "Table"
            && let Some(inner) = object.values.as_ref().and_then(|vals| vals.first())
            && let Some(map) = inner.struct_get_field("table")
        {
            return Some(map);
        }

        if let Value::Struct(object) = value
            && object.name == "LuaTable"
            && let Some(map) = value.struct_get_field("table")
        {
            return Some(map);
        }

        None
    }

    fn lua_table_key_value(value: &Value) -> Value {
        if let Value::Enum(object) = value
            && object.enum_name == "LuaValue"
        {
            let EnumObject {
                variant, values, ..
            } = object.as_ref();
            return match variant.as_str() {
                "Nil" => Value::Nil,
                "Bool" | "Int" | "Float" | "String" | "Table" | "Function" | "LightUserdata"
                | "Userdata" | "Thread" => values
                    .as_ref()
                    .and_then(|v| v.first())
                    .cloned()
                    .unwrap_or(Value::Nil),
                _ => value.clone(),
            };
        }
        value.clone()
    }

    pub fn run(&mut self) -> Result<Value> {
        let _vm_guard = crate::vm::CurrentVmGuard::from_vm(self);
        #[cfg(feature = "std")]
        if self.backend == crate::vm::VmBackend::Cranelift {
            return self.run_cranelift();
        }
        // Specialize the dispatch loop once rather than testing it around every
        // instruction; the interpreter-only copy has no trace bookkeeping or JIT backedges.
        // When a memory budget is active, effective execution is the interpreter.
        let result = if self.effective_jit_enabled() {
            self.run_inner::<true>()
        } else {
            self.run_inner::<false>()
        };
        crate::bytecode::value::sync_all_specialized_arrays();
        result
    }

    #[cold]
    #[inline(never)]
    fn run_dispatch_preamble_slow(&mut self) -> Result<Option<Value>> {
        if let Some(task) = self.current_task.and_then(|id| self.task_manager.get_mut(id)) {
            if task.poll_interrupt().is_some() {
                return Err(task.error.clone().expect("interrupt error"));
            }
        }
        if let Some(target_depth) = self.call_until_depth
            && self.call_stack.len() == target_depth
            && let Some(return_value) = self.pending_return_value.take()
        {
            self.call_until_depth = None;
            self.sync_slow_path_flags();
            return Ok(Some(return_value));
        }

        if self.pending_return_value.is_some()
            && let Some(return_value) = self.pending_return_value.take()
        {
            self.sync_slow_path_flags();
            if let Some(dest_reg) = self.pending_return_dest.take() {
                self.set_register(dest_reg, return_value)?;
            }
        }

        if self.current_task.is_some()
            && let Some(signal) = self.pending_task_signal.take()
        {
            self.last_task_signal = Some(signal);
            return Ok(Some(Value::Nil));
        }

        if self.current_task.is_some() {
            if let Some(fuel) = self.quantum_fuel.as_mut() {
                if *fuel == 0 {
                    self.last_task_signal = Some(TaskSignal::QuantumExhausted);
                    return Ok(Some(Value::Nil));
                }
                *fuel -= 1;
            }
        }
        Ok(None)
    }

    #[cfg(feature = "std")]
    #[cold]
    #[inline(never)]
    fn dispatch_instruction_hook_slow(&self, func_idx: usize, ip_before_execution: usize) {
        let func = &self.functions[func_idx];
        let line = func.chunk.lines.get(ip_before_execution).copied().unwrap_or(0);
        self.dispatch_instruction_hook(&func.name, line, ip_before_execution);
    }

    #[cold]
    #[inline(never)]
    fn record_mutation_m1_skip_gas() {
        MUTATION_M1_EXECUTION_COUNT.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    }

    #[cold]
    #[inline(never)]
    fn handle_out_of_gas_slow(&mut self, limit: u64) -> LustError {
        let e = LustError::RuntimeError {
            message: format!(
                "Out of gas (limit: {}, used: {})",
                limit, self.budgets.gas.used
            ),
        };
        if let Some(task) = self.current_task.and_then(|id| self.task_manager.get_mut(id)) {
            task.request_interrupt(crate::effect_abi::InterruptReason::OutOfGas);
            if let Some(err) = task.error.clone() {
                return err;
            }
        }
        e
    }

    // Fiber restore and native frame materialization can hand off densely
    // packed logical registers. Move the live slots back to padded windows
    // before dispatch. Reserving first makes allocation failure atomic.
    fn prepare_dispatch_register_windows(&mut self) -> Result<()> {
        let Some(first) = self.call_stack.first() else {
            return Ok(());
        };
        let mut end = first.base_reg;
        let mut needs_rebase = false;
        for frame in &self.call_stack {
            needs_rebase |= frame.base_reg != end;
            end = register_window_end(end, frame.reg_count)?;
        }
        reserve_register_window(&mut self.vm_registers, &mut self.budgets, end)?;
        if needs_rebase {
            // Restored frames are dense or already padded, so all moves
            // are to the right. Move high slots first to handle overlap.
            for frame in self.call_stack.iter_mut().rev() {
                end -= usize::from(frame.reg_count).max(BYTECODE_REGISTER_WINDOW);
                if end != frame.base_reg {
                    for slot in (0..usize::from(frame.reg_count)).rev() {
                        let value = core::mem::replace(
                            &mut self.vm_registers[frame.base_reg + slot],
                            Value::Nil,
                        );
                        self.vm_registers[end + slot] = value;
                    }
                    frame.base_reg = end;
                }
            }
        }
        Ok(())
    }

    pub(super) fn run_inner<const JIT_ENABLED: bool>(&mut self) -> Result<Value> {
        self.sync_slow_path_flags();
        let mut frame_dirty = true;
        self.prepare_dispatch_register_windows()?;
        let mut cur_frame_idx: usize = 0;
        let mut cur_func_idx: usize = 0;
        let mut cur_chunk_len: usize = 0;
        let mut cur_has_packed: bool = false;
        // Keep the allocations from returned closure frames for the next call.
        let mut closure_upvalue_pool = Vec::new();
        #[cfg(feature = "std")]
        let mut cur_has_osr_entries: bool = false;

        'dispatch: loop {
            debug_assert_eq!(
                self.slow_path_flags & !SLOW_PATH_SKIP_GAS,
                self.compute_slow_path_flags() & !SLOW_PATH_SKIP_GAS
            );
            if self.slow_path_flags & SLOW_PATH_PREAMBLE_MASK != 0 {
                if let Some(return_value) = self.run_dispatch_preamble_slow()? {
                    return Ok(return_value);
                }
                frame_dirty = true;
            }

            if frame_dirty {
                cur_frame_idx = self
                    .call_stack
                    .len()
                    .checked_sub(1)
                    .ok_or_else(|| LustError::RuntimeError {
                        message: "Empty call stack".to_string(),
                    })?;
                cur_func_idx = self.call_stack[cur_frame_idx].function_idx;
                let func = &self.functions[cur_func_idx];
                cur_chunk_len = func.chunk.instructions.len();
                cur_has_packed = func.chunk.packed_code.len() == cur_chunk_len;
                #[cfg(feature = "std")]
                {
                    cur_has_osr_entries = !func.block_entry_pcs.is_empty();
                }
                record_frame_hoist_reload();
                frame_dirty = false;
            }

            debug_assert!(!self.call_stack.is_empty());
            debug_assert_eq!(cur_frame_idx, self.call_stack.len() - 1);
            debug_assert_eq!(cur_func_idx, self.call_stack.last().unwrap().function_idx);
            debug_assert_eq!(
                cur_chunk_len,
                self.functions[cur_func_idx].chunk.instructions.len()
            );
            #[cfg(feature = "std")]
            debug_assert_eq!(
                cur_has_osr_entries,
                !self.functions[cur_func_idx].block_entry_pcs.is_empty()
            );

            #[cfg(feature = "std")]
            let can_enter_fast = !JIT_ENABLED
                && cur_has_packed
                && (self.slow_path_flags & SLOW_PATH_PREAMBLE_MASK) == 0
                && !crate::vm::tracing::has_trace_hook();
            #[cfg(not(feature = "std"))]
            let can_enter_fast = !JIT_ENABLED
                && cur_has_packed
                && (self.slow_path_flags & SLOW_PATH_PREAMBLE_MASK) == 0;

            let (instruction, ip_before_execution) = if can_enter_fast {
                let has_gas_limit = (self.slow_path_flags & SLOW_PATH_GAS_LIMIT) != 0;
                let skip_gas = (self.slow_path_flags & SLOW_PATH_SKIP_GAS) != 0;
                #[cfg(feature = "std")]
                let can_osr =
                    self.backend == crate::vm::VmBackend::Cranelift && cur_has_osr_entries;
                let allow_inloop_calls = self.backend != crate::vm::VmBackend::Cranelift;
                let has_active_finalizers = self.has_active_finalizers;
                let max_stack_depth = self.max_stack_depth;
                let min_inloop_return_depth = self.call_until_depth.unwrap_or(0).saturating_add(1);
                // Safe disjoint-field borrows scoped to intra-frame ops plus in-loop
                // Call / CallMethod / Return over `VM::vm_registers` (NG-W11-4B).
                // Zero `unsafe`: `frame`/`regs` are dropped before pushing/popping
                // `call_stack` or growing `vm_registers`, and `ic_slots` is dropped
                // only when switching to a different `function_idx`.
                // Restore the unexecuted suffix if this dispatch leaves a
                // prepaid block early (slow opcode, GC, or runtime error).
                let mut prepaid = None;
                let exit = (|| -> Result<FrameExit> {
                    let functions = &self.functions;
                    let call_meta = &mut self.call_meta;
                    let call_stack = &mut self.call_stack;
                    let vm_registers = &mut self.vm_registers;
                    let last_frame_pop_cleanup_iters = &mut self.last_frame_pop_cleanup_iters;
                    let budgets = &mut self.budgets;
                    let cycle_collector = &mut self.cycle_collector;
                    let struct_tostring_cache = &mut self.struct_tostring_cache;
                    let struct_metadata = &self.struct_metadata;
                    let unit_enums = &mut self.unit_enums;
                    let globals_vec = &self.globals_vec;
                    let global_name_to_slot = &self.global_name_to_slot;
                    let globals_generation = self.globals_generation;

                    Ok('frame_fast: loop {
                        let func = &functions[cur_func_idx];
                        let instructions = &func.chunk.instructions;
                        let packed_code = &func.chunk.packed_code;
                        let extra_args = &func.chunk.extra_args;
                        if packed_code.len() != instructions.len() {
                            break 'frame_fast FrameExit::FallbackToOuter;
                        }
                        let constants = &func.chunk.constants;
                        let bb_gas_weights = &func.chunk.bb_gas_weights;
                        let bb_gas_ends = &func.chunk.bb_gas_ends;
                        let has_bb_gas = bb_gas_weights.len() == packed_code.len()
                            && bb_gas_ends.len() == packed_code.len();
                        cur_chunk_len = packed_code.len();
                        let mut ic_slots = func.chunk.ic_slots.borrow_mut();
                        if ic_slots.len() < packed_code.len() {
                            ic_slots.resize(packed_code.len(), ChunkIcSlot::Empty);
                        }

                        'same_func: loop {
                            let base_reg = call_stack[cur_frame_idx].base_reg;
                            let reg_end = register_window_end(
                                base_reg,
                                call_stack[cur_frame_idx].reg_count,
                            )?;
                            reserve_register_window(vm_registers, budgets, reg_end)?;

                            let step: FastStep = {
                                let frame: &mut CallFrame = &mut call_stack[cur_frame_idx];
                                let regs: &mut [Value; BYTECODE_REGISTER_WINDOW] =
                                    (&mut vm_registers[base_reg..base_reg + BYTECODE_REGISTER_WINDOW])
                                        .try_into()
                                        .expect("fixed-width register slice");
                                let mut gas_edge = frame.ip;
                                let mut exact_gas_mode = true;

                                'intra: loop {
                        debug_assert_eq!(cur_func_idx, frame.function_idx);
                        debug_assert_eq!(cur_chunk_len, packed_code.len());

                        let ip = frame.ip;
                        let Some(&packed) = packed_code.get(ip) else {
                            break 'frame_fast FrameExit::EndOfChunk;
                        };
                        debug_assert_eq!(
                            packed.to_instruction(extra_args),
                            Ok(instructions[ip]),
                            "packed_code[{}] drifted out of sync with instructions[{}]",
                            ip,
                            ip
                        );
                        let ip_before_execution = ip + 1;
                        frame.ip = ip_before_execution;

                        if skip_gas {
                            Self::record_mutation_m1_skip_gas();
                        } else {
                            if has_bb_gas && ip == gas_edge {
                                // On an interior resume, charge instructions
                                // individually until the next block entry.
                                let end = bb_gas_ends[ip];
                                gas_edge = end;
                                exact_gas_mode = true;
                                prepaid = None;
                                if ip == 0 || bb_gas_ends[ip - 1] == ip {
                                    let weight = u64::from(bb_gas_weights[ip]);
                                    let used = budgets.gas.used;
                                    if !has_gas_limit || budgets.gas.limit.is_none_or(|limit| {
                                        limit.saturating_sub(used) >= weight
                                    }) {
                                        exact_gas_mode = false;
                                        budgets.gas.used = used.saturating_add(weight);
                                        prepaid = Some((cur_frame_idx, cur_func_idx, ip, end, used));
                                    }
                                }
                            }
                            if exact_gas_mode {
                                budgets.gas.used = budgets.gas.used
                                    .saturating_add(u64::from(packed.gas_weight()));
                                if has_gas_limit
                                    && let Some(limit) = budgets.gas.limit
                                    && budgets.gas.used > limit
                                {
                                    break 'frame_fast FrameExit::OutOfGas(limit);
                                }
                            }
                        }

                        let needs_collect = match packed.opcode_u8() {
                            OP_LOAD_NIL => {
                                write_frame_reg_scalar(regs, packed.a(), Value::Nil)
                            }
                            OP_LOAD_BOOL => {
                                write_frame_reg_scalar(regs, packed.a(), Value::Bool(packed.b() != 0))
                            }
                            OP_LOAD_CONST => {
                                let dest = packed.a();
                                let constant = &constants[packed.bx() as usize];
                                let scalar = match constant {
                                    Value::Int(value) => Some(Value::Int(*value)),
                                    Value::Float(value) => Some(Value::Float(*value)),
                                    Value::Bool(value) => Some(Value::Bool(*value)),
                                    _ => None,
                                };
                                if let Some(value) = scalar
                                    && regs[dest as usize].is_plain()
                                {
                                    write_frame_reg_scalar(regs, dest, value)
                                } else {
                                    write_frame_reg_fast(frame, regs, cycle_collector, dest, constant.fast_clone())
                                }
                            }
                            OP_LOAD_GLOBAL => {
                                let dest = packed.a();
                                let name_idx = packed.bx();
                                let val = match &ic_slots[ip] {
                                    ChunkIcSlot::GlobalSlot {
                                        generation,
                                        slot_idx,
                                    } if *generation == globals_generation => {
                                        globals_vec[*slot_idx as usize].fast_clone()
                                    }
                                    _ => {
                                        let name = constants[name_idx as usize]
                                            .as_string()
                                            .ok_or_else(|| LustError::RuntimeError {
                                                message: "Global name must be a string".to_string(),
                                            })?;
                                        if let Some(&slot_idx) = global_name_to_slot.get(name) {
                                            ic_slots[ip] = ChunkIcSlot::GlobalSlot {
                                                generation: globals_generation,
                                                slot_idx,
                                            };
                                            globals_vec[slot_idx as usize].fast_clone()
                                        } else {
                                            break 'frame_fast FrameExit::Slow(
                                                instructions[ip],
                                                ip_before_execution,
                                            );
                                        }
                                    }
                                };
                                if val.is_plain() {
                                    write_frame_reg_scalar(regs, dest, val)
                                } else {
                                    write_frame_reg_fast(frame, regs, cycle_collector, dest, val)
                                }
                            }
                            OP_MOVE => {
                                let value = regs[packed.b() as usize].fast_clone();
                                write_frame_reg_fast(frame, regs, cycle_collector, packed.a(), value)
                            }

                            OP_ADD_INT => {
                                frame_int_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    Ok(Value::Int(a.wrapping_add(b)))
                                })?
                            }
                            OP_ADD_INT_IMM => {
                                let dest = packed.a();
                                let src = packed.b();
                                let imm = packed.c_i8();
                                let src_val = &regs[src as usize];
                                let Value::Int(a) = src_val else {
                                    return Err(LustError::RuntimeError {
                                        message: format!(
                                            "Typed numeric instruction expected int operand, got {:?}",
                                            src_val
                                        ),
                                    });
                                };
                                write_frame_reg_scalar(
                                    regs,
                                    dest,
                                    Value::Int(a.wrapping_add(LustInt::from(imm))),
                                )
                            }
                            OP_SUB_INT => {
                                frame_int_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    Ok(Value::Int(a.wrapping_sub(b)))
                                })?
                            }
                            OP_MUL_INT => {
                                frame_int_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    Ok(Value::Int(a.wrapping_mul(b)))
                                })?
                            }
                            OP_DIV_INT => {
                                frame_int_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    if b == 0 {
                                        Err(LustError::RuntimeError {
                                            message: "Division by zero".to_string(),
                                        })
                                    } else {
                                        Ok(Value::Int(a.wrapping_div(b)))
                                    }
                                })?
                            }
                            OP_MOD_INT => {
                                frame_int_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    if b == 0 {
                                        Err(LustError::RuntimeError {
                                            message: "Modulo by zero".to_string(),
                                        })
                                    } else {
                                        Ok(Value::Int(a.wrapping_rem(b)))
                                    }
                                })?
                            }
                            OP_POW_INT => {
                                frame_int_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    int_pow(a, b)
                                })?
                            }
                            OP_FLOOR_DIV_INT => {
                                frame_int_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    floor_div_i64(a, b).map(Value::Int)
                                })?
                            }
                            OP_NEG_INT => {
                                let src = packed.b();
                                frame_int_binary_op(regs, packed.a(), src, src, |a, _| {
                                    Ok(Value::Int(a.wrapping_neg()))
                                })?
                            }
                            OP_EQ_INT => {
                                frame_int_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    Ok(Value::Bool(a == b))
                                })?
                            }
                            OP_NE_INT => {
                                frame_int_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    Ok(Value::Bool(a != b))
                                })?
                            }
                            OP_LT_INT => {
                                frame_int_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    Ok(Value::Bool(a < b))
                                })?
                            }
                            OP_LE_INT => {
                                frame_int_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    Ok(Value::Bool(a <= b))
                                })?
                            }
                            OP_GT_INT => {
                                frame_int_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    Ok(Value::Bool(a > b))
                                })?
                            }
                            OP_GE_INT => {
                                frame_int_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    Ok(Value::Bool(a >= b))
                                })?
                            }

                            OP_ADD_FLOAT => {
                                frame_float_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    Ok(Value::Float(a + b))
                                })?
                            }
                            OP_SUB_FLOAT => {
                                frame_float_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    Ok(Value::Float(a - b))
                                })?
                            }
                            OP_MUL_FLOAT => {
                                frame_float_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    Ok(Value::Float(a * b))
                                })?
                            }
                            OP_DIV_FLOAT => {
                                frame_float_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    Ok(Value::Float(a / b))
                                })?
                            }
                            OP_MOD_FLOAT => {
                                frame_float_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    if b == 0.0 {
                                        Err(LustError::RuntimeError {
                                            message: "Modulo by zero".to_string(),
                                        })
                                    } else {
                                        Ok(Value::Float(a % b))
                                    }
                                })?
                            }
                            OP_POW_FLOAT => {
                                frame_float_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    Ok(Value::Float(a.powf(b)))
                                })?
                            }
                            OP_FLOOR_DIV_FLOAT => {
                                frame_float_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    if b == 0.0 {
                                        Err(LustError::RuntimeError {
                                            message: "Division by zero".to_string(),
                                        })
                                    } else {
                                        Ok(Value::Float((a / b).floor()))
                                    }
                                })?
                            }
                            OP_NEG_FLOAT => {
                                let src = packed.b();
                                frame_float_binary_op(regs, packed.a(), src, src, |a, _| {
                                    Ok(Value::Float(-a))
                                })?
                            }
                            OP_EQ_FLOAT => {
                                frame_float_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    Ok(Value::Bool(a == b))
                                })?
                            }
                            OP_NE_FLOAT => {
                                frame_float_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    Ok(Value::Bool(a != b))
                                })?
                            }
                            OP_LT_FLOAT => {
                                frame_float_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    Ok(Value::Bool(a < b))
                                })?
                            }
                            OP_LE_FLOAT => {
                                frame_float_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    Ok(Value::Bool(a <= b))
                                })?
                            }
                            OP_GT_FLOAT => {
                                frame_float_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    Ok(Value::Bool(a > b))
                                })?
                            }
                            OP_GE_FLOAT => {
                                frame_float_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    Ok(Value::Bool(a >= b))
                                })?
                            }

                            OP_ADD => {
                                frame_binary_op(frame, regs, cycle_collector, packed.a(), packed.b(), packed.c(), |l, r| {
                                    match (l, r) {
                                        (Value::Int(a), Value::Int(b)) => {
                                            Ok(Value::Int(a.wrapping_add(*b)))
                                        }
                                        (Value::Float(a), Value::Float(b)) => {
                                            Ok(Value::Float(a + b))
                                        }
                                        (Value::Int(a), Value::Float(b)) => {
                                            Ok(Value::Float(float_from_int(*a) + *b))
                                        }
                                        (Value::Float(a), Value::Int(b)) => {
                                            Ok(Value::Float(*a + float_from_int(*b)))
                                        }
                                        (Value::String(a), Value::String(b)) => {
                                            Ok(Value::string(format!("{}{}", a, b)))
                                        }
                                        _ => Err(LustError::RuntimeError {
                                            message: format!("Cannot add {:?} and {:?}", l, r),
                                        }),
                                    }
                                })?
                            }
                            OP_SUB => {
                                frame_binary_op(frame, regs, cycle_collector, packed.a(), packed.b(), packed.c(), |l, r| {
                                    match (l, r) {
                                        (Value::Int(a), Value::Int(b)) => {
                                            Ok(Value::Int(a.wrapping_sub(*b)))
                                        }
                                        (Value::Float(a), Value::Float(b)) => {
                                            Ok(Value::Float(a - b))
                                        }
                                        (Value::Int(a), Value::Float(b)) => {
                                            Ok(Value::Float(float_from_int(*a) - *b))
                                        }
                                        (Value::Float(a), Value::Int(b)) => {
                                            Ok(Value::Float(*a - float_from_int(*b)))
                                        }
                                        _ => Err(LustError::RuntimeError {
                                            message: format!("Cannot subtract {:?} and {:?}", l, r),
                                        }),
                                    }
                                })?
                            }
                            OP_MUL => {
                                frame_binary_op(frame, regs, cycle_collector, packed.a(), packed.b(), packed.c(), |l, r| {
                                    match (l, r) {
                                        (Value::Int(a), Value::Int(b)) => {
                                            Ok(Value::Int(a.wrapping_mul(*b)))
                                        }
                                        (Value::Float(a), Value::Float(b)) => {
                                            Ok(Value::Float(a * b))
                                        }
                                        (Value::Int(a), Value::Float(b)) => {
                                            Ok(Value::Float(float_from_int(*a) * *b))
                                        }
                                        (Value::Float(a), Value::Int(b)) => {
                                            Ok(Value::Float(*a * float_from_int(*b)))
                                        }
                                        _ => Err(LustError::RuntimeError {
                                            message: format!("Cannot multiply {:?} and {:?}", l, r),
                                        }),
                                    }
                                })?
                            }
                            OP_DIV => {
                                frame_binary_op(frame, regs, cycle_collector, packed.a(), packed.b(), packed.c(), |l, r| {
                                    match (l, r) {
                                        (Value::Int(a), Value::Int(b)) => {
                                            if *b == 0 {
                                                Err(LustError::RuntimeError {
                                                    message: "Division by zero".to_string(),
                                                })
                                            } else {
                                                Ok(Value::Int(a.wrapping_div(*b)))
                                            }
                                        }
                                        (Value::Float(a), Value::Float(b)) => {
                                            Ok(Value::Float(*a / *b))
                                        }
                                        (Value::Int(a), Value::Float(b)) => {
                                            Ok(Value::Float(float_from_int(*a) / *b))
                                        }
                                        (Value::Float(a), Value::Int(b)) => {
                                            Ok(Value::Float(*a / float_from_int(*b)))
                                        }
                                        _ => Err(LustError::RuntimeError {
                                            message: format!("Cannot divide {:?} and {:?}", l, r),
                                        }),
                                    }
                                })?
                            }
                            OP_MOD => {
                                frame_binary_op(frame, regs, cycle_collector, packed.a(), packed.b(), packed.c(), |l, r| {
                                    match (l, r) {
                                        (Value::Int(a), Value::Int(b)) => {
                                            if *b == 0 {
                                                Err(LustError::RuntimeError {
                                                    message: "Modulo by zero".to_string(),
                                                })
                                            } else {
                                                Ok(Value::Int(a.wrapping_rem(*b)))
                                            }
                                        }
                                        (Value::Float(a), Value::Float(b)) => {
                                            if *b == 0.0 {
                                                Err(LustError::RuntimeError {
                                                    message: "Modulo by zero".to_string(),
                                                })
                                            } else {
                                                Ok(Value::Float(a % b))
                                            }
                                        }
                                        (Value::Int(a), Value::Float(b)) => {
                                            if *b == 0.0 {
                                                Err(LustError::RuntimeError {
                                                    message: "Modulo by zero".to_string(),
                                                })
                                            } else {
                                                Ok(Value::Float(float_from_int(*a) % *b))
                                            }
                                        }
                                        (Value::Float(a), Value::Int(b)) => {
                                            if *b == 0 {
                                                Err(LustError::RuntimeError {
                                                    message: "Modulo by zero".to_string(),
                                                })
                                            } else {
                                                Ok(Value::Float(*a % float_from_int(*b)))
                                            }
                                        }
                                        _ => Err(LustError::RuntimeError {
                                            message: format!("Cannot modulo {:?} and {:?}", l, r),
                                        }),
                                    }
                                })?
                            }
                            OP_POW => {
                                frame_binary_op(frame, regs, cycle_collector, packed.a(), packed.b(), packed.c(), |l, r| {
                                    match (l, r) {
                                        (Value::Int(a), Value::Int(b)) => int_pow(*a, *b),
                                        (Value::Float(a), Value::Float(b)) => {
                                            Ok(Value::Float(a.powf(*b)))
                                        }
                                        (Value::Int(a), Value::Float(b)) => {
                                            Ok(Value::Float(float_from_int(*a).powf(*b)))
                                        }
                                        (Value::Float(a), Value::Int(b)) => {
                                            Ok(Value::Float(a.powf(float_from_int(*b))))
                                        }
                                        _ => Err(LustError::RuntimeError {
                                            message: format!(
                                                "Cannot raise {:?} to power {:?}",
                                                l, r
                                            ),
                                        }),
                                    }
                                })?
                            }
                            OP_FLOOR_DIV => {
                                frame_binary_op(frame, regs, cycle_collector, packed.a(), packed.b(), packed.c(), |l, r| {
                                    match (l, r) {
                                        (Value::Int(a), Value::Int(b)) => {
                                            floor_div_i64(*a, *b).map(Value::Int)
                                        }
                                        (Value::Float(a), Value::Float(b)) => {
                                            if *b == 0.0 {
                                                Err(LustError::RuntimeError {
                                                    message: "Division by zero".to_string(),
                                                })
                                            } else {
                                                Ok(Value::Float((a / b).floor()))
                                            }
                                        }
                                        (Value::Int(a), Value::Float(b)) => {
                                            if *b == 0.0 {
                                                Err(LustError::RuntimeError {
                                                    message: "Division by zero".to_string(),
                                                })
                                            } else {
                                                Ok(Value::Float((float_from_int(*a) / b).floor()))
                                            }
                                        }
                                        (Value::Float(a), Value::Int(b)) => {
                                            if *b == 0 {
                                                Err(LustError::RuntimeError {
                                                    message: "Division by zero".to_string(),
                                                })
                                            } else {
                                                Ok(Value::Float((a / float_from_int(*b)).floor()))
                                            }
                                        }
                                        _ => Err(LustError::RuntimeError {
                                            message: format!(
                                                "Cannot floor divide {:?} and {:?}",
                                                l, r
                                            ),
                                        }),
                                    }
                                })?
                            }
                            OP_NEG => {
                                let result = match &regs[packed.b() as usize] {
                                    Value::Int(i) => Value::Int(i.wrapping_neg()),
                                    Value::Float(f) => Value::Float(-f),
                                    other => {
                                        return Err(LustError::RuntimeError {
                                            message: format!("Cannot negate {:?}", other),
                                        });
                                    }
                                };
                                write_frame_reg_scalar(regs, packed.a(), result)
                            }
                            OP_EQ => {
                                let eq = regs[packed.b() as usize] == regs[packed.c() as usize];
                                write_frame_reg_scalar(regs, packed.a(), Value::Bool(eq))
                            }
                            OP_NE => {
                                let ne = regs[packed.b() as usize] != regs[packed.c() as usize];
                                write_frame_reg_scalar(regs, packed.a(), Value::Bool(ne))
                            }
                            OP_LT => {
                                frame_comparison_op(
                                    regs,
                                    packed.a(),
                                    packed.b(),
                                    packed.c(),
                                    |l, r| l < r,
                                    |l, r| l < r,
                                )?
                            }
                            OP_LE => {
                                frame_comparison_op(
                                    regs,
                                    packed.a(),
                                    packed.b(),
                                    packed.c(),
                                    |l, r| l <= r,
                                    |l, r| l <= r,
                                )?
                            }
                            OP_GT => {
                                frame_comparison_op(
                                    regs,
                                    packed.a(),
                                    packed.b(),
                                    packed.c(),
                                    |l, r| l > r,
                                    |l, r| l > r,
                                )?
                            }
                            OP_GE => {
                                frame_comparison_op(
                                    regs,
                                    packed.a(),
                                    packed.b(),
                                    packed.c(),
                                    |l, r| l >= r,
                                    |l, r| l >= r,
                                )?
                            }
                            OP_AND => {
                                let res = regs[packed.b() as usize].is_truthy()
                                    && regs[packed.c() as usize].is_truthy();
                                write_frame_reg_scalar(regs, packed.a(), Value::Bool(res))
                            }
                            OP_OR => {
                                let res = regs[packed.b() as usize].is_truthy()
                                    || regs[packed.c() as usize].is_truthy();
                                write_frame_reg_scalar(regs, packed.a(), Value::Bool(res))
                            }
                            OP_NOT => {
                                let res = !regs[packed.b() as usize].is_truthy();
                                write_frame_reg_scalar(regs, packed.a(), Value::Bool(res))
                            }
                            OP_BIT_AND => {
                                frame_int_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    Ok(Value::Int(a & b))
                                })?
                            }
                            OP_BIT_OR => {
                                frame_int_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    Ok(Value::Int(a | b))
                                })?
                            }
                            OP_BIT_XOR => {
                                frame_int_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    Ok(Value::Int(a ^ b))
                                })?
                            }
                            OP_SHL => {
                                frame_int_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    Ok(Value::Int(a.wrapping_shl((b & 63) as u32)))
                                })?
                            }
                            OP_SHR => {
                                frame_int_binary_op(regs, packed.a(), packed.b(), packed.c(), |a, b| {
                                    Ok(Value::Int(a.wrapping_shr((b & 63) as u32)))
                                })?
                            }
                            OP_BIT_NOT => {
                                let res = match &regs[packed.b() as usize] {
                                    Value::Int(a) => Value::Int(!a),
                                    other => {
                                        return Err(LustError::RuntimeError {
                                            message: format!(
                                                "Cannot perform bitwise NOT on {:?}",
                                                other
                                            ),
                                        });
                                    }
                                };
                                write_frame_reg_scalar(regs, packed.a(), res)
                            }

                            OP_JUMP => {
                                let offset = packed.sbx();
                                let target_ip = (frame.ip as isize + offset as isize) as usize;
                                frame.ip = target_ip;
                                gas_edge = frame.ip;
                                prepaid = None;
                                #[cfg(feature = "std")]
                                if offset < 0 && can_osr {
                                    break 'frame_fast FrameExit::TryOsr(target_ip);
                                }
                                false
                            }
                            OP_JUMP_IF => {
                                let cond = packed.a();
                                let offset = packed.sbx();
                                let truthy = match &regs[cond as usize] {
                                    Value::Bool(b) => *b,
                                    other => other.is_truthy(),
                                };
                                if truthy {
                                    frame.ip = (frame.ip as isize + offset as isize) as usize;
                                    gas_edge = frame.ip;
                                    prepaid = None;
                                }
                                false
                            }
                            OP_JUMP_IF_NOT => {
                                let cond = packed.a();
                                let offset = packed.sbx();
                                let truthy = match &regs[cond as usize] {
                                    Value::Bool(b) => *b,
                                    other => other.is_truthy(),
                                };
                                if !truthy {
                                    frame.ip = (frame.ip as isize + offset as isize) as usize;
                                    gas_edge = frame.ip;
                                    prepaid = None;
                                }
                                false
                            }
                            OP_BRANCH_LT_INT => {
                                let lhs = packed.a();
                                let w0 = extra_args[packed.bx() as usize];
                                let rhs = (w0 & 0xFF) as u8;
                                let offset = (w0 >> 16) as u16 as i16;
                                let left = &regs[lhs as usize];
                                let right = &regs[rhs as usize];
                                let (Value::Int(a), Value::Int(b)) = (left, right) else {
                                    return Err(LustError::RuntimeError {
                                        message: format!(
                                            "Typed numeric instruction expected int operands, got {:?} and {:?}",
                                            left, right
                                        ),
                                    });
                                };
                                if !(*a < *b) {
                                    let target_ip = (frame.ip as isize + offset as isize) as usize;
                                    frame.ip = target_ip;
                                    gas_edge = frame.ip;
                                    prepaid = None;
                                    #[cfg(feature = "std")]
                                    if offset < 0 && can_osr {
                                        break 'frame_fast FrameExit::TryOsr(target_ip);
                                    }
                                }
                                false
                            }
                            OP_BRANCH_LE_INT => {
                                let lhs = packed.a();
                                let w0 = extra_args[packed.bx() as usize];
                                let rhs = (w0 & 0xFF) as u8;
                                let offset = (w0 >> 16) as u16 as i16;
                                let left = &regs[lhs as usize];
                                let right = &regs[rhs as usize];
                                let (Value::Int(a), Value::Int(b)) = (left, right) else {
                                    return Err(LustError::RuntimeError {
                                        message: format!(
                                            "Typed numeric instruction expected int operands, got {:?} and {:?}",
                                            left, right
                                        ),
                                    });
                                };
                                if !(*a <= *b) {
                                    let target_ip = (frame.ip as isize + offset as isize) as usize;
                                    frame.ip = target_ip;
                                    gas_edge = frame.ip;
                                    prepaid = None;
                                    #[cfg(feature = "std")]
                                    if offset < 0 && can_osr {
                                        break 'frame_fast FrameExit::TryOsr(target_ip);
                                    }
                                }
                                false
                            }

                            OP_NEW_ARRAY => {
                                let dest = packed.a();
                                let first_elem = packed.b();
                                let count = packed.c();
                                let element_count = count as usize;
                                budgets.charge_value_vec(element_count)?;
                                let mut elements = Vec::with_capacity(element_count);
                                for i in 0..count {
                                    elements.push(
                                        regs[(first_elem + i) as usize].fast_clone(),
                                    );
                                }
                                write_frame_reg_fast(
                                    frame,
                                    regs,
                                    cycle_collector,
                                    dest,
                                    Value::array(elements),
                                )
                            }
                            OP_TUPLE_NEW => {
                                let dest = packed.a();
                                let first_elem = packed.b();
                                let count = packed.c();
                                let mut parts = Vec::with_capacity(count as usize);
                                let mut total_elements: usize = 0;
                                for offset in 0..(count as usize) {
                                    let value =
                                        regs[(first_elem + offset as u8) as usize]
                                            .clone();
                                    total_elements = total_elements.saturating_add(match &value {
                                        Value::Tuple(existing) => existing.len(),
                                        _ => 1,
                                    });
                                    parts.push(value);
                                }
                                budgets.charge_value_vec(total_elements)?;
                                let mut elements = Vec::with_capacity(total_elements);
                                for value in parts {
                                    if let Value::Tuple(existing) = value {
                                        elements.extend(existing.iter().cloned());
                                    } else {
                                        elements.push(value);
                                    }
                                }
                                write_frame_reg_fast(
                                    frame,
                                    regs,
                                    cycle_collector,
                                    dest,
                                    Value::tuple(elements),
                                )
                            }
                            OP_TUPLE_GET => {
                                let dest = packed.a();
                                let tuple_reg = packed.b();
                                let index = packed.c();
                                let value = match &regs[tuple_reg as usize] {
                                    Value::Tuple(values) => {
                                        let idx = index as usize;
                                        if idx >= values.len() {
                                            return Err(LustError::RuntimeError {
                                                message: format!(
                                                    "Tuple index {} out of bounds (len {})",
                                                    idx,
                                                    values.len()
                                                ),
                                            });
                                        }
                                        values[idx].fast_clone()
                                    }
                                    _ => {
                                        return Err(LustError::RuntimeError {
                                            message: "Attempted to destructure non-tuple value"
                                                .to_string(),
                                        });
                                    }
                                };
                                write_frame_reg_fast(frame, regs, cycle_collector, dest, value)
                            }
                            OP_IS_ENUM_VARIANT => {
                                let dest = packed.a();
                                let ex = packed.bx() as usize;
                                let w0 = extra_args[ex];
                                let w1 = extra_args[ex + 1];
                                let value_reg = (w1 & 0xFF) as u8;
                                let enum_name_idx = (w0 & 0xFFFF) as u16;
                                let variant_idx = (w0 >> 16) as u16;
                                let slot = &mut ic_slots[ip];
                                if !matches!(slot, ChunkIcSlot::EnumVariantTag { .. }) {
                                    let enum_name = constants[enum_name_idx as usize]
                                        .as_string()
                                        .ok_or_else(|| LustError::RuntimeError {
                                            message: "Enum name must be a string".to_string(),
                                        })?;
                                    let variant_name = constants[variant_idx as usize]
                                        .as_string()
                                        .ok_or_else(|| LustError::RuntimeError {
                                            message: "Variant name must be a string".to_string(),
                                        })?;
                                    *slot = ChunkIcSlot::EnumVariantTag {
                                        enum_name: Name::from(enum_name),
                                        variant_name: Name::from(variant_name),
                                    };
                                }
                                let ChunkIcSlot::EnumVariantTag { enum_name, variant_name } = slot
                                else {
                                    unreachable!()
                                };
                                let is_variant = matches!(&regs[value_reg as usize],
                                    Value::Enum(object)
                                        if (enum_name.is_empty() || object.enum_name == *enum_name)
                                            && object.variant == *variant_name
                                );
                                write_frame_reg_scalar(regs, dest, Value::Bool(is_variant))
                            }
                            OP_GET_ENUM_VALUE => {
                                let dest = packed.a();
                                let enum_reg = packed.b();
                                let index = packed.c();
                                let value = if let Some((_, _, Some(values))) =
                                    regs[enum_reg as usize].as_enum()
                                {
                                    if (index as usize) < values.len() {
                                        values[index as usize].fast_clone()
                                    } else {
                                        return Err(LustError::RuntimeError {
                                            message: format!(
                                                "Enum value index {} out of bounds (has {} values)",
                                                index,
                                                values.len()
                                            ),
                                        });
                                    }
                                } else {
                                    return Err(LustError::RuntimeError {
                                        message:
                                            "GetEnumValue requires an enum variant with values"
                                                .to_string(),
                                    });
                                };
                                write_frame_reg_fast(frame, regs, cycle_collector, dest, value)
                            }
                            OP_NEW_STRUCT => {
                                let dest = packed.a();
                                let ex = packed.bx() as usize;
                                let w0 = extra_args[ex];
                                let w1 = extra_args[ex + 1];
                                let name_idx = (w0 & 0xFFFF) as u16;
                                let first_field_name_idx = (w0 >> 16) as u16;
                                let first_field = (w1 & 0xFF) as u8;
                                let field_count = ((w1 >> 8) & 0xFF) as u8;
                                let fcount = field_count as usize;
                                budgets.charge_value_vec(fcount)?;
                                let meta_gen = struct_metadata.generation();
                                let ic_hit = matches!(
                                    &ic_slots[ip],
                                    ChunkIcSlot::NewStructTemplate { struct_meta_gen, .. }
                                        if *struct_meta_gen == meta_gen
                                );
                                let struct_value = if !ic_hit {
                                    let struct_name = constants[name_idx as usize]
                                        .as_string()
                                        .ok_or_else(|| LustError::RuntimeError {
                                            message: "Struct name must be a string".to_string(),
                                        })?;
                                    let info = struct_metadata.get(struct_name).ok_or_else(|| {
                                        LustError::RuntimeError {
                                            message: format!("Unknown struct '{}'", struct_name),
                                        }
                                    })?;
                                    let layout = Rc::clone(&info.layout);
                                    let layout_len = layout.field_names().len();
                                    let mut slot_map = Vec::with_capacity(fcount);
                                    let mut filled = vec![false; layout_len];
                                    let mut is_identity = fcount == layout_len;
                                    for i in 0..field_count {
                                        let field_name = constants
                                            [(first_field_name_idx + i as u16) as usize]
                                            .as_string_rc()
                                            .ok_or_else(|| LustError::RuntimeError {
                                                message: "Field name must be a string".to_string(),
                                            })?;
                                        let slot_idx = layout
                                            .index_of_rc(&field_name)
                                            .or_else(|| layout.index_of_str(field_name.as_str()))
                                            .ok_or_else(|| LustError::RuntimeError {
                                                message: format!(
                                                    "Struct '{}' has no field '{}'",
                                                    struct_name, field_name
                                                ),
                                            })?;
                                        if slot_idx != i as usize {
                                            is_identity = false;
                                        }
                                        filled[slot_idx] = true;
                                        slot_map.push(slot_idx);
                                    }
                                    if filled.iter().any(|slot| !*slot) {
                                        let missing: Vec<String> = layout
                                            .field_names()
                                            .iter()
                                            .enumerate()
                                            .filter(|&(idx, _)| !filled[idx])
                                            .map(|(_, name)| (**name).clone())
                                            .collect();
                                        return Err(LustError::RuntimeError {
                                            message: format!(
                                                "Struct '{}' is missing required field(s): {}",
                                                struct_name,
                                                missing.join(", ")
                                            ),
                                        });
                                    }
                                    let interned_name = Name::from(struct_name);
                                    let ordered = if is_identity && !layout.has_weak_fields() {
                                        let mut ordered = Vec::with_capacity(fcount);
                                        for i in 0..field_count {
                                            ordered.push(
                                                regs
                                                    [(first_field as usize) + (i as usize)]
                                                    .fast_clone(),
                                            );
                                        }
                                        ordered
                                    } else {
                                        let mut ordered = vec![Value::Nil; layout_len];
                                        for (i, &slot_idx) in slot_map.iter().enumerate() {
                                            let val = regs
                                                [(first_field as usize) + i]
                                                .fast_clone();
                                            let canonical = if layout.has_weak_fields() {
                                                layout
                                                    .canonicalize_field_value(slot_idx, val)
                                                    .map_err(|message| LustError::RuntimeError {
                                                        message,
                                                    })?
                                            } else {
                                                val
                                            };
                                            ordered[slot_idx] = canonical;
                                        }
                                        ordered
                                    };
                                    if struct_name != "LuaTable" {
                                        ic_slots[ip] = ChunkIcSlot::NewStructTemplate {
                                            struct_meta_gen: meta_gen,
                                            struct_name: interned_name.clone(),
                                            layout: Rc::clone(&layout),
                                            slot_map: slot_map.into_boxed_slice(),
                                            is_identity,
                                        };
                                    }
                                    Value::Struct(StructObject::new(interned_name, layout, ordered))
                                } else {
                                    let ChunkIcSlot::NewStructTemplate {
                                        struct_name,
                                        layout,
                                        slot_map,
                                        is_identity,
                                        ..
                                    } = &ic_slots[ip]
                                    else {
                                        unreachable!()
                                    };
                                    let ordered = if *is_identity && !layout.has_weak_fields() {
                                        let mut ordered = Vec::with_capacity(fcount);
                                        for i in 0..field_count {
                                            ordered.push(
                                                regs
                                                    [(first_field as usize) + (i as usize)]
                                                    .fast_clone(),
                                            );
                                        }
                                        ordered
                                    } else {
                                        let layout_len = layout.field_names().len();
                                        let mut ordered = vec![Value::Nil; layout_len];
                                        for (i, &slot_idx) in slot_map.iter().enumerate() {
                                            let val = regs
                                                [(first_field as usize) + i]
                                                .fast_clone();
                                            let canonical = if layout.has_weak_fields() {
                                                layout
                                                    .canonicalize_field_value(slot_idx, val)
                                                    .map_err(|message| LustError::RuntimeError {
                                                        message,
                                                    })?
                                            } else {
                                                val
                                            };
                                            ordered[slot_idx] = canonical;
                                        }
                                        ordered
                                    };
                                    Value::Struct(StructObject::new(
                                        struct_name.clone(),
                                        Rc::clone(layout),
                                        ordered,
                                    ))
                                };
                                write_frame_reg_fast(frame, regs, cycle_collector, dest, struct_value)
                            }
                            OP_NEW_ENUM_UNIT => {
                                let dest = packed.a();
                                let w0 = extra_args[packed.bx() as usize];
                                let enum_name_idx = (w0 & 0xFFFF) as u16;
                                let variant_idx = (w0 >> 16) as u16;
                                let obj = match &ic_slots[ip] {
                                    ChunkIcSlot::EnumUnit { object } => Rc::clone(object),
                                    _ => {
                                        let enum_str = constants[enum_name_idx as usize]
                                            .as_string()
                                            .ok_or_else(|| LustError::RuntimeError {
                                                message: "Enum name must be a string".to_string(),
                                            })?;
                                        let variant_str = constants[variant_idx as usize]
                                            .as_string()
                                            .ok_or_else(|| LustError::RuntimeError {
                                                message: "Variant name must be a string"
                                                    .to_string(),
                                            })?;
                                        #[cfg(feature = "std")]
                                        let obj = if enum_str == "Option" && variant_str == "None"
                                        {
                                            option_none_singleton()
                                        } else {
                                            let e_name = Name::from(enum_str);
                                            let v_name = Name::from(variant_str);
                                            let key = (
                                                e_name.inner_ptr() as usize,
                                                v_name.inner_ptr() as usize,
                                            );
                                            Rc::clone(unit_enums.entry(key).or_insert_with(|| {
                                                EnumObject::new(e_name, v_name, None)
                                            }))
                                        };
                                        #[cfg(not(feature = "std"))]
                                        let obj = {
                                            let e_name = Name::from(enum_str);
                                            let v_name = Name::from(variant_str);
                                            let key = (
                                                e_name.inner_ptr() as usize,
                                                v_name.inner_ptr() as usize,
                                            );
                                            Rc::clone(unit_enums.entry(key).or_insert_with(|| {
                                                EnumObject::new(e_name, v_name, None)
                                            }))
                                        };
                                        ic_slots[ip] = ChunkIcSlot::EnumUnit {
                                            object: Rc::clone(&obj),
                                        };
                                        obj
                                    }
                                };
                                write_frame_reg_fast(frame, regs, cycle_collector, dest, Value::Enum(obj))
                            }
                            OP_NEW_ENUM_VARIANT => {
                                let dest = packed.a();
                                let ex = packed.bx() as usize;
                                let w0 = extra_args[ex];
                                let w1 = extra_args[ex + 1];
                                let enum_name_idx = (w0 & 0xFFFF) as u16;
                                let variant_idx = (w0 >> 16) as u16;
                                let first_value = (w1 & 0xFF) as u8;
                                let value_count = ((w1 >> 8) & 0xFF) as u8;
                                budgets.charge_value_vec(value_count as usize)?;
                                let (enum_name, variant_name) = match &ic_slots[ip] {
                                    ChunkIcSlot::EnumVariantNames {
                                        enum_name,
                                        variant_name,
                                    } => (enum_name.clone(), variant_name.clone()),
                                    _ => {
                                        let enum_str = constants[enum_name_idx as usize]
                                            .as_string()
                                            .ok_or_else(|| LustError::RuntimeError {
                                                message: "Enum name must be a string".to_string(),
                                            })?;
                                        let variant_str = constants[variant_idx as usize]
                                            .as_string()
                                            .ok_or_else(|| LustError::RuntimeError {
                                                message: "Variant name must be a string"
                                                    .to_string(),
                                            })?;
                                        let enum_name = Name::from(enum_str);
                                        let variant_name = Name::from(variant_str);
                                        ic_slots[ip] = ChunkIcSlot::EnumVariantNames {
                                            enum_name: enum_name.clone(),
                                            variant_name: variant_name.clone(),
                                        };
                                        (enum_name, variant_name)
                                    }
                                };
                                let mut values = Vec::with_capacity(value_count as usize);
                                for i in 0..value_count {
                                    values.push(
                                        regs[(first_value as usize) + (i as usize)]
                                            .fast_clone(),
                                    );
                                }
                                let val = Value::Enum(EnumObject::new(
                                    enum_name,
                                    variant_name,
                                    Some(values),
                                ));
                                write_frame_reg_fast(frame, regs, cycle_collector, dest, val)
                            }
                            OP_GET_FIELD_SLOT => {
                                let dest = packed.a();
                                let obj = packed.b();
                                let slot_idx = packed.c();
                                let val = match &regs[obj as usize] {
                                    Value::Struct(s) if s.name != "LuaTable" => {
                                        let idx = slot_idx as usize;
                                        let borrowed = s.fields.borrow();
                                        match borrowed.get(idx) {
                                            Some(v) => {
                                                let cloned = v.fast_clone();
                                                drop(borrowed);
                                                if s.layout.has_weak_fields() {
                                                    s.layout.materialize_field_value(idx, cloned)
                                                } else {
                                                    cloned
                                                }
                                            }
                                            None => Value::Nil,
                                        }
                                    }
                                    _ => {
                                        break 'frame_fast FrameExit::Slow(
                                            instructions[ip],
                                            ip_before_execution,
                                        );
                                    }
                                };
                                if val.is_plain() {
                                    write_frame_reg_scalar(regs, dest, val)
                                } else {
                                    write_frame_reg_fast(frame, regs, cycle_collector, dest, val)
                                }
                            }
                            OP_SET_FIELD_SLOT => {
                                let obj_reg = packed.a();
                                let slot_idx = packed.b();
                                let value_reg = packed.c();
                                let is_fast_struct = matches!(
                                    &regs[obj_reg as usize],
                                    Value::Struct(s) if s.name != "LuaTable"
                                );
                                if !is_fast_struct {
                                    break 'frame_fast FrameExit::Slow(
                                        instructions[ip],
                                        ip_before_execution,
                                    );
                                }
                                let value = regs[value_reg as usize].fast_clone();
                                if !crate::vm::cycle::CycleCollector::is_leaf(&value) {
                                    cycle_collector.record_mutation();
                                }
                                let Value::Struct(s) = &regs[obj_reg as usize] else {
                                    unreachable!()
                                };
                                let idx = slot_idx as usize;
                                {
                                    let mut borrowed = s.fields.borrow_mut();
                                    if idx < borrowed.len() {
                                        let canonical = if s.layout.has_weak_fields() {
                                            s.layout
                                                .canonicalize_field_value(idx, value)
                                                .map_err(|message| LustError::RuntimeError {
                                                    message,
                                                })?
                                        } else {
                                            value
                                        };
                                        borrowed[idx] = canonical;
                                    } else {
                                        return Err(LustError::RuntimeError {
                                            message: format!(
                                                "Struct '{}' field index {} out of bounds (len {})",
                                                s.name,
                                                idx,
                                                borrowed.len()
                                            ),
                                        });
                                    }
                                }
                                if !struct_tostring_cache.is_empty() {
                                    let key = Rc::as_ptr(s) as usize;
                                    struct_tostring_cache.remove(&key);
                                }
                                false
                            }
                            OP_GET_FIELD_ADD_SET_INT => {
                                let obj_reg = packed.a();
                                let slot_idx = packed.b();
                                let addend_reg = packed.c();
                                let (Value::Struct(s), Value::Int(delta)) =
                                    (&regs[obj_reg as usize], &regs[addend_reg as usize])
                                else {
                                    break 'frame_fast FrameExit::Slow(
                                        instructions[ip],
                                        ip_before_execution,
                                    );
                                };
                                if s.name == "LuaTable" || s.layout.has_weak_fields() {
                                    break 'frame_fast FrameExit::Slow(
                                        instructions[ip],
                                        ip_before_execution,
                                    );
                                }
                                let idx = slot_idx as usize;
                                let updated = {
                                    let mut borrowed = s.fields.borrow_mut();
                                    let Some(slot_val) = borrowed.get_mut(idx) else {
                                        return Err(LustError::RuntimeError {
                                            message: format!(
                                                "Struct '{}' field index {} out of bounds (len {})",
                                                s.name,
                                                idx,
                                                borrowed.len()
                                            ),
                                        });
                                    };
                                    if let Value::Int(cur) = *slot_val {
                                        *slot_val = Value::Int(cur.wrapping_add(*delta));
                                        true
                                    } else {
                                        false
                                    }
                                };
                                if !updated {
                                    break 'frame_fast FrameExit::Slow(
                                        instructions[ip],
                                        ip_before_execution,
                                    );
                                }
                                if !struct_tostring_cache.is_empty() {
                                    let key = Rc::as_ptr(s) as usize;
                                    struct_tostring_cache.remove(&key);
                                }
                                false
                            }
                            OP_GET_FIELD | OP_GET_FIELD_COMPACT => {
                                let dest = packed.a();
                                let (obj, field_idx) = packed.field_operands(extra_args);
                                let val = match &regs[obj as usize] {
                                    Value::Struct(s) if s.name != "LuaTable" => {
                                        let resolved_slot = match ic_slots[ip].probe_field(&s.layout) {
                                            Some(slot) => Some(slot),
                                            None => {
                                                let field_name = constants[field_idx as usize]
                                                    .as_string_rc()
                                                    .ok_or_else(|| LustError::RuntimeError {
                                                        message: "Field name must be a string"
                                                            .to_string(),
                                                    })?;
                                                let idx_opt = s
                                                    .layout
                                                    .index_of_rc(&field_name)
                                                    .or_else(|| {
                                                        s.layout.index_of_str(field_name.as_str())
                                                    })
                                                    .or_else(|| {
                                                        s.layout
                                                            .field_names()
                                                            .iter()
                                                            .position(|n| {
                                                                n.as_str() == field_name.as_str()
                                                            })
                                                    });
                                                if let Some(slot_idx) = idx_opt {
                                                    ic_slots[ip].insert_field(&s.layout, slot_idx);
                                                }
                                                idx_opt
                                            }
                                        };
                                        match resolved_slot {
                                            Some(idx) => {
                                                let borrowed = s.fields.borrow();
                                                match borrowed.get(idx) {
                                                    Some(v) => {
                                                        let cloned = v.fast_clone();
                                                        drop(borrowed);
                                                        if s.layout.has_weak_fields() {
                                                            s.layout
                                                                .materialize_field_value(idx, cloned)
                                                        } else {
                                                            cloned
                                                        }
                                                    }
                                                    None => Value::Nil,
                                                }
                                            }
                                            None => Value::Nil,
                                        }
                                    }
                                    Value::Map(map) => {
                                        let field_name = constants[field_idx as usize]
                                            .as_string_rc()
                                            .ok_or_else(|| LustError::RuntimeError {
                                                message: "Field name must be a string".to_string(),
                                            })?;
                                        let key = ValueKey::from(field_name);
                                        map.borrow().get(&key).cloned().unwrap_or(Value::Nil)
                                    }
                                    _ => {
                                        break 'frame_fast FrameExit::Slow(
                                            instructions[ip],
                                            ip_before_execution,
                                        );
                                    }
                                };
                                if val.is_plain() {
                                    write_frame_reg_scalar(regs, dest, val)
                                } else {
                                    write_frame_reg_fast(frame, regs, cycle_collector, dest, val)
                                }
                            }
                            OP_SET_FIELD | OP_SET_FIELD_COMPACT => {
                                let obj_reg = packed.a();
                                let (value_reg, field_idx) = packed.field_operands(extra_args);
                                let value = &regs[value_reg as usize];
                                match &regs[obj_reg as usize] {
                                    Value::Struct(s) if s.name != "LuaTable" => {
                                        let idx = match ic_slots[ip].probe_field(&s.layout) {
                                            Some(slot) => slot,
                                            None => {
                                                let field_name = constants[field_idx as usize]
                                                    .as_string_rc()
                                                    .ok_or_else(|| LustError::RuntimeError {
                                                        message: "Field name must be a string"
                                                            .to_string(),
                                                    })?;
                                                let slot_idx = s
                                                    .layout
                                                    .index_of_rc(&field_name)
                                                    .or_else(|| {
                                                        s.layout.index_of_str(field_name.as_str())
                                                    })
                                                    .ok_or_else(|| LustError::RuntimeError {
                                                        message: format!(
                                                            "Struct '{}' has no field '{}'",
                                                            s.layout.name(),
                                                            field_name.as_str()
                                                        ),
                                                    })?;
                                                ic_slots[ip].insert_field(&s.layout, slot_idx);
                                                slot_idx
                                            }
                                        };
                                        if !crate::vm::cycle::CycleCollector::is_leaf(value) {
                                            cycle_collector.record_mutation();
                                        }
                                        {
                                            let mut borrowed = s.fields.borrow_mut();
                                            if idx < borrowed.len() {
                                                let canonical = if s.layout.has_weak_fields() {
                                                    s.layout
                                                        .canonicalize_field_value(idx, value.fast_clone())
                                                        .map_err(|message| {
                                                            LustError::RuntimeError { message }
                                                        })?
                                                } else {
                                                    value.fast_clone()
                                                };
                                                borrowed[idx] = canonical;
                                            } else {
                                                return Err(LustError::RuntimeError {
                                                    message: format!(
                                                        "Struct '{}' field index {} out of bounds (len {})",
                                                        s.name,
                                                        idx,
                                                        borrowed.len()
                                                    ),
                                                });
                                            }
                                        }
                                        if !struct_tostring_cache.is_empty() {
                                            let key = Rc::as_ptr(s) as usize;
                                            struct_tostring_cache.remove(&key);
                                        }
                                    }
                                    Value::Map(map) => {
                                        let field_name = constants[field_idx as usize]
                                            .as_string_rc()
                                            .ok_or_else(|| LustError::RuntimeError {
                                                message: "Field name must be a string".to_string(),
                                            })?;
                                        let key = ValueKey::from(field_name);
                                        if budgets.mem_budget_enabled()
                                            && !map.borrow().contains_key(&key)
                                        {
                                            budgets.charge_map_entry_estimate()?;
                                        }
                                        map.borrow_mut().insert(key, value.fast_clone());
                                    }
                                    _ => {
                                        break 'frame_fast FrameExit::Slow(
                                            instructions[ip],
                                            ip_before_execution,
                                        );
                                    }
                                }
                                false
                            }
                            OP_CONCAT => {
                                let dest = packed.a();
                                let lhs = packed.b();
                                let rhs = packed.c();
                                let both_strings = matches!(
                                    (
                                        &regs[lhs as usize],
                                        &regs[rhs as usize]
                                    ),
                                    (Value::String(_), Value::String(_))
                                );
                                if !both_strings {
                                    break 'frame_fast FrameExit::Slow(
                                        instructions[ip],
                                        ip_before_execution,
                                    );
                                }
                                let mut in_place_done = false;
                                if dest == lhs {
                                    let right_rc = match &regs[rhs as usize] {
                                        Value::String(s) => Rc::clone(s),
                                        _ => unreachable!(),
                                    };
                                    let right_len = right_rc.len();
                                    if let Value::String(ref mut rc) =
                                        regs[dest as usize]
                                    {
                                        if let Some(s_mut) = Rc::get_mut(rc) {
                                            budgets.charge_mem_bytes(right_len)?;
                                            s_mut.push_str(right_rc.as_str());
                                            in_place_done = true;
                                        }
                                    }
                                }
                                if !in_place_done {
                                    let (left_rc, right_rc) = match (
                                        &regs[lhs as usize],
                                        &regs[rhs as usize],
                                    ) {
                                        (Value::String(l), Value::String(r)) => {
                                            (Rc::clone(l), Rc::clone(r))
                                        }
                                        _ => unreachable!(),
                                    };
                                    let cap = left_rc.len().saturating_add(right_rc.len());
                                    budgets.charge_mem_bytes(cap)?;
                                    let mut combined = String::with_capacity(if dest == lhs {
                                        cap.max(left_rc.len() * 2)
                                    } else {
                                        cap
                                    });
                                    combined.push_str(left_rc.as_str());
                                    combined.push_str(right_rc.as_str());
                                    write_frame_reg_fast(
                                        frame,
                                        regs,
                                        cycle_collector,
                                        dest,
                                        Value::string(combined),
                                    )
                                } else {
                                    false
                                }
                            }
                            OP_GET_INDEX => {
                                let dest = packed.a();
                                let array_reg = packed.b();
                                let index_reg = packed.c();
                                let Value::Array(arr) = &regs[array_reg as usize] else {
                                    break 'frame_fast FrameExit::Slow(
                                        instructions[ip],
                                        ip_before_execution,
                                    );
                                };
                                let idx = regs[index_reg as usize]
                                    .as_int()
                                    .ok_or_else(|| LustError::RuntimeError {
                                        message: "Array index must be an integer".to_string(),
                                    })?;
                                let result = if let Some(val) =
                                    crate::bytecode::value::get_specialized_cached_element(arr, idx)
                                {
                                    val
                                } else if idx >= 0
                                    && let Some(val) = arr.get_value_fast(idx as usize)
                                {
                                    val
                                } else {
                                    return Err(LustError::RuntimeError {
                                        message: format!(
                                            "Array index {} out of bounds (length: {})",
                                            idx,
                                            arr.len_fast()
                                        ),
                                    });
                                };
                                if result.is_plain() {
                                    write_frame_reg_scalar(regs, dest, result)
                                } else {
                                    write_frame_reg_fast(frame, regs, cycle_collector, dest, result)
                                }
                            }
                            OP_ARRAY_GET_UNWRAP => {
                                let dest = packed.a();
                                let array_reg = packed.b();
                                let index_reg = packed.c();
                                let Value::Array(arr) = &regs[array_reg as usize] else {
                                    let ty = regs[array_reg as usize].type_of();
                                    return Err(LustError::RuntimeError {
                                        message: format!("array.get expects an array, got {:?}", ty),
                                    });
                                };
                                let idx = regs[index_reg as usize]
                                    .as_int()
                                    .ok_or_else(|| LustError::RuntimeError {
                                        message: "Array index must be an integer".to_string(),
                                    })?;
                                let result = if let Some(val) =
                                    crate::bytecode::value::get_specialized_cached_element(arr, idx)
                                {
                                    val
                                } else if idx >= 0
                                    && let Some(val) = arr.get_value_fast(idx as usize)
                                {
                                    val
                                } else {
                                    return Err(LustError::RuntimeError {
                                        message: format!(
                                            "Array index {} out of bounds (length: {})",
                                            idx,
                                            arr.len_fast()
                                        ),
                                    });
                                };
                                if result.is_plain() {
                                    write_frame_reg_scalar(regs, dest, result)
                                } else {
                                    write_frame_reg_fast(frame, regs, cycle_collector, dest, result)
                                }
                            }
                            OP_ARRAY_PUSH => {
                                let array_reg = packed.a();
                                let value_reg = packed.b();
                                let Value::Array(arr) = &regs[array_reg as usize] else {
                                    let ty = regs[array_reg as usize].type_of();
                                    return Err(LustError::RuntimeError {
                                        message: format!(
                                            "array.push expects an array, got {:?}",
                                            ty
                                        ),
                                    });
                                };
                                let value = regs[value_reg as usize].fast_clone();
                                charge_array_push_budget(budgets, arr, &value)?;
                                if !crate::vm::cycle::CycleCollector::is_leaf(&value) {
                                    cycle_collector.record_mutation();
                                }
                                crate::bytecode::value::sync_specialized_array(arr);
                                arr.push_value_fast(value);
                                crate::bytecode::value::invalidate_specialized_array_cache(
                                    arr.as_ptr() as usize,
                                );
                                false
                            }
                            OP_SET_INDEX => {
                                let collection_reg = packed.a();
                                let index_reg = packed.b();
                                let value_reg = packed.c();
                                let Value::Array(arr) = &regs[collection_reg as usize]
                                else {
                                    break 'frame_fast FrameExit::Slow(
                                        instructions[ip],
                                        ip_before_execution,
                                    );
                                };
                                let idx = regs[index_reg as usize]
                                    .as_int()
                                    .ok_or_else(|| LustError::RuntimeError {
                                        message: "Array index must be an integer".to_string(),
                                    })?;
                                let value = regs[value_reg as usize].fast_clone();
                                if let Some(res) =
                                    crate::bytecode::value::set_specialized_cached_element(
                                        arr, idx, &value,
                                    )
                                {
                                    res.map_err(|message| LustError::RuntimeError { message })?;
                                } else {
                                    if idx < 0 {
                                        return Err(LustError::RuntimeError {
                                            message: format!(
                                                "Array index {} out of bounds (length: {})",
                                                idx,
                                                arr.len_fast()
                                            ),
                                        });
                                    }
                                    if !crate::vm::cycle::CycleCollector::is_leaf(&value) {
                                        cycle_collector.record_mutation();
                                    }
                                    if let Err(len) = arr.set_value_fast(idx as usize, value) {
                                        return Err(LustError::RuntimeError {
                                            message: format!(
                                                "Array index {} out of bounds (length: {})",
                                                idx, len
                                            ),
                                        });
                                    }
                                    crate::bytecode::value::invalidate_specialized_array_cache(
                                        arr.as_ptr() as usize,
                                    );
                                }
                                false
                            }
                            OP_ARRAY_INDEX_OK => {
                                let val_dest = packed.a();
                                let w0 = extra_args[packed.bx() as usize];
                                let cond_dest = (w0 & 0xFF) as u8;
                                let array_reg = ((w0 >> 8) & 0xFF) as u8;
                                let idx_reg = ((w0 >> 16) & 0xFF) as u8;
                                let Value::Array(arr) = &regs[array_reg as usize] else {
                                    let ty = regs[array_reg as usize].type_of();
                                    return Err(LustError::RuntimeError {
                                        message: format!("Cannot index {:?}", ty),
                                    });
                                };
                                let idx = regs[idx_reg as usize]
                                    .as_int()
                                    .ok_or_else(|| LustError::RuntimeError {
                                        message: "Array index must be an integer".to_string(),
                                    })?;
                                let found = if let Some(val) =
                                    crate::bytecode::value::get_specialized_cached_element(arr, idx)
                                {
                                    Some(val)
                                } else if let Some(len) =
                                    crate::bytecode::value::get_specialized_cached_len(arr)
                                {
                                    if idx >= 0 && (idx as usize) < len {
                                        arr.get_value_fast(idx as usize)
                                    } else {
                                        None
                                    }
                                } else if idx >= 0 {
                                    arr.get_value_fast(idx as usize)
                                } else {
                                    None
                                };
                                if let Some(val) = found {
                                    let c1 = if val.is_plain() {
                                        write_frame_reg_scalar(regs, val_dest, val)
                                    } else {
                                        write_frame_reg_fast(frame, regs, cycle_collector, val_dest, val)
                                    };
                                    let c2 = write_frame_reg_scalar(
                                        regs,
                                        cond_dest,
                                        Value::Bool(true),
                                    );
                                    c1 || c2
                                } else {
                                    write_frame_reg_scalar(regs, cond_dest, Value::Bool(false))
                                }
                            }
                            OP_MAP_GET | OP_MAP_GET_UNWRAP => {
                                let value = map_read(
                                    &regs[packed.b() as usize],
                                    &regs[packed.c() as usize],
                                    packed.opcode_u8() == OP_MAP_GET_UNWRAP,
                                )?;
                                if value.is_plain() {
                                    write_frame_reg_scalar(regs, packed.a(), value)
                                } else {
                                    write_frame_reg_fast(frame, regs, cycle_collector, packed.a(), value)
                                }
                            }
                            OP_MAP_SET => {
                                let stores_container = map_write(
                                    &regs[packed.a() as usize],
                                    &regs[packed.b() as usize],
                                    &regs[packed.c() as usize],
                                    budgets,
                                    cycle_collector,
                                )?;
                                stores_container && cycle_collector.should_collect()
                            }
                            OP_ARRAY_LEN => {
                                let dest = packed.a();
                                let array_reg = packed.b();
                                let len = match &regs[array_reg as usize] {
                                    Value::Array(arr) => {
                                        crate::bytecode::value::get_specialized_cached_len(arr)
                                            .unwrap_or_else(|| arr.len_fast())
                                    }
                                    other => {
                                        return Err(LustError::RuntimeError {
                                            message: format!(
                                                "ArrayLen requires array, got {:?}",
                                                other.type_of()
                                            ),
                                        });
                                    }
                                };
                                write_frame_reg_scalar(
                                    regs,
                                    dest,
                                    Value::Int(int_from_usize(len)),
                                )
                            }
                            OP_LOAD_UPVALUE => {
                                let dest = packed.a();
                                let upvalue_idx = packed.b();
                                if (upvalue_idx as usize) < frame.upvalues.len() {
                                    let value = frame.upvalues[upvalue_idx as usize].clone();
                                    write_frame_reg_fast(frame, regs, cycle_collector, dest, value)
                                } else {
                                    return Err(LustError::RuntimeError {
                                        message: format!(
                                            "Upvalue index {} out of bounds (have {} upvalues)",
                                            upvalue_idx,
                                            frame.upvalues.len()
                                        ),
                                    });
                                }
                            }
                            OP_STORE_UPVALUE => {
                                let upvalue_idx = packed.a();
                                let src = packed.b();
                                let value = regs[src as usize].clone();
                                let idx = upvalue_idx as usize;
                                if idx < frame.upvalues.len() {
                                    frame.upvalues[idx] = value.clone();
                                    if idx < frame.shared_upvalues.len() {
                                        frame.shared_upvalues[idx].set(value);
                                    }
                                    false
                                } else {
                                    return Err(LustError::RuntimeError {
                                        message: format!(
                                            "Upvalue index {} out of bounds (have {} upvalues)",
                                            upvalue_idx,
                                            frame.upvalues.len()
                                        ),
                                    });
                                }
                            }

                            OP_CALL => {
                                let callee_reg = packed.a();
                                if allow_inloop_calls {
                                    let target = match &regs[callee_reg as usize] {
                                        Value::Function(idx) => Some((*idx, None)),
                                        Value::Closure(closure) => {
                                            Some((closure.function_idx, Some(callee_reg)))
                                        }
                                        _ => None,
                                    };
                                    if let Some((target_func_idx, closure_reg)) = target {
                                        let w0 = extra_args[packed.bx() as usize];
                                        break 'intra FastStep::InLoopCall {
                                            target_func_idx,
                                            closure_reg,
                                            first_arg: (w0 & 0xFF) as u8,
                                            arg_count: ((w0 >> 8) & 0xFF) as u8,
                                            dest: ((w0 >> 16) & 0xFF) as u8,
                                            ip_before_execution,
                                        };
                                    }
                                }
                                break 'frame_fast FrameExit::Slow(
                                    instructions[ip],
                                    ip_before_execution,
                                );
                            }
                            OP_CALL_METHOD => {
                                let obj_reg = packed.a();
                                if allow_inloop_calls
                                    && let Value::Struct(receiver) = &regs[obj_reg as usize]
                                    && receiver.name != "LuaTable"
                                    && receiver.name == *receiver.layout.name_interned()
                                {
                                    let ex = packed.bx() as usize;
                                    let w0 = extra_args[ex];
                                    let w1 = extra_args[ex + 1];
                                    let method_const = (w0 & 0xFFFF) as u16;
                                    let first_arg = ((w0 >> 16) & 0xFF) as u8;
                                    let arg_count = (w0 >> 24) as u8;
                                    let dest = (w1 & 0xFF) as u8;
                                    let target_opt = ic_slots[ip].probe_method(&receiver.layout)
                                        .or_else(|| {
                                            let method_name = constants[method_const as usize].as_string()?;
                                            if method_name == "try_unwrap" {
                                                return None;
                                            }
                                            let mangled_name = format!("{}:{}", receiver.name, method_name);
                                            let target = functions.iter().position(|f| f.name == mangled_name)?;
                                            ic_slots[ip].insert_method(&receiver.layout, target);
                                            Some(target)
                                        });
                                    if let Some(target_func_idx) = target_opt {
                                        break 'intra FastStep::InLoopCallMethod {
                                            target_func_idx,
                                            obj_reg,
                                            first_arg,
                                            arg_count,
                                            dest,
                                            ip_before_execution,
                                        };
                                    }
                                }
                                break 'frame_fast FrameExit::Slow(
                                    instructions[ip],
                                    ip_before_execution,
                                );
                            }
                            OP_RETURN => {
                                let ret_reg = packed.a();
                                if allow_inloop_calls && !has_active_finalizers {
                                    break 'intra FastStep::InLoopReturn {
                                        ret_reg,
                                        ip_before_execution,
                                    };
                                }
                                break 'frame_fast FrameExit::Slow(
                                    instructions[ip],
                                    ip_before_execution,
                                );
                            }

                            _ => {
                                break 'frame_fast FrameExit::Slow(
                                    instructions[ip],
                                    ip_before_execution,
                                );
                            }
                        };

                        if needs_collect {
                            break 'frame_fast FrameExit::CollectCycles;
                        }
                                }
                            };

                            // An in-loop call or return ends the current gas block.
                            prepaid = None;
                            match step {
                                FastStep::InLoopCall {
                                    target_func_idx,
                                    closure_reg,
                                    first_arg,
                                    arg_count,
                                    dest,
                                    ip_before_execution,
                                } => {
                                    let Some(target_fn) = functions.get(target_func_idx) else {
                                        break 'frame_fast FrameExit::Slow(
                                            instructions[ip_before_execution - 1],
                                            ip_before_execution,
                                        );
                                    };
                                    let callee_meta = &call_meta[target_func_idx];
                                    if callee_meta.lua_function
                                        || (closure_reg.is_none() && !target_fn.upvalues.is_empty())
                                        || arg_count != target_fn.param_count
                                    {
                                        break 'frame_fast FrameExit::Slow(
                                            instructions[ip_before_execution - 1],
                                            ip_before_execution,
                                        );
                                    }
                                    let caller_frame = &call_stack[cur_frame_idx];
                                    let caller_base = caller_frame.base_reg;
                                    let caller_reg_count = caller_frame.reg_count as usize;
                                    if arg_count > 0
                                        && (first_arg as usize) + (arg_count as usize)
                                            > caller_reg_count
                                    {
                                        break 'frame_fast FrameExit::Slow(
                                            instructions[ip_before_execution - 1],
                                            ip_before_execution,
                                        );
                                    }
                                    // Non-Lua bytecode calls use the same shallow
                                    // argument checks here as in the general dispatcher.
                                    if !callee_meta.trusted_signature
                                        && callee_meta.params.iter().enumerate().any(|(i, kind)| {
                                            !kind.matches(
                                                &vm_registers[caller_base + first_arg as usize + i],
                                            )
                                        })
                                    {
                                        break 'frame_fast FrameExit::Slow(
                                            instructions[ip_before_execution - 1],
                                            ip_before_execution,
                                        );
                                    }
                                    if call_stack.len() >= max_stack_depth {
                                        return Err(LustError::RuntimeError {
                                            message: "Stack overflow".to_string(),
                                        });
                                    }
                                    let new_base =
                                        register_window_end(caller_base, caller_frame.reg_count)?;
                                    let target_reg_count: u16 = if target_fn.register_count > 0 {
                                        (target_fn.register_count as u16)
                                            .max(target_fn.param_count as u16)
                                    } else {
                                        256
                                    };
                                    let needed = register_window_end(new_base, target_reg_count)?;
                                    reserve_register_window(vm_registers, budgets, needed)?;
                                    let mut callee_non_plain_mask: u64 = 0;
                                    for i in 0..arg_count as usize {
                                        let src_idx =
                                            caller_base + first_arg.wrapping_add(i as u8) as usize;
                                        let val = vm_registers[src_idx].fast_clone();
                                        if !val.is_plain() {
                                            callee_non_plain_mask |= 1u64 << i.min(63);
                                        }
                                        vm_registers[new_base + i] = val;
                                    }
                                    if let Some(meta) = call_meta.get_mut(target_func_idx) {
                                        meta.active_depth = meta.active_depth.saturating_add(1);
                                    }
                                    let (upvalues, shared_upvalues) = if let Some(reg) = closure_reg
                                        && let Value::Closure(closure) =
                                            &vm_registers[caller_base + reg as usize]
                                        && !closure.upvalues.is_empty()
                                    {
                                        let (mut values, mut shared): (Vec<Value>, Vec<Upvalue>) =
                                            closure_upvalue_pool.pop().unwrap_or_default();
                                        #[cfg(all(feature = "std", debug_assertions))]
                                        CLOSURE_UPVALUE_BUFFER_GROWTHS.with(|growths| {
                                            let count = closure.upvalues.len();
                                            growths.set(
                                                growths.get()
                                                    + u64::from(values.capacity() < count)
                                                    + u64::from(shared.capacity() < count),
                                            );
                                        });
                                        values.extend(closure.upvalues.iter().map(Upvalue::get));
                                        shared.extend_from_slice(&closure.upvalues);
                                        (values, shared)
                                    } else {
                                        (Vec::new(), Vec::new())
                                    };
                                    call_stack.push(CallFrame {
                                        function_idx: target_func_idx,
                                        ip: 0,
                                        base_reg: new_base,
                                        reg_count: target_reg_count,
                                        return_dest: Some(dest),
                                        non_plain_mask: callee_non_plain_mask,
                                        upvalues,
                                        shared_upvalues,
                                        saved_registers: None,
                                    });
                                    cur_frame_idx = call_stack.len() - 1;
                                    if target_func_idx == cur_func_idx {
                                        continue 'same_func;
                                    }
                                    cur_func_idx = target_func_idx;
                                    continue 'frame_fast;
                                }
                                FastStep::InLoopCallMethod {
                                    target_func_idx,
                                    obj_reg,
                                    first_arg,
                                    arg_count,
                                    dest,
                                    ip_before_execution,
                                } => {
                                    let Some(target_fn) = functions.get(target_func_idx) else {
                                        break 'frame_fast FrameExit::Slow(
                                            instructions[ip_before_execution - 1],
                                            ip_before_execution,
                                        );
                                    };
                                    let is_trusted = call_meta
                                        .get(target_func_idx)
                                        .is_some_and(|m| m.trusted_signature && !m.lua_function);
                                    if !is_trusted
                                        || !target_fn.upvalues.is_empty()
                                        || (1usize + arg_count as usize)
                                            != target_fn.param_count as usize
                                    {
                                        break 'frame_fast FrameExit::Slow(
                                            instructions[ip_before_execution - 1],
                                            ip_before_execution,
                                        );
                                    }
                                    let caller_frame = &call_stack[cur_frame_idx];
                                    let caller_base = caller_frame.base_reg;
                                    let caller_reg_count = caller_frame.reg_count as usize;
                                    if (obj_reg as usize) >= caller_reg_count
                                        || (arg_count > 0
                                            && (first_arg as usize) + (arg_count as usize)
                                                > caller_reg_count)
                                    {
                                        break 'frame_fast FrameExit::Slow(
                                            instructions[ip_before_execution - 1],
                                            ip_before_execution,
                                        );
                                    }
                                    if call_stack.len() >= max_stack_depth {
                                        return Err(LustError::RuntimeError {
                                            message: "Stack overflow".to_string(),
                                        });
                                    }
                                    let new_base =
                                        register_window_end(caller_base, caller_frame.reg_count)?;
                                    let target_reg_count: u16 = if target_fn.register_count > 0 {
                                        (target_fn.register_count as u16)
                                            .max(target_fn.param_count as u16)
                                    } else {
                                        256
                                    };
                                    let needed = register_window_end(new_base, target_reg_count)?;
                                    reserve_register_window(vm_registers, budgets, needed)?;
                                    let mut callee_non_plain_mask: u64 = 0;
                                    let recv_val =
                                        vm_registers[caller_base + obj_reg as usize].fast_clone();
                                    if !recv_val.is_plain() {
                                        callee_non_plain_mask |= 1u64;
                                    }
                                    vm_registers[new_base] = recv_val;
                                    for i in 0..arg_count as usize {
                                        let src_idx =
                                            caller_base + first_arg.wrapping_add(i as u8) as usize;
                                        let val = vm_registers[src_idx].fast_clone();
                                        if !val.is_plain() {
                                            callee_non_plain_mask |= 1u64 << (i + 1).min(63);
                                        }
                                        vm_registers[new_base + 1 + i] = val;
                                    }
                                    if let Some(meta) = call_meta.get_mut(target_func_idx) {
                                        meta.active_depth = meta.active_depth.saturating_add(1);
                                    }
                                    call_stack.push(CallFrame {
                                        function_idx: target_func_idx,
                                        ip: 0,
                                        base_reg: new_base,
                                        reg_count: target_reg_count,
                                        return_dest: Some(dest),
                                        non_plain_mask: callee_non_plain_mask,
                                        upvalues: Vec::new(),
                                        shared_upvalues: Vec::new(),
                                        saved_registers: None,
                                    });
                                    cur_frame_idx = call_stack.len() - 1;
                                    if target_func_idx == cur_func_idx {
                                        continue 'same_func;
                                    }
                                    cur_func_idx = target_func_idx;
                                    continue 'frame_fast;
                                }
                                FastStep::InLoopReturn {
                                    ret_reg,
                                    ip_before_execution,
                                } => {
                                    let callee_base = call_stack[cur_frame_idx].base_reg;
                                    let ret = if ret_reg == 255 {
                                        &Value::Nil
                                    } else {
                                        &vm_registers[callee_base + ret_reg as usize]
                                    };
                                    let meta = &call_meta[cur_func_idx];
                                    if call_stack.len() <= min_inloop_return_depth
                                        || (!meta.trusted_signature
                                            && !meta.return_kind.matches(ret)
                                            && !(meta.lua_multi_return
                                                && matches!(ret, Value::Nil)))
                                    {
                                        break 'frame_fast FrameExit::Slow(
                                            instructions[ip_before_execution - 1],
                                            ip_before_execution,
                                        );
                                    }
                                    let ret_val = ret.fast_clone();
                                    let mut callee_frame = call_stack.pop().unwrap();
                                    if let Some(meta) =
                                        call_meta.get_mut(callee_frame.function_idx)
                                    {
                                        meta.active_depth = meta.active_depth.saturating_sub(1);
                                    }
                                    let start = callee_frame.base_reg;
                                    let end = (start + callee_frame.reg_count as usize)
                                        .min(vm_registers.len());
                                    let cleared = if start < end {
                                        CallFrame::clear_non_plain_slots(
                                            &mut vm_registers[start..end],
                                            callee_frame.non_plain_mask,
                                        )
                                    } else {
                                        0
                                    };
                                    *last_frame_pop_cleanup_iters = cleared;

                                    if callee_frame.shared_upvalues.capacity() != 0 {
                                        callee_frame.upvalues.clear();
                                        callee_frame.shared_upvalues.clear();
                                        closure_upvalue_pool.push((
                                            callee_frame.upvalues,
                                            callee_frame.shared_upvalues,
                                        ));
                                    }
                                    cur_frame_idx = call_stack.len() - 1;
                                    let caller_func_idx = call_stack[cur_frame_idx].function_idx;
                                    let mut needs_collect = false;
                                    if let Some(dest) = callee_frame.return_dest {
                                        let caller_frame = &mut call_stack[cur_frame_idx];
                                        let dest_idx = dest as usize;
                                        if !ret_val.is_plain() {
                                            caller_frame.mark_non_plain(dest_idx);
                                            if cycle_collector.is_enabled() {
                                                cycle_collector.register_value(&ret_val);
                                                needs_collect = cycle_collector.should_collect();
                                            }
                                        }
                                        vm_registers[caller_frame.base_reg + dest_idx] = ret_val;
                                    }
                                    if needs_collect {
                                        cur_func_idx = caller_func_idx;
                                        break 'frame_fast FrameExit::CollectCycles;
                                    }
                                    if caller_func_idx == cur_func_idx {
                                        continue 'same_func;
                                    }
                                    cur_func_idx = caller_func_idx;
                                    continue 'frame_fast;
                                }
                            }
                        }
                    })
                })();

                if let Some((frame_idx, func_idx, start, end, used)) = prepaid {
                    let pc = self.call_stack[frame_idx].ip;
                    if pc < end {
                        let executed = self.functions[func_idx].chunk.packed_code[start..pc]
                            .iter()
                            .map(|inst| u64::from(inst.gas_weight()))
                            .sum::<u64>();
                        self.budgets.gas.used = used.saturating_add(executed);
                    }
                }

                match exit? {
                    FrameExit::EndOfChunk => {
                        let _ = self.pop_call_frame();
                        if self.call_stack.is_empty() {
                            return Ok(Value::Nil);
                        }
                        frame_dirty = true;
                        continue 'dispatch;
                    }
                    FrameExit::CollectCycles => {
                        self.maybe_collect_cycles();
                        frame_dirty = true;
                        continue 'dispatch;
                    }
                    FrameExit::OutOfGas(limit) => {
                        return Err(self.handle_out_of_gas_slow(limit));
                    }
                    FrameExit::FallbackToOuter => {
                        frame_dirty = true;
                        continue 'dispatch;
                    }
                    #[cfg(feature = "std")]
                    FrameExit::TryOsr(target_ip) => {
                        frame_dirty = true;
                        if let Some(outcome) =
                            self.try_enter_osr_at_backedge(cur_func_idx, target_ip)?
                        {
                            match outcome {
                                crate::vm::OsrBackedgeOutcome::Return(Some(val)) => {
                                    return Ok(val);
                                }
                                crate::vm::OsrBackedgeOutcome::Return(None)
                                | crate::vm::OsrBackedgeOutcome::DeoptimizedResume => {
                                    continue 'dispatch;
                                }
                            }
                        }
                        continue 'dispatch;
                    }
                    FrameExit::Slow(instruction, ip_before_execution) => {
                        frame_dirty = true;
                        (instruction, ip_before_execution)
                    }
                }
            } else {
                let func_idx = cur_func_idx;
                let (instruction, ip_before_execution) = {
                    let frame = &mut self.call_stack[cur_frame_idx];
                    let ip = frame.ip;
                    if ip >= cur_chunk_len {
                        let _ = self.pop_call_frame();
                        if self.call_stack.is_empty() {
                            return Ok(Value::Nil);
                        }
                        frame_dirty = true;
                        continue 'dispatch;
                    }
                    let instruction = self.functions[cur_func_idx].chunk.instructions[ip];
                    frame.ip = ip + 1;
                    (instruction, ip + 1)
                };

                if JIT_ENABLED {
                    let (should_check_jit, loop_start_ip) =
                        if let Instruction::Jump(offset) = instruction {
                            if offset < 0 {
                                let jump_target =
                                    (ip_before_execution as isize + offset as isize) as usize;
                                (true, jump_target)
                            } else {
                                (false, 0)
                            }
                        } else {
                            (false, 0)
                        };
                    if should_check_jit && self.quantum_fuel.is_none() {
                        let count = self.jit.profiler.record_backedge(func_idx, loop_start_ip);
                        let backedge_ip = ip_before_execution.saturating_sub(1);
                        if self
                            .trace_recorder
                            .as_ref()
                            .is_some_and(|recorder| !recorder.is_recording())
                        {
                            self.abandon_trace_recording();
                        }
                        // No trace runs while one is being recorded, except the
                        // root trace of a nested loop the recording is skipping:
                        // that loop's iterations are not part of the trace, and
                        // interpreting them (a million, for a big array) would be
                        // what the recording costs.
                        let skipped_nested = self.trace_recorder.as_ref().is_some_and(|recorder| {
                            recorder.skipped_loop() == Some((func_idx, loop_start_ip))
                        });
                        let root_trace_id = if self.trace_recorder.is_none() || skipped_nested {
                            self.jit
                                .root_traces
                                .get(&(func_idx, loop_start_ip))
                                .copied()
                        } else {
                            None
                        };
                        if let Some(trace_id) = root_trace_id {
                            let (base_reg, reg_count) = {
                                let frame = self.call_stack.last().unwrap();
                                (frame.base_reg, frame.reg_count as usize)
                            };
                            let registers_ptr = self.vm_registers[base_reg..base_reg + reg_count].as_mut_ptr();
                            if let Some(trace) = self.jit.trace_handle(trace_id) {
                                frame_dirty = true;
                                self.jit.record_native_entry();
                                crate::jit::log(|| {
                                    format!(
                                        "▶️  JIT: Executing trace #{} at func {} ip {}",
                                        trace_id.0, func_idx, loop_start_ip
                                    )
                                });

                                let trace_gas_cost = {
                                    let cost = trace.trace.ops.len()
                                        + trace.trace.preamble.len()
                                        + trace.trace.postamble.len();
                                    core::cmp::max(1, cost) as u64
                                };
                                self.budgets.charge_gas(trace_gas_cost)?;
                                self.publish_gas_to_cells();
                                self.pending_jit_error = None;

                                // Capture RSP before and after to detect stack leaks (x86_64 only)
                                #[cfg(target_arch = "x86_64")]
                                let rsp_before: usize;
                                #[cfg(target_arch = "x86_64")]
                                unsafe {
                                    core::arch::asm!("mov {}, rsp", out(reg) rsp_before)
                                };

                                let vm_ptr = self as *mut VM;
                                let _jit_guard = self.enter_jit_scope();
                                let result = trace.execute(registers_ptr, vm_ptr, ptr::null());
                                drop(_jit_guard);
                                drop(trace);
                                if let Some(frame) = self.call_stack.last_mut() {
                                    frame.non_plain_mask = u64::MAX;
                                }
                                self.sync_gas_from_cells();

                                #[cfg(target_arch = "x86_64")]
                                let rsp_after: usize;
                                #[cfg(target_arch = "x86_64")]
                                unsafe {
                                    core::arch::asm!("mov {}, rsp", out(reg) rsp_after)
                                };

                                #[cfg(target_arch = "x86_64")]
                                let rsp_diff = rsp_after as isize - rsp_before as isize;
                                crate::jit::log(|| {
                                    #[cfg(target_arch = "x86_64")]
                                    return format!(
                                        "🎯 JIT: Trace #{} execution result: {} (RSP before: {:x}, after: {:x}, diff: {})",
                                        trace_id.0, result, rsp_before, rsp_after, rsp_diff
                                    );
                                    #[cfg(not(target_arch = "x86_64"))]
                                    return format!(
                                        "🎯 JIT: Trace #{} execution result: {}",
                                        trace_id.0, result
                                    );
                                });

                                if result == 0 {
                                    if self.current_task.is_some()
                                        && self.pending_task_signal.is_some()
                                    {
                                        if let Some(frame) = self.call_stack.last_mut() {
                                            frame.ip = ip_before_execution.saturating_sub(1);
                                        }
                                        continue 'dispatch;
                                    }

                                    if let Some(frame) = self.call_stack.last_mut() {
                                        frame.ip = loop_start_ip;
                                    }

                                    continue 'dispatch;
                                } else if result > 0 {
                                    self.jit.record_guard_exit();
                                    let guard_index = (result - 1) as usize;
                                    let bailout_ip = self
                                        .jit
                                        .get_trace(trace_id)
                                        .and_then(|trace| trace.guards.get(guard_index))
                                        .map(|guard| guard.bailout_ip);

                                    crate::jit::log(|| {
                                        let kind = self
                                            .jit
                                            .get_trace(trace_id)
                                            .and_then(|trace| trace.guards.get(guard_index))
                                            .map(|guard| format!("{:?}", guard.kind));
                                        format!(
                                            "↩️  JIT: guard #{guard_index} exit {kind:?} → ip {bailout_ip:?}"
                                        )
                                    });
                                    // A nested loop's trace may have bailed out
                                    // somewhere other than the guard's own ip.
                                    let bailout_ip =
                                        self.nested_loop_exit_ip.take().or(bailout_ip);
                                    if let Some(bailout_ip) = bailout_ip
                                        && let Some(frame) = self.call_stack.last_mut()
                                    {
                                        frame.ip = bailout_ip;
                                    }
                                    if let Some(error) = self.pending_jit_error.take() {
                                        return Err(error);
                                    }

                                    self.handle_guard_failure(trace_id, guard_index, func_idx)?;
                                    let reusable_exit = self
                                        .jit
                                        .get_trace(trace_id)
                                        .and_then(|trace| trace.guards.get(guard_index))
                                        .is_some_and(|guard| {
                                            matches!(
                                                guard.kind,
                                                crate::jit::GuardKind::Truthy { .. }
                                                    | crate::jit::GuardKind::Falsy { .. }
                                                    | crate::jit::GuardKind::NestedLoop { .. }
                                                    | crate::jit::GuardKind::Plain { .. }
                                            )
                                        });
                                    if !reusable_exit {
                                        self.jit.evict_root_trace(func_idx, loop_start_ip);
                                    }
                                    continue 'dispatch;
                                } else {
                                    self.jit.record_execution_failure();
                                    if let Some(error) = self.pending_jit_error.take() {
                                        return Err(error);
                                    }
                                    let resume_ip = if result <= -2 {
                                        self.jit
                                            .get_trace(trace_id)
                                            .and_then(|trace| {
                                                trace.fail_sites.get((-result - 2) as usize)
                                            })
                                            .copied()
                                    } else {
                                        None
                                    };
                                    crate::jit::log(|| {
                                        format!(
                                            "⚠️  JIT: Trace execution failed (result {result}), resuming at ip {:?}",
                                            resume_ip
                                        )
                                    });
                                    if let Some(frame) = self.call_stack.last_mut() {
                                        frame.ip = resume_ip.unwrap_or(loop_start_ip);
                                    }

                                    self.jit.evict_root_trace(func_idx, loop_start_ip);
                                    continue 'dispatch;
                                }
                            }
                        } else {
                            if let Some(recorder) = &mut self.trace_recorder {
                                let recording_this_loop = recorder.trace.function_idx == func_idx
                                    && recorder.trace.start_ip == loop_start_ip;
                                if recorder.is_recording() && recording_this_loop {
                                    crate::jit::log(|| {
                                        format!(
                                            "📝 JIT: Trace recording complete - {} ops recorded",
                                            recorder.trace.ops.len()
                                        )
                                    });
                                    let mut recorder = self.trace_recorder.take().unwrap();
                                    recorder.complete_nested_skip_at(backedge_ip);
                                    if !recorder.is_recording() {
                                        self.jit.recording_aborted(func_idx, loop_start_ip);
                                        frame_dirty = true;
                                        continue 'dispatch;
                                    }
                                    let mut trace = recorder.finish();
                                    let mut optimizer = TraceOptimizer::new();
                                    let hoisted_constants = optimizer.optimize(&mut trace);
                                    crate::jit::log(|| {
                                        "⚙️  JIT: Compiling root trace...".to_string()
                                    });
                                    let trace_id = self.jit.alloc_trace_id();
                                    match JitCompiler::new()
                                        .with_gas_checks(self.gas_checked())
                                        .compile_trace(
                                            &trace,
                                            trace_id,
                                            hoisted_constants.clone(),
                                        )
                                    {
                                        Ok(compiled_trace) => {
                                            crate::jit::log(|| {
                                                format!(
                                                    "✅ JIT: Trace #{} compiled successfully!",
                                                    trace_id.0
                                                )
                                            });
                                            crate::jit::log(|| {
                                                "🚀 JIT: Future iterations will use native code!"
                                                    .to_string()
                                            });
                                            self.jit.store_root_trace(
                                                func_idx,
                                                loop_start_ip,
                                                compiled_trace,
                                            );
                                        }

                                        Err(e) => {
                                            crate::jit::log(|| {
                                                format!("❌ JIT: Trace compilation failed: {}", e)
                                            });
                                            self.jit.recording_aborted(func_idx, loop_start_ip);
                                        }
                                    }
                                }
                            }

                            if self.trace_recorder.is_none()
                                && !self
                                    .jit
                                    .root_traces
                                    .contains_key(&(func_idx, loop_start_ip))
                            {
                                let loop_in_hierarchy = match self
                                    .jit
                                    .profiler
                                    .get_cached_loop_hierarchy(func_idx, loop_start_ip)
                                {
                                    Some(cached) => cached,
                                    None => {
                                        let calculated = self.is_loop_in_hierarchy(
                                            func_idx,
                                            loop_start_ip,
                                            backedge_ip,
                                        );
                                        self.jit.profiler.set_cached_loop_hierarchy(
                                            func_idx,
                                            loop_start_ip,
                                            calculated,
                                        );
                                        calculated
                                    }
                                };
                                if self.jit.should_record_root(
                                    func_idx,
                                    loop_start_ip,
                                    count,
                                    crate::jit::HOT_THRESHOLD + u32::from(loop_in_hierarchy),
                                ) {
                                    crate::jit::log(|| {
                                        format!(
                                            "🔥 JIT: Hot loop detected at func {} ip {} - starting trace recording!",
                                            func_idx, loop_start_ip
                                        )
                                    });
                                    let mut recorder = TraceRecorder::new(
                                        func_idx,
                                        loop_start_ip,
                                        MAX_TRACE_LENGTH,
                                    );
                                    recorder.set_root_frame_index(
                                        self.call_stack.len().saturating_sub(1),
                                    );
                                    recorder.set_intrinsics(&self.jit.intrinsics);
                                    // Specialize loop-invariant values at trace entry
                                    if !self
                                        .jit
                                        .no_specialize_sites
                                        .contains(&(func_idx, loop_start_ip))
                                    {
                                        let frame = self.call_stack.last().unwrap();
                                        let regs = &self.vm_registers[frame.base_reg..frame.base_reg + frame.reg_count as usize];
                                        let func = &self.functions[func_idx];
                                        recorder.specialize_trace_inputs(regs, func);
                                    }
                                    self.trace_recorder = Some(recorder);
                                    self.jit.recording_started();
                                    self.skip_next_trace_record = true;
                                }
                            }
                        }
                    }
                }

                #[cfg(feature = "std")]
                if crate::vm::tracing::has_trace_hook() {
                    self.dispatch_instruction_hook_slow(func_idx, ip_before_execution);
                    frame_dirty = true;
                }

                if (self.slow_path_flags & SLOW_PATH_SKIP_GAS) != 0 {
                    Self::record_mutation_m1_skip_gas();
                } else {
                    self.budgets.gas.used = self
                        .budgets
                        .gas
                        .used
                        .saturating_add(u64::from(instruction.gas_weight()));
                    if self.slow_path_flags & SLOW_PATH_GAS_LIMIT != 0
                        && let Some(limit) = self.budgets.gas.limit
                        && self.budgets.gas.used > limit
                    {
                        return Err(self.handle_out_of_gas_slow(limit));
                    }
                }

                // Structurally guarantee every outer-dispatch instruction reloads
                // cached frame metadata on the next iteration (zero cost on `'frame_fast`).
                frame_dirty = true;
                (instruction, ip_before_execution)
            };
            let executing_frame_index = cur_frame_idx;
            let func_idx = cur_func_idx;
            match instruction {
                Instruction::LoadNil(dest) => {
                    self.set_register(dest, Value::Nil)?;
                }

                Instruction::LoadBool(dest, value) => {
                    self.set_register(dest, Value::Bool(value))?;
                }

                Instruction::LoadConst(dest, const_idx) => {
                    let constant = {
                        let func = &self.functions[func_idx];
                        func.chunk.constants[const_idx as usize].fast_clone()
                    };
                    self.set_register(dest, constant)?;
                }

                Instruction::LoadGlobal(dest, name_idx) => {
                    let func = &self.functions[func_idx];
                    let name = func.chunk.constants[name_idx as usize]
                        .as_string()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "Global name must be a string".to_string(),
                        })?;
                    if let Some(value) = self.globals.get(name) {
                        self.set_register(dest, value.clone())?;
                    } else if let Some(value) = self.natives.get(name) {
                        self.set_register(dest, value.clone())?;
                    } else {
                        if let Some((_, value)) =
                            self.globals.iter().find(|(key, _)| key.as_str() == name)
                        {
                            self.set_register(dest, value.clone())?;
                        } else if let Some((_, value)) =
                            self.natives.iter().find(|(key, _)| key.as_str() == name)
                        {
                            self.set_register(dest, value.clone())?;
                        } else if name.contains('.')
                            && let Some(pos) = self.functions.iter().position(|f| f.name == name)
                        {
                            self.set_register(dest, Value::Function(pos))?;
                        } else if Self::is_effect_intrinsic(name) {
                            self.set_register(
                                dest,
                                Value::String(alloc::rc::Rc::new(alloc::format!(
                                    "__lust_effect_intrinsic:{}",
                                    name
                                ))),
                            )?;
                        } else {
                            return Err(LustError::RuntimeError {
                                message: format!("Undefined global: {}", name),
                            });
                        }
                    }
                }

                Instruction::StoreGlobal(name_idx, src) => {
                    let name = {
                        let func = &self.functions[func_idx];
                        func.chunk.constants[name_idx as usize]
                            .as_string()
                            .ok_or_else(|| LustError::RuntimeError {
                                message: "Global name must be a string".to_string(),
                            })?
                            .to_string()
                    };
                    let value = self.get_register(src)?.clone();
                    self.sync_global_insert(name, value);
                }

                Instruction::Move(dest, src) => {
                    let value = self.get_register(src)?.clone();
                    self.set_register(dest, value)?;
                }

                Instruction::AddInt(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Int(a.wrapping_add(b))))?;
                }
                Instruction::AddIntImm(dest, src, imm) => {
                    let src_val = self.get_register(src)?;
                    let Value::Int(a) = src_val else {
                        return Err(LustError::RuntimeError {
                            message: format!(
                                "Typed numeric instruction expected int operand, got {:?}",
                                src_val
                            ),
                        });
                    };
                    let result = Value::Int(a.wrapping_add(LustInt::from(imm)));
                    self.set_register(dest, result)?;
                }
                Instruction::SubInt(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Int(a.wrapping_sub(b))))?;
                }
                Instruction::MulInt(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Int(a.wrapping_mul(b))))?;
                }
                Instruction::DivInt(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| {
                        if b == 0 {
                            Err(LustError::RuntimeError {
                                message: "Division by zero".to_string(),
                            })
                        } else {
                            Ok(Value::Int(a.wrapping_div(b)))
                        }
                    })?;
                }
                Instruction::ModInt(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| {
                        if b == 0 {
                            Err(LustError::RuntimeError {
                                message: "Modulo by zero".to_string(),
                            })
                        } else {
                            Ok(Value::Int(a.wrapping_rem(b)))
                        }
                    })?;
                }
                Instruction::PowInt(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| int_pow(a, b))?;
                }
                Instruction::FloorDivInt(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| floor_div_i64(a, b).map(Value::Int))?;
                }
                Instruction::NegInt(dest, src) => {
                    self.int_binary_op(dest, src, src, |a, _| Ok(Value::Int(a.wrapping_neg())))?;
                }
                Instruction::EqInt(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Bool(a == b)))?;
                }
                Instruction::NeInt(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Bool(a != b)))?;
                }
                Instruction::LtInt(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Bool(a < b)))?;
                }
                Instruction::LeInt(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Bool(a <= b)))?;
                }
                Instruction::GtInt(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Bool(a > b)))?;
                }
                Instruction::GeInt(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Bool(a >= b)))?;
                }
                Instruction::AddFloat(dest, lhs, rhs) => {
                    self.float_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Float(a + b)))?;
                }
                Instruction::SubFloat(dest, lhs, rhs) => {
                    self.float_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Float(a - b)))?;
                }
                Instruction::MulFloat(dest, lhs, rhs) => {
                    self.float_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Float(a * b)))?;
                }
                Instruction::DivFloat(dest, lhs, rhs) => {
                    self.float_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Float(a / b)))?;
                }
                Instruction::ModFloat(dest, lhs, rhs) => {
                    self.float_binary_op(dest, lhs, rhs, |a, b| {
                        if b == 0.0 {
                            Err(LustError::RuntimeError {
                                message: "Modulo by zero".to_string(),
                            })
                        } else {
                            Ok(Value::Float(a % b))
                        }
                    })?;
                }
                Instruction::PowFloat(dest, lhs, rhs) => {
                    self.float_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Float(a.powf(b))))?;
                }
                Instruction::FloorDivFloat(dest, lhs, rhs) => {
                    self.float_binary_op(dest, lhs, rhs, |a, b| {
                        if b == 0.0 {
                            Err(LustError::RuntimeError {
                                message: "Division by zero".to_string(),
                            })
                        } else {
                            Ok(Value::Float((a / b).floor()))
                        }
                    })?;
                }
                Instruction::NegFloat(dest, src) => {
                    self.float_binary_op(dest, src, src, |a, _| Ok(Value::Float(-a)))?;
                }
                Instruction::EqFloat(dest, lhs, rhs) => {
                    self.float_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Bool(a == b)))?;
                }
                Instruction::NeFloat(dest, lhs, rhs) => {
                    self.float_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Bool(a != b)))?;
                }
                Instruction::LtFloat(dest, lhs, rhs) => {
                    self.float_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Bool(a < b)))?;
                }
                Instruction::LeFloat(dest, lhs, rhs) => {
                    self.float_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Bool(a <= b)))?;
                }
                Instruction::GtFloat(dest, lhs, rhs) => {
                    self.float_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Bool(a > b)))?;
                }
                Instruction::GeFloat(dest, lhs, rhs) => {
                    self.float_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Bool(a >= b)))?;
                }

                Instruction::Add(dest, lhs, rhs) => {
                    self.binary_op(dest, lhs, rhs, |l, r| match (l, r) {
                        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a.wrapping_add(*b))),
                        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
                        (Value::Int(a), Value::Float(b)) => {
                            Ok(Value::Float(float_from_int(*a) + *b))
                        }
                        (Value::Float(a), Value::Int(b)) => {
                            Ok(Value::Float(*a + float_from_int(*b)))
                        }
                        (Value::String(a), Value::String(b)) => {
                            Ok(Value::string(format!("{}{}", a, b)))
                        }
                        _ => Err(LustError::RuntimeError {
                            message: format!("Cannot add {:?} and {:?}", l, r),
                        }),
                    })?;
                }

                Instruction::Sub(dest, lhs, rhs) => {
                    self.binary_op(dest, lhs, rhs, |l, r| match (l, r) {
                        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a.wrapping_sub(*b))),
                        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
                        (Value::Int(a), Value::Float(b)) => {
                            Ok(Value::Float(float_from_int(*a) - *b))
                        }
                        (Value::Float(a), Value::Int(b)) => {
                            Ok(Value::Float(*a - float_from_int(*b)))
                        }
                        _ => Err(LustError::RuntimeError {
                            message: format!("Cannot subtract {:?} and {:?}", l, r),
                        }),
                    })?;
                }

                Instruction::Mul(dest, lhs, rhs) => {
                    self.binary_op(dest, lhs, rhs, |l, r| match (l, r) {
                        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a.wrapping_mul(*b))),
                        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
                        (Value::Int(a), Value::Float(b)) => {
                            Ok(Value::Float(float_from_int(*a) * *b))
                        }
                        (Value::Float(a), Value::Int(b)) => {
                            Ok(Value::Float(*a * float_from_int(*b)))
                        }
                        _ => Err(LustError::RuntimeError {
                            message: format!("Cannot multiply {:?} and {:?}", l, r),
                        }),
                    })?;
                }

                Instruction::Div(dest, lhs, rhs) => {
                    self.binary_op(dest, lhs, rhs, |l, r| match (l, r) {
                        (Value::Int(a), Value::Int(b)) => {
                            if *b == 0 {
                                Err(LustError::RuntimeError {
                                    message: "Division by zero".to_string(),
                                })
                            } else {
                                Ok(Value::Int(a.wrapping_div(*b)))
                            }
                        }

                        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(*a / *b)),
                        (Value::Int(a), Value::Float(b)) => {
                            Ok(Value::Float(float_from_int(*a) / *b))
                        }
                        (Value::Float(a), Value::Int(b)) => {
                            Ok(Value::Float(*a / float_from_int(*b)))
                        }
                        _ => Err(LustError::RuntimeError {
                            message: format!("Cannot divide {:?} and {:?}", l, r),
                        }),
                    })?;
                }

                Instruction::Mod(dest, lhs, rhs) => {
                    self.binary_op(dest, lhs, rhs, |l, r| match (l, r) {
                        (Value::Int(a), Value::Int(b)) => {
                            if *b == 0 {
                                Err(LustError::RuntimeError {
                                    message: "Modulo by zero".to_string(),
                                })
                            } else {
                                Ok(Value::Int(a.wrapping_rem(*b)))
                            }
                        }

                        (Value::Float(a), Value::Float(b)) => {
                            if *b == 0.0 {
                                Err(LustError::RuntimeError {
                                    message: "Modulo by zero".to_string(),
                                })
                            } else {
                                Ok(Value::Float(a % b))
                            }
                        }

                        (Value::Int(a), Value::Float(b)) => {
                            if *b == 0.0 {
                                Err(LustError::RuntimeError {
                                    message: "Modulo by zero".to_string(),
                                })
                            } else {
                                Ok(Value::Float(float_from_int(*a) % *b))
                            }
                        }

                        (Value::Float(a), Value::Int(b)) => {
                            if *b == 0 {
                                Err(LustError::RuntimeError {
                                    message: "Modulo by zero".to_string(),
                                })
                            } else {
                                Ok(Value::Float(*a % float_from_int(*b)))
                            }
                        }

                        _ => Err(LustError::RuntimeError {
                            message: format!("Cannot modulo {:?} and {:?}", l, r),
                        }),
                    })?;
                }

                Instruction::Pow(dest, lhs, rhs) => {
                    self.binary_op(dest, lhs, rhs, |l, r| match (l, r) {
                        (Value::Int(a), Value::Int(b)) => int_pow(*a, *b),
                        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a.powf(*b))),
                        (Value::Int(a), Value::Float(b)) => {
                            Ok(Value::Float(float_from_int(*a).powf(*b)))
                        }
                        (Value::Float(a), Value::Int(b)) => {
                            Ok(Value::Float(a.powf(float_from_int(*b))))
                        }
                        _ => Err(LustError::RuntimeError {
                            message: format!("Cannot raise {:?} to power {:?}", l, r),
                        }),
                    })?;
                }

                Instruction::FloorDiv(dest, lhs, rhs) => {
                    self.binary_op(dest, lhs, rhs, |l, r| match (l, r) {
                        (Value::Int(a), Value::Int(b)) => floor_div_i64(*a, *b).map(Value::Int),
                        (Value::Float(a), Value::Float(b)) => {
                            if *b == 0.0 {
                                Err(LustError::RuntimeError {
                                    message: "Division by zero".to_string(),
                                })
                            } else {
                                Ok(Value::Float((a / b).floor()))
                            }
                        }
                        (Value::Int(a), Value::Float(b)) => {
                            if *b == 0.0 {
                                Err(LustError::RuntimeError {
                                    message: "Division by zero".to_string(),
                                })
                            } else {
                                Ok(Value::Float((float_from_int(*a) / b).floor()))
                            }
                        }
                        (Value::Float(a), Value::Int(b)) => {
                            if *b == 0 {
                                Err(LustError::RuntimeError {
                                    message: "Division by zero".to_string(),
                                })
                            } else {
                                Ok(Value::Float((a / float_from_int(*b)).floor()))
                            }
                        }
                        _ => Err(LustError::RuntimeError {
                            message: format!("Cannot floor divide {:?} and {:?}", l, r),
                        }),
                    })?;
                }

                Instruction::Neg(dest, src) => {
                    let value = self.get_register(src)?;
                    let result = match value {
                        Value::Int(i) => Value::Int(i.wrapping_neg()),
                        Value::Float(f) => Value::Float(-f),
                        _ => {
                            return Err(LustError::RuntimeError {
                                message: format!("Cannot negate {:?}", value),
                            });
                        }
                    };
                    self.set_register(dest, result)?;
                }

                Instruction::Eq(dest, lhs, rhs) => {
                    let left = self.get_register(lhs)?;
                    let right = self.get_register(rhs)?;
                    self.set_register(dest, Value::Bool(left == right))?;
                }

                Instruction::Ne(dest, lhs, rhs) => {
                    let left = self.get_register(lhs)?;
                    let right = self.get_register(rhs)?;
                    self.set_register(dest, Value::Bool(left != right))?;
                }

                Instruction::Lt(dest, lhs, rhs) => {
                    self.comparison_op(dest, lhs, rhs, |l, r| l < r, |l, r| l < r)?;
                }

                Instruction::Le(dest, lhs, rhs) => {
                    self.comparison_op(dest, lhs, rhs, |l, r| l <= r, |l, r| l <= r)?;
                }

                Instruction::Gt(dest, lhs, rhs) => {
                    self.comparison_op(dest, lhs, rhs, |l, r| l > r, |l, r| l > r)?;
                }

                Instruction::Ge(dest, lhs, rhs) => {
                    self.comparison_op(dest, lhs, rhs, |l, r| l >= r, |l, r| l >= r)?;
                }

                Instruction::And(dest, lhs, rhs) => {
                    let left = self.get_register(lhs)?;
                    let right = self.get_register(rhs)?;
                    let result = Value::Bool(left.is_truthy() && right.is_truthy());
                    self.set_register(dest, result)?;
                }

                Instruction::Or(dest, lhs, rhs) => {
                    let left = self.get_register(lhs)?;
                    let right = self.get_register(rhs)?;
                    let result = Value::Bool(left.is_truthy() || right.is_truthy());
                    self.set_register(dest, result)?;
                }

                Instruction::Not(dest, src) => {
                    let value = self.get_register(src)?;
                    self.set_register(dest, Value::Bool(!value.is_truthy()))?;
                }

                Instruction::BitAnd(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Int(a & b)))?;
                }
                Instruction::BitOr(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Int(a | b)))?;
                }
                Instruction::BitXor(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| Ok(Value::Int(a ^ b)))?;
                }
                Instruction::Shl(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| {
                        Ok(Value::Int(a.wrapping_shl((b & 63) as u32)))
                    })?;
                }
                Instruction::Shr(dest, lhs, rhs) => {
                    self.int_binary_op(dest, lhs, rhs, |a, b| {
                        Ok(Value::Int(a.wrapping_shr((b & 63) as u32)))
                    })?;
                }
                Instruction::BitNot(dest, src) => {
                    let value = self.get_register(src)?;
                    match value {
                        Value::Int(a) => {
                            self.set_register(dest, Value::Int(!a))?;
                        }
                        _ => {
                            return Err(LustError::RuntimeError {
                                message: format!("Cannot perform bitwise NOT on {:?}", value),
                            });
                        }
                    }
                }

                Instruction::Jump(offset) => {
                    let frame = self.call_stack.last_mut().unwrap();
                    let target_ip = (frame.ip as isize + offset as isize) as usize;
                    frame.ip = target_ip;
                    #[cfg(feature = "std")]
                    if offset < 0
                        && self.backend == crate::vm::VmBackend::Cranelift
                        && cur_has_osr_entries
                    {
                        frame_dirty = true;
                        if let Some(outcome) =
                            self.try_enter_osr_at_backedge(func_idx, target_ip)?
                        {
                            match outcome {
                                crate::vm::OsrBackedgeOutcome::Return(Some(val)) => {
                                    return Ok(val);
                                }
                                crate::vm::OsrBackedgeOutcome::Return(None)
                                | crate::vm::OsrBackedgeOutcome::DeoptimizedResume => {
                                    continue 'dispatch;
                                }
                            }
                        }
                    }
                }

                Instruction::JumpIf(cond, offset) => {
                    let condition = self.get_register(cond)?;
                    let truthy = match condition {
                        Value::Bool(b) => *b,
                        other => other.is_truthy(),
                    };
                    if truthy {
                        let frame = self.call_stack.last_mut().unwrap();
                        frame.ip = (frame.ip as isize + offset as isize) as usize;
                    }
                }

                Instruction::JumpIfNot(cond, offset) => {
                    let condition = self.get_register(cond)?;
                    let truthy = match condition {
                        Value::Bool(b) => *b,
                        other => other.is_truthy(),
                    };
                    if !truthy {
                        let frame = self.call_stack.last_mut().unwrap();
                        frame.ip = (frame.ip as isize + offset as isize) as usize;
                    }
                }

                Instruction::BranchLtInt(lhs, rhs, offset) => {
                    let left = self.get_register(lhs)?;
                    let right = self.get_register(rhs)?;
                    let (Value::Int(a), Value::Int(b)) = (left, right) else {
                        return Err(LustError::RuntimeError {
                            message: format!(
                                "Typed numeric instruction expected int operands, got {:?} and {:?}",
                                left, right
                            ),
                        });
                    };
                    if !(*a < *b) {
                        let frame = self.call_stack.last_mut().unwrap();
                        frame.ip = (frame.ip as isize + offset as isize) as usize;
                    }
                }

                Instruction::BranchLeInt(lhs, rhs, offset) => {
                    let left = self.get_register(lhs)?;
                    let right = self.get_register(rhs)?;
                    let (Value::Int(a), Value::Int(b)) = (left, right) else {
                        return Err(LustError::RuntimeError {
                            message: format!(
                                "Typed numeric instruction expected int operands, got {:?} and {:?}",
                                left, right
                            ),
                        });
                    };
                    if !(*a <= *b) {
                        let frame = self.call_stack.last_mut().unwrap();
                        frame.ip = (frame.ip as isize + offset as isize) as usize;
                    }
                }

                Instruction::Call(func_reg, first_arg, arg_count, dest_reg)
                    if self.plain_function_callee(func_reg) =>
                {
                    frame_dirty = true;
                    let callee_idx =
                        self.push_bytecode_call_frame(func_reg, first_arg, arg_count, dest_reg)?;
                    #[cfg(feature = "std")]
                    if crate::vm::tracing::has_trace_hook() {
                        let callee = &self.functions[callee_idx];
                        let line = callee.chunk.lines.first().copied().unwrap_or(0);
                        self.dispatch_call_hook(&callee.name, line, self.call_stack.len());
                    }
                    #[cfg(feature = "std")]
                    if self.backend == crate::vm::VmBackend::Cranelift
                        && self.trace_recorder.is_none()
                        && let Some(finished) = self.try_run_cranelift_top_frame()?
                    {
                        if let Some(val) = finished {
                            return Ok(val);
                        }
                        continue 'dispatch;
                    }
                    if JIT_ENABLED
                        && self.trace_recorder.is_none()
                        && let Some(finished) = self.run_compiled_function(callee_idx)?
                    {
                        return Ok(finished);
                    }
                }

                Instruction::Call(func_reg, first_arg, arg_count, dest_reg) => {
                    frame_dirty = true;
                    let func_value = self.get_register(func_reg)?.clone();
                    if let Some(intrinsic_name) = Self::get_effect_intrinsic_name(&func_value) {
                        let mut args = Vec::with_capacity(arg_count as usize);
                        for i in 0..arg_count {
                            args.push(self.get_register(first_arg + i)?.clone());
                        }
                        let result = self.dispatch_effect_intrinsic(&intrinsic_name, args)?;
                        self.set_register(dest_reg, result)?;
                        continue 'dispatch;
                    }
                    #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
                    let mut func_value = func_value;
                    // Check if this is a table/userdata with __call metamethod (Lua compat)
                    let needs_call_value = {
                        let mut check_value = &func_value;
                        // Unwrap LuaValue enum if needed
                        if let Value::Enum(object) = &func_value
                            && object.enum_name == "LuaValue"
                            && (object.variant == "Table" || object.variant == "Userdata")
                            && let Some(inner) = object.values.as_ref().and_then(|v| v.first())
                        {
                            check_value = inner;
                        }
                        // Check if it's a LuaTable/LuaUserdata struct with metamethods
                        if let Value::Struct(object) = check_value {
                            (object.name == "LuaTable" || object.name == "LuaUserdata")
                                && check_value.struct_get_field("metamethods").is_some()
                        } else {
                            false
                        }
                    };

                    if needs_call_value {
                        // eprintln!("DEBUG Instruction::Call: Delegating to call_value for table/userdata");
                        // Delegate to call_value which handles __call metamethods
                        let mut args = Vec::new();
                        for i in 0..arg_count {
                            args.push(self.get_register(first_arg + i)?.clone());
                        }
                        let result = self.call_value(&func_value, args)?;
                        self.set_register(dest_reg, result)?;
                        continue;
                    } else {
                        // Check what type we're actually dealing with
                        if let Value::Enum(object) = &func_value
                            && object.enum_name == "LuaValue"
                            && (object.variant == "Table" || object.variant == "Userdata")
                        {
                            #[cfg(feature = "std")]
                            let EnumObject { variant, .. } = object.as_ref();
                            #[cfg(feature = "std")]
                            eprintln!(
                                "DEBUG Instruction::Call: Have LuaValue.{} but needs_call_value=false",
                                variant
                            );
                        }
                    }

                    #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
                    loop {
                        let handle = match &func_value {
                            Value::Enum(object)
                                if object.enum_name == "LuaValue"
                                    && object.variant == "Function" =>
                            {
                                object
                                    .values
                                    .as_ref()
                                    .and_then(|vals| vals.first())
                                    .and_then(|v| v.struct_get_field("handle"))
                                    .and_then(|v| v.as_int())
                                    .map(|i| i as usize)
                            }
                            _ => None,
                        };
                        let Some(handle) = handle else { break };
                        let inner =
                            crate::lua_compat::lookup_lust_function(handle).ok_or_else(|| {
                                LustError::RuntimeError {
                                    message: format!(
                                        "LuaValue function handle {} was not registered with VM",
                                        handle
                                    ),
                                }
                            })?;
                        func_value = inner;
                    }
                    match func_value {
                        Value::Function(func_idx) => {
                            let mut args = core::mem::take(&mut self.arg_scratch);
                            args.clear();
                            for i in 0..arg_count {
                                args.push(self.get_register(first_arg + i)?.clone());
                            }

                            self.push_checked_call_frame(
                                func_idx,
                                Some(dest_reg),
                                args,
                                Vec::new(),
                                Vec::new(),
                            )?;
                        }

                        Value::Closure(closure) => {
                            let func_idx = closure.function_idx;
                            let upvalues = &closure.upvalues;
                            let mut args = core::mem::take(&mut self.arg_scratch);
                            args.clear();
                            for i in 0..arg_count {
                                args.push(self.get_register(first_arg + i)?.clone());
                            }

                            let upvalue_values: Vec<Value> =
                                upvalues.iter().map(|uv| uv.get()).collect();
                            self.push_checked_call_frame(
                                func_idx,
                                Some(dest_reg),
                                args,
                                upvalue_values,
                                upvalues.clone(),
                            )?;
                        }

                        Value::NativeFunction(native_fn) => {
                            let mut args = core::mem::take(&mut self.arg_scratch);
                            args.clear();
                            for i in 0..arg_count {
                                args.push(self.get_register(first_arg + i)?.fast_clone());
                            }

                            let _native_guard = self.enter_native_scope();
                            let _vm_guard = crate::vm::CurrentVmGuard::from_vm(self);
                            let outcome = native_fn(&args);
                            drop(_vm_guard);
                            drop(_native_guard);
                            args.clear();
                            if args.capacity() > self.arg_scratch.capacity() {
                                self.arg_scratch = args;
                            }
                            let outcome =
                                outcome.map_err(|e| LustError::RuntimeError { message: e })?;
                            self.handle_native_call_outcome(dest_reg, outcome)?;
                        }

                        _ => {
                            return Err(LustError::RuntimeError {
                                message: format!(
                                    "Cannot call non-function value: {:?}",
                                    func_value
                                ),
                            });
                        }
                    }
                }

                Instruction::Return(value_reg) => {
                    frame_dirty = true;
                    let return_value = if value_reg == 255 {
                        Value::Nil
                    } else {
                        self.get_register(value_reg)?.fast_clone()
                    };
                    if let Some(finished) = self.finish_return(return_value)? {
                        return Ok(finished);
                    }
                }

                Instruction::NewArray(dest, first_elem, count) => {
                    let element_count = count as usize;
                    self.budgets.charge_value_vec(element_count)?;
                    let mut elements = Vec::with_capacity(element_count);
                    for i in 0..count {
                        elements.push(self.get_register(first_elem + i)?.clone());
                    }

                    self.set_register(dest, Value::array(elements))?;
                }

                Instruction::TupleNew(dest, first_elem, count) => {
                    let mut parts = Vec::with_capacity(count as usize);
                    let mut total_elements: usize = 0;
                    for offset in 0..(count as usize) {
                        let value = self.get_register(first_elem + offset as u8)?.clone();
                        total_elements = total_elements.saturating_add(match &value {
                            Value::Tuple(existing) => existing.len(),
                            _ => 1,
                        });
                        parts.push(value);
                    }

                    self.budgets.charge_value_vec(total_elements)?;
                    let mut elements = Vec::with_capacity(total_elements);
                    for value in parts {
                        if let Value::Tuple(existing) = value {
                            elements.extend(existing.iter().cloned());
                        } else {
                            elements.push(value);
                        }
                    }

                    self.set_register(dest, Value::tuple(elements))?;
                }

                Instruction::TupleGet(dest, tuple_reg, index) => {
                    let tuple_value = self.get_register(tuple_reg)?.clone();
                    if let Value::Tuple(values) = tuple_value {
                        let idx = index as usize;
                        if idx >= values.len() {
                            return Err(LustError::RuntimeError {
                                message: format!(
                                    "Tuple index {} out of bounds (len {})",
                                    idx,
                                    values.len()
                                ),
                            });
                        }

                        let value = values[idx].clone();
                        self.set_register(dest, value)?;
                    } else {
                        return Err(LustError::RuntimeError {
                            message: "Attempted to destructure non-tuple value".to_string(),
                        });
                    }
                }

                Instruction::NewMap(dest) => {
                    self.set_register(dest, self.new_map_value())?;
                }

                Instruction::NewStruct(
                    dest,
                    name_idx,
                    first_field_name_idx,
                    first_field,
                    field_count,
                ) => {
                    self.budgets.charge_value_vec(field_count as usize)?;
                    let func = &self.functions[func_idx];
                    let struct_name = func.chunk.constants[name_idx as usize]
                        .as_string()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "Struct name must be a string".to_string(),
                        })?
                        .to_string();
                    let mut fields = Vec::with_capacity(field_count as usize);
                    for i in 0..field_count {
                        let field_name = func.chunk.constants
                            [(first_field_name_idx + i as u16) as usize]
                            .as_string_rc()
                            .ok_or_else(|| LustError::RuntimeError {
                                message: "Field name must be a string".to_string(),
                            })?;
                        let value = self.get_register(first_field + i)?.clone();
                        fields.push((field_name, value));
                    }

                    let struct_value = self.instantiate_struct(&struct_name, fields)?;
                    self.set_register(dest, struct_value)?;
                }

                Instruction::NewEnumUnit(dest, enum_name_idx, variant_idx) => {
                    let func = &self.functions[func_idx];
                    let enum_name = func.chunk.constants[enum_name_idx as usize]
                        .as_string_rc()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "Enum name must be a string".to_string(),
                        })?;
                    let variant_name = func.chunk.constants[variant_idx as usize]
                        .as_string_rc()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "Variant name must be a string".to_string(),
                        })?;
                    #[cfg(feature = "std")]
                    let value = if enum_name.as_str() == "Option"
                        && variant_name.as_str() == "None"
                    {
                        Value::Enum(option_none_singleton())
                    } else {
                        self.unit_enum(&enum_name, &variant_name)
                    };
                    #[cfg(not(feature = "std"))]
                    let value = self.unit_enum(&enum_name, &variant_name);
                    self.set_register(dest, value)?;
                }

                Instruction::NewEnumVariant(
                    dest,
                    enum_name_idx,
                    variant_idx,
                    first_value,
                    value_count,
                ) => {
                    self.budgets.charge_value_vec(value_count as usize)?;
                    let func = &self.functions[func_idx];
                    let enum_name = func.chunk.constants[enum_name_idx as usize]
                        .as_string()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "Enum name must be a string".to_string(),
                        })?
                        .to_string();
                    let variant_name = func.chunk.constants[variant_idx as usize]
                        .as_string()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "Variant name must be a string".to_string(),
                        })?
                        .to_string();
                    let mut values = Vec::with_capacity(value_count as usize);
                    for i in 0..value_count {
                        values.push(self.get_register(first_value + i)?.clone());
                    }

                    self.set_register(dest, Value::enum_variant(enum_name, variant_name, values))?;
                }

                Instruction::IsEnumVariant(dest, value_reg, enum_name_idx, variant_idx) => {
                    let value = self.get_register(value_reg)?;
                    let func = &self.functions[func_idx];
                    let enum_name = func.chunk.constants[enum_name_idx as usize]
                        .as_string()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "Enum name must be a string".to_string(),
                        })?;
                    let variant_name = func.chunk.constants[variant_idx as usize]
                        .as_string()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "Variant name must be a string".to_string(),
                        })?;
                    let is_variant = value.is_enum_variant(enum_name, variant_name);
                    self.set_register(dest, Value::Bool(is_variant))?;
                }

                Instruction::GetEnumValue(dest, enum_reg, index) => {
                    let enum_value = self.get_register(enum_reg)?;
                    if let Some((_, _, Some(values))) = enum_value.as_enum() {
                        if (index as usize) < values.len() {
                            self.set_register(dest, values[index as usize].clone())?;
                        } else {
                            return Err(LustError::RuntimeError {
                                message: format!(
                                    "Enum value index {} out of bounds (has {} values)",
                                    index,
                                    values.len()
                                ),
                            });
                        }
                    } else {
                        return Err(LustError::RuntimeError {
                            message: "GetEnumValue requires an enum variant with values"
                                .to_string(),
                        });
                    }
                }

                Instruction::GetField(dest, obj, field_idx) => {
                    let object = self.get_register(obj)?;
                    let func = &self.functions[func_idx];
                    let field_name = func.chunk.constants[field_idx as usize]
                        .as_string_rc()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "Field name must be a string".to_string(),
                        })?;
                    let value = if let Some(map_val) = Self::lua_table_map(object) {
                        frame_dirty = true;
                        if let Value::Map(map) = map_val {
                            let raw_key_value = Value::String(field_name.clone());
                            let key =
                                self.make_hash_key(&Self::lua_table_key_value(&raw_key_value))?;
                            let borrowed = map.borrow();
                            borrowed.get(&key).cloned().unwrap_or(Value::Nil)
                        } else {
                            Value::Nil
                        }
                    } else {
                        match object {
                            Value::Struct(_) => object
                                .struct_get_field_rc(&field_name)
                                .unwrap_or(Value::Nil),
                            Value::Map(map) => {
                                let key = ValueKey::from(field_name.clone());
                                map.borrow().get(&key).cloned().unwrap_or(Value::Nil)
                            }

                            _ => {
                                return Err(LustError::RuntimeError {
                                    message: format!(
                                        "Cannot get field '{}' from {:?}",
                                        field_name.as_str(),
                                        object
                                    ),
                                });
                            }
                        }
                    };
                    self.set_register(dest, value)?;
                }

                Instruction::SetField(obj_reg, field_idx, value_reg) => {
                    let object = self.get_register(obj_reg)?.clone();
                    let value = self.get_register(value_reg)?.clone();
                    let func = &self.functions[func_idx];
                    let field_name = func.chunk.constants[field_idx as usize]
                        .as_string_rc()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "Field name must be a string".to_string(),
                        })?;
                    let mut invalidate_key: Option<usize> = None;
                    if let Some(map_val) = Self::lua_table_map(&object) {
                        frame_dirty = true;
                        if let Value::Map(map) = map_val {
                            let raw_key_value = Value::String(field_name.clone());
                            let key =
                                self.make_hash_key(&Self::lua_table_key_value(&raw_key_value))?;
                            if self.budgets.mem_budget_enabled() && !map.borrow().contains_key(&key)
                            {
                                self.budgets.charge_map_entry_estimate()?;
                            }
                            map.borrow_mut().insert(key, value);
                        } else {
                            return Err(LustError::RuntimeError {
                                message: format!(
                                    "Cannot set field '{}' on {:?}",
                                    field_name.as_str(),
                                    object
                                ),
                            });
                        }
                    } else {
                        match &object {
                            Value::Struct(_) => {
                                invalidate_key = Self::struct_cache_key(&object);
                                object
                                    .struct_set_field_rc(&field_name, value)
                                    .map_err(|message| LustError::RuntimeError { message })?;
                            }

                            Value::Map(map) => {
                                let key = ValueKey::from(field_name.clone());
                                if self.budgets.mem_budget_enabled()
                                    && !map.borrow().contains_key(&key)
                                {
                                    self.budgets.charge_map_entry_estimate()?;
                                }
                                map.borrow_mut().insert(key, value);
                            }

                            _ => {
                                return Err(LustError::RuntimeError {
                                    message: format!(
                                        "Cannot set field '{}' on {:?}",
                                        field_name.as_str(),
                                        object
                                    ),
                                });
                            }
                        }
                    }

                    if !self.struct_tostring_cache.is_empty()
                        && let Some(key) = invalidate_key
                    {
                        self.struct_tostring_cache.remove(&key);
                    }
                }

                Instruction::GetFieldSlot(dest, obj, slot_idx) => {
                    let object = self.get_register(obj)?;
                    let value = match object {
                        Value::Struct(s) if s.name != "LuaTable" => object
                            .struct_get_field_indexed(slot_idx as usize)
                            .unwrap_or(Value::Nil),
                        Value::Struct(s) => {
                            let field_name = s
                                .layout
                                .field_names()
                                .get(slot_idx as usize)
                                .cloned()
                                .ok_or_else(|| LustError::RuntimeError {
                                    message: format!(
                                        "Struct '{}' field index {} out of bounds",
                                        s.name, slot_idx
                                    ),
                                })?;
                            if let Some(Value::Map(map)) = Self::lua_table_map(object) {
                                frame_dirty = true;
                                let raw_key_value = Value::String(field_name);
                                let key = self
                                    .make_hash_key(&Self::lua_table_key_value(&raw_key_value))?;
                                let borrowed = map.borrow();
                                borrowed.get(&key).cloned().unwrap_or(Value::Nil)
                            } else {
                                object
                                    .struct_get_field_indexed(slot_idx as usize)
                                    .unwrap_or(Value::Nil)
                            }
                        }
                        _ => {
                            return Err(LustError::RuntimeError {
                                message: format!(
                                    "Cannot get field slot {} from {:?}",
                                    slot_idx, object
                                ),
                            });
                        }
                    };
                    self.set_register(dest, value)?;
                }

                Instruction::SetFieldSlot(obj_reg, slot_idx, value_reg) => {
                    let object = self.get_register(obj_reg)?.clone();
                    let value = self.get_register(value_reg)?.clone();
                    let mut invalidate_key: Option<usize> = None;
                    match &object {
                        Value::Struct(s) if s.name != "LuaTable" => {
                            invalidate_key = Self::struct_cache_key(&object);
                            if !crate::vm::cycle::CycleCollector::is_leaf(&value) {
                                self.cycle_collector.record_mutation();
                            }
                            object
                                .struct_set_field_indexed(slot_idx as usize, value)
                                .map_err(|message| LustError::RuntimeError { message })?;
                        }
                        Value::Struct(s) => {
                            let field_name = s
                                .layout
                                .field_names()
                                .get(slot_idx as usize)
                                .cloned()
                                .ok_or_else(|| LustError::RuntimeError {
                                    message: format!(
                                        "Struct '{}' field index {} out of bounds",
                                        s.name, slot_idx
                                    ),
                                })?;
                            if let Some(Value::Map(map)) = Self::lua_table_map(&object) {
                                frame_dirty = true;
                                let raw_key_value = Value::String(field_name);
                                let key = self
                                    .make_hash_key(&Self::lua_table_key_value(&raw_key_value))?;
                                if self.budgets.mem_budget_enabled()
                                    && !map.borrow().contains_key(&key)
                                {
                                    self.budgets.charge_map_entry_estimate()?;
                                }
                                map.borrow_mut().insert(key, value);
                            } else {
                                invalidate_key = Self::struct_cache_key(&object);
                                object
                                    .struct_set_field_indexed(slot_idx as usize, value)
                                    .map_err(|message| LustError::RuntimeError { message })?;
                            }
                        }
                        _ => {
                            return Err(LustError::RuntimeError {
                                message: format!(
                                    "Cannot set field slot {} on {:?}",
                                    slot_idx, object
                                ),
                            });
                        }
                    }
                    if !self.struct_tostring_cache.is_empty()
                        && let Some(key) = invalidate_key
                    {
                        self.struct_tostring_cache.remove(&key);
                    }
                }

                Instruction::GetFieldAddSetInt(obj_reg, slot_idx, addend_reg) => {
                    let object = self.get_register(obj_reg)?.clone();
                    let addend = self.get_register(addend_reg)?.clone();
                    let idx = slot_idx as usize;
                    let mut invalidate_key: Option<usize> = None;

                    let current = match &object {
                        Value::Struct(s) if s.name != "LuaTable" => {
                            let len = s.fields.borrow().len();
                            if idx >= len {
                                return Err(LustError::RuntimeError {
                                    message: format!(
                                        "Struct '{}' field index {} out of bounds (len {})",
                                        s.name, idx, len
                                    ),
                                });
                            }
                            object
                                .struct_get_field_indexed(idx)
                                .unwrap_or(Value::Nil)
                        }
                        Value::Struct(s) => {
                            let field_name = s
                                .layout
                                .field_names()
                                .get(idx)
                                .cloned()
                                .ok_or_else(|| LustError::RuntimeError {
                                    message: format!(
                                        "Struct '{}' field index {} out of bounds",
                                        s.name, slot_idx
                                    ),
                                })?;
                            if let Some(Value::Map(map)) = Self::lua_table_map(&object) {
                                frame_dirty = true;
                                let raw_key_value = Value::String(field_name);
                                let key = self
                                    .make_hash_key(&Self::lua_table_key_value(&raw_key_value))?;
                                let borrowed = map.borrow();
                                borrowed.get(&key).cloned().unwrap_or(Value::Nil)
                            } else {
                                object
                                    .struct_get_field_indexed(idx)
                                    .unwrap_or(Value::Nil)
                            }
                        }
                        _ => {
                            return Err(LustError::RuntimeError {
                                message: format!(
                                    "Cannot get field slot {} from {:?}",
                                    slot_idx, object
                                ),
                            });
                        }
                    };

                    let sum = match (&current, &addend) {
                        (Value::Int(a), Value::Int(b)) => Value::Int(a.wrapping_add(*b)),
                        (Value::Float(a), Value::Float(b)) => Value::Float(a + b),
                        (Value::Int(a), Value::Float(b)) => Value::Float(float_from_int(*a) + *b),
                        (Value::Float(a), Value::Int(b)) => Value::Float(*a + float_from_int(*b)),
                        (Value::String(a), Value::String(b)) => {
                            let cap = a.len().saturating_add(b.len());
                            self.budgets.charge_mem_bytes(cap)?;
                            Value::string(format!("{}{}", a, b))
                        }
                        _ => {
                            return Err(LustError::RuntimeError {
                                message: format!("Cannot add {:?} and {:?}", current, addend),
                            });
                        }
                    };

                    match &object {
                        Value::Struct(s) if s.name != "LuaTable" => {
                            invalidate_key = Self::struct_cache_key(&object);
                            if !crate::vm::cycle::CycleCollector::is_leaf(&sum) {
                                self.cycle_collector.record_mutation();
                            }
                            object
                                .struct_set_field_indexed(idx, sum)
                                .map_err(|message| LustError::RuntimeError { message })?;
                        }
                        Value::Struct(s) => {
                            let field_name = s
                                .layout
                                .field_names()
                                .get(idx)
                                .cloned()
                                .ok_or_else(|| LustError::RuntimeError {
                                    message: format!(
                                        "Struct '{}' field index {} out of bounds",
                                        s.name, slot_idx
                                    ),
                                })?;
                            if let Some(Value::Map(map)) = Self::lua_table_map(&object) {
                                frame_dirty = true;
                                let raw_key_value = Value::String(field_name);
                                let key = self
                                    .make_hash_key(&Self::lua_table_key_value(&raw_key_value))?;
                                if self.budgets.mem_budget_enabled()
                                    && !map.borrow().contains_key(&key)
                                {
                                    self.budgets.charge_map_entry_estimate()?;
                                }
                                map.borrow_mut().insert(key, sum);
                            } else {
                                invalidate_key = Self::struct_cache_key(&object);
                                object
                                    .struct_set_field_indexed(idx, sum)
                                    .map_err(|message| LustError::RuntimeError { message })?;
                            }
                        }
                        _ => unreachable!(),
                    }
                    if !self.struct_tostring_cache.is_empty()
                        && let Some(key) = invalidate_key
                    {
                        self.struct_tostring_cache.remove(&key);
                    }
                }

                Instruction::Concat(dest, lhs, rhs) => {
                    let mut in_place_done = false;
                    if dest == lhs {
                        let right = self.get_register(rhs)?.clone();
                        if matches!(right, Value::Struct(_) | Value::Enum(_)) {
                            frame_dirty = true;
                        }
                        let right_str = self.value_to_string_for_concat(&right)?;
                        let right_len = right_str.len();

                        let base_reg = self
                            .call_stack
                            .last()
                            .ok_or_else(|| LustError::RuntimeError {
                                message: "Empty call stack".to_string(),
                            })?
                            .base_reg;
                        if let Value::String(ref mut rc) = self.vm_registers[base_reg + dest as usize] {
                            if let Some(s_mut) = Rc::get_mut(rc) {
                                self.budgets.charge_mem_bytes(right_len)?;
                                s_mut.push_str(right_str.as_ref());
                                in_place_done = true;
                            }
                        }
                    }

                    if !in_place_done {
                        let left = self.get_register(lhs)?.clone();
                        let right = self.get_register(rhs)?.clone();
                        if matches!(left, Value::Struct(_) | Value::Enum(_))
                            || matches!(right, Value::Struct(_) | Value::Enum(_))
                        {
                            frame_dirty = true;
                        }
                        let left_str = self.value_to_string_for_concat(&left)?;
                        let right_str = self.value_to_string_for_concat(&right)?;
                        let cap = left_str.len().saturating_add(right_str.len());
                        self.budgets.charge_mem_bytes(cap)?;
                        let mut combined = String::with_capacity(if dest == lhs { cap.max(left_str.len() * 2) } else { cap });
                        combined.push_str(left_str.as_ref());
                        combined.push_str(right_str.as_ref());
                        let result = Value::string(combined);
                        self.set_register(dest, result)?;
                    }
                }

                Instruction::GetIndex(dest, array_reg, index_reg) => {
                    let collection = self.get_register(array_reg)?.clone();
                    let index = self.get_register(index_reg)?.clone();
                    let result = if let Some(map_val) = Self::lua_table_map(&collection) {
                        frame_dirty = true;
                        if let Value::Map(map) = map_val {
                            let raw_key = self.make_hash_key(&Self::lua_table_key_value(&index))?;
                            let borrowed = map.borrow();
                            borrowed.get(&raw_key).cloned().unwrap_or(Value::Nil)
                        } else {
                            return Err(LustError::RuntimeError {
                                message: format!("Cannot index {:?}", collection.type_of()),
                            });
                        }
                    } else {
                        match collection {
                            Value::Array(arr) => {
                                let idx =
                                    index.as_int().ok_or_else(|| LustError::RuntimeError {
                                        message: "Array index must be an integer".to_string(),
                                    })?;
                                if let Some(val) =
                                    crate::bytecode::value::get_specialized_cached_element(
                                        &arr, idx,
                                    )
                                {
                                    val
                                } else if idx >= 0
                                    && let Some(val) = arr.get_value_fast(idx as usize)
                                {
                                    val
                                } else {
                                    return Err(LustError::RuntimeError {
                                        message: format!(
                                            "Array index {} out of bounds (length: {})",
                                            idx,
                                            arr.len_fast()
                                        ),
                                    });
                                }
                            }

                            Value::Map(map) => {
                                frame_dirty = true;
                                let key = self.make_hash_key(&index)?;
                                map.borrow().get(&key).cloned().unwrap_or(Value::Nil)
                            }

                            _ => {
                                return Err(LustError::RuntimeError {
                                    message: format!("Cannot index {:?}", collection.type_of()),
                                });
                            }
                        }
                    };
                    self.set_register(dest, result)?;
                }

                Instruction::MapGet(dest, map, key) | Instruction::MapGetUnwrap(dest, map, key) => {
                    let value = map_read(
                        self.get_register(map)?,
                        self.get_register(key)?,
                        matches!(instruction, Instruction::MapGetUnwrap(..)),
                    )?;
                    self.set_register(dest, value)?;
                }

                Instruction::MapSet(map, key, value) => {
                    let map = self.get_register(map)?.fast_clone();
                    let key = self.get_register(key)?.fast_clone();
                    let value = self.get_register(value)?.fast_clone();
                    if map_write(
                        &map,
                        &key,
                        &value,
                        &mut self.budgets,
                        &mut self.cycle_collector,
                    )? {
                        self.maybe_collect_cycles();
                    }
                }

                Instruction::ArrayGetUnwrap(dest, array_reg, index_reg) => {
                    let collection = self.get_register(array_reg)?.fast_clone();
                    let index = self.get_register(index_reg)?.fast_clone();
                    match collection {
                        Value::Array(arr) => {
                            let idx = index.as_int().ok_or_else(|| LustError::RuntimeError {
                                message: "Array index must be an integer".to_string(),
                            })?;
                            let result = if let Some(val) =
                                crate::bytecode::value::get_specialized_cached_element(&arr, idx)
                            {
                                val
                            } else if idx >= 0
                                && let Some(val) = arr.get_value_fast(idx as usize)
                            {
                                val
                            } else {
                                return Err(LustError::RuntimeError {
                                    message: format!(
                                        "Array index {} out of bounds (length: {})",
                                        idx,
                                        arr.len_fast()
                                    ),
                                });
                            };
                            self.set_register(dest, result)?;
                        }
                        other => {
                            return Err(LustError::RuntimeError {
                                message: format!(
                                    "array.get expects an array, got {:?}",
                                    other.type_of()
                                ),
                            });
                        }
                    }
                }

                Instruction::ArrayPush(array_reg, value_reg) => {
                    let collection = self.get_register(array_reg)?.fast_clone();
                    let value = self.get_register(value_reg)?.fast_clone();
                    match collection {
                        Value::Array(arr) => {
                            charge_array_push_budget(&mut self.budgets, &arr, &value)?;
                            if !crate::vm::cycle::CycleCollector::is_leaf(&value) {
                                self.cycle_collector.record_mutation();
                            }
                            crate::bytecode::value::sync_specialized_array(&arr);
                            arr.push_value_fast(value);
                            crate::bytecode::value::invalidate_specialized_array_cache(
                                arr.as_ptr() as usize,
                            );
                        }
                        other => {
                            return Err(LustError::RuntimeError {
                                message: format!(
                                    "array.push expects an array, got {:?}",
                                    other.type_of()
                                ),
                            });
                        }
                    }
                }

                Instruction::TryGetIndex(dest, collection_reg, index_reg) => {
                    let collection = self.get_register(collection_reg)?.clone();
                    let index = self.get_register(index_reg)?.clone();
                    let result = if let Some(map_val) = Self::lua_table_map(&collection) {
                        frame_dirty = true;
                        if let Value::Map(map) = map_val {
                            let raw_key = self.make_hash_key(&Self::lua_table_key_value(&index))?;
                            let borrowed = map.borrow();
                            borrowed.get(&raw_key).cloned().unwrap_or(Value::Nil)
                        } else {
                            return Err(LustError::RuntimeError {
                                message: format!("Cannot index {:?}", collection.type_of()),
                            });
                        }
                    } else {
                        match &collection {
                            Value::Array(_) => {
                                let index =
                                    index.as_int().ok_or_else(|| LustError::RuntimeError {
                                        message: "Array index must be an integer".to_string(),
                                    })?;
                                self.array_index_result(&collection, index)?
                            }
                            Value::Map(map) => {
                                frame_dirty = true;
                                let key = self.make_hash_key(&index)?;
                                map.borrow().get(&key).cloned().unwrap_or(Value::Nil)
                            }
                            _ => {
                                return Err(LustError::RuntimeError {
                                    message: format!("Cannot index {:?}", collection.type_of()),
                                });
                            }
                        }
                    };
                    self.set_register(dest, result)?;
                }

                Instruction::ArrayIndexOk(val_dest, cond_dest, array_reg, idx_reg) => {
                    let collection = self.get_register(array_reg)?.fast_clone();
                    let index = self.get_register(idx_reg)?.fast_clone();
                    if let Value::Array(arr) = &collection {
                        let idx = index.as_int().ok_or_else(|| LustError::RuntimeError {
                            message: "Array index must be an integer".to_string(),
                        })?;
                        let found = if let Some(val) =
                            crate::bytecode::value::get_specialized_cached_element(arr, idx)
                        {
                            Some(val)
                        } else if let Some(len) =
                            crate::bytecode::value::get_specialized_cached_len(arr)
                        {
                            if idx >= 0 && (idx as usize) < len {
                                arr.get_value_fast(idx as usize)
                            } else {
                                None
                            }
                        } else if idx >= 0 {
                            arr.get_value_fast(idx as usize)
                        } else {
                            None
                        };
                        if let Some(val) = found {
                            self.set_register(val_dest, val)?;
                            self.set_register(cond_dest, Value::Bool(true))?;
                        } else {
                            self.set_register(cond_dest, Value::Bool(false))?;
                        }
                    } else {
                        return Err(LustError::RuntimeError {
                            message: format!("Cannot index {:?}", collection.type_of()),
                        });
                    }
                }

                Instruction::ArrayLen(dest, array_reg) => {
                    let collection = self.get_register(array_reg)?;
                    match collection {
                        Value::Array(arr) => {
                            let len = crate::bytecode::value::get_specialized_cached_len(arr)
                                .unwrap_or_else(|| arr.len_fast());
                            self.set_register(dest, Value::Int(int_from_usize(len)))?;
                        }

                        _ => {
                            return Err(LustError::RuntimeError {
                                message: format!(
                                    "ArrayLen requires array, got {:?}",
                                    collection.type_of()
                                ),
                            });
                        }
                    }
                }

                Instruction::CallMethod(
                    obj_reg,
                    method_name_idx,
                    first_arg,
                    arg_count,
                    dest_reg,
                ) => 'method: {
                    frame_dirty = true;
                    let call_ip = ip_before_execution.saturating_sub(1);
                    let object = self.get_register(obj_reg)?.fast_clone();
                    // Method entries own their layouts, so a recycled allocation
                    // address cannot select a previous receiver's method.
                    if let Value::Struct(receiver) = &object
                        && receiver.name != "LuaTable"
                        && receiver.name == *receiver.layout.name_interned()
                    {
                        let cached_target = self.functions[func_idx]
                            .chunk
                            .ic_slots
                            .borrow_mut()
                            .get_mut(call_ip)
                            .and_then(|slot| slot.probe_method(&receiver.layout));
                        if let Some(target) = cached_target {
                            let mut args = core::mem::take(&mut self.arg_scratch);
                            args.clear();
                            args.push(object.clone());
                            for i in 0..arg_count {
                                args.push(self.get_register(first_arg + i)?.fast_clone());
                            }
                            // Fall through to the trace recorder below: it
                            // inlines user-defined struct methods like calls.
                            self.push_checked_call_frame(
                                target,
                                Some(dest_reg),
                                args,
                                Vec::new(),
                                Vec::new(),
                            )?;
                            #[cfg(feature = "std")]
                            if self.backend == crate::vm::VmBackend::Cranelift
                                && self.trace_recorder.is_none()
                                && let Some(finished) = self.try_run_cranelift_top_frame()?
                            {
                                if let Some(val) = finished {
                                    return Ok(val);
                                }
                                continue 'dispatch;
                            }
                            if JIT_ENABLED
                                && self.trace_recorder.is_none()
                                && let Some(finished) = self.run_compiled_function(target)?
                            {
                                return Ok(finished);
                            }
                            break 'method;
                        }
                    }
                    let method_name = {
                        let func = &self.functions[func_idx];
                        func.chunk.constants[method_name_idx as usize]
                            .as_string()
                            .ok_or_else(|| LustError::RuntimeError {
                                message: "Method name must be a string".to_string(),
                            })?
                            .to_string()
                    };
                    if matches!(&object, Value::Closure(_)) {
                        if method_name == "provide" && arg_count == 1 {
                            self.set_register(dest_reg, object)?;
                            break 'method;
                        }
                        if method_name == "run_sync" && arg_count == 0 {
                            let result = self.call_value(&object, Vec::new())?;
                            self.set_register(dest_reg, result)?;
                            break 'method;
                        }
                    }
                    if method_name == "try_unwrap" {
                        let unwrap_res = match &object {
                            Value::Enum(obj) if obj.enum_name == "Option" => {
                                if obj.variant == "Some" {
                                    Ok(obj.values.as_ref().and_then(|v| v.first().cloned()).unwrap_or(Value::Nil))
                                } else {
                                    Err(LustError::RuntimeError { message: "Option is None".into() })
                                }
                            }
                            Value::Enum(obj) if obj.enum_name == "Result" => {
                                if obj.variant == "Ok" {
                                    Ok(obj.values.as_ref().and_then(|v| v.first().cloned()).unwrap_or(Value::Nil))
                                } else {
                                    Err(LustError::RuntimeError { message: "Result is Err".into() })
                                }
                            }
                            scalar => Ok(scalar.clone()),
                        };
                        let val = unwrap_res?;
                        self.set_register(dest_reg, val)?;
                        break 'method;
                    }
                    let object_type_name = match &object {
                        Value::Struct(object) => Some(object.name.as_str()),
                        Value::Enum(object) => Some(object.enum_name.as_str()),
                        _ => None,
                    };
                    if let Some(struct_name) = object_type_name {
                        let mangled_name = format!("{}:{}", struct_name, method_name);
                        if let Some(target_func_idx) =
                            self.functions.iter().position(|f| f.name == mangled_name)
                        {
                            if let Value::Struct(object) = &object
                                && object.name != "LuaTable"
                                && object.name == *object.layout.name_interned()
                            {
                                let StructObject { layout, .. } = object.as_ref();
                                let caller = self.call_stack.last().unwrap().function_idx;
                                let needed = self.functions[caller].chunk.instructions.len();
                                let mut slots =
                                    self.functions[caller].chunk.ic_slots.borrow_mut();
                                if slots.len() < needed {
                                    slots.resize(needed, ChunkIcSlot::Empty);
                                }
                                if call_ip < slots.len() {
                                    slots[call_ip].insert_method(layout, target_func_idx);
                                }
                            }
                            let mut args = Vec::with_capacity(1 + arg_count as usize);
                            args.push(object.clone());
                            for i in 0..arg_count {
                                args.push(self.get_register(first_arg + i)?.clone());
                            }
                            self.push_deep_call_frame(
                                target_func_idx,
                                Some(dest_reg),
                                args,
                                Vec::new(),
                                Vec::new(),
                            )?;
                            #[cfg(feature = "std")]
                            if self.backend == crate::vm::VmBackend::Cranelift
                                && self.trace_recorder.is_none()
                                && let Some(finished) = self.try_run_cranelift_top_frame()?
                            {
                                if let Some(val) = finished {
                                    return Ok(val);
                                }
                                continue 'dispatch;
                            }
                            break 'method;
                        }

                        let mut candidate_names = vec![mangled_name.clone()];
                        if let Some(simple) = struct_name.rsplit(['.', ':']).next() {
                            candidate_names.push(format!("{}:{}", simple, method_name));
                        }

                        let mut handled = false;
                        for candidate in candidate_names {
                            let mut resolved = None;
                            for variant in [candidate.clone(), candidate.replace('.', "::")] {
                                if let Some((_name, value)) =
                                    self.globals.iter().find(|(name, _)| *name == &variant)
                                {
                                    resolved = Some(value.clone());
                                    break;
                                }
                                if let Some((_name, value)) =
                                    self.natives.iter().find(|(name, _)| *name == &variant)
                                {
                                    resolved = Some(value.clone());
                                    break;
                                }
                            }

                            if let Some(global_func) = resolved {
                                let mut call_args = Vec::with_capacity(1 + arg_count as usize);
                                call_args.push(object.clone());
                                for i in 0..arg_count {
                                    call_args.push(self.get_register(first_arg + i)?.clone());
                                }
                                if self.trace_recorder.is_some() {
                                    self.abandon_trace_recording();
                                }
                                let result = self.call_value(&global_func, call_args)?;
                                self.set_register(dest_reg, result)?;
                                handled = true;
                                break;
                            }
                        }
                        if handled {
                            continue 'dispatch;
                        }
                    }

                    let mut args = Vec::new();
                    for i in 0..arg_count {
                        args.push(self.get_register(first_arg + i)?.clone());
                    }

                    let result = self.call_builtin_method(&object, &method_name, args)?;
                    self.set_register(dest_reg, result)?;
                }

                Instruction::Closure(dest, func_idx, first_upvalue_reg, upvalue_count) => {
                    use crate::bytecode::Upvalue;
                    self.budgets
                        .charge_upvalues_estimate(upvalue_count as usize)?;
                    let mut upvalues = Vec::with_capacity(upvalue_count as usize);
                    for i in 0..upvalue_count {
                        let value = self.get_register(first_upvalue_reg + i)?.clone();
                        upvalues.push(Upvalue::new(value));
                    }

                    let closure = Value::Closure(Rc::new(ClosureObject {
                        function_idx: func_idx as usize,
                        upvalues,
                    }));
                    self.set_register(dest, closure)?;
                }

                Instruction::LoadUpvalue(dest, upvalue_idx) => {
                    let frame = self
                        .call_stack
                        .last()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "Empty call stack".to_string(),
                        })?;
                    if (upvalue_idx as usize) < frame.upvalues.len() {
                        let value = frame.upvalues[upvalue_idx as usize].clone();
                        self.set_register(dest, value)?;
                    } else {
                        return Err(LustError::RuntimeError {
                            message: format!(
                                "Upvalue index {} out of bounds (have {} upvalues)",
                                upvalue_idx,
                                frame.upvalues.len()
                            ),
                        });
                    }
                }

                Instruction::StoreUpvalue(upvalue_idx, src) => {
                    let value = self.get_register(src)?.clone();
                    let frame =
                        self.call_stack
                            .last_mut()
                            .ok_or_else(|| LustError::RuntimeError {
                                message: "Empty call stack".to_string(),
                            })?;
                    let idx = upvalue_idx as usize;
                    if idx < frame.upvalues.len() {
                        frame.upvalues[idx] = value.clone();
                        if idx < frame.shared_upvalues.len() {
                            frame.shared_upvalues[idx].set(value);
                        }
                    } else {
                        return Err(LustError::RuntimeError {
                            message: format!(
                                "Upvalue index {} out of bounds (have {} upvalues)",
                                upvalue_idx,
                                frame.upvalues.len()
                            ),
                        });
                    }
                }

                Instruction::SetIndex(collection_reg, index_reg, value_reg) => {
                    let collection = self.get_register(collection_reg)?.clone();
                    let index = self.get_register(index_reg)?.clone();
                    let value = self.get_register(value_reg)?.clone();
                    if let Some(map_val) = Self::lua_table_map(&collection) {
                        frame_dirty = true;
                        if let Value::Map(map) = map_val {
                            let key = self.make_hash_key(&Self::lua_table_key_value(&index))?;
                            if self.budgets.mem_budget_enabled() && !map.borrow().contains_key(&key)
                            {
                                self.budgets.charge_map_entry_estimate()?;
                            }
                            map.borrow_mut().insert(key, value);
                        } else {
                            return Err(LustError::RuntimeError {
                                message: format!("Cannot index {:?}", collection.type_of()),
                            });
                        }
                    } else {
                        match collection {
                            Value::Array(arr) => {
                                let idx =
                                    index.as_int().ok_or_else(|| LustError::RuntimeError {
                                        message: "Array index must be an integer".to_string(),
                                    })?;
                                if let Some(res) =
                                    crate::bytecode::value::set_specialized_cached_element(
                                        &arr, idx, &value,
                                    )
                                {
                                    res.map_err(|message| LustError::RuntimeError { message })?;
                                } else {
                                    if idx < 0 {
                                        return Err(LustError::RuntimeError {
                                            message: format!(
                                                "Array index {} out of bounds (length: {})",
                                                idx,
                                                arr.len_fast()
                                            ),
                                        });
                                    }
                                    if !crate::vm::cycle::CycleCollector::is_leaf(&value) {
                                        self.cycle_collector.record_mutation();
                                    }
                                    if let Err(len) = arr.set_value_fast(idx as usize, value) {
                                        return Err(LustError::RuntimeError {
                                            message: format!(
                                                "Array index {} out of bounds (length: {})",
                                                idx, len
                                            ),
                                        });
                                    }
                                    crate::bytecode::value::invalidate_specialized_array_cache(
                                        arr.as_ptr() as usize,
                                    );
                                }
                            }

                            Value::Map(map) => {
                                frame_dirty = true;
                                let key = self.make_hash_key(&index)?;
                                if self.budgets.mem_budget_enabled()
                                    && !map.borrow().contains_key(&key)
                                {
                                    self.budgets.charge_map_entry_estimate()?;
                                }
                                map.borrow_mut().insert(key, value);
                            }

                            _ => {
                                return Err(LustError::RuntimeError {
                                    message: format!("Cannot index {:?}", collection.type_of()),
                                });
                            }
                        }
                    }
                }

                Instruction::TypeIs(dest, value_reg, type_name_idx) => {
                    let value = self.get_register(value_reg)?.clone();
                    let func = &self.functions[func_idx];
                    let type_name = func.chunk.constants[type_name_idx as usize]
                        .as_string()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "Type name must be a string".to_string(),
                        })?
                        .to_string();
                    let matches = self.value_is_type(&value, &type_name);
                    self.set_register(dest, Value::Bool(matches))?;
                }

                Instruction::TryCast(dest, value_reg, type_name_idx) => {
                    let value = self.get_register(value_reg)?.clone();
                    let func = &self.functions[func_idx];
                    let type_name = func.chunk.constants[type_name_idx as usize]
                        .as_string()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: "Cast type name must be a string".to_string(),
                        })?
                        .to_string();
                    let result = if self.value_is_type(&value, &type_name) {
                        self.budgets.charge_value_vec(1)?;
                        Value::some(value)
                    } else {
                        Value::none()
                    };
                    self.set_register(dest, result)?;
                }
            }

            if JIT_ENABLED
                && let Some(recorder) = &mut self.trace_recorder
                && recorder.is_recording()
            {
                if self.skip_next_trace_record {
                    self.skip_next_trace_record = false;
                } else {
                    let function = &self.functions[func_idx];
                    let registers_opt =
                        if let Some(frame) = self.call_stack.get(executing_frame_index) {
                            Some(&self.vm_registers[frame.base_reg..frame.base_reg + frame.reg_count as usize])
                        } else if executing_frame_index > 0 {
                            self.call_stack
                                .get(executing_frame_index - 1)
                                .map(|frame| {
                                    &self.vm_registers[frame.base_reg..frame.base_reg + frame.reg_count as usize]
                                })
                        } else {
                            None
                        };
                    let frame_pushed = self.call_stack.len() > executing_frame_index + 1;
                    recorder.globals_version = self.globals_version;
                    if let Some(registers) = registers_opt
                        && let Err(e) = recorder.record_instruction_at_frame(
                            executing_frame_index,
                            instruction,
                            ip_before_execution,
                            registers,
                            function,
                            func_idx,
                            &self.functions,
                            frame_pushed,
                        )
                    {
                        crate::jit::log(|| format!("⚠️  JIT: {}", e));
                        self.abandon_trace_recording();
                    }
                }
            }
        }
    }

    pub(super) fn binary_op<F>(
        &mut self,
        dest: Register,
        lhs: Register,
        rhs: Register,
        op: F,
    ) -> Result<()>
    where
        F: FnOnce(&Value, &Value) -> Result<Value>,
    {
        let left = self.get_register(lhs)?;
        let right = self.get_register(rhs)?;
        let result = op(left, right)?;
        self.set_register(dest, result)
    }

    fn int_binary_op<F>(
        &mut self,
        dest: Register,
        lhs: Register,
        rhs: Register,
        op: F,
    ) -> Result<()>
    where
        F: FnOnce(LustInt, LustInt) -> Result<Value>,
    {
        let left = self.get_register(lhs)?;
        let right = self.get_register(rhs)?;
        let (Value::Int(a), Value::Int(b)) = (left, right) else {
            return Err(LustError::RuntimeError {
                message: format!(
                    "Typed numeric instruction expected int operands, got {:?} and {:?}",
                    left, right
                ),
            });
        };
        let result = op(*a, *b)?;
        self.set_register(dest, result)
    }

    fn float_binary_op<F>(
        &mut self,
        dest: Register,
        lhs: Register,
        rhs: Register,
        op: F,
    ) -> Result<()>
    where
        F: FnOnce(LustFloat, LustFloat) -> Result<Value>,
    {
        let left = self.get_register(lhs)?;
        let right = self.get_register(rhs)?;
        let (Value::Float(a), Value::Float(b)) = (left, right) else {
            return Err(LustError::RuntimeError {
                message: format!(
                    "Typed numeric instruction expected float operands, got {:?} and {:?}",
                    left, right
                ),
            });
        };
        let result = op(*a, *b)?;
        self.set_register(dest, result)
    }

    pub(super) fn comparison_op<I, F>(
        &mut self,
        dest: Register,
        lhs: Register,
        rhs: Register,
        int_op: I,
        op: F,
    ) -> Result<()>
    where
        I: FnOnce(LustInt, LustInt) -> bool,
        F: FnOnce(LustFloat, LustFloat) -> bool,
    {
        let left = self.get_register(lhs)?;
        let right = self.get_register(rhs)?;
        let result = match (left, right) {
            (Value::Int(a), Value::Int(b)) => int_op(*a, *b),
            (Value::Float(a), Value::Float(b)) => op(*a, *b),
            (Value::Int(a), Value::Float(b)) => op(float_from_int(*a), *b),
            (Value::Float(a), Value::Int(b)) => op(*a, float_from_int(*b)),
            _ => {
                return Err(LustError::RuntimeError {
                    message: format!("Cannot compare {:?} and {:?}", left, right),
                });
            }
        };
        self.set_register(dest, Value::Bool(result))
    }

    pub fn value_is_type(&self, value: &Value, type_name: &str) -> bool {
        // Lua compatibility intentionally treats LuaValue as a dynamic carrier. Transpiled
        // values may still be represented by their underlying VM value at call boundaries.
        if type_name == "LuaValue" || type_name == "unknown" {
            return true;
        }

        let value_type_name = match value {
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::String(_) => "string",
            Value::Bool(_) => "bool",
            Value::Nil => "nil",
            Value::Array(_) => "Array",
            Value::Tuple(_) => "Tuple",
            Value::Map(_) => "Map",
            Value::Struct(object) => object.name.as_str(),
            Value::WeakStruct(weak) => {
                return weak
                    .struct_name()
                    .is_some_and(|name| name.as_str() == type_name);
            }
            Value::Enum(object) => object.enum_name.as_str(),
            Value::Function(_) | Value::NativeFunction(_) | Value::Closure(_) => "function",
            Value::Iterator(_) => "Iterator",
            Value::Task(_) => "Task",
        };
        if value_type_name == type_name || (matches!(value, Value::Nil) && type_name == "()") {
            return true;
        }

        if type_name.contains('|')
            && type_name
                .split('|')
                .map(str::trim)
                .any(|member| member != type_name && self.value_is_type(value, member))
        {
            return true;
        }

        if type_name.starts_with("Array") && matches!(value, Value::Array(_)) {
            return true;
        }

        if type_name.starts_with("Map") && matches!(value, Value::Map(_)) {
            return true;
        }

        if type_name.starts_with("Tuple") && matches!(value, Value::Tuple(_)) {
            return true;
        }

        if (type_name == "Option" || type_name.starts_with("Option<"))
            && matches!(value, Value::Enum(object) if object.enum_name == "Option")
        {
            return true;
        }

        if (type_name == "Result" || type_name.starts_with("Result<"))
            && matches!(value, Value::Enum(object) if object.enum_name == "Result")
        {
            return true;
        }

        if let Some(matches) = self.match_function_type(value, type_name) {
            return matches;
        }

        if !self.trait_impls.is_empty()
            && self
                .trait_impls
                .get(&(value_type_name.to_string(), type_name.to_string()))
                .is_some()
        {
            return true;
        }

        false
    }

    pub(crate) fn value_matches_type(&self, value: &Value, ty: &Type) -> bool {
        match &ty.kind {
            TypeKind::Int => matches!(value, Value::Int(_)),
            TypeKind::Float => matches!(value, Value::Float(_)),
            TypeKind::String => matches!(value, Value::String(_)),
            TypeKind::Bool => matches!(value, Value::Bool(_)),
            TypeKind::Unit => matches!(value, Value::Nil),
            TypeKind::Unknown | TypeKind::Generic(_) | TypeKind::Infer => true,
            TypeKind::Named(expected) => self.value_is_type(value, expected),
            TypeKind::GenericInstance { name, .. } => self.value_is_type(value, name),
            TypeKind::Array(element_type) => match value {
                Value::Array(values) => {
                    if values.is_unboxed() {
                        values.with_unboxed_data(|data| match data {
                            crate::bytecode::value::UnboxedArrayData::Int(v) => {
                                v.is_empty()
                                    || self.value_matches_type(&Value::Int(0), element_type)
                            }
                            crate::bytecode::value::UnboxedArrayData::Float(v) => {
                                v.is_empty()
                                    || self.value_matches_type(&Value::Float(0.0), element_type)
                            }
                            crate::bytecode::value::UnboxedArrayData::Bool(v) => {
                                v.is_empty()
                                    || self.value_matches_type(&Value::Bool(false), element_type)
                            }
                            crate::bytecode::value::UnboxedArrayData::None => true,
                        })
                    } else {
                        values.try_borrow().is_ok_and(|items| {
                            items
                                .iter()
                                .all(|value| self.value_matches_type(value, element_type))
                        })
                    }
                }
                _ => false,
            },
            TypeKind::Map(key_type, value_type) => match value {
                Value::Map(entries) => entries.borrow().iter().all(|(key, value)| {
                    self.value_matches_type(&key.to_value(), key_type)
                        && self.value_matches_type(value, value_type)
                }),
                _ => false,
            },
            TypeKind::Tuple(element_types) => match value {
                Value::Tuple(values) => {
                    values.len() == element_types.len()
                        && values
                            .iter()
                            .zip(element_types)
                            .all(|(value, ty)| self.value_matches_type(value, ty))
                }
                _ => false,
            },
            TypeKind::Option(inner) => match value {
                Value::Enum(object) if object.enum_name == "Option" => {
                    match object.variant.as_str() {
                        "None" => object
                            .values
                            .as_ref()
                            .is_none_or(|values| values.is_empty()),
                        "Some" => object.values.as_ref().is_some_and(|values| {
                            values.len() == 1 && self.value_matches_type(&values[0], inner)
                        }),
                        _ => false,
                    }
                }
                _ => false,
            },
            TypeKind::Result(ok_type, err_type) => match value {
                Value::Enum(object) if object.enum_name == "Result" => {
                    let payload_type = match object.variant.as_str() {
                        "Ok" => ok_type,
                        "Err" => err_type,
                        _ => return false,
                    };
                    object.values.as_ref().is_some_and(|values| {
                        values.len() == 1 && self.value_matches_type(&values[0], payload_type)
                    })
                }
                _ => false,
            },
            TypeKind::Function {
                params,
                return_type,
            } => match value {
                Value::Function(_) | Value::Closure(_) => self
                    .functions
                    .get(match value {
                        Value::Function(index) => *index,
                        Value::Closure(closure) => closure.function_idx,
                        _ => unreachable!(),
                    })
                    .and_then(|function| function.signature.as_ref())
                    .is_some_and(|signature| {
                        signature.params == *params && signature.return_type == **return_type
                    }),
                Value::NativeFunction(_) => true,
                _ => false,
            },
            TypeKind::Union(types) => types.iter().any(|ty| self.value_matches_type(value, ty)),
            TypeKind::Trait(name) => {
                let value_name = self.value_trait_name(value);
                self.trait_impls
                    .contains_key(&(value_name, name.to_string()))
            }
            TypeKind::TraitBound(names) => {
                let value_name = self.value_trait_name(value);
                names.iter().all(|name| {
                    self.trait_impls
                        .contains_key(&(value_name.clone(), name.clone()))
                })
            }
            TypeKind::Ref(inner) | TypeKind::MutRef(inner) => self.value_matches_type(value, inner),
            TypeKind::Pointer { .. } => true,
            TypeKind::Never => false,
            TypeKind::Effect { .. }
            | TypeKind::Typeof(_)
            | TypeKind::InferType(_)
            | TypeKind::InferInput(_)
            | TypeKind::InferOutput(_)
            | TypeKind::Schema { .. } => true,
        }
    }

    /// The tail of `Instruction::Return` once the value is in hand: check
    /// it against the signature, pop the frame and deliver the value to
    /// the caller. Returns the value itself when the call stack is now
    /// empty and `run` should hand it back.
    pub(super) fn finish_return(&mut self, return_value: Value) -> Result<Option<Value>> {
        let frame = self.call_stack.last().unwrap();
        let return_dest = frame.return_dest;
        let is_outer_return = self.call_stack.len() <= 1
            || self.call_until_depth == Some(self.call_stack.len() - 1);
        let skip_return_check = !is_outer_return
            && self
                .call_meta
                .get(frame.function_idx)
                .is_some_and(|meta| meta.trusted_signature);
        if !skip_return_check {
            self.validate_function_return(frame.function_idx, &return_value)?;
        }
        #[cfg(feature = "std")]
        if crate::vm::tracing::has_trace_hook() {
            let func = &self.functions[frame.function_idx];
            let ip_index = frame.ip.saturating_sub(1);
            let line = func.chunk.lines.get(ip_index).copied().unwrap_or(0);
            self.dispatch_return_hook(&func.name, line, self.call_stack.len());
        }
        let _ = self.pop_call_frame();
        if self.call_stack.is_empty() {
            return Ok(Some(return_value));
        }

        // Deliver straight into the caller unless this return ends a host
        // call into the VM, which `run` hands back from the top of the
        // loop.
        if self.call_until_depth == Some(self.call_stack.len()) {
            self.pending_return_value = Some(return_value);
            self.pending_return_dest = return_dest;
            self.sync_slow_path_flags();
        } else if let Some(dest) = return_dest {
            self.set_register(dest, return_value)?;
        }
        Ok(None)
    }

    /// Run the frame just pushed for `func_idx` through the function's
    /// compiled code, compiling it first when it has just become hot (see
    /// `jit::function`). Does nothing when there is no code. Afterwards
    /// the call stack is wherever the native code left it: the frame
    /// popped (it returned; `Some` when that emptied the stack), or one or
    /// more frames — native callees materialized on exit included —
    /// positioned at the instruction the interpreter continues from.
    ///
    /// Only the backends that implement `JitCompiler::compile_function`
    /// (x86_64 and aarch64, with `std`) have function code; elsewhere the
    /// interpreter runs every call itself and loops are still traced.
    #[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub(super) fn run_compiled_function(&mut self, func_idx: usize) -> Result<Option<Value>> {
        if !self.effective_jit_enabled() {
            return Ok(None);
        }
        let code = match self.jit.function_code(func_idx) {
            Some(code) => code,
            None => {
                if !self.jit.record_function_entry(func_idx) {
                    return Ok(None);
                }
                self.compile_function(func_idx);
                match self.jit.function_code(func_idx) {
                    Some(code) => code,
                    None => return Ok(None),
                }
            }
        };

        // Native-to-native calls grow the machine stack; give them at most
        // `NATIVE_STACK_RESERVE` below here before they hand calls to the
        // interpreter, whose frames live on the heap.
        // And no deeper than the interpreter's frame depth limit allows:
        // each native call level takes at least `MIN_NATIVE_FRAME` bytes.
        let marker = 0u8;
        let sp = &marker as *const u8 as usize;
        let depth_budget = self.max_stack_depth.saturating_sub(self.call_stack.len());
        let limit = sp
            .saturating_sub(crate::jit::NATIVE_STACK_RESERVE)
            .max(sp.saturating_sub(depth_budget.saturating_mul(crate::jit::MIN_NATIVE_FRAME)));
        let outer_limit = core::mem::replace(&mut self.jit_cells.stack_limit, limit);

        let cost = code.trace.ops.len();
        self.budgets.charge_gas(core::cmp::max(1, cost) as u64)?;
        self.publish_gas_to_cells();
        self.jit.record_native_entry();
        self.pending_jit_error = None;
        self.jit_cells.exit_info = usize::MAX;
        let (base_reg, reg_count) = {
            let f = self.call_stack.last().unwrap();
            (f.base_reg, f.reg_count as usize)
        };
        let registers_ptr = self.vm_registers[base_reg..base_reg + reg_count].as_mut_ptr();
        let vm_ptr = self as *mut VM;
        let _jit_guard = self.enter_jit_scope();
        let result = code.execute(registers_ptr, vm_ptr, ptr::null());
        drop(_jit_guard);
        if let Some(frame) = self.call_stack.last_mut() {
            frame.non_plain_mask = u64::MAX;
        }
        self.jit_cells.stack_limit = outer_limit;
        self.sync_gas_from_cells();
        crate::jit::log(|| format!("🎯 JIT: function {} native result {}", func_idx, result));
        drop(code);

        if (crate::jit::FUNCTION_RETURN_BASE..crate::jit::NATIVE_RETURNED).contains(&result) {
            let reg = (result - crate::jit::FUNCTION_RETURN_BASE) as usize;
            let return_value = if reg == 255 {
                Value::Nil
            } else {
                let top = self.call_stack.last().unwrap();
                self.vm_registers[top.base_reg + reg].fast_clone()
            };
            return self.finish_return(return_value);
        }

        // An exit, from this function or a native callee whose frames are
        // now on the call stack: resume the innermost frame where the
        // exiting site said.
        let info = self.jit_cells.exit_info;
        if info == usize::MAX {
            // Every exit stub of function code records its site; an exit
            // without one is a compiler bug. Evict the code and resume at
            // the function's entry, which at least keeps the interpreter
            // consistent.
            debug_assert!(
                false,
                "function {func_idx} exited with {result} and no exit info"
            );
            crate::jit::log(|| {
                format!("❌ JIT: function {func_idx} exited with {result} and no exit info")
            });
            self.jit.evict_function_code(func_idx);
            if let Some(frame) = self.call_stack.last_mut() {
                frame.ip = 0;
            }
            return Ok(None);
        }
        let ip = info & ((1usize << crate::jit::EXIT_KIND_SHIFT) - 1);
        let kind = info >> crate::jit::EXIT_KIND_SHIFT;
        if let Some(error) = self.pending_jit_error.take() {
            return Err(error);
        }
        let frame = self.call_stack.last_mut().unwrap();
        let exited_idx = frame.function_idx;
        frame.ip = ip;
        crate::jit::log(|| {
            format!(
                "↩️  JIT: function {} exit kind {} → func {} ip {}",
                func_idx, kind, exited_idx, ip
            )
        });
        match kind {
            crate::jit::EXIT_KIND_HANDOFF | crate::jit::EXIT_KIND_FAIL => {}
            _ => {
                self.jit.record_guard_exit();
                self.jit.evict_function_code(exited_idx);
            }
        }
        Ok(None)
    }

    /// Translate and compile `func_idx` (see `jit::function`), or mark it
    /// as not compilable.
    #[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn compile_function(&mut self, func_idx: usize) {
        if !self.effective_jit_enabled() {
            return;
        }
        use crate::jit::function::{Context, FunctionSig, translate};
        let sig_of = |meta: &CallMeta, function: &Function| FunctionSig {
            params: meta.params.iter().map(|kind| kind.value_type()).collect(),
            ret: meta.return_kind.value_type(),
            lua_function: meta.lua_function,
            register_count: function.register_count,
            param_types: match &function.signature {
                Some(signature) if signature.params.len() == function.param_count as usize => {
                    signature
                        .params
                        .iter()
                        .map(|ty| Some(ty.kind.clone()))
                        .collect()
                }
                _ => alloc::vec![None; function.param_count as usize],
            },
            written_registers: crate::jit::function::written_registers(function),
            returned_registers: crate::jit::function::returned_registers(function),
        };
        let Some((function, meta)) = self
            .functions
            .get(func_idx)
            .zip(self.call_meta.get(func_idx))
        else {
            return;
        };
        let sig = sig_of(meta, function);
        let functions = &self.functions;
        let call_meta = &self.call_meta;
        let callee_sig = |idx: usize| {
            functions
                .get(idx)
                .zip(call_meta.get(idx))
                .map(|(function, meta)| sig_of(meta, function))
        };
        let struct_metadata = &self.struct_metadata;
        let layout_of = |name: &str| struct_metadata.get(name).map(|info| info.layout.clone());
        let function_named = |name: &str| functions.iter().position(|f| f.name == name);
        let globals = &self.globals;
        let natives = &self.natives;
        let global = |name: &str| globals.get(name).or_else(|| natives.get(name)).cloned();
        let field_pure = |idx: usize| crate::jit::function::field_pure(functions, idx);
        let ctx = Context {
            callee_sig: &callee_sig,
            layout_of: &layout_of,
            function_named: &function_named,
            global: &global,
            globals_version: self.globals_version,
            intrinsics: &self.jit.intrinsics,
            field_pure: &field_pure,
        };
        let Some(trace) = translate(function, func_idx, &sig, &ctx) else {
            crate::jit::log(|| format!("🚫 JIT: function {} is not compilable", func_idx));
            self.jit.function_not_compilable(func_idx);
            return;
        };
        let trace_id = self.jit.alloc_trace_id();
        let register_count = function.register_count;
        let entry_table = self.jit.function_entry_table();
        let result_type = sig.ret.filter(|ty| {
            matches!(
                ty,
                crate::jit::trace::ValueType::Int
                    | crate::jit::trace::ValueType::Float
                    | crate::jit::trace::ValueType::Bool
            )
        });
        match JitCompiler::new()
            .with_gas_checks(self.gas_checked())
            .compile_function(
                &trace,
                trace_id,
                register_count,
                entry_table,
                result_type,
                Vec::new(),
            )
        {
            Ok(code) => {
                crate::jit::log(|| format!("✅ JIT: function {} compiled", func_idx));
                self.jit.store_function_code(func_idx, code);
            }
            Err(e) => {
                crate::jit::log(|| format!("❌ JIT: function {} compile failed: {}", func_idx, e));
                self.jit.function_not_compilable(func_idx);
            }
        }
    }

    /// Targets without a whole-function backend: the call stays
    /// interpreted (see the compiled version above).
    #[cfg(not(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64"))))]
    pub(super) fn run_compiled_function(&mut self, _func_idx: usize) -> Result<Option<Value>> {
        Ok(None)
    }

    /// Is the callee register a plain bytecode function (not a closure,
    /// native, Lua value or callable table) whose call can take the fast
    /// path?
    #[inline]
    fn plain_function_callee(&self, func_reg: Register) -> bool {
        let Some(frame) = self.call_stack.last() else {
            return false;
        };
        let slot = frame.base_reg + func_reg as usize;
        match self.vm_registers[slot] {
            Value::Function(idx) => self
                .call_meta
                .get(idx)
                .is_some_and(|meta| !meta.lua_function),
            _ => false,
        }
    }

    /// The fast path of `Instruction::Call` for a plain bytecode callee:
    /// same checks as `push_call_frame_with` with `ArgCheck::Shallow`, but
    /// the arguments are copied straight from the caller's register window into
    /// the new frame window in `vm_registers`, with no intermediate buffer and no
    /// per-call signature walk. `plain_function_callee` must have said yes.
    fn try_push_dispatch_frame(
        &mut self,
        function_idx: usize,
        reg_count: u16,
        return_dest: Option<Register>,
        upvalues: Vec<Value>,
        shared_upvalues: Vec<Upvalue>,
    ) -> Result<usize> {
        if self.call_stack.len() >= self.max_stack_depth {
            return Err(LustError::RuntimeError {
                message: "Stack overflow".to_string(),
            });
        }
        let base = match self.call_stack.last() {
            Some(parent) => register_window_end(parent.base_reg, parent.reg_count)?,
            None => 0,
        };
        let end = register_window_end(base, reg_count)?;
        reserve_register_window(&mut self.vm_registers, &mut self.budgets, end)?;
        // The shared push updates call metadata and has enough storage
        // already. Set the padded base before initializing any live slots.
        let idx =
            self.try_push_call_frame(function_idx, reg_count, return_dest, upvalues, shared_upvalues)?;
        self.call_stack[idx].base_reg = base;
        Ok(idx)
    }

    fn push_bytecode_call_frame(
        &mut self,
        func_reg: Register,
        first_arg: Register,
        arg_count: u8,
        dest_reg: Register,
    ) -> Result<usize> {
        let caller = self.call_stack.len() - 1;
        let caller_base = self.call_stack[caller].base_reg;
        let Value::Function(function_idx) = self.vm_registers[caller_base + func_reg as usize]
        else {
            unreachable!("plain_function_callee checked the callee");
        };
        let function = &self.functions[function_idx];
        if arg_count != function.param_count {
            return Err(LustError::RuntimeError {
                message: format!(
                    "Function {} expects {} arguments, got {}",
                    function.name, function.param_count, arg_count
                ),
            });
        }
        // Validate all argument types before pushing the new window (fail-closed, zero mutation).
        // Skip redundant shallow re-checks for statically verified `trusted_signature` callees.
        if !self.call_meta[function_idx].trusted_signature {
            for index in 0..arg_count as usize {
                let value =
                    &self.vm_registers[caller_base + first_arg.wrapping_add(index as u8) as usize];
                let expected = self.call_meta[function_idx].params[index];
                if !expected.matches(value) {
                    let function = &self.functions[function_idx];
                    let ty = function
                        .signature
                        .as_ref()
                        .map(|signature| signature.params[index].to_string())
                        .unwrap_or_default();
                    let message = format!(
                        "Function {} argument {} expects {}, got {:?}",
                        function.name,
                        index + 1,
                        ty,
                        value.type_of()
                    );
                    return Err(LustError::RuntimeError { message });
                }
            }
        }
        let reg_count = self.effective_reg_count(function_idx);
        let callee_frame_idx = self.try_push_dispatch_frame(
            function_idx,
            reg_count,
            Some(dest_reg),
            Vec::new(),
            Vec::new(),
        )?;
        let callee_base = self.call_stack[callee_frame_idx].base_reg;
        for index in 0..arg_count as usize {
            let src_idx = caller_base + first_arg.wrapping_add(index as u8) as usize;
            let value = self.vm_registers[src_idx].fast_clone();
            if !value.is_plain() {
                self.call_stack[callee_frame_idx].mark_non_plain(index);
            }
            self.vm_registers[callee_base + index] = value;
        }
        if self.jit.enabled {
            let recursive = self
                .call_meta
                .get(function_idx)
                .is_some_and(|meta| meta.active_depth > 1);
            self.jit.record_function_call(recursive);
        }
        Ok(function_idx)
    }

    pub(super) fn push_checked_call_frame(
        &mut self,
        function_idx: usize,
        return_dest: Option<Register>,
        args: Vec<Value>,
        upvalues: Vec<Value>,
        shared_upvalues: Vec<Upvalue>,
    ) -> Result<()> {
        self.push_call_frame_with(
            function_idx,
            return_dest,
            args,
            upvalues,
            shared_upvalues,
            ArgCheck::Shallow,
        )
    }

    /// Push a frame window for a call from the host or a dynamic value: every
    /// argument is validated in full against the signature, including the
    /// elements of containers.
    pub(super) fn push_deep_call_frame(
        &mut self,
        function_idx: usize,
        return_dest: Option<Register>,
        args: Vec<Value>,
        upvalues: Vec<Value>,
        shared_upvalues: Vec<Upvalue>,
    ) -> Result<()> {
        self.push_call_frame_with(
            function_idx,
            return_dest,
            args,
            upvalues,
            shared_upvalues,
            ArgCheck::Deep,
        )
    }

    pub(super) fn push_call_frame_slice(
        &mut self,
        function_idx: usize,
        return_dest: Option<Register>,
        args: &[Value],
        check: ArgCheck,
    ) -> Result<()> {
        let function = self
            .functions
            .get(function_idx)
            .ok_or_else(|| LustError::RuntimeError {
                message: format!("Invalid function index {}", function_idx),
            })?;
        if args.len() != function.param_count as usize {
            return Err(LustError::RuntimeError {
                message: format!(
                    "Function {} expects {} arguments, got {}",
                    function.name,
                    function.param_count,
                    args.len()
                ),
            });
        }
        let skip_shallow = matches!(check, ArgCheck::Shallow)
            && self
                .call_meta
                .get(function_idx)
                .is_some_and(|meta| meta.trusted_signature);
        if !skip_shallow
            && let Some(signature) = &function.signature
            && signature.params.len() == args.len()
        {
            for (index, (value, ty)) in args.iter().zip(&signature.params).enumerate() {
                let ok = match check {
                    ArgCheck::Deep => self.value_matches_type(value, ty),
                    ArgCheck::Shallow => Self::value_matches_type_shallow(value, ty),
                };
                if !ok {
                    if matches!(value, Value::Array(arr) if arr.try_borrow().is_err()) {
                        return Err(LustError::RuntimeError {
                            message: format!(
                                "Failed to acquire borrow for argument {} of function {}: mutable borrow active",
                                index + 1,
                                function.name
                            ),
                        });
                    }
                    return Err(LustError::RuntimeError {
                        message: format!(
                            "Function {} argument {} expects {}, got {:?}",
                            function.name,
                            index + 1,
                            ty,
                            value.type_of()
                        ),
                    });
                }
            }
        }

        let reg_count = self.effective_reg_count(function_idx);
        let frame_idx = self.try_push_dispatch_frame(
            function_idx,
            reg_count,
            return_dest,
            Vec::new(),
            Vec::new(),
        )?;
        if self.jit.enabled {
            let recursive = self
                .call_meta
                .get(function_idx)
                .is_some_and(|meta| meta.active_depth > 1);
            self.jit.record_function_call(recursive);
        }
        let base_reg = self.call_stack[frame_idx].base_reg;
        for (index, arg) in args.iter().enumerate() {
            self.observe_value(arg);
            let cloned = arg.fast_clone();
            if !cloned.is_plain() {
                self.call_stack[frame_idx].mark_non_plain(index);
            }
            self.vm_registers[base_reg + index] = cloned;
        }
        Ok(())
    }

    fn validate_call_args_vec(
        &self,
        function_idx: usize,
        args: &mut Vec<Value>,
        check: ArgCheck,
    ) -> Result<()> {
        let function = self
            .functions
            .get(function_idx)
            .ok_or_else(|| LustError::RuntimeError {
                message: format!("Invalid function index {}", function_idx),
            })?;
        let is_lua_function = function.signature.as_ref().is_some_and(|signature| {
            let lua_params = signature
                .params
                .iter()
                .all(|ty| matches!(&ty.kind, TypeKind::Named(name) if name == "LuaValue"));
            let lua_multi_return = matches!(
                &signature.return_type.kind,
                TypeKind::Array(inner)
                    if matches!(&inner.kind, TypeKind::Named(name) if name == "LuaValue")
            );
            lua_params && (!signature.params.is_empty() || lua_multi_return)
        });
        if is_lua_function {
            args.resize(function.param_count as usize, Value::Nil);
            args.truncate(function.param_count as usize);
        }
        if args.len() != function.param_count as usize {
            return Err(LustError::RuntimeError {
                message: format!(
                    "Function {} expects {} arguments, got {}",
                    function.name,
                    function.param_count,
                    args.len()
                ),
            });
        }
        let skip_shallow = matches!(check, ArgCheck::Shallow)
            && self
                .call_meta
                .get(function_idx)
                .is_some_and(|meta| meta.trusted_signature);
        if !skip_shallow
            && let Some(signature) = &function.signature
            && signature.params.len() == args.len()
        {
            for (index, (value, ty)) in args.iter().zip(&signature.params).enumerate() {
                let ok = match check {
                    ArgCheck::Deep => self.value_matches_type(value, ty),
                    ArgCheck::Shallow => Self::value_matches_type_shallow(value, ty),
                };
                if !ok {
                    if matches!(value, Value::Array(arr) if arr.try_borrow().is_err()) {
                        return Err(LustError::RuntimeError {
                            message: format!(
                                "Failed to acquire borrow for argument {} of function {}: mutable borrow active",
                                index + 1,
                                function.name
                            ),
                        });
                    }
                    return Err(LustError::RuntimeError {
                        message: format!(
                            "Function {} argument {} expects {}, got {:?}",
                            function.name,
                            index + 1,
                            ty,
                            value.type_of()
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    fn push_call_frame_with(
        &mut self,
        function_idx: usize,
        return_dest: Option<Register>,
        mut args: Vec<Value>,
        upvalues: Vec<Value>,
        shared_upvalues: Vec<Upvalue>,
        check: ArgCheck,
    ) -> Result<()> {
        self.validate_call_args_vec(function_idx, &mut args, check)?;
        let reg_count = self.effective_reg_count(function_idx);
        let frame_idx = self.try_push_dispatch_frame(
            function_idx,
            reg_count,
            return_dest,
            upvalues,
            shared_upvalues,
        )?;
        if self.jit.enabled {
            let recursive = self
                .call_meta
                .get(function_idx)
                .is_some_and(|meta| meta.active_depth > 1);
            self.jit.record_function_call(recursive);
        }
        let base_reg = self.call_stack[frame_idx].base_reg;
        for (index, arg) in args.drain(..).enumerate() {
            self.observe_value(&arg);
            if !arg.is_plain() {
                self.call_stack[frame_idx].mark_non_plain(index);
            }
            self.vm_registers[base_reg + index] = arg;
        }
        // Keep the (now empty) argument buffer so the next call allocates
        // nothing.
        if args.capacity() > self.arg_scratch.capacity() {
            self.arg_scratch = args;
        }
        Ok(())
    }

    /// Creates a detached `Box<CallFrame>` carrying its initial arguments in `saved_registers`
    /// for a newly created fiber task (does not mutate the active VM call stack or `vm_registers`).
    pub(super) fn make_detached_task_frame(
        &mut self,
        function_idx: usize,
        mut args: Vec<Value>,
        upvalues: Vec<Value>,
        shared_upvalues: Vec<Upvalue>,
    ) -> Result<Box<CallFrame>> {
        self.validate_call_args_vec(function_idx, &mut args, ArgCheck::Deep)?;
        let reg_count = self.effective_reg_count(function_idx);
        let mut regs = alloc::vec![Value::Nil; reg_count as usize];
        let mut frame = CallFrame::new(function_idx, None, reg_count);
        frame.upvalues = upvalues;
        frame.shared_upvalues = shared_upvalues;
        for (index, arg) in args.drain(..).enumerate() {
            self.observe_value(&arg);
            if !arg.is_plain() {
                frame.mark_non_plain(index);
            }
            regs[index] = arg;
        }
        frame.saved_registers = Some(Box::new(regs));
        if args.capacity() > self.arg_scratch.capacity() {
            self.arg_scratch = args;
        }
        Ok(Box::new(frame))
    }

    /// An O(1) argument check for calls the typechecker already validated:
    /// scalar kinds and container kinds are verified, contents are not.
    /// Anything the checker treats dynamically (`unknown`, generics,
    /// unions, function types, Lua values) is accepted; typed instructions
    /// still guard the payloads they read.
    fn value_matches_type_shallow(value: &Value, ty: &Type) -> bool {
        match &ty.kind {
            TypeKind::Int => matches!(value, Value::Int(_)),
            TypeKind::Float => matches!(value, Value::Float(_)),
            TypeKind::String => matches!(value, Value::String(_)),
            TypeKind::Bool => matches!(value, Value::Bool(_)),
            TypeKind::Unit => matches!(value, Value::Nil),
            TypeKind::Array(_) => matches!(value, Value::Array(_)),
            TypeKind::Map(..) => matches!(value, Value::Map(_)),
            TypeKind::Tuple(_) => matches!(value, Value::Tuple(_)),
            _ => true,
        }
    }

    fn validate_function_return(&self, function_idx: usize, value: &Value) -> Result<()> {
        // A return executes bytecode the typechecker validated; check the
        // kind only (precomputed in `CallMeta`). Container contents are
        // checked where they are read, by typed instructions.
        let meta = &self.call_meta[function_idx];
        if meta.return_kind.matches(value) || (meta.lua_multi_return && matches!(value, Value::Nil))
        {
            return Ok(());
        }
        let function = &self.functions[function_idx];
        Err(LustError::RuntimeError {
            message: format!(
                "Function {} must return {}, got {:?}",
                function.name,
                function
                    .signature
                    .as_ref()
                    .map(|signature| signature.return_type.to_string())
                    .unwrap_or_default(),
                value.type_of()
            ),
        })
    }

    fn value_trait_name(&self, value: &Value) -> String {
        match value {
            Value::Int(_) => "int".to_string(),
            Value::Float(_) => "float".to_string(),
            Value::String(_) => "string".to_string(),
            Value::Bool(_) => "bool".to_string(),
            Value::Nil => "nil".to_string(),
            Value::Array(_) => "Array".to_string(),
            Value::Tuple(_) => "Tuple".to_string(),
            Value::Map(_) => "Map".to_string(),
            Value::Struct(object) => object.name.to_string(),
            Value::WeakStruct(weak) => weak
                .struct_name()
                .map(|name| name.to_string())
                .unwrap_or_else(|| "struct".to_string()),
            Value::Enum(object) => object.enum_name.to_string(),
            Value::Function(_) | Value::NativeFunction(_) | Value::Closure(_) => {
                "function".to_string()
            }
            Value::Iterator(_) => "Iterator".to_string(),
            Value::Task(_) => "task".to_string(),
        }
    }

    fn invoke_hashkey(&mut self, value: &Value, type_name: &str) -> Result<Value> {
        let mut candidates = vec![format!("{}:{}", type_name, HASH_KEY_METHOD)];
        if let Some(last) = type_name.rsplit('.').next()
            && last != type_name
        {
            candidates.push(format!("{}:{}", last, HASH_KEY_METHOD));
        }

        for candidate in candidates {
            if let Some(idx) = self.functions.iter().position(|f| f.name == candidate) {
                return self.call_value(&Value::Function(idx), vec![value.clone()]);
            }
        }

        Err(LustError::RuntimeError {
            message: format!(
                "HashKey trait declared but method '{}' not found for type '{}'",
                HASH_KEY_METHOD, type_name
            ),
        })
    }

    pub fn make_hash_key(&mut self, value: &Value) -> Result<ValueKey> {
        let type_name = self.value_trait_name(value);
        if self.type_has_hashkey(&type_name) {
            let hashed = self.invoke_hashkey(value, &type_name)?;
            Ok(ValueKey::with_hashed(value.clone(), hashed))
        } else {
            Ok(ValueKey::from_value(value))
        }
    }

    fn match_function_type(&self, value: &Value, type_name: &str) -> Option<bool> {
        let wants_signature = type_name.starts_with("function(");
        let wants_generic = type_name == "function";
        if !wants_signature && !wants_generic {
            return None;
        }

        let matches = match value {
            Value::Function(idx) => self.function_signature_matches(*idx, type_name),
            Value::Closure(closure) => {
                let ClosureObject { function_idx, .. } = closure.as_ref();
                self.function_signature_matches(*function_idx, type_name)
            }
            Value::NativeFunction(_) => wants_generic,
            _ => false,
        };
        Some(matches)
    }

    fn function_signature_matches(&self, func_idx: usize, type_name: &str) -> bool {
        if type_name == "function" {
            return true;
        }

        let Some(signature) = self
            .functions
            .get(func_idx)
            .and_then(|func| func.signature.as_ref())
        else {
            return false;
        };

        let rendered = signature.to_string();
        if rendered == type_name {
            return true;
        }

        let (Some(rendered_params), Some(expected_params)) = (
            function_type_params(&rendered),
            function_type_params(type_name),
        ) else {
            return false;
        };
        if rendered_params != expected_params {
            return false;
        }

        match &signature.return_type.kind {
            TypeKind::Unit => !type_string_declares_return(type_name),
            _ => false,
        }
    }

    pub(super) fn get_register(&self, reg: Register) -> Result<&Value> {
        let frame = self
            .call_stack
            .last()
            .ok_or_else(|| LustError::RuntimeError {
                message: "Empty call stack".to_string(),
            })?;
        let idx = reg as usize;
        if idx >= frame.reg_count as usize {
            return Err(LustError::RuntimeError {
                message: format!(
                    "Register {} out of bounds (window len {})",
                    idx, frame.reg_count
                ),
            });
        }
        Ok(&self.vm_registers[frame.base_reg + idx])
    }

    // Inlining lets scalar instructions write their tag/payload directly,
    // without materializing a temporary Value for an out-of-line call.
    #[inline(always)]
    pub(super) fn set_register(&mut self, reg: Register, value: Value) -> Result<()> {
        let is_non_plain = !value.is_plain();
        if is_non_plain && self.cycle_collector.is_enabled() {
            self.cycle_collector.register_value(&value);
        }
        let frame = self
            .call_stack
            .last_mut()
            .ok_or_else(|| LustError::RuntimeError {
                message: "Empty call stack".to_string(),
            })?;
        let idx = reg as usize;
        if idx >= frame.reg_count as usize {
            return Err(LustError::RuntimeError {
                message: format!(
                    "Register {} out of bounds (window len {})",
                    idx, frame.reg_count
                ),
            });
        }
        if is_non_plain {
            frame.mark_non_plain(idx);
        }
        let slot = frame.base_reg + idx;
        self.vm_registers[slot] = value;
        if self.cycle_collector.is_enabled() {
            self.maybe_collect_cycles();
        }
        Ok(())
    }

    pub(super) fn handle_native_call_outcome(
        &mut self,
        dest: Register,
        outcome: NativeCallResult,
    ) -> Result<()> {
        #[cfg(feature = "std")]
        if lua_socket_trace_enabled()
            && let NativeCallResult::Return(value) = &outcome
            && let Value::Array(arr) = value
        {
            let borrowed = arr.borrow();
            let interesting = borrowed.len() > 1
                && matches!(
                    borrowed.first(),
                    Some(Value::Enum(object))
                        if object.enum_name == "LuaValue" && object.variant == "Nil"
                );
            if interesting {
                let func_name = self
                    .call_stack
                    .last()
                    .and_then(|frame| self.functions.get(frame.function_idx))
                    .map(|f| f.name.as_str())
                    .unwrap_or("<unknown>");
                eprintln!(
                    "[lua-socket] native return in {} dest=R{} len={} value={}",
                    func_name,
                    dest,
                    borrowed.len(),
                    value
                );
            }
        }
        match outcome {
            NativeCallResult::Return(value) => self.set_register(dest, value),
            NativeCallResult::Yield(value) => {
                if self.current_task.is_some() {
                    self.set_register(dest, Value::Nil)?;
                    self.pending_task_signal = Some(TaskSignal::Yield { dest, value });
                    Ok(())
                } else {
                    Err(LustError::RuntimeError {
                        message: "task.yield() can only be used inside a task".to_string(),
                    })
                }
            }

            NativeCallResult::YieldTimer(duration, value) => {
                if self.clock.is_simulated() && self.current_task.is_some() {
                    let duration_nanos = duration.as_nanos().min(u64::MAX as u128) as u64;
                    let now = self.clock.now_nanos();
                    let deadline = now.saturating_add(duration_nanos);
                    let timer_id = self.clock.schedule_timer(deadline, 0, move || {});
                    self.set_register(dest, Value::Nil)?;
                    self.pending_task_signal = Some(TaskSignal::YieldTimer {
                        dest,
                        value,
                        timer_id,
                    });
                    Ok(())
                } else {
                    #[cfg(feature = "std")]
                    std::thread::sleep(duration);
                    self.set_register(dest, value)
                }
            }

            NativeCallResult::Stop(value) => {
                if self.current_task.is_some() {
                    self.set_register(dest, Value::Nil)?;
                    self.pending_task_signal = Some(TaskSignal::Stop { value });
                    Ok(())
                } else {
                    Err(LustError::RuntimeError {
                        message: "task.stop() can only be used inside a task".to_string(),
                    })
                }
            }
        }
    }

    pub fn value_to_string_for_concat(&mut self, value: &Value) -> Result<Rc<String>> {
        match value {
            Value::String(s) => Ok(s.clone()),
            Value::Struct(object) => {
                let name = object.name.clone();
                self.invoke_tostring(value, &name)
            }
            Value::Enum(object) => {
                let name = object.enum_name.clone();
                self.invoke_tostring(value, &name)
            }
            _ => Ok(Rc::new(value.to_string())),
        }
    }

    #[inline(never)]
    pub fn call_value(&mut self, func: &Value, args: Vec<Value>) -> Result<Value> {
        let _vm_guard = crate::vm::CurrentVmGuard::from_vm(self);
        // Lua compatibility: honor __call metamethod on Lua tables/userdata.
        // Lua semantics: if value has metatable.__call, calling it invokes that function with the
        // receiver as the first argument.
        let mut args = args;
        let maybe_lua_call = {
            let mut current = func;
            let mut receiver_wrapped = func.clone();

            if let Value::Struct(object) = func {
                let name = &object.name;
                if name == "LuaTable" {
                    receiver_wrapped = Value::enum_variant("LuaValue", "Table", vec![func.clone()]);
                } else if name == "LuaUserdata" {
                    receiver_wrapped =
                        Value::enum_variant("LuaValue", "Userdata", vec![func.clone()]);
                }
            }

            if let Value::Enum(object) = func
                && object.enum_name == "LuaValue"
                && (object.variant == "Table" || object.variant == "Userdata")
                && let Some(inner) = object.values.as_ref().and_then(|vals| vals.first())
            {
                current = inner;
            }

            if let Value::Struct(object) = current {
                let name = &object.name;
                if name == "LuaTable" || name == "LuaUserdata" {
                    if let Some(Value::Map(meta_rc)) = current.struct_get_field("metamethods") {
                        meta_rc
                            .borrow()
                            .get(&ValueKey::string("__call".to_string()))
                            .cloned()
                            .map(|callable| (callable, receiver_wrapped))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        };

        if let Some((callable, receiver)) = maybe_lua_call {
            // eprintln!("DEBUG: __call metamethod found, invoking with receiver");
            let mut call_args = Vec::with_capacity(args.len() + 1);
            call_args.push(receiver);
            call_args.append(&mut args);
            return self.call_value(&callable, call_args);
        }

        if let Some(intrinsic_name) = Self::get_effect_intrinsic_name(func) {
            return self.dispatch_effect_intrinsic(&intrinsic_name, args);
        }

        match func {
            #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
            Value::Enum(object)
                if object.enum_name == "LuaValue" && object.variant == "Function" =>
            {
                let handle = object
                    .values
                    .as_ref()
                    .and_then(|vals| vals.first())
                    .and_then(|v| v.struct_get_field("handle"))
                    .and_then(|v| v.as_int())
                    .map(|i| i as usize)
                    .ok_or_else(|| LustError::RuntimeError {
                        message: "LuaValue function missing handle".to_string(),
                    })?;
                let inner = crate::lua_compat::lookup_lust_function(handle).ok_or_else(|| {
                    LustError::RuntimeError {
                        message: format!(
                            "LuaValue function handle {} was not registered with VM",
                            handle
                        ),
                    }
                })?;
                self.call_value(&inner, args)
            }
            Value::Function(func_idx) => {
                let saved_pending_return_value = self.pending_return_value.clone();
                let saved_pending_return_dest = self.pending_return_dest;
                let saved_pending_task_signal = self.pending_task_signal.clone();
                let saved_last_task_signal = self.last_task_signal.clone();

                let stack_depth_before = self.call_stack.len();
                self.push_deep_call_frame(*func_idx, None, args, Vec::new(), Vec::new())?;
                let previous_target = self.call_until_depth;
                self.call_until_depth = Some(stack_depth_before);
                // Compiled code for the function runs it now; `run` then
                // either hands back the pending return value or carries on
                // interpreting from wherever the native code exited.
                let native = if self.effective_jit_enabled() && self.trace_recorder.is_none() {
                    self.run_compiled_function(*func_idx).map(|_| ())
                } else {
                    Ok(())
                };
                let run_result = native.and_then(|()| self.run());
                self.call_until_depth = previous_target;
                match run_result {
                    Ok(value) => Ok(value),
                    Err(err) => {
                        let annotated = self.annotate_runtime_error(err);
                        self.truncate_call_stack(stack_depth_before);
                        self.pending_return_value = saved_pending_return_value;
                        self.pending_return_dest = saved_pending_return_dest;
                        self.pending_task_signal = saved_pending_task_signal;
                        self.last_task_signal = saved_last_task_signal;
                        self.sync_slow_path_flags();
                        Err(annotated)
                    }
                }
            }

            Value::Closure(closure) => {
                let ClosureObject {
                    function_idx: func_idx,
                    upvalues,
                } = closure.as_ref();
                let saved_pending_return_value = self.pending_return_value.clone();
                let saved_pending_return_dest = self.pending_return_dest;
                let saved_pending_task_signal = self.pending_task_signal.clone();
                let saved_last_task_signal = self.last_task_signal.clone();

                let upvalue_values: Vec<Value> = upvalues.iter().map(|uv| uv.get()).collect();
                let stack_depth_before = self.call_stack.len();
                self.push_deep_call_frame(
                    *func_idx,
                    None,
                    args,
                    upvalue_values,
                    upvalues.clone(),
                )?;
                let previous_target = self.call_until_depth;
                self.call_until_depth = Some(stack_depth_before);
                let run_result = self.run();
                self.call_until_depth = previous_target;
                match run_result {
                    Ok(value) => Ok(value),
                    Err(err) => {
                        let annotated = self.annotate_runtime_error(err);
                        self.truncate_call_stack(stack_depth_before);
                        self.pending_return_value = saved_pending_return_value;
                        self.pending_return_dest = saved_pending_return_dest;
                        self.pending_task_signal = saved_pending_task_signal;
                        self.last_task_signal = saved_last_task_signal;
                        self.sync_slow_path_flags();
                        Err(annotated)
                    }
                }
            }

            Value::NativeFunction(native_fn) => {
                let _native_guard = self.enter_native_scope();
                let _vm_guard = crate::vm::CurrentVmGuard::from_vm(self);
                let outcome = native_fn(&args);
                drop(_vm_guard);
                drop(_native_guard);
                let outcome = outcome.map_err(|e| LustError::RuntimeError { message: e })?;
                match outcome {
                    NativeCallResult::Return(value) => Ok(value),
                    NativeCallResult::YieldTimer(duration, value) => {
                        #[cfg(feature = "std")]
                        std::thread::sleep(duration);
                        Ok(value)
                    }
                    NativeCallResult::Yield(_) | NativeCallResult::Stop(_) => {
                        Err(LustError::RuntimeError {
                            message: "Yielding or stopping is not allowed from this context"
                                .to_string(),
                        })
                    }
                }
            }

            _ => Err(LustError::RuntimeError {
                message: format!("Cannot call non-function value: {:?}", func),
            }),
        }
    }
}

fn type_string_declares_return(type_name: &str) -> bool {
    if !type_name.starts_with("function(") {
        return true;
    }
    let bytes = type_name.as_bytes();
    let mut depth = 0usize;
    for (index, byte) in bytes.iter().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return bytes[index + 1..].first() == Some(&b':');
                }
            }
            _ => {}
        }
    }
    true
}

fn function_type_params(type_name: &str) -> Option<&str> {
    let rest = type_name.strip_prefix("function(")?;
    let bytes = rest.as_bytes();
    let mut depth = 1usize;
    for (index, byte) in bytes.iter().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&rest[..index]);
                }
            }
            _ => {}
        }
    }
    None
}

/// `LUST_LUA_SOCKET_TRACE` is consulted on every native call; read the
/// environment once rather than per call.
#[cfg(feature = "std")]
fn lua_socket_trace_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("LUST_LUA_SOCKET_TRACE").is_some())
}

// =========================================================================
// Track 3D (L-HANDLERS): Algebraic Effect Handlers & Deopt Continuation
// =========================================================================

#[derive(Debug, Clone)]
pub struct EffectHandlerArm {
    pub effect_name: String,
    pub handler_fn: Value,
}

#[derive(Debug, Clone)]
pub struct EffectHandlerFrame {
    pub id: u64,
    pub arms: Vec<EffectHandlerArm>,
    pub active: bool,
    pub finalizer_baseline: usize,
}

#[derive(Debug, Clone)]
pub struct ResumptionState {
    pub handler_id: u64,
    pub effect_name: String,
    pub resumed: bool,
    pub resume_value: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct AbortPayload {
    pub handler_id: u64,
    pub value: Value,
}

pub enum EffectFinalizer {
    Rust(Box<dyn FnOnce() + Send + 'static>),
    Lust(Value),
}

#[cfg(feature = "std")]
std::thread_local! {
    static EFFECT_HANDLER_STACK: std::cell::RefCell<Vec<EffectHandlerFrame>> = const { std::cell::RefCell::new(Vec::new()) };
    static EFFECT_RESUMPTION_STACK: std::cell::RefCell<Vec<ResumptionState>> = const { std::cell::RefCell::new(Vec::new()) };
    static EFFECT_ABORT_PAYLOAD: std::cell::RefCell<Option<AbortPayload>> = const { std::cell::RefCell::new(None) };
    static EFFECT_FINALIZERS: std::cell::RefCell<Vec<EffectFinalizer>> = const { std::cell::RefCell::new(Vec::new()) };
    static NEXT_EFFECT_HANDLER_ID: std::cell::Cell<u64> = const { std::cell::Cell::new(1) };
}

pub fn register_effect_finalizer(f: impl FnOnce() + Send + 'static) {
    #[cfg(feature = "std")]
    EFFECT_FINALIZERS.with(|fins| {
        fins.borrow_mut().push(EffectFinalizer::Rust(Box::new(f)));
    });
    if let Some(vm_ptr) = crate::vm::current_vm_ptr() {
        unsafe {
            (*vm_ptr).has_active_finalizers = true;
        }
    }
}

pub fn get_effect_finalizer_count() -> usize {
    #[cfg(feature = "std")]
    {
        EFFECT_FINALIZERS.with(|fins| fins.borrow().len())
    }
    #[cfg(not(feature = "std"))]
    {
        0
    }
}

pub fn reset_effect_handler_state() {
    #[cfg(feature = "std")]
    {
        EFFECT_HANDLER_STACK.with(|s| s.borrow_mut().clear());
        EFFECT_RESUMPTION_STACK.with(|s| s.borrow_mut().clear());
        EFFECT_ABORT_PAYLOAD.with(|s| *s.borrow_mut() = None);
        EFFECT_FINALIZERS.with(|s| s.borrow_mut().clear());
    }
    if let Some(vm_ptr) = crate::vm::current_vm_ptr() {
        unsafe {
            (*vm_ptr).has_active_finalizers = false;
        }
    }
}

impl VM {
    pub fn run_finalizers_down_to(&mut self, target_depth: usize) -> Result<()> {
        #[cfg(feature = "std")]
        {
            loop {
                let maybe_fin = EFFECT_FINALIZERS.with(|fins| {
                    let mut b = fins.borrow_mut();
                    if b.len() > target_depth {
                        b.pop()
                    } else {
                        None
                    }
                });
                let Some(fin) = maybe_fin else { break };
                match fin {
                    EffectFinalizer::Rust(f) => {
                        f();
                    }
                    EffectFinalizer::Lust(val) => {
                        let _ = self.call_value(&val, Vec::new());
                    }
                }
            }
            self.has_active_finalizers = get_effect_finalizer_count() > 0;
        }
        Ok(())
    }

    pub(crate) fn is_effect_intrinsic(name: &str) -> bool {
        matches!(
            name,
            "__lust_handle"
                | "__lust_perform"
                | "perform"
                | "__lust_resume"
                | "resume"
                | "__lust_register_finalizer"
                | "register_finalizer"
        )
    }

    pub(crate) fn get_effect_intrinsic_name(func: &Value) -> Option<String> {
        if let Value::String(s) = func {
            if let Some(rest) = s.strip_prefix("__lust_effect_intrinsic:") {
                return Some(rest.to_string());
            }
            if Self::is_effect_intrinsic(s) {
                return Some(s.to_string());
            }
        }
        None
    }

    pub fn dispatch_effect_intrinsic(&mut self, name: &str, args: Vec<Value>) -> Result<Value> {
        match name {
            "__lust_handle" => {
                if args.len() < 2 {
                    return Err(LustError::RuntimeError {
                        message: "__lust_handle requires 2 arguments: (body_thunk, arms)".to_string(),
                    });
                }
                let body_thunk = args[0].clone();
                let arms_val = &args[1];
                let mut arms = Vec::new();
                match arms_val {
                    Value::Array(arr_rc) => {
                        let arr = arr_rc.borrow();
                        for item in arr.iter() {
                            if let Value::Tuple(tup) = item {
                                if tup.len() >= 2 {
                                    let effect_name = tup[0].as_string().map(|s| s.to_string()).unwrap_or_default();
                                    arms.push(EffectHandlerArm {
                                        effect_name,
                                        handler_fn: tup[1].clone(),
                                    });
                                }
                            } else if let Value::Array(sub_rc) = item {
                                let sub = sub_rc.borrow();
                                if sub.len() >= 2 {
                                    let effect_name = sub[0].as_string().map(|s| s.to_string()).unwrap_or_default();
                                    arms.push(EffectHandlerArm {
                                        effect_name,
                                        handler_fn: sub[1].clone(),
                                    });
                                }
                            }
                        }
                    }
                    Value::Tuple(tup_rc) => {
                        for item in tup_rc.iter() {
                            if let Value::Tuple(tup) = item {
                                if tup.len() >= 2 {
                                    let effect_name = tup[0].as_string().map(|s| s.to_string()).unwrap_or_default();
                                    arms.push(EffectHandlerArm {
                                        effect_name,
                                        handler_fn: tup[1].clone(),
                                    });
                                }
                            }
                        }
                    }
                    _ => {}
                }
                self.execute_effect_handle(body_thunk, arms)
            }
            "__lust_perform" | "perform" => {
                if args.is_empty() {
                    return Err(LustError::RuntimeError {
                        message: "perform requires at least an effect name".to_string(),
                    });
                }
                let mut effect_name = args[0].as_string().map(|s| s.to_string()).ok_or_else(|| {
                    LustError::RuntimeError {
                        message: "Effect name in perform must be a string".to_string(),
                    }
                })?;
                if effect_name.starts_with('"') && effect_name.ends_with('"') && effect_name.len() >= 2 {
                    effect_name = effect_name[1..effect_name.len() - 1].to_string();
                }
                let effect_args = if args.len() > 1 {
                    match &args[1] {
                        Value::Array(arr_rc) => arr_rc.borrow().clone(),
                        Value::Tuple(tup_rc) => (**tup_rc).clone(),
                        other => vec![other.clone()],
                    }
                } else {
                    Vec::new()
                };
                self.execute_effect_perform(effect_name, effect_args)
            }
            "__lust_resume" | "resume" => {
                if args.len() != 1 {
                    return Err(LustError::RuntimeError {
                        message: "resume requires exactly 1 argument".to_string(),
                    });
                }
                self.execute_effect_resume(args[0].clone())
            }
            "__lust_register_finalizer" | "register_finalizer" => {
                if args.is_empty() {
                    return Err(LustError::RuntimeError {
                        message: "register_finalizer requires a callable argument".to_string(),
                    });
                }
                #[cfg(feature = "std")]
                {
                    EFFECT_FINALIZERS.with(|fins| {
                        fins.borrow_mut().push(EffectFinalizer::Lust(args[0].clone()));
                    });
                    self.has_active_finalizers = true;
                }
                Ok(Value::Nil)
            }
            other => Err(LustError::RuntimeError {
                message: format!("Unknown effect intrinsic: {}", other),
            }),
        }
    }

    pub fn execute_effect_handle(
        &mut self,
        body_thunk: Value,
        arms: Vec<EffectHandlerArm>,
    ) -> Result<Value> {
        #[cfg(feature = "std")]
        {
            let handler_id = NEXT_EFFECT_HANDLER_ID.with(|id| {
                let cur = id.get();
                id.set(cur + 1);
                cur
            });
            let fin_baseline = get_effect_finalizer_count();
            let frame = EffectHandlerFrame {
                id: handler_id,
                arms,
                active: true,
                finalizer_baseline: fin_baseline,
            };
            EFFECT_HANDLER_STACK.with(|s| s.borrow_mut().push(frame));

            let body_res = self.call_value(&body_thunk, Vec::new());

            // Remove this handler frame from stack
            EFFECT_HANDLER_STACK.with(|s| {
                let mut b = s.borrow_mut();
                if let Some(pos) = b.iter().rposition(|f| f.id == handler_id) {
                    b.remove(pos);
                }
            });

            match body_res {
                Ok(val) => {
                    self.run_finalizers_down_to(fin_baseline)?;
                    Ok(val)
                }
                Err(err) => {
                    // Check if error is an abort directed at this handler
                    let mut matched_abort = None;
                    EFFECT_ABORT_PAYLOAD.with(|abort_cell| {
                        let mut b = abort_cell.borrow_mut();
                        if let Some(payload) = b.as_ref() {
                            if payload.handler_id == handler_id {
                                matched_abort = b.take();
                            }
                        }
                    });

                    if let Some(payload) = matched_abort {
                        self.run_finalizers_down_to(fin_baseline)?;
                        Ok(payload.value)
                    } else {
                        let _ = self.run_finalizers_down_to(fin_baseline);
                        Err(err)
                    }
                }
            }
        }
        #[cfg(not(feature = "std"))]
        {
            self.call_value(&body_thunk, Vec::new())
        }
    }

    pub fn execute_effect_perform(
        &mut self,
        effect_name: String,
        args: Vec<Value>,
    ) -> Result<Value> {
        #[cfg(feature = "std")]
        {
            let found = EFFECT_HANDLER_STACK.with(|s| {
                let b = s.borrow();
                for (idx, frame) in b.iter().enumerate().rev() {
                    if frame.active {
                        for arm in &frame.arms {
                            if arm.effect_name == effect_name {
                                return Some((idx, frame.id, arm.handler_fn.clone()));
                            }
                        }
                    }
                }
                None
            });

            let Some((frame_idx, handler_id, handler_fn)) = found else {
                let _ = self.run_finalizers_down_to(0);
                return Err(LustError::RuntimeError {
                    message: format!("Unhandled effect: {}", effect_name),
                });
            };

            EFFECT_HANDLER_STACK.with(|s| {
                s.borrow_mut()[frame_idx].active = false;
            });

            let resumption = ResumptionState {
                handler_id,
                effect_name: effect_name.clone(),
                resumed: false,
                resume_value: None,
            };
            EFFECT_RESUMPTION_STACK.with(|r| r.borrow_mut().push(resumption));

            let arm_res = self.call_value(&handler_fn, args);

            let popped_resumption = EFFECT_RESUMPTION_STACK.with(|r| r.borrow_mut().pop());

            EFFECT_HANDLER_STACK.with(|s| {
                let mut b = s.borrow_mut();
                if let Some(frame) = b.iter_mut().find(|f| f.id == handler_id) {
                    frame.active = true;
                }
            });

            match arm_res {
                Ok(arm_val) => {
                    let resumption = popped_resumption.ok_or_else(|| LustError::RuntimeError {
                        message: "Corrupted effect resumption stack".to_string(),
                    })?;
                    if resumption.resumed {
                        Ok(resumption.resume_value.unwrap_or(Value::Nil))
                    } else {
                        EFFECT_ABORT_PAYLOAD.with(|abort_cell| {
                            *abort_cell.borrow_mut() = Some(AbortPayload {
                                handler_id,
                                value: arm_val,
                            });
                        });
                        Err(LustError::RuntimeError {
                            message: format!("__lust_effect_abort:{}", handler_id),
                        })
                    }
                }
                Err(err) => Err(err),
            }
        }
        #[cfg(not(feature = "std"))]
        {
            Err(LustError::RuntimeError {
                message: format!("Unhandled effect in no_std: {}", effect_name),
            })
        }
    }

    pub fn execute_effect_resume(&mut self, val: Value) -> Result<Value> {
        #[cfg(feature = "std")]
        {
            EFFECT_RESUMPTION_STACK.with(|r| {
                let mut b = r.borrow_mut();
                let Some(resumption) = b.last_mut() else {
                    return Err(LustError::RuntimeError {
                        message: "Cannot resume outside of an effect handler arm".to_string(),
                    });
                };
                if resumption.resumed {
                    return Err(LustError::RuntimeError {
                        message: "Effect already resumed: resumption can only be called once".to_string(),
                    });
                }
                resumption.resumed = true;
                resumption.resume_value = Some(val.clone());
                Ok(val)
            })
        }
        #[cfg(not(feature = "std"))]
        {
            Ok(val)
        }
    }

    /// 3A -> 3D deopt interface consumer:
    /// Consumes a pending DeoptPayload from `take_deopt_payload()`, reconstructs the target CallFrame
    /// state (target IP and live registers) via `DeoptExitState::from_payload` / `apply_to_call_frame`,
    /// and resumes execution in the interpreter without duplicate execution.
    pub fn resume_from_deopt(
        &mut self,
        types: Option<&[(usize, crate::mir::ir::MirType)]>,
    ) -> Result<Value> {
        if self.call_stack.is_empty() {
            return Err(LustError::RuntimeError {
                message: "Cannot resume deopt: call stack is empty".to_string(),
            });
        }

        let payload = crate::mir::cranelift_backend::take_deopt_payload()
            .ok_or_else(|| LustError::RuntimeError {
                message: "No deopt payload available to reconstruct VM state".to_string(),
            })?;

        // 1. Validate call frame identity before mutation
        let top_frame = self.call_stack.last().expect("call stack validated non-empty");
        let fn_idx = top_frame.function_idx;
        let func = self.functions.get(fn_idx).ok_or_else(|| LustError::RuntimeError {
            message: format!("Function index {} out of bounds on top call frame", fn_idx),
        })?;

        if let Some(ref expected_fn_name) = payload.func_name {
            if func.name != *expected_fn_name {
                return Err(LustError::RuntimeError {
                    message: format!(
                        "Deopt payload function identity mismatch: expected '{}', top frame has '{}'",
                        expected_fn_name, func.name
                    ),
                });
            }
        }

        if let Some(ref expected_ident) = payload.frame_identity {
            if expected_ident.frame_depth != self.call_stack.len() {
                return Err(LustError::RuntimeError {
                    message: format!(
                        "Deopt payload frame depth mismatch: expected {}, call stack depth is {}",
                        expected_ident.frame_depth,
                        self.call_stack.len()
                    ),
                });
            }
        }

        // 2. Separate site/block identity from authoritative bytecode resume PC
        let site_id = payload.target_pc;
        let resume_pc = match payload.site_kind {
            crate::mir::cranelift_backend::DeoptSiteKind::BlockEntry => {
                if let Some(pc) = func.block_entry_pcs.get(&site_id) {
                    *pc
                } else if func.block_entry_pcs.is_empty() && site_id <= func.chunk.instructions.len() {
                    // Hand-crafted bytecode function where site_id directly indicates the bytecode PC
                    site_id
                } else {
                    return Err(LustError::RuntimeError {
                        message: format!(
                            "Deopt payload site/block ID {} has no proven block-entry bytecode resume PC mapping for function '{}'",
                            site_id, func.name
                        ),
                    });
                }
            }
            crate::mir::cranelift_backend::DeoptSiteKind::PreTerminator => {
                if let Some(pc) = func.terminator_pcs.get(&site_id) {
                    *pc
                } else if func.terminator_pcs.is_empty() && site_id <= func.chunk.instructions.len() {
                    site_id
                } else {
                    return Err(LustError::RuntimeError {
                        message: format!(
                            "Deopt payload site/block ID {} has no proven pre-terminator bytecode resume PC mapping for function '{}'",
                            site_id, func.name
                        ),
                    });
                }
            }
        };

        if resume_pc > func.chunk.instructions.len() {
            return Err(LustError::RuntimeError {
                message: format!(
                    "Deopt resume PC {} is out of bounds for function '{}' (instruction count {})",
                    resume_pc, func.name, func.chunk.instructions.len()
                ),
            });
        }

        // 3. Validate entire payload and types before mutating any frame state
        let provided_types = types.ok_or_else(|| LustError::RuntimeError {
            message: "Cannot resume deopt: type metadata required but None provided".to_string(),
        })?;

        let mut seen_type_indices = hashbrown::HashSet::with_capacity(provided_types.len());
        for &(t_idx, _) in provided_types {
            if !seen_type_indices.insert(t_idx) {
                return Err(LustError::RuntimeError {
                    message: format!("Duplicate type metadata for local index {}", t_idx),
                });
            }
        }

        let mut validated_restorations = Vec::with_capacity(payload.live_locals.len());
        let mut seen_indices = hashbrown::HashSet::new();

        for &(loc_idx, raw_val) in &payload.live_locals {
            if !seen_indices.insert(loc_idx) {
                return Err(LustError::RuntimeError {
                    message: format!("Duplicate local index {} in deopt payload", loc_idx),
                });
            }
            if loc_idx >= top_frame.reg_count as usize {
                return Err(LustError::RuntimeError {
                    message: format!(
                        "Deopt payload local index {} exceeds frame register count {}",
                        loc_idx,
                        top_frame.reg_count
                    ),
                });
            }
            let mir_ty = provided_types
                .iter()
                .find(|(i, _)| *i == loc_idx)
                .map(|(_, ty)| ty)
                .ok_or_else(|| LustError::RuntimeError {
                    message: format!("Missing type metadata for deopt local index {}", loc_idx),
                })?;
            let restored_val = match mir_ty {
                crate::mir::ir::MirType::Int => Value::Int(raw_val),
                crate::mir::ir::MirType::Float => Value::Float(f64::from_bits(raw_val as u64)),
                crate::mir::ir::MirType::Bool => Value::Bool(raw_val != 0),
                crate::mir::ir::MirType::Unit => Value::Nil,
                crate::mir::ir::MirType::String
                | crate::mir::ir::MirType::Array(_)
                | crate::mir::ir::MirType::Tuple(_)
                | crate::mir::ir::MirType::Map(_, _)
                | crate::mir::ir::MirType::Struct(_)
                | crate::mir::ir::MirType::Enum(_) => {
                    let handle_val = crate::mir::runtime_helpers::with_current_mir_context(|ctx| ctx.get_handle(raw_val));
                    match handle_val {
                        Some(val) => val,
                        None => {
                            return Err(LustError::RuntimeError {
                                message: format!(
                                    "Invalid or absent reference handle {} for deopt local {} of type {:?}",
                                    raw_val, loc_idx, mir_ty
                                ),
                            });
                        }
                    }
                }
                _ => {
                    return Err(LustError::RuntimeError {
                        message: format!(
                            "Unsupported type {:?} for deopt local {}",
                            mir_ty, loc_idx
                        ),
                    });
                }
            };
            validated_restorations.push((loc_idx, restored_val));
        }

        // 4. All validations passed; apply restored state to top call frame
        let top_frame = self.call_stack.last_mut().expect("call stack validated non-empty");
        top_frame.ip = resume_pc;
        let base_reg = top_frame.base_reg;
        for (loc_idx, val) in validated_restorations {
            if !val.is_plain() {
                top_frame.mark_non_plain(loc_idx);
            }
            self.vm_registers[base_reg + loc_idx] = val;
        }

        self.run()
    }
}
