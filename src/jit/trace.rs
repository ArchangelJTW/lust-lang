use crate::LustError;
use crate::bytecode::Instruction;
use crate::bytecode::value::NativeFn;
use crate::bytecode::value::StructObject;
use crate::bytecode::{Register, Value};
use alloc::{
    boxed::Box,
    format,
    rc::Rc,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::fmt;
use hashbrown::{HashMap, HashSet};

#[derive(Clone)]
pub struct TracedNativeFn {
    function: Rc<NativeFn>,
}

impl TracedNativeFn {
    pub fn new(function: Rc<NativeFn>) -> Self {
        Self { function }
    }

    pub fn pointer(&self) -> *const () {
        Rc::as_ptr(&self.function) as *const ()
    }

    /// The allocation pointer a `Value::NativeFunction` holding this
    /// function carries (the `Rc`'s own pointer word, not the data
    /// pointer `pointer` returns).
    pub fn inner_ptr(&self) -> usize {
        // The `Rc` stores the allocation's address (`RcInner { strong,
        // weak, value }`); `as_ptr` gives the value, two words in.
        (Rc::as_ptr(&self.function) as usize) - 2 * core::mem::size_of::<usize>()
    }
}

impl fmt::Debug for TracedNativeFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NativeFn({:p})", Rc::as_ptr(&self.function))
    }
}

#[derive(Debug, Clone)]
pub struct Trace {
    pub function_idx: usize,
    pub start_ip: usize,
    /// Compiled function code (not a loop trace): the body runs once from
    /// the function's entry, and `Return` ops return.
    pub is_function: bool,
    /// Function code only: may some register hold an owned value when the
    /// function returns? If not, the frame needs no drop pass.
    pub frame_may_own: bool,
    /// Function code only: what the registers hold on entry — the scalar
    /// parameters by their declared types, everything else `Nil`.
    pub entry_scalars: Vec<(Register, ValueType)>,
    /// Function code only: the parameters a native caller may alias into
    /// the frame (bit `i` for parameter `i`; see
    /// `FunctionSig::can_alias_param`). They never own anything in a
    /// native frame, so a return does not drop them.
    pub alias_params: u64,
    /// Function code only: registers that hold a borrow at some point
    /// (see `TraceOp::BorrowField`). They never hold an owned value, so
    /// an exit to the interpreter retains whatever they hold (a no-op for
    /// a scalar or Nil) and a native return does not release them.
    pub borrowed_registers: Vec<Register>,
    /// Operations executed once at trace entry (unboxing, guards, etc.)
    pub preamble: Vec<TraceOp>,
    /// Operations in the trace loop body
    pub ops: Vec<TraceOp>,
    /// Operations executed once at trace exit (reboxing to restore state)
    pub postamble: Vec<TraceOp>,
    pub inputs: Vec<Register>,
    pub outputs: Vec<Register>,
}

#[derive(Debug, Clone)]
pub struct InlineTrace {
    pub function_idx: usize,
    pub register_count: u8,
    pub first_arg: Register,
    pub arg_count: u8,
    pub arg_registers: Vec<Register>,
    pub body: Vec<TraceOp>,
    pub return_register: Option<Register>,
    pub is_closure: bool,
    pub upvalues_ptr: Option<*const ()>,
    /// Instructions after the call at which the interpreter resumes when
    /// the callee's frames were materialized: 1, or 2 when the move of
    /// the result into a local was folded into the call.
    pub resume_offset: usize,
}

#[derive(Debug, Clone)]
pub enum TraceOp {
    /// Marks the bytecode instruction the following ops came from. Emits no
    /// code; a backend uses it as the resume point when one of those ops
    /// fails, so the interpreter re-executes exactly the failing instruction
    /// (and raises its error) instead of restarting the loop iteration with
    /// half of its side effects already applied.
    At {
        ip: usize,
    },
    LoadConst {
        dest: Register,
        value: Value,
    },
    Move {
        dest: Register,
        src: Register,
    },
    Add {
        dest: Register,
        lhs: Register,
        rhs: Register,
        lhs_type: ValueType,
        rhs_type: ValueType,
    },
    Sub {
        dest: Register,
        lhs: Register,
        rhs: Register,
        lhs_type: ValueType,
        rhs_type: ValueType,
    },
    Mul {
        dest: Register,
        lhs: Register,
        rhs: Register,
        lhs_type: ValueType,
        rhs_type: ValueType,
    },
    Div {
        dest: Register,
        lhs: Register,
        rhs: Register,
        lhs_type: ValueType,
        rhs_type: ValueType,
    },
    Mod {
        dest: Register,
        lhs: Register,
        rhs: Register,
        lhs_type: ValueType,
        rhs_type: ValueType,
    },
    Neg {
        dest: Register,
        src: Register,
    },
    Eq {
        dest: Register,
        lhs: Register,
        rhs: Register,
        lhs_type: ValueType,
        rhs_type: ValueType,
    },
    Ne {
        dest: Register,
        lhs: Register,
        rhs: Register,
        lhs_type: ValueType,
        rhs_type: ValueType,
    },
    Lt {
        dest: Register,
        lhs: Register,
        rhs: Register,
        lhs_type: ValueType,
        rhs_type: ValueType,
    },
    Le {
        dest: Register,
        lhs: Register,
        rhs: Register,
        lhs_type: ValueType,
        rhs_type: ValueType,
    },
    Gt {
        dest: Register,
        lhs: Register,
        rhs: Register,
        lhs_type: ValueType,
        rhs_type: ValueType,
    },
    Ge {
        dest: Register,
        lhs: Register,
        rhs: Register,
        lhs_type: ValueType,
        rhs_type: ValueType,
    },
    And {
        dest: Register,
        lhs: Register,
        rhs: Register,
    },
    Or {
        dest: Register,
        lhs: Register,
        rhs: Register,
    },
    Not {
        dest: Register,
        src: Register,
    },
    Concat {
        dest: Register,
        lhs: Register,
        rhs: Register,
    },
    GetIndex {
        dest: Register,
        array: Register,
        index: Register,
    },
    TryGetIndex {
        dest: Register,
        array: Register,
        index: Register,
    },
    /// `array[index] = value` on an `Array`: a clone of the value replaces
    /// the element (the old one dropped), or the op fails to the
    /// interpreter (out of bounds, not an array, not an int index) which
    /// raises the error.
    SetIndex {
        array: Register,
        index: Register,
        value: Register,
    },
    /// `array.push(array, value)` on a plain (unspecialized) `Array`: a
    /// clone of the value is appended and `dest` becomes Nil, or the op
    /// fails to the interpreter (not an array, memory budget exceeded on
    /// growth) which performs the call and raises the error.
    ArrayPush {
        dest: Register,
        array: Register,
        value: Register,
    },
    ArrayIndexOk {
        value_dest: Register,
        condition_dest: Register,
        array: Register,
        index: Register,
        /// The scalar type a guard right after the read expects the element
        /// to have; the read then fails to the interpreter on any other
        /// element and leaves `value_dest` a known scalar.
        value_type: Option<ValueType>,
    },
    ArrayLen {
        dest: Register,
        array: Register,
    },
    GuardNativeFunction {
        register: Register,
        function: TracedNativeFn,
    },
    /// The VM's globals are unchanged since recording (`VM::globals_version`
    /// still equals `version`), so the snapshots the trace loaded from them
    /// (recorded as `LoadConst`) are still what `LoadGlobal` would produce.
    GuardGlobals {
        version: u64,
    },
    /// The register holds a struct of exactly this layout (identity of its
    /// `StructLayout`), so a method resolved for it stays valid.
    GuardStructLayout {
        register: Register,
        layout: *const (),
    },
    GuardFunction {
        register: Register,
        function_idx: usize,
    },
    GuardClosure {
        register: Register,
        function_idx: usize,
        upvalues_ptr: *const (),
    },
    CallNative {
        dest: Register,
        callee: Register,
        function: TracedNativeFn,
        first_arg: Register,
        arg_count: u8,
    },
    CallFunction {
        dest: Register,
        callee: Register,
        function_idx: usize,
        first_arg: Register,
        arg_count: u8,
        is_closure: bool,
        upvalues_ptr: Option<*const ()>,
    },
    InlineCall {
        dest: Register,
        callee: Register,
        trace: InlineTrace,
    },
    CallMethod {
        dest: Register,
        object: Register,
        method_name: String,
        first_arg: Register,
        arg_count: u8,
    },
    GetField {
        dest: Register,
        object: Register,
        field_name: String,
        field_index: Option<usize>,
        value_type: Option<ValueType>,
        is_weak: bool,
    },
    SetField {
        object: Register,
        field_name: String,
        value: Register,
        field_index: Option<usize>,
        value_type: Option<ValueType>,
        is_weak: bool,
    },
    NewArray {
        dest: Register,
        first_element: Register,
        count: u8,
    },
    NewStruct {
        dest: Register,
        struct_name: String,
        field_names: Vec<String>,
        field_registers: Vec<Register>,
    },
    NewEnumUnit {
        dest: Register,
        enum_name: String,
        variant_name: String,
    },
    NewEnumVariant {
        dest: Register,
        enum_name: String,
        variant_name: String,
        value_registers: Vec<Register>,
    },
    IsEnumVariant {
        dest: Register,
        value: Register,
        enum_name: String,
        variant_name: String,
    },
    TypeIs {
        dest: Register,
        value: Register,
        type_name: String,
    },
    TryCast {
        dest: Register,
        value: Register,
        type_name: String,
    },
    GetEnumValue {
        dest: Register,
        enum_reg: Register,
        index: u8,
    },
    /// Function code only: `dest` becomes a *borrow* of the strong field
    /// `field_index` of the struct in `object` — the bits of the value,
    /// with no reference count taken. The translator emits one only when
    /// the struct outlives the function's frame (it is a parameter the
    /// function never writes, or a borrow itself), nothing the function
    /// runs can write a field, and `dest` is never returned or overwritten
    /// with an owned value; every exit to the interpreter retains what the
    /// borrowed registers hold first (see `Trace::borrowed_registers`).
    BorrowField {
        dest: Register,
        object: Register,
        field_index: usize,
    },
    /// Function code only: `dest` becomes a borrow of payload value
    /// `index` of the enum in `enum_reg`, itself a borrow (an enum's
    /// payload is immutable, so it lives as long as the enum).
    BorrowEnumValue {
        dest: Register,
        enum_reg: Register,
        index: u8,
    },
    Guard {
        register: Register,
        expected_type: ValueType,
    },
    GuardLoopContinue {
        condition_register: Register,
        expect_truthy: bool,
        bailout_ip: usize,
    },
    /// Run the inner loop `[loop_start_ip, bailout_ip]` of `function_idx` to
    /// completion through its own root trace, then continue at `resume_ip`,
    /// the instruction the recording resumed at after the inner loop ended.
    /// Exits to the interpreter at `bailout_ip` (the inner back-edge) when
    /// the inner loop has no trace, and wherever the inner trace bails out
    /// otherwise.
    NestedLoopCall {
        function_idx: usize,
        loop_start_ip: usize,
        bailout_ip: usize,
        resume_ip: usize,
    },
    Return {
        value: Option<Register>,
    },
    /// A branch target in compiled function code (see `jit::function`).
    /// `scalars` is the static type environment every path into the label
    /// agrees on.
    Label {
        id: usize,
        scalars: Vec<(Register, ValueType)>,
    },
    Jump {
        label: usize,
    },
    /// Branch to `label` when the register's truthiness equals
    /// `expect_truthy`.
    BranchIf {
        condition_register: Register,
        expect_truthy: bool,
        label: usize,
    },
    /// A call from compiled function code to a bytecode function whose
    /// identity is known at compile time. Runs the callee's compiled code
    /// natively when it has some, otherwise exits to the interpreter at
    /// `call_ip` (the interpreter performs the call and carries on).
    CallDirect {
        dest: Register,
        callee: Register,
        /// A method call's receiver, passed as the callee's first
        /// argument ahead of `first_arg..`.
        receiver: Option<Register>,
        function_idx: usize,
        first_arg: Register,
        arg_count: u8,
        /// Registers the callee's frame needs.
        callee_registers: u8,
        call_ip: usize,
        /// Where the interpreter resumes in the caller when the callee's
        /// frames were materialized: the instruction after the call, or
        /// one further when the move of the result into a local was
        /// folded into the call.
        resume_ip: usize,
        /// Callee registers (bit i = register i) that receive a bitwise
        /// copy of the argument instead of a clone: the callee never
        /// writes them, so nothing owns them and they are not dropped
        /// with its frame (materialization clones them).
        alias_mask: u64,
        /// The callee's declared return kind when scalar (its compiled code
        /// guards its return value against it).
        result_type: Option<ValueType>,
    },
    /// Unbox a Value into specialized representation
    Unbox {
        specialized_id: usize,
        source_reg: Register,
        layout: crate::jit::specialization::SpecializedLayout,
    },
    /// Rebox a specialized value back to Value
    Rebox {
        dest_reg: Register,
        specialized_id: usize,
        layout: crate::jit::specialization::SpecializedLayout,
    },
    /// Drop a specialized value without reboxing (cleanup for leaked specializations)
    DropSpecialized {
        specialized_id: usize,
        layout: crate::jit::specialization::SpecializedLayout,
    },
    /// Operation on specialized values
    SpecializedOp {
        op: SpecializedOpKind,
        operands: Vec<Operand>,
    },
}

/// Operand for specialized operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operand {
    Register(u8),
    Specialized(usize),
    Immediate(i64),
}

/// Types of operations on specialized values
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpecializedOpKind {
    // Vector operations
    VecPush,
    VecPop,
    VecGet,
    VecSet,
    VecLen,

    // Map operations
    MapInsert,
    MapGet,
    MapRemove,

    // Struct operations
    StructGetField { field_index: usize },
    StructSetField { field_index: usize },

    // Arithmetic on unboxed values
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Neg,

    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    Int,
    Float,
    Bool,
    String,
    Array,
    Tuple,
    Struct,
    /// A register that holds nothing owned — `Nil`, a `Bool`, an `Int` or
    /// a `Float` — without saying which: a fresh frame's unwritten
    /// registers, or one that holds different scalars on different paths.
    /// Never recorded or guarded; the fact only tells the backend that a
    /// store into the register has nothing to drop.
    Plain,
}

impl TraceOp {
    /// The one register this op writes, for ops whose result can be sent
    /// to another register instead (see `register_dead_after`).
    pub fn single_dest(&self) -> Option<Register> {
        match self {
            TraceOp::LoadConst { dest, .. }
            | TraceOp::Move { dest, .. }
            | TraceOp::Concat { dest, .. }
            | TraceOp::GetIndex { dest, .. }
            | TraceOp::TryGetIndex { dest, .. }
            | TraceOp::ArrayLen { dest, .. }
            | TraceOp::CallNative { dest, .. }
            | TraceOp::CallFunction { dest, .. }
            | TraceOp::InlineCall { dest, .. }
            | TraceOp::CallMethod { dest, .. }
            | TraceOp::CallDirect { dest, .. }
            | TraceOp::GetField { dest, .. }
            | TraceOp::NewArray { dest, .. }
            | TraceOp::NewStruct { dest, .. }
            | TraceOp::NewEnumUnit { dest, .. }
            | TraceOp::NewEnumVariant { dest, .. }
            | TraceOp::IsEnumVariant { dest, .. }
            | TraceOp::TypeIs { dest, .. }
            | TraceOp::TryCast { dest, .. }
            | TraceOp::GetEnumValue { dest, .. }
            | TraceOp::BorrowField { dest, .. }
            | TraceOp::BorrowEnumValue { dest, .. } => Some(*dest),
            _ => None,
        }
    }

    /// Send a `single_dest` op's result to `register` instead.
    pub fn set_dest(&mut self, register: Register) {
        match self {
            TraceOp::LoadConst { dest, .. }
            | TraceOp::Move { dest, .. }
            | TraceOp::Concat { dest, .. }
            | TraceOp::GetIndex { dest, .. }
            | TraceOp::TryGetIndex { dest, .. }
            | TraceOp::ArrayLen { dest, .. }
            | TraceOp::CallNative { dest, .. }
            | TraceOp::CallFunction { dest, .. }
            | TraceOp::InlineCall { dest, .. }
            | TraceOp::CallMethod { dest, .. }
            | TraceOp::CallDirect { dest, .. }
            | TraceOp::GetField { dest, .. }
            | TraceOp::NewArray { dest, .. }
            | TraceOp::NewStruct { dest, .. }
            | TraceOp::NewEnumUnit { dest, .. }
            | TraceOp::NewEnumVariant { dest, .. }
            | TraceOp::IsEnumVariant { dest, .. }
            | TraceOp::TypeIs { dest, .. }
            | TraceOp::TryCast { dest, .. }
            | TraceOp::GetEnumValue { dest, .. }
            | TraceOp::BorrowField { dest, .. }
            | TraceOp::BorrowEnumValue { dest, .. } => *dest = register,
            _ => {}
        }
    }
}

/// Is `register` written before it is read on every bytecode path from
/// `ip` (so its value there is unobservable)? A temporary the compiler
/// moves into a local right after producing it is; the producer can then
/// write the local directly and the copy — a clone and a drop for an owned
/// value — is skipped. Conservative: an instruction that both reads and
/// writes it counts as a read, and a call's arguments and callee count as
/// reads.
pub fn register_dead_after(
    function: &crate::bytecode::Function,
    ip: usize,
    register: Register,
) -> bool {
    let instructions = &function.chunk.instructions;
    let mut visited = HashSet::new();
    let mut pending = alloc::vec![ip];
    while let Some(ip) = pending.pop() {
        if !visited.insert(ip) {
            continue;
        }
        let Some(instruction) = instructions.get(ip) else {
            return false;
        };
        if instruction.reads_register(register) {
            return false;
        }
        if instruction.defined_register() == Some(register) {
            continue;
        }
        match *instruction {
            Instruction::Return(_) => continue,
            Instruction::Jump(offset) => {
                let Some(target) = (ip as i64 + 1 + i64::from(offset)).try_into().ok() else {
                    return false;
                };
                pending.push(target);
            }
            Instruction::JumpIf(_, offset) | Instruction::JumpIfNot(_, offset) => {
                let Some(target) = (ip as i64 + 1 + i64::from(offset)).try_into().ok() else {
                    return false;
                };
                pending.push(target);
                pending.push(ip + 1);
            }
            _ => pending.push(ip + 1),
        }
    }
    true
}

pub struct TraceRecorder {
    pub trace: Trace,
    max_length: usize,
    recording: bool,
    completed: bool,
    finalized: bool,
    root_frame_index: usize,
    guarded_registers: HashSet<Register>,
    inline_stack: Vec<InlineContext>,
    /// `VM::globals_version` at the instruction being recorded; the VM keeps
    /// it current.
    pub globals_version: u64,
    /// A `GuardGlobals` for `globals_version` is in force: nothing recorded
    /// since it could have changed the globals.
    globals_guarded: bool,
    /// Natives the recorder may turn into specialized ops (`JitState::intrinsics`).
    intrinsics: HashMap<usize, crate::jit::Intrinsic>,
    /// Set when the recording was abandoned because a mutated specialized
    /// array escaped; the site is then recorded again without specializing.
    pub specialization_escaped: bool,
    /// A specialized array whose copy has been written was overwritten in
    /// its register (see `remove_specialization_tracking`): the recording
    /// is abandoned after the op that did it.
    overwritten_specialization: bool,
    /// Bytecode ip of the instruction being recorded, not yet written as an
    /// `At` marker (see `flush_marker`).
    pending_marker: Option<usize>,
    /// Bytecode ip of the instruction being recorded.
    current_ip: usize,
    op_count: usize,
    /// Track which registers contain specialized values (register -> (specialized_id, layout))
    specialized_registers:
        HashMap<Register, (usize, crate::jit::specialization::SpecializedLayout)>,
    /// Counter for generating specialized IDs
    next_specialized_id: usize,
    /// Registry for type specializations
    specialization_registry: crate::jit::specialization::SpecializationRegistry,
    /// Track how many times we've seen each loop backedge to enable unrolling
    loop_iterations: HashMap<(usize, usize), usize>,
    /// Track specialized values that were unboxed but later invalidated (need cleanup/drop)
    leaked_specialized_values: Vec<(usize, crate::jit::specialization::SpecializedLayout)>,
    /// While execution is inside a nested loop that a `NestedLoopCall` will
    /// run through its own trace, nothing is recorded (see `skip_nested`).
    nested_skip: Option<NestedSkip>,
}

/// The nested loop currently being skipped: its bytecode range in
/// `function_idx`, and where its `NestedLoopCall` op sits so the ip the
/// recording resumes at can be patched in once the loop is left.
#[derive(Debug, Clone, Copy)]
struct NestedSkip {
    function_idx: usize,
    loop_start_ip: usize,
    backedge_ip: usize,
    inline_depth: usize,
    op_index: usize,
}

#[derive(Debug, Clone)]
struct InlineContext {
    function_idx: usize,
    register_count: u8,
    dest: Register,
    callee_reg: Register,
    first_arg: Register,
    arg_count: u8,
    arg_registers: Vec<Register>,
    ops: Vec<TraceOp>,
    guarded_registers: HashSet<Register>,
    return_register: Option<Register>,
    is_closure: bool,
    upvalues_ptr: Option<*const ()>,
    /// Ip of the call instruction in the caller: the `InlineCall` op's
    /// marker, and where the caller resumes if the call must be redone.
    call_ip: usize,
}

impl TraceRecorder {
    pub fn new(function_idx: usize, start_ip: usize, max_length: usize) -> Self {
        Self {
            trace: Trace {
                function_idx,
                start_ip,
                is_function: false,
                frame_may_own: true,
                entry_scalars: Vec::new(),
                alias_params: 0,
                borrowed_registers: Vec::new(),
                preamble: Vec::new(),
                ops: Vec::new(),
                postamble: Vec::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
            },
            max_length,
            recording: true,
            completed: false,
            finalized: false,
            root_frame_index: 0,
            guarded_registers: HashSet::new(),
            inline_stack: Vec::new(),
            globals_version: 0,
            globals_guarded: false,
            intrinsics: HashMap::new(),
            specialization_escaped: false,
            overwritten_specialization: false,
            pending_marker: None,
            current_ip: 0,
            op_count: 0,
            specialized_registers: HashMap::new(),
            next_specialized_id: 0,
            specialization_registry: crate::jit::specialization::SpecializationRegistry::new(),
            loop_iterations: HashMap::new(),
            leaked_specialized_values: Vec::new(),
            nested_skip: None,
        }
    }

    /// Scan live registers at trace entry and specialize any loop-invariant arrays
    /// This should be called right after trace recording starts
    pub fn specialize_trace_inputs(
        &mut self,
        registers: &[Value],
        function: &crate::bytecode::Function,
    ) {
        crate::jit::log(|| "🔍 JIT: Scanning trace inputs for specialization...".to_string());

        // Only slots below `register_count` belong to this frame.  Everything
        // above is stale data left by previously executed frames, and a leftover
        // `Value::Array` up there would otherwise be unboxed and specialized —
        // reading, and on rebox writing, memory that is not ours.
        let live_registers = usize::from(function.register_count).min(registers.len());

        // An array can sit in more than one register of the same frame (a
        // temporary left over from constructing it, say).  Those registers share
        // one Rc, so specializing each of them separately would hand out two
        // independent unboxed copies of the same buffer and let the second
        // rebox clobber the first — which surfaces as the array spontaneously
        // having length 0.  Specialize each distinct buffer at most once.
        let mut seen_buffers: Vec<*const ()> = Vec::new();

        for reg in 0..live_registers {
            let reg = reg as u8;
            // Check if this register contains an Array at runtime
            if let Value::Array(ref arr_rc) = registers[reg as usize] {
                crate::jit::log(|| format!("🔍 JIT: Found array in reg {}", reg));

                let identity = Rc::as_ptr(arr_rc) as *const ();
                if seen_buffers.contains(&identity) {
                    crate::jit::log(|| {
                        format!(
                            "🔍 JIT: reg {} aliases an already-specialized array, skipping",
                            reg
                        )
                    });
                    continue;
                }
                // Selection is speculative; the generated unbox helper validates
                // every element before the specialized representation is used.
                // Only `Array<int>` may be specialized: the unbox/rebox helpers
                // (`jit_unbox_array_int` / `jit_rebox_array_int`) are the only
                // ones that exist, and the layout alone cannot distinguish an
                // int array from a float or bool one — every scalar element is
                // 8 bytes wide.  Claiming a float array is specializable made
                // codegen emit the *int* helpers for float data; the unbox then
                // correctly refused the elements and bailed out, but the trace
                // made no progress and was re-entered forever.
                //
                // Widen this only together with element-typed helpers.
                let element_type = arr_rc.borrow().first().and_then(|value| match value {
                    Value::Int(_) => Some(crate::ast::TypeKind::Int),
                    _ => None,
                });

                if let Some(elem_type) = element_type {
                    use crate::ast::{Span, Type};
                    let array_type =
                        crate::ast::TypeKind::Array(Box::new(Type::new(elem_type, Span::dummy())));

                    // Check if this array type is specializable
                    if let Some(layout) =
                        self.specialization_registry.get_specialization(&array_type)
                    {
                        crate::jit::log(|| {
                            format!(
                                "🔬 JIT: Specializing trace input reg {} ({:?})",
                                reg, array_type
                            )
                        });

                        // Emit Unbox in PREAMBLE (executes once at trace entry, not in loop)
                        let specialized_id = self.next_specialized_id;
                        self.next_specialized_id += 1;

                        self.trace.preamble.push(TraceOp::Unbox {
                            specialized_id,
                            source_reg: reg,
                            layout: layout.clone(),
                        });

                        // Track this specialized value
                        self.specialized_registers
                            .insert(reg, (specialized_id, layout));
                        seen_buffers.push(identity);
                    }
                }
            }
        }
    }

    /// A nested loop is not recorded into the trace that contains it: the
    /// `NestedLoopCall` op runs it through the inner loop's own root trace.
    /// Returns `true` while the instruction is inside the skipped loop. The
    /// first instruction outside it is where the outer trace resumes; that
    /// ip is patched into the op, and every type fact is forgotten, since
    /// the inner loop may have written any register.
    fn skip_nested(
        &mut self,
        instruction: Instruction,
        current_ip: usize,
        function_idx: usize,
    ) -> Result<bool, LustError> {
        let Some(skip) = self.nested_skip else {
            return Ok(false);
        };
        let ip = current_ip.saturating_sub(1);
        if function_idx == skip.function_idx
            && (skip.loop_start_ip..=skip.backedge_ip).contains(&ip)
        {
            if matches!(instruction, Instruction::Return(_)) {
                // Leaves the frame from inside the loop; the trace cannot
                // continue past the loop.
                self.stop_recording();
            }
            return Ok(true);
        }
        self.nested_skip = None;
        let ops = if skip.inline_depth == 0 {
            &mut self.trace.ops
        } else {
            &mut self.inline_stack[skip.inline_depth - 1].ops
        };
        match ops.get_mut(skip.op_index) {
            Some(TraceOp::NestedLoopCall { resume_ip, .. }) => *resume_ip = ip,
            _ => {
                self.stop_recording();
                return Err(LustError::RuntimeError {
                    message: "Trace aborted: lost the NestedLoopCall being skipped".to_string(),
                });
            }
        }
        self.current_guard_set_mut().clear();
        Ok(false)
    }

    /// The recording is being completed at the loop's own back-edge (which
    /// the VM handles before the instruction reaches the recorder). If a
    /// nested loop was being skipped, its `NestedLoopCall` resumes there:
    /// the inner loop ended on the last instruction before the back-edge.
    pub fn complete_nested_skip_at(&mut self, backedge_ip: usize) {
        if let Some(skip) = self.nested_skip.take() {
            let ops = if skip.inline_depth == 0 {
                &mut self.trace.ops
            } else {
                &mut self.inline_stack[skip.inline_depth - 1].ops
            };
            if let Some(TraceOp::NestedLoopCall { resume_ip, .. }) = ops.get_mut(skip.op_index) {
                *resume_ip = backedge_ip;
            } else {
                self.stop_recording();
            }
        }
    }

    fn current_function_idx(&self) -> usize {
        self.inline_stack
            .last()
            .map(|ctx| ctx.function_idx)
            .unwrap_or(self.trace.function_idx)
    }

    fn expected_frame_index(&self) -> usize {
        self.root_frame_index + self.inline_stack.len()
    }

    pub fn set_root_frame_index(&mut self, frame_index: usize) {
        self.root_frame_index = frame_index;
    }

    pub fn set_intrinsics(&mut self, intrinsics: &HashMap<usize, crate::jit::Intrinsic>) {
        self.intrinsics = intrinsics.clone();
    }

    /// The tracked register whose specialized array `register` holds (the
    /// register itself or an alias sharing its `Rc`).
    fn specialized_owner(&self, register: Register, registers: &[Value]) -> Option<Register> {
        if self.specialized_registers.contains_key(&register) {
            return Some(register);
        }
        let Some(Value::Array(array)) = registers.get(register as usize) else {
            return None;
        };
        self.specialized_registers
            .keys()
            .copied()
            .find(|candidate| {
                matches!(
                    registers.get(*candidate as usize),
                    Some(Value::Array(candidate_array)) if Rc::ptr_eq(array, candidate_array)
                )
            })
    }

    /// Code the recorder cannot see (a native, a call it does not inline)
    /// is about to run: it may read or write any array through the boxed
    /// value, which an unboxed copy would not reflect. Unused
    /// specializations are dropped; one already used aborts the recording,
    /// and the site is recorded again without specializing.
    fn specializations_escape(&mut self, what: &str) -> Result<(), LustError> {
        let tracked: Vec<Register> = self.specialized_registers.keys().copied().collect();
        for register in tracked {
            if !self.disable_unused_specialization(register) {
                self.specialization_escaped = true;
                self.stop_recording();
                return Err(LustError::RuntimeError {
                    message: format!(
                        "Trace aborted: specialized array escapes to {what} after a specialized op"
                    ),
                });
            }
        }
        Ok(())
    }

    fn current_guard_set(&self) -> &HashSet<Register> {
        self.inline_stack
            .last()
            .map(|ctx| &ctx.guarded_registers)
            .unwrap_or(&self.guarded_registers)
    }

    fn current_guard_set_mut(&mut self) -> &mut HashSet<Register> {
        self.inline_stack
            .last_mut()
            .map(|ctx| &mut ctx.guarded_registers)
            .unwrap_or(&mut self.guarded_registers)
    }

    fn is_guarded(&self, register: Register) -> bool {
        self.current_guard_set().contains(&register)
    }

    fn mark_guarded(&mut self, register: Register) {
        let set = self.current_guard_set_mut();
        set.insert(register);
    }

    fn forget_guard(&mut self, register: Register) {
        self.current_guard_set_mut().remove(&register);
    }

    /// The boxed register an op overwrites, if any.
    ///
    /// Guards, `SetField`, `Return` and `NestedLoopCall` write no register.
    /// `Unbox`/`Rebox`/`DropSpecialized` and `SpecializedOp` address specialized
    /// slots rather than boxed registers and are deliberately excluded: the
    /// rebox path removes its own tracking entry before pushing the op.
    fn written_register(op: &TraceOp) -> Option<Register> {
        match op {
            TraceOp::At { .. }
            | TraceOp::Label { .. }
            | TraceOp::Jump { .. }
            | TraceOp::BranchIf { .. } => None,
            TraceOp::CallDirect { dest, .. } | TraceOp::ArrayPush { dest, .. } => Some(*dest),
            TraceOp::LoadConst { dest, .. }
            | TraceOp::Move { dest, .. }
            | TraceOp::Add { dest, .. }
            | TraceOp::Sub { dest, .. }
            | TraceOp::Mul { dest, .. }
            | TraceOp::Div { dest, .. }
            | TraceOp::Mod { dest, .. }
            | TraceOp::Neg { dest, .. }
            | TraceOp::Eq { dest, .. }
            | TraceOp::Ne { dest, .. }
            | TraceOp::Lt { dest, .. }
            | TraceOp::Le { dest, .. }
            | TraceOp::Gt { dest, .. }
            | TraceOp::Ge { dest, .. }
            | TraceOp::And { dest, .. }
            | TraceOp::Or { dest, .. }
            | TraceOp::Not { dest, .. }
            | TraceOp::Concat { dest, .. }
            | TraceOp::GetIndex { dest, .. }
            | TraceOp::TryGetIndex { dest, .. }
            | TraceOp::ArrayLen { dest, .. }
            | TraceOp::CallNative { dest, .. }
            | TraceOp::CallFunction { dest, .. }
            | TraceOp::InlineCall { dest, .. }
            | TraceOp::CallMethod { dest, .. }
            | TraceOp::GetField { dest, .. }
            | TraceOp::NewArray { dest, .. }
            | TraceOp::NewStruct { dest, .. }
            | TraceOp::NewEnumUnit { dest, .. }
            | TraceOp::NewEnumVariant { dest, .. }
            | TraceOp::IsEnumVariant { dest, .. }
            | TraceOp::TypeIs { dest, .. }
            | TraceOp::TryCast { dest, .. }
            | TraceOp::GetEnumValue { dest, .. }
            | TraceOp::BorrowField { dest, .. }
            | TraceOp::BorrowEnumValue { dest, .. } => Some(*dest),
            TraceOp::SetField { .. }
            | TraceOp::SetIndex { .. }
            | TraceOp::ArrayIndexOk { .. }
            | TraceOp::Guard { .. }
            | TraceOp::GuardNativeFunction { .. }
            | TraceOp::GuardGlobals { .. }
            | TraceOp::GuardStructLayout { .. }
            | TraceOp::GuardFunction { .. }
            | TraceOp::GuardClosure { .. }
            | TraceOp::GuardLoopContinue { .. }
            | TraceOp::NestedLoopCall { .. }
            | TraceOp::Return { .. }
            | TraceOp::Unbox { .. }
            | TraceOp::Rebox { .. }
            | TraceOp::DropSpecialized { .. }
            | TraceOp::SpecializedOp { .. } => None,
        }
    }

    /// Write the pending `At` marker, if any, without counting it as a trace
    /// op or touching specialization tracking.
    fn flush_marker(&mut self) {
        let Some(ip) = self.pending_marker.take() else {
            return;
        };
        let op = TraceOp::At { ip };
        if let Some(ctx) = self.inline_stack.last_mut() {
            ctx.ops.push(op);
        } else {
            self.trace.ops.push(op);
        }
    }

    /// The single destination of the last op recorded in the current
    /// context, if the op is one whose result can be redirected.
    fn last_op_dest(&self) -> Option<Register> {
        let ops = self
            .inline_stack
            .last()
            .map(|ctx| &ctx.ops)
            .unwrap_or(&self.trace.ops);
        ops.iter()
            .rev()
            .find(|op| !matches!(op, TraceOp::At { .. }))
            .and_then(|op| op.single_dest())
    }

    /// Redirect the last recorded op's result to `dest` (see
    /// `register_dead_after`); a guard fact the op established moves along.
    fn retarget_last_op(&mut self, dest: Register) {
        let ops = self
            .inline_stack
            .last_mut()
            .map(|ctx| &mut ctx.ops)
            .unwrap_or(&mut self.trace.ops);
        let Some(op) = ops
            .iter_mut()
            .rev()
            .find(|op| !matches!(op, TraceOp::At { .. }))
        else {
            return;
        };
        let Some(old) = op.single_dest() else {
            return;
        };
        op.set_dest(dest);
        // The elided move must not run again after an exit inside the
        // callee: its frames resume the caller one instruction further.
        match op {
            TraceOp::InlineCall { trace, .. } => trace.resume_offset = 2,
            TraceOp::CallDirect { resume_ip, .. } => *resume_ip += 1,
            _ => {}
        }
        if self.current_guard_set().contains(&old) {
            self.forget_guard(old);
            self.mark_guarded(dest);
        }
    }

    fn current_ops_len(&self) -> usize {
        self.inline_stack
            .last()
            .map(|ctx| ctx.ops.len())
            .unwrap_or(self.trace.ops.len())
    }

    fn push_op(&mut self, op: TraceOp) {
        self.flush_marker();
        // A specialization describes the array a register held at trace entry.
        // The instant the trace writes something else into that register the
        // specialization is stale, and the postamble rebox would otherwise dump
        // the entry-time copy into whatever unrelated array the register now
        // holds.  That is what corrupted `grid[ctr % 4][i]`: the temp holding the
        // inner array was specialized at entry, then reassigned by
        // `GetIndex { dest: <temp> }` on every iteration, and the rebox wrote the
        // stale row back over a different row of `grid`.
        //
        // Invalidation used to be an explicit call at a handful of recording
        // sites, which is why `GetIndex` was missed.  Do it centrally instead so
        // no op can forget.
        if let TraceOp::ArrayIndexOk {
            value_dest,
            condition_dest,
            ..
        } = &op
        {
            self.remove_specialization_tracking(*value_dest);
            self.remove_specialization_tracking(*condition_dest);
        } else if let Some(dest) = Self::written_register(&op) {
            self.remove_specialization_tracking(dest);
        }

        // Anything that runs code the recorder does not see may change the
        // globals; the next `LoadGlobal` then needs a fresh version guard.
        if matches!(
            op,
            TraceOp::CallNative { .. }
                | TraceOp::CallFunction { .. }
                | TraceOp::CallMethod { .. }
                | TraceOp::CallDirect { .. }
                | TraceOp::NestedLoopCall { .. }
        ) {
            self.globals_guarded = false;
        }

        self.op_count += 1;
        if let Some(ctx) = self.inline_stack.last_mut() {
            ctx.ops.push(op);
        } else {
            self.trace.ops.push(op);
        }
    }

    /// Finalize the trace by adding postamble operations (rebox all specialized values)
    fn finalize_trace(&mut self) {
        if self.finalized {
            return;
        }
        self.finalized = true;

        crate::jit::log(|| {
            format!(
                "🏁 JIT: Finalizing trace - reboxing {} specialized values, dropping {} leaked values",
                self.specialized_registers.len(),
                self.leaked_specialized_values.len()
            )
        });

        // NOTE: We do NOT emit drops for leaked_specialized_values!
        // Those values were invalidated during trace RECORDING, so they never
        // actually exist on the JIT stack during trace EXECUTION.
        // The arrays are still managed by their Rc<RefCell<>> wrappers.

        // Rebox all remaining specialized values in the postamble
        for (&register, &(specialized_id, ref layout)) in self.specialized_registers.iter() {
            crate::jit::log(|| {
                format!(
                    "📦 JIT: Adding rebox to postamble for specialized #{} in reg {}",
                    specialized_id, register
                )
            });

            self.trace.postamble.push(TraceOp::Rebox {
                dest_reg: register,
                specialized_id,
                layout: layout.clone(),
            });
        }
    }

    /// Stop recording and finalize the trace
    fn stop_recording(&mut self) {
        crate::jit::log(|| {
            format!(
                "🛑 JIT: stop_recording called, recording={}, specialized_regs={}",
                self.recording,
                self.specialized_registers.len()
            )
        });
        if self.recording {
            self.finalize_trace();
            self.recording = false;
        }
    }

    fn complete_recording(&mut self) {
        self.stop_recording();
        self.completed = true;
    }

    /// Rebox all currently active specialized values
    /// This must be called before any side exit to restore interpreter-compatible state
    fn rebox_all_specialized_values(&mut self) {
        // Collect all specialized values that need reboxing
        let to_rebox: Vec<(
            Register,
            usize,
            crate::jit::specialization::SpecializedLayout,
        )> = self
            .specialized_registers
            .iter()
            .map(|(&reg, &(id, ref layout))| (reg, id, layout.clone()))
            .collect();

        // Emit Rebox operations
        for (register, specialized_id, layout) in to_rebox {
            crate::jit::log(|| {
                format!(
                    "📦 JIT: Reboxing specialized #{} back to reg {} before side exit",
                    specialized_id, register
                )
            });

            self.push_op(TraceOp::Rebox {
                dest_reg: register,
                specialized_id,
                layout,
            });

            // Remove from tracking
            self.specialized_registers.remove(&register);
        }
    }

    /// Invalidate specialization for a register that's about to be overwritten
    /// The specialized Vec data needs to be dropped since it won't be reboxed
    #[allow(dead_code)]
    fn invalidate_specialization(&mut self, register: Register) {
        if let Some((specialized_id, layout)) = self.specialized_registers.remove(&register) {
            crate::jit::log(|| {
                format!(
                    "🚫 JIT: Invalidating specialization for reg {} (being overwritten) - will drop specialized #{}",
                    register, specialized_id
                )
            });
            // Track this for cleanup in postamble - the Vec data needs to be dropped
            self.leaked_specialized_values
                .push((specialized_id, layout));
        }
    }

    /// The register a specialization describes is about to be overwritten.
    /// A specialization nothing has used yet is dropped outright, `Unbox`
    /// and all: the trace exit publishes every remaining copy back into the
    /// array it was taken from, and a copy taken at entry would overwrite
    /// whatever the trace itself wrote into that array through the boxed
    /// value (`array.push` on an alias, `a[i] = v`) with the entry-time
    /// contents. One already written to cannot simply be dropped, and after
    /// the overwrite the next iteration would run its ops on a copy of the
    /// wrong array: the recording is abandoned, and the site recorded again
    /// without specializing.
    fn remove_specialization_tracking(&mut self, register: Register) {
        if !self.specialized_registers.contains_key(&register) {
            return;
        }
        if self.disable_unused_specialization(register) {
            crate::jit::log(|| {
                format!(
                    "🗑️  JIT: Dropping the unused specialization of reg {} (being overwritten)",
                    register
                )
            });
            return;
        }
        self.specialized_registers.remove(&register);
        self.overwritten_specialization = true;
    }

    fn rebox_specialized_register(&mut self, register: Register, context: &str) {
        if let Some((specialized_id, layout)) = self.specialized_registers.remove(&register) {
            crate::jit::log(|| {
                format!(
                    "📦 JIT: Reboxing specialized #{} from reg {} before {}",
                    specialized_id, register, context
                )
            });
            self.push_op(TraceOp::Rebox {
                dest_reg: register,
                specialized_id,
                layout,
            });
        }
    }

    /// Checked source indexing reads the boxed array. If the recorder eagerly
    /// specialized that array at trace entry, discard the unused specialization
    /// instead of emitting a consuming Rebox into the loop body. Rebox cannot be
    /// unrolled safely because it transfers the specialized buffer's ownership.
    fn disable_unused_specialization(&mut self, register: Register) -> bool {
        let Some((specialized_id, _)) = self.specialized_registers.get(&register).cloned() else {
            return true;
        };
        let uses_specialization = |op: &TraceOp| {
            matches!(
                op,
                TraceOp::SpecializedOp { operands, .. }
                    if operands.iter().any(
                        |operand| matches!(operand, Operand::Specialized(id) if *id == specialized_id)
                    )
            )
        };
        if self.trace.ops.iter().any(&uses_specialization)
            || self
                .inline_stack
                .iter()
                .any(|context| context.ops.iter().any(&uses_specialization))
        {
            return false;
        }

        self.specialized_registers
            .retain(|_, (id, _)| *id != specialized_id);
        self.trace.preamble.retain(
            |op| !matches!(op, TraceOp::Unbox { specialized_id: id, .. } if *id == specialized_id),
        );
        self.trace.ops.retain(
            |op| !matches!(op, TraceOp::Unbox { specialized_id: id, .. } if *id == specialized_id),
        );
        for context in &mut self.inline_stack {
            context.ops.retain(|op| {
                !matches!(op, TraceOp::Unbox { specialized_id: id, .. } if *id == specialized_id)
            });
        }
        true
    }

    fn disable_unused_array_specialization(
        &mut self,
        register: Register,
        registers: &[Value],
    ) -> bool {
        self.specialized_owner(register, registers)
            .is_none_or(|owner| self.disable_unused_specialization(owner))
    }

    /// `array.push(a, v)` on a plain array, as one op instead of a native
    /// call: the array's type is guarded, the value may be anything (a
    /// specialized value is reboxed first). Nothing else can observe the
    /// push, so live specializations stay unboxed. False when the call is
    /// not that.
    fn record_array_push(
        &mut self,
        native_ptr: usize,
        first_arg: Register,
        arg_count: u8,
        dest_reg: Register,
        registers: &[Value],
    ) -> bool {
        if self.intrinsics.get(&native_ptr) != Some(&crate::jit::Intrinsic::ArrayPush)
            || arg_count != 2
            || !matches!(registers.get(first_arg as usize), Some(Value::Array(_)))
            || self.specialized_owner(first_arg, registers).is_some()
        {
            return false;
        }
        let value_reg = first_arg + 1;
        self.rebox_specialized_register(value_reg, "array.push");
        if !self.is_guarded(first_arg) {
            self.push_op(TraceOp::Guard {
                register: first_arg,
                expected_type: ValueType::Array,
            });
            self.mark_guarded(first_arg);
        }
        self.push_op(TraceOp::ArrayPush {
            dest: dest_reg,
            array: first_arg,
            value: value_reg,
        });
        true
    }

    /// `array.push(a, v)` / `array.len(a)` on a specialized array, as the
    /// specialized ops (the same ones `a:push(v)` records). `None` when the
    /// call is not one of these; `Some` is the recording's result.
    fn record_intrinsic_call(
        &mut self,
        native_ptr: usize,
        first_arg: Register,
        arg_count: u8,
        dest_reg: Register,
        registers: &[Value],
    ) -> Option<Result<(), LustError>> {
        use crate::jit::Intrinsic;
        let intrinsic = *self.intrinsics.get(&native_ptr)?;
        let owner = self.specialized_owner(first_arg, registers)?;
        let specialized_id = self.specialized_registers.get(&owner)?.0;
        match intrinsic {
            Intrinsic::ArrayPush if arg_count == 2 => {
                let value_reg = first_arg + 1;
                if !matches!(registers[value_reg as usize], Value::Int(_)) {
                    return None;
                }
                crate::jit::log(|| {
                    format!(
                        "⚡ JIT: array.push on reg {} (specialized #{})",
                        first_arg, specialized_id
                    )
                });
                if !self.is_guarded(value_reg) {
                    self.push_op(TraceOp::Guard {
                        register: value_reg,
                        expected_type: ValueType::Int,
                    });
                    self.mark_guarded(value_reg);
                }
                self.push_op(TraceOp::SpecializedOp {
                    op: SpecializedOpKind::VecPush,
                    operands: vec![
                        Operand::Specialized(specialized_id),
                        Operand::Register(value_reg),
                    ],
                });
                // `array.push` returns Nil.
                self.push_op(TraceOp::LoadConst {
                    dest: dest_reg,
                    value: Value::Nil,
                });
                Some(Ok(()))
            }
            Intrinsic::ArrayLen if arg_count == 1 => {
                crate::jit::log(|| {
                    format!(
                        "⚡ JIT: array.len on reg {} (specialized #{})",
                        first_arg, specialized_id
                    )
                });
                self.push_op(TraceOp::SpecializedOp {
                    op: SpecializedOpKind::VecLen,
                    operands: vec![
                        Operand::Specialized(specialized_id),
                        Operand::Register(dest_reg),
                    ],
                });
                Some(Ok(()))
            }
            _ => None,
        }
    }

    fn should_inline(&self, function_idx: usize, callee_fn: &crate::bytecode::Function) -> bool {
        if function_idx == self.trace.function_idx {
            return false;
        }

        if self
            .inline_stack
            .iter()
            .any(|ctx| ctx.function_idx == function_idx)
        {
            return false;
        }

        // Disable inlining when specialized values are active to avoid
        // stack layout conflicts between inline frames and specialized storage
        if !self.specialized_registers.is_empty() {
            return false;
        }

        if callee_fn.chunk.instructions.iter().any(|inst| {
            matches!(
                inst,
                Instruction::Jump(_) | Instruction::JumpIf(..) | Instruction::JumpIfNot(..)
            )
        }) {
            return false;
        }

        true
    }

    #[allow(clippy::too_many_arguments)]
    fn push_inline_context(
        &mut self,
        function_idx: usize,
        register_count: u8,
        dest: Register,
        callee_reg: Register,
        first_arg: Register,
        arg_count: u8,
        arg_registers: Vec<Register>,
        is_closure: bool,
        upvalues_ptr: Option<*const ()>,
    ) {
        self.inline_stack.push(InlineContext {
            function_idx,
            register_count,
            dest,
            callee_reg,
            first_arg,
            arg_count,
            arg_registers,
            ops: Vec::new(),
            guarded_registers: HashSet::new(),
            return_register: None,
            is_closure,
            upvalues_ptr,
            call_ip: self.current_ip,
        });
    }

    fn finalize_inline_context(&mut self) -> Option<TraceOp> {
        let context = self.inline_stack.pop()?;
        // The callee's `Return` left its own ip pending; the `InlineCall` op
        // belongs to the call instruction.
        self.pending_marker = Some(context.call_ip);
        let trace = InlineTrace {
            function_idx: context.function_idx,
            register_count: context.register_count,
            first_arg: context.first_arg,
            arg_count: context.arg_count,
            arg_registers: context.arg_registers,
            body: context.ops,
            return_register: context.return_register,
            is_closure: context.is_closure,
            upvalues_ptr: context.upvalues_ptr,
            resume_offset: 1,
        };
        Some(TraceOp::InlineCall {
            dest: context.dest,
            callee: context.callee_reg,
            trace,
        })
    }

    pub fn record_instruction(
        &mut self,
        instruction: Instruction,
        current_ip: usize,
        registers: &[Value],
        function: &crate::bytecode::Function,
        function_idx: usize,
        functions: &[crate::bytecode::Function],
    ) -> Result<(), LustError> {
        self.record_instruction_at_frame(
            self.expected_frame_index(),
            instruction,
            current_ip,
            registers,
            function,
            function_idx,
            functions,
            false,
        )
    }

    /// `frame_pushed`: the instruction was a call that pushed a new frame,
    /// so the executing frame's registers are still exactly as they were
    /// before it (the destination is written only when the callee returns).
    #[allow(clippy::too_many_arguments)]
    pub fn record_instruction_at_frame(
        &mut self,
        frame_index: usize,
        instruction: Instruction,
        current_ip: usize,
        registers: &[Value],
        function: &crate::bytecode::Function,
        function_idx: usize,
        functions: &[crate::bytecode::Function],
        frame_pushed: bool,
    ) -> Result<(), LustError> {
        if !self.recording {
            return Ok(());
        }

        // A non-inlined call is emitted as one guarded CallFunction operation.
        // Its interpreter frames are opaque to this trace; resume recording when
        // execution returns to the exact activation that owns the trace.
        if frame_index != self.expected_frame_index() {
            return Ok(());
        }

        if function_idx != self.current_function_idx() {
            // Execution has entered a function this trace is not inlining, so
            // its instructions are not being recorded.  Skipping them and
            // carrying on produces a body that omits the call entirely: its side
            // effects are lost and, worse, the register meant to receive its
            // result is left holding whatever it happened to contain before.
            //
            // A method call whose result register had last been used for the
            // loop-condition flag turned `acc = acc + c:bump()` into `acc`
            // plus the bit pattern of a bool.  There is no way to record a
            // correct trace from here, so abandon it.
            self.stop_recording();
            crate::jit::log(|| {
                format!(
                    "Trace aborted: execution left the traced function (recording {}, now in {})",
                    self.trace.function_idx, function_idx
                )
            });
            return Err(LustError::RuntimeError {
                message: "Trace aborted: execution left the traced function".to_string(),
            });
        }

        if self.skip_nested(instruction, current_ip, function_idx)? {
            return Ok(());
        }

        if let Some(dest) = instruction.defined_register() {
            let preserves_numeric_inputs =
                instruction
                    .numeric_specialization()
                    .is_some_and(|(generic, _, _, _)| {
                        matches!(
                            generic,
                            Instruction::Add(..)
                                | Instruction::Sub(..)
                                | Instruction::Mul(..)
                                | Instruction::Div(..)
                                | Instruction::Mod(..)
                                | Instruction::Neg(..)
                        )
                    });
            if instruction.reads_register(dest) && !preserves_numeric_inputs && !frame_pushed {
                self.stop_recording();
                return Err(LustError::RuntimeError {
                    message: format!(
                        "Trace aborted: {:?} aliases its destination and requires pre-execution operands",
                        instruction.opcode()
                    ),
                });
            }
            self.forget_guard(dest);
        }

        // `current_ip` is the ip after the fetch; the instruction itself is one
        // before it (the same convention as guard bailout ips). The marker is
        // written lazily, in front of the first op this instruction records.
        self.current_ip = current_ip.saturating_sub(1);
        self.pending_marker = Some(self.current_ip);

        // Reuse the numeric trace IR, but retain the bytecode's input contract.
        // Host mutation and trace entry still require guards before payload loads.
        let instruction =
            if let Some((generic, ty, lhs, rhs)) = instruction.numeric_specialization() {
                let expected = match ty {
                    crate::number::NumericType::Int => ValueType::Int,
                    crate::number::NumericType::Float => ValueType::Float,
                };
                if [lhs, rhs]
                    .iter()
                    .any(|&reg| Self::get_value_type(&registers[reg as usize]) != Some(expected))
                {
                    self.stop_recording();
                    return Err(LustError::RuntimeError {
                        message: "Trace aborted: typed numeric operand mismatch".to_string(),
                    });
                }
                generic
            } else {
                instruction
            };

        let outcome: Result<(), LustError> = match instruction {
            Instruction::LoadConst(dest, _) => {
                // Rebox specialized value if dest contains one
                self.remove_specialization_tracking(dest);

                // A constant establishes what the register holds until the
                // next write: its scalar type, or for a function constant its
                // identity, so a call through it needs no `GuardFunction`.
                if Self::get_value_type(&registers[dest as usize]).is_some()
                    || matches!(registers[dest as usize], Value::Function(_))
                {
                    self.mark_guarded(dest);
                }

                self.push_op(TraceOp::LoadConst {
                    dest,
                    value: registers[dest as usize].clone(),
                });
                Ok(())
            }

            Instruction::LoadBool(dest, value) => {
                self.remove_specialization_tracking(dest);
                self.mark_guarded(dest);
                self.push_op(TraceOp::LoadConst {
                    dest,
                    value: Value::Bool(value),
                });
                Ok(())
            }

            Instruction::LoadGlobal(dest, _) => {
                // The value is a snapshot: valid while `VM::globals_version`
                // is what it was now, which every mutation of the globals
                // bumps. One guard covers every load until something
                // recorded (a call) could have changed them.
                if !self.globals_guarded {
                    self.push_op(TraceOp::GuardGlobals {
                        version: self.globals_version,
                    });
                    self.globals_guarded = true;
                }
                self.push_op(TraceOp::LoadConst {
                    dest,
                    value: registers[dest as usize].clone(),
                });
                Ok(())
            }

            Instruction::StoreGlobal(_, _) => {
                // A store bumps the version, so a trace containing one would
                // fail its own guard on the next iteration.
                self.stop_recording();
                Err(LustError::RuntimeError {
                    message: "Trace aborted: global store".to_string(),
                })
            }

            Instruction::Move(dest, src) => {
                // A temporary moved into a local right after being produced:
                // let the producer write the local, and skip the copy.
                if dest != src
                    && !self.specialized_registers.contains_key(&src)
                    && !self.specialized_registers.contains_key(&dest)
                    && self.last_op_dest() == Some(src)
                    && register_dead_after(function, current_ip, src)
                {
                    self.remove_specialization_tracking(dest);
                    self.retarget_last_op(dest);
                    return Ok(());
                }
                // If dest contains a specialized value, rebox it first before overwriting
                self.remove_specialization_tracking(dest);

                // Check if we're moving a specialized value
                let moved_specialization = if dest != src {
                    self.specialized_registers.get(&src).cloned()
                } else {
                    None
                };
                if let Some((specialized_id, _)) = &moved_specialization {
                    crate::jit::log(|| {
                        format!(
                            "📦 JIT: Moving specialized #{} from reg {} to reg {}",
                            specialized_id, src, dest
                        )
                    });
                }

                // A scalar source, once guarded, moves as a payload copy
                // and a typed store instead of a runtime clone and drop.
                if let Some(ty @ (ValueType::Int | ValueType::Float | ValueType::Bool)) =
                    Self::get_value_type(&registers[src as usize])
                    && !self.is_guarded(src)
                {
                    self.push_op(TraceOp::Guard {
                        register: src,
                        expected_type: ty,
                    });
                    self.mark_guarded(src);
                }
                self.push_op(TraceOp::Move { dest, src });
                if let Some((specialized_id, layout)) = moved_specialization {
                    // `push_op` invalidates the destination first; alias it
                    // only after that generic invalidation. Both registers
                    // hold the same `Rc`, so both stay tracked: the next
                    // iteration's `Move` finds the source still specialized.
                    self.specialized_registers
                        .insert(dest, (specialized_id, layout));
                }
                Ok(())
            }

            Instruction::Add(dest, lhs, rhs) => {
                // Rebox specialized value if dest contains one
                self.remove_specialization_tracking(dest);

                self.add_type_guards(lhs, rhs, registers, function)?;
                let lhs_type =
                    Self::get_value_type(&registers[lhs as usize]).unwrap_or(ValueType::Int);
                let rhs_type =
                    Self::get_value_type(&registers[rhs as usize]).unwrap_or(ValueType::Int);
                self.push_op(TraceOp::Add {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                });
                self.mark_guarded(dest);
                Ok(())
            }

            Instruction::Sub(dest, lhs, rhs) => {
                self.remove_specialization_tracking(dest);
                self.add_type_guards(lhs, rhs, registers, function)?;
                let lhs_type =
                    Self::get_value_type(&registers[lhs as usize]).unwrap_or(ValueType::Int);
                let rhs_type =
                    Self::get_value_type(&registers[rhs as usize]).unwrap_or(ValueType::Int);
                self.push_op(TraceOp::Sub {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                });
                self.mark_guarded(dest);
                Ok(())
            }

            Instruction::Mul(dest, lhs, rhs) => {
                self.add_type_guards(lhs, rhs, registers, function)?;
                let lhs_type =
                    Self::get_value_type(&registers[lhs as usize]).unwrap_or(ValueType::Int);
                let rhs_type =
                    Self::get_value_type(&registers[rhs as usize]).unwrap_or(ValueType::Int);
                self.push_op(TraceOp::Mul {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                });
                self.mark_guarded(dest);
                Ok(())
            }

            Instruction::Div(dest, lhs, rhs) => {
                self.add_type_guards(lhs, rhs, registers, function)?;
                let lhs_type =
                    Self::get_value_type(&registers[lhs as usize]).unwrap_or(ValueType::Int);
                let rhs_type =
                    Self::get_value_type(&registers[rhs as usize]).unwrap_or(ValueType::Int);
                self.push_op(TraceOp::Div {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                });
                self.mark_guarded(dest);
                Ok(())
            }

            Instruction::Mod(dest, lhs, rhs) => {
                self.add_type_guards(lhs, rhs, registers, function)?;
                let lhs_type =
                    Self::get_value_type(&registers[lhs as usize]).unwrap_or(ValueType::Int);
                let rhs_type =
                    Self::get_value_type(&registers[rhs as usize]).unwrap_or(ValueType::Int);
                self.push_op(TraceOp::Mod {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                });
                self.mark_guarded(dest);
                Ok(())
            }

            Instruction::Neg(dest, src) => {
                self.numeric_comparison_types(src, src, registers)?;
                self.add_type_guards(src, src, registers, function)?;
                self.push_op(TraceOp::Neg { dest, src });
                self.mark_guarded(dest);
                Ok(())
            }

            Instruction::Eq(dest, lhs, rhs) => {
                let (lhs_type, rhs_type) = self.scalar_comparison_types(lhs, rhs, registers)?;
                self.add_type_guards(lhs, rhs, registers, function)?;
                self.push_op(TraceOp::Eq {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                });
                self.mark_guarded(dest);
                Ok(())
            }

            Instruction::Ne(dest, lhs, rhs) => {
                let (lhs_type, rhs_type) = self.scalar_comparison_types(lhs, rhs, registers)?;
                self.add_type_guards(lhs, rhs, registers, function)?;
                self.push_op(TraceOp::Ne {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                });
                self.mark_guarded(dest);
                Ok(())
            }

            Instruction::Lt(dest, lhs, rhs) => {
                let (lhs_type, rhs_type) = self.numeric_comparison_types(lhs, rhs, registers)?;
                self.add_type_guards(lhs, rhs, registers, function)?;
                self.push_op(TraceOp::Lt {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                });
                self.mark_guarded(dest);
                Ok(())
            }

            Instruction::Le(dest, lhs, rhs) => {
                let (lhs_type, rhs_type) = self.numeric_comparison_types(lhs, rhs, registers)?;
                self.add_type_guards(lhs, rhs, registers, function)?;
                self.push_op(TraceOp::Le {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                });
                self.mark_guarded(dest);
                Ok(())
            }

            Instruction::Gt(dest, lhs, rhs) => {
                let (lhs_type, rhs_type) = self.numeric_comparison_types(lhs, rhs, registers)?;
                self.add_type_guards(lhs, rhs, registers, function)?;
                self.push_op(TraceOp::Gt {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                });
                self.mark_guarded(dest);
                Ok(())
            }

            Instruction::Ge(dest, lhs, rhs) => {
                let (lhs_type, rhs_type) = self.numeric_comparison_types(lhs, rhs, registers)?;
                self.add_type_guards(lhs, rhs, registers, function)?;
                self.push_op(TraceOp::Ge {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                });
                self.mark_guarded(dest);
                Ok(())
            }

            Instruction::And(dest, lhs, rhs) => {
                self.push_op(TraceOp::And { dest, lhs, rhs });
                Ok(())
            }

            Instruction::Or(dest, lhs, rhs) => {
                self.push_op(TraceOp::Or { dest, lhs, rhs });
                Ok(())
            }

            Instruction::Not(dest, src) => {
                self.push_op(TraceOp::Not { dest, src });
                Ok(())
            }

            Instruction::Concat(dest, lhs, rhs) => {
                if let Some(ty) = Self::get_value_type(&registers[lhs as usize])
                    && !self.is_guarded(lhs)
                {
                    self.push_op(TraceOp::Guard {
                        register: lhs,
                        expected_type: ty,
                    });
                    self.mark_guarded(lhs);
                }

                if let Some(ty) = Self::get_value_type(&registers[rhs as usize])
                    && !self.is_guarded(rhs)
                {
                    self.push_op(TraceOp::Guard {
                        register: rhs,
                        expected_type: ty,
                    });
                    self.mark_guarded(rhs);
                }

                self.push_op(TraceOp::Concat { dest, lhs, rhs });
                Ok(())
            }

            Instruction::GetIndex(dest, array, index) => {
                if !self.disable_unused_array_specialization(array, registers) {
                    self.specialization_escaped = true;
                    self.stop_recording();
                    return Err(LustError::RuntimeError {
                        message: "Trace aborted: array read follows a specialized mutation"
                            .to_string(),
                    });
                }
                if let Some(ty) = Self::get_value_type(&registers[array as usize])
                    && !self.is_guarded(array)
                {
                    self.push_op(TraceOp::Guard {
                        register: array,
                        expected_type: ty,
                    });
                    self.mark_guarded(array);
                }

                if let Some(ty) = Self::get_value_type(&registers[index as usize])
                    && !self.is_guarded(index)
                {
                    self.push_op(TraceOp::Guard {
                        register: index,
                        expected_type: ty,
                    });
                    self.mark_guarded(index);
                }

                self.push_op(TraceOp::GetIndex { dest, array, index });
                Ok(())
            }

            Instruction::TryGetIndex(dest, array, index) => {
                if !matches!(registers.get(array as usize), Some(Value::Array(_))) {
                    self.stop_recording();
                    return Err(LustError::RuntimeError {
                        message: "Trace aborted: checked indexing currently supports arrays only"
                            .to_string(),
                    });
                }

                if !self.disable_unused_array_specialization(array, registers) {
                    self.specialization_escaped = true;
                    self.stop_recording();
                    return Err(LustError::RuntimeError {
                        message: "Trace aborted: checked read follows a specialized array mutation"
                            .to_string(),
                    });
                }
                if let Some(ty) = Self::get_value_type(&registers[array as usize])
                    && !self.is_guarded(array)
                {
                    self.push_op(TraceOp::Guard {
                        register: array,
                        expected_type: ty,
                    });
                    self.mark_guarded(array);
                }

                if let Some(ty) = Self::get_value_type(&registers[index as usize])
                    && !self.is_guarded(index)
                {
                    self.push_op(TraceOp::Guard {
                        register: index,
                        expected_type: ty,
                    });
                    self.mark_guarded(index);
                }

                self.push_op(TraceOp::TryGetIndex { dest, array, index });
                Ok(())
            }

            Instruction::ArrayLen(dest, array) => {
                if let Some(ty) = Self::get_value_type(&registers[array as usize])
                    && !self.is_guarded(array)
                {
                    self.push_op(TraceOp::Guard {
                        register: array,
                        expected_type: ty,
                    });
                    self.mark_guarded(array);
                }

                self.push_op(TraceOp::ArrayLen { dest, array });
                self.mark_guarded(dest);
                Ok(())
            }

            Instruction::CallMethod(obj_reg, method_name_idx, first_arg, arg_count, dest_reg) => {
                // Rebox specialized value if dest_reg contains one
                self.remove_specialization_tracking(dest_reg);

                let method_name = function.chunk.constants[method_name_idx as usize]
                    .as_string()
                    .unwrap_or("unknown")
                    .to_string();

                // Check if this is a method on a specialized value
                if let Some(&(specialized_id, _)) = self.specialized_registers.get(&obj_reg) {
                    // This is a method call on a specialized value
                    match method_name.as_str() {
                        "push" if arg_count == 1 => {
                            // Specialized array push
                            crate::jit::log(|| {
                                format!(
                                    "⚡ JIT: Specializing push on reg {} (specialized #{})",
                                    obj_reg, specialized_id
                                )
                            });

                            // Guard the argument
                            let value_reg = first_arg;
                            if let Some(ty) = Self::get_value_type(&registers[value_reg as usize])
                                && !self.is_guarded(value_reg)
                            {
                                self.push_op(TraceOp::Guard {
                                    register: value_reg,
                                    expected_type: ty,
                                });
                                self.mark_guarded(value_reg);
                            }

                            // Emit specialized push operation
                            self.push_op(TraceOp::SpecializedOp {
                                op: SpecializedOpKind::VecPush,
                                operands: vec![
                                    Operand::Specialized(specialized_id),
                                    Operand::Register(value_reg),
                                ],
                            });

                            return Ok(());
                        }
                        "len" if arg_count == 0 => {
                            // Specialized array len
                            crate::jit::log(|| {
                                format!(
                                    "⚡ JIT: Specializing len on reg {} (specialized #{})",
                                    obj_reg, specialized_id
                                )
                            });

                            // Emit specialized len operation
                            self.push_op(TraceOp::SpecializedOp {
                                op: SpecializedOpKind::VecLen,
                                operands: vec![
                                    Operand::Specialized(specialized_id),
                                    Operand::Register(dest_reg),
                                ],
                            });

                            return Ok(());
                        }
                        _ => {
                            // Any other method reads the boxed array: the
                            // specialization must go first.
                            self.specializations_escape("a method call")?;
                        }
                    }
                }

                // A user-defined struct method: resolve it the way the
                // interpreter does and inline it like a call, with the
                // receiver as the first argument. The receiver's layout is
                // guarded because trait dispatch can bring different structs
                // to one call site.
                if let Value::Struct(object) = &registers[obj_reg as usize] {
                    let StructObject { name, layout, .. } = object.as_ref();
                    let mangled = format!("{}:{}", name, method_name);
                    if let Some(function_idx) = functions.iter().position(|f| f.name == mangled) {
                        let callee_fn = &functions[function_idx];
                        let total_args = arg_count as usize + 1;
                        if self.should_inline(function_idx, callee_fn)
                            && callee_fn.register_count > 0
                            && total_args <= callee_fn.register_count as usize
                        {
                            self.push_op(TraceOp::GuardStructLayout {
                                register: obj_reg,
                                layout: Rc::as_ptr(layout) as *const (),
                            });
                            self.mark_guarded(obj_reg);
                            for i in 0..arg_count {
                                let arg_reg = first_arg + i;
                                if let Some(ty) = Self::get_value_type(&registers[arg_reg as usize])
                                    && !self.is_guarded(arg_reg)
                                {
                                    self.push_op(TraceOp::Guard {
                                        register: arg_reg,
                                        expected_type: ty,
                                    });
                                    self.mark_guarded(arg_reg);
                                }
                            }
                            let mut arg_registers = Vec::with_capacity(total_args);
                            arg_registers.push(obj_reg);
                            for i in 0..arg_count {
                                arg_registers.push(first_arg + i);
                            }
                            self.push_inline_context(
                                function_idx,
                                callee_fn.register_count,
                                dest_reg,
                                obj_reg,
                                first_arg,
                                arg_count,
                                arg_registers,
                                false,
                                None,
                            );
                            return Ok(());
                        }
                    }
                }

                // Normal (non-specialized) method call
                if let Some(ty) = Self::get_value_type(&registers[obj_reg as usize])
                    && !self.is_guarded(obj_reg)
                {
                    self.push_op(TraceOp::Guard {
                        register: obj_reg,
                        expected_type: ty,
                    });
                    self.mark_guarded(obj_reg);
                }

                for i in 0..arg_count {
                    let arg_reg = first_arg + i;
                    if let Some(ty) = Self::get_value_type(&registers[arg_reg as usize])
                        && !self.is_guarded(arg_reg)
                    {
                        self.push_op(TraceOp::Guard {
                            register: arg_reg,
                            expected_type: ty,
                        });
                        self.mark_guarded(arg_reg);
                    }
                }

                // Only receivers that `call_builtin_method_simple` can actually
                // execute may be traced.  It handles arrays, iterators and enums;
                // everything else — ints, floats, bools, strings, maps, and
                // structs, which `jit_call_method_safe` rejects outright — makes
                // the compiled trace fail mid-body.
                //
                // A mid-body failure is unrecoverable: the trace returns -1 with
                // the registers already mutated, and the interpreter then
                // restarts the iteration from the loop header, executing it a
                // second time.  That is how `acc = acc + neg:abs()` over 1..10
                // produced 57 instead of 55.  Refuse the trace instead and let
                // the loop stay interpreted.
                //
                // Lifting this means giving `call_builtin_method_simple` real
                // Int/Float arms, not relaxing the check.
                let receiver_supported = match &registers[obj_reg as usize] {
                    Value::Iterator(_) => true,
                    Value::Enum(object) => {
                        object.enum_name == "Option" || object.enum_name == "Result"
                    }
                    _ => false,
                };
                if !receiver_supported {
                    self.stop_recording();
                    crate::jit::log(|| {
                        format!(
                            "Trace aborted: method '{}' on unsupported receiver in reg {}",
                            method_name, obj_reg
                        )
                    });
                    return Err(LustError::RuntimeError {
                        message: "Trace aborted: method receiver not supported by the JIT"
                            .to_string(),
                    });
                }

                self.specializations_escape("a method call")?;
                self.push_op(TraceOp::CallMethod {
                    dest: dest_reg,
                    object: obj_reg,
                    method_name,
                    first_arg,
                    arg_count,
                });
                Ok(())
            }

            Instruction::GetField(dest, obj_reg, field_name_idx) => {
                let field_name = function.chunk.constants[field_name_idx as usize]
                    .as_string()
                    .unwrap_or("unknown")
                    .to_string();
                let (field_index, is_weak_field) = match &registers[obj_reg as usize] {
                    Value::Struct(object) => {
                        let StructObject { layout, .. } = object.as_ref();
                        let idx = layout.index_of_str(&field_name);
                        let is_weak = idx.map(|i| layout.is_weak(i)).unwrap_or(false);
                        (idx, is_weak)
                    }

                    _ => (None, false),
                };
                if let Some(ty) = Self::get_value_type(&registers[obj_reg as usize])
                    && !self.is_guarded(obj_reg)
                {
                    self.push_op(TraceOp::Guard {
                        register: obj_reg,
                        expected_type: ty,
                    });
                    self.mark_guarded(obj_reg);
                }

                let value_type = Self::get_value_type(&registers[dest as usize]);
                self.push_op(TraceOp::GetField {
                    dest,
                    object: obj_reg,
                    field_name,
                    field_index,
                    value_type,
                    is_weak: is_weak_field,
                });
                Ok(())
            }

            Instruction::SetField(obj_reg, field_name_idx, value_reg) => {
                let field_name = function.chunk.constants[field_name_idx as usize]
                    .as_string()
                    .unwrap_or("unknown")
                    .to_string();
                let (field_index, is_weak_field) = match &registers[obj_reg as usize] {
                    Value::Struct(object) => {
                        let StructObject { layout, .. } = object.as_ref();
                        let idx = layout.index_of_str(&field_name);
                        let is_weak = idx.map(|i| layout.is_weak(i)).unwrap_or(false);
                        (idx, is_weak)
                    }

                    _ => (None, false),
                };
                if let Some(ty) = Self::get_value_type(&registers[obj_reg as usize])
                    && !self.is_guarded(obj_reg)
                {
                    self.push_op(TraceOp::Guard {
                        register: obj_reg,
                        expected_type: ty,
                    });
                    self.mark_guarded(obj_reg);
                }

                let value_type = Self::get_value_type(&registers[value_reg as usize]);
                if let Some(ty) = value_type
                    && !self.is_guarded(value_reg)
                {
                    self.push_op(TraceOp::Guard {
                        register: value_reg,
                        expected_type: ty,
                    });
                    self.mark_guarded(value_reg);
                }

                self.rebox_specialized_register(value_reg, "SetField");

                self.push_op(TraceOp::SetField {
                    object: obj_reg,
                    field_name,
                    value: value_reg,
                    field_index,
                    value_type,
                    is_weak: is_weak_field,
                });
                Ok(())
            }

            Instruction::NewStruct(
                dest,
                struct_name_idx,
                first_field_name_idx,
                first_field_reg,
                field_count,
            ) => {
                let struct_name = function.chunk.constants[struct_name_idx as usize]
                    .as_string()
                    .unwrap_or("unknown")
                    .to_string();
                let mut field_names = Vec::new();
                for i in 0..field_count {
                    let field_name_idx = first_field_name_idx + (i as u16);
                    let field_name = function.chunk.constants[field_name_idx as usize]
                        .as_string()
                        .unwrap_or("unknown")
                        .to_string();
                    field_names.push(field_name);
                }

                let mut field_registers = Vec::new();
                for i in 0..field_count {
                    let field_reg = first_field_reg + i;
                    field_registers.push(field_reg);
                    if let Some(ty) = Self::get_value_type(&registers[field_reg as usize])
                        && !self.is_guarded(field_reg)
                    {
                        self.push_op(TraceOp::Guard {
                            register: field_reg,
                            expected_type: ty,
                        });
                        self.mark_guarded(field_reg);
                    }
                }

                for &field_reg in &field_registers {
                    self.rebox_specialized_register(field_reg, "struct literal field");
                }

                self.push_op(TraceOp::NewStruct {
                    dest,
                    struct_name,
                    field_names,
                    field_registers,
                });
                Ok(())
            }

            Instruction::NewEnumUnit(dest, enum_name_idx, variant_idx) => {
                let enum_name = function.chunk.constants[enum_name_idx as usize]
                    .as_string()
                    .unwrap_or("unknown")
                    .to_string();
                let variant_name = function.chunk.constants[variant_idx as usize]
                    .as_string()
                    .unwrap_or("unknown")
                    .to_string();
                self.push_op(TraceOp::NewEnumUnit {
                    dest,
                    enum_name,
                    variant_name,
                });
                Ok(())
            }

            Instruction::NewEnumVariant(
                dest,
                enum_name_idx,
                variant_idx,
                first_value,
                value_count,
            ) => {
                let enum_name = function.chunk.constants[enum_name_idx as usize]
                    .as_string()
                    .unwrap_or("unknown")
                    .to_string();
                let variant_name = function.chunk.constants[variant_idx as usize]
                    .as_string()
                    .unwrap_or("unknown")
                    .to_string();
                let mut value_registers = Vec::new();
                for i in 0..value_count {
                    value_registers.push(first_value + i);
                }

                for &value_reg in &value_registers {
                    self.rebox_specialized_register(value_reg, "enum variant value");
                }

                self.push_op(TraceOp::NewEnumVariant {
                    dest,
                    enum_name,
                    variant_name,
                    value_registers,
                });
                Ok(())
            }

            Instruction::IsEnumVariant(dest, value_reg, enum_name_idx, variant_idx) => {
                let enum_name = function.chunk.constants[enum_name_idx as usize]
                    .as_string()
                    .unwrap_or("unknown")
                    .to_string();
                let variant_name = function.chunk.constants[variant_idx as usize]
                    .as_string()
                    .unwrap_or("unknown")
                    .to_string();
                self.push_op(TraceOp::IsEnumVariant {
                    dest,
                    value: value_reg,
                    enum_name,
                    variant_name,
                });
                self.mark_guarded(dest);
                Ok(())
            }

            Instruction::TypeIs(dest, value_reg, type_name_idx) => {
                let type_name = function.chunk.constants[type_name_idx as usize]
                    .as_string()
                    .unwrap_or("unknown")
                    .to_string();
                self.push_op(TraceOp::TypeIs {
                    dest,
                    value: value_reg,
                    type_name,
                });
                self.mark_guarded(dest);
                Ok(())
            }

            Instruction::TryCast(dest, value_reg, type_name_idx) => {
                // The cast helper reads the boxed value. Do not emit a consuming
                // Rebox into the cyclic body; discard an unused eager
                // specialization, or abort if specialized mutations came first.
                if !self.disable_unused_array_specialization(value_reg, registers) {
                    self.specialization_escaped = true;
                    self.stop_recording();
                    return Err(LustError::RuntimeError {
                        message: "Trace aborted: cast follows a specialized array mutation"
                            .to_string(),
                    });
                }
                let type_name = function.chunk.constants[type_name_idx as usize]
                    .as_string()
                    .unwrap_or("unknown")
                    .to_string();
                self.push_op(TraceOp::TryCast {
                    dest,
                    value: value_reg,
                    type_name,
                });
                Ok(())
            }

            Instruction::GetEnumValue(dest, enum_reg, index) => {
                self.push_op(TraceOp::GetEnumValue {
                    dest,
                    enum_reg,
                    index,
                });
                Ok(())
            }

            Instruction::Call(func_reg, first_arg, arg_count, dest_reg) => {
                // Rebox specialized value if dest_reg contains one
                self.remove_specialization_tracking(dest_reg);

                match &registers[func_reg as usize] {
                    Value::NativeFunction(native_fn) => {
                        let traced = TracedNativeFn::new(native_fn.clone());
                        if !self.is_guarded(func_reg) {
                            self.push_op(TraceOp::GuardNativeFunction {
                                register: func_reg,
                                function: traced.clone(),
                            });
                            self.mark_guarded(func_reg);
                        }

                        if let Some(op) = self.record_intrinsic_call(
                            Rc::as_ptr(native_fn) as *const () as usize,
                            first_arg,
                            arg_count,
                            dest_reg,
                            registers,
                        ) {
                            return op;
                        }
                        if self.record_array_push(
                            Rc::as_ptr(native_fn) as *const () as usize,
                            first_arg,
                            arg_count,
                            dest_reg,
                            registers,
                        ) {
                            return Ok(());
                        }
                        self.specializations_escape("a native call")?;

                        self.push_op(TraceOp::CallNative {
                            dest: dest_reg,
                            callee: func_reg,
                            function: traced,
                            first_arg,
                            arg_count,
                        });
                        Ok(())
                    }

                    Value::Function(function_idx) => {
                        if !self.is_guarded(func_reg) {
                            self.push_op(TraceOp::GuardFunction {
                                register: func_reg,
                                function_idx: *function_idx,
                            });
                            self.mark_guarded(func_reg);
                        }

                        let mut did_inline = false;
                        if let Some(callee_fn) = functions.get(*function_idx)
                            && self.should_inline(*function_idx, callee_fn)
                            && (arg_count as usize) <= callee_fn.register_count as usize
                        {
                            let mut arg_registers = Vec::with_capacity(arg_count as usize);
                            for i in 0..arg_count {
                                arg_registers.push(first_arg + i);
                            }
                            self.push_inline_context(
                                *function_idx,
                                callee_fn.register_count,
                                dest_reg,
                                func_reg,
                                first_arg,
                                arg_count,
                                arg_registers,
                                false,
                                None,
                            );
                            did_inline = true;
                        }

                        if !did_inline {
                            self.specializations_escape("a call")?;
                            self.push_op(TraceOp::CallFunction {
                                dest: dest_reg,
                                callee: func_reg,
                                function_idx: *function_idx,
                                first_arg,
                                arg_count,
                                is_closure: false,
                                upvalues_ptr: None,
                            });
                        }

                        Ok(())
                    }

                    Value::Closure(closure) => {
                        let function_idx = &closure.function_idx;
                        // The closure's identity: the object every clone shares.
                        let upvalues_ptr = Rc::as_ptr(closure) as *const ();
                        if !self.is_guarded(func_reg) {
                            self.push_op(TraceOp::GuardClosure {
                                register: func_reg,
                                function_idx: *function_idx,
                                upvalues_ptr,
                            });
                            self.mark_guarded(func_reg);
                        }

                        let mut did_inline = false;
                        if let Some(callee_fn) = functions.get(*function_idx)
                            && self.should_inline(*function_idx, callee_fn)
                            && (arg_count as usize) <= callee_fn.register_count as usize
                        {
                            let mut arg_registers = Vec::with_capacity(arg_count as usize);
                            for i in 0..arg_count {
                                arg_registers.push(first_arg + i);
                            }
                            self.push_inline_context(
                                *function_idx,
                                callee_fn.register_count,
                                dest_reg,
                                func_reg,
                                first_arg,
                                arg_count,
                                arg_registers,
                                true,
                                Some(upvalues_ptr),
                            );
                            did_inline = true;
                        }

                        if !did_inline {
                            self.specializations_escape("a closure call")?;
                            self.push_op(TraceOp::CallFunction {
                                dest: dest_reg,
                                callee: func_reg,
                                function_idx: *function_idx,
                                first_arg,
                                arg_count,
                                is_closure: true,
                                upvalues_ptr: Some(upvalues_ptr),
                            });
                        }

                        Ok(())
                    }

                    _ => {
                        self.stop_recording();
                        crate::jit::log(|| {
                            format!(
                                "Trace aborted: unsupported call operation on register {} (value {:?})",
                                func_reg,
                                registers[func_reg as usize].tag()
                            )
                        });
                        Err(LustError::RuntimeError {
                            message: "Trace aborted: unsupported call operation".to_string(),
                        })
                    }
                }
            }

            Instruction::NewArray(dest, first_elem, count) => {
                // Rebox specialized value if dest contains one
                self.remove_specialization_tracking(dest);

                // Disable specialization inside inlined functions for now to avoid
                // register aliasing issues between inline frames and the parent trace.
                if !self.inline_stack.is_empty() {
                    self.push_op(TraceOp::NewArray {
                        dest,
                        first_element: first_elem,
                        count,
                    });
                    return Ok(());
                }

                let element_type = if count == 0 {
                    None
                } else {
                    match &registers[first_elem as usize] {
                        Value::Int(_) => Some(crate::ast::TypeKind::Int),
                        Value::Float(_) => Some(crate::ast::TypeKind::Float),
                        Value::Bool(_) => Some(crate::ast::TypeKind::Bool),
                        _ => None,
                    }
                };

                if let Some(element_type) = element_type {
                    use crate::ast::{Span, Type};
                    let array_type = crate::ast::TypeKind::Array(Box::new(Type::new(
                        element_type.clone(),
                        Span::dummy(),
                    )));

                    let Some(layout) = self.specialization_registry.get_specialization(&array_type)
                    else {
                        self.push_op(TraceOp::NewArray {
                            dest,
                            first_element: first_elem,
                            count,
                        });
                        return Ok(());
                    };

                    crate::jit::log(|| {
                        format!(
                            "🔬 JIT: Specializing NewArray for reg {} with element type {:?}",
                            dest, element_type
                        )
                    });

                    self.push_op(TraceOp::NewArray {
                        dest,
                        first_element: first_elem,
                        count,
                    });

                    // Then unbox it for specialized operations
                    let specialized_id = self.next_specialized_id;
                    self.next_specialized_id += 1;

                    self.push_op(TraceOp::Unbox {
                        specialized_id,
                        source_reg: dest,
                        layout: layout.clone(),
                    });

                    // Track that this register now contains a specialized value
                    self.specialized_registers
                        .insert(dest, (specialized_id, layout));
                } else {
                    // Normal non-specialized array
                    self.push_op(TraceOp::NewArray {
                        dest,
                        first_element: first_elem,
                        count,
                    });
                }
                Ok(())
            }

            Instruction::SetIndex(array, index, value) => {
                if !matches!(registers.get(array as usize), Some(Value::Array(_))) {
                    self.stop_recording();
                    return Err(LustError::RuntimeError {
                        message: "Trace aborted: index assignment currently supports arrays only"
                            .to_string(),
                    });
                }
                if !self.disable_unused_array_specialization(array, registers) {
                    self.specialization_escaped = true;
                    self.stop_recording();
                    return Err(LustError::RuntimeError {
                        message: "Trace aborted: array write follows a specialized mutation"
                            .to_string(),
                    });
                }
                self.rebox_specialized_register(value, "SetIndex");
                for reg in [array, index] {
                    if let Some(ty) = Self::get_value_type(&registers[reg as usize])
                        && !self.is_guarded(reg)
                    {
                        self.push_op(TraceOp::Guard {
                            register: reg,
                            expected_type: ty,
                        });
                        self.mark_guarded(reg);
                    }
                }
                self.push_op(TraceOp::SetIndex {
                    array,
                    index,
                    value,
                });
                Ok(())
            }

            Instruction::NewMap(_) => {
                self.stop_recording();
                Err(LustError::RuntimeError {
                    message: "Trace aborted: unsupported index operation".to_string(),
                })
            }

            Instruction::Return(value_reg) => {
                let return_reg = if value_reg == 255 {
                    None
                } else {
                    Some(value_reg)
                };

                // Rebox any specialized values before return
                if let Some(reg) = return_reg
                    && let Some(&(specialized_id, ref layout)) =
                        self.specialized_registers.get(&reg)
                {
                    crate::jit::log(|| {
                        format!(
                            "📦 JIT: Reboxing specialized #{} in reg {} before return",
                            specialized_id, reg
                        )
                    });

                    self.push_op(TraceOp::Rebox {
                        dest_reg: reg,
                        specialized_id,
                        layout: layout.clone(),
                    });

                    self.specialized_registers.remove(&reg);
                }

                // Ensure no specialized values leak past the return
                self.rebox_all_specialized_values();

                if let Some(ctx) = self.inline_stack.last_mut() {
                    ctx.return_register = return_reg;
                    crate::jit::log(|| {
                        format!(
                            "🔧 JIT: Inline return detected, return_reg={:?}",
                            return_reg
                        )
                    });
                    if let Some(inline_op) = self.finalize_inline_context() {
                        self.push_op(inline_op);
                    }
                    Ok(())
                } else if function_idx == self.trace.function_idx {
                    self.stop_recording();
                    Ok(())
                } else {
                    self.push_op(TraceOp::Return { value: return_reg });
                    Ok(())
                }
            }

            Instruction::Jump(offset) => {
                if offset < 0 {
                    let target_calc = (current_ip as isize) + (offset as isize);
                    if target_calc < 0 {
                        self.stop_recording();
                        Err(LustError::RuntimeError {
                            message: format!(
                                "Invalid jump target: offset={}, current_ip={}, target={}",
                                offset, current_ip, target_calc
                            ),
                        })
                    } else {
                        let jump_target = target_calc as usize;
                        let loop_key = (function_idx, jump_target);

                        // Track how many times we've seen this loop backedge
                        let iteration_count = self.loop_iterations.entry(loop_key).or_insert(0);
                        *iteration_count += 1;

                        if function_idx == self.trace.function_idx
                            && jump_target == self.trace.start_ip
                        {
                            // This is our main trace loop closing - check if we should unroll more
                            if *iteration_count < crate::jit::LOOP_UNROLL_COUNT {
                                crate::jit::log(|| {
                                    format!(
                                        "🔄 JIT: Unrolling main loop (iteration {}/{})",
                                        iteration_count,
                                        crate::jit::LOOP_UNROLL_COUNT
                                    )
                                });
                                // Continue recording to unroll the loop
                                Ok(())
                            } else {
                                crate::jit::log(|| {
                                    format!(
                                        "✅ JIT: Loop unrolled {} times, stopping trace",
                                        iteration_count
                                    )
                                });
                                self.complete_recording();
                                Ok(())
                            }
                        } else if function_idx == self.trace.function_idx
                            && jump_target < self.trace.start_ip
                        {
                            self.stop_recording();
                            Err(LustError::RuntimeError {
                                message:
                                    "Trace aborted: inner-loop recording reached an enclosing backedge"
                                        .to_string(),
                            })
                        } else {
                            // This is a nested loop that should be compiled as a separate trace
                            // Following LuaJIT's approach: don't inline loops, compile them separately
                            let bailout_ip = current_ip.saturating_sub(1);

                            crate::jit::log(|| {
                                format!(
                                    "🔄 JIT: Nested loop detected at func {} ip {} - will call as separate trace",
                                    function_idx, jump_target
                                )
                            });

                            // The inner loop runs through its own trace on
                            // the boxed arrays, so this trace's unboxed
                            // copies would go stale; and a `Rebox` in the
                            // body would empty its slot for the next
                            // iteration's specialized ops. Give them up.
                            self.specializations_escape("a nested loop")?;

                            // The inner loop runs through its own root trace; its
                            // remaining iterations are not recorded here. The
                            // resume ip is patched in when execution leaves it.
                            self.push_op(TraceOp::NestedLoopCall {
                                function_idx,
                                loop_start_ip: jump_target,
                                bailout_ip,
                                resume_ip: 0,
                            });
                            let inline_depth = self.inline_stack.len();
                            let op_index = self.current_ops_len() - 1;
                            self.nested_skip = Some(NestedSkip {
                                function_idx,
                                loop_start_ip: jump_target,
                                backedge_ip: bailout_ip,
                                inline_depth,
                                op_index,
                            });
                            Ok(())
                        }
                    }
                } else {
                    Ok(())
                }
            }

            Instruction::JumpIf(cond, offset) => {
                let condition = &registers[cond as usize];
                let is_truthy = condition.is_truthy();
                let target_offset = (current_ip as isize) + (offset as isize);
                let target = if target_offset < 0 {
                    0
                } else {
                    target_offset as usize
                };
                let bailout_ip = if is_truthy { current_ip } else { target };
                self.push_op(TraceOp::GuardLoopContinue {
                    condition_register: cond,
                    expect_truthy: is_truthy,
                    bailout_ip,
                });
                Ok(())
            }

            Instruction::JumpIfNot(cond, offset) => {
                let condition = &registers[cond as usize];
                let is_truthy = condition.is_truthy();
                let target_offset = (current_ip as isize) + (offset as isize);
                let target = if target_offset < 0 {
                    0
                } else {
                    target_offset as usize
                };
                let bailout_ip = if !is_truthy { current_ip } else { target };
                self.push_op(TraceOp::GuardLoopContinue {
                    condition_register: cond,
                    expect_truthy: is_truthy,
                    bailout_ip,
                });
                Ok(())
            }

            _ => {
                self.stop_recording();
                crate::jit::log(|| {
                    format!(
                        "Trace aborted: unsupported instruction {:?}",
                        instruction.opcode()
                    )
                });
                Err(LustError::RuntimeError {
                    message: "Trace aborted: unsupported instruction".to_string(),
                })
            }
        };

        outcome?;

        if self.overwritten_specialization {
            self.specialization_escaped = true;
            self.stop_recording();
            return Err(LustError::RuntimeError {
                message: "Trace aborted: a written specialized array was overwritten".to_string(),
            });
        }

        if self.op_count >= self.max_length {
            self.stop_recording();
            return Err(LustError::RuntimeError {
                message: "Trace too long".to_string(),
            });
        }

        Ok(())
    }
    fn add_type_guards(
        &mut self,
        lhs: Register,
        rhs: Register,
        registers: &[Value],
        _function: &crate::bytecode::Function,
    ) -> Result<(), LustError> {
        if let Some(ty) = Self::get_value_type(&registers[lhs as usize]) {
            let needs_guard = !self.is_guarded(lhs);
            if needs_guard {
                self.push_op(TraceOp::Guard {
                    register: lhs,
                    expected_type: ty,
                });
                self.mark_guarded(lhs);
            } else {
                self.mark_guarded(lhs);
            }
        }

        if let Some(ty) = Self::get_value_type(&registers[rhs as usize]) {
            let needs_guard = !self.is_guarded(rhs);
            if needs_guard {
                self.push_op(TraceOp::Guard {
                    register: rhs,
                    expected_type: ty,
                });
                self.mark_guarded(rhs);
            } else {
                self.mark_guarded(rhs);
            }
        }

        Ok(())
    }

    fn numeric_comparison_types(
        &mut self,
        lhs: Register,
        rhs: Register,
        registers: &[Value],
    ) -> Result<(ValueType, ValueType), LustError> {
        let types = (
            Self::get_value_type(&registers[lhs as usize]),
            Self::get_value_type(&registers[rhs as usize]),
        );
        match types {
            (
                Some(lhs_type @ (ValueType::Int | ValueType::Float)),
                Some(rhs_type @ (ValueType::Int | ValueType::Float)),
            ) => Ok((lhs_type, rhs_type)),
            _ => {
                self.stop_recording();
                Err(LustError::RuntimeError {
                    message: "Trace aborted: ordered comparison requires numeric operands"
                        .to_string(),
                })
            }
        }
    }

    fn scalar_comparison_types(
        &mut self,
        lhs: Register,
        rhs: Register,
        registers: &[Value],
    ) -> Result<(ValueType, ValueType), LustError> {
        let types = (
            Self::get_value_type(&registers[lhs as usize]),
            Self::get_value_type(&registers[rhs as usize]),
        );
        match types {
            (
                Some(lhs_type @ (ValueType::Int | ValueType::Float | ValueType::Bool)),
                Some(rhs_type @ (ValueType::Int | ValueType::Float | ValueType::Bool)),
            ) => Ok((lhs_type, rhs_type)),
            _ => {
                self.stop_recording();
                Err(LustError::RuntimeError {
                    message: "Trace aborted: equality requires a supported scalar specialization"
                        .to_string(),
                })
            }
        }
    }

    fn get_value_type(value: &Value) -> Option<ValueType> {
        match value {
            Value::Int(_) => Some(ValueType::Int),
            Value::Float(_) => Some(ValueType::Float),
            Value::Bool(_) => Some(ValueType::Bool),
            Value::String(_) => Some(ValueType::String),
            Value::Array(_) => Some(ValueType::Array),
            Value::Tuple(_) => Some(ValueType::Tuple),
            Value::Struct(_) => Some(ValueType::Struct),
            _ => None,
        }
    }

    pub fn finish(mut self) -> Trace {
        #[cfg(feature = "std")]
        if std::env::var("LUST_TRACE_DEBUG").is_ok() {
            eprintln!(
                "🧵 Trace dump (func {}, start_ip {}):",
                self.trace.function_idx, self.trace.start_ip
            );
            if !self.trace.preamble.is_empty() {
                eprintln!("  Preamble:");
                for (idx, op) in self.trace.preamble.iter().enumerate() {
                    eprintln!("    {:03}: {:?}", idx, op);
                }
            }
            eprintln!("  Body:");
            for (idx, op) in self.trace.ops.iter().enumerate() {
                eprintln!("    {:03}: {:?}", idx, op);
            }
            if !self.trace.postamble.is_empty() {
                eprintln!("  Postamble:");
                for (idx, op) in self.trace.postamble.iter().enumerate() {
                    eprintln!("    {:03}: {:?}", idx, op);
                }
            }
        }

        // Finalize before returning (add rebox ops to postamble)
        self.finalize_trace();
        self.trace
    }

    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// The nested loop whose iterations are not being recorded, as
    /// `(function_idx, loop_start_ip)`: its own root trace may run them.
    pub fn skipped_loop(&self) -> Option<(usize, usize)> {
        self.nested_skip
            .map(|skip| (skip.function_idx, skip.loop_start_ip))
    }

    pub fn is_complete(&self) -> bool {
        self.completed
    }

    pub fn abort(&mut self) {
        self.stop_recording();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TypeKind;

    #[test]
    fn global_load_is_a_version_guarded_snapshot() {
        let functions = vec![crate::bytecode::Function::new("global_loop", 0, false)];
        let mut registers = vec![Value::Nil; 4];
        registers[0] = Value::Int(7);
        let mut recorder = TraceRecorder::new(0, 0, 32);
        recorder.globals_version = 5;

        for ip in 1..=2 {
            recorder
                .record_instruction(
                    Instruction::LoadGlobal(0, 0),
                    ip,
                    &registers,
                    &functions[0],
                    0,
                    &functions,
                )
                .unwrap();
        }

        // One guard covers both loads; each load is the value seen.
        let ops: Vec<&TraceOp> = recorder
            .trace
            .ops
            .iter()
            .filter(|op| !matches!(op, TraceOp::At { .. }))
            .collect();
        assert!(matches!(ops[0], TraceOp::GuardGlobals { version: 5 }));
        assert!(matches!(
            ops[1],
            TraceOp::LoadConst {
                dest: 0,
                value: Value::Int(7)
            }
        ));
        assert!(matches!(
            ops[2],
            TraceOp::LoadConst {
                dest: 0,
                value: Value::Int(7)
            }
        ));
        assert_eq!(ops.len(), 3);

        // A store still aborts: it would fail the trace's own guard.
        let result = recorder.record_instruction(
            Instruction::StoreGlobal(0, 0),
            3,
            &registers,
            &functions[0],
            0,
            &functions,
        );
        assert!(result.is_err());
        assert!(!recorder.is_recording());
    }

    #[test]
    fn trace_finalization_only_reboxes_each_specialized_value_once() {
        let mut functions = vec![crate::bytecode::Function::new("array_loop", 0, false)];
        // `specialize_trace_inputs` only considers slots inside the frame, so the
        // frame has to actually claim register 0.
        functions[0].set_register_count(1);
        let mut registers = vec![Value::Nil; 256];
        registers[0] = Value::array(vec![Value::Int(1)]);
        let mut recorder = TraceRecorder::new(0, 0, 32);
        recorder.specialize_trace_inputs(&registers, &functions[0]);

        assert!(
            recorder
                .record_instruction(
                    Instruction::StoreGlobal(0, 1),
                    1,
                    &registers,
                    &functions[0],
                    0,
                    &functions,
                )
                .is_err()
        );
        let trace = recorder.finish();

        assert_eq!(trace.postamble.len(), 1);
        assert!(matches!(trace.postamble[0], TraceOp::Rebox { .. }));
    }

    #[test]
    fn overwritten_register_requires_a_new_type_guard() {
        let functions = vec![crate::bytecode::Function::new("overwrite", 0, false)];
        let mut registers = vec![Value::Nil; 256];
        let mut recorder = TraceRecorder::new(0, 0, 32);

        registers[0] = Value::Int(1);
        recorder
            .record_instruction(
                Instruction::LoadConst(0, 0),
                1,
                &registers,
                &functions[0],
                0,
                &functions,
            )
            .unwrap();
        registers[0] = Value::Float(1.5);
        registers[1] = Value::Float(1.5);
        recorder
            .record_instruction(
                Instruction::Move(0, 1),
                2,
                &registers,
                &functions[0],
                0,
                &functions,
            )
            .unwrap();
        registers[2] = Value::Float(3.0);
        recorder
            .record_instruction(
                Instruction::Add(2, 0, 0),
                3,
                &registers,
                &functions[0],
                0,
                &functions,
            )
            .unwrap();

        assert!(recorder.trace.ops.iter().any(|op| matches!(
            op,
            TraceOp::Guard {
                register: 0,
                expected_type: ValueType::Float
            }
        )));
    }

    #[test]
    fn static_register_type_does_not_replace_a_runtime_guard() {
        let mut function = crate::bytecode::Function::new("typed", 0, false);
        function.register_types.insert(0, TypeKind::Int);
        let functions = vec![function];
        let mut registers = vec![Value::Nil; 256];
        registers[0] = Value::Int(2);
        registers[1] = Value::Int(3);
        registers[2] = Value::Int(5);
        let mut recorder = TraceRecorder::new(0, 0, 32);

        recorder
            .record_instruction(
                Instruction::Add(2, 0, 1),
                1,
                &registers,
                &functions[0],
                0,
                &functions,
            )
            .unwrap();

        assert!(
            recorder
                .trace
                .ops
                .iter()
                .any(|op| matches!(op, TraceOp::Guard { register: 0, .. }))
        );
    }

    #[test]
    fn typed_arithmetic_can_alias_an_input_but_cannot_change_its_contract() {
        let functions = vec![crate::bytecode::Function::new("typed_alias", 0, false)];
        for (instruction, registers, expected) in [
            (
                Instruction::AddInt(0, 0, 1),
                vec![Value::Int(5), Value::Int(3)],
                ValueType::Int,
            ),
            (
                Instruction::AddFloat(0, 0, 1),
                vec![Value::Float(5.0), Value::Float(3.0)],
                ValueType::Float,
            ),
        ] {
            let mut recorder = TraceRecorder::new(0, 0, 32);
            recorder
                .record_instruction(instruction, 1, &registers, &functions[0], 0, &functions)
                .unwrap();
            assert!(recorder.trace.ops.iter().any(|op| matches!(op,
                TraceOp::Guard { register: 0, expected_type } if *expected_type == expected
            )));
            assert!(recorder.trace.ops.iter().any(|op| matches!(op,
                TraceOp::Add { dest: 0, lhs: 0, lhs_type, rhs_type, .. }
                    if *lhs_type == expected && *rhs_type == expected
            )));
        }
        let mut recorder = TraceRecorder::new(0, 0, 32);
        assert!(
            recorder
                .record_instruction(
                    Instruction::AddInt(0, 0, 1),
                    1,
                    &[Value::Float(5.0), Value::Float(3.0)],
                    &functions[0],
                    0,
                    &functions
                )
                .is_err()
        );
        assert!(!recorder.is_recording());
    }

    #[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[test]
    fn compiled_typed_arithmetic_bails_before_reading_wrong_payloads() {
        let functions = vec![crate::bytecode::Function::new("typed_guard", 0, false)];
        let mut recorder = TraceRecorder::new(0, 0, 32);
        recorder
            .record_instruction(
                Instruction::AddInt(0, 0, 1),
                1,
                &[Value::Int(5), Value::Int(3)],
                &functions[0],
                0,
                &functions,
            )
            .unwrap();
        let mut trace = recorder.finish();
        trace.ops.push(TraceOp::GuardLoopContinue {
            condition_register: 2,
            expect_truthy: true,
            bailout_ip: 9,
        });
        let compiled = crate::jit::JitCompiler::new()
            .compile_trace(&trace, crate::jit::TraceId(0), Vec::new())
            .unwrap();
        let mut registers = vec![Value::Int(2), Value::Int(3), Value::Bool(false)];
        compiled.execute(
            registers.as_mut_ptr(),
            core::ptr::null_mut(),
            core::ptr::null(),
        );
        assert_eq!(registers[0], Value::Int(5));
        registers[0] = Value::Float(2.5);
        assert_eq!(
            compiled.execute(
                registers.as_mut_ptr(),
                core::ptr::null_mut(),
                core::ptr::null()
            ),
            1
        );
        assert_eq!(registers[0], Value::Float(2.5));
    }

    #[test]
    fn aliased_arithmetic_aborts_until_inputs_are_recorded_pre_execution() {
        let functions = vec![crate::bytecode::Function::new("aliased", 0, false)];
        let registers = vec![Value::Int(2); 256];
        let mut recorder = TraceRecorder::new(0, 0, 32);

        let result = recorder.record_instruction(
            Instruction::Add(0, 0, 1),
            1,
            &registers,
            &functions[0],
            0,
            &functions,
        );

        assert!(result.is_err());
        assert!(!recorder.is_recording());
        assert!(recorder.trace.ops.is_empty());
    }

    #[test]
    fn empty_array_is_not_specialized_from_stale_register_metadata() {
        let mut function = crate::bytecode::Function::new("empty_array", 0, false);
        function.register_types.insert(0, TypeKind::Int);
        let functions = vec![function];
        let mut registers = vec![Value::Nil; 256];
        registers[0] = Value::array(Vec::new());
        let mut recorder = TraceRecorder::new(0, 0, 32);

        recorder
            .record_instruction(
                Instruction::NewArray(0, 0, 0),
                1,
                &registers,
                &functions[0],
                0,
                &functions,
            )
            .unwrap();

        assert!(matches!(
            recorder.trace.ops.as_slice(),
            [TraceOp::At { .. }, TraceOp::NewArray { count: 0, .. }]
        ));
        assert!(recorder.specialized_registers.is_empty());
    }
}
