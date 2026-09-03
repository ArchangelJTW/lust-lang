use crate::bytecode::value::NativeFn;
use crate::bytecode::Instruction;
use crate::bytecode::{Register, Value};
use crate::LustError;
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
    function: NativeFn,
}

impl TracedNativeFn {
    pub fn new(function: NativeFn) -> Self {
        Self { function }
    }

    pub fn pointer(&self) -> *const () {
        Rc::as_ptr(&self.function) as *const ()
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
}

#[derive(Debug, Clone)]
pub enum TraceOp {
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
    ArrayIndexOk {
        value_dest: Register,
        condition_dest: Register,
        array: Register,
        index: Register,
    },
    ArrayLen {
        dest: Register,
        array: Register,
    },
    GuardNativeFunction {
        register: Register,
        function: TracedNativeFn,
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
    Guard {
        register: Register,
        expected_type: ValueType,
    },
    GuardLoopContinue {
        condition_register: Register,
        expect_truthy: bool,
        bailout_ip: usize,
    },
    NestedLoopCall {
        function_idx: usize,
        loop_start_ip: usize,
        bailout_ip: usize,
    },
    Return {
        value: Option<Register>,
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
}

impl TraceRecorder {
    pub fn new(function_idx: usize, start_ip: usize, max_length: usize) -> Self {
        Self {
            trace: Trace {
                function_idx,
                start_ip,
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
            op_count: 0,
            specialized_registers: HashMap::new(),
            next_specialized_id: 0,
            specialization_registry: crate::jit::specialization::SpecializationRegistry::new(),
            loop_iterations: HashMap::new(),
            leaked_specialized_values: Vec::new(),
        }
    }

    /// Scan live registers at trace entry and specialize any loop-invariant arrays
    /// This should be called right after trace recording starts
    pub fn specialize_trace_inputs(
        &mut self,
        registers: &[Value],
        function: &crate::bytecode::Function,
    ) {
        crate::jit::log(|| format!("🔍 JIT: Scanning trace inputs for specialization..."));

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
            | TraceOp::GetEnumValue { dest, .. } => Some(*dest),
            TraceOp::SetField { .. }
            | TraceOp::ArrayIndexOk { .. }
            | TraceOp::Guard { .. }
            | TraceOp::GuardNativeFunction { .. }
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

    fn push_op(&mut self, op: TraceOp) {
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

    /// Remove specialization tracking if register is about to be overwritten
    /// The Vec data becomes "leaked" on the JIT stack but that's fine - it's cleaned
    /// up when the stack frame is destroyed. The array is still managed by Rc<RefCell<>>.
    fn remove_specialization_tracking(&mut self, register: Register) {
        if let Some((_specialized_id, _layout)) = self.specialized_registers.remove(&register) {
            crate::jit::log(|| {
                format!(
                    "🗑️  JIT: Removing specialization tracking for reg {} (being overwritten)",
                    register
                )
            });
            // Don't emit rebox - the Vec data stays on JIT stack but that's OK
            // It will be cleaned up when the JIT stack frame is destroyed
        }
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

        self.specialized_registers.remove(&register);
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
        let owner =
            if self.specialized_registers.contains_key(&register) {
                Some(register)
            } else if let Some(Value::Array(array)) = registers.get(register as usize) {
                self.specialized_registers.keys().copied().find(|candidate| {
                matches!(
                    registers.get(*candidate as usize),
                    Some(Value::Array(candidate_array)) if Rc::ptr_eq(array, candidate_array)
                )
            })
            } else {
                None
            };

        owner.map_or(true, |owner| self.disable_unused_specialization(owner))
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
        });
    }

    fn finalize_inline_context(&mut self) -> Option<TraceOp> {
        let context = self.inline_stack.pop()?;
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
        )
    }

    pub fn record_instruction_at_frame(
        &mut self,
        frame_index: usize,
        instruction: Instruction,
        current_ip: usize,
        registers: &[Value],
        function: &crate::bytecode::Function,
        function_idx: usize,
        functions: &[crate::bytecode::Function],
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

        if let Some(dest) = instruction.defined_register() {
            if instruction.reads_register(dest) {
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

        let outcome: Result<(), LustError> = match instruction {
            Instruction::LoadConst(dest, _) => {
                // Rebox specialized value if dest contains one
                self.remove_specialization_tracking(dest);

                if let Some(_ty) = Self::get_value_type(&registers[dest as usize]) {
                    self.mark_guarded(dest);
                }

                self.push_op(TraceOp::LoadConst {
                    dest,
                    value: registers[dest as usize].clone(),
                });
                Ok(())
            }

            Instruction::LoadGlobal(_, _) | Instruction::StoreGlobal(_, _) => {
                // Globals can be replaced by Lust or embedding code. Until
                // traces carry global slots and versions, snapshotting or
                // omitting these operations would miscompile the loop.
                self.stop_recording();
                Err(LustError::RuntimeError {
                    message: "Trace aborted: global access requires versioning".to_string(),
                })
            }

            Instruction::Move(dest, src) => {
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

                self.push_op(TraceOp::Move { dest, src });
                if let Some((specialized_id, layout)) = moved_specialization {
                    // `push_op` invalidates the destination first; transfer the
                    // specialization only after that generic invalidation.
                    self.specialized_registers.remove(&src);
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
                if let Some(ty) = Self::get_value_type(&registers[lhs as usize]) {
                    if !self.is_guarded(lhs) {
                        self.push_op(TraceOp::Guard {
                            register: lhs,
                            expected_type: ty,
                        });
                        self.mark_guarded(lhs);
                    }
                }

                if let Some(ty) = Self::get_value_type(&registers[rhs as usize]) {
                    if !self.is_guarded(rhs) {
                        self.push_op(TraceOp::Guard {
                            register: rhs,
                            expected_type: ty,
                        });
                        self.mark_guarded(rhs);
                    }
                }

                self.push_op(TraceOp::Concat { dest, lhs, rhs });
                Ok(())
            }

            Instruction::GetIndex(dest, array, index) => {
                if !self.disable_unused_array_specialization(array, registers) {
                    self.stop_recording();
                    return Err(LustError::RuntimeError {
                        message: "Trace aborted: array read follows a specialized mutation"
                            .to_string(),
                    });
                }
                if let Some(ty) = Self::get_value_type(&registers[array as usize]) {
                    if !self.is_guarded(array) {
                        self.push_op(TraceOp::Guard {
                            register: array,
                            expected_type: ty,
                        });
                        self.mark_guarded(array);
                    }
                }

                if let Some(ty) = Self::get_value_type(&registers[index as usize]) {
                    if !self.is_guarded(index) {
                        self.push_op(TraceOp::Guard {
                            register: index,
                            expected_type: ty,
                        });
                        self.mark_guarded(index);
                    }
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
                    self.stop_recording();
                    return Err(LustError::RuntimeError {
                        message: "Trace aborted: checked read follows a specialized array mutation"
                            .to_string(),
                    });
                }
                if let Some(ty) = Self::get_value_type(&registers[array as usize]) {
                    if !self.is_guarded(array) {
                        self.push_op(TraceOp::Guard {
                            register: array,
                            expected_type: ty,
                        });
                        self.mark_guarded(array);
                    }
                }

                if let Some(ty) = Self::get_value_type(&registers[index as usize]) {
                    if !self.is_guarded(index) {
                        self.push_op(TraceOp::Guard {
                            register: index,
                            expected_type: ty,
                        });
                        self.mark_guarded(index);
                    }
                }

                self.push_op(TraceOp::TryGetIndex { dest, array, index });
                Ok(())
            }

            Instruction::ArrayLen(dest, array) => {
                if let Some(ty) = Self::get_value_type(&registers[array as usize]) {
                    if !self.is_guarded(array) {
                        self.push_op(TraceOp::Guard {
                            register: array,
                            expected_type: ty,
                        });
                        self.mark_guarded(array);
                    }
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
                            if let Some(ty) = Self::get_value_type(&registers[value_reg as usize]) {
                                if !self.is_guarded(value_reg) {
                                    self.push_op(TraceOp::Guard {
                                        register: value_reg,
                                        expected_type: ty,
                                    });
                                    self.mark_guarded(value_reg);
                                }
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
                            // Other methods on specialized values - need to rebox first
                            // For now, fall through to normal handling (will be wrong!)
                            crate::jit::log(|| {
                                format!(
                                    "⚠️  JIT: Method '{}' on specialized value not supported, will be incorrect!",
                                    method_name
                                )
                            });
                        }
                    }
                }

                // Normal (non-specialized) method call
                if let Some(ty) = Self::get_value_type(&registers[obj_reg as usize]) {
                    if !self.is_guarded(obj_reg) {
                        self.push_op(TraceOp::Guard {
                            register: obj_reg,
                            expected_type: ty,
                        });
                        self.mark_guarded(obj_reg);
                    }
                }

                for i in 0..arg_count {
                    let arg_reg = first_arg + i;
                    if let Some(ty) = Self::get_value_type(&registers[arg_reg as usize]) {
                        if !self.is_guarded(arg_reg) {
                            self.push_op(TraceOp::Guard {
                                register: arg_reg,
                                expected_type: ty,
                            });
                            self.mark_guarded(arg_reg);
                        }
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
                    Value::Enum { enum_name, .. } => enum_name == "Option" || enum_name == "Result",
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
                    Value::Struct { layout, .. } => {
                        let idx = layout.index_of_str(&field_name);
                        let is_weak = idx.map(|i| layout.is_weak(i)).unwrap_or(false);
                        (idx, is_weak)
                    }

                    _ => (None, false),
                };
                if let Some(ty) = Self::get_value_type(&registers[obj_reg as usize]) {
                    if !self.is_guarded(obj_reg) {
                        self.push_op(TraceOp::Guard {
                            register: obj_reg,
                            expected_type: ty,
                        });
                        self.mark_guarded(obj_reg);
                    }
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
                    Value::Struct { layout, .. } => {
                        let idx = layout.index_of_str(&field_name);
                        let is_weak = idx.map(|i| layout.is_weak(i)).unwrap_or(false);
                        (idx, is_weak)
                    }

                    _ => (None, false),
                };
                if let Some(ty) = Self::get_value_type(&registers[obj_reg as usize]) {
                    if !self.is_guarded(obj_reg) {
                        self.push_op(TraceOp::Guard {
                            register: obj_reg,
                            expected_type: ty,
                        });
                        self.mark_guarded(obj_reg);
                    }
                }

                let value_type = Self::get_value_type(&registers[value_reg as usize]);
                if let Some(ty) = value_type {
                    if !self.is_guarded(value_reg) {
                        self.push_op(TraceOp::Guard {
                            register: value_reg,
                            expected_type: ty,
                        });
                        self.mark_guarded(value_reg);
                    }
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
                    if let Some(ty) = Self::get_value_type(&registers[field_reg as usize]) {
                        if !self.is_guarded(field_reg) {
                            self.push_op(TraceOp::Guard {
                                register: field_reg,
                                expected_type: ty,
                            });
                            self.mark_guarded(field_reg);
                        }
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
                        if let Some(callee_fn) = functions.get(*function_idx) {
                            if self.should_inline(*function_idx, callee_fn)
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
                        }

                        if !did_inline {
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

                    Value::Closure {
                        function_idx,
                        upvalues,
                    } => {
                        let upvalues_ptr = Rc::as_ptr(upvalues) as *const ();
                        if !self.is_guarded(func_reg) {
                            self.push_op(TraceOp::GuardClosure {
                                register: func_reg,
                                function_idx: *function_idx,
                                upvalues_ptr,
                            });
                            self.mark_guarded(func_reg);
                        }

                        let mut did_inline = false;
                        if let Some(callee_fn) = functions.get(*function_idx) {
                            if self.should_inline(*function_idx, callee_fn)
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
                        }

                        if !did_inline {
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

            Instruction::NewMap(_) | Instruction::SetIndex(_, _, _) => {
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
                if let Some(reg) = return_reg {
                    if let Some(&(specialized_id, ref layout)) =
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

                            // Rebox all specialized values before calling nested trace
                            self.rebox_all_specialized_values();

                            // Emit NestedLoopCall which will eventually call the compiled inner trace
                            self.push_op(TraceOp::NestedLoopCall {
                                function_idx,
                                loop_start_ip: jump_target,
                                bailout_ip,
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
            Value::Struct { .. } => Some(ValueType::Struct),
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
    fn global_access_aborts_recording_instead_of_becoming_a_constant() {
        let functions = vec![crate::bytecode::Function::new("global_loop", 0, false)];
        let mut recorder = TraceRecorder::new(0, 0, 32);

        let result = recorder.record_instruction(
            Instruction::LoadGlobal(0, 0),
            1,
            &[],
            &functions[0],
            0,
            &functions,
        );

        assert!(result.is_err());
        assert!(!recorder.is_recording());
        assert!(recorder.trace.ops.is_empty());
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

        assert!(recorder
            .record_instruction(
                Instruction::LoadGlobal(1, 0),
                1,
                &registers,
                &functions[0],
                0,
                &functions,
            )
            .is_err());
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

        assert!(recorder
            .trace
            .ops
            .iter()
            .any(|op| matches!(op, TraceOp::Guard { register: 0, .. })));
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
            [TraceOp::NewArray { count: 0, .. }]
        ));
        assert!(recorder.specialized_registers.is_empty());
    }
}
