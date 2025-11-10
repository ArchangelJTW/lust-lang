use crate::bytecode::value::NativeFn;
use crate::bytecode::Instruction;
use crate::bytecode::{Register, Value};
use crate::LustError;
use alloc::{
    format,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};
use core::fmt;
use hashbrown::HashSet;

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
    pub ops: Vec<TraceOp>,
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
    },
    Ne {
        dest: Register,
        lhs: Register,
        rhs: Register,
    },
    Lt {
        dest: Register,
        lhs: Register,
        rhs: Register,
    },
    Le {
        dest: Register,
        lhs: Register,
        rhs: Register,
    },
    Gt {
        dest: Register,
        lhs: Register,
        rhs: Register,
    },
    Ge {
        dest: Register,
        lhs: Register,
        rhs: Register,
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
    guarded_registers: HashSet<Register>,
    inline_stack: Vec<InlineContext>,
    op_count: usize,
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
                ops: Vec::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
            },
            max_length,
            recording: true,
            guarded_registers: HashSet::new(),
            inline_stack: Vec::new(),
            op_count: 0,
        }
    }

    fn current_function_idx(&self) -> usize {
        self.inline_stack
            .last()
            .map(|ctx| ctx.function_idx)
            .unwrap_or(self.trace.function_idx)
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

    fn push_op(&mut self, op: TraceOp) {
        self.op_count += 1;
        if let Some(ctx) = self.inline_stack.last_mut() {
            ctx.ops.push(op);
        } else {
            self.trace.ops.push(op);
        }
    }

    fn should_inline(&self, function_idx: usize) -> bool {
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
        registers: &[Value; 256],
        function: &crate::bytecode::Function,
        function_idx: usize,
        functions: &[crate::bytecode::Function],
    ) -> Result<(), LustError> {
        if !self.recording {
            return Ok(());
        }

        if function_idx != self.current_function_idx() {
            return Ok(());
        }

        let outcome: Result<(), LustError> = match instruction {
            Instruction::LoadConst(dest, _) => {
                if let Some(_ty) = Self::get_value_type(&registers[dest as usize]) {
                    self.mark_guarded(dest);
                }

                self.push_op(TraceOp::LoadConst {
                    dest,
                    value: registers[dest as usize].clone(),
                });
                Ok(())
            }

            Instruction::LoadGlobal(dest, _) => {
                if let Some(_ty) = Self::get_value_type(&registers[dest as usize]) {
                    self.mark_guarded(dest);
                }

                self.push_op(TraceOp::LoadConst {
                    dest,
                    value: registers[dest as usize].clone(),
                });
                Ok(())
            }

            Instruction::StoreGlobal(_, _) => Ok(()),

            Instruction::Move(dest, src) => {
                self.push_op(TraceOp::Move { dest, src });
                Ok(())
            }

            Instruction::Add(dest, lhs, rhs) => {
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
                Ok(())
            }

            Instruction::Sub(dest, lhs, rhs) => {
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
                Ok(())
            }

            Instruction::Neg(dest, src) => {
                self.push_op(TraceOp::Neg { dest, src });
                Ok(())
            }

            Instruction::Eq(dest, lhs, rhs) => {
                self.push_op(TraceOp::Eq { dest, lhs, rhs });
                Ok(())
            }

            Instruction::Ne(dest, lhs, rhs) => {
                self.push_op(TraceOp::Ne { dest, lhs, rhs });
                Ok(())
            }

            Instruction::Lt(dest, lhs, rhs) => {
                self.push_op(TraceOp::Lt { dest, lhs, rhs });
                Ok(())
            }

            Instruction::Le(dest, lhs, rhs) => {
                self.push_op(TraceOp::Le { dest, lhs, rhs });
                Ok(())
            }

            Instruction::Gt(dest, lhs, rhs) => {
                self.push_op(TraceOp::Gt { dest, lhs, rhs });
                Ok(())
            }

            Instruction::Ge(dest, lhs, rhs) => {
                self.push_op(TraceOp::Ge { dest, lhs, rhs });
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
                Ok(())
            }

            Instruction::CallMethod(obj_reg, method_name_idx, first_arg, arg_count, dest_reg) => {
                let method_name = function.chunk.constants[method_name_idx as usize]
                    .as_string()
                    .unwrap_or("unknown")
                    .to_string();
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
                            if self.should_inline(*function_idx)
                                && (arg_count as usize)
                                    <= callee_fn.register_count as usize
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
                            if self.should_inline(*function_idx)
                                && (arg_count as usize)
                                    <= callee_fn.register_count as usize
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
                        self.recording = false;
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

            Instruction::NewArray(_, _, _)
            | Instruction::NewMap(_)
            | Instruction::SetIndex(_, _, _) => {
                self.recording = false;
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

                if let Some(ctx) = self.inline_stack.last_mut() {
                    ctx.return_register = return_reg;
                    if let Some(inline_op) = self.finalize_inline_context() {
                        self.push_op(inline_op);
                    }
                    Ok(())
                } else if function_idx == self.trace.function_idx {
                    self.recording = false;
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
                        self.recording = false;
                        Err(LustError::RuntimeError {
                            message: format!(
                                "Invalid jump target: offset={}, current_ip={}, target={}",
                                offset, current_ip, target_calc
                            ),
                        })
                    } else {
                        let jump_target = target_calc as usize;
                        if function_idx == self.trace.function_idx
                            && jump_target == self.trace.start_ip
                        {
                            self.recording = false;
                            Ok(())
                        } else {
                            let bailout_ip = current_ip.saturating_sub(1);
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

            Instruction::JumpIf(_, _) | Instruction::JumpIfNot(_, _) => Ok(()),

            _ => {
                self.recording = false;
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
            self.recording = false;
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
        registers: &[Value; 256],
        function: &crate::bytecode::Function,
    ) -> Result<(), LustError> {
        if let Some(ty) = Self::get_value_type(&registers[lhs as usize]) {
            let needs_guard = if self.is_guarded(lhs) {
                false
            } else if let Some(static_type) = function.register_types.get(&lhs) {
                !Self::type_kind_matches_value_type(static_type, ty)
            } else {
                true
            };
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
            let needs_guard = if self.is_guarded(rhs) {
                false
            } else if let Some(static_type) = function.register_types.get(&rhs) {
                !Self::type_kind_matches_value_type(static_type, ty)
            } else {
                true
            };
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

    fn type_kind_matches_value_type(
        type_kind: &crate::ast::TypeKind,
        value_type: ValueType,
    ) -> bool {
        use crate::ast::TypeKind;
        match (type_kind, value_type) {
            (TypeKind::Int, ValueType::Int) => true,
            (TypeKind::Float, ValueType::Float) => true,
            (TypeKind::Bool, ValueType::Bool) => true,
            (TypeKind::String, ValueType::String) => true,
            (TypeKind::Array(_), ValueType::Array) => true,
            (TypeKind::Tuple(_), ValueType::Tuple) => true,
            _ => false,
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

    pub fn finish(self) -> Trace {
        self.trace
    }

    pub fn is_recording(&self) -> bool {
        self.recording
    }

    pub fn abort(&mut self) {
        self.recording = false;
    }
}
