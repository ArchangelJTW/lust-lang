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
    CallNative {
        dest: Register,
        callee: Register,
        function: TracedNativeFn,
        first_arg: Register,
        arg_count: u8,
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
        }
    }

    pub fn record_instruction(
        &mut self,
        instruction: Instruction,
        current_ip: usize,
        registers: &[Value; 256],
        function: &crate::bytecode::Function,
        function_idx: usize,
    ) -> Result<(), LustError> {
        if !self.recording {
            return Ok(());
        }

        if function_idx != self.trace.function_idx {
            return Ok(());
        }

        let trace_op = match instruction {
            Instruction::LoadConst(dest, _) => {
                if let Some(_ty) = Self::get_value_type(&registers[dest as usize]) {
                    self.guarded_registers.insert(dest);
                }

                TraceOp::LoadConst {
                    dest,
                    value: registers[dest as usize].clone(),
                }
            }

            Instruction::LoadGlobal(dest, _) => {
                if let Some(_ty) = Self::get_value_type(&registers[dest as usize]) {
                    self.guarded_registers.insert(dest);
                }

                TraceOp::LoadConst {
                    dest,
                    value: registers[dest as usize].clone(),
                }
            }

            Instruction::StoreGlobal(_, _) => {
                return Ok(());
            }

            Instruction::Move(dest, src) => TraceOp::Move { dest, src },
            Instruction::Add(dest, lhs, rhs) => {
                self.add_type_guards(lhs, rhs, registers, function)?;
                let lhs_type =
                    Self::get_value_type(&registers[lhs as usize]).unwrap_or(ValueType::Int);
                let rhs_type =
                    Self::get_value_type(&registers[rhs as usize]).unwrap_or(ValueType::Int);
                TraceOp::Add {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                }
            }

            Instruction::Sub(dest, lhs, rhs) => {
                self.add_type_guards(lhs, rhs, registers, function)?;
                let lhs_type =
                    Self::get_value_type(&registers[lhs as usize]).unwrap_or(ValueType::Int);
                let rhs_type =
                    Self::get_value_type(&registers[rhs as usize]).unwrap_or(ValueType::Int);
                TraceOp::Sub {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                }
            }

            Instruction::Mul(dest, lhs, rhs) => {
                self.add_type_guards(lhs, rhs, registers, function)?;
                let lhs_type =
                    Self::get_value_type(&registers[lhs as usize]).unwrap_or(ValueType::Int);
                let rhs_type =
                    Self::get_value_type(&registers[rhs as usize]).unwrap_or(ValueType::Int);
                TraceOp::Mul {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                }
            }

            Instruction::Div(dest, lhs, rhs) => {
                self.add_type_guards(lhs, rhs, registers, function)?;
                let lhs_type =
                    Self::get_value_type(&registers[lhs as usize]).unwrap_or(ValueType::Int);
                let rhs_type =
                    Self::get_value_type(&registers[rhs as usize]).unwrap_or(ValueType::Int);
                TraceOp::Div {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                }
            }

            Instruction::Mod(dest, lhs, rhs) => {
                self.add_type_guards(lhs, rhs, registers, function)?;
                let lhs_type =
                    Self::get_value_type(&registers[lhs as usize]).unwrap_or(ValueType::Int);
                let rhs_type =
                    Self::get_value_type(&registers[rhs as usize]).unwrap_or(ValueType::Int);
                TraceOp::Mod {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                }
            }

            Instruction::Neg(dest, src) => TraceOp::Neg { dest, src },
            Instruction::Eq(dest, lhs, rhs) => TraceOp::Eq { dest, lhs, rhs },
            Instruction::Ne(dest, lhs, rhs) => TraceOp::Ne { dest, lhs, rhs },
            Instruction::Lt(dest, lhs, rhs) => TraceOp::Lt { dest, lhs, rhs },
            Instruction::Le(dest, lhs, rhs) => TraceOp::Le { dest, lhs, rhs },
            Instruction::Gt(dest, lhs, rhs) => TraceOp::Gt { dest, lhs, rhs },
            Instruction::Ge(dest, lhs, rhs) => TraceOp::Ge { dest, lhs, rhs },
            Instruction::And(dest, lhs, rhs) => TraceOp::And { dest, lhs, rhs },
            Instruction::Or(dest, lhs, rhs) => TraceOp::Or { dest, lhs, rhs },
            Instruction::Not(dest, src) => TraceOp::Not { dest, src },
            Instruction::Concat(dest, lhs, rhs) => {
                if let Some(ty) = Self::get_value_type(&registers[lhs as usize]) {
                    if !self.guarded_registers.contains(&lhs) {
                        self.trace.ops.push(TraceOp::Guard {
                            register: lhs,
                            expected_type: ty,
                        });
                        self.guarded_registers.insert(lhs);
                    }
                }

                if let Some(ty) = Self::get_value_type(&registers[rhs as usize]) {
                    if !self.guarded_registers.contains(&rhs) {
                        self.trace.ops.push(TraceOp::Guard {
                            register: rhs,
                            expected_type: ty,
                        });
                        self.guarded_registers.insert(rhs);
                    }
                }

                TraceOp::Concat { dest, lhs, rhs }
            }

            Instruction::GetIndex(dest, array, index) => {
                if let Some(ty) = Self::get_value_type(&registers[array as usize]) {
                    if !self.guarded_registers.contains(&array) {
                        self.trace.ops.push(TraceOp::Guard {
                            register: array,
                            expected_type: ty,
                        });
                        self.guarded_registers.insert(array);
                    }
                }

                if let Some(ty) = Self::get_value_type(&registers[index as usize]) {
                    if !self.guarded_registers.contains(&index) {
                        self.trace.ops.push(TraceOp::Guard {
                            register: index,
                            expected_type: ty,
                        });
                        self.guarded_registers.insert(index);
                    }
                }

                TraceOp::GetIndex { dest, array, index }
            }

            Instruction::ArrayLen(dest, array) => {
                if let Some(ty) = Self::get_value_type(&registers[array as usize]) {
                    if !self.guarded_registers.contains(&array) {
                        self.trace.ops.push(TraceOp::Guard {
                            register: array,
                            expected_type: ty,
                        });
                        self.guarded_registers.insert(array);
                    }
                }

                TraceOp::ArrayLen { dest, array }
            }

            Instruction::CallMethod(obj_reg, method_name_idx, first_arg, arg_count, dest_reg) => {
                let method_name = function.chunk.constants[method_name_idx as usize]
                    .as_string()
                    .unwrap_or("unknown")
                    .to_string();
                if let Some(ty) = Self::get_value_type(&registers[obj_reg as usize]) {
                    if !self.guarded_registers.contains(&obj_reg) {
                        self.trace.ops.push(TraceOp::Guard {
                            register: obj_reg,
                            expected_type: ty,
                        });
                        self.guarded_registers.insert(obj_reg);
                    }
                }

                for i in 0..arg_count {
                    let arg_reg = first_arg + i;
                    if let Some(ty) = Self::get_value_type(&registers[arg_reg as usize]) {
                        if !self.guarded_registers.contains(&arg_reg) {
                            self.trace.ops.push(TraceOp::Guard {
                                register: arg_reg,
                                expected_type: ty,
                            });
                            self.guarded_registers.insert(arg_reg);
                        }
                    }
                }

                TraceOp::CallMethod {
                    dest: dest_reg,
                    object: obj_reg,
                    method_name,
                    first_arg,
                    arg_count,
                }
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
                    if !self.guarded_registers.contains(&obj_reg) {
                        self.trace.ops.push(TraceOp::Guard {
                            register: obj_reg,
                            expected_type: ty,
                        });
                        self.guarded_registers.insert(obj_reg);
                    }
                }

                let value_type = Self::get_value_type(&registers[dest as usize]);
                TraceOp::GetField {
                    dest,
                    object: obj_reg,
                    field_name,
                    field_index,
                    value_type,
                    is_weak: is_weak_field,
                }
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
                    if !self.guarded_registers.contains(&obj_reg) {
                        self.trace.ops.push(TraceOp::Guard {
                            register: obj_reg,
                            expected_type: ty,
                        });
                        self.guarded_registers.insert(obj_reg);
                    }
                }

                let value_type = Self::get_value_type(&registers[value_reg as usize]);
                if let Some(ty) = value_type {
                    if !self.guarded_registers.contains(&value_reg) {
                        self.trace.ops.push(TraceOp::Guard {
                            register: value_reg,
                            expected_type: ty,
                        });
                        self.guarded_registers.insert(value_reg);
                    }
                }

                TraceOp::SetField {
                    object: obj_reg,
                    field_name,
                    value: value_reg,
                    field_index,
                    value_type,
                    is_weak: is_weak_field,
                }
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
                        if !self.guarded_registers.contains(&field_reg) {
                            self.trace.ops.push(TraceOp::Guard {
                                register: field_reg,
                                expected_type: ty,
                            });
                            self.guarded_registers.insert(field_reg);
                        }
                    }
                }

                TraceOp::NewStruct {
                    dest,
                    struct_name,
                    field_names,
                    field_registers,
                }
            }

            Instruction::Call(func_reg, first_arg, arg_count, dest_reg) => {
                match &registers[func_reg as usize] {
                    Value::NativeFunction(native_fn) => {
                        let traced = TracedNativeFn::new(native_fn.clone());
                        if !self.guarded_registers.contains(&func_reg) {
                            self.trace.ops.push(TraceOp::GuardNativeFunction {
                                register: func_reg,
                                function: traced.clone(),
                            });
                            self.guarded_registers.insert(func_reg);
                        }

                        self.trace.ops.push(TraceOp::CallNative {
                            dest: dest_reg,
                            callee: func_reg,
                            function: traced,
                            first_arg,
                            arg_count,
                        });
                        return Ok(());
                    }

                    _ => {
                        self.recording = false;
                        return Err(LustError::RuntimeError {
                            message: "Trace aborted: unsupported operation".to_string(),
                        });
                    }
                }
            }

            Instruction::NewArray(_, _, _)
            | Instruction::NewMap(_)
            | Instruction::SetIndex(_, _, _) => {
                self.recording = false;
                return Err(LustError::RuntimeError {
                    message: "Trace aborted: unsupported operation".to_string(),
                });
            }

            Instruction::Return(value_reg) => {
                if function_idx == self.trace.function_idx {
                    self.recording = false;
                    return Ok(());
                } else {
                    TraceOp::Return {
                        value: if value_reg == 255 {
                            None
                        } else {
                            Some(value_reg)
                        },
                    }
                }
            }

            Instruction::Jump(offset) => {
                if offset < 0 {
                    let target_calc = (current_ip as isize) + (offset as isize);
                    if target_calc < 0 {
                        self.recording = false;
                        return Err(LustError::RuntimeError {
                            message: format!(
                                "Invalid jump target: offset={}, current_ip={}, target={}",
                                offset, current_ip, target_calc
                            ),
                        });
                    }

                    let jump_target = target_calc as usize;
                    if function_idx == self.trace.function_idx && jump_target == self.trace.start_ip
                    {
                        self.recording = false;
                        return Ok(());
                    } else {
                        let bailout_ip = current_ip.saturating_sub(1);
                        TraceOp::NestedLoopCall {
                            function_idx,
                            loop_start_ip: jump_target,
                            bailout_ip,
                        }
                    }
                } else {
                    return Ok(());
                }
            }

            Instruction::JumpIf(_cond, _) | Instruction::JumpIfNot(_cond, _) => {
                return Ok(());
            }

            _ => {
                self.recording = false;
                return Err(LustError::RuntimeError {
                    message: "Trace aborted: unsupported instruction".to_string(),
                });
            }
        };
        self.trace.ops.push(trace_op);
        if self.trace.ops.len() >= self.max_length {
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
            let needs_guard = if self.guarded_registers.contains(&lhs) {
                false
            } else if let Some(static_type) = function.register_types.get(&lhs) {
                !Self::type_kind_matches_value_type(static_type, ty)
            } else {
                true
            };
            if needs_guard {
                self.trace.ops.push(TraceOp::Guard {
                    register: lhs,
                    expected_type: ty,
                });
                self.guarded_registers.insert(lhs);
            } else {
                self.guarded_registers.insert(lhs);
            }
        }

        if let Some(ty) = Self::get_value_type(&registers[rhs as usize]) {
            let needs_guard = if self.guarded_registers.contains(&rhs) {
                false
            } else if let Some(static_type) = function.register_types.get(&rhs) {
                !Self::type_kind_matches_value_type(static_type, ty)
            } else {
                true
            };
            if needs_guard {
                self.trace.ops.push(TraceOp::Guard {
                    register: rhs,
                    expected_type: ty,
                });
                self.guarded_registers.insert(rhs);
            } else {
                self.guarded_registers.insert(rhs);
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
