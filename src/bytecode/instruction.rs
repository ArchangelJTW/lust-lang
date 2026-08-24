use core::fmt;
pub type Register = u8;
pub type ConstIndex = u16;
pub type JumpOffset = i16;
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Instruction {
    LoadNil(Register),
    LoadBool(Register, bool),
    LoadConst(Register, ConstIndex),
    LoadGlobal(Register, ConstIndex),
    StoreGlobal(ConstIndex, Register),
    Move(Register, Register),
    Add(Register, Register, Register),
    Sub(Register, Register, Register),
    Mul(Register, Register, Register),
    Div(Register, Register, Register),
    Mod(Register, Register, Register),
    Neg(Register, Register),
    Eq(Register, Register, Register),
    Ne(Register, Register, Register),
    Lt(Register, Register, Register),
    Le(Register, Register, Register),
    Gt(Register, Register, Register),
    Ge(Register, Register, Register),
    And(Register, Register, Register),
    Or(Register, Register, Register),
    Not(Register, Register),
    Jump(JumpOffset),
    JumpIf(Register, JumpOffset),
    JumpIfNot(Register, JumpOffset),
    Call(Register, Register, u8, Register),
    Return(Register),
    NewArray(Register, Register, u8),
    NewMap(Register),
    NewStruct(Register, ConstIndex, ConstIndex, Register, u8),
    NewEnumUnit(Register, ConstIndex, ConstIndex),
    NewEnumVariant(Register, ConstIndex, ConstIndex, Register, u8),
    TupleNew(Register, Register, u8),
    TupleGet(Register, Register, u8),
    IsEnumVariant(Register, Register, ConstIndex, ConstIndex),
    GetEnumValue(Register, Register, u8),
    GetField(Register, Register, ConstIndex),
    SetField(Register, ConstIndex, Register),
    GetIndex(Register, Register, Register),
    TryGetIndex(Register, Register, Register),
    ArrayLen(Register, Register),
    SetIndex(Register, Register, Register),
    Concat(Register, Register, Register),
    CallMethod(Register, ConstIndex, Register, u8, Register),
    TypeIs(Register, Register, ConstIndex),
    TryCast(Register, Register, ConstIndex),
    LoadUpvalue(Register, u8),
    StoreUpvalue(u8, Register),
    Closure(Register, ConstIndex, Register, u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpCode {
    LoadNil,
    LoadBool,
    LoadConst,
    LoadGlobal,
    StoreGlobal,
    Move,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Neg,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    Not,
    Jump,
    JumpIf,
    JumpIfNot,
    Call,
    Return,
    NewArray,
    NewMap,
    NewStruct,
    NewEnumUnit,
    NewEnumVariant,
    TupleNew,
    TupleGet,
    IsEnumVariant,
    GetEnumValue,
    GetField,
    SetField,
    GetIndex,
    TryGetIndex,
    ArrayLen,
    SetIndex,
    Concat,
    CallMethod,
    TypeIs,
    TryCast,
    LoadUpvalue,
    StoreUpvalue,
    Closure,
}

impl Instruction {
    pub fn opcode(&self) -> OpCode {
        match self {
            Instruction::LoadNil(_) => OpCode::LoadNil,
            Instruction::LoadBool(_, _) => OpCode::LoadBool,
            Instruction::LoadConst(_, _) => OpCode::LoadConst,
            Instruction::LoadGlobal(_, _) => OpCode::LoadGlobal,
            Instruction::StoreGlobal(_, _) => OpCode::StoreGlobal,
            Instruction::Move(_, _) => OpCode::Move,
            Instruction::Add(_, _, _) => OpCode::Add,
            Instruction::Sub(_, _, _) => OpCode::Sub,
            Instruction::Mul(_, _, _) => OpCode::Mul,
            Instruction::Div(_, _, _) => OpCode::Div,
            Instruction::Mod(_, _, _) => OpCode::Mod,
            Instruction::Neg(_, _) => OpCode::Neg,
            Instruction::Eq(_, _, _) => OpCode::Eq,
            Instruction::Ne(_, _, _) => OpCode::Ne,
            Instruction::Lt(_, _, _) => OpCode::Lt,
            Instruction::Le(_, _, _) => OpCode::Le,
            Instruction::Gt(_, _, _) => OpCode::Gt,
            Instruction::Ge(_, _, _) => OpCode::Ge,
            Instruction::And(_, _, _) => OpCode::And,
            Instruction::Or(_, _, _) => OpCode::Or,
            Instruction::Not(_, _) => OpCode::Not,
            Instruction::Jump(_) => OpCode::Jump,
            Instruction::JumpIf(_, _) => OpCode::JumpIf,
            Instruction::JumpIfNot(_, _) => OpCode::JumpIfNot,
            Instruction::Call(_, _, _, _) => OpCode::Call,
            Instruction::Return(_) => OpCode::Return,
            Instruction::NewArray(_, _, _) => OpCode::NewArray,
            Instruction::NewMap(_) => OpCode::NewMap,
            Instruction::NewStruct(_, _, _, _, _) => OpCode::NewStruct,
            Instruction::NewEnumUnit(_, _, _) => OpCode::NewEnumUnit,
            Instruction::NewEnumVariant(_, _, _, _, _) => OpCode::NewEnumVariant,
            Instruction::TupleNew(_, _, _) => OpCode::TupleNew,
            Instruction::TupleGet(_, _, _) => OpCode::TupleGet,
            Instruction::IsEnumVariant(_, _, _, _) => OpCode::IsEnumVariant,
            Instruction::GetEnumValue(_, _, _) => OpCode::GetEnumValue,
            Instruction::GetField(_, _, _) => OpCode::GetField,
            Instruction::SetField(_, _, _) => OpCode::SetField,
            Instruction::GetIndex(_, _, _) => OpCode::GetIndex,
            Instruction::TryGetIndex(_, _, _) => OpCode::TryGetIndex,
            Instruction::ArrayLen(_, _) => OpCode::ArrayLen,
            Instruction::SetIndex(_, _, _) => OpCode::SetIndex,
            Instruction::Concat(_, _, _) => OpCode::Concat,
            Instruction::CallMethod(_, _, _, _, _) => OpCode::CallMethod,
            Instruction::TypeIs(_, _, _) => OpCode::TypeIs,
            Instruction::TryCast(_, _, _) => OpCode::TryCast,
            Instruction::LoadUpvalue(_, _) => OpCode::LoadUpvalue,
            Instruction::StoreUpvalue(_, _) => OpCode::StoreUpvalue,
            Instruction::Closure(_, _, _, _) => OpCode::Closure,
        }
    }

    pub fn defined_register(&self) -> Option<Register> {
        match *self {
            Instruction::LoadNil(dest)
            | Instruction::LoadBool(dest, _)
            | Instruction::LoadConst(dest, _)
            | Instruction::LoadGlobal(dest, _)
            | Instruction::Move(dest, _)
            | Instruction::Add(dest, _, _)
            | Instruction::Sub(dest, _, _)
            | Instruction::Mul(dest, _, _)
            | Instruction::Div(dest, _, _)
            | Instruction::Mod(dest, _, _)
            | Instruction::Neg(dest, _)
            | Instruction::Eq(dest, _, _)
            | Instruction::Ne(dest, _, _)
            | Instruction::Lt(dest, _, _)
            | Instruction::Le(dest, _, _)
            | Instruction::Gt(dest, _, _)
            | Instruction::Ge(dest, _, _)
            | Instruction::And(dest, _, _)
            | Instruction::Or(dest, _, _)
            | Instruction::Not(dest, _)
            | Instruction::Call(_, _, _, dest)
            | Instruction::NewArray(dest, _, _)
            | Instruction::NewMap(dest)
            | Instruction::NewStruct(dest, _, _, _, _)
            | Instruction::NewEnumUnit(dest, _, _)
            | Instruction::NewEnumVariant(dest, _, _, _, _)
            | Instruction::TupleNew(dest, _, _)
            | Instruction::TupleGet(dest, _, _)
            | Instruction::IsEnumVariant(dest, _, _, _)
            | Instruction::GetEnumValue(dest, _, _)
            | Instruction::GetField(dest, _, _)
            | Instruction::GetIndex(dest, _, _)
            | Instruction::TryGetIndex(dest, _, _)
            | Instruction::ArrayLen(dest, _)
            | Instruction::Concat(dest, _, _)
            | Instruction::CallMethod(_, _, _, _, dest)
            | Instruction::TypeIs(dest, _, _)
            | Instruction::TryCast(dest, _, _)
            | Instruction::LoadUpvalue(dest, _)
            | Instruction::Closure(dest, _, _, _) => Some(dest),
            Instruction::StoreGlobal(_, _)
            | Instruction::Jump(_)
            | Instruction::JumpIf(_, _)
            | Instruction::JumpIfNot(_, _)
            | Instruction::Return(_)
            | Instruction::SetField(_, _, _)
            | Instruction::SetIndex(_, _, _)
            | Instruction::StoreUpvalue(_, _) => None,
        }
    }

    pub fn reads_register(&self, register: Register) -> bool {
        let in_range = |first: Register, count: u8| {
            register >= first && (register as usize) < first as usize + count as usize
        };
        match *self {
            Instruction::Move(_, src)
            | Instruction::Neg(_, src)
            | Instruction::Not(_, src)
            | Instruction::JumpIf(src, _)
            | Instruction::JumpIfNot(src, _)
            | Instruction::Return(src)
            | Instruction::TupleGet(_, src, _)
            | Instruction::GetEnumValue(_, src, _)
            | Instruction::ArrayLen(_, src)
            | Instruction::StoreGlobal(_, src)
            | Instruction::StoreUpvalue(_, src) => register == src,
            Instruction::Add(_, lhs, rhs)
            | Instruction::Sub(_, lhs, rhs)
            | Instruction::Mul(_, lhs, rhs)
            | Instruction::Div(_, lhs, rhs)
            | Instruction::Mod(_, lhs, rhs)
            | Instruction::Eq(_, lhs, rhs)
            | Instruction::Ne(_, lhs, rhs)
            | Instruction::Lt(_, lhs, rhs)
            | Instruction::Le(_, lhs, rhs)
            | Instruction::Gt(_, lhs, rhs)
            | Instruction::Ge(_, lhs, rhs)
            | Instruction::And(_, lhs, rhs)
            | Instruction::Or(_, lhs, rhs)
            | Instruction::GetIndex(_, lhs, rhs)
            | Instruction::TryGetIndex(_, lhs, rhs)
            | Instruction::Concat(_, lhs, rhs) => register == lhs || register == rhs,
            Instruction::SetIndex(collection, index, value) => {
                register == collection || register == index || register == value
            }
            Instruction::SetField(object, _, value) => register == object || register == value,
            Instruction::IsEnumVariant(_, value, _, _)
            | Instruction::GetField(_, value, _)
            | Instruction::TypeIs(_, value, _)
            | Instruction::TryCast(_, value, _) => register == value,
            Instruction::Call(function, first, count, _) => {
                register == function || in_range(first, count)
            }
            Instruction::CallMethod(object, _, first, count, _) => {
                register == object || in_range(first, count)
            }
            Instruction::NewArray(_, first, count)
            | Instruction::TupleNew(_, first, count)
            | Instruction::Closure(_, _, first, count) => in_range(first, count),
            Instruction::NewStruct(_, _, _, first, count)
            | Instruction::NewEnumVariant(_, _, _, first, count) => in_range(first, count),
            Instruction::LoadNil(_)
            | Instruction::LoadBool(_, _)
            | Instruction::LoadConst(_, _)
            | Instruction::LoadGlobal(_, _)
            | Instruction::Jump(_)
            | Instruction::NewMap(_)
            | Instruction::NewEnumUnit(_, _, _)
            | Instruction::LoadUpvalue(_, _) => false,
        }
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::LoadNil(r) => write!(f, "LoadNil R{}", r),
            Instruction::LoadBool(r, b) => write!(f, "LoadBool R{}, {}", r, b),
            Instruction::LoadConst(r, c) => write!(f, "LoadConst R{}, K{}", r, c),
            Instruction::LoadGlobal(r, c) => write!(f, "LoadGlobal R{}, K{}", r, c),
            Instruction::StoreGlobal(c, r) => write!(f, "StoreGlobal K{}, R{}", c, r),
            Instruction::Move(d, s) => write!(f, "Move R{}, R{}", d, s),
            Instruction::Add(d, l, r) => write!(f, "Add R{}, R{}, R{}", d, l, r),
            Instruction::Sub(d, l, r) => write!(f, "Sub R{}, R{}, R{}", d, l, r),
            Instruction::Mul(d, l, r) => write!(f, "Mul R{}, R{}, R{}", d, l, r),
            Instruction::Div(d, l, r) => write!(f, "Div R{}, R{}, R{}", d, l, r),
            Instruction::Mod(d, l, r) => write!(f, "Mod R{}, R{}, R{}", d, l, r),
            Instruction::Neg(d, s) => write!(f, "Neg R{}, R{}", d, s),
            Instruction::Eq(d, l, r) => write!(f, "Eq R{}, R{}, R{}", d, l, r),
            Instruction::Ne(d, l, r) => write!(f, "Ne R{}, R{}, R{}", d, l, r),
            Instruction::Lt(d, l, r) => write!(f, "Lt R{}, R{}, R{}", d, l, r),
            Instruction::Le(d, l, r) => write!(f, "Le R{}, R{}, R{}", d, l, r),
            Instruction::Gt(d, l, r) => write!(f, "Gt R{}, R{}, R{}", d, l, r),
            Instruction::Ge(d, l, r) => write!(f, "Ge R{}, R{}, R{}", d, l, r),
            Instruction::And(d, l, r) => write!(f, "And R{}, R{}, R{}", d, l, r),
            Instruction::Or(d, l, r) => write!(f, "Or R{}, R{}, R{}", d, l, r),
            Instruction::Not(d, s) => write!(f, "Not R{}, R{}", d, s),
            Instruction::Jump(offset) => write!(f, "Jump {}", offset),
            Instruction::JumpIf(r, offset) => write!(f, "JumpIf R{}, {}", r, offset),
            Instruction::JumpIfNot(r, offset) => write!(f, "JumpIfNot R{}, {}", r, offset),
            Instruction::Call(func, args, cnt, dest) => {
                if *cnt == 0 {
                    write!(f, "Call R{}, <no args>, R{}", func, dest)
                } else {
                    let arg_start = *args as usize;
                    let arg_end = arg_start + (*cnt as usize) - 1;
                    write!(f, "Call R{}, R{}..R{}, R{}", func, args, arg_end, dest)
                }
            }

            Instruction::Return(r) => write!(f, "Return R{}", r),
            Instruction::NewArray(d, elems, cnt) => {
                if *cnt == 0 {
                    write!(f, "NewArray R{}, <no elements>", d)
                } else {
                    let end = (*elems as usize) + (*cnt as usize) - 1;
                    write!(f, "NewArray R{}, R{}..R{}", d, elems, end)
                }
            }

            Instruction::NewMap(r) => write!(f, "NewMap R{}", r),
            Instruction::NewStruct(d, name, field_names, fields, cnt) => {
                if *cnt == 0 {
                    write!(
                        f,
                        "NewStruct R{}, K{}, <no fields>, R{}..R{}",
                        d, name, fields, fields
                    )
                } else {
                    let field_name_start = *field_names as usize;
                    let field_name_end = field_name_start + (*cnt as usize) - 1;
                    let value_start = *fields as usize;
                    let value_end = value_start + (*cnt as usize) - 1;
                    write!(
                        f,
                        "NewStruct R{}, K{}, K{}..K{}, R{}..R{}",
                        d, name, field_names, field_name_end, fields, value_end
                    )
                }
            }

            Instruction::NewEnumUnit(d, enum_name, variant) => {
                write!(f, "NewEnumUnit R{}, K{}, K{}", d, enum_name, variant)
            }

            Instruction::NewEnumVariant(d, enum_name, variant, values, cnt) => {
                if *cnt == 0 {
                    write!(
                        f,
                        "NewEnumVariant R{}, K{}, K{}, <no values>",
                        d, enum_name, variant
                    )
                } else {
                    let value_start = *values as usize;
                    let value_end = value_start + (*cnt as usize) - 1;
                    write!(
                        f,
                        "NewEnumVariant R{}, K{}, K{}, R{}..R{}",
                        d, enum_name, variant, values, value_end
                    )
                }
            }

            Instruction::TupleNew(d, first, cnt) => {
                if *cnt == 0 {
                    write!(f, "TupleNew R{}, <empty>", d)
                } else {
                    let start = *first as usize;
                    let end = start + (*cnt as usize) - 1;
                    write!(f, "TupleNew R{}, R{}..R{}", d, first, end)
                }
            }

            Instruction::TupleGet(d, tuple, idx) => {
                write!(f, "TupleGet R{}, R{}, {}", d, tuple, idx)
            }

            Instruction::IsEnumVariant(d, val, enum_name, variant) => {
                write!(
                    f,
                    "IsEnumVariant R{}, R{}, K{}, K{}",
                    d, val, enum_name, variant
                )
            }

            Instruction::GetEnumValue(d, enum_reg, idx) => {
                write!(f, "GetEnumValue R{}, R{}, {}", d, enum_reg, idx)
            }

            Instruction::GetField(d, obj, field) => {
                write!(f, "GetField R{}, R{}, K{}", d, obj, field)
            }

            Instruction::SetField(obj, field, val) => {
                write!(f, "SetField R{}, K{}, R{}", obj, field, val)
            }

            Instruction::GetIndex(d, arr, idx) => write!(f, "GetIndex R{}, R{}, R{}", d, arr, idx),
            Instruction::TryGetIndex(d, arr, idx) => {
                write!(f, "TryGetIndex R{}, R{}, R{}", d, arr, idx)
            }
            Instruction::ArrayLen(d, arr) => write!(f, "ArrayLen R{}, R{}", d, arr),
            Instruction::SetIndex(arr, idx, val) => {
                write!(f, "SetIndex R{}, R{}, R{}", arr, idx, val)
            }

            Instruction::Concat(d, l, r) => write!(f, "Concat R{}, R{}, R{}", d, l, r),
            Instruction::CallMethod(obj, method, args, cnt, dest) => {
                if *cnt == 0 {
                    write!(f, "CallMethod R{}, K{}, <no args>, R{}", obj, method, dest)
                } else {
                    let arg_start = *args as usize;
                    let arg_end = arg_start + (*cnt as usize) - 1;
                    write!(
                        f,
                        "CallMethod R{}, K{}, R{}..R{}, R{}",
                        obj, method, args, arg_end, dest
                    )
                }
            }

            Instruction::TypeIs(d, val, type_name) => {
                write!(f, "TypeIs R{}, R{}, K{}", d, val, type_name)
            }

            Instruction::TryCast(dest, value, type_name) => {
                write!(f, "TryCast R{}, R{}, K{}", dest, value, type_name)
            }

            Instruction::LoadUpvalue(d, idx) => write!(f, "LoadUpvalue R{}, U{}", d, idx),
            Instruction::StoreUpvalue(idx, s) => write!(f, "StoreUpvalue U{}, R{}", idx, s),
            Instruction::Closure(d, func, upvals, cnt) => {
                if *cnt == 0 {
                    write!(f, "Closure R{}, F{}, <no upvalues>", d, func)
                } else {
                    let start = *upvals as usize;
                    let end = start + (*cnt as usize) - 1;
                    write!(f, "Closure R{}, F{}, R{}..R{}", d, func, upvals, end)
                }
            }
        }
    }
}
