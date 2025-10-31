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
    NewTable(Register),
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
    ArrayLen(Register, Register),
    SetIndex(Register, Register, Register),
    Concat(Register, Register, Register),
    CallMethod(Register, ConstIndex, Register, u8, Register),
    TypeIs(Register, Register, ConstIndex),
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
    NewTable,
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
    ArrayLen,
    SetIndex,
    Concat,
    CallMethod,
    TypeIs,
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
            Instruction::NewTable(_) => OpCode::NewTable,
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
            Instruction::ArrayLen(_, _) => OpCode::ArrayLen,
            Instruction::SetIndex(_, _, _) => OpCode::SetIndex,
            Instruction::Concat(_, _, _) => OpCode::Concat,
            Instruction::CallMethod(_, _, _, _, _) => OpCode::CallMethod,
            Instruction::TypeIs(_, _, _) => OpCode::TypeIs,
            Instruction::LoadUpvalue(_, _) => OpCode::LoadUpvalue,
            Instruction::StoreUpvalue(_, _) => OpCode::StoreUpvalue,
            Instruction::Closure(_, _, _, _) => OpCode::Closure,
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
                    write!(
                        f,
                        "Call R{}, R{}..R{}, R{}",
                        func,
                        args,
                        args + cnt - 1,
                        dest
                    )
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
            Instruction::NewTable(r) => write!(f, "NewTable R{}", r),
            Instruction::NewStruct(d, name, field_names, fields, cnt) => {
                write!(
                    f,
                    "NewStruct R{}, K{}, K{}..K{}, R{}..R{}",
                    d,
                    name,
                    field_names,
                    field_names + (*cnt as u16) - 1,
                    fields,
                    fields + cnt - 1
                )
            }

            Instruction::NewEnumUnit(d, enum_name, variant) => {
                write!(f, "NewEnumUnit R{}, K{}, K{}", d, enum_name, variant)
            }

            Instruction::NewEnumVariant(d, enum_name, variant, values, cnt) => {
                write!(
                    f,
                    "NewEnumVariant R{}, K{}, K{}, R{}..R{}",
                    d,
                    enum_name,
                    variant,
                    values,
                    values + cnt - 1
                )
            }

            Instruction::TupleNew(d, first, cnt) => {
                write!(f, "TupleNew R{}, R{}..R{}", d, first, first + cnt - 1)
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
            Instruction::ArrayLen(d, arr) => write!(f, "ArrayLen R{}, R{}", d, arr),
            Instruction::SetIndex(arr, idx, val) => {
                write!(f, "SetIndex R{}, R{}, R{}", arr, idx, val)
            }

            Instruction::Concat(d, l, r) => write!(f, "Concat R{}, R{}, R{}", d, l, r),
            Instruction::CallMethod(obj, method, args, cnt, dest) => {
                write!(
                    f,
                    "CallMethod R{}, K{}, R{}..R{}, R{}",
                    obj,
                    method,
                    args,
                    args + cnt - 1,
                    dest
                )
            }

            Instruction::TypeIs(d, val, type_name) => {
                write!(f, "TypeIs R{}, R{}, K{}", d, val, type_name)
            }

            Instruction::LoadUpvalue(d, idx) => write!(f, "LoadUpvalue R{}, U{}", d, idx),
            Instruction::StoreUpvalue(idx, s) => write!(f, "StoreUpvalue U{}, R{}", idx, s),
            Instruction::Closure(d, func, upvals, cnt) => {
                write!(
                    f,
                    "Closure R{}, F{}, R{}..R{}",
                    d,
                    func,
                    upvals,
                    upvals + cnt - 1
                )
            }
        }
    }
}
