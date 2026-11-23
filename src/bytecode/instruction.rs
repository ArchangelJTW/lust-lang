use crate::number::NumericType;
use alloc::vec::Vec;
use core::fmt;
pub type Register = u8;
pub type ConstIndex = u16;
pub type JumpOffset = i16;

/// Compact 4-byte packed bytecode instruction for L1d-friendly VM dispatch.
///
/// Bit layouts:
/// - `ABC`:  bits `[0..8] = op`, `[8..16] = a`, `[16..24] = b`, `[24..32] = c`
/// - `ABx`:  bits `[0..8] = op`, `[8..16] = a`, `[16..32] = bx` (`u16`)
/// - `AsBx`: bits `[0..8] = op`, `[8..16] = a`, `[16..32] = sbx` (`i16` as `u16`)
/// - `AX`:   `ABx` where `bx` indexes into `Chunk::extra_args: Vec<u32>`
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedInstruction(pub u32);

const _: () = assert!(core::mem::size_of::<PackedInstruction>() == 4);
const _: () = assert!(core::mem::align_of::<PackedInstruction>() == 4);

impl PackedInstruction {
    /// Format ABC: bits [0..8] = op, [8..16] = a, [16..24] = b, [24..32] = c
    #[inline(always)]
    pub const fn encode_abc(op: OpCode, a: u8, b: u8, c: u8) -> Self {
        Self((op as u32) | ((a as u32) << 8) | ((b as u32) << 16) | ((c as u32) << 24))
    }

    /// Format ABx: bits [0..8] = op, [8..16] = a, [16..32] = bx (u16)
    #[inline(always)]
    pub const fn encode_abx(op: OpCode, a: u8, bx: u16) -> Self {
        Self((op as u32) | ((a as u32) << 8) | ((bx as u32) << 16))
    }

    /// Format AsBx: bits [0..8] = op, [8..16] = a, [16..32] = sbx (i16 as u16)
    #[inline(always)]
    pub const fn encode_asbx(op: OpCode, a: u8, sbx: i16) -> Self {
        Self::encode_abx(op, a, sbx as u16)
    }

    #[inline(always)]
    pub const fn opcode_u8(self) -> u8 {
        (self.0 & 0xFF) as u8
    }

    #[inline(always)]
    pub const fn a(self) -> u8 {
        ((self.0 >> 8) & 0xFF) as u8
    }

    #[inline(always)]
    pub const fn b(self) -> u8 {
        ((self.0 >> 16) & 0xFF) as u8
    }

    #[inline(always)]
    pub const fn c(self) -> u8 {
        (self.0 >> 24) as u8
    }

    #[inline(always)]
    pub const fn c_i8(self) -> i8 {
        ((self.0 >> 24) as u8) as i8
    }

    #[inline(always)]
    pub const fn bx(self) -> u16 {
        (self.0 >> 16) as u16
    }

    #[inline(always)]
    pub const fn sbx(self) -> i16 {
        ((self.0 >> 16) as u16) as i16
    }

    /// Logical gas weight matching `Instruction::gas_weight()`.
    #[inline(always)]
    pub const fn gas_weight(self) -> u16 {
        let op = self.opcode_u8();
        if op == OpCode::BranchLtInt as u8 || op == OpCode::BranchLeInt as u8 {
            2
        } else if op == OpCode::GetFieldAddSetInt as u8 {
            3
        } else {
            1
        }
    }

    /// Returns the register operand and the field-name constant index.
    #[inline(always)]
    pub(crate) fn field_operands(self, extra_args: &[u32]) -> (Register, ConstIndex) {
        let op = self.opcode_u8();
        if op == OpCode::GetFieldCompact as u8 || op == OpCode::SetFieldCompact as u8 {
            (self.b(), u16::from(self.c()))
        } else {
            let word = extra_args[self.bx() as usize];
            (word as u8, (word >> 16) as u16)
        }
    }

    #[inline]
    fn push_extra1(extra_args: &mut Vec<u32>, w0: u32) -> Option<u16> {
        let idx = extra_args.len();
        if idx > u16::MAX as usize {
            return None;
        }
        extra_args.push(w0);
        Some(idx as u16)
    }

    #[inline]
    fn push_extra2(extra_args: &mut Vec<u32>, w0: u32, w1: u32) -> Option<u16> {
        let idx = extra_args.len();
        if idx > u16::MAX as usize {
            return None;
        }
        extra_args.push(w0);
        extra_args.push(w1);
        Some(idx as u16)
    }

    /// Encodes an `Instruction` into a 4-byte `PackedInstruction`, appending 0, 1, or 2
    /// extension words to `extra_args` for `AX`-format opcodes.
    ///
    /// Returns `None` if `extra_args` cannot be indexed by a `u16` (`extra_args.len() > u16::MAX`).
    pub fn from_instruction(inst: Instruction, extra_args: &mut Vec<u32>) -> Option<Self> {
        let packed = match inst {
            // Direct ABC format (0 extra_args words)
            Instruction::LoadNil(dest) => Self::encode_abc(OpCode::LoadNil, dest, 0, 0),
            Instruction::LoadBool(dest, val) => {
                Self::encode_abc(OpCode::LoadBool, dest, if val { 1 } else { 0 }, 0)
            }
            Instruction::Move(dest, src) => Self::encode_abc(OpCode::Move, dest, src, 0),
            Instruction::Return(reg) => Self::encode_abc(OpCode::Return, reg, 0, 0),
            Instruction::NewMap(dest) => Self::encode_abc(OpCode::NewMap, dest, 0, 0),
            Instruction::Neg(d, s) => Self::encode_abc(OpCode::Neg, d, s, 0),
            Instruction::NegInt(d, s) => Self::encode_abc(OpCode::NegInt, d, s, 0),
            Instruction::NegFloat(d, s) => Self::encode_abc(OpCode::NegFloat, d, s, 0),
            Instruction::Not(d, s) => Self::encode_abc(OpCode::Not, d, s, 0),
            Instruction::BitNot(d, s) => Self::encode_abc(OpCode::BitNot, d, s, 0),
            Instruction::ArrayLen(d, s) => Self::encode_abc(OpCode::ArrayLen, d, s, 0),
            Instruction::ArrayPush(d, s) => Self::encode_abc(OpCode::ArrayPush, d, s, 0),
            Instruction::LoadUpvalue(dest, idx) => {
                Self::encode_abc(OpCode::LoadUpvalue, dest, idx, 0)
            }
            Instruction::StoreUpvalue(idx, src) => {
                Self::encode_abc(OpCode::StoreUpvalue, idx, src, 0)
            }
            Instruction::Add(d, l, r) => Self::encode_abc(OpCode::Add, d, l, r),
            Instruction::Sub(d, l, r) => Self::encode_abc(OpCode::Sub, d, l, r),
            Instruction::Mul(d, l, r) => Self::encode_abc(OpCode::Mul, d, l, r),
            Instruction::Div(d, l, r) => Self::encode_abc(OpCode::Div, d, l, r),
            Instruction::Mod(d, l, r) => Self::encode_abc(OpCode::Mod, d, l, r),
            Instruction::Pow(d, l, r) => Self::encode_abc(OpCode::Pow, d, l, r),
            Instruction::FloorDiv(d, l, r) => Self::encode_abc(OpCode::FloorDiv, d, l, r),
            Instruction::Eq(d, l, r) => Self::encode_abc(OpCode::Eq, d, l, r),
            Instruction::Ne(d, l, r) => Self::encode_abc(OpCode::Ne, d, l, r),
            Instruction::Lt(d, l, r) => Self::encode_abc(OpCode::Lt, d, l, r),
            Instruction::Le(d, l, r) => Self::encode_abc(OpCode::Le, d, l, r),
            Instruction::Gt(d, l, r) => Self::encode_abc(OpCode::Gt, d, l, r),
            Instruction::Ge(d, l, r) => Self::encode_abc(OpCode::Ge, d, l, r),
            Instruction::AddInt(d, l, r) => Self::encode_abc(OpCode::AddInt, d, l, r),
            Instruction::SubInt(d, l, r) => Self::encode_abc(OpCode::SubInt, d, l, r),
            Instruction::MulInt(d, l, r) => Self::encode_abc(OpCode::MulInt, d, l, r),
            Instruction::DivInt(d, l, r) => Self::encode_abc(OpCode::DivInt, d, l, r),
            Instruction::ModInt(d, l, r) => Self::encode_abc(OpCode::ModInt, d, l, r),
            Instruction::PowInt(d, l, r) => Self::encode_abc(OpCode::PowInt, d, l, r),
            Instruction::FloorDivInt(d, l, r) => Self::encode_abc(OpCode::FloorDivInt, d, l, r),
            Instruction::EqInt(d, l, r) => Self::encode_abc(OpCode::EqInt, d, l, r),
            Instruction::NeInt(d, l, r) => Self::encode_abc(OpCode::NeInt, d, l, r),
            Instruction::LtInt(d, l, r) => Self::encode_abc(OpCode::LtInt, d, l, r),
            Instruction::LeInt(d, l, r) => Self::encode_abc(OpCode::LeInt, d, l, r),
            Instruction::GtInt(d, l, r) => Self::encode_abc(OpCode::GtInt, d, l, r),
            Instruction::GeInt(d, l, r) => Self::encode_abc(OpCode::GeInt, d, l, r),
            Instruction::AddFloat(d, l, r) => Self::encode_abc(OpCode::AddFloat, d, l, r),
            Instruction::SubFloat(d, l, r) => Self::encode_abc(OpCode::SubFloat, d, l, r),
            Instruction::MulFloat(d, l, r) => Self::encode_abc(OpCode::MulFloat, d, l, r),
            Instruction::DivFloat(d, l, r) => Self::encode_abc(OpCode::DivFloat, d, l, r),
            Instruction::ModFloat(d, l, r) => Self::encode_abc(OpCode::ModFloat, d, l, r),
            Instruction::PowFloat(d, l, r) => Self::encode_abc(OpCode::PowFloat, d, l, r),
            Instruction::FloorDivFloat(d, l, r) => Self::encode_abc(OpCode::FloorDivFloat, d, l, r),
            Instruction::EqFloat(d, l, r) => Self::encode_abc(OpCode::EqFloat, d, l, r),
            Instruction::NeFloat(d, l, r) => Self::encode_abc(OpCode::NeFloat, d, l, r),
            Instruction::LtFloat(d, l, r) => Self::encode_abc(OpCode::LtFloat, d, l, r),
            Instruction::LeFloat(d, l, r) => Self::encode_abc(OpCode::LeFloat, d, l, r),
            Instruction::GtFloat(d, l, r) => Self::encode_abc(OpCode::GtFloat, d, l, r),
            Instruction::GeFloat(d, l, r) => Self::encode_abc(OpCode::GeFloat, d, l, r),
            Instruction::And(d, l, r) => Self::encode_abc(OpCode::And, d, l, r),
            Instruction::Or(d, l, r) => Self::encode_abc(OpCode::Or, d, l, r),
            Instruction::BitAnd(d, l, r) => Self::encode_abc(OpCode::BitAnd, d, l, r),
            Instruction::BitOr(d, l, r) => Self::encode_abc(OpCode::BitOr, d, l, r),
            Instruction::BitXor(d, l, r) => Self::encode_abc(OpCode::BitXor, d, l, r),
            Instruction::Shl(d, l, r) => Self::encode_abc(OpCode::Shl, d, l, r),
            Instruction::Shr(d, l, r) => Self::encode_abc(OpCode::Shr, d, l, r),
            Instruction::NewArray(d, first, cnt) => Self::encode_abc(OpCode::NewArray, d, first, cnt),
            Instruction::TupleNew(d, first, cnt) => Self::encode_abc(OpCode::TupleNew, d, first, cnt),
            Instruction::TupleGet(d, tuple, idx) => Self::encode_abc(OpCode::TupleGet, d, tuple, idx),
            Instruction::GetEnumValue(d, enum_reg, idx) => {
                Self::encode_abc(OpCode::GetEnumValue, d, enum_reg, idx)
            }
            Instruction::GetFieldSlot(d, obj, slot) => {
                Self::encode_abc(OpCode::GetFieldSlot, d, obj, slot)
            }
            Instruction::SetFieldSlot(obj, slot, val) => {
                Self::encode_abc(OpCode::SetFieldSlot, obj, slot, val)
            }
            Instruction::GetIndex(d, arr, idx) => Self::encode_abc(OpCode::GetIndex, d, arr, idx),
            Instruction::TryGetIndex(d, arr, idx) => {
                Self::encode_abc(OpCode::TryGetIndex, d, arr, idx)
            }
            Instruction::ArrayGetUnwrap(d, arr, idx) => {
                Self::encode_abc(OpCode::ArrayGetUnwrap, d, arr, idx)
            }
            Instruction::SetIndex(arr, idx, val) => {
                Self::encode_abc(OpCode::SetIndex, arr, idx, val)
            }
            Instruction::Concat(d, l, r) => Self::encode_abc(OpCode::Concat, d, l, r),
            Instruction::AddIntImm(d, src, imm) => {
                Self::encode_abc(OpCode::AddIntImm, d, src, imm as u8)
            }
            Instruction::GetFieldAddSetInt(obj, slot, addend) => {
                Self::encode_abc(OpCode::GetFieldAddSetInt, obj, slot, addend)
            }

            // Direct ABx / AsBx format (0 extra_args words)
            Instruction::LoadConst(dest, idx) => Self::encode_abx(OpCode::LoadConst, dest, idx),
            Instruction::LoadGlobal(dest, idx) => Self::encode_abx(OpCode::LoadGlobal, dest, idx),
            Instruction::StoreGlobal(idx, src) => Self::encode_abx(OpCode::StoreGlobal, src, idx),
            Instruction::Jump(offset) => Self::encode_asbx(OpCode::Jump, 0, offset),
            Instruction::JumpIf(reg, offset) => Self::encode_asbx(OpCode::JumpIf, reg, offset),
            Instruction::JumpIfNot(reg, offset) => {
                Self::encode_asbx(OpCode::JumpIfNot, reg, offset)
            }

            // AX format: 1 word in extra_args
            Instruction::BranchLtInt(lhs, rhs, off) => {
                let w0 = u32::from(rhs) | (u32::from(off as u16) << 16);
                let ex = Self::push_extra1(extra_args, w0)?;
                Self::encode_abx(OpCode::BranchLtInt, lhs, ex)
            }
            Instruction::BranchLeInt(lhs, rhs, off) => {
                let w0 = u32::from(rhs) | (u32::from(off as u16) << 16);
                let ex = Self::push_extra1(extra_args, w0)?;
                Self::encode_abx(OpCode::BranchLeInt, lhs, ex)
            }
            Instruction::Call(func, args, cnt, dest) => {
                let w0 = u32::from(args) | (u32::from(cnt) << 8) | (u32::from(dest) << 16);
                let ex = Self::push_extra1(extra_args, w0)?;
                Self::encode_abx(OpCode::Call, func, ex)
            }
            Instruction::ArrayIndexOk(val, cond, arr, idx) => {
                let w0 = u32::from(cond) | (u32::from(arr) << 8) | (u32::from(idx) << 16);
                let ex = Self::push_extra1(extra_args, w0)?;
                Self::encode_abx(OpCode::ArrayIndexOk, val, ex)
            }
            Instruction::GetField(dest, obj, field_idx) if field_idx <= u16::from(u8::MAX) => {
                Self::encode_abc(OpCode::GetFieldCompact, dest, obj, field_idx as u8)
            }
            Instruction::SetField(obj, field_idx, val) if field_idx <= u16::from(u8::MAX) => {
                Self::encode_abc(OpCode::SetFieldCompact, obj, val, field_idx as u8)
            }
            Instruction::GetField(dest, obj, field_idx) => {
                let w0 = u32::from(obj) | (u32::from(field_idx) << 16);
                let ex = Self::push_extra1(extra_args, w0)?;
                Self::encode_abx(OpCode::GetField, dest, ex)
            }
            Instruction::SetField(obj, field_idx, val) => {
                let w0 = u32::from(val) | (u32::from(field_idx) << 16);
                let ex = Self::push_extra1(extra_args, w0)?;
                Self::encode_abx(OpCode::SetField, obj, ex)
            }
            Instruction::TypeIs(dest, src, type_idx) => {
                let w0 = u32::from(src) | (u32::from(type_idx) << 16);
                let ex = Self::push_extra1(extra_args, w0)?;
                Self::encode_abx(OpCode::TypeIs, dest, ex)
            }
            Instruction::TryCast(dest, src, type_idx) => {
                let w0 = u32::from(src) | (u32::from(type_idx) << 16);
                let ex = Self::push_extra1(extra_args, w0)?;
                Self::encode_abx(OpCode::TryCast, dest, ex)
            }

            // AX format: 2 consecutive words in extra_args
            Instruction::NewStruct(dest, name, field_names, fields, cnt) => {
                let w0 = u32::from(name) | (u32::from(field_names) << 16);
                let w1 = u32::from(fields) | (u32::from(cnt) << 8);
                let ex = Self::push_extra2(extra_args, w0, w1)?;
                Self::encode_abx(OpCode::NewStruct, dest, ex)
            }
            Instruction::NewEnumUnit(dest, enum_name, variant) => {
                let w0 = u32::from(enum_name) | (u32::from(variant) << 16);
                let ex = Self::push_extra2(extra_args, w0, 0)?;
                Self::encode_abx(OpCode::NewEnumUnit, dest, ex)
            }
            Instruction::NewEnumVariant(dest, enum_name, variant, values, cnt) => {
                let w0 = u32::from(enum_name) | (u32::from(variant) << 16);
                let w1 = u32::from(values) | (u32::from(cnt) << 8);
                let ex = Self::push_extra2(extra_args, w0, w1)?;
                Self::encode_abx(OpCode::NewEnumVariant, dest, ex)
            }
            Instruction::IsEnumVariant(dest, val, enum_name, variant) => {
                let w0 = u32::from(enum_name) | (u32::from(variant) << 16);
                let w1 = u32::from(val);
                let ex = Self::push_extra2(extra_args, w0, w1)?;
                Self::encode_abx(OpCode::IsEnumVariant, dest, ex)
            }
            Instruction::CallMethod(obj, method, args, cnt, dest) => {
                let w0 = u32::from(method) | (u32::from(args) << 16) | (u32::from(cnt) << 24);
                let w1 = u32::from(dest);
                let ex = Self::push_extra2(extra_args, w0, w1)?;
                Self::encode_abx(OpCode::CallMethod, obj, ex)
            }
            Instruction::Closure(dest, func, upvals, cnt) => {
                let w0 = u32::from(func) | (u32::from(upvals) << 16) | (u32::from(cnt) << 24);
                let ex = Self::push_extra2(extra_args, w0, 0)?;
                Self::encode_abx(OpCode::Closure, dest, ex)
            }
        };
        Some(packed)
    }

    /// Decodes a `PackedInstruction` (paired with `Chunk::extra_args`) back into an `Instruction`.
    pub fn to_instruction(self, extra_args: &[u32]) -> Result<Instruction, DecodeError> {
        let opcode = OpCode::from_u8(self.opcode_u8())?;
        let extra1 = |idx: u16| -> Result<u32, DecodeError> {
            let i = idx as usize;
            extra_args.get(i).copied().ok_or(DecodeError::Truncated {
                expected: i + 1,
                found: extra_args.len(),
            })
        };
        let extra2 = |idx: u16| -> Result<(u32, u32), DecodeError> {
            let i = idx as usize;
            if i + 1 >= extra_args.len() {
                return Err(DecodeError::Truncated {
                    expected: i + 2,
                    found: extra_args.len(),
                });
            }
            Ok((extra_args[i], extra_args[i + 1]))
        };

        let inst = match opcode {
            OpCode::LoadNil => Instruction::LoadNil(self.a()),
            OpCode::LoadBool => {
                let b = match self.b() {
                    0 => false,
                    1 => true,
                    other => return Err(DecodeError::InvalidBoolean(other)),
                };
                Instruction::LoadBool(self.a(), b)
            }
            OpCode::LoadConst => Instruction::LoadConst(self.a(), self.bx()),
            OpCode::LoadGlobal => Instruction::LoadGlobal(self.a(), self.bx()),
            OpCode::StoreGlobal => Instruction::StoreGlobal(self.bx(), self.a()),
            OpCode::Move => Instruction::Move(self.a(), self.b()),
            OpCode::Add => Instruction::Add(self.a(), self.b(), self.c()),
            OpCode::Sub => Instruction::Sub(self.a(), self.b(), self.c()),
            OpCode::Mul => Instruction::Mul(self.a(), self.b(), self.c()),
            OpCode::Div => Instruction::Div(self.a(), self.b(), self.c()),
            OpCode::Mod => Instruction::Mod(self.a(), self.b(), self.c()),
            OpCode::Pow => Instruction::Pow(self.a(), self.b(), self.c()),
            OpCode::FloorDiv => Instruction::FloorDiv(self.a(), self.b(), self.c()),
            OpCode::Neg => Instruction::Neg(self.a(), self.b()),
            OpCode::Eq => Instruction::Eq(self.a(), self.b(), self.c()),
            OpCode::Ne => Instruction::Ne(self.a(), self.b(), self.c()),
            OpCode::Lt => Instruction::Lt(self.a(), self.b(), self.c()),
            OpCode::Le => Instruction::Le(self.a(), self.b(), self.c()),
            OpCode::Gt => Instruction::Gt(self.a(), self.b(), self.c()),
            OpCode::Ge => Instruction::Ge(self.a(), self.b(), self.c()),
            OpCode::AddInt => Instruction::AddInt(self.a(), self.b(), self.c()),
            OpCode::SubInt => Instruction::SubInt(self.a(), self.b(), self.c()),
            OpCode::MulInt => Instruction::MulInt(self.a(), self.b(), self.c()),
            OpCode::DivInt => Instruction::DivInt(self.a(), self.b(), self.c()),
            OpCode::ModInt => Instruction::ModInt(self.a(), self.b(), self.c()),
            OpCode::PowInt => Instruction::PowInt(self.a(), self.b(), self.c()),
            OpCode::FloorDivInt => Instruction::FloorDivInt(self.a(), self.b(), self.c()),
            OpCode::NegInt => Instruction::NegInt(self.a(), self.b()),
            OpCode::EqInt => Instruction::EqInt(self.a(), self.b(), self.c()),
            OpCode::NeInt => Instruction::NeInt(self.a(), self.b(), self.c()),
            OpCode::LtInt => Instruction::LtInt(self.a(), self.b(), self.c()),
            OpCode::LeInt => Instruction::LeInt(self.a(), self.b(), self.c()),
            OpCode::GtInt => Instruction::GtInt(self.a(), self.b(), self.c()),
            OpCode::GeInt => Instruction::GeInt(self.a(), self.b(), self.c()),
            OpCode::AddFloat => Instruction::AddFloat(self.a(), self.b(), self.c()),
            OpCode::SubFloat => Instruction::SubFloat(self.a(), self.b(), self.c()),
            OpCode::MulFloat => Instruction::MulFloat(self.a(), self.b(), self.c()),
            OpCode::DivFloat => Instruction::DivFloat(self.a(), self.b(), self.c()),
            OpCode::ModFloat => Instruction::ModFloat(self.a(), self.b(), self.c()),
            OpCode::PowFloat => Instruction::PowFloat(self.a(), self.b(), self.c()),
            OpCode::FloorDivFloat => Instruction::FloorDivFloat(self.a(), self.b(), self.c()),
            OpCode::NegFloat => Instruction::NegFloat(self.a(), self.b()),
            OpCode::EqFloat => Instruction::EqFloat(self.a(), self.b(), self.c()),
            OpCode::NeFloat => Instruction::NeFloat(self.a(), self.b(), self.c()),
            OpCode::LtFloat => Instruction::LtFloat(self.a(), self.b(), self.c()),
            OpCode::LeFloat => Instruction::LeFloat(self.a(), self.b(), self.c()),
            OpCode::GtFloat => Instruction::GtFloat(self.a(), self.b(), self.c()),
            OpCode::GeFloat => Instruction::GeFloat(self.a(), self.b(), self.c()),
            OpCode::And => Instruction::And(self.a(), self.b(), self.c()),
            OpCode::Or => Instruction::Or(self.a(), self.b(), self.c()),
            OpCode::Not => Instruction::Not(self.a(), self.b()),
            OpCode::BitAnd => Instruction::BitAnd(self.a(), self.b(), self.c()),
            OpCode::BitOr => Instruction::BitOr(self.a(), self.b(), self.c()),
            OpCode::BitXor => Instruction::BitXor(self.a(), self.b(), self.c()),
            OpCode::Shl => Instruction::Shl(self.a(), self.b(), self.c()),
            OpCode::Shr => Instruction::Shr(self.a(), self.b(), self.c()),
            OpCode::BitNot => Instruction::BitNot(self.a(), self.b()),
            OpCode::Jump => Instruction::Jump(self.sbx()),
            OpCode::JumpIf => Instruction::JumpIf(self.a(), self.sbx()),
            OpCode::JumpIfNot => Instruction::JumpIfNot(self.a(), self.sbx()),
            OpCode::Call => {
                let w0 = extra1(self.bx())?;
                Instruction::Call(
                    self.a(),
                    (w0 & 0xFF) as u8,
                    ((w0 >> 8) & 0xFF) as u8,
                    ((w0 >> 16) & 0xFF) as u8,
                )
            }
            OpCode::Return => Instruction::Return(self.a()),
            OpCode::NewArray => Instruction::NewArray(self.a(), self.b(), self.c()),
            OpCode::NewMap => Instruction::NewMap(self.a()),
            OpCode::NewStruct => {
                let (w0, w1) = extra2(self.bx())?;
                Instruction::NewStruct(
                    self.a(),
                    (w0 & 0xFFFF) as u16,
                    (w0 >> 16) as u16,
                    (w1 & 0xFF) as u8,
                    ((w1 >> 8) & 0xFF) as u8,
                )
            }
            OpCode::NewEnumUnit => {
                let (w0, _) = extra2(self.bx())?;
                Instruction::NewEnumUnit(self.a(), (w0 & 0xFFFF) as u16, (w0 >> 16) as u16)
            }
            OpCode::NewEnumVariant => {
                let (w0, w1) = extra2(self.bx())?;
                Instruction::NewEnumVariant(
                    self.a(),
                    (w0 & 0xFFFF) as u16,
                    (w0 >> 16) as u16,
                    (w1 & 0xFF) as u8,
                    ((w1 >> 8) & 0xFF) as u8,
                )
            }
            OpCode::TupleNew => Instruction::TupleNew(self.a(), self.b(), self.c()),
            OpCode::TupleGet => Instruction::TupleGet(self.a(), self.b(), self.c()),
            OpCode::IsEnumVariant => {
                let (w0, w1) = extra2(self.bx())?;
                Instruction::IsEnumVariant(
                    self.a(),
                    (w1 & 0xFF) as u8,
                    (w0 & 0xFFFF) as u16,
                    (w0 >> 16) as u16,
                )
            }
            OpCode::GetEnumValue => Instruction::GetEnumValue(self.a(), self.b(), self.c()),
            OpCode::GetField => {
                let w0 = extra1(self.bx())?;
                Instruction::GetField(self.a(), (w0 & 0xFF) as u8, (w0 >> 16) as u16)
            }
            OpCode::SetField => {
                let w0 = extra1(self.bx())?;
                Instruction::SetField(self.a(), (w0 >> 16) as u16, (w0 & 0xFF) as u8)
            }
            OpCode::GetFieldCompact => {
                Instruction::GetField(self.a(), self.b(), u16::from(self.c()))
            }
            OpCode::SetFieldCompact => {
                Instruction::SetField(self.a(), u16::from(self.c()), self.b())
            }
            OpCode::GetFieldSlot => Instruction::GetFieldSlot(self.a(), self.b(), self.c()),
            OpCode::SetFieldSlot => Instruction::SetFieldSlot(self.a(), self.b(), self.c()),
            OpCode::GetIndex => Instruction::GetIndex(self.a(), self.b(), self.c()),
            OpCode::TryGetIndex => Instruction::TryGetIndex(self.a(), self.b(), self.c()),
            OpCode::ArrayIndexOk => {
                let w0 = extra1(self.bx())?;
                Instruction::ArrayIndexOk(
                    self.a(),
                    (w0 & 0xFF) as u8,
                    ((w0 >> 8) & 0xFF) as u8,
                    ((w0 >> 16) & 0xFF) as u8,
                )
            }
            OpCode::ArrayLen => Instruction::ArrayLen(self.a(), self.b()),
            OpCode::ArrayPush => Instruction::ArrayPush(self.a(), self.b()),
            OpCode::ArrayGetUnwrap => Instruction::ArrayGetUnwrap(self.a(), self.b(), self.c()),
            OpCode::SetIndex => Instruction::SetIndex(self.a(), self.b(), self.c()),
            OpCode::Concat => Instruction::Concat(self.a(), self.b(), self.c()),
            OpCode::CallMethod => {
                let (w0, w1) = extra2(self.bx())?;
                Instruction::CallMethod(
                    self.a(),
                    (w0 & 0xFFFF) as u16,
                    ((w0 >> 16) & 0xFF) as u8,
                    (w0 >> 24) as u8,
                    (w1 & 0xFF) as u8,
                )
            }
            OpCode::TypeIs => {
                let w0 = extra1(self.bx())?;
                Instruction::TypeIs(self.a(), (w0 & 0xFF) as u8, (w0 >> 16) as u16)
            }
            OpCode::TryCast => {
                let w0 = extra1(self.bx())?;
                Instruction::TryCast(self.a(), (w0 & 0xFF) as u8, (w0 >> 16) as u16)
            }
            OpCode::LoadUpvalue => Instruction::LoadUpvalue(self.a(), self.b()),
            OpCode::StoreUpvalue => Instruction::StoreUpvalue(self.a(), self.b()),
            OpCode::Closure => {
                let (w0, _) = extra2(self.bx())?;
                Instruction::Closure(
                    self.a(),
                    (w0 & 0xFFFF) as u16,
                    ((w0 >> 16) & 0xFF) as u8,
                    (w0 >> 24) as u8,
                )
            }
            OpCode::AddIntImm => Instruction::AddIntImm(self.a(), self.b(), self.c_i8()),
            OpCode::BranchLtInt => {
                let w0 = extra1(self.bx())?;
                Instruction::BranchLtInt(self.a(), (w0 & 0xFF) as u8, (w0 >> 16) as u16 as i16)
            }
            OpCode::BranchLeInt => {
                let w0 = extra1(self.bx())?;
                Instruction::BranchLeInt(self.a(), (w0 & 0xFF) as u8, (w0 >> 16) as u16 as i16)
            }
            OpCode::GetFieldAddSetInt => {
                Instruction::GetFieldAddSetInt(self.a(), self.b(), self.c())
            }
        };
        Ok(inst)
    }
}
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
    Pow(Register, Register, Register),
    FloorDiv(Register, Register, Register),
    Neg(Register, Register),
    Eq(Register, Register, Register),
    Ne(Register, Register, Register),
    Lt(Register, Register, Register),
    Le(Register, Register, Register),
    Gt(Register, Register, Register),
    Ge(Register, Register, Register),
    AddInt(Register, Register, Register),
    SubInt(Register, Register, Register),
    MulInt(Register, Register, Register),
    DivInt(Register, Register, Register),
    ModInt(Register, Register, Register),
    PowInt(Register, Register, Register),
    FloorDivInt(Register, Register, Register),
    NegInt(Register, Register),
    EqInt(Register, Register, Register),
    NeInt(Register, Register, Register),
    LtInt(Register, Register, Register),
    LeInt(Register, Register, Register),
    GtInt(Register, Register, Register),
    GeInt(Register, Register, Register),
    AddFloat(Register, Register, Register),
    SubFloat(Register, Register, Register),
    MulFloat(Register, Register, Register),
    DivFloat(Register, Register, Register),
    ModFloat(Register, Register, Register),
    PowFloat(Register, Register, Register),
    FloorDivFloat(Register, Register, Register),
    NegFloat(Register, Register),
    EqFloat(Register, Register, Register),
    NeFloat(Register, Register, Register),
    LtFloat(Register, Register, Register),
    LeFloat(Register, Register, Register),
    GtFloat(Register, Register, Register),
    GeFloat(Register, Register, Register),
    And(Register, Register, Register),
    Or(Register, Register, Register),
    Not(Register, Register),
    BitAnd(Register, Register, Register),
    BitOr(Register, Register, Register),
    BitXor(Register, Register, Register),
    Shl(Register, Register, Register),
    Shr(Register, Register, Register),
    BitNot(Register, Register),
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
    GetFieldSlot(Register, Register, u8),
    SetFieldSlot(Register, u8, Register),
    GetIndex(Register, Register, Register),
    TryGetIndex(Register, Register, Register),
    ArrayIndexOk(Register, Register, Register, Register),
    ArrayLen(Register, Register),
    ArrayPush(Register, Register),
    ArrayGetUnwrap(Register, Register, Register),
    SetIndex(Register, Register, Register),
    Concat(Register, Register, Register),
    CallMethod(Register, ConstIndex, Register, u8, Register),
    TypeIs(Register, Register, ConstIndex),
    TryCast(Register, Register, ConstIndex),
    LoadUpvalue(Register, u8),
    StoreUpvalue(u8, Register),
    Closure(Register, ConstIndex, Register, u8),
    AddIntImm(Register, Register, i8),
    BranchLtInt(Register, Register, JumpOffset),
    BranchLeInt(Register, Register, JumpOffset),
    GetFieldAddSetInt(Register, u8, Register),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
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
    Pow,
    FloorDiv,
    Neg,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    AddInt,
    SubInt,
    MulInt,
    DivInt,
    ModInt,
    PowInt,
    FloorDivInt,
    NegInt,
    EqInt,
    NeInt,
    LtInt,
    LeInt,
    GtInt,
    GeInt,
    AddFloat,
    SubFloat,
    MulFloat,
    DivFloat,
    ModFloat,
    PowFloat,
    FloorDivFloat,
    NegFloat,
    EqFloat,
    NeFloat,
    LtFloat,
    LeFloat,
    GtFloat,
    GeFloat,
    And,
    Or,
    Not,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    BitNot,
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
    ArrayIndexOk,
    ArrayLen,
    SetIndex,
    Concat,
    CallMethod,
    TypeIs,
    TryCast,
    LoadUpvalue,
    StoreUpvalue,
    Closure,
    ArrayPush,
    ArrayGetUnwrap,
    GetFieldSlot,
    SetFieldSlot,
    AddIntImm,
    BranchLtInt,
    BranchLeInt,
    GetFieldAddSetInt,
    // Packed-only forms. The legacy encoding keeps the original field opcodes.
    GetFieldCompact,
    SetFieldCompact,
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
            Instruction::Pow(_, _, _) => OpCode::Pow,
            Instruction::FloorDiv(_, _, _) => OpCode::FloorDiv,
            Instruction::Neg(_, _) => OpCode::Neg,
            Instruction::Eq(_, _, _) => OpCode::Eq,
            Instruction::Ne(_, _, _) => OpCode::Ne,
            Instruction::Lt(_, _, _) => OpCode::Lt,
            Instruction::Le(_, _, _) => OpCode::Le,
            Instruction::Gt(_, _, _) => OpCode::Gt,
            Instruction::Ge(_, _, _) => OpCode::Ge,
            Instruction::AddInt(_, _, _) => OpCode::AddInt,
            Instruction::SubInt(_, _, _) => OpCode::SubInt,
            Instruction::MulInt(_, _, _) => OpCode::MulInt,
            Instruction::DivInt(_, _, _) => OpCode::DivInt,
            Instruction::ModInt(_, _, _) => OpCode::ModInt,
            Instruction::PowInt(_, _, _) => OpCode::PowInt,
            Instruction::FloorDivInt(_, _, _) => OpCode::FloorDivInt,
            Instruction::NegInt(_, _) => OpCode::NegInt,
            Instruction::EqInt(_, _, _) => OpCode::EqInt,
            Instruction::NeInt(_, _, _) => OpCode::NeInt,
            Instruction::LtInt(_, _, _) => OpCode::LtInt,
            Instruction::LeInt(_, _, _) => OpCode::LeInt,
            Instruction::GtInt(_, _, _) => OpCode::GtInt,
            Instruction::GeInt(_, _, _) => OpCode::GeInt,
            Instruction::AddFloat(_, _, _) => OpCode::AddFloat,
            Instruction::SubFloat(_, _, _) => OpCode::SubFloat,
            Instruction::MulFloat(_, _, _) => OpCode::MulFloat,
            Instruction::DivFloat(_, _, _) => OpCode::DivFloat,
            Instruction::ModFloat(_, _, _) => OpCode::ModFloat,
            Instruction::PowFloat(_, _, _) => OpCode::PowFloat,
            Instruction::FloorDivFloat(_, _, _) => OpCode::FloorDivFloat,
            Instruction::NegFloat(_, _) => OpCode::NegFloat,
            Instruction::EqFloat(_, _, _) => OpCode::EqFloat,
            Instruction::NeFloat(_, _, _) => OpCode::NeFloat,
            Instruction::LtFloat(_, _, _) => OpCode::LtFloat,
            Instruction::LeFloat(_, _, _) => OpCode::LeFloat,
            Instruction::GtFloat(_, _, _) => OpCode::GtFloat,
            Instruction::GeFloat(_, _, _) => OpCode::GeFloat,
            Instruction::And(_, _, _) => OpCode::And,
            Instruction::Or(_, _, _) => OpCode::Or,
            Instruction::Not(_, _) => OpCode::Not,
            Instruction::BitAnd(_, _, _) => OpCode::BitAnd,
            Instruction::BitOr(_, _, _) => OpCode::BitOr,
            Instruction::BitXor(_, _, _) => OpCode::BitXor,
            Instruction::Shl(_, _, _) => OpCode::Shl,
            Instruction::Shr(_, _, _) => OpCode::Shr,
            Instruction::BitNot(_, _) => OpCode::BitNot,
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
            Instruction::GetFieldSlot(_, _, _) => OpCode::GetFieldSlot,
            Instruction::SetFieldSlot(_, _, _) => OpCode::SetFieldSlot,
            Instruction::GetIndex(_, _, _) => OpCode::GetIndex,
            Instruction::TryGetIndex(_, _, _) => OpCode::TryGetIndex,
            Instruction::ArrayIndexOk(_, _, _, _) => OpCode::ArrayIndexOk,
            Instruction::ArrayLen(_, _) => OpCode::ArrayLen,
            Instruction::ArrayPush(_, _) => OpCode::ArrayPush,
            Instruction::ArrayGetUnwrap(_, _, _) => OpCode::ArrayGetUnwrap,
            Instruction::SetIndex(_, _, _) => OpCode::SetIndex,
            Instruction::Concat(_, _, _) => OpCode::Concat,
            Instruction::CallMethod(_, _, _, _, _) => OpCode::CallMethod,
            Instruction::TypeIs(_, _, _) => OpCode::TypeIs,
            Instruction::TryCast(_, _, _) => OpCode::TryCast,
            Instruction::LoadUpvalue(_, _) => OpCode::LoadUpvalue,
            Instruction::StoreUpvalue(_, _) => OpCode::StoreUpvalue,
            Instruction::Closure(_, _, _, _) => OpCode::Closure,
            Instruction::AddIntImm(_, _, _) => OpCode::AddIntImm,
            Instruction::BranchLtInt(_, _, _) => OpCode::BranchLtInt,
            Instruction::BranchLeInt(_, _, _) => OpCode::BranchLeInt,
            Instruction::GetFieldAddSetInt(_, _, _) => OpCode::GetFieldAddSetInt,
        }
    }

    /// Logical gas weight of a single bytecode instruction.
    ///
    /// Superinstructions charge the exact sum of the constituent instructions
    /// they fuse so gas accounting remains invariant under fusion:
    /// - `BranchLtInt` / `BranchLeInt`: 2 (`LtInt`/`LeInt` + `JumpIfNot`)
    /// - `GetFieldAddSetInt`: 3 (`GetFieldSlot` + `Add`/`AddInt` + `SetFieldSlot`)
    /// - all other instructions: 1
    #[inline(always)]
    pub const fn gas_weight(&self) -> u16 {
        match self {
            Instruction::BranchLtInt(..) | Instruction::BranchLeInt(..) => 2,
            Instruction::GetFieldAddSetInt(..) => 3,
            _ => 1,
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
            | Instruction::Pow(dest, _, _)
            | Instruction::FloorDiv(dest, _, _)
            | Instruction::Neg(dest, _)
            | Instruction::Eq(dest, _, _)
            | Instruction::Ne(dest, _, _)
            | Instruction::Lt(dest, _, _)
            | Instruction::Le(dest, _, _)
            | Instruction::Gt(dest, _, _)
            | Instruction::Ge(dest, _, _)
            | Instruction::AddInt(dest, _, _)
            | Instruction::SubInt(dest, _, _)
            | Instruction::MulInt(dest, _, _)
            | Instruction::DivInt(dest, _, _)
            | Instruction::ModInt(dest, _, _)
            | Instruction::PowInt(dest, _, _)
            | Instruction::FloorDivInt(dest, _, _)
            | Instruction::NegInt(dest, _)
            | Instruction::EqInt(dest, _, _)
            | Instruction::NeInt(dest, _, _)
            | Instruction::LtInt(dest, _, _)
            | Instruction::LeInt(dest, _, _)
            | Instruction::GtInt(dest, _, _)
            | Instruction::GeInt(dest, _, _)
            | Instruction::AddFloat(dest, _, _)
            | Instruction::SubFloat(dest, _, _)
            | Instruction::MulFloat(dest, _, _)
            | Instruction::DivFloat(dest, _, _)
            | Instruction::ModFloat(dest, _, _)
            | Instruction::PowFloat(dest, _, _)
            | Instruction::FloorDivFloat(dest, _, _)
            | Instruction::NegFloat(dest, _)
            | Instruction::EqFloat(dest, _, _)
            | Instruction::NeFloat(dest, _, _)
            | Instruction::LtFloat(dest, _, _)
            | Instruction::LeFloat(dest, _, _)
            | Instruction::GtFloat(dest, _, _)
            | Instruction::GeFloat(dest, _, _)
            | Instruction::And(dest, _, _)
            | Instruction::Or(dest, _, _)
            | Instruction::Not(dest, _)
            | Instruction::BitAnd(dest, _, _)
            | Instruction::BitOr(dest, _, _)
            | Instruction::BitXor(dest, _, _)
            | Instruction::Shl(dest, _, _)
            | Instruction::Shr(dest, _, _)
            | Instruction::BitNot(dest, _)
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
            | Instruction::GetFieldSlot(dest, _, _)
            | Instruction::GetIndex(dest, _, _)
            | Instruction::TryGetIndex(dest, _, _)
            | Instruction::ArrayLen(dest, _)
            | Instruction::ArrayGetUnwrap(dest, _, _)
            | Instruction::Concat(dest, _, _)
            | Instruction::CallMethod(_, _, _, _, dest)
            | Instruction::TypeIs(dest, _, _)
            | Instruction::TryCast(dest, _, _)
            | Instruction::LoadUpvalue(dest, _)
            | Instruction::Closure(dest, _, _, _)
            | Instruction::AddIntImm(dest, _, _) => Some(dest),
            Instruction::StoreGlobal(_, _)
            | Instruction::Jump(_)
            | Instruction::JumpIf(_, _)
            | Instruction::JumpIfNot(_, _)
            | Instruction::BranchLtInt(_, _, _)
            | Instruction::BranchLeInt(_, _, _)
            | Instruction::Return(_)
            | Instruction::SetField(_, _, _)
            | Instruction::SetFieldSlot(_, _, _)
            | Instruction::GetFieldAddSetInt(_, _, _)
            | Instruction::SetIndex(_, _, _)
            | Instruction::ArrayIndexOk(_, _, _, _)
            | Instruction::ArrayPush(_, _)
            | Instruction::StoreUpvalue(_, _) => None,
        }
    }

    pub fn reads_register(&self, register: Register) -> bool {
        let in_range = |first: Register, count: u8| {
            register >= first && (register as usize) < first as usize + count as usize
        };
        match *self {
            Instruction::Move(_, src)
            | Instruction::AddIntImm(_, src, _)
            | Instruction::Neg(_, src)
            | Instruction::NegInt(_, src)
            | Instruction::NegFloat(_, src)
            | Instruction::Not(_, src)
            | Instruction::BitNot(_, src)
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
            | Instruction::Pow(_, lhs, rhs)
            | Instruction::FloorDiv(_, lhs, rhs)
            | Instruction::Eq(_, lhs, rhs)
            | Instruction::Ne(_, lhs, rhs)
            | Instruction::Lt(_, lhs, rhs)
            | Instruction::Le(_, lhs, rhs)
            | Instruction::Gt(_, lhs, rhs)
            | Instruction::Ge(_, lhs, rhs)
            | Instruction::AddInt(_, lhs, rhs)
            | Instruction::SubInt(_, lhs, rhs)
            | Instruction::MulInt(_, lhs, rhs)
            | Instruction::DivInt(_, lhs, rhs)
            | Instruction::ModInt(_, lhs, rhs)
            | Instruction::PowInt(_, lhs, rhs)
            | Instruction::FloorDivInt(_, lhs, rhs)
            | Instruction::EqInt(_, lhs, rhs)
            | Instruction::NeInt(_, lhs, rhs)
            | Instruction::LtInt(_, lhs, rhs)
            | Instruction::LeInt(_, lhs, rhs)
            | Instruction::GtInt(_, lhs, rhs)
            | Instruction::GeInt(_, lhs, rhs)
            | Instruction::AddFloat(_, lhs, rhs)
            | Instruction::SubFloat(_, lhs, rhs)
            | Instruction::MulFloat(_, lhs, rhs)
            | Instruction::DivFloat(_, lhs, rhs)
            | Instruction::ModFloat(_, lhs, rhs)
            | Instruction::PowFloat(_, lhs, rhs)
            | Instruction::FloorDivFloat(_, lhs, rhs)
            | Instruction::EqFloat(_, lhs, rhs)
            | Instruction::NeFloat(_, lhs, rhs)
            | Instruction::LtFloat(_, lhs, rhs)
            | Instruction::LeFloat(_, lhs, rhs)
            | Instruction::GtFloat(_, lhs, rhs)
            | Instruction::GeFloat(_, lhs, rhs)
            | Instruction::And(_, lhs, rhs)
            | Instruction::Or(_, lhs, rhs)
            | Instruction::BitAnd(_, lhs, rhs)
            | Instruction::BitOr(_, lhs, rhs)
            | Instruction::BitXor(_, lhs, rhs)
            | Instruction::Shl(_, lhs, rhs)
            | Instruction::Shr(_, lhs, rhs)
            | Instruction::GetIndex(_, lhs, rhs)
            | Instruction::TryGetIndex(_, lhs, rhs)
            | Instruction::ArrayIndexOk(_, _, lhs, rhs)
            | Instruction::ArrayPush(lhs, rhs)
            | Instruction::ArrayGetUnwrap(_, lhs, rhs)
            | Instruction::Concat(_, lhs, rhs)
            | Instruction::BranchLtInt(lhs, rhs, _)
            | Instruction::BranchLeInt(lhs, rhs, _) => register == lhs || register == rhs,
            Instruction::SetIndex(collection, index, value) => {
                register == collection || register == index || register == value
            }
            Instruction::SetField(object, _, value)
            | Instruction::SetFieldSlot(object, _, value)
            | Instruction::GetFieldAddSetInt(object, _, value) => {
                register == object || register == value
            }
            Instruction::IsEnumVariant(_, value, _, _)
            | Instruction::GetField(_, value, _)
            | Instruction::GetFieldSlot(_, value, _)
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

impl Instruction {
    pub fn specialize_numeric(self, ty: NumericType) -> Self {
        use NumericType::{Float, Int};
        match (self, ty) {
            (Self::Add(d, l, r), Int) => Self::AddInt(d, l, r),
            (Self::Sub(d, l, r), Int) => Self::SubInt(d, l, r),
            (Self::Mul(d, l, r), Int) => Self::MulInt(d, l, r),
            (Self::Div(d, l, r), Int) => Self::DivInt(d, l, r),
            (Self::Mod(d, l, r), Int) => Self::ModInt(d, l, r),
            (Self::Pow(d, l, r), Int) => Self::PowInt(d, l, r),
            (Self::FloorDiv(d, l, r), Int) => Self::FloorDivInt(d, l, r),
            (Self::Neg(d, s), Int) => Self::NegInt(d, s),
            (Self::Eq(d, l, r), Int) => Self::EqInt(d, l, r),
            (Self::Ne(d, l, r), Int) => Self::NeInt(d, l, r),
            (Self::Lt(d, l, r), Int) => Self::LtInt(d, l, r),
            (Self::Le(d, l, r), Int) => Self::LeInt(d, l, r),
            (Self::Gt(d, l, r), Int) => Self::GtInt(d, l, r),
            (Self::Ge(d, l, r), Int) => Self::GeInt(d, l, r),
            (Self::Add(d, l, r), Float) => Self::AddFloat(d, l, r),
            (Self::Sub(d, l, r), Float) => Self::SubFloat(d, l, r),
            (Self::Mul(d, l, r), Float) => Self::MulFloat(d, l, r),
            (Self::Div(d, l, r), Float) => Self::DivFloat(d, l, r),
            (Self::Mod(d, l, r), Float) => Self::ModFloat(d, l, r),
            (Self::Pow(d, l, r), Float) => Self::PowFloat(d, l, r),
            (Self::FloorDiv(d, l, r), Float) => Self::FloorDivFloat(d, l, r),
            (Self::Neg(d, s), Float) => Self::NegFloat(d, s),
            (Self::Eq(d, l, r), Float) => Self::EqFloat(d, l, r),
            (Self::Ne(d, l, r), Float) => Self::NeFloat(d, l, r),
            (Self::Lt(d, l, r), Float) => Self::LtFloat(d, l, r),
            (Self::Le(d, l, r), Float) => Self::LeFloat(d, l, r),
            (Self::Gt(d, l, r), Float) => Self::GtFloat(d, l, r),
            (Self::Ge(d, l, r), Float) => Self::GeFloat(d, l, r),
            _ => self,
        }
    }

    /// Generic operation and checked input contract for trace lowering.
    /// Negation repeats its single source in both operand positions.
    pub fn numeric_specialization(self) -> Option<(Self, NumericType, Register, Register)> {
        use NumericType::{Float, Int};
        Some(match self {
            Self::AddInt(d, l, r) => (Self::Add(d, l, r), Int, l, r),
            Self::SubInt(d, l, r) => (Self::Sub(d, l, r), Int, l, r),
            Self::MulInt(d, l, r) => (Self::Mul(d, l, r), Int, l, r),
            Self::DivInt(d, l, r) => (Self::Div(d, l, r), Int, l, r),
            Self::ModInt(d, l, r) => (Self::Mod(d, l, r), Int, l, r),
            Self::NegInt(d, s) => (Self::Neg(d, s), Int, s, s),
            Self::EqInt(d, l, r) => (Self::Eq(d, l, r), Int, l, r),
            Self::NeInt(d, l, r) => (Self::Ne(d, l, r), Int, l, r),
            Self::LtInt(d, l, r) => (Self::Lt(d, l, r), Int, l, r),
            Self::LeInt(d, l, r) => (Self::Le(d, l, r), Int, l, r),
            Self::GtInt(d, l, r) => (Self::Gt(d, l, r), Int, l, r),
            Self::GeInt(d, l, r) => (Self::Ge(d, l, r), Int, l, r),
            Self::AddFloat(d, l, r) => (Self::Add(d, l, r), Float, l, r),
            Self::SubFloat(d, l, r) => (Self::Sub(d, l, r), Float, l, r),
            Self::MulFloat(d, l, r) => (Self::Mul(d, l, r), Float, l, r),
            Self::DivFloat(d, l, r) => (Self::Div(d, l, r), Float, l, r),
            Self::ModFloat(d, l, r) => (Self::Mod(d, l, r), Float, l, r),
            Self::NegFloat(d, s) => (Self::Neg(d, s), Float, s, s),
            Self::EqFloat(d, l, r) => (Self::Eq(d, l, r), Float, l, r),
            Self::NeFloat(d, l, r) => (Self::Ne(d, l, r), Float, l, r),
            Self::LtFloat(d, l, r) => (Self::Lt(d, l, r), Float, l, r),
            Self::LeFloat(d, l, r) => (Self::Le(d, l, r), Float, l, r),
            Self::GtFloat(d, l, r) => (Self::Gt(d, l, r), Float, l, r),
            Self::GeFloat(d, l, r) => (Self::Ge(d, l, r), Float, l, r),
            _ => return None,
        })
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
            Instruction::Pow(d, l, r) => write!(f, "Pow R{}, R{}, R{}", d, l, r),
            Instruction::FloorDiv(d, l, r) => write!(f, "FloorDiv R{}, R{}, R{}", d, l, r),
            Instruction::Neg(d, s) => write!(f, "Neg R{}, R{}", d, s),
            Instruction::Eq(d, l, r) => write!(f, "Eq R{}, R{}, R{}", d, l, r),
            Instruction::Ne(d, l, r) => write!(f, "Ne R{}, R{}, R{}", d, l, r),
            Instruction::Lt(d, l, r) => write!(f, "Lt R{}, R{}, R{}", d, l, r),
            Instruction::Le(d, l, r) => write!(f, "Le R{}, R{}, R{}", d, l, r),
            Instruction::Gt(d, l, r) => write!(f, "Gt R{}, R{}, R{}", d, l, r),
            Instruction::Ge(d, l, r) => write!(f, "Ge R{}, R{}, R{}", d, l, r),
            Instruction::AddInt(d, l, r) => write!(f, "AddInt R{}, R{}, R{}", d, l, r),
            Instruction::SubInt(d, l, r) => write!(f, "SubInt R{}, R{}, R{}", d, l, r),
            Instruction::MulInt(d, l, r) => write!(f, "MulInt R{}, R{}, R{}", d, l, r),
            Instruction::DivInt(d, l, r) => write!(f, "DivInt R{}, R{}, R{}", d, l, r),
            Instruction::ModInt(d, l, r) => write!(f, "ModInt R{}, R{}, R{}", d, l, r),
            Instruction::PowInt(d, l, r) => write!(f, "PowInt R{}, R{}, R{}", d, l, r),
            Instruction::FloorDivInt(d, l, r) => write!(f, "FloorDivInt R{}, R{}, R{}", d, l, r),
            Instruction::NegInt(d, s) => write!(f, "NegInt R{}, R{}", d, s),
            Instruction::EqInt(d, l, r) => write!(f, "EqInt R{}, R{}, R{}", d, l, r),
            Instruction::NeInt(d, l, r) => write!(f, "NeInt R{}, R{}, R{}", d, l, r),
            Instruction::LtInt(d, l, r) => write!(f, "LtInt R{}, R{}, R{}", d, l, r),
            Instruction::LeInt(d, l, r) => write!(f, "LeInt R{}, R{}, R{}", d, l, r),
            Instruction::GtInt(d, l, r) => write!(f, "GtInt R{}, R{}, R{}", d, l, r),
            Instruction::GeInt(d, l, r) => write!(f, "GeInt R{}, R{}, R{}", d, l, r),
            Instruction::AddFloat(d, l, r) => write!(f, "AddFloat R{}, R{}, R{}", d, l, r),
            Instruction::SubFloat(d, l, r) => write!(f, "SubFloat R{}, R{}, R{}", d, l, r),
            Instruction::MulFloat(d, l, r) => write!(f, "MulFloat R{}, R{}, R{}", d, l, r),
            Instruction::DivFloat(d, l, r) => write!(f, "DivFloat R{}, R{}, R{}", d, l, r),
            Instruction::ModFloat(d, l, r) => write!(f, "ModFloat R{}, R{}, R{}", d, l, r),
            Instruction::PowFloat(d, l, r) => write!(f, "PowFloat R{}, R{}, R{}", d, l, r),
            Instruction::FloorDivFloat(d, l, r) => write!(f, "FloorDivFloat R{}, R{}, R{}", d, l, r),
            Instruction::NegFloat(d, s) => write!(f, "NegFloat R{}, R{}", d, s),
            Instruction::EqFloat(d, l, r) => write!(f, "EqFloat R{}, R{}, R{}", d, l, r),
            Instruction::NeFloat(d, l, r) => write!(f, "NeFloat R{}, R{}, R{}", d, l, r),
            Instruction::LtFloat(d, l, r) => write!(f, "LtFloat R{}, R{}, R{}", d, l, r),
            Instruction::LeFloat(d, l, r) => write!(f, "LeFloat R{}, R{}, R{}", d, l, r),
            Instruction::GtFloat(d, l, r) => write!(f, "GtFloat R{}, R{}, R{}", d, l, r),
            Instruction::GeFloat(d, l, r) => write!(f, "GeFloat R{}, R{}, R{}", d, l, r),
            Instruction::And(d, l, r) => write!(f, "And R{}, R{}, R{}", d, l, r),
            Instruction::Or(d, l, r) => write!(f, "Or R{}, R{}, R{}", d, l, r),
            Instruction::Not(d, s) => write!(f, "Not R{}, R{}", d, s),
            Instruction::BitAnd(d, l, r) => write!(f, "BitAnd R{}, R{}, R{}", d, l, r),
            Instruction::BitOr(d, l, r) => write!(f, "BitOr R{}, R{}, R{}", d, l, r),
            Instruction::BitXor(d, l, r) => write!(f, "BitXor R{}, R{}, R{}", d, l, r),
            Instruction::Shl(d, l, r) => write!(f, "Shl R{}, R{}, R{}", d, l, r),
            Instruction::Shr(d, l, r) => write!(f, "Shr R{}, R{}, R{}", d, l, r),
            Instruction::BitNot(d, s) => write!(f, "BitNot R{}, R{}", d, s),
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

            Instruction::GetFieldSlot(d, obj, slot) => {
                write!(f, "GetFieldSlot R{}, R{}, #{}", d, obj, slot)
            }

            Instruction::SetFieldSlot(obj, slot, val) => {
                write!(f, "SetFieldSlot R{}, #{}, R{}", obj, slot, val)
            }

            Instruction::GetIndex(d, arr, idx) => write!(f, "GetIndex R{}, R{}, R{}", d, arr, idx),
            Instruction::TryGetIndex(d, arr, idx) => {
                write!(f, "TryGetIndex R{}, R{}, R{}", d, arr, idx)
            }
            Instruction::ArrayIndexOk(v, c, arr, idx) => {
                write!(f, "ArrayIndexOk R{}, R{}, R{}, R{}", v, c, arr, idx)
            }
            Instruction::ArrayLen(d, arr) => write!(f, "ArrayLen R{}, R{}", d, arr),
            Instruction::ArrayPush(arr, val) => write!(f, "ArrayPush R{}, R{}", arr, val),
            Instruction::ArrayGetUnwrap(d, arr, idx) => {
                write!(f, "ArrayGetUnwrap R{}, R{}, R{}", d, arr, idx)
            }
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
            Instruction::AddIntImm(d, s, imm) => write!(f, "AddIntImm R{}, R{}, {}", d, s, imm),
            Instruction::BranchLtInt(lhs, rhs, offset) => {
                write!(f, "BranchLtInt R{}, R{}, {}", lhs, rhs, offset)
            }
            Instruction::BranchLeInt(lhs, rhs, offset) => {
                write!(f, "BranchLeInt R{}, R{}, {}", lhs, rhs, offset)
            }
            Instruction::GetFieldAddSetInt(obj, slot, addend) => {
                write!(f, "GetFieldAddSetInt R{}, #{}, R{}", obj, slot, addend)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    Truncated { expected: usize, found: usize },
    UnknownOpCode(u8),
    InvalidBoolean(u8),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { expected, found } => {
                write!(
                    f,
                    "truncated instruction stream: expected at least {} bytes, found {}",
                    expected, found
                )
            }
            Self::UnknownOpCode(op) => write!(f, "unknown instruction opcode: 0x{:02x} ({})", op, op),
            Self::InvalidBoolean(b) => {
                write!(f, "invalid boolean encoding in LoadBool: expected 0 or 1, found {}", b)
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DecodeError {}

impl OpCode {
    #[inline]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    pub fn from_u8(val: u8) -> Result<Self, DecodeError> {
        match val {
            0 => Ok(OpCode::LoadNil),
            1 => Ok(OpCode::LoadBool),
            2 => Ok(OpCode::LoadConst),
            3 => Ok(OpCode::LoadGlobal),
            4 => Ok(OpCode::StoreGlobal),
            5 => Ok(OpCode::Move),
            6 => Ok(OpCode::Add),
            7 => Ok(OpCode::Sub),
            8 => Ok(OpCode::Mul),
            9 => Ok(OpCode::Div),
            10 => Ok(OpCode::Mod),
            11 => Ok(OpCode::Pow),
            12 => Ok(OpCode::FloorDiv),
            13 => Ok(OpCode::Neg),
            14 => Ok(OpCode::Eq),
            15 => Ok(OpCode::Ne),
            16 => Ok(OpCode::Lt),
            17 => Ok(OpCode::Le),
            18 => Ok(OpCode::Gt),
            19 => Ok(OpCode::Ge),
            20 => Ok(OpCode::AddInt),
            21 => Ok(OpCode::SubInt),
            22 => Ok(OpCode::MulInt),
            23 => Ok(OpCode::DivInt),
            24 => Ok(OpCode::ModInt),
            25 => Ok(OpCode::PowInt),
            26 => Ok(OpCode::FloorDivInt),
            27 => Ok(OpCode::NegInt),
            28 => Ok(OpCode::EqInt),
            29 => Ok(OpCode::NeInt),
            30 => Ok(OpCode::LtInt),
            31 => Ok(OpCode::LeInt),
            32 => Ok(OpCode::GtInt),
            33 => Ok(OpCode::GeInt),
            34 => Ok(OpCode::AddFloat),
            35 => Ok(OpCode::SubFloat),
            36 => Ok(OpCode::MulFloat),
            37 => Ok(OpCode::DivFloat),
            38 => Ok(OpCode::ModFloat),
            39 => Ok(OpCode::PowFloat),
            40 => Ok(OpCode::FloorDivFloat),
            41 => Ok(OpCode::NegFloat),
            42 => Ok(OpCode::EqFloat),
            43 => Ok(OpCode::NeFloat),
            44 => Ok(OpCode::LtFloat),
            45 => Ok(OpCode::LeFloat),
            46 => Ok(OpCode::GtFloat),
            47 => Ok(OpCode::GeFloat),
            48 => Ok(OpCode::And),
            49 => Ok(OpCode::Or),
            50 => Ok(OpCode::Not),
            51 => Ok(OpCode::BitAnd),
            52 => Ok(OpCode::BitOr),
            53 => Ok(OpCode::BitXor),
            54 => Ok(OpCode::Shl),
            55 => Ok(OpCode::Shr),
            56 => Ok(OpCode::BitNot),
            57 => Ok(OpCode::Jump),
            58 => Ok(OpCode::JumpIf),
            59 => Ok(OpCode::JumpIfNot),
            60 => Ok(OpCode::Call),
            61 => Ok(OpCode::Return),
            62 => Ok(OpCode::NewArray),
            63 => Ok(OpCode::NewMap),
            64 => Ok(OpCode::NewStruct),
            65 => Ok(OpCode::NewEnumUnit),
            66 => Ok(OpCode::NewEnumVariant),
            67 => Ok(OpCode::TupleNew),
            68 => Ok(OpCode::TupleGet),
            69 => Ok(OpCode::IsEnumVariant),
            70 => Ok(OpCode::GetEnumValue),
            71 => Ok(OpCode::GetField),
            72 => Ok(OpCode::SetField),
            73 => Ok(OpCode::GetIndex),
            74 => Ok(OpCode::TryGetIndex),
            75 => Ok(OpCode::ArrayIndexOk),
            76 => Ok(OpCode::ArrayLen),
            77 => Ok(OpCode::SetIndex),
            78 => Ok(OpCode::Concat),
            79 => Ok(OpCode::CallMethod),
            80 => Ok(OpCode::TypeIs),
            81 => Ok(OpCode::TryCast),
            82 => Ok(OpCode::LoadUpvalue),
            83 => Ok(OpCode::StoreUpvalue),
            84 => Ok(OpCode::Closure),
            85 => Ok(OpCode::ArrayPush),
            86 => Ok(OpCode::ArrayGetUnwrap),
            87 => Ok(OpCode::GetFieldSlot),
            88 => Ok(OpCode::SetFieldSlot),
            89 => Ok(OpCode::AddIntImm),
            90 => Ok(OpCode::BranchLtInt),
            91 => Ok(OpCode::BranchLeInt),
            92 => Ok(OpCode::GetFieldAddSetInt),
            93 => Ok(OpCode::GetFieldCompact),
            94 => Ok(OpCode::SetFieldCompact),
            other => Err(DecodeError::UnknownOpCode(other)),
        }
    }
}

impl Instruction {
    pub const ENCODED_SIZE: usize = 8;

    pub fn encode(&self) -> [u8; 8] {
        let mut buf = [0u8; 8];
        buf[0] = self.opcode().as_u8();
        match *self {
            Instruction::LoadNil(dest) => {
                buf[1] = dest;
            }
            Instruction::LoadBool(dest, val) => {
                buf[1] = dest;
                buf[2] = if val { 1 } else { 0 };
            }
            Instruction::LoadConst(dest, idx) => {
                buf[1] = dest;
                let b = idx.to_le_bytes();
                buf[2] = b[0];
                buf[3] = b[1];
            }
            Instruction::LoadGlobal(dest, idx) => {
                buf[1] = dest;
                let b = idx.to_le_bytes();
                buf[2] = b[0];
                buf[3] = b[1];
            }
            Instruction::StoreGlobal(idx, src) => {
                buf[1] = src;
                let b = idx.to_le_bytes();
                buf[2] = b[0];
                buf[3] = b[1];
            }
            Instruction::Move(dest, src) => {
                buf[1] = dest;
                buf[2] = src;
            }
            Instruction::Add(d, l, r)
            | Instruction::Sub(d, l, r)
            | Instruction::Mul(d, l, r)
            | Instruction::Div(d, l, r)
            | Instruction::Mod(d, l, r)
            | Instruction::Pow(d, l, r)
            | Instruction::FloorDiv(d, l, r)
            | Instruction::Eq(d, l, r)
            | Instruction::Ne(d, l, r)
            | Instruction::Lt(d, l, r)
            | Instruction::Le(d, l, r)
            | Instruction::Gt(d, l, r)
            | Instruction::Ge(d, l, r)
            | Instruction::AddInt(d, l, r)
            | Instruction::SubInt(d, l, r)
            | Instruction::MulInt(d, l, r)
            | Instruction::DivInt(d, l, r)
            | Instruction::ModInt(d, l, r)
            | Instruction::PowInt(d, l, r)
            | Instruction::FloorDivInt(d, l, r)
            | Instruction::EqInt(d, l, r)
            | Instruction::NeInt(d, l, r)
            | Instruction::LtInt(d, l, r)
            | Instruction::LeInt(d, l, r)
            | Instruction::GtInt(d, l, r)
            | Instruction::GeInt(d, l, r)
            | Instruction::AddFloat(d, l, r)
            | Instruction::SubFloat(d, l, r)
            | Instruction::MulFloat(d, l, r)
            | Instruction::DivFloat(d, l, r)
            | Instruction::ModFloat(d, l, r)
            | Instruction::PowFloat(d, l, r)
            | Instruction::FloorDivFloat(d, l, r)
            | Instruction::EqFloat(d, l, r)
            | Instruction::NeFloat(d, l, r)
            | Instruction::LtFloat(d, l, r)
            | Instruction::LeFloat(d, l, r)
            | Instruction::GtFloat(d, l, r)
            | Instruction::GeFloat(d, l, r)
            | Instruction::And(d, l, r)
            | Instruction::Or(d, l, r)
            | Instruction::BitAnd(d, l, r)
            | Instruction::BitOr(d, l, r)
            | Instruction::BitXor(d, l, r)
            | Instruction::Shl(d, l, r)
            | Instruction::Shr(d, l, r)
            | Instruction::GetIndex(d, l, r)
            | Instruction::TryGetIndex(d, l, r)
            | Instruction::ArrayGetUnwrap(d, l, r)
            | Instruction::Concat(d, l, r) => {
                buf[1] = d;
                buf[2] = l;
                buf[3] = r;
            }
            Instruction::Neg(d, s)
            | Instruction::NegInt(d, s)
            | Instruction::NegFloat(d, s)
            | Instruction::Not(d, s)
            | Instruction::BitNot(d, s)
            | Instruction::ArrayLen(d, s)
            | Instruction::ArrayPush(d, s) => {
                buf[1] = d;
                buf[2] = s;
            }
            Instruction::Jump(offset) => {
                let b = offset.to_le_bytes();
                buf[1] = b[0];
                buf[2] = b[1];
            }
            Instruction::JumpIf(reg, offset) => {
                buf[1] = reg;
                let b = offset.to_le_bytes();
                buf[2] = b[0];
                buf[3] = b[1];
            }
            Instruction::JumpIfNot(reg, offset) => {
                buf[1] = reg;
                let b = offset.to_le_bytes();
                buf[2] = b[0];
                buf[3] = b[1];
            }
            Instruction::Call(func, args, cnt, dest) => {
                buf[1] = func;
                buf[2] = args;
                buf[3] = cnt;
                buf[4] = dest;
            }
            Instruction::Return(reg) => {
                buf[1] = reg;
            }
            Instruction::NewArray(dest, elems, cnt) => {
                buf[1] = dest;
                buf[2] = elems;
                buf[3] = cnt;
            }
            Instruction::NewMap(dest) => {
                buf[1] = dest;
            }
            Instruction::NewStruct(dest, name, field_names, fields, cnt) => {
                buf[1] = dest;
                let b1 = name.to_le_bytes();
                buf[2] = b1[0];
                buf[3] = b1[1];
                let b2 = field_names.to_le_bytes();
                buf[4] = b2[0];
                buf[5] = b2[1];
                buf[6] = fields;
                buf[7] = cnt;
            }
            Instruction::NewEnumUnit(dest, enum_name, variant) => {
                buf[1] = dest;
                let b1 = enum_name.to_le_bytes();
                buf[2] = b1[0];
                buf[3] = b1[1];
                let b2 = variant.to_le_bytes();
                buf[4] = b2[0];
                buf[5] = b2[1];
            }
            Instruction::NewEnumVariant(dest, enum_name, variant, values, cnt) => {
                buf[1] = dest;
                let b1 = enum_name.to_le_bytes();
                buf[2] = b1[0];
                buf[3] = b1[1];
                let b2 = variant.to_le_bytes();
                buf[4] = b2[0];
                buf[5] = b2[1];
                buf[6] = values;
                buf[7] = cnt;
            }
            Instruction::TupleNew(dest, first, cnt) => {
                buf[1] = dest;
                buf[2] = first;
                buf[3] = cnt;
            }
            Instruction::TupleGet(dest, tuple, idx) => {
                buf[1] = dest;
                buf[2] = tuple;
                buf[3] = idx;
            }
            Instruction::IsEnumVariant(dest, val, enum_name, variant) => {
                buf[1] = dest;
                buf[2] = val;
                let b1 = enum_name.to_le_bytes();
                buf[3] = b1[0];
                buf[4] = b1[1];
                let b2 = variant.to_le_bytes();
                buf[5] = b2[0];
                buf[6] = b2[1];
            }
            Instruction::GetEnumValue(dest, enum_reg, idx) => {
                buf[1] = dest;
                buf[2] = enum_reg;
                buf[3] = idx;
            }
            Instruction::GetField(dest, obj, field) => {
                buf[1] = dest;
                buf[2] = obj;
                let b = field.to_le_bytes();
                buf[3] = b[0];
                buf[4] = b[1];
            }
            Instruction::SetField(obj, field, val) => {
                buf[1] = obj;
                let b = field.to_le_bytes();
                buf[2] = b[0];
                buf[3] = b[1];
                buf[4] = val;
            }
            Instruction::GetFieldSlot(dest, obj, slot) => {
                buf[1] = dest;
                buf[2] = obj;
                buf[3] = slot;
            }
            Instruction::SetFieldSlot(obj, slot, val) => {
                buf[1] = obj;
                buf[2] = slot;
                buf[3] = val;
            }
            Instruction::ArrayIndexOk(v, c, arr, idx) => {
                buf[1] = v;
                buf[2] = c;
                buf[3] = arr;
                buf[4] = idx;
            }
            Instruction::SetIndex(arr, idx, val) => {
                buf[1] = arr;
                buf[2] = idx;
                buf[3] = val;
            }
            Instruction::CallMethod(obj, method, args, cnt, dest) => {
                buf[1] = obj;
                let b = method.to_le_bytes();
                buf[2] = b[0];
                buf[3] = b[1];
                buf[4] = args;
                buf[5] = cnt;
                buf[6] = dest;
            }
            Instruction::TypeIs(dest, val, type_name) => {
                buf[1] = dest;
                buf[2] = val;
                let b = type_name.to_le_bytes();
                buf[3] = b[0];
                buf[4] = b[1];
            }
            Instruction::TryCast(dest, val, type_name) => {
                buf[1] = dest;
                buf[2] = val;
                let b = type_name.to_le_bytes();
                buf[3] = b[0];
                buf[4] = b[1];
            }
            Instruction::LoadUpvalue(dest, idx) => {
                buf[1] = dest;
                buf[2] = idx;
            }
            Instruction::StoreUpvalue(idx, src) => {
                buf[1] = idx;
                buf[2] = src;
            }
            Instruction::Closure(dest, func, upvals, cnt) => {
                buf[1] = dest;
                let b = func.to_le_bytes();
                buf[2] = b[0];
                buf[3] = b[1];
                buf[4] = upvals;
                buf[5] = cnt;
            }
            Instruction::AddIntImm(dest, src, imm) => {
                buf[1] = dest;
                buf[2] = src;
                buf[3] = imm as u8;
            }
            Instruction::BranchLtInt(lhs, rhs, offset)
            | Instruction::BranchLeInt(lhs, rhs, offset) => {
                buf[1] = lhs;
                buf[2] = rhs;
                let b = offset.to_le_bytes();
                buf[3] = b[0];
                buf[4] = b[1];
            }
            Instruction::GetFieldAddSetInt(obj, slot, addend) => {
                buf[1] = obj;
                buf[2] = slot;
                buf[3] = addend;
            }
        }
        buf
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
        if bytes.len() < Self::ENCODED_SIZE {
            return Err(DecodeError::Truncated {
                expected: Self::ENCODED_SIZE,
                found: bytes.len(),
            });
        }
        let opcode = OpCode::from_u8(bytes[0])?;
        let u16_at = |idx: usize| u16::from_le_bytes([bytes[idx], bytes[idx + 1]]);
        let i16_at = |idx: usize| i16::from_le_bytes([bytes[idx], bytes[idx + 1]]);

        match opcode {
            OpCode::LoadNil => Ok(Instruction::LoadNil(bytes[1])),
            OpCode::LoadBool => {
                let val = match bytes[2] {
                    0 => false,
                    1 => true,
                    other => return Err(DecodeError::InvalidBoolean(other)),
                };
                Ok(Instruction::LoadBool(bytes[1], val))
            }
            OpCode::LoadConst => Ok(Instruction::LoadConst(bytes[1], u16_at(2))),
            OpCode::LoadGlobal => Ok(Instruction::LoadGlobal(bytes[1], u16_at(2))),
            OpCode::StoreGlobal => Ok(Instruction::StoreGlobal(u16_at(2), bytes[1])),
            OpCode::Move => Ok(Instruction::Move(bytes[1], bytes[2])),
            OpCode::Add => Ok(Instruction::Add(bytes[1], bytes[2], bytes[3])),
            OpCode::Sub => Ok(Instruction::Sub(bytes[1], bytes[2], bytes[3])),
            OpCode::Mul => Ok(Instruction::Mul(bytes[1], bytes[2], bytes[3])),
            OpCode::Div => Ok(Instruction::Div(bytes[1], bytes[2], bytes[3])),
            OpCode::Mod => Ok(Instruction::Mod(bytes[1], bytes[2], bytes[3])),
            OpCode::Pow => Ok(Instruction::Pow(bytes[1], bytes[2], bytes[3])),
            OpCode::FloorDiv => Ok(Instruction::FloorDiv(bytes[1], bytes[2], bytes[3])),
            OpCode::Neg => Ok(Instruction::Neg(bytes[1], bytes[2])),
            OpCode::Eq => Ok(Instruction::Eq(bytes[1], bytes[2], bytes[3])),
            OpCode::Ne => Ok(Instruction::Ne(bytes[1], bytes[2], bytes[3])),
            OpCode::Lt => Ok(Instruction::Lt(bytes[1], bytes[2], bytes[3])),
            OpCode::Le => Ok(Instruction::Le(bytes[1], bytes[2], bytes[3])),
            OpCode::Gt => Ok(Instruction::Gt(bytes[1], bytes[2], bytes[3])),
            OpCode::Ge => Ok(Instruction::Ge(bytes[1], bytes[2], bytes[3])),
            OpCode::AddInt => Ok(Instruction::AddInt(bytes[1], bytes[2], bytes[3])),
            OpCode::SubInt => Ok(Instruction::SubInt(bytes[1], bytes[2], bytes[3])),
            OpCode::MulInt => Ok(Instruction::MulInt(bytes[1], bytes[2], bytes[3])),
            OpCode::DivInt => Ok(Instruction::DivInt(bytes[1], bytes[2], bytes[3])),
            OpCode::ModInt => Ok(Instruction::ModInt(bytes[1], bytes[2], bytes[3])),
            OpCode::PowInt => Ok(Instruction::PowInt(bytes[1], bytes[2], bytes[3])),
            OpCode::FloorDivInt => Ok(Instruction::FloorDivInt(bytes[1], bytes[2], bytes[3])),
            OpCode::NegInt => Ok(Instruction::NegInt(bytes[1], bytes[2])),
            OpCode::EqInt => Ok(Instruction::EqInt(bytes[1], bytes[2], bytes[3])),
            OpCode::NeInt => Ok(Instruction::NeInt(bytes[1], bytes[2], bytes[3])),
            OpCode::LtInt => Ok(Instruction::LtInt(bytes[1], bytes[2], bytes[3])),
            OpCode::LeInt => Ok(Instruction::LeInt(bytes[1], bytes[2], bytes[3])),
            OpCode::GtInt => Ok(Instruction::GtInt(bytes[1], bytes[2], bytes[3])),
            OpCode::GeInt => Ok(Instruction::GeInt(bytes[1], bytes[2], bytes[3])),
            OpCode::AddFloat => Ok(Instruction::AddFloat(bytes[1], bytes[2], bytes[3])),
            OpCode::SubFloat => Ok(Instruction::SubFloat(bytes[1], bytes[2], bytes[3])),
            OpCode::MulFloat => Ok(Instruction::MulFloat(bytes[1], bytes[2], bytes[3])),
            OpCode::DivFloat => Ok(Instruction::DivFloat(bytes[1], bytes[2], bytes[3])),
            OpCode::ModFloat => Ok(Instruction::ModFloat(bytes[1], bytes[2], bytes[3])),
            OpCode::PowFloat => Ok(Instruction::PowFloat(bytes[1], bytes[2], bytes[3])),
            OpCode::FloorDivFloat => Ok(Instruction::FloorDivFloat(bytes[1], bytes[2], bytes[3])),
            OpCode::NegFloat => Ok(Instruction::NegFloat(bytes[1], bytes[2])),
            OpCode::EqFloat => Ok(Instruction::EqFloat(bytes[1], bytes[2], bytes[3])),
            OpCode::NeFloat => Ok(Instruction::NeFloat(bytes[1], bytes[2], bytes[3])),
            OpCode::LtFloat => Ok(Instruction::LtFloat(bytes[1], bytes[2], bytes[3])),
            OpCode::LeFloat => Ok(Instruction::LeFloat(bytes[1], bytes[2], bytes[3])),
            OpCode::GtFloat => Ok(Instruction::GtFloat(bytes[1], bytes[2], bytes[3])),
            OpCode::GeFloat => Ok(Instruction::GeFloat(bytes[1], bytes[2], bytes[3])),
            OpCode::And => Ok(Instruction::And(bytes[1], bytes[2], bytes[3])),
            OpCode::Or => Ok(Instruction::Or(bytes[1], bytes[2], bytes[3])),
            OpCode::Not => Ok(Instruction::Not(bytes[1], bytes[2])),
            OpCode::BitAnd => Ok(Instruction::BitAnd(bytes[1], bytes[2], bytes[3])),
            OpCode::BitOr => Ok(Instruction::BitOr(bytes[1], bytes[2], bytes[3])),
            OpCode::BitXor => Ok(Instruction::BitXor(bytes[1], bytes[2], bytes[3])),
            OpCode::Shl => Ok(Instruction::Shl(bytes[1], bytes[2], bytes[3])),
            OpCode::Shr => Ok(Instruction::Shr(bytes[1], bytes[2], bytes[3])),
            OpCode::BitNot => Ok(Instruction::BitNot(bytes[1], bytes[2])),
            OpCode::Jump => Ok(Instruction::Jump(i16_at(1))),
            OpCode::JumpIf => Ok(Instruction::JumpIf(bytes[1], i16_at(2))),
            OpCode::JumpIfNot => Ok(Instruction::JumpIfNot(bytes[1], i16_at(2))),
            OpCode::Call => Ok(Instruction::Call(bytes[1], bytes[2], bytes[3], bytes[4])),
            OpCode::Return => Ok(Instruction::Return(bytes[1])),
            OpCode::NewArray => Ok(Instruction::NewArray(bytes[1], bytes[2], bytes[3])),
            OpCode::NewMap => Ok(Instruction::NewMap(bytes[1])),
            OpCode::NewStruct => Ok(Instruction::NewStruct(
                bytes[1],
                u16_at(2),
                u16_at(4),
                bytes[6],
                bytes[7],
            )),
            OpCode::NewEnumUnit => Ok(Instruction::NewEnumUnit(bytes[1], u16_at(2), u16_at(4))),
            OpCode::NewEnumVariant => Ok(Instruction::NewEnumVariant(
                bytes[1],
                u16_at(2),
                u16_at(4),
                bytes[6],
                bytes[7],
            )),
            OpCode::TupleNew => Ok(Instruction::TupleNew(bytes[1], bytes[2], bytes[3])),
            OpCode::TupleGet => Ok(Instruction::TupleGet(bytes[1], bytes[2], bytes[3])),
            OpCode::IsEnumVariant => Ok(Instruction::IsEnumVariant(
                bytes[1],
                bytes[2],
                u16_at(3),
                u16_at(5),
            )),
            OpCode::GetEnumValue => Ok(Instruction::GetEnumValue(bytes[1], bytes[2], bytes[3])),
            OpCode::GetFieldCompact | OpCode::SetFieldCompact => {
                Err(DecodeError::UnknownOpCode(bytes[0]))
            }
            OpCode::GetField => Ok(Instruction::GetField(bytes[1], bytes[2], u16_at(3))),
            OpCode::SetField => Ok(Instruction::SetField(bytes[1], u16_at(2), bytes[4])),
            OpCode::GetFieldSlot => Ok(Instruction::GetFieldSlot(bytes[1], bytes[2], bytes[3])),
            OpCode::SetFieldSlot => Ok(Instruction::SetFieldSlot(bytes[1], bytes[2], bytes[3])),
            OpCode::GetIndex => Ok(Instruction::GetIndex(bytes[1], bytes[2], bytes[3])),
            OpCode::TryGetIndex => Ok(Instruction::TryGetIndex(bytes[1], bytes[2], bytes[3])),
            OpCode::ArrayIndexOk => Ok(Instruction::ArrayIndexOk(
                bytes[1], bytes[2], bytes[3], bytes[4],
            )),
            OpCode::ArrayLen => Ok(Instruction::ArrayLen(bytes[1], bytes[2])),
            OpCode::ArrayPush => Ok(Instruction::ArrayPush(bytes[1], bytes[2])),
            OpCode::ArrayGetUnwrap => Ok(Instruction::ArrayGetUnwrap(bytes[1], bytes[2], bytes[3])),
            OpCode::SetIndex => Ok(Instruction::SetIndex(bytes[1], bytes[2], bytes[3])),
            OpCode::Concat => Ok(Instruction::Concat(bytes[1], bytes[2], bytes[3])),
            OpCode::CallMethod => Ok(Instruction::CallMethod(
                bytes[1],
                u16_at(2),
                bytes[4],
                bytes[5],
                bytes[6],
            )),
            OpCode::TypeIs => Ok(Instruction::TypeIs(bytes[1], bytes[2], u16_at(3))),
            OpCode::TryCast => Ok(Instruction::TryCast(bytes[1], bytes[2], u16_at(3))),
            OpCode::LoadUpvalue => Ok(Instruction::LoadUpvalue(bytes[1], bytes[2])),
            OpCode::StoreUpvalue => Ok(Instruction::StoreUpvalue(bytes[1], bytes[2])),
            OpCode::Closure => Ok(Instruction::Closure(
                bytes[1],
                u16_at(2),
                bytes[4],
                bytes[5],
            )),
            OpCode::AddIntImm => Ok(Instruction::AddIntImm(
                bytes[1],
                bytes[2],
                bytes[3] as i8,
            )),
            OpCode::BranchLtInt => Ok(Instruction::BranchLtInt(
                bytes[1],
                bytes[2],
                i16_at(3),
            )),
            OpCode::BranchLeInt => Ok(Instruction::BranchLeInt(
                bytes[1],
                bytes[2],
                i16_at(3),
            )),
            OpCode::GetFieldAddSetInt => Ok(Instruction::GetFieldAddSetInt(
                bytes[1], bytes[2], bytes[3],
            )),
        }
    }
}
