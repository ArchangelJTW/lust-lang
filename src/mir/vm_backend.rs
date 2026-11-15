use crate::bytecode::{Chunk, Function, Instruction, Register, Value};
use crate::bytecode::value::{native_fn, NativeCallResult};
use crate::data::layout::CanonicalTypeTag;
use crate::data::materialize::{
    decode_json_into_contiguous, ContiguousDecodeOptions, ContiguousStruct, LayoutPerfCounters,
};
use crate::mir::ir::*;
use crate::number::{LustFloat, LustInt};
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;
use hashbrown::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MirVmBackendError {
    #[error("Binary operator '{0:?}' is not supported by MIR VM backend")]
    UnsupportedBinaryOp(MirBinaryOp),
}

pub struct MirToBytecodeCompiler<'a> {
    func: &'a MirFunction,
    chunk: Chunk,
    scratch_start: usize,
    scratch_count: usize,
    max_scratch: usize,
    block_pcs: HashMap<BlockId, usize>,
    pending_jumps: Vec<(usize, BlockId)>,
    direct_callees: Option<&'a HashSet<String>>,
}

impl<'a> MirToBytecodeCompiler<'a> {
    pub fn compile(func: &'a MirFunction) -> Result<Function, MirVmBackendError> {
        Self::compile_inner(func, None)
    }

    pub fn compile_with_direct_callees(
        func: &'a MirFunction,
        direct_callees: &'a HashSet<String>,
    ) -> Result<Function, MirVmBackendError> {
        Self::compile_inner(func, Some(direct_callees))
    }

    fn compile_inner(
        func: &'a MirFunction,
        direct_callees: Option<&'a HashSet<String>>,
    ) -> Result<Function, MirVmBackendError> {
        let param_count = func.params.len() as u8;
        let mut bytecode_fn = Function::new(func.signature.name.clone(), param_count, false);

        let mut compiler = MirToBytecodeCompiler {
            func,
            chunk: Chunk::new(),
            scratch_start: func.locals.len(),
            scratch_count: 0,
            max_scratch: 0,
            block_pcs: HashMap::new(),
            pending_jumps: Vec::new(),
            direct_callees,
        };

        compiler.compile_blocks()?;
        compiler.patch_all_jumps();

        bytecode_fn.chunk = compiler.chunk;
        bytecode_fn.register_count = (func.locals.len() + compiler.max_scratch) as u8;
        Ok(bytecode_fn)
    }

    fn next_scratch(&mut self) -> Register {
        let reg = (self.scratch_start + self.scratch_count) as Register;
        self.scratch_count += 1;
        if self.scratch_count > self.max_scratch {
            self.max_scratch = self.scratch_count;
        }
        reg
    }

    fn reset_scratch(&mut self) {
        self.scratch_count = 0;
    }

    fn compile_blocks(&mut self) -> Result<(), MirVmBackendError> {
        for block in &self.func.blocks {
            let pc = self.chunk.instructions.len();
            self.block_pcs.insert(block.id, pc);

            for stmt in &block.statements {
                self.reset_scratch();
                self.compile_statement(stmt)?;
            }

            self.reset_scratch();
            if let Some(term) = &block.terminator {
                self.compile_terminator(term);
            }
        }
        Ok(())
    }

    fn operand_to_reg(&mut self, op: &Operand, line: usize) -> Register {
        match op {
            Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => loc.0 as Register,
            Operand::Copy(Place::Index(base, idx)) | Operand::Move(Place::Index(base, idx)) => {
                let reg = self.next_scratch();
                let base_reg = self.place_to_reg(base);
                let idx_reg = idx.0 as Register;
                self.chunk.emit(Instruction::GetIndex(reg, base_reg, idx_reg), line);
                reg
            }
            Operand::Copy(Place::Field(base, f)) | Operand::Move(Place::Field(base, f)) => {
                let reg = self.next_scratch();
                let base_reg = self.place_to_reg(base);
                let idx_reg = self.next_scratch();
                self.load_constant_into_reg(&MirConstant::Int(*f as i64), idx_reg, line);
                self.chunk.emit(Instruction::GetIndex(reg, base_reg, idx_reg), line);
                reg
            }
            Operand::Constant(c) => {
                let reg = self.next_scratch();
                self.load_constant_into_reg(c, reg, line);
                reg
            }
        }
    }

    fn load_constant_into_reg(&mut self, c: &MirConstant, reg: Register, line: usize) {
        match c {
            MirConstant::Int(i) => {
                let const_idx = self.chunk.add_constant(Value::Int(*i as LustInt));
                self.chunk.emit(Instruction::LoadConst(reg, const_idx), line);
            }
            MirConstant::Float(f) => {
                let const_idx = self.chunk.add_constant(Value::Float(*f as LustFloat));
                self.chunk.emit(Instruction::LoadConst(reg, const_idx), line);
            }
            MirConstant::Bool(b) => {
                self.chunk.emit(Instruction::LoadBool(reg, *b), line);
            }
            MirConstant::String(s) => {
                let const_idx = self.chunk.add_constant(Value::string(s.clone()));
                self.chunk.emit(Instruction::LoadConst(reg, const_idx), line);
            }
            MirConstant::Unit => {
                self.chunk.emit(Instruction::LoadNil(reg), line);
            }
        }
    }

    fn place_to_reg(&self, place: &Place) -> Register {
        match place {
            Place::Local(id) => id.0 as Register,
            Place::Index(base, _) => self.place_to_reg(base),
            Place::Field(base, _) => self.place_to_reg(base),
        }
    }

    fn compile_rvalue(&mut self, rvalue: &Rvalue, dst_reg: Register, line: usize) -> Result<(), MirVmBackendError> {
        match rvalue {
            Rvalue::Use(op) => {
                match op {
                    Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src)) => {
                        if dst_reg != src.0 as Register {
                            self.chunk.emit(Instruction::Move(dst_reg, src.0 as Register), line);
                        }
                    }
                    Operand::Copy(Place::Index(base, idx)) | Operand::Move(Place::Index(base, idx)) => {
                        let base_reg = self.place_to_reg(base);
                        let idx_reg = idx.0 as Register;
                        self.chunk.emit(Instruction::GetIndex(dst_reg, base_reg, idx_reg), line);
                    }
                    Operand::Copy(Place::Field(base, f)) | Operand::Move(Place::Field(base, f)) => {
                        let base_reg = self.place_to_reg(base);
                        let idx_reg = self.next_scratch();
                        self.load_constant_into_reg(&MirConstant::Int(*f as i64), idx_reg, line);
                        self.chunk.emit(Instruction::GetIndex(dst_reg, base_reg, idx_reg), line);
                    }
                    Operand::Constant(c) => {
                        self.load_constant_into_reg(c, dst_reg, line);
                    }
                }
            }
            Rvalue::Cast(op, target_ty) => {
                let r_op = self.operand_to_reg(op, line);
                match target_ty {
                    MirType::Float => {
                        let zero_f_reg = self.next_scratch();
                        self.load_constant_into_reg(&MirConstant::Float(0.0), zero_f_reg, line);
                        self.chunk.emit(Instruction::Add(dst_reg, r_op, zero_f_reg), line);
                    }
                    MirType::Int => {
                        let obj_idx = self.chunk.add_constant(Value::string("math".to_string()));
                        let field_idx = self.chunk.add_constant(Value::string("floor".to_string()));
                        let func_reg = self.next_scratch();
                        self.chunk.emit(Instruction::LoadGlobal(func_reg, obj_idx), line);
                        self.chunk.emit(Instruction::GetField(func_reg, func_reg, field_idx), line);
                        let arg_reg = self.next_scratch();
                        self.chunk.emit(Instruction::Move(arg_reg, r_op), line);
                        self.chunk.emit(Instruction::Call(func_reg, arg_reg, 1, dst_reg), line);
                    }
                    _ => {
                        if dst_reg != r_op {
                            self.chunk.emit(Instruction::Move(dst_reg, r_op), line);
                        }
                    }
                }
            }
            Rvalue::BinaryOp(binop, lhs, rhs) => {
                let is_string = match (lhs, rhs) {
                    (Operand::Constant(MirConstant::String(_)), _) | (_, Operand::Constant(MirConstant::String(_))) => true,
                    (Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l)), _) if self.func.locals[l.0].ty == MirType::String => true,
                    (_, Operand::Copy(Place::Local(r)) | Operand::Move(Place::Local(r))) if self.func.locals[r.0].ty == MirType::String => true,
                    _ => false,
                };
                let r_lhs = self.operand_to_reg(lhs, line);
                let r_rhs = self.operand_to_reg(rhs, line);
                if is_string && *binop == MirBinaryOp::Add {
                    self.chunk.emit(Instruction::Concat(dst_reg, r_lhs, r_rhs), line);
                } else {
                    self.compile_binop(*binop, dst_reg, r_lhs, r_rhs, line)?;
                }
            }
            Rvalue::UnaryOp(unop, op) => {
                let r_op = self.operand_to_reg(op, line);
                match unop {
                    MirUnaryOp::Neg => {
                        self.chunk.emit(Instruction::Neg(dst_reg, r_op), line);
                    }
                    MirUnaryOp::Not => {
                        self.chunk.emit(Instruction::Not(dst_reg, r_op), line);
                    }
                    MirUnaryOp::BitNot => {
                        self.chunk.emit(Instruction::BitNot(dst_reg, r_op), line);
                    }
                    MirUnaryOp::Len => {
                        self.chunk.emit(Instruction::ArrayLen(dst_reg, r_op), line);
                    }
                }
            }
            Rvalue::Aggregate(agg_ty, elements) => {
                let count = elements.len() as u8;
                let scratch_base = if count == 0 {
                    0
                } else {
                    let s_base = (self.scratch_start + self.scratch_count) as Register;
                    self.scratch_count += elements.len();
                    if self.scratch_count > self.max_scratch {
                        self.max_scratch = self.scratch_count;
                    }
                    for (i, elem) in elements.iter().enumerate() {
                        let target_reg = s_base + i as u8;
                        match elem {
                            Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => {
                                self.chunk.emit(Instruction::Move(target_reg, loc.0 as Register), line);
                            }
                            Operand::Copy(Place::Index(base, idx)) | Operand::Move(Place::Index(base, idx)) => {
                                let base_reg = self.place_to_reg(base);
                                let idx_reg = idx.0 as Register;
                                self.chunk.emit(Instruction::GetIndex(target_reg, base_reg, idx_reg), line);
                            }
                            Operand::Copy(Place::Field(base, f)) | Operand::Move(Place::Field(base, f)) => {
                                let base_reg = self.place_to_reg(base);
                                let idx_reg = self.next_scratch();
                                self.load_constant_into_reg(&MirConstant::Int(*f as i64), idx_reg, line);
                                self.chunk.emit(Instruction::GetIndex(target_reg, base_reg, idx_reg), line);
                            }
                            Operand::Constant(c) => {
                                self.load_constant_into_reg(c, target_reg, line);
                            }
                        }
                    }
                    s_base
                };
                match agg_ty {
                    MirType::Tuple(_) => {
                        self.chunk.emit(Instruction::TupleNew(dst_reg, scratch_base, count), line);
                    }
                    _ => {
                        self.chunk.emit(Instruction::NewArray(dst_reg, scratch_base, count), line);
                    }
                }
            }
        }
        Ok(())
    }

    fn compile_statement(&mut self, stmt: &Statement) -> Result<(), MirVmBackendError> {
        match stmt {
            Statement::Assign(Place::Local(dst), rvalue, span) => {
                let dst_reg = dst.0 as Register;
                let line = span.start_line;
                self.compile_rvalue(rvalue, dst_reg, line)?;
            }
            Statement::Assign(Place::Index(base, idx), rvalue, span) => {
                let line = span.start_line;
                let val_reg = match rvalue {
                    Rvalue::Use(Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src))) => {
                        src.0 as Register
                    }
                    _ => {
                        let temp_reg = self.next_scratch();
                        self.compile_rvalue(rvalue, temp_reg, line)?;
                        temp_reg
                    }
                };
                let base_reg = self.place_to_reg(base);
                let idx_reg = idx.0 as Register;
                self.chunk.emit(Instruction::SetIndex(base_reg, idx_reg, val_reg), line);
            }
            Statement::Assign(Place::Field(base, f), rvalue, span) => {
                let line = span.start_line;
                let val_reg = match rvalue {
                    Rvalue::Use(Operand::Copy(Place::Local(src)) | Operand::Move(Place::Local(src))) => {
                        src.0 as Register
                    }
                    _ => {
                        let temp_reg = self.next_scratch();
                        self.compile_rvalue(rvalue, temp_reg, line)?;
                        temp_reg
                    }
                };
                let base_reg = self.place_to_reg(base);
                let idx_reg = self.next_scratch();
                self.load_constant_into_reg(&MirConstant::Int(*f as i64), idx_reg, line);
                self.chunk.emit(Instruction::SetIndex(base_reg, idx_reg, val_reg), line);
            }
            Statement::StorageLive(_, _) | Statement::StorageDead(_, _) | Statement::ResourceCheck(_, _, _) => {}
        }
        Ok(())
    }

    fn compile_binop(&mut self, op: MirBinaryOp, dst: Register, lhs: Register, rhs: Register, line: usize) -> Result<(), MirVmBackendError> {
        let instr = match op {
            MirBinaryOp::Add => Instruction::Add(dst, lhs, rhs),
            MirBinaryOp::Sub => Instruction::Sub(dst, lhs, rhs),
            MirBinaryOp::Mul => Instruction::Mul(dst, lhs, rhs),
            MirBinaryOp::Div => Instruction::Div(dst, lhs, rhs),
            MirBinaryOp::Mod => Instruction::Mod(dst, lhs, rhs),
            MirBinaryOp::FloorDiv => Instruction::FloorDiv(dst, lhs, rhs),
            MirBinaryOp::Pow => Instruction::Pow(dst, lhs, rhs),
            MirBinaryOp::BitAnd => Instruction::BitAnd(dst, lhs, rhs),
            MirBinaryOp::BitOr => Instruction::BitOr(dst, lhs, rhs),
            MirBinaryOp::BitXor => Instruction::BitXor(dst, lhs, rhs),
            MirBinaryOp::Shl => Instruction::Shl(dst, lhs, rhs),
            MirBinaryOp::Shr => Instruction::Shr(dst, lhs, rhs),
            MirBinaryOp::Eq => Instruction::Eq(dst, lhs, rhs),
            MirBinaryOp::NotEq => Instruction::Ne(dst, lhs, rhs),
            MirBinaryOp::Lt => Instruction::Lt(dst, lhs, rhs),
            MirBinaryOp::LtEq => Instruction::Le(dst, lhs, rhs),
            MirBinaryOp::Gt => Instruction::Gt(dst, lhs, rhs),
            MirBinaryOp::GtEq => Instruction::Ge(dst, lhs, rhs),
            MirBinaryOp::And => Instruction::And(dst, lhs, rhs),
            MirBinaryOp::Or => Instruction::Or(dst, lhs, rhs),
        };
        self.chunk.emit(instr, line);
        Ok(())
    }

    fn compile_terminator(&mut self, term: &Terminator) {
        match term {
            Terminator::Goto(target, span) => {
                let jump_idx = self.chunk.emit(Instruction::Jump(0), span.start_line);
                self.pending_jumps.push((jump_idx, *target));
            }
            Terminator::Cond { cond, then_block, else_block, span } => {
                let r_cond = self.operand_to_reg(cond, span.start_line);
                let jump_if_not = self.chunk.emit(Instruction::JumpIfNot(r_cond, 0), span.start_line);
                self.pending_jumps.push((jump_if_not, *else_block));

                let jump_then = self.chunk.emit(Instruction::Jump(0), span.start_line);
                self.pending_jumps.push((jump_then, *then_block));
            }
            Terminator::Switch { discr, cases, default, span } => {
                let line = span.start_line;
                let discr_reg = self.operand_to_reg(discr, line);
                for (val, target_bb) in cases {
                    let val_reg = self.next_scratch();
                    self.load_constant_into_reg(&MirConstant::Int(*val), val_reg, line);
                    let cmp_reg = self.next_scratch();
                    self.chunk.emit(Instruction::EqInt(cmp_reg, discr_reg, val_reg), line);
                    let jump_next_case = self.chunk.emit(Instruction::JumpIfNot(cmp_reg, 0), line);
                    let jump_target = self.chunk.emit(Instruction::Jump(0), line);
                    self.pending_jumps.push((jump_target, *target_bb));
                    let next_pc = self.chunk.instructions.len();
                    self.chunk.patch_jump(jump_next_case, next_pc);
                }
                let default_jump = self.chunk.emit(Instruction::Jump(0), line);
                self.pending_jumps.push((default_jump, *default));
            }
            Terminator::Return(opt_op, span) => {
                let r_ret = if let Some(op) = opt_op {
                    self.operand_to_reg(op, span.start_line)
                } else {
                    0
                };
                self.chunk.emit(Instruction::Return(r_ret), span.start_line);
            }
            Terminator::Call { callee, args, destination, success, error: _, span } => {
                let line = span.start_line;
                let arg_count = args.len() as u8;
                let first_arg_reg = if arg_count > 0 {
                    let scratch_base = self.next_scratch();
                    for _ in 1..args.len() {
                        self.next_scratch();
                    }
                    for (i, arg) in args.iter().enumerate() {
                        let target_reg = scratch_base + i as u8;
                        match arg {
                            Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => {
                                self.chunk.emit(Instruction::Move(target_reg, loc.0 as Register), line);
                            }
                            Operand::Copy(Place::Index(base, idx)) | Operand::Move(Place::Index(base, idx)) => {
                                let base_reg = self.place_to_reg(base);
                                let idx_reg = idx.0 as Register;
                                self.chunk.emit(Instruction::GetIndex(target_reg, base_reg, idx_reg), line);
                            }
                            Operand::Copy(Place::Field(base, f)) | Operand::Move(Place::Field(base, f)) => {
                                let base_reg = self.place_to_reg(base);
                                let idx_reg = self.next_scratch();
                                self.load_constant_into_reg(&MirConstant::Int(*f as i64), idx_reg, line);
                                self.chunk.emit(Instruction::GetIndex(target_reg, base_reg, idx_reg), line);
                            }
                            Operand::Constant(c) => {
                                self.load_constant_into_reg(c, target_reg, line);
                            }
                        }
                    }
                    scratch_base
                } else {
                    0
                };

                let func_reg = self.next_scratch();
                let is_direct = self
                    .direct_callees
                    .is_some_and(|set| set.contains(callee.as_str()));
                if !is_direct
                    && let Some((obj_name, field_name)) = callee.split_once('.')
                {
                    let obj_idx = self.chunk.add_constant(Value::string(obj_name.to_string()));
                    let field_idx = self.chunk.add_constant(Value::string(field_name.to_string()));
                    self.chunk.emit(Instruction::LoadGlobal(func_reg, obj_idx), line);
                    self.chunk.emit(Instruction::GetField(func_reg, func_reg, field_idx), line);
                } else {
                    let name_idx = self.chunk.add_constant(Value::string(callee.clone()));
                    self.chunk.emit(Instruction::LoadGlobal(func_reg, name_idx), line);
                }

                let dest_reg = self.place_to_reg(destination);
                self.chunk.emit(Instruction::Call(func_reg, first_arg_reg, arg_count, dest_reg), line);

                let jump_idx = self.chunk.emit(Instruction::Jump(0), line);
                self.pending_jumps.push((jump_idx, *success));
            }
            Terminator::Unreachable(span) => {
                self.chunk.emit(Instruction::Return(0), span.start_line);
            }
            _ => {}
        }
    }

    fn patch_all_jumps(&mut self) {
        for (jump_idx, target) in &self.pending_jumps {
            if let Some(target_pc) = self.block_pcs.get(target) {
                self.chunk.patch_jump(*jump_idx, *target_pc);
            } else {
                panic!("Target basic block {:?} not found during jump patching", target);
            }
        }
    }

    pub fn compile_module(module: &'a MirModule) -> Result<Vec<Function>, MirVmBackendError> {
        let mut out = Vec::with_capacity(module.functions.len());
        for func in &module.functions {
            out.push(Self::compile(func)?);
        }
        Ok(out)
    }
}

/// Structural bytecode audit for Gate-SCHEMA-09 zero name/hash lookup verification.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChunkFieldAccessAudit {
    pub get_field_count: usize,
    pub set_field_count: usize,
    pub get_index_count: usize,
    pub set_index_count: usize,
}

impl ChunkFieldAccessAudit {
    pub fn has_zero_name_lookup(&self) -> bool {
        self.get_field_count == 0 && self.set_field_count == 0
    }
}

/// Audits a compiled VM bytecode chunk to verify zero `GetField`/`SetField` name lookups.
pub fn audit_chunk_field_access(chunk: &Chunk) -> ChunkFieldAccessAudit {
    let mut audit = ChunkFieldAccessAudit::default();
    for instr in &chunk.instructions {
        match instr {
            Instruction::GetField(..) => audit.get_field_count += 1,
            Instruction::SetField(..) => audit.set_field_count += 1,
            Instruction::GetIndex(..) => audit.get_index_count += 1,
            Instruction::SetIndex(..) => audit.set_index_count += 1,
            _ => {}
        }
    }
    audit
}

fn contiguous_to_vm_slots(s: &ContiguousStruct) -> Result<Value, String> {
    let mut slots = Vec::with_capacity(s.layout.fields.len());
    for field in &s.layout.fields {
        let ord = field.ordinal;
        let off = field.byte_offset;
        if s.is_null(ord) || !s.is_initialized(ord) {
            slots.push(Value::Nil);
            continue;
        }
        let val = match field.tag {
            CanonicalTypeTag::Bool => {
                Value::Bool(s.read_bool_at(ord, off, None).map_err(|e| e.to_string())?.unwrap_or(false))
            }
            CanonicalTypeTag::Int64 | CanonicalTypeTag::DateTimeUtc => {
                Value::Int(s.read_i64_at(ord, off, None).map_err(|e| e.to_string())?.unwrap_or(0) as LustInt)
            }
            CanonicalTypeTag::UInt64 => {
                let u = s.read_u64_at(ord, off, None).map_err(|e| e.to_string())?.unwrap_or(0);
                let i = i64::try_from(u).map_err(|_| "UInt64 overflows VM Int".to_string())?;
                Value::Int(i as LustInt)
            }
            CanonicalTypeTag::Float64 => {
                Value::Float(s.read_f64_at(ord, off, None).map_err(|e| e.to_string())?.unwrap_or(0.0) as LustFloat)
            }
            CanonicalTypeTag::StringSlice => {
                let text = s.read_str_at(ord, off, None).map_err(|e| e.to_string())?.unwrap_or("");
                Value::string(text.to_string())
            }
            CanonicalTypeTag::BinarySlice => {
                let bytes = s.read_bytes_at(ord, off, None).map_err(|e| e.to_string())?.unwrap_or(&[]);
                Value::string(String::from_utf8_lossy(bytes).into_owned())
            }
            CanonicalTypeTag::NestedStruct(_) => {
                if let Some(nested) = s.read_nested_at(ord, None).map_err(|e| e.to_string())? {
                    contiguous_to_vm_slots(nested)?
                } else {
                    Value::Nil
                }
            }
            CanonicalTypeTag::ObjectId => {
                let oid = s.read_object_id_at(ord, off, None).map_err(|e| e.to_string())?.unwrap_or([0u8; 12]);
                let mut hex = String::with_capacity(24);
                for b in oid {
                    use core::fmt::Write;
                    let _ = write!(&mut hex, "{:02x}", b);
                }
                Value::string(hex)
            }
            CanonicalTypeTag::Decimal128 => {
                let dec = s.read_decimal128_at(ord, off, None).map_err(|e| e.to_string())?.unwrap_or(0);
                Value::Int(dec as LustInt)
            }
        };
        slots.push(val);
    }
    Ok(Value::array(slots))
}

/// Registers `StructDecodeInto` native decoders for all layouts in `module` onto `vm`.
pub fn register_module_struct_decoders(vm: &mut crate::vm::VM, module: &MirModule) {
    let counters = Rc::new(RefCell::new(LayoutPerfCounters::default()));
    register_struct_decode_natives_with_counters(vm, &module.struct_layouts, counters);
}

/// Registers `StructDecodeInto` native decoders with shared instrumentation counters.
pub fn register_struct_decode_natives_with_counters(
    vm: &mut crate::vm::VM,
    layouts: &[MirStructLayout],
    counters: Rc<RefCell<LayoutPerfCounters>>,
) {
    let mut nested_map = std::collections::HashMap::new();
    for l in layouts {
        nested_map.insert(l.type_id, l.to_logical_layout());
    }
    let nested_map = Rc::new(nested_map);

    for layout in layouts {
        let callee = mir_struct_decode_callee(&layout.type_name);
        let logical = layout.to_logical_layout();
        let counters_ref = Rc::clone(&counters);
        let nested_ref = Rc::clone(&nested_map);
        let func = native_fn(move |args: &[Value]| {
            let input_str = args
                .first()
                .and_then(|v| v.as_string())
                .ok_or_else(|| "StructDecodeInto expects a JSON string argument".to_string())?;
            let opts = ContiguousDecodeOptions {
                cancel_check: None,
                nested_layouts: Some(nested_ref.as_ref()),
                require_nominal_tag: false,
            };
            let mut c = counters_ref.borrow_mut();
            let decoded = decode_json_into_contiguous(&logical, input_str.as_bytes(), &mut c, &opts)
                .map_err(|e| e.to_string())?;
            let slots = contiguous_to_vm_slots(&decoded)?;
            Ok(NativeCallResult::Return(slots))
        });
        vm.register_native(callee, Value::NativeFunction(func));
    }
}
