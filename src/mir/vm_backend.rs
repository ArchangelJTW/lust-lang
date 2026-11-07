use crate::bytecode::{Chunk, Function, Instruction, Register, Value};
use crate::mir::ir::*;
use crate::number::{LustFloat, LustInt};
use alloc::vec::Vec;
use hashbrown::HashMap;

pub struct MirToBytecodeCompiler<'a> {
    func: &'a MirFunction,
    chunk: Chunk,
    scratch_start: usize,
    scratch_count: usize,
    max_scratch: usize,
    block_pcs: HashMap<BlockId, usize>,
    pending_jumps: Vec<(usize, BlockId)>,
}

impl<'a> MirToBytecodeCompiler<'a> {
    pub fn compile(func: &'a MirFunction) -> Function {
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
        };

        compiler.compile_blocks();
        compiler.patch_all_jumps();

        bytecode_fn.chunk = compiler.chunk;
        bytecode_fn.register_count = (func.locals.len() + compiler.max_scratch) as u8;
        bytecode_fn
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

    fn compile_blocks(&mut self) {
        for block in &self.func.blocks {
            let pc = self.chunk.instructions.len();
            self.block_pcs.insert(block.id, pc);

            for stmt in &block.statements {
                self.reset_scratch();
                self.compile_statement(stmt);
            }

            self.reset_scratch();
            if let Some(term) = &block.terminator {
                self.compile_terminator(term);
            }
        }
    }

    fn operand_to_reg(&mut self, op: &Operand, line: usize) -> Register {
        match op {
            Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => loc.0 as Register,
            Operand::Constant(c) => {
                let reg = self.next_scratch();
                self.load_constant_into_reg(c, reg, line);
                reg
            }
            _ => 0,
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

    fn compile_statement(&mut self, stmt: &Statement) {
        match stmt {
            Statement::Assign(Place::Local(dst), rvalue, span) => {
                let dst_reg = dst.0 as Register;
                let line = span.start_line;
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
                            Operand::Constant(c) => {
                                self.load_constant_into_reg(c, dst_reg, line);
                            }
                            _ => {}
                        }
                    }
                    Rvalue::BinaryOp(binop, lhs, rhs) => {
                        let r_lhs = self.operand_to_reg(lhs, line);
                        let r_rhs = self.operand_to_reg(rhs, line);
                        self.compile_binop(*binop, dst_reg, r_lhs, r_rhs, line);
                    }
                    Rvalue::UnaryOp(unop, op) => {
                        let r_op = self.operand_to_reg(op, line);
                        match unop {
                            MirUnaryOp::Neg => {
                                self.chunk.emit(Instruction::NegInt(dst_reg, r_op), line);
                            }
                            MirUnaryOp::Not => {
                                self.chunk.emit(Instruction::Not(dst_reg, r_op), line);
                            }
                            MirUnaryOp::Len => {
                                self.chunk.emit(Instruction::ArrayLen(dst_reg, r_op), line);
                            }
                        }
                    }
                    Rvalue::Aggregate(MirType::Array(_), elements) => {
                        let count = elements.len() as u8;
                        if count == 0 {
                            self.chunk.emit(Instruction::NewArray(dst_reg, 0, 0), line);
                        } else {
                            let scratch_base = (self.scratch_start + self.scratch_count) as Register;
                            self.scratch_count += elements.len();
                            for (i, elem) in elements.iter().enumerate() {
                                let target_reg = scratch_base + i as u8;
                                match elem {
                                    Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => {
                                        self.chunk.emit(Instruction::Move(target_reg, loc.0 as Register), line);
                                    }
                                    Operand::Constant(c) => {
                                        self.load_constant_into_reg(c, target_reg, line);
                                    }
                                    _ => {}
                                }
                            }
                            self.chunk.emit(Instruction::NewArray(dst_reg, scratch_base, count), line);
                        }
                    }
                    _ => {}
                }
            }
            Statement::Assign(Place::Index(base, idx), rvalue, span) => {
                let line = span.start_line;
                let val_reg = match rvalue {
                    Rvalue::Use(op) => self.operand_to_reg(op, line),
                    _ => 0,
                };
                let base_reg = self.place_to_reg(base);
                let idx_reg = idx.0 as Register;
                self.chunk.emit(Instruction::SetIndex(base_reg, idx_reg, val_reg), line);
            }
            Statement::StorageLive(_, _) | Statement::StorageDead(_, _) | Statement::ResourceCheck(_, _, _) => {}
            _ => {}
        }
    }

    fn compile_binop(&mut self, op: MirBinaryOp, dst: Register, lhs: Register, rhs: Register, line: usize) {
        let instr = match op {
            MirBinaryOp::Add => Instruction::AddInt(dst, lhs, rhs),
            MirBinaryOp::Sub => Instruction::SubInt(dst, lhs, rhs),
            MirBinaryOp::Mul => Instruction::MulInt(dst, lhs, rhs),
            MirBinaryOp::Div => Instruction::DivInt(dst, lhs, rhs),
            MirBinaryOp::Mod => Instruction::ModInt(dst, lhs, rhs),
            MirBinaryOp::Eq => Instruction::EqInt(dst, lhs, rhs),
            MirBinaryOp::NotEq => Instruction::NeInt(dst, lhs, rhs),
            MirBinaryOp::Lt => Instruction::LtInt(dst, lhs, rhs),
            MirBinaryOp::LtEq => Instruction::LeInt(dst, lhs, rhs),
            MirBinaryOp::Gt => Instruction::GtInt(dst, lhs, rhs),
            MirBinaryOp::GtEq => Instruction::GeInt(dst, lhs, rhs),
            MirBinaryOp::And => Instruction::And(dst, lhs, rhs),
            MirBinaryOp::Or => Instruction::Or(dst, lhs, rhs),
            _ => Instruction::AddInt(dst, lhs, rhs),
        };
        self.chunk.emit(instr, line);
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
                            Operand::Constant(c) => {
                                self.load_constant_into_reg(c, target_reg, line);
                            }
                            _ => {}
                        }
                    }
                    scratch_base
                } else {
                    0
                };

                let func_reg = self.next_scratch();
                if let Some((obj_name, field_name)) = callee.split_once('.') {
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
}
