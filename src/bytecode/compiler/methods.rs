use super::*;
impl Compiler {
    pub(super) fn compile_map_method(
        &mut self,
        array_expr: &Expr,
        lambda_expr: &Expr,
    ) -> Result<Register> {
        let array_reg = self.compile_expr(array_expr)?;
        let lambda_reg = self.compile_expr(lambda_expr)?;
        let result_reg = self.allocate_register();
        let dummy_elem_reg = self.allocate_register();
        self.emit(Instruction::NewArray(result_reg, dummy_elem_reg, 0), 0);
        let len_reg = self.allocate_register();
        self.emit(Instruction::ArrayLen(len_reg, array_reg), 0);
        let i_reg = self.allocate_register();
        let zero_const_idx = self.add_int_const(0);
        self.emit(Instruction::LoadConst(i_reg, zero_const_idx), 0);
        let end_reg = self.allocate_register();
        let one_const_idx = self.add_int_const(1);
        let one_reg = self.allocate_register();
        self.emit(Instruction::LoadConst(one_reg, one_const_idx), 0);
        self.emit(Instruction::Sub(end_reg, len_reg, one_reg), 0);
        let loop_watermark = self.next_register;
        let loop_start = self.current_chunk().instructions.len();
        let cond_reg = self.allocate_register();
        self.emit(Instruction::Le(cond_reg, i_reg, end_reg), 0);
        let jump_to_end = self.emit(Instruction::JumpIfNot(cond_reg, 0), 0);
        let elem_reg = self.allocate_register();
        self.emit(Instruction::GetIndex(elem_reg, array_reg, i_reg), 0);
        let mapped_reg = self.allocate_register();
        self.emit(Instruction::Call(lambda_reg, elem_reg, 1, mapped_reg), 0);
        let push_method_idx = self.add_string_constant("push");
        let push_result_reg = self.allocate_register();
        self.emit(
            Instruction::CallMethod(result_reg, push_method_idx, mapped_reg, 1, push_result_reg),
            0,
        );
        let inc_reg = self.allocate_register();
        let one_reg2 = self.allocate_register();
        self.emit(Instruction::LoadConst(one_reg2, one_const_idx), 0);
        self.emit(Instruction::Add(inc_reg, i_reg, one_reg2), 0);
        self.emit(Instruction::Move(i_reg, inc_reg), 0);
        self.next_register = loop_watermark;
        self.emit_jump_back_to(loop_start);
        let end_pos = self.current_chunk().instructions.len();
        self.current_chunk_mut().patch_jump(jump_to_end, end_pos);
        Ok(result_reg)
    }

    pub(super) fn compile_filter_method(
        &mut self,
        array_expr: &Expr,
        predicate_expr: &Expr,
    ) -> Result<Register> {
        let array_reg = self.compile_expr(array_expr)?;
        let predicate_reg = self.compile_expr(predicate_expr)?;
        let result_reg = self.allocate_register();
        let dummy_elem_reg = self.allocate_register();
        self.emit(Instruction::NewArray(result_reg, dummy_elem_reg, 0), 0);
        let len_reg = self.allocate_register();
        self.emit(Instruction::ArrayLen(len_reg, array_reg), 0);
        let i_reg = self.allocate_register();
        let zero_const_idx = self.add_int_const(0);
        self.emit(Instruction::LoadConst(i_reg, zero_const_idx), 0);
        let end_reg = self.allocate_register();
        let one_const_idx = self.add_int_const(1);
        let one_reg = self.allocate_register();
        self.emit(Instruction::LoadConst(one_reg, one_const_idx), 0);
        self.emit(Instruction::Sub(end_reg, len_reg, one_reg), 0);
        let loop_watermark = self.next_register;
        let loop_start = self.current_chunk().instructions.len();
        let cond_reg = self.allocate_register();
        self.emit(Instruction::Le(cond_reg, i_reg, end_reg), 0);
        let jump_to_end = self.emit(Instruction::JumpIfNot(cond_reg, 0), 0);
        let elem_reg = self.allocate_register();
        self.emit(Instruction::GetIndex(elem_reg, array_reg, i_reg), 0);
        let pred_result_reg = self.allocate_register();
        self.emit(
            Instruction::Call(predicate_reg, elem_reg, 1, pred_result_reg),
            0,
        );
        let jump_to_next_iter = self.emit(Instruction::JumpIfNot(pred_result_reg, 0), 0);
        let push_method_idx = self.add_string_constant("push");
        let push_result_reg = self.allocate_register();
        self.emit(
            Instruction::CallMethod(result_reg, push_method_idx, elem_reg, 1, push_result_reg),
            0,
        );
        let next_iter_pos = self.current_chunk().instructions.len();
        self.current_chunk_mut()
            .patch_jump(jump_to_next_iter, next_iter_pos);
        let inc_reg = self.allocate_register();
        let one_reg2 = self.allocate_register();
        self.emit(Instruction::LoadConst(one_reg2, one_const_idx), 0);
        self.emit(Instruction::Add(inc_reg, i_reg, one_reg2), 0);
        self.emit(Instruction::Move(i_reg, inc_reg), 0);
        self.next_register = loop_watermark;
        self.emit_jump_back_to(loop_start);
        let end_pos = self.current_chunk().instructions.len();
        self.current_chunk_mut().patch_jump(jump_to_end, end_pos);
        Ok(result_reg)
    }

    pub(super) fn compile_reduce_method(
        &mut self,
        array_expr: &Expr,
        init_expr: &Expr,
        reducer_expr: &Expr,
    ) -> Result<Register> {
        let array_reg = self.compile_expr(array_expr)?;
        let acc_reg = self.compile_expr(init_expr)?;
        let reducer_reg = self.compile_expr(reducer_expr)?;
        let len_reg = self.allocate_register();
        self.emit(Instruction::ArrayLen(len_reg, array_reg), 0);
        let i_reg = self.allocate_register();
        let zero_const_idx = self.add_int_const(0);
        self.emit(Instruction::LoadConst(i_reg, zero_const_idx), 0);
        let end_reg = self.allocate_register();
        let one_const_idx = self.add_int_const(1);
        let one_reg = self.allocate_register();
        self.emit(Instruction::LoadConst(one_reg, one_const_idx), 0);
        self.emit(Instruction::Sub(end_reg, len_reg, one_reg), 0);
        let loop_watermark = self.next_register;
        let loop_start = self.current_chunk().instructions.len();
        let cond_reg = self.allocate_register();
        self.emit(Instruction::Le(cond_reg, i_reg, end_reg), 0);
        let jump_to_end = self.emit(Instruction::JumpIfNot(cond_reg, 0), 0);
        let elem_reg = self.allocate_register();
        self.emit(Instruction::GetIndex(elem_reg, array_reg, i_reg), 0);
        let arg_base = self.next_register;
        let arg1_reg = self.allocate_register();
        let arg2_reg = self.allocate_register();
        if acc_reg != arg1_reg {
            self.emit(Instruction::Move(arg1_reg, acc_reg), 0);
        }

        if elem_reg != arg2_reg {
            self.emit(Instruction::Move(arg2_reg, elem_reg), 0);
        }

        let new_acc_reg = self.allocate_register();
        self.emit(Instruction::Call(reducer_reg, arg_base, 2, new_acc_reg), 0);
        self.emit(Instruction::Move(acc_reg, new_acc_reg), 0);
        let inc_reg = self.allocate_register();
        let one_reg2 = self.allocate_register();
        self.emit(Instruction::LoadConst(one_reg2, one_const_idx), 0);
        self.emit(Instruction::Add(inc_reg, i_reg, one_reg2), 0);
        self.emit(Instruction::Move(i_reg, inc_reg), 0);
        self.next_register = loop_watermark;
        self.emit_jump_back_to(loop_start);
        let end_pos = self.current_chunk().instructions.len();
        self.current_chunk_mut().patch_jump(jump_to_end, end_pos);
        Ok(acc_reg)
    }
}
