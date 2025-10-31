use super::*;
impl Compiler {
    pub(super) fn max_local_register_index(&self) -> i32 {
        let mut max_reg: i32 = -1;
        for sc in &self.scopes {
            for &(reg, _) in sc.locals.values() {
                if (reg as i32) > max_reg {
                    max_reg = reg as i32;
                }
            }
        }

        max_reg
    }

    pub(super) fn next_local_slot(&mut self) -> Register {
        let reg = (self.max_local_register_index() + 1).max(0) as u8;
        if self.next_register <= reg {
            self.next_register = reg + 1;
        }

        if reg > self.max_register {
            self.max_register = reg;
        }

        reg
    }

    pub(super) fn add_int_const(&mut self, n: LustInt) -> u16 {
        self.add_constant(Value::Int(n))
    }

    pub(super) fn emit_jump_back_to(&mut self, target_pos: usize) {
        let jump_pos = self.current_chunk().instructions.len();
        let offset = (target_pos as isize - jump_pos as isize - 1) as i16;
        self.emit(Instruction::Jump(offset), 0);
    }

    pub(super) fn begin_scope(&mut self) {
        let depth = self.scopes.last().map(|s| s.depth + 1).unwrap_or(0);
        self.scopes.push(Scope {
            locals: HashMap::new(),
            depth,
        });
    }

    pub(super) fn end_scope(&mut self) {
        if let Some(_scope) = self.scopes.pop() {}
    }

    pub(super) fn resolve_local(&self, name: &str) -> Result<Register> {
        for scope in self.scopes.iter().rev() {
            if let Some(&(reg, _)) = scope.locals.get(name) {
                return Ok(reg);
            }
        }

        Err(LustError::CompileError(format!(
            "Undefined variable: {}",
            name
        )))
    }

    pub(super) fn allocate_register(&mut self) -> Register {
        let base = (self.max_local_register_index() + 1).max(0) as u8;
        if self.next_register < base {
            self.next_register = base;
        }

        let reg = self.next_register;
        self.next_register = self.next_register.saturating_add(1);
        if reg > self.max_register {
            self.max_register = reg;
        }

        if reg == 255 {
            panic!("Register overflow (max 255 registers per function)");
        }

        reg
    }

    pub(super) fn free_register(&mut self, reg: Register) {
        let base = (self.max_local_register_index() + 1).max(0) as u8;
        if reg >= base {
            if reg + 1 == self.next_register {
                self.next_register = reg;
            } else {
                debug_assert!(
                    reg < self.next_register,
                    "Attempted to free register {} beyond current watermark {}",
                    reg,
                    self.next_register
                );
            }
        }
    }

    pub(super) fn reset_temp_registers(&mut self) {
        let base = (self.max_local_register_index() + 1).max(0) as u8;
        self.next_register = base;
    }

    pub(super) fn add_constant(&mut self, value: Value) -> u16 {
        self.functions[self.current_function]
            .chunk
            .add_constant(value)
    }

    pub(super) fn add_string_constant(&mut self, s: &str) -> u16 {
        self.add_constant(Value::string(s))
    }

    pub(super) fn register_type(&mut self, reg: Register, type_kind: crate::ast::TypeKind) {
        self.functions[self.current_function]
            .register_types
            .insert(reg, type_kind);
    }

    pub(super) fn emit(&mut self, instruction: Instruction, line: usize) -> usize {
        let effective_line = if line == 0 { self.current_line } else { line };
        self.functions[self.current_function]
            .chunk
            .emit(instruction, effective_line)
    }

    pub(super) fn current_chunk(&self) -> &Chunk {
        &self.functions[self.current_function].chunk
    }

    pub(super) fn current_chunk_mut(&mut self) -> &mut Chunk {
        &mut self.functions[self.current_function].chunk
    }
}
