use super::{Instruction, Value};
use crate::ast::TypeKind;
use crate::typechecker::FunctionSignature;
use alloc::{format, string::String, vec::Vec};
use hashbrown::HashMap;
#[derive(Debug, Clone)]
pub struct Chunk {
    pub instructions: Vec<Instruction>,
    pub constants: Vec<Value>,
    #[cfg(feature = "std")]
    pub lines: Vec<usize>,
}

impl Chunk {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            constants: Vec::new(),
            #[cfg(feature = "std")]
            lines: Vec::new(),
        }
    }

    pub fn emit(&mut self, instruction: Instruction, line: usize) -> usize {
        let idx = self.instructions.len();
        self.instructions.push(instruction);
        #[cfg(feature = "std")]
        self.lines.push(line);
        #[cfg(not(feature = "std"))]
        let _ = line;
        idx
    }

    pub fn add_constant(&mut self, value: Value) -> u16 {
        if let Some(idx) = self.constants.iter().position(|v| v == &value) {
            return idx as u16;
        }

        let idx = self.constants.len();
        if idx > u16::MAX as usize {
            panic!("Too many constants in chunk (max: {})", u16::MAX);
        }

        self.constants.push(value);
        idx as u16
    }

    pub fn patch_jump(&mut self, jump_idx: usize, target_idx: usize) {
        let offset = (target_idx as isize - jump_idx as isize - 1) as i16;
        match &mut self.instructions[jump_idx] {
            Instruction::Jump(ref mut off) => *off = offset,
            Instruction::JumpIf(_, ref mut off) => *off = offset,
            Instruction::JumpIfNot(_, ref mut off) => *off = offset,
            _ => panic!("Attempted to patch non-jump instruction"),
        }
    }

    pub fn disassemble(&self, name: &str) -> String {
        let mut output = String::new();
        output.push_str(&format!("===== {} =====\n", name));
        output.push_str(&format!("Constants: {}\n", self.constants.len()));
        for (i, constant) in self.constants.iter().enumerate() {
            output.push_str(&format!("  K{}: {}\n", i, constant));
        }

        output.push_str("\nInstructions:\n");
        for (i, instruction) in self.instructions.iter().enumerate() {
            #[cfg(feature = "std")]
            let line = self.lines.get(i).copied().unwrap_or(0);
            #[cfg(not(feature = "std"))]
            let line = 0usize;
            output.push_str(&format!("{:04} [L{:03}] {}\n", i, line, instruction));
        }

        output
    }
}

impl Default for Chunk {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub param_count: u8,
    pub register_count: u8,
    pub is_method: bool,
    pub chunk: Chunk,
    pub upvalues: Vec<(bool, u8)>,
    pub register_types: HashMap<u8, TypeKind>,
    pub signature: Option<FunctionSignature>,
}

impl Function {
    pub fn new(name: impl Into<String>, param_count: u8, is_method: bool) -> Self {
        Self {
            name: name.into(),
            param_count,
            register_count: 0,
            is_method,
            chunk: Chunk::new(),
            upvalues: Vec::new(),
            register_types: HashMap::new(),
            signature: None,
        }
    }

    pub fn set_register_count(&mut self, count: u8) {
        self.register_count = count;
    }

    pub fn add_upvalue(&mut self, is_local: bool, index: u8) -> u8 {
        let idx = self.upvalues.len();
        if idx > u8::MAX as usize {
            panic!("Too many upvalues in function (max: 255)");
        }

        self.upvalues.push((is_local, index));
        idx as u8
    }

    pub fn set_signature(&mut self, signature: FunctionSignature) {
        self.signature = Some(signature);
    }

    pub fn disassemble(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!(
            "Function: {} (params: {}, registers: {}, method: {})\n",
            self.name, self.param_count, self.register_count, self.is_method
        ));
        if !self.upvalues.is_empty() {
            output.push_str("Upvalues:\n");
            for (i, (is_local, idx)) in self.upvalues.iter().enumerate() {
                let kind = if *is_local { "local" } else { "upvalue" };
                output.push_str(&format!("  U{}: {} {}\n", i, kind, idx));
            }
        }

        output.push_str(&self.chunk.disassemble(&self.name));
        output
    }
}
