use super::instruction::PackedInstruction;
use super::{EnumObject, Instruction, OpCode, StructLayout, Value};
use crate::ast::TypeKind;
use crate::bytecode::value::Name;
use crate::typechecker::FunctionSignature;
use alloc::{boxed::Box, format, rc::Rc, string::String, vec::Vec};
use core::cell::RefCell;
use hashbrown::HashMap;

#[derive(Debug, Clone, Default)]
pub enum ChunkIcSlot {
    #[default]
    Empty,
    FieldSlot {
        layout_ptr: usize,
        layout: Rc<StructLayout>,
        slot_idx: usize,
    },
    NewStructTemplate {
        struct_meta_gen: u64,
        struct_name: Name,
        layout: Rc<StructLayout>,
        slot_map: Box<[usize]>,
        is_identity: bool,
    },
    EnumUnit {
        object: Rc<EnumObject>,
    },
    EnumVariantNames {
        enum_name: Name,
        variant_name: Name,
    },
    MethodCall {
        layout_ptr: usize,
        layout: Rc<StructLayout>,
        target_func_idx: usize,
    },
    GlobalSlot {
        generation: u64,
        slot_idx: u32,
    },
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub instructions: Vec<Instruction>,
    /// Compact 4-byte packed instruction stream aligned 1:1 with `instructions`.
    pub packed_code: Vec<PackedInstruction>,
    /// Extension operand words referenced by `AX`-format `PackedInstruction`s.
    pub extra_args: Vec<u32>,
    pub constants: Vec<Value>,
    #[cfg(feature = "std")]
    pub lines: Vec<usize>,
    pub ic_slots: RefCell<Vec<ChunkIcSlot>>,
    /// Basic-block gas weights aligned 1:1 with `instructions`.
    /// Each basic-block leader holds the sum of `inst.gas_weight()` across the
    /// block; interior instructions hold `0`.
    pub bb_gas_weights: Vec<u16>,
}

impl Chunk {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            packed_code: Vec::new(),
            extra_args: Vec::new(),
            constants: Vec::new(),
            #[cfg(feature = "std")]
            lines: Vec::new(),
            ic_slots: RefCell::new(Vec::new()),
            bb_gas_weights: Vec::new(),
        }
    }

    pub fn emit(&mut self, instruction: Instruction, line: usize) -> usize {
        let idx = self.instructions.len();
        self.instructions.push(instruction);
        if self.packed_code.len() == idx {
            if let Some(packed) =
                PackedInstruction::from_instruction(instruction, &mut self.extra_args)
            {
                self.packed_code.push(packed);
            } else {
                self.packed_code.clear();
                self.extra_args.clear();
            }
        }
        #[cfg(feature = "std")]
        self.lines.push(line);
        #[cfg(not(feature = "std"))]
        let _ = line;
        idx
    }

    /// Replaces the instruction at `idx` and synchronously updates `packed_code[idx]`.
    pub fn set_instruction(&mut self, idx: usize, instruction: Instruction) {
        self.instructions[idx] = instruction;
        if self.packed_code.len() == self.instructions.len() {
            if let Some(packed) =
                PackedInstruction::from_instruction(instruction, &mut self.extra_args)
            {
                self.packed_code[idx] = packed;
            } else {
                self.packed_code.clear();
                self.extra_args.clear();
            }
        }
    }

    /// Pops the last instruction (and matching line / packed entry when synchronized).
    pub fn pop_instruction(&mut self) -> Option<Instruction> {
        let was_synced = self.packed_code.len() == self.instructions.len();
        let popped = self.instructions.pop()?;
        if was_synced {
            let _ = self.packed_code.pop();
        }
        #[cfg(feature = "std")]
        let _ = self.lines.pop();
        Some(popped)
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
            Instruction::Jump(off) => {
                *off = offset;
                if let Some(slot) = self.packed_code.get_mut(jump_idx) {
                    *slot = PackedInstruction::encode_asbx(OpCode::Jump, 0, offset);
                }
            }
            Instruction::JumpIf(reg, off) => {
                let reg = *reg;
                *off = offset;
                if let Some(slot) = self.packed_code.get_mut(jump_idx) {
                    *slot = PackedInstruction::encode_asbx(OpCode::JumpIf, reg, offset);
                }
            }
            Instruction::JumpIfNot(reg, off) => {
                let reg = *reg;
                *off = offset;
                if let Some(slot) = self.packed_code.get_mut(jump_idx) {
                    *slot = PackedInstruction::encode_asbx(OpCode::JumpIfNot, reg, offset);
                }
            }
            Instruction::BranchLtInt(lhs, rhs, off) => {
                let (lhs, rhs) = (*lhs, *rhs);
                *off = offset;
                if let Some(&packed) = self.packed_code.get(jump_idx)
                    && packed.opcode_u8() == OpCode::BranchLtInt as u8
                    && packed.a() == lhs
                {
                    let extra_idx = packed.bx() as usize;
                    if let Some(word) = self.extra_args.get_mut(extra_idx) {
                        *word = u32::from(rhs) | (u32::from(offset as u16) << 16);
                    }
                }
            }
            Instruction::BranchLeInt(lhs, rhs, off) => {
                let (lhs, rhs) = (*lhs, *rhs);
                *off = offset;
                if let Some(&packed) = self.packed_code.get(jump_idx)
                    && packed.opcode_u8() == OpCode::BranchLeInt as u8
                    && packed.a() == lhs
                {
                    let extra_idx = packed.bx() as usize;
                    if let Some(word) = self.extra_args.get_mut(extra_idx) {
                        *word = u32::from(rhs) | (u32::from(offset as u16) << 16);
                    }
                }
            }
            _ => panic!("Attempted to patch non-jump instruction"),
        }
    }

    /// Rebuilds `packed_code` and `extra_args` directly from `self.instructions`,
    /// guaranteeing strict 1:1 PC correspondence (`packed_code.len() == instructions.len()`)
    /// and compact `extra_args` with no dead entries left behind by peephole rewrites.
    ///
    /// If a pathological chunk exceeds `u16::MAX` extension words, leaves `packed_code`
    /// empty so the VM cleanly falls back to `instructions`.
    pub fn build_packed_code(&mut self) {
        let n = self.instructions.len();
        self.packed_code.clear();
        self.extra_args.clear();
        if n == 0 {
            return;
        }
        self.packed_code.reserve(n);
        for &inst in &self.instructions {
            let Some(packed) = PackedInstruction::from_instruction(inst, &mut self.extra_args)
            else {
                self.packed_code.clear();
                self.extra_args.clear();
                return;
            };
            self.packed_code.push(packed);
        }
        debug_assert_eq!(self.packed_code.len(), self.instructions.len());
    }

    /// Computes basic-block gas weights aligned 1:1 with `self.instructions`.
    ///
    /// Basic-block leaders (`bb_leaders`) are:
    /// - `pc = 0`
    /// - any instruction immediately following a branch/call/return terminator
    ///   (`Jump`, `JumpIf`, `JumpIfNot`, `BranchLtInt`, `BranchLeInt`, `Call`,
    ///    `CallMethod`, `Return`)
    /// - any in-bounds jump target `pc + 1 + offset`.
    ///
    /// Each leader stores `sum(inst.gas_weight())` for its basic block; interior
    /// instructions store `0`. If a synthetic straight-line block would exceed
    /// `u16::MAX` total gas weight, it splits cleanly at the overflowing PC.
    pub fn compute_bb_gas_weights(&mut self) {
        let n = self.instructions.len();
        if n == 0 {
            self.bb_gas_weights.clear();
            return;
        }

        let mut is_leader = alloc::vec![false; n];
        is_leader[0] = true;

        for (pc, inst) in self.instructions.iter().enumerate() {
            match *inst {
                Instruction::Jump(offset)
                | Instruction::JumpIf(_, offset)
                | Instruction::JumpIfNot(_, offset)
                | Instruction::BranchLtInt(_, _, offset)
                | Instruction::BranchLeInt(_, _, offset) => {
                    let target = pc as isize + 1 + offset as isize;
                    if target >= 0 && (target as usize) < n {
                        is_leader[target as usize] = true;
                    }
                    if pc + 1 < n {
                        is_leader[pc + 1] = true;
                    }
                }
                Instruction::Call(..) | Instruction::CallMethod(..) | Instruction::Return(_) => {
                    if pc + 1 < n {
                        is_leader[pc + 1] = true;
                    }
                }
                _ => {}
            }
        }

        self.bb_gas_weights.clear();
        self.bb_gas_weights.resize(n, 0);

        let mut leader_pc = 0usize;
        let mut acc: u32 = 0;
        for pc in 0..n {
            let w = u32::from(self.instructions[pc].gas_weight());
            if pc > 0 && is_leader[pc] {
                self.bb_gas_weights[leader_pc] = acc as u16;
                leader_pc = pc;
                acc = w;
            } else if acc + w > u32::from(u16::MAX) {
                self.bb_gas_weights[leader_pc] = acc as u16;
                leader_pc = pc;
                acc = w;
            } else {
                acc += w;
            }
        }
        self.bb_gas_weights[leader_pc] = acc as u16;
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
    /// Only stored when minimal_runtime_types is disabled
    pub register_types: HashMap<u8, TypeKind>,
    pub signature: Option<FunctionSignature>,
    /// True when all parameters and return type are statically typed without
    /// dynamic contracts (`Schema` / `LuaValue` / `Unknown` / `Infer` / `Generic` / `Effect`).
    /// Allows internal bytecode-to-bytecode calls (`Call`/`CallMethod`/`Return`) to
    /// skip redundant runtime shallow-type re-validation.
    pub trusted_signature: bool,
    /// Bytecode PC immediately before each basic block terminator (keyed by block id index)
    pub terminator_pcs: HashMap<usize, usize>,
    /// Authoritative bytecode PC at the entry of each basic block (keyed by block id index)
    pub block_entry_pcs: HashMap<usize, usize>,
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
            trusted_signature: false,
            terminator_pcs: HashMap::new(),
            block_entry_pcs: HashMap::new(),
        }
    }

    /// Creates a new function without register type tracking (saves memory)
    pub fn new_minimal(name: impl Into<String>, param_count: u8, is_method: bool) -> Self {
        Self {
            name: name.into(),
            param_count,
            register_count: 0,
            is_method,
            chunk: Chunk::new(),
            upvalues: Vec::new(),
            register_types: HashMap::new(), // Empty, won't be populated
            signature: None,
            trusted_signature: false,
            terminator_pcs: HashMap::new(),
            block_entry_pcs: HashMap::new(),
        }
    }

    #[inline]
    fn is_dynamic_type_name(name: &str) -> bool {
        matches!(
            name,
            "LuaValue"
                | "LuaTable"
                | "LuaFunction"
                | "LuaUserdata"
                | "LuaThread"
                | "unknown"
                | "Schema"
                | "SchemaBuilder"
        )
    }

    fn is_trusted_static_type(ty: &crate::ast::Type) -> bool {
        match &ty.kind {
            TypeKind::Int
            | TypeKind::Float
            | TypeKind::String
            | TypeKind::Bool
            | TypeKind::Unit
            | TypeKind::Never => true,
            TypeKind::Named(name) => !Self::is_dynamic_type_name(name),
            TypeKind::Array(inner)
            | TypeKind::Option(inner)
            | TypeKind::Ref(inner)
            | TypeKind::MutRef(inner) => Self::is_trusted_static_type(inner),
            TypeKind::Pointer { pointee, .. } => Self::is_trusted_static_type(pointee),
            TypeKind::Map(key, val) | TypeKind::Result(key, val) => {
                Self::is_trusted_static_type(key) && Self::is_trusted_static_type(val)
            }
            TypeKind::Tuple(elems) => elems.iter().all(Self::is_trusted_static_type),
            TypeKind::Function {
                params,
                return_type,
            } => {
                params.iter().all(Self::is_trusted_static_type)
                    && Self::is_trusted_static_type(return_type)
            }
            TypeKind::GenericInstance { name, type_args } => {
                !Self::is_dynamic_type_name(name)
                    && type_args.iter().all(Self::is_trusted_static_type)
            }
            TypeKind::Union(types) => {
                !types.is_empty() && types.iter().all(Self::is_trusted_static_type)
            }
            TypeKind::Unknown
            | TypeKind::Infer
            | TypeKind::Generic(_)
            | TypeKind::Trait(_)
            | TypeKind::TraitBound(_)
            | TypeKind::Effect { .. }
            | TypeKind::Typeof(_)
            | TypeKind::InferType(_)
            | TypeKind::InferInput(_)
            | TypeKind::InferOutput(_)
            | TypeKind::Schema { .. } => false,
        }
    }

    /// Computes whether this function's signature is fully statically typed with no
    /// dynamic contracts (`Schema`, `LuaValue`, `Unknown`, `Infer`, generics, or effects).
    pub fn compute_trusted_signature(&self) -> bool {
        let Some(signature) = &self.signature else {
            return false;
        };
        if !signature.type_params.is_empty()
            || !signature.trait_bounds.is_empty()
            || signature.effects.is_some()
        {
            return false;
        }
        if signature.params.len() != self.param_count as usize {
            return false;
        }
        signature.params.iter().all(Self::is_trusted_static_type)
            && Self::is_trusted_static_type(&signature.return_type)
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
        self.trusted_signature = self.compute_trusted_signature();
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
