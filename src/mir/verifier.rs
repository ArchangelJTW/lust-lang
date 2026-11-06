use super::ir::*;
use crate::ast::Span;
use alloc::{string::String, vec::Vec};
use core::fmt;
use hashbrown::HashSet;

#[derive(Debug, Clone, PartialEq)]
pub struct MirVerificationError {
    pub message: String,
    pub function: String,
    pub block: Option<BlockId>,
    pub span: Option<Span>,
}

impl fmt::Display for MirVerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MIR verification error in function '{}'", self.function)?;
        if let Some(b) = self.block {
            write!(f, " at {b}")?;
        }
        write!(f, ": {}", self.message)
    }
}

pub struct MirVerifier<'a> {
    func: &'a MirFunction,
    errors: Vec<MirVerificationError>,
}

impl<'a> MirVerifier<'a> {
    pub fn verify_module(module: &MirModule) -> Result<(), Vec<MirVerificationError>> {
        let mut all_errors = Vec::new();
        for func in &module.functions {
            let mut verifier = MirVerifier {
                func,
                errors: Vec::new(),
            };
            verifier.verify_function();
            all_errors.extend(verifier.errors);
        }

        if all_errors.is_empty() {
            Ok(())
        } else {
            Err(all_errors)
        }
    }

    pub fn verify_fn(func: &MirFunction) -> Result<(), Vec<MirVerificationError>> {
        let mut verifier = MirVerifier {
            func,
            errors: Vec::new(),
        };
        verifier.verify_function();
        if verifier.errors.is_empty() {
            Ok(())
        } else {
            Err(verifier.errors)
        }
    }

    pub fn verify_function(&mut self) {
        // 1. Entry block exists
        if self.func.blocks.is_empty() {
            self.error("Function has no basic blocks", None, None);
            return;
        }

        if self.func.entry.0 >= self.func.blocks.len() {
            self.error("Entry block ID out of bounds", Some(self.func.entry), None);
        }

        // 2. Validate every block has a terminator
        for (idx, block) in self.func.blocks.iter().enumerate() {
            let block_id = BlockId(idx);
            if block.terminator.is_none() {
                self.error("Basic block missing terminator", Some(block_id), None);
            }
        }

        // 3. Validate control flow targets, places, types, and def-before-use
        self.verify_def_before_use();

        for block in &self.func.blocks {
            for stmt in &block.statements {
                self.verify_statement(stmt, block.id);
            }
            if let Some(term) = &block.terminator {
                self.verify_terminator(term, block.id);
            }
        }
    }

    fn verify_def_before_use(&mut self) {
        let mut block_defined: Vec<Option<HashSet<LocalId>>> = vec![None; self.func.blocks.len()];
        let mut entry_inits = HashSet::new();
        for p in &self.func.params {
            entry_inits.insert(*p);
        }
        block_defined[self.func.entry.0] = Some(entry_inits);

        let mut changed = true;
        let mut iterations = 0;
        let max_iterations = self.func.blocks.len() * 3 + 10;

        while changed && iterations < max_iterations {
            changed = false;
            iterations += 1;

            for block in &self.func.blocks {
                let current_inits = match &block_defined[block.id.0] {
                    Some(inits) => inits.clone(),
                    None => continue,
                };

                let mut out_inits = current_inits;
                for stmt in &block.statements {
                    match stmt {
                        Statement::Assign(place, _, _) => {
                            if let Place::Local(loc) = place {
                                out_inits.insert(*loc);
                            }
                        }
                        Statement::StorageDead(loc, _) => {
                            out_inits.remove(loc);
                        }
                        _ => {}
                    }
                }

                if let Some(term) = &block.terminator {
                    match term {
                        Terminator::Cond { then_block, else_block, .. } => {
                            changed |= Self::propagate(&mut block_defined, *then_block, &out_inits);
                            changed |= Self::propagate(&mut block_defined, *else_block, &out_inits);
                        }
                        Terminator::Goto(target, _) => {
                            changed |= Self::propagate(&mut block_defined, *target, &out_inits);
                        }
                        Terminator::Switch { cases, default, .. } => {
                            for (_, target) in cases {
                                changed |= Self::propagate(&mut block_defined, *target, &out_inits);
                            }
                            changed |= Self::propagate(&mut block_defined, *default, &out_inits);
                        }
                        Terminator::Call { destination, success, error, .. } => {
                            let mut succ_inits = out_inits.clone();
                            if let Place::Local(loc) = destination {
                                succ_inits.insert(*loc);
                            }
                            changed |= Self::propagate(&mut block_defined, *success, &succ_inits);
                            if let Some(err_block) = error {
                                changed |= Self::propagate(&mut block_defined, *err_block, &out_inits);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Verification pass: check reads in all reachable blocks
        for block in &self.func.blocks {
            let mut current_inits = match &block_defined[block.id.0] {
                Some(inits) => inits.clone(),
                None => continue,
            };

            for stmt in &block.statements {
                match stmt {
                    Statement::Assign(place, rvalue, span) => {
                        self.check_rvalue_reads(rvalue, &current_inits, block.id, *span);
                        if let Place::Local(loc) = place {
                            current_inits.insert(*loc);
                        }
                    }
                    Statement::StorageDead(loc, _) => {
                        current_inits.remove(loc);
                    }
                    _ => {}
                }
            }

            if let Some(term) = &block.terminator {
                match term {
                    Terminator::Cond { cond, span, .. } => {
                        self.check_operand_read(cond, &current_inits, block.id, *span);
                    }
                    Terminator::Switch { discr, span, .. } => {
                        self.check_operand_read(discr, &current_inits, block.id, *span);
                    }
                    Terminator::Call { args, span, .. } => {
                        for arg in args {
                            self.check_operand_read(arg, &current_inits, block.id, *span);
                        }
                    }
                    Terminator::Return(opt_op, span) => {
                        if let Some(op) = opt_op {
                            self.check_operand_read(op, &current_inits, block.id, *span);
                        }
                    }
                    Terminator::Raise(op, span) => {
                        self.check_operand_read(op, &current_inits, block.id, *span);
                    }
                    _ => {}
                }
            }
        }
    }

    fn propagate(
        block_defined: &mut [Option<HashSet<LocalId>>],
        target: BlockId,
        inits: &HashSet<LocalId>,
    ) -> bool {
        if target.0 >= block_defined.len() {
            return false;
        }
        match &mut block_defined[target.0] {
            None => {
                block_defined[target.0] = Some(inits.clone());
                true
            }
            Some(existing) => {
                let old_len = existing.len();
                existing.retain(|id| inits.contains(id));
                existing.len() != old_len
            }
        }
    }

    fn check_rvalue_reads(
        &mut self,
        rvalue: &Rvalue,
        inits: &HashSet<LocalId>,
        block: BlockId,
        span: Span,
    ) {
        match rvalue {
            Rvalue::Use(op) => self.check_operand_read(op, inits, block, span),
            Rvalue::BinaryOp(_, lhs, rhs) => {
                self.check_operand_read(lhs, inits, block, span);
                self.check_operand_read(rhs, inits, block, span);
            }
            Rvalue::UnaryOp(_, op) => self.check_operand_read(op, inits, block, span),
            Rvalue::Cast(op, _) => self.check_operand_read(op, inits, block, span),
            Rvalue::Aggregate(_, operands) => {
                for op in operands {
                    self.check_operand_read(op, inits, block, span);
                }
            }
        }
    }

    fn check_operand_read(
        &mut self,
        op: &Operand,
        inits: &HashSet<LocalId>,
        block: BlockId,
        span: Span,
    ) {
        match op {
            Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => {
                if !inits.contains(loc) {
                    self.error(
                        alloc::format!("Use of uninitialized local {loc}"),
                        Some(block),
                        Some(span),
                    );
                }
            }
            _ => {}
        }
    }

    fn verify_statement(&mut self, stmt: &Statement, block: BlockId) {
        match stmt {
            Statement::Assign(place, _rvalue, span) => {
                self.verify_place(place, block, *span);
            }
            Statement::StorageLive(loc, span) | Statement::StorageDead(loc, span) => {
                if loc.0 >= self.func.locals.len() {
                    self.error(alloc::format!("Local {loc} out of bounds"), Some(block), Some(*span));
                }
            }
            Statement::ResourceCheck(_, _, _) => {}
        }
    }

    fn verify_place(&mut self, place: &Place, block: BlockId, span: Span) {
        match place {
            Place::Local(loc) => {
                if loc.0 >= self.func.locals.len() {
                    self.error(alloc::format!("Place local {loc} out of bounds"), Some(block), Some(span));
                }
            }
            Place::Field(base, _) => self.verify_place(base, block, span),
            Place::Index(base, idx) => {
                self.verify_place(base, block, span);
                if idx.0 >= self.func.locals.len() {
                    self.error(alloc::format!("Index local {idx} out of bounds"), Some(block), Some(span));
                }
            }
        }
    }

    fn verify_terminator(&mut self, term: &Terminator, block: BlockId) {
        match term {
            Terminator::Goto(target, span) => {
                self.check_block_target(*target, block, *span);
            }
            Terminator::Cond { cond: _, then_block, else_block, span } => {
                self.check_block_target(*then_block, block, *span);
                self.check_block_target(*else_block, block, *span);
            }
            Terminator::Switch { cases, default, span, .. } => {
                let mut seen_cases = HashSet::new();
                for (val, target) in cases {
                    if !seen_cases.insert(*val) {
                        self.error(
                            alloc::format!("Duplicate case value {val} in switch terminator"),
                            Some(block),
                            Some(*span),
                        );
                    }
                    self.check_block_target(*target, block, *span);
                }
                self.check_block_target(*default, block, *span);
            }
            Terminator::Call { destination, success, error, span, .. } => {
                self.verify_place(destination, block, *span);
                self.check_block_target(*success, block, *span);
                if let Some(err) = error {
                    self.check_block_target(*err, block, *span);
                }
            }
            Terminator::Return(_opt_op, _span) => {
                // Verify return type matches signature
            }
            Terminator::Raise(_, _) => {}
            Terminator::Unreachable(_) => {}
        }
    }

    fn check_block_target(&mut self, target: BlockId, from: BlockId, span: Span) {
        if target.0 >= self.func.blocks.len() {
            self.error(
                alloc::format!("Target block {target} out of bounds (from {from})"),
                Some(from),
                Some(span),
            );
        }
    }

    fn error(&mut self, message: impl Into<String>, block: Option<BlockId>, span: Option<Span>) {
        self.errors.push(MirVerificationError {
            message: message.into(),
            function: self.func.signature.name.clone(),
            block,
            span,
        });
    }
}
