use alloc::collections::BTreeSet;
use hashbrown::HashMap;
use crate::mir::ir::{
    LocalId, MirFunction, MirModule, Operand, Place, Rvalue, Statement, Terminator,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscapeStatus {
    NoEscape,
    EscapesReturn,
    EscapesCall,
    EscapesGlobal,
}

impl EscapeStatus {
    pub fn escapes(&self) -> bool {
        !matches!(self, EscapeStatus::NoEscape)
    }
}

pub struct EscapeAnalysis;

impl EscapeAnalysis {
    /// Computes the escape status of each local in the given MIR function.
    pub fn analyze_function(func: &MirFunction) -> HashMap<LocalId, EscapeStatus> {
        let mut status: HashMap<LocalId, EscapeStatus> = HashMap::new();

        // Parameters escape or are externally owned.
        for param in &func.params {
            status.insert(*param, EscapeStatus::EscapesCall);
        }

        // Return local escapes via return.
        status.insert(func.return_local, EscapeStatus::EscapesReturn);

        // Alias tracking: if local A is assigned from local B or shares storage, they share escape status.
        let mut aliases: HashMap<LocalId, BTreeSet<LocalId>> = HashMap::new();

        for block in &func.blocks {
            for stmt in &block.statements {
                if let Statement::Assign(lhs, rhs, _) = stmt {
                    if let Some(lhs_loc) = Self::place_root_local(lhs) {
                        match lhs {
                            Place::Local(_) => {
                                match rhs {
                                    Rvalue::Use(op) | Rvalue::Cast(op, _) => {
                                        if let Some(rhs_loc) = Self::operand_root_local(op) {
                                            Self::add_alias(&mut aliases, lhs_loc, rhs_loc);
                                        }
                                    }
                                    Rvalue::Aggregate(_, operands) => {
                                        for op in operands {
                                            if let Some(elem_loc) = Self::operand_root_local(op) {
                                                Self::add_alias(&mut aliases, lhs_loc, elem_loc);
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            Place::Field(base, _) | Place::Index(base, _) => {
                                if let Some(base_loc) = Self::place_root_local(base) {
                                    match rhs {
                                        Rvalue::Use(op) | Rvalue::Cast(op, _) => {
                                            if let Some(rhs_loc) = Self::operand_root_local(op) {
                                                Self::add_alias(&mut aliases, base_loc, rhs_loc);
                                            }
                                        }
                                        Rvalue::Aggregate(_, operands) => {
                                            for op in operands {
                                                if let Some(elem_loc) = Self::operand_root_local(op) {
                                                    Self::add_alias(&mut aliases, base_loc, elem_loc);
                                                }
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if let Some(ref term) = block.terminator {
                match term {
                    Terminator::Return(Some(op), _) => {
                        Self::mark_operand_escape(op, EscapeStatus::EscapesReturn, &mut status);
                    }
                    Terminator::Raise(op, _) => {
                        Self::mark_operand_escape(op, EscapeStatus::EscapesReturn, &mut status);
                    }
                    Terminator::Call { callee, args, .. } => {
                        let escape_kind = if callee.starts_with("global") || callee.contains("global") {
                            EscapeStatus::EscapesGlobal
                        } else {
                            EscapeStatus::EscapesCall
                        };
                        for arg in args {
                            Self::mark_operand_escape(arg, escape_kind, &mut status);
                        }
                    }
                    _ => {}
                }
            }
        }

        // Propagate escape status through aliases until fixed-point
        let mut changed = true;
        while changed {
            changed = false;
            for (loc, connected) in &aliases {
                if let Some(&st) = status.get(loc) {
                    if st.escapes() {
                        for other in connected {
                            let curr = status.get(other).copied().unwrap_or(EscapeStatus::NoEscape);
                            if !curr.escapes() {
                                status.insert(*other, st);
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        // Fill in remaining locals as NoEscape
        for local in &func.locals {
            status.entry(local.id).or_insert(EscapeStatus::NoEscape);
        }

        status
    }

    /// Runs escape analysis on `func` and marks `is_stack_promoted`:
    /// Non-escaping locals (`EscapeStatus::NoEscape`) are marked `is_stack_promoted = true`,
    /// while escaping aliases/calls/returns/globals stay `false`.
    pub fn run_on_function(func: &mut MirFunction) -> bool {
        let status = Self::analyze_function(func);
        let mut changed = false;
        for local in &mut func.locals {
            let is_promoted = matches!(status.get(&local.id), Some(EscapeStatus::NoEscape));
            if local.is_stack_promoted != is_promoted {
                local.is_stack_promoted = is_promoted;
                changed = true;
            }
        }
        changed
    }

    /// Runs escape analysis across all functions in `module`.
    pub fn run_on_module(module: &mut MirModule) -> bool {
        let mut changed = false;
        for func in &mut module.functions {
            changed |= Self::run_on_function(func);
        }
        changed
    }

    fn place_root_local(place: &Place) -> Option<LocalId> {
        match place {
            Place::Local(loc) => Some(*loc),
            Place::Field(base, _) | Place::Index(base, _) => Self::place_root_local(base),
        }
    }

    fn operand_root_local(op: &Operand) -> Option<LocalId> {
        match op {
            Operand::Copy(p) | Operand::Move(p) => Self::place_root_local(p),
            Operand::Constant(_) => None,
        }
    }

    fn add_alias(aliases: &mut HashMap<LocalId, BTreeSet<LocalId>>, a: LocalId, b: LocalId) {
        if a != b {
            aliases.entry(a).or_default().insert(b);
            aliases.entry(b).or_default().insert(a);
        }
    }

    fn mark_operand_escape(
        op: &Operand,
        escape_type: EscapeStatus,
        status: &mut HashMap<LocalId, EscapeStatus>,
    ) {
        if let Some(loc) = Self::operand_root_local(op) {
            status.insert(loc, escape_type);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Span;
    use crate::mir::ir::{FunctionSignature, MirConstant, MirType};

    #[test]
    fn test_escape_analysis_local_does_not_escape() {
        let sig = FunctionSignature {
            name: "test_fn".into(),
            param_types: vec![],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };

        let mut func = MirFunction::new(sig, Span::dummy());
        let non_escaping = func.add_local(Some("local_arr".into()), MirType::Array(alloc::boxed::Box::new(MirType::Int)), true, Span::dummy());

        let bb0 = &mut func.blocks[0];
        bb0.statements.push(Statement::Assign(
            Place::Local(non_escaping),
            Rvalue::Aggregate(MirType::Array(alloc::boxed::Box::new(MirType::Int)), vec![Operand::Constant(MirConstant::Int(10))]),
            Span::dummy(),
        ));
        bb0.statements.push(Statement::Assign(
            Place::Local(func.return_local),
            Rvalue::Use(Operand::Constant(MirConstant::Int(42))),
            Span::dummy(),
        ));
        bb0.terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(func.return_local))),
            Span::dummy(),
        ));

        let results = EscapeAnalysis::analyze_function(&func);
        assert_eq!(results.get(&non_escaping), Some(&EscapeStatus::NoEscape));
        assert_eq!(results.get(&func.return_local), Some(&EscapeStatus::EscapesReturn));

        // Test stack promotion pass
        let changed = EscapeAnalysis::run_on_function(&mut func);
        assert!(changed);
        assert!(func.locals[non_escaping.0].is_stack_promoted);
        assert!(!func.locals[func.return_local.0].is_stack_promoted);
    }

    #[test]
    fn test_escape_analysis_alias_and_call_propagation() {
        let sig = FunctionSignature {
            name: "test_alias".into(),
            param_types: vec![MirType::Int],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        };

        let mut func = MirFunction::new(sig, Span::dummy());
        let arg0 = func.params[0];
        let loc_a = func.add_local(Some("a".into()), MirType::Int, true, Span::dummy());
        let loc_b = func.add_local(Some("b".into()), MirType::Int, true, Span::dummy());
        let loc_pure = func.add_local(Some("pure".into()), MirType::Int, true, Span::dummy());

        let bb0 = &mut func.blocks[0];
        // a = arg0 (aliases param -> escapes call)
        bb0.statements.push(Statement::Assign(
            Place::Local(loc_a),
            Rvalue::Use(Operand::Copy(Place::Local(arg0))),
            Span::dummy(),
        ));
        // b = a (aliases a -> escapes call)
        bb0.statements.push(Statement::Assign(
            Place::Local(loc_b),
            Rvalue::Use(Operand::Copy(Place::Local(loc_a))),
            Span::dummy(),
        ));
        // pure = 123 (does not escape)
        bb0.statements.push(Statement::Assign(
            Place::Local(loc_pure),
            Rvalue::Use(Operand::Constant(MirConstant::Int(123))),
            Span::dummy(),
        ));
        bb0.terminator = Some(Terminator::Return(
            Some(Operand::Copy(Place::Local(func.return_local))),
            Span::dummy(),
        ));

        EscapeAnalysis::run_on_function(&mut func);
        assert!(!func.locals[arg0.0].is_stack_promoted, "param must not be stack promoted");
        assert!(!func.locals[loc_a.0].is_stack_promoted, "alias of param must not be stack promoted");
        assert!(!func.locals[loc_b.0].is_stack_promoted, "transitive alias of param must not be stack promoted");
        assert!(func.locals[loc_pure.0].is_stack_promoted, "pure local must be stack promoted");
    }
}
