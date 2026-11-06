use alloc::collections::BTreeSet;
use hashbrown::HashMap;
use crate::mir::ir::{
    LocalId, MirFunction, Operand, Place, Rvalue, Statement, Terminator,
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

        // Alias tracking: if local A is assigned from local B, they share escape status.
        let mut aliases: HashMap<LocalId, BTreeSet<LocalId>> = HashMap::new();

        for block in &func.blocks {
            for stmt in &block.statements {
                if let Statement::Assign(lhs, rhs, _) = stmt {
                    if let Place::Local(lhs_loc) = lhs {
                        match rhs {
                            Rvalue::Use(Operand::Copy(Place::Local(rhs_loc)))
                            | Rvalue::Use(Operand::Move(Place::Local(rhs_loc))) => {
                                aliases.entry(*lhs_loc).or_default().insert(*rhs_loc);
                                aliases.entry(*rhs_loc).or_default().insert(*lhs_loc);
                            }
                            _ => {}
                        }
                    }
                }
            }

            if let Some(ref term) = block.terminator {
                match term {
                    Terminator::Return(Some(op), _) => {
                        Self::mark_operand_escape(op, EscapeStatus::EscapesReturn, &mut status);
                    }
                    Terminator::Call { args, .. } => {
                        for arg in args {
                            Self::mark_operand_escape(arg, EscapeStatus::EscapesCall, &mut status);
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

    fn mark_operand_escape(
        op: &Operand,
        escape_type: EscapeStatus,
        status: &mut HashMap<LocalId, EscapeStatus>,
    ) {
        match op {
            Operand::Copy(Place::Local(loc)) | Operand::Move(Place::Local(loc)) => {
                status.insert(*loc, escape_type);
            }
            _ => {}
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
        };

        let mut func = MirFunction::new(sig, Span::dummy());
        let non_escaping = func.add_local(Some("local_arr".into()), MirType::Array(Box::new(MirType::Int)), true, Span::dummy());

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
    }
}
