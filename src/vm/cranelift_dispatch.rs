//! Cranelift JIT dispatch, eligibility checks, and compound value marshaling for the Lust VM.

use crate::ast::types::TypeKind;
use crate::bytecode::{LustMap, StructLayout, StructObject, Value};
use crate::error::{LustError, Result};
use crate::mir::cranelift_backend::{
    MIR_STATUS_ERR_DIV_BY_ZERO, MIR_STATUS_ERR_NEG_EXPONENT, MIR_STATUS_ERR_OOM,
    MIR_STATUS_ERR_OUT_OF_GAS, MIR_STATUS_OK, lust_mir_alloc_array,
};
use crate::mir::ir::{MirConstant, MirFunction, MirType, Operand, ResourceKind, Rvalue, Statement, Terminator};
use crate::vm::VM;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;

#[derive(Debug, Clone, Copy)]
pub enum RawArg {
    Int(i64),
    Float(f64),
}

impl VM {
    #[cfg(feature = "std")]
    pub(crate) fn is_cranelift_supported_type(ty: &MirType) -> bool {
        match ty {
            MirType::Int | MirType::Bool | MirType::Unit | MirType::Float | MirType::String | MirType::Unknown => true,
            MirType::Array(inner) => Self::is_cranelift_supported_type(inner),
            MirType::Tuple(elems) => elems.iter().all(Self::is_cranelift_supported_type),
            MirType::Struct(_) => true,
            MirType::Map(k, v) => Self::is_cranelift_supported_type(k) && Self::is_cranelift_supported_type(v),
            MirType::Enum(_) => true,
            MirType::Function { .. } | MirType::Effect { .. } => false,
        }
    }

    #[cfg(feature = "std")]
    pub(crate) fn is_cranelift_eligible_operand(op: &Operand) -> bool {
        match op {
            Operand::Copy(_) | Operand::Move(_) => true,
            Operand::Constant(c) => matches!(
                c,
                MirConstant::Int(_)
                    | MirConstant::Float(_)
                    | MirConstant::Bool(_)
                    | MirConstant::String(_)
            ),
        }
    }

    #[cfg(feature = "std")]
    pub(crate) fn is_cranelift_eligible_rvalue(rval: &Rvalue) -> bool {
        match rval {
            Rvalue::Use(op) | Rvalue::UnaryOp(_, op) => Self::is_cranelift_eligible_operand(op),
            Rvalue::BinaryOp(_, l, r) => {
                Self::is_cranelift_eligible_operand(l) && Self::is_cranelift_eligible_operand(r)
            }
            Rvalue::Aggregate(ty, elems) => {
                Self::is_cranelift_supported_type(ty)
                    && elems.iter().all(Self::is_cranelift_eligible_operand)
            }
            Rvalue::Cast(op, target_ty) => {
                Self::is_cranelift_eligible_operand(op)
                    && Self::is_cranelift_supported_type(target_ty)
            }
        }
    }

    #[cfg(feature = "std")]
    pub(crate) fn is_cranelift_eligible_signature(mir_fn: &MirFunction) -> bool {
        if mir_fn.params.len() > 6
            || !Self::is_cranelift_supported_type(&mir_fn.signature.return_type)
        {
            return false;
        }
        if !mir_fn
            .params
            .iter()
            .all(|pid| Self::is_cranelift_supported_type(&mir_fn.locals[pid.0].ty))
        {
            return false;
        }
        if !mir_fn
            .locals
            .iter()
            .all(|loc| Self::is_cranelift_supported_type(&loc.ty))
        {
            return false;
        }
        for block in &mir_fn.blocks {
            for stmt in &block.statements {
                match stmt {
                    Statement::Assign(_, rval, _) => {
                        if !Self::is_cranelift_eligible_rvalue(rval) {
                            return false;
                        }
                    }
                    Statement::ResourceCheck(ResourceKind::Memory, _, _) => {}
                    Statement::StorageLive(..)
                    | Statement::StorageDead(..)
                    | Statement::ResourceCheck(ResourceKind::Gas, _, _) => {}
                }
            }
            if let Some(term) = &block.terminator {
                match term {
                    Terminator::Goto(..) | Terminator::Unreachable(_) => {}
                    Terminator::Cond { cond, .. } => {
                        if !Self::is_cranelift_eligible_operand(cond) {
                            return false;
                        }
                    }
                    Terminator::Switch { discr, .. } => {
                        if !Self::is_cranelift_eligible_operand(discr) {
                            return false;
                        }
                    }
                    Terminator::Call { args, .. } => {
                        if !args.iter().all(Self::is_cranelift_eligible_operand) {
                            return false;
                        }
                    }
                    Terminator::Return(opt_op, _) => {
                        if let Some(op) = opt_op
                            && !Self::is_cranelift_eligible_operand(op)
                        {
                            return false;
                        }
                    }
                    Terminator::Raise(..) => return false,
                }
            }
        }
        true
    }

    /// Compiles all uncompiled eligible registered MIR functions using CraneliftJitCompiler.
    #[cfg(feature = "std")]
    pub fn compile_cranelift(&mut self) -> Result<()> {
        let _guard = self.mir_context.enter();
        let mut eligible: hashbrown::HashSet<String> = self
            .mir_functions
            .iter()
            .filter(|(_, f)| Self::is_cranelift_eligible_signature(f))
            .map(|(name, _)| name.clone())
            .collect();

        loop {
            let to_remove: Vec<String> = eligible
                .iter()
                .filter(|name| {
                    let Some(mir_fn) = self.mir_functions.get(*name) else {
                        return true;
                    };
                    mir_fn.blocks.iter().any(|block| {
                        let Some(Terminator::Call { callee, args, .. }) = &block.terminator else {
                            return false;
                        };
                        let is_builtin = matches!(
                            callee.as_str(),
                            "print"
                                | "println"
                                | "math.abs"
                                | "abs"
                                | "math.min"
                                | "min"
                                | "math.max"
                                | "max"
                                | "math.clamp"
                                | "clamp"
                                | "table.len"
                                | "array.len"
                                | "string.len"
                                | "map.len"
                                | "map.get"
                                | "map.set"
                                | "map.new"
                                | "enum.tag"
                                | "enum.payload"
                                | "enum.new"
                                | "enum.alloc"
                        );
                        if is_builtin {
                            return false;
                        }
                        if !eligible.contains(callee) {
                            return true;
                        }
                        self.mir_functions
                            .get(callee)
                            .is_none_or(|target_fn| target_fn.params.len() != args.len())
                    })
                })
                .cloned()
                .collect();
            if to_remove.is_empty() {
                break;
            }
            for name in to_remove {
                eligible.remove(&name);
            }
        }

        if eligible.is_empty() {
            return Ok(());
        }

        let all_compiled = self.cranelift_jit.is_some()
            && eligible
                .iter()
                .all(|name| self.cranelift_compiled.contains_key(name));
        if all_compiled {
            return Ok(());
        }

        if self.cranelift_jit.is_none() || !self.cranelift_compiled.is_empty() {
            let compiler = crate::mir::cranelift_backend::jit_engine::CraneliftJitCompiler::new()
                .map_err(|e| LustError::RuntimeError {
                    message: format!("Failed to initialize Cranelift JIT compiler: {}", e),
                })?;
            self.cranelift_jit = Some(compiler);
            self.cranelift_compiled.clear();
        }

        let mut sorted_names: Vec<String> = eligible.into_iter().collect();
        sorted_names.sort();

        for (name, info) in &self.struct_metadata {
            let field_types: Vec<MirType> = (0..info.layout.field_names().len())
                .map(|i| crate::mir::lowering::mir_type_from_ast(info.layout.field_type(i)))
                .collect();
            self.mir_context.register_struct_field_types(name, field_types);
        }

        let mut module = crate::mir::ir::MirModule::new("vm_cranelift_module".into());
        for name in &sorted_names {
            if let Some(f) = self.mir_functions.get(name) {
                module.add_function(f.clone());
            }
        }

        let compiler = self.cranelift_jit.as_mut().unwrap();
        let compiled_map = compiler
            .compile_module(&module)
            .map_err(|e| LustError::RuntimeError {
                message: format!("Cranelift JIT module compilation failed: {:?}", e),
            })?;
        for (name, ptr) in compiled_map {
            self.cranelift_compiled.insert(name, ptr);
            self.cranelift_compilation_count += 1;
        }
        Ok(())
    }

    #[cfg(not(feature = "std"))]
    pub fn compile_cranelift(&mut self) -> Result<()> {
        Ok(())
    }

    /// Attempts to execute the top call frame natively using Cranelift JIT.
    #[cfg(feature = "std")]
    pub(crate) fn try_run_cranelift_top_frame(&mut self) -> Result<Option<Option<Value>>> {
        let (func_idx, param_count, has_upvalues) = match self.call_stack.last() {
            Some(f) => {
                let p_cnt = self
                    .functions
                    .get(f.function_idx)
                    .map(|func| func.param_count as usize)
                    .unwrap_or(0);
                (
                    f.function_idx,
                    p_cnt,
                    !f.upvalues.is_empty() || !f.shared_upvalues.is_empty(),
                )
            }
            None => return Ok(Some(Some(Value::Nil))),
        };
        if has_upvalues || param_count > 6 {
            return Ok(None);
        }
        let func_name = match self.functions.get(func_idx) {
            Some(f) => f.name.clone(),
            None => {
                return Err(LustError::RuntimeError {
                    message: format!("Function index {} not found in VM", func_idx),
                });
            }
        };

        let Some(mir_sig_info) = self.mir_functions.get(&func_name).and_then(|f| {
            if !Self::is_cranelift_eligible_signature(f) {
                return None;
            }
            let param_types: Vec<MirType> = f
                .params
                .iter()
                .map(|pid| f.locals[pid.0].ty.clone())
                .collect();
            Some((param_types, f.signature.return_type.clone()))
        }) else {
            return Ok(None);
        };
        let (param_types, ret_type) = mir_sig_info;
        if param_types.len() != param_count || !Self::is_cranelift_supported_type(&ret_type) {
            return Ok(None);
        }

        let baseline_handles = self.mir_context.handle_count();

        for (name, info) in &self.struct_metadata {
            let field_types: Vec<MirType> = (0..info.layout.field_names().len())
                .map(|i| crate::mir::lowering::mir_type_from_ast(info.layout.field_type(i)))
                .collect();
            self.mir_context.register_struct_field_types(name, field_types);
        }

        // Convert parameters to RawArg
        let mut raw_args = Vec::with_capacity(param_count);
        let mut array_args: Vec<(usize, Rc<RefCell<Vec<Value>>>, *mut i64)> = Vec::new();
        let mut struct_args: Vec<(usize, Rc<StructObject>, *mut i64)> = Vec::new();

        {
            let frame = self.call_stack.last().unwrap();
            if frame.registers.len() < param_count {
                return Ok(None);
            }
            for (i, expected_ty) in param_types.iter().enumerate() {
                let val = &frame.registers[i];
                let arg = match (val, expected_ty) {
                    (Value::Int(n), MirType::Int | MirType::Unknown) => RawArg::Int(*n),
                    (Value::Bool(b), MirType::Bool | MirType::Unknown) => RawArg::Int(i64::from(*b)),
                    (Value::Float(f), MirType::Float | MirType::Unknown) => RawArg::Float(*f),
                    (Value::String(_), MirType::String | MirType::Unknown) => {
                        let h = self.mir_context.register_handle(val.clone());
                        RawArg::Int(h)
                    }
                    (Value::Array(rc), MirType::Array(elem_ty)) => {
                        let elements = rc.borrow();
                        let len = elements.len();
                        let ptr = lust_mir_alloc_array(len as i64);
                        if ptr.is_null() {
                            self.mir_context.clear_to_baseline(baseline_handles);
                            return Err(LustError::RuntimeError {
                                message: "Out of memory allocating array buffer for Cranelift JIT".into(),
                            });
                        }
                        for (idx, elem) in elements.iter().enumerate() {
                            let raw_elem = match (elem, &**elem_ty) {
                                (Value::Int(n), _) => *n,
                                (Value::Bool(b), _) => i64::from(*b),
                                (Value::Float(f), _) => f.to_bits() as i64,
                                (Value::String(_), _) => self.mir_context.register_handle(elem.clone()),
                                _ => 0,
                            };
                            unsafe {
                                *ptr.offset(2 + idx as isize) = raw_elem;
                            }
                        }
                        array_args.push((i, rc.clone(), ptr));
                        RawArg::Int(ptr as i64)
                    }
                    (Value::Tuple(rc), MirType::Tuple(elem_types)) => {
                        let len = rc.len();
                        let ptr = lust_mir_alloc_array(len as i64);
                        if ptr.is_null() {
                            self.mir_context.clear_to_baseline(baseline_handles);
                            return Err(LustError::RuntimeError {
                                message: "Out of memory allocating tuple buffer for Cranelift JIT".into(),
                            });
                        }
                        for (idx, elem) in rc.iter().enumerate() {
                            let elem_ty = elem_types.get(idx).unwrap_or(&MirType::Int);
                            let raw_elem = match (elem, elem_ty) {
                                (Value::Int(n), _) => *n,
                                (Value::Bool(b), _) => i64::from(*b),
                                (Value::Float(f), _) => f.to_bits() as i64,
                                (Value::String(_), _) => self.mir_context.register_handle(elem.clone()),
                                _ => 0,
                            };
                            unsafe {
                                *ptr.offset(2 + idx as isize) = raw_elem;
                            }
                        }
                        RawArg::Int(ptr as i64)
                    }
                    (Value::Struct(rc), MirType::Struct(_) | MirType::Unknown) => {
                        let fields = rc.fields.borrow();
                        let len = fields.len();
                        let ptr = lust_mir_alloc_array(len as i64);
                        if ptr.is_null() {
                            self.mir_context.clear_to_baseline(baseline_handles);
                            return Err(LustError::RuntimeError {
                                message: "Out of memory allocating struct buffer for Cranelift JIT".into(),
                            });
                        }
                        for (idx, fld) in fields.iter().enumerate() {
                            let raw_fld = match fld {
                                Value::Int(n) => *n,
                                Value::Bool(b) => i64::from(*b),
                                Value::Float(f) => f.to_bits() as i64,
                                Value::String(_) | Value::Map(_) | Value::Enum(_) => {
                                    self.mir_context.register_handle(fld.clone())
                                }
                                _ => 0,
                            };
                            unsafe {
                                *ptr.offset(2 + idx as isize) = raw_fld;
                            }
                        }
                        if !self.struct_metadata.contains_key(rc.name.as_str()) {
                            self.struct_metadata.insert(
                                rc.name.to_string(),
                                crate::vm::RuntimeStructInfo {
                                    layout: rc.layout.clone(),
                                },
                            );
                            let field_types: Vec<MirType> = (0..rc.layout.field_names().len())
                                .map(|fi| crate::mir::lowering::mir_type_from_ast(rc.layout.field_type(fi)))
                                .collect();
                            self.mir_context.register_struct_field_types(rc.name.as_str(), field_types);
                        }
                        struct_args.push((i, rc.clone(), ptr));
                        RawArg::Int(ptr as i64)
                    }
                    (Value::Map(_), MirType::Map(_, _) | MirType::Unknown) => {
                        let h = self.mir_context.register_handle(val.clone());
                        RawArg::Int(h)
                    }
                    (Value::Enum(_), MirType::Enum(_) | MirType::Unknown) => {
                        let h = self.mir_context.register_handle(val.clone());
                        RawArg::Int(h)
                    }
                    _ => {
                        self.mir_context.clear_to_baseline(baseline_handles);
                        return Ok(None);
                    }
                };
                raw_args.push(arg);
            }
        }

        if !self.cranelift_compiled.contains_key(&func_name) {
            self.compile_cranelift()?;
        }
        let Some(code_ptr) = self.cranelift_compiled.get(&func_name).copied() else {
            self.mir_context.clear_to_baseline(baseline_handles);
            return Ok(None);
        };

        let vm_gas_budget = self.gas_remaining();
        let vm_has_mem_budget = self.memory_budget_bytes().is_some();
        let mem_limit = match self.memory_remaining_bytes() {
            Some(rem_mem) => i64::try_from(rem_mem).unwrap_or(i64::MAX),
            None => i64::MAX,
        };

        self.mir_context.set_status(MIR_STATUS_OK);
        self.mir_context.set_memory_used(0);
        self.mir_context.set_max_memory_limit(mem_limit);
        if let Some(rem_gas) = vm_gas_budget {
            self.mir_context
                .set_gas(i64::try_from(rem_gas).unwrap_or(i64::MAX));
        }
        let gas_entry_snapshot = self.mir_context.gas_remaining();

        self.cranelift_entry_count += 1;
        let _jit_guard = self.enter_jit_scope();
        let _mir_guard = self.mir_context.enter();

        // Charge base gas for entry
        crate::mir::cranelift_backend::jit_engine::lust_mir_charge_gas(1);

        // Execute JIT compiled code matching argument types and return type
        let (raw_int, raw_float): (i64, f64) = match (raw_args.as_slice(), &ret_type) {
            // 0 args
            (&[], MirType::Float) => {
                let func: extern "C" fn() -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func())
            }
            (&[], MirType::Unit) => {
                let func: extern "C" fn() = unsafe { core::mem::transmute(code_ptr) };
                func();
                (0, 0.0)
            }
            (&[], _) => {
                let func: extern "C" fn() -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(), 0.0)
            }

            // 1 arg
            (&[RawArg::Float(a)], MirType::Float) => {
                let func: extern "C" fn(f64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a))
            }
            (&[RawArg::Float(a)], MirType::Unit) => {
                let func: extern "C" fn(f64) = unsafe { core::mem::transmute(code_ptr) };
                func(a);
                (0, 0.0)
            }
            (&[RawArg::Float(a)], _) => {
                let func: extern "C" fn(f64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a), 0.0)
            }
            (&[RawArg::Int(a)], MirType::Float) => {
                let func: extern "C" fn(i64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a))
            }
            (&[RawArg::Int(a)], MirType::Unit) => {
                let func: extern "C" fn(i64) = unsafe { core::mem::transmute(code_ptr) };
                func(a);
                (0, 0.0)
            }
            (&[RawArg::Int(a)], _) => {
                let func: extern "C" fn(i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a), 0.0)
            }

            // 2 args
            (&[RawArg::Float(a), RawArg::Float(b)], MirType::Float) => {
                let func: extern "C" fn(f64, f64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b))
            }
            (&[RawArg::Float(a), RawArg::Float(b)], MirType::Unit) => {
                let func: extern "C" fn(f64, f64) = unsafe { core::mem::transmute(code_ptr) };
                func(a, b);
                (0, 0.0)
            }
            (&[RawArg::Float(a), RawArg::Float(b)], _) => {
                let func: extern "C" fn(f64, f64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b), 0.0)
            }
            (&[RawArg::Int(a), RawArg::Float(b)], MirType::Float) => {
                let func: extern "C" fn(i64, f64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b))
            }
            (&[RawArg::Int(a), RawArg::Float(b)], MirType::Unit) => {
                let func: extern "C" fn(i64, f64) = unsafe { core::mem::transmute(code_ptr) };
                func(a, b);
                (0, 0.0)
            }
            (&[RawArg::Int(a), RawArg::Float(b)], _) => {
                let func: extern "C" fn(i64, f64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b), 0.0)
            }
            (&[RawArg::Float(a), RawArg::Int(b)], MirType::Float) => {
                let func: extern "C" fn(f64, i64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b))
            }
            (&[RawArg::Float(a), RawArg::Int(b)], MirType::Unit) => {
                let func: extern "C" fn(f64, i64) = unsafe { core::mem::transmute(code_ptr) };
                func(a, b);
                (0, 0.0)
            }
            (&[RawArg::Float(a), RawArg::Int(b)], _) => {
                let func: extern "C" fn(f64, i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b), 0.0)
            }
            (&[RawArg::Int(a), RawArg::Int(b)], MirType::Float) => {
                let func: extern "C" fn(i64, i64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b))
            }
            (&[RawArg::Int(a), RawArg::Int(b)], MirType::Unit) => {
                let func: extern "C" fn(i64, i64) = unsafe { core::mem::transmute(code_ptr) };
                func(a, b);
                (0, 0.0)
            }
            (&[RawArg::Int(a), RawArg::Int(b)], _) => {
                let func: extern "C" fn(i64, i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b), 0.0)
            }

            // 3 args
            (&[RawArg::Float(a), RawArg::Float(b), RawArg::Float(c)], MirType::Float) => {
                let func: extern "C" fn(f64, f64, f64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b, c))
            }
            (&[RawArg::Float(a), RawArg::Float(b), RawArg::Float(c)], _) => {
                let func: extern "C" fn(f64, f64, f64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b, c), 0.0)
            }
            (&[RawArg::Int(a), RawArg::Int(b), RawArg::Int(c)], MirType::Float) => {
                let func: extern "C" fn(i64, i64, i64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b, c))
            }
            (&[RawArg::Int(a), RawArg::Int(b), RawArg::Int(c)], _) => {
                let func: extern "C" fn(i64, i64, i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b, c), 0.0)
            }
            (&[RawArg::Int(a), RawArg::Int(b), RawArg::Float(c)], MirType::Float) => {
                let func: extern "C" fn(i64, i64, f64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b, c))
            }
            (&[RawArg::Int(a), RawArg::Int(b), RawArg::Float(c)], _) => {
                let func: extern "C" fn(i64, i64, f64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b, c), 0.0)
            }
            (&[RawArg::Float(a), RawArg::Float(b), RawArg::Int(c)], MirType::Float) => {
                let func: extern "C" fn(f64, f64, i64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b, c))
            }
            (&[RawArg::Float(a), RawArg::Float(b), RawArg::Int(c)], _) => {
                let func: extern "C" fn(f64, f64, i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b, c), 0.0)
            }
            (&[RawArg::Int(a), RawArg::Float(b), RawArg::Int(c)], MirType::Float) => {
                let func: extern "C" fn(i64, f64, i64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b, c))
            }
            (&[RawArg::Int(a), RawArg::Float(b), RawArg::Int(c)], _) => {
                let func: extern "C" fn(i64, f64, i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b, c), 0.0)
            }
            (&[RawArg::Float(a), RawArg::Int(b), RawArg::Float(c)], MirType::Float) => {
                let func: extern "C" fn(f64, i64, f64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b, c))
            }
            (&[RawArg::Float(a), RawArg::Int(b), RawArg::Float(c)], _) => {
                let func: extern "C" fn(f64, i64, f64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b, c), 0.0)
            }
            (&[RawArg::Int(a), RawArg::Float(b), RawArg::Float(c)], MirType::Float) => {
                let func: extern "C" fn(i64, f64, f64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b, c))
            }
            (&[RawArg::Int(a), RawArg::Float(b), RawArg::Float(c)], _) => {
                let func: extern "C" fn(i64, f64, f64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b, c), 0.0)
            }
            (&[RawArg::Float(a), RawArg::Int(b), RawArg::Int(c)], MirType::Float) => {
                let func: extern "C" fn(f64, i64, i64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b, c))
            }
            (&[RawArg::Float(a), RawArg::Int(b), RawArg::Int(c)], _) => {
                let func: extern "C" fn(f64, i64, i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b, c), 0.0)
            }

            // 4 args
            (&[RawArg::Float(a), RawArg::Float(b), RawArg::Float(c), RawArg::Float(d)], MirType::Float) => {
                let func: extern "C" fn(f64, f64, f64, f64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b, c, d))
            }
            (&[RawArg::Float(a), RawArg::Float(b), RawArg::Float(c), RawArg::Float(d)], _) => {
                let func: extern "C" fn(f64, f64, f64, f64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b, c, d), 0.0)
            }
            (&[RawArg::Int(a), RawArg::Int(b), RawArg::Int(c), RawArg::Int(d)], MirType::Float) => {
                let func: extern "C" fn(i64, i64, i64, i64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b, c, d))
            }
            (&[RawArg::Int(a), RawArg::Int(b), RawArg::Int(c), RawArg::Int(d)], _) => {
                let func: extern "C" fn(i64, i64, i64, i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b, c, d), 0.0)
            }

            // 5 args
            (&[RawArg::Float(a), RawArg::Float(b), RawArg::Float(c), RawArg::Float(d), RawArg::Float(e)], MirType::Float) => {
                let func: extern "C" fn(f64, f64, f64, f64, f64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b, c, d, e))
            }
            (&[RawArg::Float(a), RawArg::Float(b), RawArg::Float(c), RawArg::Float(d), RawArg::Float(e)], _) => {
                let func: extern "C" fn(f64, f64, f64, f64, f64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b, c, d, e), 0.0)
            }
            (&[RawArg::Int(a), RawArg::Int(b), RawArg::Int(c), RawArg::Int(d), RawArg::Int(e)], MirType::Float) => {
                let func: extern "C" fn(i64, i64, i64, i64, i64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b, c, d, e))
            }
            (&[RawArg::Int(a), RawArg::Int(b), RawArg::Int(c), RawArg::Int(d), RawArg::Int(e)], _) => {
                let func: extern "C" fn(i64, i64, i64, i64, i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b, c, d, e), 0.0)
            }

            // 6 args
            (&[RawArg::Float(a), RawArg::Float(b), RawArg::Float(c), RawArg::Float(d), RawArg::Float(e), RawArg::Float(f)], MirType::Float) => {
                let func: extern "C" fn(f64, f64, f64, f64, f64, f64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b, c, d, e, f))
            }
            (&[RawArg::Float(a), RawArg::Float(b), RawArg::Float(c), RawArg::Float(d), RawArg::Float(e), RawArg::Float(f)], _) => {
                let func: extern "C" fn(f64, f64, f64, f64, f64, f64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b, c, d, e, f), 0.0)
            }
            (&[RawArg::Int(a), RawArg::Int(b), RawArg::Int(c), RawArg::Int(d), RawArg::Int(e), RawArg::Int(f)], MirType::Float) => {
                let func: extern "C" fn(i64, i64, i64, i64, i64, i64) -> f64 = unsafe { core::mem::transmute(code_ptr) };
                (0, func(a, b, c, d, e, f))
            }
            (&[RawArg::Int(a), RawArg::Int(b), RawArg::Int(c), RawArg::Int(d), RawArg::Int(e), RawArg::Int(f)], _) => {
                let func: extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };
                (func(a, b, c, d, e, f), 0.0)
            }

            _ => {
                self.mir_context.clear_to_baseline(baseline_handles);
                return Ok(None);
            }
        };

        let status = crate::mir::cranelift_backend::jit_engine::lust_mir_get_status();
        drop(_mir_guard);
        drop(_jit_guard);

        let gas_after = self.mir_context.gas_remaining();
        let gas_spent = (gas_entry_snapshot.saturating_sub(gas_after)).max(0) as u64;
        let gas_charge_res = if vm_gas_budget.is_some() && gas_spent > 0 {
            self.budgets.charge_gas(gas_spent)
        } else {
            Ok(())
        };

        let mem_spent = usize::try_from(self.mir_context.memory_used().max(0)).unwrap_or(usize::MAX);
        let mem_charge_res = if vm_has_mem_budget && mem_spent > 0 {
            self.budgets.charge_mem_bytes(mem_spent)
        } else {
            Ok(())
        };

        if status != MIR_STATUS_OK {
            self.mir_context.clear_to_baseline(baseline_handles);
            if let Some(frame) = self.call_stack.pop() {
                self.recycle_frame(frame);
            }
            let message = match status {
                MIR_STATUS_ERR_OUT_OF_GAS => "Out of gas during Cranelift JIT execution".into(),
                MIR_STATUS_ERR_OOM => "Out of memory during Cranelift JIT execution".into(),
                MIR_STATUS_ERR_DIV_BY_ZERO => {
                    "Division or modulo by zero during Cranelift JIT execution".into()
                }
                MIR_STATUS_ERR_NEG_EXPONENT => {
                    "Negative exponent in integer power during Cranelift JIT execution".into()
                }
                _ => format!("Cranelift native execution failed with status {}", status),
            };
            return Err(LustError::RuntimeError { message });
        }

        if let Err(err) = gas_charge_res {
            self.mir_context.clear_to_baseline(baseline_handles);
            if let Some(frame) = self.call_stack.pop() {
                self.recycle_frame(frame);
            }
            return Err(err);
        }
        if let Err(err) = mem_charge_res {
            self.mir_context.clear_to_baseline(baseline_handles);
            if let Some(frame) = self.call_stack.pop() {
                self.recycle_frame(frame);
            }
            return Err(err);
        }

        // Synchronize in-place modifications for mutable array arguments
        for (_, rc, ptr) in array_args {
            if !ptr.is_null() {
                let len = unsafe { *ptr } as usize;
                let mut elements = rc.borrow_mut();
                for i in 0..elements.len().min(len) {
                    let raw = unsafe { *ptr.offset(2 + i as isize) };
                    if let Value::Int(n) = &mut elements[i] {
                        *n = raw;
                    } else if let Value::Float(f) = &mut elements[i] {
                        *f = f64::from_bits(raw as u64);
                    }
                }
            }
        }

        // Synchronize in-place modifications for mutable struct arguments
        for (_, rc, ptr) in struct_args {
            if !ptr.is_null() {
                let len = unsafe { *ptr } as usize;
                let mut fields = rc.fields.borrow_mut();
                for i in 0..fields.len().min(len) {
                    let raw = unsafe { *ptr.offset(2 + i as isize) };
                    match &mut fields[i] {
                        Value::Int(n) => *n = raw,
                        Value::Bool(b) => *b = raw != 0,
                        Value::Float(f) => *f = f64::from_bits(raw as u64),
                        Value::String(_) | Value::Map(_) | Value::Enum(_) => {
                            if let Some(val) = self.mir_context.get_handle(raw) {
                                fields[i] = val;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Reconstruct return value
        let result_val = match &ret_type {
            MirType::Float => Value::Float(raw_float),
            MirType::Int => Value::Int(raw_int),
            MirType::Unknown => {
                if let Some(val) = self.mir_context.get_handle(raw_int) {
                    val
                } else {
                    Value::Int(raw_int)
                }
            }
            MirType::Bool => Value::Bool((raw_int as u8) != 0),
            MirType::Unit => Value::Nil,
            MirType::String => {
                self.mir_context.get_handle(raw_int).unwrap_or_else(|| Value::string(""))
            }
            MirType::Array(elem_ty) => {
                let ptr = raw_int as *const i64;
                if ptr.is_null() {
                    Value::Array(Rc::new(RefCell::new(Vec::new())))
                } else {
                    let len = unsafe { *ptr };
                    let count = if len < 0 { 0 } else { len as usize };
                    let mut elems = Vec::with_capacity(count);
                    for i in 0..count {
                        let raw = unsafe { *ptr.offset(2 + i as isize) };
                        let elem = match &**elem_ty {
                            MirType::Int => Value::Int(raw),
                            MirType::Bool => Value::Bool(raw != 0),
                            MirType::Float => Value::Float(f64::from_bits(raw as u64)),
                            MirType::String => self.mir_context.get_handle(raw).unwrap_or_else(|| Value::string("")),
                            _ => Value::Int(raw),
                        };
                        elems.push(elem);
                    }
                    Value::Array(Rc::new(RefCell::new(elems)))
                }
            }
            MirType::Tuple(elem_types) => {
                let ptr = raw_int as *const i64;
                if ptr.is_null() {
                    Value::Tuple(Rc::new(Vec::new()))
                } else {
                    let len = unsafe { *ptr };
                    let count = if len < 0 { 0 } else { len as usize };
                    let mut elems = Vec::with_capacity(count);
                    for i in 0..count {
                        let raw = unsafe { *ptr.offset(2 + i as isize) };
                        let elem_ty = elem_types.get(i).unwrap_or(&MirType::Int);
                        let elem = match elem_ty {
                            MirType::Int => Value::Int(raw),
                            MirType::Bool => Value::Bool(raw != 0),
                            MirType::Float => Value::Float(f64::from_bits(raw as u64)),
                            MirType::String => self.mir_context.get_handle(raw).unwrap_or_else(|| Value::string("")),
                            _ => Value::Int(raw),
                        };
                        elems.push(elem);
                    }
                    Value::Tuple(Rc::new(elems))
                }
            }
            MirType::Struct(name) => {
                let ptr = raw_int as *const i64;
                if ptr.is_null() {
                    Value::Nil
                } else {
                    let len = unsafe { *ptr };
                    let count = if len < 0 { 0 } else { len as usize };
                    let layout_opt = self.struct_metadata.get(name).map(|info| info.layout.clone());
                    let mut fields = Vec::with_capacity(count);
                    for i in 0..count {
                        let raw = unsafe { *ptr.offset(2 + i as isize) };
                        let fld_val = if let Some(layout) = &layout_opt {
                            if i < layout.field_names().len() {
                                match layout.field_type(i).kind {
                                    TypeKind::Float => Value::Float(f64::from_bits(raw as u64)),
                                    TypeKind::Bool => Value::Bool(raw != 0),
                                    TypeKind::String => {
                                        self.mir_context.get_handle(raw).unwrap_or_else(|| Value::string(""))
                                    }
                                    TypeKind::Map(_, _) => {
                                        self.mir_context.get_handle(raw).unwrap_or_else(|| Value::map(LustMap::default()))
                                    }
                                    TypeKind::Option(_) | TypeKind::Result(_, _) => {
                                        self.mir_context.get_handle(raw).unwrap_or(Value::Nil)
                                    }
                                    _ => Value::Int(raw),
                                }
                            } else {
                                Value::Int(raw)
                            }
                        } else {
                            if let Some(h_val) = self.mir_context.get_handle(raw) {
                                h_val
                            } else {
                                Value::Int(raw)
                            }
                        };
                        fields.push(fld_val);
                    }
                    let layout = layout_opt.unwrap_or_else(|| {
                        Rc::new(StructLayout::new(
                            name.clone(),
                            vec![],
                            vec![],
                            vec![],
                            vec![],
                        ))
                    });
                    Value::Struct(StructObject::new(name.clone(), layout, fields))
                }
            }
            MirType::Map(_, _) => {
                self.mir_context.get_handle(raw_int).unwrap_or_else(|| Value::map(LustMap::default()))
            }
            MirType::Enum(name) => {
                if name == "Option" {
                    if raw_int == 0 {
                        Value::none()
                    } else if let Some(val) = self.mir_context.get_handle(raw_int) {
                        if matches!(&val, Value::Enum(_)) {
                            val
                        } else {
                            Value::some(val)
                        }
                    } else {
                        Value::some(Value::Int(raw_int))
                    }
                } else if name == "Result" {
                    if let Some(val) = self.mir_context.get_handle(raw_int) {
                        if matches!(&val, Value::Enum(_)) {
                            val
                        } else {
                            Value::ok(val)
                        }
                    } else {
                        Value::ok(Value::Int(raw_int))
                    }
                } else {
                    self.mir_context.get_handle(raw_int).unwrap_or(Value::Nil)
                }
            }
            _ => unreachable!("validated supported return type"),
        };

        // Clear temporary frame-scoped handles and allocated buffers on success
        self.mir_context.clear_to_baseline(baseline_handles);

        let finished = self.finish_return(result_val)?;
        Ok(Some(finished))
    }

    #[cfg(not(feature = "std"))]
    pub(crate) fn try_run_cranelift_top_frame(&mut self) -> Result<Option<Option<Value>>> {
        Ok(None)
    }

    /// Executes the active call frame using the native Cranelift JIT compiler backend.
    #[cfg(feature = "std")]
    pub fn run_cranelift(&mut self) -> Result<Value> {
        if let Some(finished) = self.try_run_cranelift_top_frame()? {
            if let Some(val) = finished {
                return Ok(val);
            }
            if let Some(target_depth) = self.call_until_depth
                && self.call_stack.len() == target_depth
                && let Some(return_value) = self.pending_return_value.take()
            {
                self.call_until_depth = None;
                return Ok(return_value);
            }
        }
        let result = if self.effective_jit_enabled() {
            self.run_inner::<true>()
        } else {
            self.run_inner::<false>()
        };
        crate::bytecode::value::sync_all_specialized_arrays();
        result
    }

    /// In no_std mode, fall back to interpreter dispatch.
    #[cfg(not(feature = "std"))]
    pub fn run_cranelift(&mut self) -> Result<Value> {
        self.run_inner::<false>()
    }
}
