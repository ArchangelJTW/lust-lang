#[cfg(feature = "std")]
pub mod jit_engine {
    use crate::mir::ir::*;
    use alloc::{boxed::Box, format, string::String, vec::Vec};
    use cranelift_codegen::Context;
    use cranelift_codegen::ir::{
        AbiParam, FuncRef, InstBuilder, MemFlagsData, Value as ClifValue,
        condcodes::IntCC, types,
    };
    use cranelift_codegen::settings::{self, Configurable};
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::{Linkage, Module};

    static GAS_REMAINING: core::sync::atomic::AtomicI64 = core::sync::atomic::AtomicI64::new(1_000_000_000);

    #[unsafe(no_mangle)]
    pub extern "C" fn lust_mir_set_gas(gas: i64) {
        GAS_REMAINING.store(gas, core::sync::atomic::Ordering::SeqCst);
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn lust_mir_get_gas() -> i64 {
        GAS_REMAINING.load(core::sync::atomic::Ordering::SeqCst)
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn lust_mir_charge_gas(amt: i64) -> i64 {
        let cur = GAS_REMAINING.fetch_sub(amt, core::sync::atomic::Ordering::SeqCst);
        cur - amt
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn lust_mir_alloc_array(len: i64) -> *mut i64 {
        let size = (len as usize + 2).max(2);
        let mut vec = Vec::<i64>::with_capacity(size);
        vec.push(len);
        vec.push(len);
        vec.resize(size, 0);
        let slice = vec.into_boxed_slice();
        Box::into_raw(slice) as *mut i64
    }

    pub struct CraneliftJitCompiler {
        builder_ctx: FunctionBuilderContext,
        ctx: Context,
        module: JITModule,
    }

    impl CraneliftJitCompiler {
        pub fn new() -> Result<Self, String> {
            let mut flag_builder = settings::builder();
            flag_builder
                .set("use_colocated_libcalls", "false")
                .map_err(|e| format!("{e}"))?;
            if cfg!(target_arch = "x86_64") {
                flag_builder
                    .set("is_pic", "true")
                    .map_err(|e| format!("{e}"))?;
            }

            let isa_builder = cranelift_native::builder()
                .map_err(|msg| format!("Host machine not supported by cranelift: {msg}"))?;
            let isa = isa_builder
                .finish(settings::Flags::new(flag_builder))
                .map_err(|e| format!("{e}"))?;

            let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
            builder.symbol("lust_mir_alloc_array", lust_mir_alloc_array as *const u8);
            builder.symbol("lust_mir_charge_gas", lust_mir_charge_gas as *const u8);
            let module = JITModule::new(builder);

            Ok(Self {
                builder_ctx: FunctionBuilderContext::new(),
                ctx: module.make_context(),
                module,
            })
        }

        pub fn compile_function(&mut self, mir_fn: &MirFunction) -> Result<*const u8, String> {
            // Build signature
            for param_id in &mir_fn.params {
                let p_ty = &mir_fn.locals[param_id.0].ty;
                self.ctx.func.signature.params.push(AbiParam::new(mir_to_clif_type(p_ty)));
            }

            let ret_clif_ty = mir_to_clif_type(&mir_fn.signature.return_type);
            if ret_clif_ty != types::I8 || mir_fn.signature.return_type != MirType::Unit {
                self.ctx.func.signature.returns.push(AbiParam::new(ret_clif_ty));
            }

            // Declare lust_mir_alloc_array import
            let mut alloc_sig = self.module.make_signature();
            alloc_sig.params.push(AbiParam::new(types::I64));
            alloc_sig.returns.push(AbiParam::new(types::I64));
            let alloc_callee = self
                .module
                .declare_function("lust_mir_alloc_array", Linkage::Import, &alloc_sig)
                .map_err(|e| format!("{e}"))?;

            // Declare lust_mir_charge_gas import
            let mut gas_sig = self.module.make_signature();
            gas_sig.params.push(AbiParam::new(types::I64));
            let gas_callee = self
                .module
                .declare_function("lust_mir_charge_gas", Linkage::Import, &gas_sig)
                .map_err(|e| format!("{e}"))?;

            {
                let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);
                let alloc_fn = self.module.declare_func_in_func(alloc_callee, &mut builder.func);
                let gas_fn = self.module.declare_func_in_func(gas_callee, &mut builder.func);

                // Create CLIF blocks corresponding 1-to-1 to MIR basic blocks
                let mut clif_blocks = Vec::new();
                for _ in &mir_fn.blocks {
                    clif_blocks.push(builder.create_block());
                }

                // Declare variables for each MIR local
                let mut local_vars = Vec::with_capacity(mir_fn.locals.len());
                for local in &mir_fn.locals {
                    let clif_ty = mir_to_clif_type(&local.ty);
                    let var = builder.declare_var(clif_ty);
                    local_vars.push(var);
                }

                // Append parameters to entry block
                let entry_clif = clif_blocks[mir_fn.entry.0];
                builder.append_block_params_for_function_params(entry_clif);
                builder.switch_to_block(entry_clif);

                for (idx, p_id) in mir_fn.params.iter().enumerate() {
                    let param_val = builder.block_params(entry_clif)[idx];
                    builder.def_var(local_vars[p_id.0], param_val);
                }

                // Lower each basic block
                for (b_idx, mir_block) in mir_fn.blocks.iter().enumerate() {
                    let clif_bb = clif_blocks[b_idx];
                    builder.switch_to_block(clif_bb);

                    for stmt in &mir_block.statements {
                        match stmt {
                            Statement::Assign(Place::Local(dst), rvalue, _) => {
                                let val = lower_rvalue(rvalue, &mut builder, &local_vars, alloc_fn);
                                builder.def_var(local_vars[dst.0], val);
                            }
                            Statement::Assign(Place::Index(base, idx), rvalue, _) => {
                                let val = lower_rvalue(rvalue, &mut builder, &local_vars, alloc_fn);
                                let base_ptr = lower_place(base, &mut builder, &local_vars);
                                let idx_val = builder.use_var(local_vars[idx.0]);
                                let eight = builder.ins().iconst(types::I64, 8);
                                let byte_idx = builder.ins().imul(idx_val, eight);
                                let sixteen = builder.ins().iconst(types::I64, 16);
                                let offset = builder.ins().iadd(byte_idx, sixteen);
                                let elem_addr = builder.ins().iadd(base_ptr, offset);
                                builder.ins().store(MemFlagsData::trusted(), val, elem_addr, 0);
                            }
                            Statement::ResourceCheck(ResourceKind::Gas, amt, _) => {
                                let amt_val = builder.ins().iconst(types::I64, *amt as i64);
                                builder.ins().call(gas_fn, &[amt_val]);
                            }
                            _ => {}
                        }
                    }

                    if let Some(term) = &mir_block.terminator {
                        match term {
                            Terminator::Goto(target, _) => {
                                builder.ins().jump(clif_blocks[target.0], &[]);
                            }
                            Terminator::Cond { cond, then_block, else_block, .. } => {
                                let c_val = lower_operand(cond, &mut builder, &local_vars);
                                builder.ins().brif(c_val, clif_blocks[then_block.0], &[], clif_blocks[else_block.0], &[]);
                            }
                            Terminator::Return(opt_op, _) => {
                                if ret_clif_ty != types::I8 || mir_fn.signature.return_type != MirType::Unit {
                                    let ret_val = if let Some(op) = opt_op {
                                        lower_operand(op, &mut builder, &local_vars)
                                    } else {
                                        builder.ins().iconst(ret_clif_ty, 0)
                                    };
                                    builder.ins().return_(&[ret_val]);
                                } else {
                                    builder.ins().return_(&[]);
                                }
                            }
                            Terminator::Unreachable(_) => {
                                builder.ins().trap(cranelift_codegen::ir::TrapCode::unwrap_user(1));
                            }
                            _ => {
                                if ret_clif_ty != types::I8 || mir_fn.signature.return_type != MirType::Unit {
                                    let ret_val = builder.ins().iconst(ret_clif_ty, 0);
                                    builder.ins().return_(&[ret_val]);
                                } else {
                                    builder.ins().return_(&[]);
                                }
                            }
                        }
                    }
                }

                builder.seal_all_blocks();
                builder.finalize(self.module.target_config());
            }

            // Declare & define function in JIT module
            let func_id = self
                .module
                .declare_function(&mir_fn.signature.name, Linkage::Export, &self.ctx.func.signature)
                .map_err(|e| format!("{e}"))?;

            self.module
                .define_function(func_id, &mut self.ctx)
                .map_err(|e| format!("{e}\nCLIF:\n{}", self.ctx.func))?;

            self.module.clear_context(&mut self.ctx);
            self.module.finalize_definitions().map_err(|e| format!("{e}"))?;

            let code_ptr = self.module.get_finalized_function(func_id);
            Ok(code_ptr)
        }
    }

    fn mir_to_clif_type(ty: &MirType) -> types::Type {
        match ty {
            MirType::Int => types::I64,
            MirType::Float => types::F64,
            MirType::Bool => types::I8,
            MirType::Array(_) => types::I64,
            _ => types::I64,
        }
    }

    fn lower_place(
        place: &Place,
        builder: &mut FunctionBuilder,
        local_vars: &[Variable],
    ) -> ClifValue {
        match place {
            Place::Local(loc) => builder.use_var(local_vars[loc.0]),
            Place::Index(base, idx) => {
                let base_ptr = lower_place(base, builder, local_vars);
                let idx_val = builder.use_var(local_vars[idx.0]);
                let eight = builder.ins().iconst(types::I64, 8);
                let byte_idx = builder.ins().imul(idx_val, eight);
                let sixteen = builder.ins().iconst(types::I64, 16);
                let offset = builder.ins().iadd(byte_idx, sixteen);
                let elem_addr = builder.ins().iadd(base_ptr, offset);
                builder.ins().load(types::I64, MemFlagsData::trusted(), elem_addr, 0)
            }
            Place::Field(base, _) => lower_place(base, builder, local_vars),
        }
    }

    fn lower_operand(op: &Operand, builder: &mut FunctionBuilder, local_vars: &[Variable]) -> ClifValue {
        match op {
            Operand::Copy(place) | Operand::Move(place) => lower_place(place, builder, local_vars),
            Operand::Constant(c) => match c {
                MirConstant::Int(i) => builder.ins().iconst(types::I64, *i),
                MirConstant::Float(f) => builder.ins().f64const(*f),
                MirConstant::Bool(b) => builder.ins().iconst(types::I8, if *b { 1 } else { 0 }),
                _ => builder.ins().iconst(types::I64, 0),
            },
        }
    }

    fn lower_rvalue(
        rval: &Rvalue,
        builder: &mut FunctionBuilder,
        local_vars: &[Variable],
        alloc_fn: FuncRef,
    ) -> ClifValue {
        match rval {
            Rvalue::Use(op) => lower_operand(op, builder, local_vars),
            Rvalue::BinaryOp(binop, lhs, rhs) => {
                let l = lower_operand(lhs, builder, local_vars);
                let r = lower_operand(rhs, builder, local_vars);
                match binop {
                    MirBinaryOp::Add => builder.ins().iadd(l, r),
                    MirBinaryOp::Sub => builder.ins().isub(l, r),
                    MirBinaryOp::Mul => builder.ins().imul(l, r),
                    MirBinaryOp::Div => builder.ins().sdiv(l, r),
                    MirBinaryOp::Mod => builder.ins().srem(l, r),
                    MirBinaryOp::Eq => builder.ins().icmp(IntCC::Equal, l, r),
                    MirBinaryOp::NotEq => builder.ins().icmp(IntCC::NotEqual, l, r),
                    MirBinaryOp::Lt => builder.ins().icmp(IntCC::SignedLessThan, l, r),
                    MirBinaryOp::LtEq => builder.ins().icmp(IntCC::SignedLessThanOrEqual, l, r),
                    MirBinaryOp::Gt => builder.ins().icmp(IntCC::SignedGreaterThan, l, r),
                    MirBinaryOp::GtEq => builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, l, r),
                    MirBinaryOp::And => builder.ins().band(l, r),
                    MirBinaryOp::Or => builder.ins().bor(l, r),
                    _ => builder.ins().iadd(l, r),
                }
            }
            Rvalue::UnaryOp(unop, op) => {
                let v = lower_operand(op, builder, local_vars);
                match unop {
                    MirUnaryOp::Neg => builder.ins().ineg(v),
                    MirUnaryOp::Not => builder.ins().bnot(v),
                    MirUnaryOp::Len => {
                        // Offset 0 holds length
                        builder.ins().load(types::I64, MemFlagsData::trusted(), v, 0)
                    }
                }
            }
            Rvalue::Aggregate(MirType::Array(_), elements) => {
                let len = elements.len() as i64;
                let len_val = builder.ins().iconst(types::I64, len);
                let call_inst = builder.ins().call(alloc_fn, &[len_val]);
                let array_ptr = builder.inst_results(call_inst)[0];
                for (i, elem) in elements.iter().enumerate() {
                    let el_val = lower_operand(elem, builder, local_vars);
                    builder.ins().store(MemFlagsData::trusted(), el_val, array_ptr, 16 + (i as i32) * 8);
                }
                array_ptr
            }
            _ => builder.ins().iconst(types::I64, 0),
        }
    }
}
