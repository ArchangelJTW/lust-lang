#[cfg(feature = "cranelift")]
pub use jit_engine::MirCraneliftBackendError;

#[cfg(feature = "cranelift")]
pub mod jit_engine {
    use crate::mir::ir::*;
    pub use crate::mir::runtime_helpers::*;
    use alloc::{format, string::String, vec::Vec};
    use cranelift_codegen::Context;
    use cranelift_codegen::ir::{
        condcodes::IntCC, types, AbiParam, FuncRef, InstBuilder, MemFlagsData, Value as ClifValue,
    };
    use cranelift_codegen::settings::{self, Configurable};
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::{FuncId, Linkage, Module};

    #[derive(Debug, Clone, PartialEq, thiserror::Error)]
    pub enum MirCraneliftBackendError {
        #[error("Binary operator '{0:?}' is not supported by Cranelift backend")]
        UnsupportedBinaryOp(MirBinaryOp),
        #[error("Constant '{0:?}' is not supported by Cranelift backend")]
        UnsupportedConstant(MirConstant),
        #[error("Cranelift codegen error: {0}")]
        CodegenError(String),
        #[error("Failed or unlowerable value cannot be replaced with silent Unit: {0}")]
        UnlowerableValue(String),
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
            builder.symbol("lust_mir_set_gas", lust_mir_set_gas as *const u8);
            builder.symbol("lust_mir_get_gas", lust_mir_get_gas as *const u8);
            builder.symbol("lust_mir_charge_gas", lust_mir_charge_gas as *const u8);
            builder.symbol("lust_mir_alloc_array", lust_mir_alloc_array as *const u8);
            builder.symbol("lust_mir_array_len", lust_mir_array_len as *const u8);
            builder.symbol("lust_mir_array_get", lust_mir_array_get as *const u8);
            builder.symbol("lust_mir_array_set", lust_mir_array_set as *const u8);
            builder.symbol("lust_mir_floordiv_int", lust_mir_floordiv_int as *const u8);
            builder.symbol("lust_mir_pow_int", lust_mir_pow_int as *const u8);
            builder.symbol("lust_mir_abs_int", lust_mir_abs_int as *const u8);
            builder.symbol("lust_mir_min_int", lust_mir_min_int as *const u8);
            builder.symbol("lust_mir_max_int", lust_mir_max_int as *const u8);
            builder.symbol("lust_mir_clamp_int", lust_mir_clamp_int as *const u8);
            builder.symbol("lust_mir_get_status", lust_mir_get_status as *const u8);
            builder.symbol("lust_mir_set_status", lust_mir_set_status as *const u8);
            builder.symbol("lust_mir_print_int", lust_mir_print_int as *const u8);
            builder.symbol("lust_mir_div_int", lust_mir_div_int as *const u8);
            builder.symbol("lust_mir_rem_int", lust_mir_rem_int as *const u8);
            let module = JITModule::new(builder);

            Ok(Self {
                builder_ctx: FunctionBuilderContext::new(),
                ctx: module.make_context(),
                module,
            })
        }

        pub fn compile_function(&mut self, mir_fn: &MirFunction) -> Result<*const u8, MirCraneliftBackendError> {
            let mut module = MirModule::new(mir_fn.signature.name.clone());
            module.add_function(mir_fn.clone());
            let mut map = self.compile_module(&module)?;
            map.remove(&mir_fn.signature.name)
                .ok_or_else(|| MirCraneliftBackendError::CodegenError("Failed to get compiled function pointer".into()))
        }

        pub fn compile_module(
            &mut self,
            module: &MirModule,
        ) -> Result<hashbrown::HashMap<String, *const u8>, MirCraneliftBackendError> {
            let mut module_funcs = hashbrown::HashMap::new();

            // 1. Declare all functions first so they can reference each other
            for mir_fn in &module.functions {
                let mut sig = self.module.make_signature();
                for param_id in &mir_fn.params {
                    let p_ty = &mir_fn.locals[param_id.0].ty;
                    sig.params.push(AbiParam::new(mir_to_clif_type(p_ty)));
                }

                let ret_clif_ty = mir_to_clif_type(&mir_fn.signature.return_type);
                if ret_clif_ty != types::I8 || mir_fn.signature.return_type != MirType::Unit {
                    sig.returns.push(AbiParam::new(ret_clif_ty));
                }

                let func_id = self
                    .module
                    .declare_function(&mir_fn.signature.name, Linkage::Export, &sig)
                    .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;
                module_funcs.insert(mir_fn.signature.name.clone(), func_id);
            }

            // 2. Compile each function definition
            for mir_fn in &module.functions {
                let func_id = module_funcs[&mir_fn.signature.name];
                self.compile_function_internal(mir_fn, func_id, &module_funcs)?;
            }

            // 3. Finalize definitions
            self.module
                .finalize_definitions()
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            // 4. Collect finalized code pointers
            let mut res = hashbrown::HashMap::new();
            for mir_fn in &module.functions {
                let func_id = module_funcs[&mir_fn.signature.name];
                let code_ptr = self.module.get_finalized_function(func_id);
                res.insert(mir_fn.signature.name.clone(), code_ptr);
            }

            Ok(res)
        }

        fn compile_function_internal(
            &mut self,
            mir_fn: &MirFunction,
            func_id: FuncId,
            module_funcs: &hashbrown::HashMap<String, FuncId>,
        ) -> Result<(), MirCraneliftBackendError> {
            self.ctx.func.signature.params.clear();
            self.ctx.func.signature.returns.clear();

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
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            // Declare lust_mir_charge_gas import
            let mut gas_sig = self.module.make_signature();
            gas_sig.params.push(AbiParam::new(types::I64));
            gas_sig.returns.push(AbiParam::new(types::I64));
            let gas_callee = self
                .module
                .declare_function("lust_mir_charge_gas", Linkage::Import, &gas_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let mut status_sig = self.module.make_signature();
            status_sig.returns.push(AbiParam::new(types::I64));
            let status_callee = self
                .module
                .declare_function("lust_mir_get_status", Linkage::Import, &status_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let mut print_sig = self.module.make_signature();
            print_sig.params.push(AbiParam::new(types::I64));
            let print_callee = self
                .module
                .declare_function("lust_mir_print_int", Linkage::Import, &print_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            // Declare binop helpers
            let mut binop_sig = self.module.make_signature();
            binop_sig.params.push(AbiParam::new(types::I64));
            binop_sig.params.push(AbiParam::new(types::I64));
            binop_sig.returns.push(AbiParam::new(types::I64));

            let floordiv_callee = self
                .module
                .declare_function("lust_mir_floordiv_int", Linkage::Import, &binop_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let pow_callee = self
                .module
                .declare_function("lust_mir_pow_int", Linkage::Import, &binop_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let div_callee = self
                .module
                .declare_function("lust_mir_div_int", Linkage::Import, &binop_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let rem_callee = self
                .module
                .declare_function("lust_mir_rem_int", Linkage::Import, &binop_sig)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            // Declare builtin call helpers
            let mut sig1 = self.module.make_signature();
            sig1.params.push(AbiParam::new(types::I64));
            sig1.returns.push(AbiParam::new(types::I64));

            let abs_callee = self
                .module
                .declare_function("lust_mir_abs_int", Linkage::Import, &sig1)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let len_callee = self
                .module
                .declare_function("lust_mir_array_len", Linkage::Import, &sig1)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let mut sig2 = self.module.make_signature();
            sig2.params.push(AbiParam::new(types::I64));
            sig2.params.push(AbiParam::new(types::I64));
            sig2.returns.push(AbiParam::new(types::I64));

            let min_callee = self
                .module
                .declare_function("lust_mir_min_int", Linkage::Import, &sig2)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let max_callee = self
                .module
                .declare_function("lust_mir_max_int", Linkage::Import, &sig2)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            let mut sig3 = self.module.make_signature();
            sig3.params.push(AbiParam::new(types::I64));
            sig3.params.push(AbiParam::new(types::I64));
            sig3.params.push(AbiParam::new(types::I64));
            sig3.returns.push(AbiParam::new(types::I64));

            let clamp_callee = self
                .module
                .declare_function("lust_mir_clamp_int", Linkage::Import, &sig3)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;

            {
                let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);
                let alloc_fn = self.module.declare_func_in_func(alloc_callee, &mut builder.func);
                let gas_fn = self.module.declare_func_in_func(gas_callee, &mut builder.func);
                let status_fn = self.module.declare_func_in_func(status_callee, &mut builder.func);
                let print_fn = self.module.declare_func_in_func(print_callee, &mut builder.func);
                let floordiv_fn = self.module.declare_func_in_func(floordiv_callee, &mut builder.func);
                let pow_fn = self.module.declare_func_in_func(pow_callee, &mut builder.func);
                let div_fn = self.module.declare_func_in_func(div_callee, &mut builder.func);
                let rem_fn = self.module.declare_func_in_func(rem_callee, &mut builder.func);
                let emergency_exit_bb = builder.create_block();

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
                                let val = lower_rvalue(rvalue, &mut builder, &local_vars, alloc_fn, floordiv_fn, pow_fn, div_fn, rem_fn)?;
                                builder.def_var(local_vars[dst.0], val);
                            }
                            Statement::Assign(Place::Index(base, idx), rvalue, _) => {
                                let val = lower_rvalue(rvalue, &mut builder, &local_vars, alloc_fn, floordiv_fn, pow_fn, div_fn, rem_fn)?;
                                let base_ptr = lower_place(base, &mut builder, &local_vars);
                                let idx_val = builder.use_var(local_vars[idx.0]);
                                let eight = builder.ins().iconst(types::I64, 8);
                                let byte_idx = builder.ins().imul(idx_val, eight);
                                let sixteen = builder.ins().iconst(types::I64, 16);
                                let offset = builder.ins().iadd(byte_idx, sixteen);
                                let elem_addr = builder.ins().iadd(base_ptr, offset);
                                builder.ins().store(MemFlagsData::trusted(), val, elem_addr, 0);
                            }
                            Statement::Assign(Place::Field(base, field_idx), rvalue, _) => {
                                let val = lower_rvalue(rvalue, &mut builder, &local_vars, alloc_fn, floordiv_fn, pow_fn, div_fn, rem_fn)?;
                                let base_ptr = lower_place(base, &mut builder, &local_vars);
                                builder.ins().store(MemFlagsData::trusted(), val, base_ptr, 16 + (*field_idx as i32) * 8);
                            }
                            Statement::ResourceCheck(ResourceKind::Gas, amt, _) => {
                                let amt_val = builder.ins().iconst(types::I64, *amt as i64);
                                let call_inst = builder.ins().call(gas_fn, &[amt_val]);
                                let rem = builder.inst_results(call_inst)[0];
                                let is_neg = builder.ins().icmp_imm_s(IntCC::SignedLessThan, rem, 0);
                                let cont_bb = builder.create_block();
                                builder.ins().brif(is_neg, emergency_exit_bb, &[], cont_bb, &[]);
                                builder.switch_to_block(cont_bb);
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
                                let c_val = lower_operand(cond, &mut builder, &local_vars)?;
                                builder.ins().brif(c_val, clif_blocks[then_block.0], &[], clif_blocks[else_block.0], &[]);
                            }
                            Terminator::Switch { discr, cases, default, .. } => {
                                let d_val = lower_operand(discr, &mut builder, &local_vars)?;
                                for (val, target_bb) in cases {
                                    let is_eq = builder.ins().icmp_imm_s(IntCC::Equal, d_val, *val);
                                    let next_check = builder.create_block();
                                    builder.ins().brif(is_eq, clif_blocks[target_bb.0], &[], next_check, &[]);
                                    builder.switch_to_block(next_check);
                                }
                                builder.ins().jump(clif_blocks[default.0], &[]);
                            }
                            Terminator::Call { callee, args, destination, success, .. } => {
                                if callee == "print" || callee == "println" {
                                    let arg_val = if let Some(first) = args.first() {
                                        lower_operand(first, &mut builder, &local_vars)?
                                    } else {
                                        builder.ins().iconst(types::I64, 0)
                                    };
                                    builder.ins().call(print_fn, &[arg_val]);
                                    match destination {
                                        Place::Local(loc) => {
                                            let zero = builder.ins().iconst(types::I64, 0);
                                            builder.def_var(local_vars[loc.0], zero);
                                        }
                                        _ => {}
                                    }
                                    builder.ins().jump(clif_blocks[success.0], &[]);
                                } else {
                                    let (callee_func_id, param_count) = match callee.as_str() {
                                        "math.abs" | "abs" => (abs_callee, 1),
                                        "math.min" | "min" => (min_callee, 2),
                                        "math.max" | "max" => (max_callee, 2),
                                        "math.clamp" | "clamp" => (clamp_callee, 3),
                                        "table.len" | "array.len" => (len_callee, 1),
                                        _ => {
                                            if let Some(&fid) = module_funcs.get(callee) {
                                                (fid, args.len())
                                            } else {
                                                let mut call_sig = self.module.make_signature();
                                                for _ in 0..args.len() {
                                                    call_sig.params.push(AbiParam::new(types::I64));
                                                }
                                                call_sig.returns.push(AbiParam::new(types::I64));
                                                let fid = self.module
                                                    .declare_function(callee, Linkage::Import, &call_sig)
                                                    .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}")))?;
                                                (fid, args.len())
                                            }
                                        }
                                    };
                                    let callee_ref = self.module.declare_func_in_func(callee_func_id, &mut builder.func);
                                    let mut arg_vals = Vec::with_capacity(param_count);
                                    for a in args {
                                        arg_vals.push(lower_operand(a, &mut builder, &local_vars)?);
                                    }
                                    while arg_vals.len() < param_count {
                                        arg_vals.push(builder.ins().iconst(types::I64, 0));
                                    }
                                    let call_inst = builder.ins().call(callee_ref, &arg_vals);
                                    let ret_vals = builder.inst_results(call_inst);
                                    if !ret_vals.is_empty() {
                                        let res_val = ret_vals[0];
                                        match destination {
                                            Place::Local(loc) => {
                                                builder.def_var(local_vars[loc.0], res_val);
                                            }
                                            Place::Index(base, idx) => {
                                                let base_ptr = lower_place(base, &mut builder, &local_vars);
                                                let idx_val = builder.use_var(local_vars[idx.0]);
                                                let eight = builder.ins().iconst(types::I64, 8);
                                                let byte_idx = builder.ins().imul(idx_val, eight);
                                                let sixteen = builder.ins().iconst(types::I64, 16);
                                                let offset = builder.ins().iadd(byte_idx, sixteen);
                                                let elem_addr = builder.ins().iadd(base_ptr, offset);
                                                builder.ins().store(MemFlagsData::trusted(), res_val, elem_addr, 0);
                                            }
                                            Place::Field(base, field_idx) => {
                                                let base_ptr = lower_place(base, &mut builder, &local_vars);
                                                builder.ins().store(MemFlagsData::trusted(), res_val, base_ptr, 16 + (*field_idx as i32) * 8);
                                            }
                                        }
                                    }
                                    let stat_call = builder.ins().call(status_fn, &[]);
                                    let stat_val = builder.inst_results(stat_call)[0];
                                    let is_err = builder.ins().icmp_imm_s(IntCC::NotEqual, stat_val, 0);
                                    let call_cont_bb = builder.create_block();
                                    builder.ins().brif(is_err, emergency_exit_bb, &[], call_cont_bb, &[]);
                                    builder.switch_to_block(call_cont_bb);

                                    builder.ins().jump(clif_blocks[success.0], &[]);
                                }
                            }
                            Terminator::Return(opt_op, _) => {
                                if ret_clif_ty != types::I8 || mir_fn.signature.return_type != MirType::Unit {
                                    let ret_val = if let Some(op) = opt_op {
                                        lower_operand(op, &mut builder, &local_vars)?
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

                builder.switch_to_block(emergency_exit_bb);
                if ret_clif_ty != types::I8 || mir_fn.signature.return_type != MirType::Unit {
                    let stat_call = builder.ins().call(status_fn, &[]);
                    let stat_val = builder.inst_results(stat_call)[0];
                    let is_zero = builder.ins().icmp_imm_s(IntCC::Equal, stat_val, 0);
                    let default_err = builder.ins().iconst(types::I64, -(crate::effect_abi::LUST_STATUS_ERR_OUT_OF_GAS as i64));
                    let neg_stat = builder.ins().ineg(stat_val);
                    let err_code = builder.ins().select(is_zero, default_err, neg_stat);
                    builder.ins().return_(&[err_code]);
                } else {
                    builder.ins().return_(&[]);
                }

                builder.seal_all_blocks();
                builder.finalize(self.module.target_config());
            }

            self.module
                .define_function(func_id, &mut self.ctx)
                .map_err(|e| MirCraneliftBackendError::CodegenError(format!("{e}\nCLIF:\n{}", self.ctx.func)))?;

            self.module.clear_context(&mut self.ctx);
            Ok(())
        }
    }

    fn mir_to_clif_type(ty: &MirType) -> types::Type {
        match ty {
            MirType::Int => types::I64,
            MirType::Float => types::F64,
            MirType::Bool => types::I8,
            MirType::Array(_) | MirType::Tuple(_) | MirType::Struct(_) | MirType::Effect { .. } => types::I64,
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
            Place::Field(base, field_idx) => {
                let base_ptr = lower_place(base, builder, local_vars);
                builder.ins().load(types::I64, MemFlagsData::trusted(), base_ptr, 16 + (*field_idx as i32) * 8)
            }
        }
    }

    fn lower_operand(
        op: &Operand,
        builder: &mut FunctionBuilder,
        local_vars: &[Variable],
    ) -> Result<ClifValue, MirCraneliftBackendError> {
        match op {
            Operand::Copy(place) | Operand::Move(place) => Ok(lower_place(place, builder, local_vars)),
            Operand::Constant(c) => match c {
                MirConstant::Int(i) => Ok(builder.ins().iconst(types::I64, *i)),
                MirConstant::Float(f) => Ok(builder.ins().f64const(*f)),
                MirConstant::Bool(b) => Ok(builder.ins().iconst(types::I8, if *b { 1 } else { 0 })),
                MirConstant::Unit => Err(MirCraneliftBackendError::UnlowerableValue(
                    "failed or unlowerable value cannot be evaluated as Unit in Cranelift backend".to_string(),
                )),
                _ => Err(MirCraneliftBackendError::UnsupportedConstant(c.clone())),
            },
        }
    }

    fn lower_rvalue(
        rval: &Rvalue,
        builder: &mut FunctionBuilder,
        local_vars: &[Variable],
        alloc_fn: FuncRef,
        floordiv_fn: FuncRef,
        pow_fn: FuncRef,
        div_fn: FuncRef,
        rem_fn: FuncRef,
    ) -> Result<ClifValue, MirCraneliftBackendError> {
        match rval {
            Rvalue::Use(op) => lower_operand(op, builder, local_vars),
            Rvalue::BinaryOp(binop, lhs, rhs) => {
                let l = lower_operand(lhs, builder, local_vars)?;
                let r = lower_operand(rhs, builder, local_vars)?;
                let val = match binop {
                    MirBinaryOp::Add => builder.ins().iadd(l, r),
                    MirBinaryOp::Sub => builder.ins().isub(l, r),
                    MirBinaryOp::Mul => builder.ins().imul(l, r),
                    MirBinaryOp::Div => {
                        let call = builder.ins().call(div_fn, &[l, r]);
                        builder.inst_results(call)[0]
                    }
                    MirBinaryOp::Mod => {
                        let call = builder.ins().call(rem_fn, &[l, r]);
                        builder.inst_results(call)[0]
                    }
                    MirBinaryOp::FloorDiv => {
                        let call = builder.ins().call(floordiv_fn, &[l, r]);
                        builder.inst_results(call)[0]
                    }
                    MirBinaryOp::Pow => {
                        let call = builder.ins().call(pow_fn, &[l, r]);
                        builder.inst_results(call)[0]
                    }
                    MirBinaryOp::BitAnd => builder.ins().band(l, r),
                    MirBinaryOp::BitOr => builder.ins().bor(l, r),
                    MirBinaryOp::BitXor => builder.ins().bxor(l, r),
                    MirBinaryOp::Shl => {
                        let mask = builder.ins().iconst(types::I64, 63);
                        let sh = builder.ins().band(r, mask);
                        builder.ins().ishl(l, sh)
                    }
                    MirBinaryOp::Shr => {
                        let mask = builder.ins().iconst(types::I64, 63);
                        let sh = builder.ins().band(r, mask);
                        builder.ins().sshr(l, sh)
                    }
                    MirBinaryOp::Eq => builder.ins().icmp(IntCC::Equal, l, r),
                    MirBinaryOp::NotEq => builder.ins().icmp(IntCC::NotEqual, l, r),
                    MirBinaryOp::Lt => builder.ins().icmp(IntCC::SignedLessThan, l, r),
                    MirBinaryOp::LtEq => builder.ins().icmp(IntCC::SignedLessThanOrEqual, l, r),
                    MirBinaryOp::Gt => builder.ins().icmp(IntCC::SignedGreaterThan, l, r),
                    MirBinaryOp::GtEq => builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, l, r),
                    MirBinaryOp::And => builder.ins().band(l, r),
                    MirBinaryOp::Or => builder.ins().bor(l, r),
                };
                Ok(val)
            }
            Rvalue::UnaryOp(unop, op) => {
                let v = lower_operand(op, builder, local_vars)?;
                let val = match unop {
                    MirUnaryOp::Neg => builder.ins().ineg(v),
                    MirUnaryOp::Not => builder.ins().bnot(v),
                    MirUnaryOp::BitNot => builder.ins().bnot(v),
                    MirUnaryOp::Len => {
                        // Offset 0 holds length
                        builder.ins().load(types::I64, MemFlagsData::trusted(), v, 0)
                    }
                };
                Ok(val)
            }
            Rvalue::Aggregate(MirType::Array(_) | MirType::Tuple(_) | MirType::Struct(_), elements) => {
                let len = elements.len() as i64;
                let len_val = builder.ins().iconst(types::I64, len);
                let call_inst = builder.ins().call(alloc_fn, &[len_val]);
                let array_ptr = builder.inst_results(call_inst)[0];
                for (i, elem) in elements.iter().enumerate() {
                    let el_val = lower_operand(elem, builder, local_vars)?;
                    builder.ins().store(MemFlagsData::trusted(), el_val, array_ptr, 16 + (i as i32) * 8);
                }
                Ok(array_ptr)
            }
            _ => Err(MirCraneliftBackendError::UnsupportedConstant(MirConstant::Unit)),
        }
    }
}
