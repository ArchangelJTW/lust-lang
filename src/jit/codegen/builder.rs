use super::*;
use crate::VM;
use hashbrown::HashMap;
impl JitCompiler {
    pub fn new() -> Self {
        Self {
            ops: Assembler::new().unwrap(),
            data: Vec::new(),
            exit_stack: Vec::new(),
            inline_depth: 0,
            specialization_registry: SpecializationRegistry::new(),
            specialized_values: HashMap::new(),
            scalar_registers: HashMap::new(),
            pending_scalar: None,
            hot_rax: None,
            hot_xmm0: None,
            hot_rax_in: None,
            hot_xmm0_in: None,
            trace_start_ip: 0,
            function_mode: false,
            function_alias_params: 0,
            function_borrows: 0,
            function_frame: (0, true),
            function_result: None,
            function_entry_table: 0,
            function_epilogue: None,
            function_self: None,
            function_labels: HashMap::new(),
            exit_is_handoff: false,
            current_fail_ip: None,
            fail_sites: Vec::new(),
            next_specialized_id: 0,
        }
    }

    /// Where the interpreter resumes when a guard that is not a branch
    /// (type, function identity, struct layout) fails: the instruction the
    /// guard was recorded for, so it is re-executed with the value the guard
    /// rejected. Inside an inlined body the ip is the call's, since the
    /// callee frame is torn down on exit; the call then runs in the
    /// interpreter.
    pub(super) fn guard_bailout_ip(&self) -> usize {
        self.current_fail_ip.unwrap_or(self.trace_start_ip)
    }

    pub(super) fn current_exit_label(&self) -> dynasmrt::DynamicLabel {
        *self
            .exit_stack
            .last()
            .expect("JIT exit label stack is empty")
    }

    /// Keep an interned name alive with the code; returns the address a
    /// value holding that name carries (see `Name::inner_ptr`).
    pub(super) fn retain_name(&mut self, text: &str) -> usize {
        let name = crate::bytecode::value::Name::from(text);
        let ptr = name.inner_ptr() as usize;
        self.data.push(JitData::Name(name));
        ptr
    }

    /// Keep a map key alive with the code (a field name looked up in a
    /// map, built once here rather than per lookup).
    pub(super) fn retain_key(&mut self, name: &str) -> *const crate::bytecode::ValueKey {
        let key = Box::new(crate::bytecode::ValueKey::from(name));
        let ptr = key.as_ref() as *const crate::bytecode::ValueKey;
        self.data.push(JitData::Key(key));
        ptr
    }

    /// Keep a call site's fixed facts with the code (see `JitInlineRecord`).
    pub(super) fn retain_call_site(&mut self, site: crate::vm::JitCallSite) -> usize {
        let site = Box::new(site);
        let ptr = site.as_ref() as *const crate::vm::JitCallSite as usize;
        self.data.push(JitData::CallSite(site));
        ptr
    }

    pub(super) fn retain_value(&mut self, value: Value) -> *const Value {
        let value = Box::new(value);
        let ptr = value.as_ref() as *const Value;
        self.data.push(JitData::Value(value));
        ptr
    }

    pub(super) fn retain_string(&mut self, value: &str) -> (*const u8, usize) {
        let value: Box<str> = value.into();
        let result = (value.as_ptr(), value.len());
        self.data.push(JitData::String(value));
        result
    }

    pub(super) fn retain_string_pointers(&mut self, pointers: Vec<*const u8>) -> *const *const u8 {
        let pointers = pointers.into_boxed_slice();
        let ptr = pointers.as_ptr();
        self.data.push(JitData::StringPointers(pointers));
        ptr
    }

    pub(super) fn retain_string_lengths(&mut self, lengths: Vec<usize>) -> *const usize {
        let lengths = lengths.into_boxed_slice();
        let ptr = lengths.as_ptr();
        self.data.push(JitData::StringLengths(lengths));
        ptr
    }

    /// Compile whole-function code (see `jit::function`). `entry_table` is
    /// the address of the compiled-function entry table, which the code
    /// reads at each `CallDirect`.
    pub fn compile_function(
        &mut self,
        trace: &Trace,
        trace_id: TraceId,
        register_count: u8,
        entry_table: usize,
        result_type: Option<ValueType>,
    ) -> Result<CompiledTrace> {
        self.function_mode = true;
        self.function_frame = (register_count, trace.frame_may_own);
        self.function_alias_params = trace.alias_params;
        self.function_borrows = trace
            .borrowed_registers
            .iter()
            .filter(|r| **r < 64)
            .fold(0u64, |mask, r| mask | (1u64 << r));
        self.function_result = result_type;
        self.function_entry_table = entry_table;
        self.function_labels.clear();
        let result = self.compile_trace(trace, trace_id, Vec::new());
        self.function_mode = false;
        result
    }

    pub(super) fn function_label(&mut self, id: usize) -> dynasmrt::DynamicLabel {
        if let Some(label) = self.function_labels.get(&id) {
            return *label;
        }
        let label = self.ops.new_dynamic_label();
        self.function_labels.insert(id, label);
        label
    }

    /// Generic exit with a guard return value (guard_index + 1). Function
    /// code also records where and why it left (see `JIT_EXIT_INFO`), since
    /// the result propagates through native callers unchanged.
    /// Before function code hands its frame to the interpreter: take a
    /// reference count for whatever the borrowed registers hold (see
    /// `Trace::borrowed_registers`), since the interpreter will own them.
    /// A scalar or Nil there costs a tag compare. Clobbers the caller-saved
    /// registers.
    pub(super) fn emit_retain_borrows(&mut self) {
        let mask = self.function_borrows;
        if mask == 0 {
            return;
        }
        let value_size = mem::size_of::<Value>() as i32;
        for reg in 0..64u8 {
            if mask & (1u64 << reg) != 0 {
                let offset = i32::from(reg) * value_size;
                dynasm!(self.ops ; .arch x64 ; lea rsi, [r12 + offset]);
                self.emit_retain_at_rsi();
            }
        }
    }

    pub(super) fn emit_guard_exit(&mut self, guard_return_value: i32) {
        let exit_label = self.current_exit_label();
        if self.function_mode {
            self.emit_retain_borrows();
            let kind = if self.exit_is_handoff {
                jit::EXIT_KIND_HANDOFF
            } else {
                jit::EXIT_KIND_GUARD
            };
            let ip = self.current_fail_ip.unwrap_or(usize::MAX >> 16);
            self.emit_exit_info(ip, kind);
        }
        dynasm!(self.ops
            ; .arch x64
            ; mov eax, DWORD guard_return_value
            ; jmp => exit_label
        );
    }

    /// `JIT_EXIT_INFO = ip | kind << EXIT_KIND_SHIFT`.
    pub(super) fn emit_exit_info(&mut self, ip: usize, kind: usize) {
        let info = (ip & ((1usize << jit::EXIT_KIND_SHIFT) - 1)) | (kind << jit::EXIT_KIND_SHIFT);
        dynasm!(self.ops
            ; .arch x64
            ; mov rcx, QWORD info as _
            ; mov [r13 + jit::EXIT_INFO_OFFSET as i32], rcx
        );
    }

    pub fn compile_trace(
        &mut self,
        trace: &Trace,
        trace_id: TraceId,
        hoisted_constants: Vec<(u8, Value)>,
    ) -> Result<CompiledTrace> {
        self.scalar_registers = trace.entry_scalars.iter().copied().collect();
        self.trace_start_ip = trace.start_ip;
        self.current_fail_ip = None;
        self.fail_sites.clear();
        let stack_size = self.compute_stack_size(trace);
        let mut guards = Vec::new();
        let mut guard_index = 0i32;
        let exit_label = self.ops.new_dynamic_label();
        let fail_label = self.ops.new_dynamic_label();
        let epilogue_label = self.ops.new_dynamic_label();
        self.function_epilogue = Some(epilogue_label);
        self.exit_stack.push(exit_label);
        crate::jit::log(|| format!("🔧 JIT: Emitting prologue with sub rsp, {}", stack_size));
        // Entry: rdi = registers, rsi = VM, rdx = the inline record a native
        // caller pushed for this frame (function mode; null from the
        // interpreter). Traces are only ever entered by the interpreter.
        if self.function_mode {
            let entry = self.ops.new_dynamic_label();
            dynasm!(self.ops ; .arch x64 ; => entry);
            self.function_self = Some((trace.function_idx, entry));
        }
        dynasm!(self.ops
            ; .arch x64
            ; push rbp
            ; mov rbp, rsp
            ; push rbx
            ; push r12
            ; push r13
            ; push r14
            ; push r15
            ; sub rsp, stack_size
            ; mov r12, rdi
            ; mov r13, rsi
        );
        if self.function_mode {
            // rcx = where a native caller wants the result (null from the
            // interpreter, which takes it from the return code).
            dynasm!(self.ops ; .arch x64 ; mov r15, rdx ; mov r14, rcx);
        } else {
            dynasm!(self.ops ; .arch x64 ; xor r15, r15);
        }
        for slot in 0..Self::count_specialized_slots(trace) as i32 {
            let offset = SPECIALIZED_BASE_OFFSET - slot * SPECIALIZED_SLOT_SIZE;
            dynasm!(self.ops
                ; .arch x64
                ; mov QWORD [rbp + offset], 0
                ; mov QWORD [rbp + offset + 8], 0
                ; mov QWORD [rbp + offset + 16], 0
                ; mov QWORD [rbp + offset + 24], 0
            );
        }
        for (dest, value) in &hoisted_constants {
            self.compile_load_const(*dest, value)?;
        }

        // Compile preamble (executed once at trace entry)
        jit::log(|| format!("🔧 JIT: Compiling preamble ({} ops)", trace.preamble.len()));
        self.compile_ops(&trace.preamble, &mut guard_index, &mut guards)?;

        // `>fail` references so far (preamble) bind here: a failed unbox at
        // entry is a bare failure (-1), restarting the iteration; left
        // pending they would bind to the body's first fail stub and resume
        // at that instruction instead.
        dynasm!(self.ops
            ; .arch x64
            ; jmp >preamble_fail_skip
            ; fail:
            ; jmp => fail_label
            ; preamble_fail_skip:
        );

        // Create a loop_start label AFTER preamble, BEFORE loop body
        self.current_fail_ip = None;
        let loop_start_label = self.ops.new_dynamic_label();
        dynasm!(self.ops
            ; .arch x64
            ; => loop_start_label
            ; loop_start:
        );

        // Compile main trace body (the loop)
        let compile_result = self.compile_ops(&trace.ops, &mut guard_index, &mut guards);
        compile_result?;

        if self.function_mode {
            // Every path through function code ends in a `Return`; falling
            // off the end is a compiler bug, reported as a bare failure.
            dynasm!(self.ops
                ; .arch x64
                ; mov eax, DWORD -1
                ; jmp => exit_label
            );
        } else {
            // At end of loop body, jump back to loop_start to loop
            dynasm!(self.ops
                ; .arch x64
                ; jmp => loop_start_label
            );
        }

        // Body exits: preserve the exit code, then turn any inline-call
        // frames into interpreter frames so r12 is the trace's own register
        // array again before the postamble runs (a guard inside an inlined
        // body exits from here with the callee frame still on the stack).
        dynasm!(self.ops
            ; .arch x64
            ; => exit_label
            ; exit:
            // Postamble helpers may overwrite eax. Preserve the exit reason in
            // a callee-saved register until specialized state is materialized.
            ; mov r14d, eax
        );
        self.emit_unwind_inline_frames();

        // Compile postamble (executed once at trace exit)
        jit::log(|| {
            format!(
                "🔧 JIT: Compiling postamble ({} ops)",
                trace.postamble.len()
            )
        });
        self.current_fail_ip = None;
        self.compile_ops(&trace.postamble, &mut guard_index, &mut guards)?;
        self.publish_remaining_specialized(trace)?;

        // Now pop the label stacks after everything is compiled
        self.exit_stack.pop();

        dynasm!(self.ops
            ; .arch x64
            ; mov eax, r14d
        );

        // Epilogue: rsp is recovered from the frame pointer (five pushes sit
        // below the saved rbp), so the exit path is valid regardless of how
        // deep an inline frame we came from. A bare failure (-1) takes the
        // same exit, which unwinds the inline frames.
        dynasm!(self.ops
            ; .arch x64
            ; => epilogue_label
            ; lea rsp, [rbp - 40]
            ; pop r15
            ; pop r14
            ; pop r13
            ; pop r12
            ; pop rbx
            ; pop rbp
            ; ret
            ; => fail_label
            ; fail:
            ; mov eax, DWORD -1
            ; jmp => exit_label
        );
        let ops = mem::replace(&mut self.ops, Assembler::new().unwrap());
        let exec_buffer = ops.finalize().unwrap();
        let entry_point = exec_buffer.ptr(dynasmrt::AssemblyOffset(0));
        let entry: extern "C" fn(*mut Value, *mut VM, *const Function, *mut Value) -> i32 =
            unsafe { mem::transmute(entry_point) };
        #[cfg(feature = "std")]
        {
            if std::env::var("LUST_JIT_DUMP").is_ok() {
                use std::{fs, path::PathBuf};
                let len = exec_buffer.len();
                let bytes = unsafe { std::slice::from_raw_parts(entry_point as *const u8, len) };
                let mut path = PathBuf::from("target");
                let _ = fs::create_dir_all(&path);
                path.push(format!(
                    "jit_trace_{}_{}.bin",
                    trace_id.0,
                    trace.function_idx
                ));
                if let Err(err) = fs::write(&path, bytes) {
                    crate::jit::log(|| {
                        format!("⚠️  JIT: failed to dump trace to {:?}: {}", path, err)
                    });
                } else {
                    crate::jit::log(|| format!("📝 JIT: Dumped trace bytes to {:?}", path));
                }
            }
        }
        let data = mem::take(&mut self.data);
        Ok(CompiledTrace {
            id: trace_id,
            entry,
            _executable: exec_buffer,
            _data: data,
            trace: trace.clone(),
            guards,
            fail_sites: mem::take(&mut self.fail_sites),
            hoisted_constants,
        })
    }

    /// Hand the inline-call frame chain in r15 to the interpreter (see
    /// `jit_materialize_inline_frames`): the callee frames become real
    /// frames and execution resumes inside the innermost one. Leaves r12 =
    /// the trace's own register array and r15 = 0. No-op without a chain.
    /// Only registers are restored here — rsp is recovered from rbp by the
    /// epilogue.
    fn emit_unwind_inline_frames(&mut self) {
        unsafe extern "C" {
            fn jit_materialize_inline_frames(
                vm: *mut VM,
                record: *const u8,
                regs: *mut Value,
            ) -> *mut Value;
        }
        let done = self.ops.new_dynamic_label();
        dynasm!(self.ops
            ; .arch x64
            ; test r15, r15
            ; jz => done
            ; mov rdi, r13
            ; mov rsi, r15
            ; mov rdx, r12
            ; mov rax, QWORD jit_materialize_inline_frames as *const () as _
            ; call rax
            ; mov r12, rax
            ; xor r15, r15
            ; => done
        );
    }

    /// After the postamble: publish and release every specialized slot the
    /// postamble did not rebox. A slot whose register the trace overwrote
    /// is dropped from the recorder's tracking (no `Rebox` is generated for
    /// it), but its unbox still runs on every entry, so left alone it
    /// leaked its copy and kept the array alive. Reboxing an empty slot is
    /// a no-op, so this is safe for the slots the postamble did handle.
    fn publish_remaining_specialized(&mut self, trace: &Trace) -> Result<()> {
        let reboxed: Vec<usize> = trace
            .postamble
            .iter()
            .filter_map(|op| match op {
                TraceOp::Rebox { specialized_id, .. } => Some(*specialized_id),
                _ => None,
            })
            .collect();
        let layouts: Vec<(usize, SpecializedLayout)> = trace
            .preamble
            .iter()
            .chain(trace.ops.iter())
            .filter_map(|op| match op {
                TraceOp::Unbox {
                    specialized_id,
                    layout,
                    ..
                } if !reboxed.contains(specialized_id) => Some((*specialized_id, layout.clone())),
                _ => None,
            })
            .collect();
        let mut done = Vec::new();
        for (id, layout) in layouts {
            if done.contains(&id) || !self.specialized_values.contains_key(&id) {
                continue;
            }
            done.push(id);
            self.compile_rebox(0, id, &layout)?;
        }
        Ok(())
    }

    fn compile_ops(
        &mut self,
        ops: &[TraceOp],
        guard_index: &mut i32,
        guards: &mut Vec<Guard>,
    ) -> Result<()> {
        let mut skip_through: Option<usize> = None;
        // Each segment (preamble, body, postamble, an inlined body) starts
        // at a label or in another frame: nothing is in rax / xmm0.
        self.hot_rax = None;
        self.hot_xmm0 = None;
        for (op_index, op) in ops.iter().enumerate() {
            if skip_through.is_some_and(|last| op_index <= last) {
                continue;
            }
            if let TraceOp::At { ip } = op {
                // Inside an inlined body the markers carry the callee's ips;
                // an exit there resumes inside the callee, whose frame the
                // exit path materializes.
                self.current_fail_ip = Some(*ip);
                continue;
            }
            // What the previous op left in rax / xmm0 is this op's to use.
            self.hot_rax_in = self.hot_rax.take();
            self.hot_xmm0_in = self.hot_xmm0.take();
            // Fusion looks at the next real op; `At` markers are transparent.
            let next_index = (op_index + 1..ops.len())
                .find(|&j| !matches!(ops[j], TraceOp::At { .. }));
            if let Some(next_index) = next_index {
                let next = &ops[next_index];
                if let TraceOp::GuardLoopContinue {
                    condition_register, ..
                } = next
                    && self.scalar_registers.contains_key(condition_register)
                    && Self::register_overwritten_before_read(
                        &ops[next_index + 1..],
                        *condition_register,
                    )
                    && let Some(guard) =
                        self.compile_numeric_comparison_guard(op, next, *guard_index as usize)?
                {
                    guards.push(guard);
                    *guard_index += 1;
                    skip_through = Some(next_index);
                    // The failed result's store is on the exit path only.
                    self.hot_rax = None;
                    self.hot_xmm0 = None;
                    continue;
                }
                if let TraceOp::LoadConst {
                    dest: constant_register,
                    ..
                } = op
                    && self.scalar_registers.contains_key(constant_register)
                    && Self::register_overwritten_before_read(
                        &ops[next_index + 1..],
                        *constant_register,
                    )
                    && self.compile_integer_add_immediate(op, next)?
                {
                    self.update_scalar_registers(next);
                    skip_through = Some(next_index);
                    continue;
                }
                // A constant that stays live is still stored, but the add
                // takes it as an immediate rather than loading it back.
                if let TraceOp::LoadConst {
                    dest: constant_register,
                    value: value @ Value::Int(_),
                } = op
                    && self.scalar_registers.contains_key(constant_register)
                    && matches!(
                        next,
                        TraceOp::Add {
                            dest,
                            lhs,
                            rhs,
                            lhs_type: ValueType::Int,
                            rhs_type: ValueType::Int,
                        } if (lhs == constant_register) != (rhs == constant_register)
                            && dest != constant_register
                    )
                {
                    self.compile_load_const(*constant_register, value)?;
                    self.update_scalar_registers(op);
                    // The constant's store is what rax now holds.
                    self.hot_rax_in = self.hot_rax.take();
                    self.hot_xmm0_in = self.hot_xmm0.take();
                    if self.compile_integer_add_immediate(op, next)? {
                        self.update_scalar_registers(next);
                        skip_through = Some(next_index);
                    }
                    continue;
                }
            }
            match op {
                TraceOp::At { .. } => unreachable!("markers are consumed above"),
                TraceOp::LoadConst { dest, value } => {
                    self.compile_load_const(*dest, value)?;
                }

                TraceOp::Move { dest, src } => {
                    self.compile_move(*dest, *src)?;
                }

                TraceOp::Add {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                } => {
                    self.compile_add_specialized(*dest, *lhs, *rhs, *lhs_type, *rhs_type)?;
                }

                TraceOp::Sub {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                } => {
                    self.compile_sub_specialized(*dest, *lhs, *rhs, *lhs_type, *rhs_type)?;
                }

                TraceOp::Mul {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                } => {
                    self.compile_mul_specialized(*dest, *lhs, *rhs, *lhs_type, *rhs_type)?;
                }

                TraceOp::Div {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                } => {
                    self.compile_div_specialized(*dest, *lhs, *rhs, *lhs_type, *rhs_type)?;
                }

                TraceOp::Mod {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                } => {
                    self.compile_mod_specialized(*dest, *lhs, *rhs, *lhs_type, *rhs_type)?;
                }

                TraceOp::Neg { dest, src } => {
                    self.compile_neg(*dest, *src)?;
                }

                TraceOp::Lt {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                } => {
                    self.compile_lt(*dest, *lhs, *rhs, *lhs_type, *rhs_type)?;
                }

                TraceOp::Le {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                } => {
                    self.compile_le(*dest, *lhs, *rhs, *lhs_type, *rhs_type)?;
                }

                TraceOp::Gt {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                } => {
                    self.compile_gt(*dest, *lhs, *rhs, *lhs_type, *rhs_type)?;
                }

                TraceOp::Ge {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                } => {
                    self.compile_ge(*dest, *lhs, *rhs, *lhs_type, *rhs_type)?;
                }

                TraceOp::Eq {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                } => {
                    self.compile_eq(*dest, *lhs, *rhs, *lhs_type, *rhs_type)?;
                }

                TraceOp::Ne {
                    dest,
                    lhs,
                    rhs,
                    lhs_type,
                    rhs_type,
                } => {
                    self.compile_ne(*dest, *lhs, *rhs, *lhs_type, *rhs_type)?;
                }

                TraceOp::And { dest, lhs, rhs } => {
                    self.compile_and(*dest, *lhs, *rhs)?;
                }

                TraceOp::Or { dest, lhs, rhs } => {
                    self.compile_or(*dest, *lhs, *rhs)?;
                }

                TraceOp::Not { dest, src } => {
                    self.compile_not(*dest, *src)?;
                }

                TraceOp::Concat { dest, lhs, rhs } => {
                    self.compile_concat(*dest, *lhs, *rhs)?;
                }

                TraceOp::GetIndex { dest, array, index } => {
                    self.compile_get_index(*dest, *array, *index)?;
                }

                TraceOp::TryGetIndex { dest, array, index } => {
                    self.compile_try_get_index(*dest, *array, *index)?;
                }

                TraceOp::ArrayIndexOk {
                    value_dest,
                    condition_dest,
                    array,
                    index,
                    value_type,
                } => {
                    self.compile_array_index_ok(
                        *value_dest,
                        *condition_dest,
                        *array,
                        *index,
                        *value_type,
                    )?;
                }

                TraceOp::ArrayLen { dest, array } => {
                    self.compile_array_len(*dest, *array)?;
                }

                TraceOp::GuardStructLayout { register, layout } => {
                    let guard = self.compile_guard_struct_layout(
                        *register,
                        *layout,
                        *guard_index as usize,
                    )?;
                    guards.push(guard);
                    *guard_index += 1;
                }

                TraceOp::GuardNativeFunction { register, function } => {
                    let expected_ptr = function.pointer();
                    crate::jit::log(|| format!("🔒 JIT: guard native reg {}", register));
                    let guard = self.compile_guard_native_function(
                        *register,
                        expected_ptr,
                        function.inner_ptr(),
                        *guard_index as usize,
                    )?;
                    guards.push(guard);
                    *guard_index += 1;
                }

                TraceOp::GuardGlobals { version } => {
                    crate::jit::log(|| format!("🔒 JIT: guard globals version {}", version));
                    let guard = self.compile_guard_globals(*version, *guard_index as usize);
                    guards.push(guard);
                    *guard_index += 1;
                }

                TraceOp::GuardFunction {
                    register,
                    function_idx,
                } => {
                    crate::jit::log(|| {
                        format!(
                            "🔒 JIT: guard function reg {} -> idx {}",
                            register, function_idx
                        )
                    });
                    let guard = self.compile_guard_function(
                        *register,
                        *function_idx,
                        *guard_index as usize,
                    )?;
                    guards.push(guard);
                    *guard_index += 1;
                }

                TraceOp::GuardClosure {
                    register,
                    function_idx,
                    upvalues_ptr,
                } => {
                    crate::jit::log(|| {
                        format!(
                            "🔒 JIT: guard closure reg {} -> idx {}",
                            register, function_idx
                        )
                    });
                    let guard = self.compile_guard_closure(
                        *register,
                        *function_idx,
                        *upvalues_ptr,
                        *guard_index as usize,
                    )?;
                    guards.push(guard);
                    *guard_index += 1;
                }

                TraceOp::CallNative {
                    dest,
                    callee,
                    function,
                    first_arg,
                    arg_count,
                } => {
                    let expected_ptr = function.pointer();
                    self.compile_call_native(*dest, *callee, expected_ptr, *first_arg, *arg_count)?;
                }

                TraceOp::CallFunction {
                    dest,
                    callee,
                    function_idx,
                    first_arg,
                    arg_count,
                    is_closure,
                    upvalues_ptr,
                } => {
                    self.compile_call_function(
                        *dest,
                        *callee,
                        *function_idx,
                        *first_arg,
                        *arg_count,
                        *is_closure,
                        *upvalues_ptr,
                    )?;
                }

                TraceOp::InlineCall {
                    dest,
                    callee,
                    trace,
                } => {
                    // Arguments of known scalar type are copied natively and
                    // seed the callee's environment; the result's type comes
                    // back the same way.
                    let arg_types: Vec<Option<ValueType>> = trace
                        .arg_registers
                        .iter()
                        .map(|reg| self.scalar_registers.get(reg).copied())
                        .collect();
                    let outer_scalar_registers = mem::take(&mut self.scalar_registers);
                    let result =
                        self.compile_inline_call(*dest, *callee, trace, &arg_types, guard_index, guards);
                    self.scalar_registers = outer_scalar_registers;
                    let result_type = result?;
                    self.update_scalar_registers(op);
                    self.pending_scalar = None;
                    if let Some(ty) = result_type {
                        self.scalar_registers.insert(*dest, ty);
                    }
                    if Self::op_may_fail(op) {
                        self.emit_fail_stub();
                    }
                    continue;
                }

                TraceOp::CallMethod {
                    dest,
                    object,
                    method_name,
                    first_arg,
                    arg_count,
                } => {
                    // Optimize common method calls with specialized JIT helpers
                    match (method_name.as_str(), *arg_count) {
                        ("push", 1) => {
                            self.compile_array_push(*object, *first_arg)?;
                        }
                        ("is_some", 0) => {
                            self.compile_enum_is_some(*dest, *object)?;
                        }
                        ("unwrap", 0) => {
                            self.compile_enum_unwrap(*dest, *object)?;
                        }
                        _ => {
                            self.compile_call_method(
                                *dest,
                                *object,
                                method_name,
                                *first_arg,
                                *arg_count,
                            )?;
                        }
                    }
                }

                TraceOp::GetField {
                    dest,
                    object,
                    field_name,
                    field_index,
                    value_type,
                    is_weak,
                } => {
                    self.compile_get_field(
                        *dest,
                        *object,
                        field_name,
                        *field_index,
                        *value_type,
                        *is_weak,
                    )?;
                }

                TraceOp::SetField {
                    object,
                    field_name,
                    value,
                    field_index,
                    value_type,
                    is_weak,
                } => {
                    self.compile_set_field(
                        *object,
                        field_name,
                        *value,
                        *field_index,
                        *value_type,
                        *is_weak,
                    )?;
                }

                TraceOp::NewArray {
                    dest,
                    first_element,
                    count,
                } => {
                    self.compile_new_array(*dest, *first_element, *count)?;
                }

                TraceOp::NewStruct {
                    dest,
                    struct_name,
                    field_names,
                    field_registers,
                } => {
                    self.compile_new_struct(*dest, struct_name, field_names, field_registers)?;
                }

                TraceOp::NewEnumUnit {
                    dest,
                    enum_name,
                    variant_name,
                } => {
                    self.compile_new_enum_unit(*dest, enum_name, variant_name)?;
                }

                TraceOp::NewEnumVariant {
                    dest,
                    enum_name,
                    variant_name,
                    value_registers,
                } => {
                    self.compile_new_enum_variant(*dest, enum_name, variant_name, value_registers)?;
                }

                TraceOp::IsEnumVariant {
                    dest,
                    value,
                    enum_name,
                    variant_name,
                } => {
                    self.compile_is_enum_variant(*dest, *value, enum_name, variant_name)?;
                }

                TraceOp::TypeIs {
                    dest,
                    value,
                    type_name,
                } => {
                    self.compile_type_is(*dest, *value, type_name)?;
                }

                TraceOp::TryCast {
                    dest,
                    value,
                    type_name,
                } => {
                    self.compile_try_cast(*dest, *value, type_name)?;
                }

                TraceOp::GetEnumValue {
                    dest,
                    enum_reg,
                    index,
                } => {
                    self.compile_get_enum_value(*dest, *enum_reg, *index)?;
                }

                TraceOp::BorrowField {
                    dest,
                    object,
                    field_index,
                } => {
                    self.compile_borrow_field(*dest, *object, *field_index)?;
                }

                TraceOp::BorrowEnumValue {
                    dest,
                    enum_reg,
                    index,
                } => {
                    self.compile_borrow_enum_value(*dest, *enum_reg, *index)?;
                }

                TraceOp::Guard {
                    register,
                    expected_type,
                } => {
                    // A guard on a register the static environment already
                    // knows to hold that scalar is redundant: only typed
                    // writes reach the environment.
                    if self.scalar_registers.get(register) != Some(expected_type) {
                        let guard =
                            self.compile_guard(*register, *expected_type, *guard_index as usize)?;
                        guards.push(guard);
                        *guard_index += 1;
                    }
                }

                TraceOp::GuardLoopContinue {
                    condition_register,
                    expect_truthy,
                    bailout_ip,
                } => {
                    let guard = self.compile_truth_guard(
                        *condition_register,
                        *expect_truthy,
                        *bailout_ip,
                        *guard_index as usize,
                    )?;
                    guards.push(guard);
                    *guard_index += 1;
                }

                TraceOp::NestedLoopCall {
                    function_idx,
                    loop_start_ip,
                    bailout_ip,
                    resume_ip,
                } => {
                    let guard = self.compile_nested_loop_call(
                        *function_idx,
                        *loop_start_ip,
                        *bailout_ip,
                        *resume_ip,
                        *guard_index as usize,
                    );
                    guards.push(guard);
                    *guard_index += 1;
                }

                TraceOp::Unbox {
                    specialized_id,
                    source_reg,
                    layout,
                } => {
                    self.compile_unbox(*specialized_id, *source_reg, layout)?;
                }

                TraceOp::Rebox {
                    dest_reg,
                    specialized_id,
                    layout,
                } => {
                    self.compile_rebox(*dest_reg, *specialized_id, layout)?;
                }

                TraceOp::DropSpecialized {
                    specialized_id,
                    layout,
                } => {
                    self.compile_drop_specialized(*specialized_id, layout)?;
                }

                TraceOp::SpecializedOp { op, operands } => {
                    self.compile_specialized_op(op, operands)?;
                }

                TraceOp::Return { value } => {
                    if self.function_mode && self.inline_depth == 0 {
                        self.compile_function_return(*value)?;
                    }
                }

                TraceOp::Label { id, scalars } => {
                    let label = self.function_label(*id);
                    dynasm!(self.ops ; .arch x64 ; => label);
                    self.scalar_registers = scalars.iter().copied().collect();
                    // Control also arrives here by jump.
                    self.hot_rax_in = None;
                    self.hot_xmm0_in = None;
                }

                TraceOp::Jump { label } => {
                    let label = self.function_label(*label);
                    dynasm!(self.ops ; .arch x64 ; jmp => label);
                }

                TraceOp::BranchIf {
                    condition_register,
                    expect_truthy,
                    label,
                } => {
                    self.compile_branch_if(*condition_register, *expect_truthy, *label);
                }

                TraceOp::CallDirect {
                    dest,
                    callee,
                    receiver,
                    function_idx,
                    first_arg,
                    arg_count,
                    callee_registers,
                    call_ip,
                    resume_ip,
                    alias_mask,
                    result_type,
                } => {
                    self.compile_call_direct(
                        *dest,
                        *callee,
                        *receiver,
                        *function_idx,
                        *first_arg,
                        *arg_count,
                        *callee_registers,
                        *call_ip,
                        *resume_ip,
                        *alias_mask,
                        *result_type,
                        guard_index,
                        guards,
                    )?;
                }
            }
            self.update_scalar_registers(op);
            if let Some((reg, ty)) = self.pending_scalar.take() {
                self.scalar_registers.insert(reg, ty);
            }
            if !self.result_stays_hot(op) {
                self.hot_rax = None;
                self.hot_xmm0 = None;
            }
            if Self::op_may_fail(op) {
                self.emit_fail_stub();
            }
        }

        Ok(())
    }

    /// Ops whose scalar store (`store_from_rax` / `store_xmm0_as_float`)
    /// is the last instruction they emit on every path that continues, so
    /// the payload is still in rax / xmm0 for the next op. Anything else
    /// — a runtime merge after the store (a generic arithmetic path, an
    /// `ArrayIndexOk` slow path), an inlined body, a call — forgets it.
    /// The fail stub emitted after an op is jumped over on the normal
    /// path and does not touch either register.
    fn result_stays_hot(&self, op: &TraceOp) -> bool {
        let numeric = |ty: ValueType| matches!(ty, ValueType::Int | ValueType::Float);
        match op {
            TraceOp::LoadConst {
                value: Value::Int(_) | Value::Float(_) | Value::Bool(_),
                ..
            } => true,
            TraceOp::Move { src, .. } => self.scalar_registers.get(src).is_some_and(|ty| {
                matches!(ty, ValueType::Int | ValueType::Float | ValueType::Bool)
            }),
            TraceOp::Add {
                lhs_type, rhs_type, ..
            }
            | TraceOp::Sub {
                lhs_type, rhs_type, ..
            }
            | TraceOp::Mul {
                lhs_type, rhs_type, ..
            }
            | TraceOp::Div {
                lhs_type, rhs_type, ..
            }
            | TraceOp::Mod {
                lhs_type, rhs_type, ..
            } => numeric(*lhs_type) && numeric(*rhs_type),
            TraceOp::Neg { src, .. } => self
                .scalar_registers
                .get(src)
                .is_some_and(|ty| numeric(*ty)),
            TraceOp::Lt { .. }
            | TraceOp::Le { .. }
            | TraceOp::Gt { .. }
            | TraceOp::Ge { .. }
            | TraceOp::Eq { .. }
            | TraceOp::Ne { .. }
            | TraceOp::And { .. }
            | TraceOp::Or { .. }
            | TraceOp::Not { .. }
            | TraceOp::IsEnumVariant { .. }
            | TraceOp::TypeIs { .. }
            | TraceOp::ArrayLen { .. }
            | TraceOp::GetField { .. }
            | TraceOp::CallDirect { .. } => true,
            _ => false,
        }
    }

    /// After an op that can branch to `>fail`: bind those branches to a stub
    /// that exits with this op's fail-site code, so the interpreter resumes
    /// at the instruction the op came from (inside an inlined body, in the
    /// callee frame the exit path materializes). Without a known ip the
    /// branches fall through to the trace's generic `fail:` (-1).
    pub(super) fn emit_fail_stub(&mut self) {
        let Some(ip) = self.current_fail_ip else {
            return;
        };
        let code = -((self.fail_sites.len() as i32) + 2);
        self.fail_sites.push(ip);
        let exit_label = self.current_exit_label();
        dynasm!(self.ops
            ; .arch x64
            ; jmp >fail_stub_skip
            ; fail:
        );
        if self.function_mode {
            self.emit_retain_borrows();
            self.emit_exit_info(ip, jit::EXIT_KIND_FAIL);
        }
        dynasm!(self.ops
            ; .arch x64
            ; mov eax, DWORD code
            ; jmp => exit_label
            ; fail_stub_skip:
        );
    }

    /// Ops that can branch to `>fail`.
    fn op_may_fail(op: &TraceOp) -> bool {
        matches!(
            op,
            TraceOp::LoadConst { .. }
                | TraceOp::Div { .. }
                | TraceOp::Mod { .. }
                | TraceOp::Concat { .. }
                | TraceOp::GetIndex { .. }
                | TraceOp::TryGetIndex { .. }
                | TraceOp::ArrayIndexOk { .. }
                | TraceOp::ArrayLen { .. }
                | TraceOp::CallNative { .. }
                | TraceOp::CallFunction { .. }
                | TraceOp::InlineCall { .. }
                | TraceOp::CallDirect { .. }
                | TraceOp::CallMethod { .. }
                | TraceOp::GetField { .. }
                | TraceOp::SetField { .. }
                | TraceOp::NewArray { .. }
                | TraceOp::NewStruct { .. }
                | TraceOp::NewEnumUnit { .. }
                | TraceOp::NewEnumVariant { .. }
                | TraceOp::TryCast { .. }
                | TraceOp::GetEnumValue { .. }
                | TraceOp::BorrowField { .. }
                | TraceOp::BorrowEnumValue { .. }
                | TraceOp::Unbox { .. }
                | TraceOp::Rebox { .. }
                | TraceOp::DropSpecialized { .. }
                | TraceOp::SpecializedOp { .. }
        ) && !matches!(
            op,
            TraceOp::LoadConst {
                value: Value::Int(_) | Value::Float(_) | Value::Bool(_),
                ..
            }
        )
    }

    fn compile_integer_add_immediate(
        &mut self,
        load: &TraceOp,
        arithmetic: &TraceOp,
    ) -> Result<bool> {
        let TraceOp::LoadConst {
            dest: constant_register,
            value: Value::Int(immediate),
        } = load
        else {
            return Ok(false);
        };
        let TraceOp::Add {
            dest,
            lhs,
            rhs,
            lhs_type: ValueType::Int,
            rhs_type: ValueType::Int,
        } = arithmetic
        else {
            return Ok(false);
        };
        let source = if lhs == constant_register && rhs != constant_register {
            *rhs
        } else if rhs == constant_register && lhs != constant_register {
            *lhs
        } else {
            return Ok(false);
        };

        self.operand_rax(source);
        if *immediate == 1 {
            dynasm!(self.ops ; .arch x64 ; inc rax);
        } else if *immediate == -1 {
            dynasm!(self.ops ; .arch x64 ; dec rax);
        } else if let Ok(immediate) = i32::try_from(*immediate) {
            dynasm!(self.ops ; .arch x64 ; add rax, immediate);
        } else {
            dynasm!(self.ops
                ; .arch x64
                ; mov rcx, QWORD *immediate
                ; add rax, rcx
            );
        }
        self.store_from_rax(*dest, ValueTag::Int.as_u8());
        Ok(true)
    }

    fn compile_numeric_comparison_guard(
        &mut self,
        comparison: &TraceOp,
        guard: &TraceOp,
        guard_index: usize,
    ) -> Result<Option<Guard>> {
        let (condition_register, lhs, rhs, lhs_type, rhs_type, comparison_kind) = match comparison {
            TraceOp::Lt {
                dest,
                lhs,
                rhs,
                lhs_type,
                rhs_type,
            } => (*dest, *lhs, *rhs, *lhs_type, *rhs_type, 0),
            TraceOp::Le {
                dest,
                lhs,
                rhs,
                lhs_type,
                rhs_type,
            } => (*dest, *lhs, *rhs, *lhs_type, *rhs_type, 1),
            TraceOp::Gt {
                dest,
                lhs,
                rhs,
                lhs_type,
                rhs_type,
            } => (*dest, *lhs, *rhs, *lhs_type, *rhs_type, 2),
            TraceOp::Ge {
                dest,
                lhs,
                rhs,
                lhs_type,
                rhs_type,
            } => (*dest, *lhs, *rhs, *lhs_type, *rhs_type, 3),
            _ => return Ok(None),
        };
        if !matches!(lhs_type, ValueType::Int | ValueType::Float)
            || !matches!(rhs_type, ValueType::Int | ValueType::Float)
        {
            return Ok(None);
        }
        let TraceOp::GuardLoopContinue {
            condition_register: guarded_register,
            expect_truthy,
            bailout_ip,
        } = guard
        else {
            return Ok(None);
        };
        if condition_register != *guarded_register {
            return Ok(None);
        }

        let guard_ok = self.ops.new_dynamic_label();
        if self.load_numeric_comparison_operands(lhs, rhs, lhs_type, rhs_type) {
            let guard_fail = self.ops.new_dynamic_label();
            dynasm!(self.ops ; .arch x64 ; ucomisd xmm0, xmm1);
            // Every ordered comparison is false for NaN. CF/ZF alone would
            // incorrectly treat unordered operands as less-than or equal.
            if *expect_truthy {
                dynasm!(self.ops ; .arch x64 ; jp =>guard_fail);
            } else {
                dynasm!(self.ops ; .arch x64 ; jp =>guard_ok);
            }
            match (comparison_kind, *expect_truthy) {
                (0, true) | (3, false) => dynasm!(self.ops ; .arch x64 ; jb =>guard_ok),
                (1, true) | (2, false) => dynasm!(self.ops ; .arch x64 ; jbe =>guard_ok),
                (2, true) | (1, false) => dynasm!(self.ops ; .arch x64 ; ja =>guard_ok),
                (3, true) | (0, false) => dynasm!(self.ops ; .arch x64 ; jae =>guard_ok),
                _ => unreachable!(),
            }
            dynasm!(self.ops ; .arch x64 ; =>guard_fail);
        } else {
            dynasm!(self.ops ; .arch x64 ; cmp rax, rcx);
            match (comparison_kind, *expect_truthy) {
                (0, true) | (3, false) => dynasm!(self.ops ; .arch x64 ; jl =>guard_ok),
                (1, true) | (2, false) => dynasm!(self.ops ; .arch x64 ; jle =>guard_ok),
                (2, true) | (1, false) => dynasm!(self.ops ; .arch x64 ; jg =>guard_ok),
                (3, true) | (0, false) => dynasm!(self.ops ; .arch x64 ; jge =>guard_ok),
                _ => unreachable!(),
            }
        }

        // The interpreter resumes at the branch bytecode, which reads this
        // register. Materialize only the uncommon failed result.
        let failed_value = i64::from(!*expect_truthy);
        dynasm!(self.ops ; .arch x64 ; mov rax, QWORD failed_value);
        self.store_from_rax(condition_register, ValueTag::Bool.as_u8());
        let guard_return_value = (guard_index + 1) as i32;
        dynasm!(self.ops
            ; .arch x64
        );
        self.emit_guard_exit(guard_return_value);
        dynasm!(self.ops
            ; .arch x64
            ; =>guard_ok
        );

        Ok(Some(Guard {
            index: guard_index,
            bailout_ip: *bailout_ip,
            kind: if *expect_truthy {
                GuardKind::Truthy {
                    register: condition_register,
                }
            } else {
                GuardKind::Falsy {
                    register: condition_register,
                }
            },
            fail_count: 0,
        }))
    }

    fn register_overwritten_before_read(ops: &[TraceOp], register: u8) -> bool {
        for op in ops {
            if Self::op_reads_register(op, register) {
                return false;
            }
            if Self::op_may_exit(op) {
                return false;
            }
            if Self::op_writes_register(op, register) {
                return true;
            }
        }
        false
    }

    fn op_may_exit(op: &TraceOp) -> bool {
        matches!(
            op,
            TraceOp::Div { .. }
                | TraceOp::Mod { .. }
                | TraceOp::Concat { .. }
                | TraceOp::GetIndex { .. }
                | TraceOp::TryGetIndex { .. }
                | TraceOp::ArrayIndexOk { .. }
                | TraceOp::ArrayLen { .. }
                | TraceOp::GuardNativeFunction { .. }
                | TraceOp::GuardGlobals { .. }
                | TraceOp::GuardStructLayout { .. }
                | TraceOp::GuardFunction { .. }
                | TraceOp::GuardClosure { .. }
                | TraceOp::CallNative { .. }
                | TraceOp::CallFunction { .. }
                | TraceOp::InlineCall { .. }
                | TraceOp::CallMethod { .. }
                | TraceOp::GetField { .. }
                | TraceOp::SetField { .. }
                | TraceOp::NewArray { .. }
                | TraceOp::NewStruct { .. }
                | TraceOp::NewEnumUnit { .. }
                | TraceOp::NewEnumVariant { .. }
                | TraceOp::IsEnumVariant { .. }
                | TraceOp::TypeIs { .. }
                | TraceOp::TryCast { .. }
                | TraceOp::GetEnumValue { .. }
                | TraceOp::BorrowField { .. }
                | TraceOp::BorrowEnumValue { .. }
                | TraceOp::Guard { .. }
                | TraceOp::GuardLoopContinue { .. }
                | TraceOp::NestedLoopCall { .. }
                | TraceOp::Return { .. }
                | TraceOp::Unbox { .. }
                | TraceOp::Rebox { .. }
                | TraceOp::DropSpecialized { .. }
                | TraceOp::SpecializedOp { .. }
                | TraceOp::Label { .. }
                | TraceOp::Jump { .. }
                | TraceOp::BranchIf { .. }
                | TraceOp::CallDirect { .. }
        )
    }

    fn op_reads_register(op: &TraceOp, register: u8) -> bool {
        let in_args =
            |first: u8, count: u8| register >= first && register < first.saturating_add(count);
        match op {
            TraceOp::At { .. } | TraceOp::LoadConst { .. } | TraceOp::GuardGlobals { .. } => false,
            TraceOp::Move { src, .. } | TraceOp::Neg { src, .. } => *src == register,
            TraceOp::Add { lhs, rhs, .. }
            | TraceOp::Sub { lhs, rhs, .. }
            | TraceOp::Mul { lhs, rhs, .. }
            | TraceOp::Div { lhs, rhs, .. }
            | TraceOp::Mod { lhs, rhs, .. }
            | TraceOp::Eq { lhs, rhs, .. }
            | TraceOp::Ne { lhs, rhs, .. }
            | TraceOp::Lt { lhs, rhs, .. }
            | TraceOp::Le { lhs, rhs, .. }
            | TraceOp::Gt { lhs, rhs, .. }
            | TraceOp::Ge { lhs, rhs, .. }
            | TraceOp::And { lhs, rhs, .. }
            | TraceOp::Or { lhs, rhs, .. }
            | TraceOp::Concat { lhs, rhs, .. } => *lhs == register || *rhs == register,
            TraceOp::Not { src, .. } => *src == register,
            TraceOp::GetIndex { array, index, .. }
            | TraceOp::TryGetIndex { array, index, .. }
            | TraceOp::ArrayIndexOk { array, index, .. } => {
                *array == register || *index == register
            }
            TraceOp::ArrayLen { array, .. } => *array == register,
            TraceOp::GuardNativeFunction { register: source, .. }
            | TraceOp::GuardStructLayout { register: source, .. }
            | TraceOp::GuardFunction { register: source, .. }
            | TraceOp::GuardClosure { register: source, .. }
            | TraceOp::Guard {
                register: source, ..
            } => *source == register,
            TraceOp::CallNative {
                callee,
                first_arg,
                arg_count,
                ..
            }
            | TraceOp::CallFunction {
                callee,
                first_arg,
                arg_count,
                ..
            } => *callee == register || in_args(*first_arg, *arg_count),
            TraceOp::InlineCall { callee, trace, .. } => {
                *callee == register || trace.arg_registers.contains(&register)
            }
            TraceOp::CallMethod {
                object,
                first_arg,
                arg_count,
                ..
            } => *object == register || in_args(*first_arg, *arg_count),
            TraceOp::GetField { object, .. } => *object == register,
            TraceOp::SetField { object, value, .. } => {
                *object == register || *value == register
            }
            TraceOp::NewArray {
                first_element,
                count,
                ..
            } => in_args(*first_element, *count),
            TraceOp::NewStruct {
                field_registers, ..
            } => field_registers.contains(&register),
            TraceOp::NewEnumVariant {
                value_registers, ..
            } => value_registers.contains(&register),
            TraceOp::IsEnumVariant { value, .. }
            | TraceOp::TypeIs { value, .. }
            | TraceOp::TryCast { value, .. } => *value == register,
            TraceOp::GetEnumValue { enum_reg, .. } => *enum_reg == register,
            TraceOp::BorrowField { object, .. } => *object == register,
            TraceOp::BorrowEnumValue { enum_reg, .. } => *enum_reg == register,
            TraceOp::GuardLoopContinue {
                condition_register, ..
            } => *condition_register == register,
            TraceOp::Return { value } => *value == Some(register),
            TraceOp::Unbox { source_reg, .. } => *source_reg == register,
            TraceOp::SpecializedOp { operands, .. } => operands.iter().any(|operand| {
                matches!(operand, crate::jit::trace::Operand::Register(source) if *source == register)
            }),
            TraceOp::NestedLoopCall { .. } => true,
            TraceOp::NewEnumUnit { .. } | TraceOp::Rebox { .. } | TraceOp::DropSpecialized { .. } => {
                false
            }
            TraceOp::Label { .. } | TraceOp::Jump { .. } => false,
            TraceOp::BranchIf {
                condition_register, ..
            } => *condition_register == register,
            TraceOp::CallDirect {
                callee,
                receiver,
                first_arg,
                arg_count,
                ..
            } => {
                *callee == register
                    || *receiver == Some(register)
                    || in_args(*first_arg, *arg_count)
            }
        }
    }

    fn op_writes_register(op: &TraceOp, register: u8) -> bool {
        match op {
            TraceOp::ArrayIndexOk {
                value_dest,
                condition_dest,
                ..
            } => *value_dest == register || *condition_dest == register,
            TraceOp::Rebox { dest_reg, .. } => *dest_reg == register,
            TraceOp::LoadConst { dest, .. }
            | TraceOp::Move { dest, .. }
            | TraceOp::Add { dest, .. }
            | TraceOp::Sub { dest, .. }
            | TraceOp::Mul { dest, .. }
            | TraceOp::Div { dest, .. }
            | TraceOp::Mod { dest, .. }
            | TraceOp::Neg { dest, .. }
            | TraceOp::Eq { dest, .. }
            | TraceOp::Ne { dest, .. }
            | TraceOp::Lt { dest, .. }
            | TraceOp::Le { dest, .. }
            | TraceOp::Gt { dest, .. }
            | TraceOp::Ge { dest, .. }
            | TraceOp::And { dest, .. }
            | TraceOp::Or { dest, .. }
            | TraceOp::Not { dest, .. }
            | TraceOp::Concat { dest, .. }
            | TraceOp::GetIndex { dest, .. }
            | TraceOp::TryGetIndex { dest, .. }
            | TraceOp::ArrayLen { dest, .. }
            | TraceOp::CallNative { dest, .. }
            | TraceOp::CallFunction { dest, .. }
            | TraceOp::InlineCall { dest, .. }
            | TraceOp::CallDirect { dest, .. }
            | TraceOp::CallMethod { dest, .. }
            | TraceOp::GetField { dest, .. }
            | TraceOp::NewArray { dest, .. }
            | TraceOp::NewStruct { dest, .. }
            | TraceOp::NewEnumUnit { dest, .. }
            | TraceOp::NewEnumVariant { dest, .. }
            | TraceOp::IsEnumVariant { dest, .. }
            | TraceOp::TypeIs { dest, .. }
            | TraceOp::TryCast { dest, .. }
            | TraceOp::GetEnumValue { dest, .. }
            | TraceOp::BorrowField { dest, .. }
            | TraceOp::BorrowEnumValue { dest, .. } => *dest == register,
            _ => false,
        }
    }

    fn update_scalar_registers(&mut self, op: &TraceOp) {
        if matches!(op, TraceOp::At { .. }) {
            return;
        }
        let scalar_type = |ty: ValueType| {
            matches!(
                ty,
                ValueType::Bool | ValueType::Int | ValueType::Float | ValueType::Plain
            )
            .then_some(ty)
        };
        let set = |registers: &mut HashMap<u8, ValueType>, register, ty| {
            if let Some(ty) = ty {
                registers.insert(register, ty);
            } else {
                registers.remove(&register);
            }
        };

        match op {
            TraceOp::At { .. } => {}
            TraceOp::LoadConst { dest, value } => {
                let ty = match value {
                    Value::Bool(_) => Some(ValueType::Bool),
                    Value::Int(_) => Some(ValueType::Int),
                    Value::Float(_) => Some(ValueType::Float),
                    // A function index or Nil: nothing to drop, no particular
                    // type.
                    Value::Function(_) | Value::Nil => Some(ValueType::Plain),
                    _ => None,
                };
                set(&mut self.scalar_registers, *dest, ty);
            }
            TraceOp::Move { dest, src } => {
                // A copy of a borrow is a clone the destination owns; a
                // register that ever holds a borrow is read as one.
                let ty = if *src < 64 && self.function_borrows & (1u64 << src) != 0 {
                    None
                } else {
                    self.scalar_registers.get(src).copied()
                };
                set(&mut self.scalar_registers, *dest, ty);
            }
            TraceOp::Add {
                dest,
                lhs_type,
                rhs_type,
                ..
            }
            | TraceOp::Sub {
                dest,
                lhs_type,
                rhs_type,
                ..
            }
            | TraceOp::Mul {
                dest,
                lhs_type,
                rhs_type,
                ..
            }
            | TraceOp::Div {
                dest,
                lhs_type,
                rhs_type,
                ..
            } => {
                let ty = match (*lhs_type, *rhs_type) {
                    (ValueType::Int, ValueType::Int) => Some(ValueType::Int),
                    (ValueType::Int | ValueType::Float, ValueType::Int | ValueType::Float) => {
                        Some(ValueType::Float)
                    }
                    _ => None,
                };
                set(&mut self.scalar_registers, *dest, ty);
            }
            TraceOp::Mod {
                dest,
                lhs_type,
                rhs_type,
                ..
            } => {
                let ty = match (*lhs_type, *rhs_type) {
                    (ValueType::Int, ValueType::Int) => Some(ValueType::Int),
                    _ => None,
                };
                set(&mut self.scalar_registers, *dest, ty);
            }
            TraceOp::Neg { dest, src } => {
                let ty = self.scalar_registers.get(src).copied();
                set(&mut self.scalar_registers, *dest, ty);
            }
            TraceOp::Eq { dest, .. }
            | TraceOp::Ne { dest, .. }
            | TraceOp::Lt { dest, .. }
            | TraceOp::Le { dest, .. }
            | TraceOp::Gt { dest, .. }
            | TraceOp::Ge { dest, .. }
            | TraceOp::And { dest, .. }
            | TraceOp::Or { dest, .. }
            | TraceOp::Not { dest, .. }
            | TraceOp::IsEnumVariant { dest, .. }
            | TraceOp::TypeIs { dest, .. } => {
                self.scalar_registers.insert(*dest, ValueType::Bool);
            }
            TraceOp::ArrayLen { dest, .. } => {
                self.scalar_registers.insert(*dest, ValueType::Int);
            }
            TraceOp::GetField { dest, .. } => {
                self.scalar_registers.remove(dest);
            }
            TraceOp::Guard {
                register,
                expected_type,
            } => {
                set(
                    &mut self.scalar_registers,
                    *register,
                    scalar_type(*expected_type),
                );
            }
            TraceOp::ArrayIndexOk {
                value_dest,
                condition_dest,
                ..
            } => {
                self.scalar_registers.remove(value_dest);
                self.scalar_registers
                    .insert(*condition_dest, ValueType::Bool);
            }
            TraceOp::Concat { dest, .. }
            | TraceOp::GetIndex { dest, .. }
            | TraceOp::TryGetIndex { dest, .. }
            | TraceOp::CallNative { dest, .. }
            | TraceOp::CallFunction { dest, .. }
            | TraceOp::InlineCall { dest, .. }
            | TraceOp::CallMethod { dest, .. }
            | TraceOp::NewArray { dest, .. }
            | TraceOp::NewStruct { dest, .. }
            | TraceOp::NewEnumUnit { dest, .. }
            | TraceOp::NewEnumVariant { dest, .. }
            | TraceOp::TryCast { dest, .. }
            | TraceOp::GetEnumValue { dest, .. } => {
                self.scalar_registers.remove(dest);
            }
            // A borrow owns nothing; the value is of no particular type.
            TraceOp::BorrowField { dest, .. } | TraceOp::BorrowEnumValue { dest, .. } => {
                self.scalar_registers.insert(*dest, ValueType::Plain);
            }
            TraceOp::Rebox { dest_reg, .. } => {
                self.scalar_registers.remove(dest_reg);
            }
            TraceOp::SpecializedOp { operands, .. } => {
                for operand in operands {
                    if let crate::jit::trace::Operand::Register(register) = operand {
                        self.scalar_registers.remove(register);
                    }
                }
            }
            TraceOp::SetField { .. }
            | TraceOp::GuardNativeFunction { .. }
            | TraceOp::GuardGlobals { .. }
            | TraceOp::GuardStructLayout { .. }
            | TraceOp::GuardFunction { .. }
            | TraceOp::GuardClosure { .. }
            | TraceOp::GuardLoopContinue { .. }
            | TraceOp::Return { .. }
            | TraceOp::Unbox { .. }
            | TraceOp::DropSpecialized { .. } => {}
            // The inner loop may have written any register.
            TraceOp::NestedLoopCall { .. } => self.scalar_registers.clear(),
            // Set by the `Label` arm itself.
            TraceOp::Label { .. } | TraceOp::Jump { .. } | TraceOp::BranchIf { .. } => {}
            TraceOp::CallDirect {
                dest, result_type, ..
            } => set(&mut self.scalar_registers, *dest, *result_type),
        }
    }

    fn compute_stack_size(&self, trace: &Trace) -> i32 {
        let specialized_slots = Self::count_specialized_slots(trace) as i32;
        let specialized_bytes =
            SPECIALIZED_STACK_BASE + (specialized_slots * SPECIALIZED_SLOT_SIZE);
        // Function code pays for its local area on every call, so it takes
        // only what its specialized values need.
        let mut size = if self.function_mode {
            specialized_bytes
        } else {
            MIN_JIT_STACK_SIZE.max(specialized_bytes)
        };
        let remainder = size % 16;
        if remainder != 8 {
            size += (8 - remainder + 16) % 16;
        }
        crate::jit::log(|| {
            format!(
                "🧮 JIT: Trace requires {} specialized slots → stack {} bytes",
                specialized_slots, size
            )
        });
        size
    }

    /// Distinct specialized ids: each gets one slot, however many times an
    /// unrolled body unboxes it.
    fn count_specialized_slots(trace: &Trace) -> usize {
        let mut ids: Vec<usize> = trace
            .preamble
            .iter()
            .chain(trace.ops.iter())
            .chain(trace.postamble.iter())
            .filter_map(|op| match op {
                TraceOp::Unbox { specialized_id, .. } => Some(*specialized_id),
                _ => None,
            })
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids.len()
    }

    /// Returns the scalar type of the value delivered to `dest`, when the
    /// callee's environment proves one.
    fn compile_inline_call(
        &mut self,
        dest: u8,
        callee: u8,
        trace: &InlineTrace,
        arg_types: &[Option<ValueType>],
        guard_index: &mut i32,
        guards: &mut Vec<Guard>,
    ) -> Result<Option<ValueType>> {
        self.inline_depth += 1;
        let result = (|| -> Result<Option<ValueType>> {
            if trace.register_count == 0 {
                crate::jit::log(|| {
                    format!(
                        "⚠️  JIT: Inline fallback for func {} (no registers)",
                        trace.function_idx
                    )
                });
                self.compile_call_function(
                    dest,
                    callee,
                    trace.function_idx,
                    trace.first_arg,
                    trace.arg_count,
                    trace.is_closure,
                    trace.upvalues_ptr,
                )?;
                return Ok(None);
            }

            crate::jit::log(|| {
                format!(
                    "✨ JIT: Inlining call to func {} into register R{}",
                    trace.function_idx, dest
                )
            });

            let value_size = mem::size_of::<Value>() as i32;
            let frame_value_count = trace.register_count as i32;
            // Round the frame up so rsp stays 16-byte aligned.
            let frame_size = (frame_value_count * value_size + 15) & !15;
            let metadata_size = INLINE_METADATA_SIZE;
            let inline_end = self.ops.new_dynamic_label();
            unsafe extern "C" {
                fn jit_move_safe(src_ptr: *const Value, dest_ptr: *mut Value) -> u8;
                fn jit_drop_values(values: *mut Value, len: usize);
            }

            // Push the inline record (see `JitInlineRecord`). It is linked
            // into the unwind chain only once the frame below it is fully
            // built, so a failure while building it exits cleanly.
            let caller_resume_ip = self.current_fail_ip.map_or(0, |ip| ip + trace.resume_offset);
            // An argument of unknown type whose callee register the body
            // never writes is aliased: its bits are copied without touching
            // the reference count, and the register is neither dropped at
            // teardown nor moved out on exit (the record's mask tells
            // `jit_materialize_inline_frames` to clone it instead). The
            // caller's register keeps the value alive throughout, since the
            // body only ever writes its own frame.
            let alias_mask: u64 = trace
                .arg_registers
                .iter()
                .enumerate()
                .filter(|(arg_index, _)| {
                    !matches!(
                        arg_types.get(*arg_index).copied().flatten(),
                        Some(ValueType::Int | ValueType::Bool | ValueType::Float)
                    ) && *arg_index < 64
                        && !trace
                            .body
                            .iter()
                            .any(|op| Self::op_writes_register(op, *arg_index as u8))
                })
                .map(|(arg_index, _)| 1u64 << arg_index)
                .fold(0, |mask, bit| mask | bit);
            let site = self.retain_call_site(crate::vm::JitCallSite {
                value_count: frame_value_count as usize,
                alias_mask: alias_mask as usize,
                borrow_mask: self.function_borrows as usize,
                function_idx: trace.function_idx,
                return_dest: dest as usize,
                callee_reg: callee as usize,
                caller_resume_ip,
            });
            dynasm!(self.ops
                ; .arch x64
                ; sub rsp, metadata_size
                ; mov [rsp], r12
                ; mov [rsp + 8], r15
                ; mov rax, QWORD site as i64
                ; mov [rsp + 16], rax
                // Allocate the callee frame and initialise every register to
                // Nil (discriminant 0), so the argument moves below find
                // nothing to drop. r12 still addresses the caller's registers.
                ; sub rsp, frame_size
            );
            for reg in 0..trace.register_count {
                let offset = reg as i32 * value_size;
                dynasm!(self.ops
                    ; .arch x64
                    ; mov BYTE [rsp + offset], 0
                );
            }

            // Copy positional arguments into the callee frame. A scalar of
            // known type is a tag + payload store into the fresh Nil slot,
            // an aliased argument a bit copy; anything else goes through the
            // helper, which clones it.
            let mut helper_copied: Vec<u8> = Vec::new();
            for (arg_index, src_reg) in trace.arg_registers.iter().enumerate() {
                let src_offset = (*src_reg as i32) * value_size;
                let dest_offset = (arg_index as i32) * value_size;
                match arg_types.get(arg_index).copied().flatten() {
                    Some(ty @ (ValueType::Int | ValueType::Bool | ValueType::Float)) => {
                        let tag = match ty {
                            ValueType::Int => ValueTag::Int,
                            ValueType::Bool => ValueTag::Bool,
                            _ => ValueTag::Float,
                        }
                        .as_u8() as i8;
                        dynasm!(self.ops
                            ; .arch x64
                            ; mov rax, [r12 + src_offset + 8]
                            ; mov BYTE [rsp + dest_offset], tag
                            ; mov [rsp + dest_offset + 8], rax
                        );
                    }
                    _ if alias_mask & (1u64 << arg_index) != 0 => {
                        for word in (0..value_size).step_by(8) {
                            dynasm!(self.ops
                                ; .arch x64
                                ; mov rax, [r12 + src_offset + word]
                                ; mov [rsp + dest_offset + word], rax
                            );
                        }
                    }
                    _ => {
                        helper_copied.push(arg_index as u8);
                        dynasm!(self.ops
                            ; .arch x64
                            ; lea rdi, [r12 + src_offset]
                            ; lea rsi, [rsp + dest_offset]
                            ; mov rax, QWORD jit_move_safe as *const () as _
                            ; call rax
                            ; test al, al
                            ; jz >fail
                        );
                    }
                }
            }
            // A failed argument move resumes at the call, in the caller.
            self.emit_fail_stub();
            for (arg_index, ty) in arg_types.iter().enumerate() {
                if let Some(ty @ (ValueType::Int | ValueType::Bool | ValueType::Float)) = ty {
                    self.scalar_registers.insert(arg_index as u8, *ty);
                }
            }
            // The rest of the frame is still Nil: the first store into
            // any of those registers has nothing to drop.
            for reg in trace.arg_registers.len() as u8..trace.register_count {
                self.scalar_registers.insert(reg, ValueType::Plain);
            }

            // Point r12 at the callee frame and link the record.
            dynasm!(self.ops
                ; .arch x64
                ; mov r12, rsp
                ; lea r15, [rsp + frame_size]
            );

            let call_ip = self.current_fail_ip;
            // The callee's registers are numbered from its own frame:
            // nothing left in rax / xmm0 carries across the frame switch,
            // either way (`compile_ops` clears on entry).
            let inline_result = self.compile_ops(&trace.body, guard_index, guards);
            self.hot_rax = None;
            self.hot_xmm0 = None;
            self.hot_rax_in = None;
            self.hot_xmm0_in = None;
            inline_result?;
            // A failed result move exits from the callee frame at its last
            // instruction; everything after the frame is popped is the
            // caller's again.
            let callee_last_ip = self.current_fail_ip;
            self.current_fail_ip = call_ip;

            // The result: a scalar of known type rides in r14 (callee-saved,
            // unused until the exit path) across the frame teardown and is
            // stored into the caller's register afterwards; anything else is
            // moved by the helper while the callee frame is still live.
            let result_type = trace.return_register.and_then(|ret_reg| {
                self.scalar_registers
                    .get(&ret_reg)
                    .copied()
                    .filter(|ty| matches!(ty, ValueType::Int | ValueType::Bool | ValueType::Float))
            });
            if let Some(ret_reg) = trace.return_register {
                let ret_offset = (ret_reg as i32) * value_size;
                if let Some(ty) = result_type {
                    if ty == ValueType::Bool {
                        dynasm!(self.ops ; .arch x64 ; movzx r14d, BYTE [r12 + ret_offset + 8]);
                    } else {
                        dynasm!(self.ops ; .arch x64 ; mov r14, [r12 + ret_offset + 8]);
                    }
                } else {
                    self.current_fail_ip = callee_last_ip;
                    let dest_offset = (dest as i32) * value_size;
                    dynasm!(self.ops
                        ; .arch x64
                        ; mov r14, [r15]
                        ; lea rdi, [r12 + ret_offset]
                        ; lea rsi, [r14 + dest_offset]
                        ; mov rax, QWORD jit_move_safe as *const () as _
                        ; call rax
                        ; test al, al
                        ; jz >fail
                    );
                    self.emit_fail_stub();
                    self.current_fail_ip = call_ip;
                }
            }

            // Drop callee registers — unless every one of them is a scalar,
            // an alias, or was never written (still Nil) — and pop the frame
            // + record. Aliased registers are blanked first so the drop
            // leaves the caller's value alone.
            let may_own = (0..trace.register_count).any(|reg| {
                !self.scalar_registers.contains_key(&reg)
                    && (helper_copied.contains(&reg)
                        || trace.body.iter().any(|op| Self::op_writes_register(op, reg)))
            });
            if may_own {
                for reg in 0..trace.register_count.min(64) {
                    if alias_mask & (1u64 << reg) != 0 {
                        let offset = reg as i32 * value_size;
                        dynasm!(self.ops ; .arch x64 ; mov BYTE [r12 + offset], 0);
                    }
                }
                dynasm!(self.ops
                    ; .arch x64
                    ; mov rdi, r12
                    ; mov esi, DWORD frame_value_count
                    ; mov rax, QWORD jit_drop_values as *const () as _
                    ; call rax
                );
            }
            dynasm!(self.ops
                ; .arch x64
                ; lea rsp, [r15 + metadata_size]
                ; mov r12, [r15]
                ; mov r15, [r15 + 8]
            );

            // Back in the caller's frame: the caller's environment is
            // restored by our caller, so store through the generic path
            // (which drops whatever `dest` held).
            self.scalar_registers.clear();
            match result_type {
                Some(ValueType::Float) => {
                    dynasm!(self.ops ; .arch x64 ; movq xmm0, r14);
                    self.store_xmm0_as_float(dest);
                }
                Some(ValueType::Int) => {
                    dynasm!(self.ops ; .arch x64 ; mov rax, r14);
                    self.store_from_rax(dest, ValueTag::Int.as_u8());
                }
                Some(ValueType::Bool) => {
                    dynasm!(self.ops ; .arch x64 ; mov rax, r14);
                    self.store_from_rax(dest, ValueTag::Bool.as_u8());
                }
                _ => {}
            }
            if trace.return_register.is_none() {
                self.compile_load_const(dest, &Value::Nil)?;
            }
            dynasm!(self.ops
                ; .arch x64
                ; => inline_end
            );

            Ok(result_type)
        })();
        self.inline_depth -= 1;
        result
    }
}
