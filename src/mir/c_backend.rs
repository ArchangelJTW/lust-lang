#[cfg(feature = "std")]
pub use c_engine::{CArtifactKind, CCompileOptions, MirCBackendError, MirToCCompiler};

#[cfg(feature = "std")]
pub mod c_engine {
    use crate::mir::ir::*;
    use alloc::{format, string::String, vec::Vec};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum CArtifactKind {
        Executable,
        CDylib,
        StaticLib,
    }

    #[derive(Debug, Clone)]
    pub struct CCompileOptions {
        pub artifact_kind: CArtifactKind,
        pub emit_c_header: bool,
        pub max_memory_bytes: Option<usize>,
        /// Optimization flag passed to the C compiler (e.g. "-O3", "-O2", "-O0").
        pub opt_level: String,
        /// Optional entry function name to invoke from `main()` when building an `Executable`.
        pub entry_function: Option<String>,
    }

    impl Default for CCompileOptions {
        fn default() -> Self {
            Self {
                artifact_kind: CArtifactKind::CDylib,
                emit_c_header: false,
                max_memory_bytes: None,
                opt_level: "-O3".to_string(),
                entry_function: None,
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, thiserror::Error)]
    pub enum MirCBackendError {
        #[error("Binary operator '{0:?}' is not supported by MIR C backend")]
        UnsupportedBinaryOp(MirBinaryOp),
        #[error("Constant '{0:?}' is not supported by MIR C backend")]
        UnsupportedConstant(MirConstant),
        #[error("Failed to write C file: {0}")]
        WriteFile(String),
        #[error("Failed to invoke C compiler '{compiler}': {detail}")]
        InvokeCompiler { compiler: String, detail: String },
        #[error("C compilation failed:\n{0}")]
        CompilationFailed(String),
    }

    fn function_has_io_or_print(func: &MirFunction) -> bool {
        for block in &func.blocks {
            if let Some(Terminator::Call { callee, .. }) = &block.terminator {
                if callee == "print" || callee == "println" || callee.starts_with("io.") {
                    return true;
                }
            }
        }
        false
    }

    fn module_has_io_or_print(module: &MirModule) -> bool {
        module.functions.iter().any(function_has_io_or_print)
    }

    pub struct MirToCCompiler;

    impl MirToCCompiler {
        pub fn generate_source(func: &MirFunction) -> Result<String, MirCBackendError> {
            Self::generate_source_with_options(func, &CCompileOptions::default())
        }

        pub fn generate_source_with_options(
            func: &MirFunction,
            opts: &CCompileOptions,
        ) -> Result<String, MirCBackendError> {
            let needs_io = function_has_io_or_print(func)
                || opts.artifact_kind == CArtifactKind::Executable;
            let mut s = Self::generate_preamble(opts, needs_io);
            let known_fns = [func.signature.name.as_str()];
            s.push_str(&Self::generate_prototype(func));
            s.push_str(";\n\n");
            s.push_str(&Self::generate_function_code(func, &known_fns)?);
            if opts.artifact_kind == CArtifactKind::Executable {
                s.push('\n');
                s.push_str(&Self::generate_main_wrapper(func));
            }
            Ok(s)
        }

        pub fn generate_module_source_with_options(
            module: &MirModule,
            opts: &CCompileOptions,
        ) -> Result<String, MirCBackendError> {
            let needs_io =
                module_has_io_or_print(module) || opts.artifact_kind == CArtifactKind::Executable;
            let mut s = Self::generate_preamble(opts, needs_io);
            let known_fns: Vec<&str> = module
                .functions
                .iter()
                .map(|f| f.signature.name.as_str())
                .collect();
            // Forward declarations for all functions in the module
            for func in &module.functions {
                s.push_str(&Self::generate_prototype(func));
                s.push_str(";\n");
            }
            s.push('\n');
            for func in &module.functions {
                s.push_str(&Self::generate_function_code(func, &known_fns)?);
                s.push('\n');
            }
            if opts.artifact_kind == CArtifactKind::Executable {
                let entry_fn = opts
                    .entry_function
                    .as_deref()
                    .and_then(|name| {
                        module
                            .functions
                            .iter()
                            .find(|f| f.signature.name == name || f.signature.name.ends_with(&format!(".{name}")))
                    })
                    .or_else(|| {
                        module
                            .functions
                            .iter()
                            .find(|f| f.signature.name == "main" || f.signature.name.ends_with(".main"))
                    })
                    .or_else(|| module.functions.last());
                if let Some(entry) = entry_fn {
                    s.push_str(&Self::generate_main_wrapper(entry));
                }
            }
            Ok(s)
        }

        fn generate_preamble(opts: &CCompileOptions, needs_io: bool) -> String {
            let mut s = String::new();
            s.push_str("/* Generated by Lust AOT compiler (MIR C backend) */\n");
            s.push_str("#include <stdint.h>\n");
            s.push_str("#include <stddef.h>\n");
            s.push_str("#include <stdbool.h>\n");
            s.push_str("#include <stdlib.h>\n");
            s.push_str("#include <string.h>\n");
            s.push_str("#include <inttypes.h>\n");
            s.push_str("#include <math.h>\n");
            if needs_io {
                s.push_str("#include <stdio.h>\n");
            }
            s.push('\n');

            s.push_str("#if defined(_WIN32)\n");
            s.push_str("#define LUST_EXPORT __declspec(dllexport)\n");
            s.push_str("#else\n");
            s.push_str("#define LUST_EXPORT __attribute__((visibility(\"default\")))\n");
            s.push_str("#endif\n\n");

            s.push_str("typedef struct {\n");
            s.push_str("    int64_t *ptr;\n");
            s.push_str("    size_t len;\n");
            s.push_str("} LustSlice;\n\n");

            // Arena allocator (linked chunks so reset is O(chunks) and allocation is O(1) bump)
            s.push_str("typedef struct LustArenaChunk {\n");
            s.push_str("    struct LustArenaChunk *next;\n");
            s.push_str("    size_t cap;\n");
            s.push_str("    size_t used;\n");
            s.push_str("    int64_t data[];\n");
            s.push_str("} LustArenaChunk;\n\n");

            s.push_str("static LustArenaChunk *__arena_head = NULL;\n");
            s.push_str("static int64_t __gas_remaining = 1000000000LL;\n");
            let mem_limit_str = match opts.max_memory_bytes {
                Some(b) => format!("{b}ULL"),
                None => "SIZE_MAX".to_string(),
            };
            s.push_str(&format!("static size_t __memory_limit = {mem_limit_str};\n"));
            s.push_str("static size_t __memory_used = 0;\n");
            s.push_str("static int32_t __last_status = 0;\n\n");

            s.push_str("LUST_EXPORT void lust_set_gas_limit(int64_t gas) { __gas_remaining = gas; }\n");
            s.push_str("LUST_EXPORT int64_t lust_get_gas_remaining(void) { return __gas_remaining; }\n");
            s.push_str("LUST_EXPORT void lust_set_memory_limit(size_t bytes) { __memory_limit = bytes; }\n");
            s.push_str("LUST_EXPORT size_t lust_get_memory_used(void) { return __memory_used; }\n");
            s.push_str("LUST_EXPORT int32_t lust_get_last_status(void) { return __last_status; }\n");
            s.push_str("LUST_EXPORT void lust_reset_arena(void) {\n");
            s.push_str("    LustArenaChunk *c = __arena_head;\n");
            s.push_str("    while (c) {\n");
            s.push_str("        LustArenaChunk *next = c->next;\n");
            s.push_str("        free(c);\n");
            s.push_str("        c = next;\n");
            s.push_str("    }\n");
            s.push_str("    __arena_head = NULL;\n");
            s.push_str("    __memory_used = 0;\n");
            s.push_str("    __last_status = 0;\n");
            s.push_str("}\n\n");

            s.push_str("static LustSlice __alloc_slice(const int64_t *items, size_t count) {\n");
            s.push_str("    LustSlice empty = { NULL, 0 };\n");
            s.push_str("    if (count == 0) return empty;\n");
            s.push_str("    if (count > SIZE_MAX / sizeof(int64_t)) {\n");
            s.push_str("        __last_status = 2;\n");
            s.push_str("        return empty;\n");
            s.push_str("    }\n");
            s.push_str("    size_t bytes_needed = count * sizeof(int64_t);\n");
            s.push_str("    if (bytes_needed > __memory_limit || __memory_used > __memory_limit - bytes_needed) {\n");
            s.push_str("        __last_status = 2;\n");
            s.push_str("        return empty;\n");
            s.push_str("    }\n");
            s.push_str("    if (!__arena_head || (__arena_head->cap - __arena_head->used) < count) {\n");
            s.push_str("        size_t chunk_cap = count > 4096 ? count : 4096;\n");
            s.push_str("        LustArenaChunk *chunk = (LustArenaChunk *)malloc(sizeof(LustArenaChunk) + chunk_cap * sizeof(int64_t));\n");
            s.push_str("        if (!chunk) {\n");
            s.push_str("            __last_status = 2;\n");
            s.push_str("            return empty;\n");
            s.push_str("        }\n");
            s.push_str("        chunk->next = __arena_head;\n");
            s.push_str("        chunk->cap = chunk_cap;\n");
            s.push_str("        chunk->used = 0;\n");
            s.push_str("        __arena_head = chunk;\n");
            s.push_str("    }\n");
            s.push_str("    int64_t *dst = &__arena_head->data[__arena_head->used];\n");
            s.push_str("    __arena_head->used += count;\n");
            s.push_str("    memcpy(dst, items, bytes_needed);\n");
            s.push_str("    __memory_used += bytes_needed;\n");
            s.push_str("    LustSlice res = { dst, count };\n");
            s.push_str("    return res;\n");
            s.push_str("}\n\n");

            if needs_io {
                s.push_str("static void __lust_mir_print_i64(int64_t val) { printf(\"%\" PRId64 \"\\n\", val); }\n");
                s.push_str("static void __lust_mir_print_f64(double val) { printf(\"%g\\n\", val); }\n");
                s.push_str("static void __lust_mir_print_bool(bool val) { printf(\"%s\\n\", val ? \"true\" : \"false\"); }\n\n");
            }

            s.push_str("static inline int64_t __lust_mir_div(int64_t a, int64_t b) {\n");
            s.push_str("    if (b == 0) abort();\n");
            s.push_str("    if (a == INT64_MIN && b == -1) return INT64_MIN;\n");
            s.push_str("    return a / b;\n");
            s.push_str("}\n\n");

            s.push_str("static inline int64_t __lust_mir_rem(int64_t a, int64_t b) {\n");
            s.push_str("    if (b == 0) abort();\n");
            s.push_str("    if (a == INT64_MIN && b == -1) return 0;\n");
            s.push_str("    return a % b;\n");
            s.push_str("}\n\n");

            s.push_str("static inline int64_t __lust_mir_floordiv(int64_t a, int64_t b) {\n");
            s.push_str("    if (b == 0) abort();\n");
            s.push_str("    if (a == INT64_MIN && b == -1) return INT64_MIN;\n");
            s.push_str("    int64_t q = a / b;\n");
            s.push_str("    int64_t r = a % b;\n");
            s.push_str("    if (r != 0 && ((r < 0) != (b < 0))) q -= 1;\n");
            s.push_str("    return q;\n");
            s.push_str("}\n\n");

            s.push_str("static inline int64_t __lust_mir_pow(int64_t base, int64_t exp) {\n");
            s.push_str("    if (exp < 0) abort();\n");
            s.push_str("    uint64_t b = (uint64_t)base;\n");
            s.push_str("    uint64_t e = (uint64_t)exp;\n");
            s.push_str("    uint64_t res = 1;\n");
            s.push_str("    while (e > 0) {\n");
            s.push_str("        if (e & 1ULL) res *= b;\n");
            s.push_str("        b *= b;\n");
            s.push_str("        e >>= 1;\n");
            s.push_str("    }\n");
            s.push_str("    return (int64_t)res;\n");
            s.push_str("}\n\n");

            s.push_str("static inline int64_t __lust_mir_abs(int64_t x) {\n");
            s.push_str("    if (x == INT64_MIN) return INT64_MIN;\n");
            s.push_str("    return x < 0 ? -x : x;\n");
            s.push_str("}\n\n");

            s.push_str("static inline int64_t __lust_mir_min(int64_t a, int64_t b) { return a < b ? a : b; }\n");
            s.push_str("static inline int64_t __lust_mir_max(int64_t a, int64_t b) { return a > b ? a : b; }\n");
            s.push_str("static inline int64_t __lust_mir_clamp(int64_t v, int64_t lo, int64_t hi) {\n");
            s.push_str("    return v < lo ? lo : (v > hi ? hi : v);\n");
            s.push_str("}\n\n");

            s
        }

        fn c_fn_name(name: &str) -> String {
            if name == "main" {
                return "__lust_entry_main".to_string();
            }
            let mut out = String::with_capacity(name.len());
            for ch in name.chars() {
                if ch.is_ascii_alphanumeric() || ch == '_' {
                    out.push(ch);
                } else {
                    out.push_str("__");
                }
            }
            if out.is_empty() || out.as_bytes()[0].is_ascii_digit() {
                format!("__lust_{out}")
            } else {
                out
            }
        }

        fn resolve_callee_name(callee: &str, known_fns: &[&str]) -> String {
            if known_fns.contains(&callee) {
                return Self::c_fn_name(callee);
            }
            let suffix = format!(".{callee}");
            if let Some(matched) = known_fns.iter().find(|f| f.ends_with(&suffix)) {
                return Self::c_fn_name(matched);
            }
            Self::c_fn_name(callee)
        }

        fn generate_prototype(func: &MirFunction) -> String {
            let ret_ty = mir_to_c_type(&func.signature.return_type);
            let fn_name = Self::c_fn_name(&func.signature.name);
            let mut s = format!("LUST_EXPORT {ret_ty} {fn_name}(");
            if func.params.is_empty() {
                s.push_str("void");
            } else {
                for (i, p_id) in func.params.iter().enumerate() {
                    if i > 0 {
                        s.push_str(", ");
                    }
                    let p_ty = mir_to_c_type(&func.locals[p_id.0].ty);
                    s.push_str(&format!("{p_ty} arg{i}"));
                }
            }
            s.push(')');
            s
        }

        fn generate_main_wrapper(entry: &MirFunction) -> String {
            let fn_name = Self::c_fn_name(&entry.signature.name);
            let mut s = String::new();
            s.push_str("int main(void) {\n");
            match entry.signature.return_type {
                MirType::Unit => {
                    s.push_str(&format!("    {fn_name}();\n"));
                }
                MirType::Int => {
                    s.push_str(&format!("    int64_t res = {fn_name}();\n"));
                    s.push_str("    (void)res;\n");
                }
                MirType::Float => {
                    s.push_str(&format!("    double res = {fn_name}();\n"));
                    s.push_str("    (void)res;\n");
                }
                MirType::Bool => {
                    s.push_str(&format!("    bool res = {fn_name}();\n"));
                    s.push_str("    (void)res;\n");
                }
                _ => {
                    s.push_str(&format!("    (void){fn_name}();\n"));
                }
            }
            s.push_str("    lust_reset_arena();\n");
            s.push_str("    return __last_status == 0 ? 0 : (int)__last_status;\n");
            s.push_str("}\n");
            s
        }

        fn generate_function_code(
            func: &MirFunction,
            known_fns: &[&str],
        ) -> Result<String, MirCBackendError> {
            let mut s = Self::generate_prototype(func);
            s.push_str(" {\n");

            // Declare locals
            for local in &func.locals {
                let ty = mir_to_c_type(&local.ty);
                let default_val = default_c_value(&local.ty);
                s.push_str(&format!("    {ty} _{} = {default_val};\n", local.id.0));
            }

            // Initialize parameter locals
            for (i, p_id) in func.params.iter().enumerate() {
                s.push_str(&format!("    _{} = arg{i};\n", p_id.0));
            }

            if !func.blocks.is_empty() {
                s.push_str("    goto bb_0;\n");
            }

            let err_ret_stmt = if func.signature.return_type == MirType::Unit {
                "return;".to_string()
            } else {
                format!("return {};", default_c_value(&func.signature.return_type))
            };

            for block in &func.blocks {
                s.push_str(&format!("bb_{}:\n", block.id.0));
                for stmt in &block.statements {
                    match stmt {
                        Statement::Assign(place, rval, _) => {
                            let p_expr = lower_place(place, &func.locals);
                            let r_expr = lower_rvalue(rval, &func.locals)?;
                            s.push_str(&format!("    {p_expr} = {r_expr};\n"));
                            if matches!(rval, Rvalue::Aggregate(_, _)) {
                                s.push_str(&format!(
                                    "    if (__last_status != 0) {{ {err_ret_stmt} }}\n"
                                ));
                            }
                        }
                        Statement::ResourceCheck(ResourceKind::Gas, amt, _) => {
                            s.push_str(&format!("    __gas_remaining -= {amt}LL;\n"));
                            s.push_str(&format!(
                                "    if (__gas_remaining < 0) {{ __last_status = 1; {err_ret_stmt} }}\n"
                            ));
                        }
                        _ => {}
                    }
                }

                if let Some(term) = &block.terminator {
                    match term {
                        Terminator::Goto(target, _) => {
                            s.push_str(&format!("    goto bb_{};\n", target.0));
                        }
                        Terminator::Cond {
                            cond,
                            then_block,
                            else_block,
                            ..
                        } => {
                            let c_expr = lower_operand(cond, &func.locals)?;
                            s.push_str(&format!(
                                "    if ({c_expr}) goto bb_{}; else goto bb_{};\n",
                                then_block.0, else_block.0
                            ));
                        }
                        Terminator::Switch {
                            discr,
                            cases,
                            default,
                            ..
                        } => {
                            let d_expr = lower_operand(discr, &func.locals)?;
                            s.push_str(&format!("    switch ({d_expr}) {{\n"));
                            for (val, target) in cases {
                                s.push_str(&format!(
                                    "        case {val}LL: goto bb_{};\n",
                                    target.0
                                ));
                            }
                            s.push_str(&format!(
                                "        default: goto bb_{};\n",
                                default.0
                            ));
                            s.push_str("    }\n");
                        }
                        Terminator::Call {
                            callee,
                            args,
                            destination,
                            success,
                            ..
                        } => {
                            let mut args_strs = Vec::new();
                            for arg in args {
                                args_strs.push(lower_operand(arg, &func.locals)?);
                            }
                            let dest_expr = lower_place(destination, &func.locals);
                            if callee == "print" || callee == "println" || callee.starts_with("io.")
                            {
                                if let Some(first_op) = args.first() {
                                    let first = &args_strs[0];
                                    match operand_type(first_op, &func.locals) {
                                        Some(MirType::Float) => {
                                            s.push_str(&format!(
                                                "    __lust_mir_print_f64({first});\n"
                                            ));
                                        }
                                        Some(MirType::Bool) => {
                                            s.push_str(&format!(
                                                "    __lust_mir_print_bool({first});\n"
                                            ));
                                        }
                                        _ => {
                                            s.push_str(&format!(
                                                "    __lust_mir_print_i64((int64_t)({first}));\n"
                                            ));
                                        }
                                    }
                                } else {
                                    s.push_str("    __lust_mir_print_i64(0);\n");
                                }
                                s.push_str(&format!("    goto bb_{};\n", success.0));
                            } else {
                                let call_expr = match callee.as_str() {
                                    "math.abs" | "abs" => format!(
                                        "__lust_mir_abs({})",
                                        args_strs.first().cloned().unwrap_or_default()
                                    ),
                                    "math.min" | "min" => format!(
                                        "__lust_mir_min({}, {})",
                                        args_strs.first().cloned().unwrap_or_default(),
                                        args_strs.get(1).cloned().unwrap_or_default()
                                    ),
                                    "math.max" | "max" => format!(
                                        "__lust_mir_max({}, {})",
                                        args_strs.first().cloned().unwrap_or_default(),
                                        args_strs.get(1).cloned().unwrap_or_default()
                                    ),
                                    "math.clamp" | "clamp" => format!(
                                        "__lust_mir_clamp({}, {}, {})",
                                        args_strs.first().cloned().unwrap_or_default(),
                                        args_strs.get(1).cloned().unwrap_or_default(),
                                        args_strs.get(2).cloned().unwrap_or_default()
                                    ),
                                    "table.len" | "array.len" => format!(
                                        "(int64_t)({}).len",
                                        args_strs.first().cloned().unwrap_or_default()
                                    ),
                                    _ => {
                                        let target_name = Self::resolve_callee_name(callee, known_fns);
                                        let args_joined = args_strs.join(", ");
                                        format!("{target_name}({args_joined})")
                                    }
                                };
                                s.push_str(&format!("    {dest_expr} = {call_expr};\n"));
                                s.push_str(&format!(
                                    "    if (__last_status != 0) {{ {err_ret_stmt} }}\n"
                                ));
                                s.push_str(&format!("    goto bb_{};\n", success.0));
                            }
                        }
                        Terminator::Return(opt_op, _) => {
                            if func.signature.return_type != MirType::Unit {
                                let ret_expr = if let Some(op) = opt_op {
                                    lower_operand(op, &func.locals)?
                                } else {
                                    default_c_value(&func.signature.return_type).into()
                                };
                                s.push_str(&format!("    return {ret_expr};\n"));
                            } else {
                                s.push_str("    return;\n");
                            }
                        }
                        Terminator::Unreachable(_) => {
                            s.push_str("    abort();\n");
                        }
                        _ => {
                            s.push_str(&format!("    {err_ret_stmt}\n"));
                        }
                    }
                }
            }

            s.push_str("}\n");
            Ok(s)
        }

        pub fn compile_with_options(
            func: &MirFunction,
            out_dir: &Path,
            opts: &CCompileOptions,
        ) -> Result<PathBuf, MirCBackendError> {
            let c_source = Self::generate_source_with_options(func, opts)?;
            let c_file = out_dir.join(format!("{}.c", func.signature.name));
            fs::write(&c_file, c_source)
                .map_err(|e| MirCBackendError::WriteFile(format!("{e}")))?;

            if opts.emit_c_header {
                crate::mir::c_header_gen::CHeaderGenerator::write_header_for_function(func, out_dir)
                    .map_err(|e| MirCBackendError::WriteFile(format!("{e}")))?;
            }

            Self::build_c_artifact(&func.signature.name, &c_file, out_dir, opts)
        }

        pub fn compile_module_with_options(
            module: &MirModule,
            out_dir: &Path,
            opts: &CCompileOptions,
        ) -> Result<PathBuf, MirCBackendError> {
            let c_source = Self::generate_module_source_with_options(module, opts)?;
            let c_file = out_dir.join(format!("{}.c", module.name));
            fs::write(&c_file, c_source)
                .map_err(|e| MirCBackendError::WriteFile(format!("{e}")))?;

            if opts.emit_c_header {
                crate::mir::c_header_gen::CHeaderGenerator::write_header_for_module(module, out_dir)
                    .map_err(|e| MirCBackendError::WriteFile(format!("{e}")))?;
            }

            Self::build_c_artifact(&module.name, &c_file, out_dir, opts)
        }

        fn find_c_compiler() -> String {
            if let Ok(cc) = std::env::var("CC") {
                if !cc.trim().is_empty() {
                    return cc;
                }
            }
            for candidate in ["cc", "clang", "gcc"] {
                if Command::new(candidate)
                    .arg("--version")
                    .output()
                    .map(|o| o.status.success())
                    .unwrap_or(false)
                {
                    return candidate.to_string();
                }
            }
            "cc".to_string()
        }

        fn build_c_artifact(
            base_name: &str,
            c_file: &Path,
            out_dir: &Path,
            opts: &CCompileOptions,
        ) -> Result<PathBuf, MirCBackendError> {
            let cc = Self::find_c_compiler();
            match opts.artifact_kind {
                CArtifactKind::Executable => {
                    let out_name = if cfg!(target_os = "windows") {
                        format!("{base_name}.exe")
                    } else {
                        base_name.to_string()
                    };
                    let out_path = out_dir.join(&out_name);
                    let mut cmd = Command::new(&cc);
                    cmd.args([
                        "-std=c11",
                        &opts.opt_level,
                        c_file.to_str().unwrap(),
                        "-o",
                        out_path.to_str().unwrap(),
                    ]);
                    if !cfg!(target_os = "windows") {
                        cmd.arg("-lm");
                    }
                    let output = cmd.current_dir(out_dir).output().map_err(|e| {
                        MirCBackendError::InvokeCompiler {
                            compiler: cc.clone(),
                            detail: format!("{e}"),
                        }
                    })?;
                    if !output.status.success() {
                        return Err(MirCBackendError::CompilationFailed(
                            String::from_utf8_lossy(&output.stderr).into_owned(),
                        ));
                    }
                    Ok(out_path)
                }
                CArtifactKind::CDylib => {
                    let dylib_name = if cfg!(target_os = "macos") {
                        format!("lib{base_name}.dylib")
                    } else if cfg!(target_os = "windows") {
                        format!("{base_name}.dll")
                    } else {
                        format!("lib{base_name}.so")
                    };
                    let out_path = out_dir.join(&dylib_name);
                    let mut cmd = Command::new(&cc);
                    cmd.args(["-std=c11", &opts.opt_level, "-shared"]);
                    if !cfg!(target_os = "windows") {
                        cmd.arg("-fPIC");
                    }
                    cmd.args([
                        c_file.to_str().unwrap(),
                        "-o",
                        out_path.to_str().unwrap(),
                    ]);
                    if !cfg!(target_os = "windows") {
                        cmd.arg("-lm");
                    }
                    let output = cmd.current_dir(out_dir).output().map_err(|e| {
                        MirCBackendError::InvokeCompiler {
                            compiler: cc.clone(),
                            detail: format!("{e}"),
                        }
                    })?;
                    if !output.status.success() {
                        return Err(MirCBackendError::CompilationFailed(
                            String::from_utf8_lossy(&output.stderr).into_owned(),
                        ));
                    }
                    Ok(out_path)
                }
                CArtifactKind::StaticLib => {
                    let obj_path = out_dir.join(format!("{base_name}.o"));
                    let static_name = if cfg!(target_os = "windows") {
                        format!("{base_name}.lib")
                    } else {
                        format!("lib{base_name}.a")
                    };
                    let out_path = out_dir.join(&static_name);

                    let mut cmd = Command::new(&cc);
                    cmd.args(["-std=c11", &opts.opt_level, "-c"]);
                    if !cfg!(target_os = "windows") {
                        cmd.arg("-fPIC");
                    }
                    cmd.args([
                        c_file.to_str().unwrap(),
                        "-o",
                        obj_path.to_str().unwrap(),
                    ]);
                    let output = cmd.current_dir(out_dir).output().map_err(|e| {
                        MirCBackendError::InvokeCompiler {
                            compiler: cc.clone(),
                            detail: format!("{e}"),
                        }
                    })?;
                    if !output.status.success() {
                        return Err(MirCBackendError::CompilationFailed(
                            String::from_utf8_lossy(&output.stderr).into_owned(),
                        ));
                    }

                    let ar_out = Command::new("ar")
                        .args(["rcs", out_path.to_str().unwrap(), obj_path.to_str().unwrap()])
                        .current_dir(out_dir)
                        .output()
                        .map_err(|e| MirCBackendError::InvokeCompiler {
                            compiler: "ar".to_string(),
                            detail: format!("{e}"),
                        })?;
                    if !ar_out.status.success() {
                        return Err(MirCBackendError::CompilationFailed(
                            String::from_utf8_lossy(&ar_out.stderr).into_owned(),
                        ));
                    }
                    Ok(out_path)
                }
            }
        }

        pub fn compile_to_dylib(
            func: &MirFunction,
            out_dir: &Path,
        ) -> Result<PathBuf, MirCBackendError> {
            Self::compile_with_options(func, out_dir, &CCompileOptions::default())
        }

        pub fn compile_module_to_dylib(
            module: &MirModule,
            out_dir: &Path,
        ) -> Result<PathBuf, MirCBackendError> {
            Self::compile_module_with_options(module, out_dir, &CCompileOptions::default())
        }
    }

    fn is_slice_type(ty: &MirType) -> bool {
        matches!(ty, MirType::Array(_) | MirType::Tuple(_) | MirType::Struct(_))
    }

    fn mir_to_c_type(ty: &MirType) -> &'static str {
        match ty {
            MirType::Int => "int64_t",
            MirType::Float => "double",
            MirType::Bool => "bool",
            MirType::Unit => "void",
            MirType::Array(_) | MirType::Tuple(_) | MirType::Struct(_) => "LustSlice",
            _ => "int64_t",
        }
    }

    fn default_c_value(ty: &MirType) -> &'static str {
        match ty {
            MirType::Int => "0",
            MirType::Float => "0.0",
            MirType::Bool => "false",
            MirType::Array(_) | MirType::Tuple(_) | MirType::Struct(_) => "{ NULL, 0 }",
            _ => "0",
        }
    }

    fn place_type<'a>(place: &Place, locals: &'a [LocalDecl]) -> Option<&'a MirType> {
        match place {
            Place::Local(loc) => locals.get(loc.0).map(|l| &l.ty),
            Place::Index(_, _) | Place::Field(_, _) => None,
        }
    }

    fn operand_type<'a>(op: &Operand, locals: &'a [LocalDecl]) -> Option<&'a MirType> {
        match op {
            Operand::Copy(place) | Operand::Move(place) => place_type(place, locals),
            Operand::Constant(c) => match c {
                MirConstant::Int(_) => Some(&MirType::Int),
                MirConstant::Float(_) => Some(&MirType::Float),
                MirConstant::Bool(_) => Some(&MirType::Bool),
                MirConstant::Unit => Some(&MirType::Unit),
                _ => None,
            },
        }
    }

    fn lower_place(place: &Place, locals: &[LocalDecl]) -> String {
        match place {
            Place::Local(loc) => format!("_{}", loc.0),
            Place::Index(base, idx) => {
                let b = lower_place(base, locals);
                let is_slice = place_type(base, locals).map(is_slice_type).unwrap_or(false);
                if is_slice {
                    format!("({b}).ptr[(size_t)_{}]", idx.0)
                } else {
                    format!("((int64_t *)(uintptr_t)({b}))[(size_t)_{}]", idx.0)
                }
            }
            Place::Field(base, f) => {
                let b = lower_place(base, locals);
                let is_slice = place_type(base, locals).map(is_slice_type).unwrap_or(false);
                if is_slice {
                    format!("({b}).ptr[{f}]")
                } else {
                    format!("((int64_t *)(uintptr_t)({b}))[{f}]")
                }
            }
        }
    }

    fn lower_operand(op: &Operand, locals: &[LocalDecl]) -> Result<String, MirCBackendError> {
        match op {
            Operand::Copy(place) | Operand::Move(place) => Ok(lower_place(place, locals)),
            Operand::Constant(c) => match c {
                MirConstant::Int(i) => {
                    if *i == i64::MIN {
                        Ok("INT64_MIN".to_string())
                    } else {
                        Ok(format!("{i}LL"))
                    }
                }
                MirConstant::Float(f) => Ok(format!("{f:?}")),
                MirConstant::Bool(b) => Ok(if *b { "true" } else { "false" }.to_string()),
                MirConstant::Unit => Ok("0".to_string()),
                _ => Err(MirCBackendError::UnsupportedConstant(c.clone())),
            },
        }
    }

    fn lower_rvalue(rval: &Rvalue, locals: &[LocalDecl]) -> Result<String, MirCBackendError> {
        match rval {
            Rvalue::Use(op) => lower_operand(op, locals),
            Rvalue::BinaryOp(binop, lhs, rhs) => {
                let l = lower_operand(lhs, locals)?;
                let r = lower_operand(rhs, locals)?;
                let is_float = matches!(operand_type(lhs, locals), Some(MirType::Float))
                    || matches!(operand_type(rhs, locals), Some(MirType::Float));
                let s = match binop {
                    MirBinaryOp::Add => {
                        if is_float {
                            format!("({l} + {r})")
                        } else {
                            format!("(int64_t)((uint64_t)({l}) + (uint64_t)({r}))")
                        }
                    }
                    MirBinaryOp::Sub => {
                        if is_float {
                            format!("({l} - {r})")
                        } else {
                            format!("(int64_t)((uint64_t)({l}) - (uint64_t)({r}))")
                        }
                    }
                    MirBinaryOp::Mul => {
                        if is_float {
                            format!("({l} * {r})")
                        } else {
                            format!("(int64_t)((uint64_t)({l}) * (uint64_t)({r}))")
                        }
                    }
                    MirBinaryOp::Div => {
                        if is_float {
                            format!("({l} / {r})")
                        } else {
                            format!("__lust_mir_div({l}, {r})")
                        }
                    }
                    MirBinaryOp::Mod => {
                        if is_float {
                            format!("fmod({l}, {r})")
                        } else {
                            format!("__lust_mir_rem({l}, {r})")
                        }
                    }
                    MirBinaryOp::FloorDiv => {
                        if is_float {
                            format!("floor({l} / {r})")
                        } else {
                            format!("__lust_mir_floordiv({l}, {r})")
                        }
                    }
                    MirBinaryOp::Pow => {
                        if is_float {
                            format!("pow({l}, {r})")
                        } else {
                            format!("__lust_mir_pow({l}, {r})")
                        }
                    }
                    MirBinaryOp::BitAnd => format!("({l} & {r})"),
                    MirBinaryOp::BitOr => format!("({l} | {r})"),
                    MirBinaryOp::BitXor => format!("({l} ^ {r})"),
                    MirBinaryOp::Shl => {
                        format!("(int64_t)((uint64_t)({l}) << ((uint64_t)({r}) & 63ULL))")
                    }
                    MirBinaryOp::Shr => format!("(({l}) >> ((uint64_t)({r}) & 63ULL))"),
                    MirBinaryOp::Eq => format!("({l} == {r})"),
                    MirBinaryOp::NotEq => format!("({l} != {r})"),
                    MirBinaryOp::Lt => format!("({l} < {r})"),
                    MirBinaryOp::LtEq => format!("({l} <= {r})"),
                    MirBinaryOp::Gt => format!("({l} > {r})"),
                    MirBinaryOp::GtEq => format!("({l} >= {r})"),
                    MirBinaryOp::And => format!("({l} && {r})"),
                    MirBinaryOp::Or => format!("({l} || {r})"),
                };
                Ok(s)
            }
            Rvalue::UnaryOp(unop, op) => {
                let v = lower_operand(op, locals)?;
                let s = match unop {
                    MirUnaryOp::Neg => {
                        if matches!(operand_type(op, locals), Some(MirType::Float)) {
                            format!("(-({v}))")
                        } else {
                            format!("(int64_t)(0ULL - (uint64_t)({v}))")
                        }
                    }
                    MirUnaryOp::Not => format!("(!({v}))"),
                    MirUnaryOp::BitNot => format!("(~({v}))"),
                    MirUnaryOp::Len => format!("(int64_t)({v}).len"),
                };
                Ok(s)
            }
            Rvalue::Aggregate(
                MirType::Array(_) | MirType::Tuple(_) | MirType::Struct(_),
                elements,
            ) => {
                if elements.is_empty() {
                    return Ok("__alloc_slice(NULL, 0)".to_string());
                }
                let mut elems = String::new();
                for (i, el) in elements.iter().enumerate() {
                    if i > 0 {
                        elems.push_str(", ");
                    }
                    let operand_str = lower_operand(el, locals)?;
                    if operand_type(el, locals).map(is_slice_type).unwrap_or(false) {
                        elems.push_str(&format!("(int64_t)(uintptr_t)({operand_str}).ptr"));
                    } else {
                        elems.push_str(&format!("(int64_t)({operand_str})"));
                    }
                }
                let count = elements.len();
                Ok(format!(
                    "__alloc_slice((const int64_t[]){{{elems}}}, {count})"
                ))
            }
            _ => Err(MirCBackendError::UnsupportedConstant(MirConstant::Unit)),
        }
    }
}
