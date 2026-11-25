// Host CLI / tooling boundary: Zig compiler backend invokes external zig compiler and writes to host filesystem.
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

#[cfg(feature = "std")]
pub use zig_engine::{
    MirToZigCompiler, MirZigBackendError, ZigArtifactKind, ZigCompileOptions,
};

#[cfg(feature = "std")]
pub mod zig_engine {
    use crate::mir::ir::*;
    use alloc::{format, string::String, vec::Vec};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ZigArtifactKind {
        CDylib,
        StaticLib,
    }

    #[derive(Debug, Clone)]
    pub struct ZigCompileOptions {
        pub artifact_kind: ZigArtifactKind,
        pub emit_c_header: bool,
        pub max_memory_bytes: Option<usize>,
    }

    impl Default for ZigCompileOptions {
        fn default() -> Self {
            Self {
                artifact_kind: ZigArtifactKind::CDylib,
                emit_c_header: false,
                max_memory_bytes: None,
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, thiserror::Error)]
    pub enum MirZigBackendError {
        #[error("Binary operator '{0:?}' is not supported by MIR Zig backend")]
        UnsupportedBinaryOp(MirBinaryOp),
        #[error("Constant '{0:?}' is not supported by MIR Zig backend")]
        UnsupportedConstant(MirConstant),
        #[error("Failed to write zig file: {0}")]
        WriteFile(String),
        #[error("Failed to invoke zig: {0}")]
        InvokeZig(String),
        #[error("Zig compilation failed")]
        CompilationFailed,
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

    pub struct MirToZigCompiler;

    impl MirToZigCompiler {
        pub fn generate_source(func: &MirFunction) -> Result<String, MirZigBackendError> {
            Self::generate_source_with_options(func, &ZigCompileOptions::default())
        }

        pub fn generate_source_with_options(
            func: &MirFunction,
            opts: &ZigCompileOptions,
        ) -> Result<String, MirZigBackendError> {
            let needs_io_print = function_has_io_or_print(func);
            let mut s = Self::generate_preamble(opts, needs_io_print);
            s.push_str(&Self::generate_function_code(func)?);
            Ok(s)
        }

        pub fn generate_module_source_with_options(
            module: &MirModule,
            opts: &ZigCompileOptions,
        ) -> Result<String, MirZigBackendError> {
            let needs_io_print = module_has_io_or_print(module);
            let mut s = Self::generate_preamble(opts, needs_io_print);
            for func in &module.functions {
                s.push_str(&Self::generate_function_code(func)?);
                s.push('\n');
            }
            Ok(s)
        }

        fn generate_preamble(opts: &ZigCompileOptions, needs_io_print: bool) -> String {
            let mut s = String::new();
            s.push_str("const std = @import(\"std\");\n\n");
            if needs_io_print {
                s.push_str("fn __lust_mir_print(val: anytype) void {\n");
                s.push_str("    std.debug.print(\"{any}\\n\", .{val});\n");
                s.push_str("}\n\n");
            }
            s.push_str("var __arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);\n");
            s.push_str("var __gas_remaining: i64 = 1000000000;\n");
            let mem_limit_str = match opts.max_memory_bytes {
                Some(b) => format!("{b}"),
                None => "std.math.maxInt(usize)".to_string(),
            };
            s.push_str(&format!("var __memory_limit: usize = {mem_limit_str};\n"));
            s.push_str("var __memory_used: usize = 0;\n");
            s.push_str("var __last_status: i32 = 0;\n\n");
            s.push_str("pub export fn lust_set_gas_limit(gas: i64) void { __gas_remaining = gas; }\n");
            s.push_str("pub export fn lust_get_gas_remaining() i64 { return __gas_remaining; }\n");
            s.push_str("pub export fn lust_set_memory_limit(bytes: usize) void { __memory_limit = bytes; }\n");
            s.push_str("pub export fn lust_get_memory_used() usize { return __memory_used; }\n");
            s.push_str("pub export fn lust_get_last_status() i32 { return __last_status; }\n");
            s.push_str("pub export fn lust_reset_arena() void {\n");
            s.push_str("    _ = __arena.reset(.retain_capacity);\n");
            s.push_str("    __memory_used = 0;\n");
            s.push_str("    __last_status = 0;\n");
            s.push_str("}\n\n");

            s.push_str("fn __lust_mir_div(a: i64, b: i64) i64 {\n");
            s.push_str("    if (b == 0) @panic(\"division by zero\");\n");
            s.push_str("    if (a == std.math.minInt(i64) and b == -1) return std.math.minInt(i64);\n");
            s.push_str("    return @divTrunc(a, b);\n");
            s.push_str("}\n\n");

            s.push_str("fn __lust_mir_rem(a: i64, b: i64) i64 {\n");
            s.push_str("    if (b == 0) @panic(\"division by zero\");\n");
            s.push_str("    if (a == std.math.minInt(i64) and b == -1) return 0;\n");
            s.push_str("    return @rem(a, b);\n");
            s.push_str("}\n\n");

            s.push_str("fn __lust_mir_floordiv(a: i64, b: i64) i64 {\n");
            s.push_str("    if (b == 0) @panic(\"division by zero\");\n");
            s.push_str("    if (a == std.math.minInt(i64) and b == -1) return std.math.minInt(i64);\n");
            s.push_str("    return @divFloor(a, b);\n");
            s.push_str("}\n\n");

            s.push_str("fn __lust_mir_pow(base: i64, exp: i64) i64 {\n");
            s.push_str("    if (exp < 0) @panic(\"Negative exponent in integer power\");\n");
            s.push_str("    var b = base;\n");
            s.push_str("    var e: u64 = @intCast(exp);\n");
            s.push_str("    var res: i64 = 1;\n");
            s.push_str("    while (e > 0) {\n");
            s.push_str("        if ((e & 1) == 1) {\n");
            s.push_str("            res = res *% b;\n");
            s.push_str("        }\n");
            s.push_str("        b = b *% b;\n");
            s.push_str("        e >>= 1;\n");
            s.push_str("    }\n");
            s.push_str("    return res;\n");
            s.push_str("}\n\n");

            s.push_str("fn __to_i64(val: anytype) i64 {\n");
            s.push_str("    const T = @TypeOf(val);\n");
            s.push_str("    if (T == i64) return val;\n");
            s.push_str("    if (T == []i64 or T == []const i64) return @as(i64, @bitCast(@intFromPtr(val.ptr)));\n");
            s.push_str("    if (@typeInfo(T) == .pointer) return @as(i64, @bitCast(@intFromPtr(val)));\n");
            s.push_str("    return @as(i64, @intCast(val));\n");
            s.push_str("}\n\n");

            s.push_str("fn __get_field_ptr(base: anytype, offset: usize) *i64 {\n");
            s.push_str("    const T = @TypeOf(base);\n");
            s.push_str("    if (T == []i64) {\n");
            s.push_str("        return &base[offset];\n");
            s.push_str("    } else if (T == i64) {\n");
            s.push_str("        const addr: usize = @bitCast(base);\n");
            s.push_str("        const ptr: [*]i64 = @ptrFromInt(addr);\n");
            s.push_str("        return &ptr[offset];\n");
            s.push_str("    } else {\n");
            s.push_str("        const ptr: [*]i64 = @ptrCast(base);\n");
            s.push_str("        return &ptr[offset];\n");
            s.push_str("    }\n");
            s.push_str("}\n\n");

            s.push_str("fn __alloc_slice(items: []const i64) []i64 {\n");
            s.push_str("    const bytes_needed = items.len * @sizeOf(i64);\n");
            s.push_str("    if (__memory_used + bytes_needed > __memory_limit) {\n");
            s.push_str("        __last_status = 2;\n");
            s.push_str("        return &[_]i64{};\n");
            s.push_str("    }\n");
            s.push_str("    const allocator = __arena.allocator();\n");
            s.push_str("    const slice = allocator.alloc(i64, items.len) catch {\n");
            s.push_str("        __last_status = 2;\n");
            s.push_str("        return &[_]i64{};\n");
            s.push_str("    };\n");
            s.push_str("    @memcpy(slice, items);\n");
            s.push_str("    __memory_used += bytes_needed;\n");
            s.push_str("    return slice;\n");
            s.push_str("}\n\n");
            s
        }

        fn generate_function_code(func: &MirFunction) -> Result<String, MirZigBackendError> {
            let mut s = String::new();
            let ret_ty = mir_to_zig_type(&func.signature.return_type);
            let has_slice = func.signature.param_types.iter().any(|p| matches!(p, MirType::Array(_)))
                || matches!(func.signature.return_type, MirType::Array(_));
            if has_slice {
                s.push_str(&format!("pub fn {}(", func.signature.name));
            } else {
                s.push_str(&format!("pub export fn {}(", func.signature.name));
            }

            for (i, p_id) in func.params.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                let p_ty = mir_to_zig_type(&func.locals[p_id.0].ty);
                s.push_str(&format!("arg{i}: {p_ty}"));
            }

            if has_slice {
                s.push_str(&format!(") {ret_ty} {{\n"));
            } else {
                s.push_str(&format!(") callconv(.c) {ret_ty} {{\n"));
            }

            // Declare locals
            for local in &func.locals {
                let ty = mir_to_zig_type(&local.ty);
                let default_val = default_zig_value(&local.ty);
                s.push_str(&format!("    var _{}: {ty} = {default_val}; _ = &_{};\n", local.id.0, local.id.0));
            }

            // Initialize parameter locals
            for (i, p_id) in func.params.iter().enumerate() {
                s.push_str(&format!("    _{} = arg{i};\n", p_id.0));
            }

            s.push_str("\n    var block_id: usize = 0; _ = &block_id;\n");
            s.push_str("    while (true) {\n");
            s.push_str("        switch (block_id) {\n");

            for block in &func.blocks {
                s.push_str(&format!("            {} => {{\n", block.id.0));
                for stmt in &block.statements {
                    match stmt {
                        Statement::Assign(place, rval, _) => {
                            let p_expr = lower_place(place);
                            let r_expr = lower_rvalue(rval)?;
                            s.push_str(&format!("                {p_expr} = {r_expr};\n"));
                        }
                        Statement::ResourceCheck(ResourceKind::Gas, amt, _) => {
                            let default_ret = if func.signature.return_type == MirType::Unit {
                                "return;".to_string()
                            } else {
                                format!("return {};", default_zig_value(&func.signature.return_type))
                            };
                            s.push_str(&format!("                __gas_remaining -= {amt};\n"));
                            s.push_str(&format!(
                                "                if (__gas_remaining < 0) {{\n                    __last_status = 1;\n                    {default_ret}\n                }}\n"
                            ));
                        }
                        _ => {}
                    }
                }

                if let Some(term) = &block.terminator {
                    match term {
                        Terminator::Goto(target, _) => {
                            s.push_str(&format!("                block_id = {};\n", target.0));
                            s.push_str("                continue;\n");
                        }
                        Terminator::Cond { cond, then_block, else_block, .. } => {
                            let c_expr = lower_operand(cond)?;
                            s.push_str(&format!("                if ({c_expr}) {{\n"));
                            s.push_str(&format!("                    block_id = {};\n", then_block.0));
                            s.push_str("                } else {\n");
                            s.push_str(&format!("                    block_id = {};\n", else_block.0));
                            s.push_str("                }\n");
                            s.push_str("                continue;\n");
                        }
                        Terminator::Switch { discr, cases, default, .. } => {
                            let d_expr = lower_operand(discr)?;
                            s.push_str(&format!("                switch ({d_expr}) {{\n"));
                            for (val, target) in cases {
                                s.push_str(&format!("                    {val} => {{\n                        block_id = {};\n                        continue;\n                    }},\n", target.0));
                            }
                            s.push_str(&format!("                    else => {{\n                        block_id = {};\n                        continue;\n                    }},\n", default.0));
                            s.push_str("                }\n");
                        }
                        Terminator::Call { callee, args, destination, success, .. } => {
                            let mut args_strs = Vec::new();
                            for arg in args {
                                args_strs.push(lower_operand(arg)?);
                            }
                            let dest_expr = lower_place(destination);
                            if callee == "print" || callee == "println" || callee.starts_with("io.") {
                                let first = args_strs.first().cloned().unwrap_or("0".to_string());
                                s.push_str(&format!("                __lust_mir_print({first});\n"));
                                s.push_str(&format!("                block_id = {};\n                continue;\n", success.0));
                            } else {
                                let call_expr = match callee.as_str() {
                                    "math.abs" | "abs" => format!("@as(i64, @intCast(@abs({})))", args_strs.first().cloned().unwrap_or_default()),
                                    "math.min" | "min" => format!("@min({}, {})", args_strs.get(0).cloned().unwrap_or_default(), args_strs.get(1).cloned().unwrap_or_default()),
                                    "math.max" | "max" => format!("@max({}, {})", args_strs.get(0).cloned().unwrap_or_default(), args_strs.get(1).cloned().unwrap_or_default()),
                                    "math.clamp" | "clamp" => format!("std.math.clamp({}, {}, {})", args_strs.get(0).cloned().unwrap_or_default(), args_strs.get(1).cloned().unwrap_or_default(), args_strs.get(2).cloned().unwrap_or_default()),
                                    "table.len" | "array.len" => format!("@as(i64, @intCast({}.len))", args_strs.first().cloned().unwrap_or_default()),
                                    _ => {
                                        let args_joined = args_strs.join(", ");
                                        format!("{callee}({args_joined})")
                                    }
                                };
                                s.push_str(&format!("                {dest_expr} = {call_expr};\n"));
                                let err_ret = if func.signature.return_type != MirType::Unit { " 0" } else { "" };
                                s.push_str(&format!("                if (__last_status != 0) return{err_ret};\n"));
                                s.push_str(&format!("                block_id = {};\n                continue;\n", success.0));
                            }
                        }
                        Terminator::Return(opt_op, _) => {
                            if func.signature.return_type != MirType::Unit {
                                let ret_expr = if let Some(op) = opt_op {
                                    lower_operand(op)?
                                } else {
                                    default_zig_value(&func.signature.return_type).into()
                                };
                                s.push_str(&format!("                return {ret_expr};\n"));
                            } else {
                                s.push_str("                return;\n");
                            }
                        }
                        Terminator::Unreachable(_) => {
                            s.push_str("                unreachable;\n");
                        }
                        _ => {
                            s.push_str("                return;\n");
                        }
                    }
                }
                s.push_str("            },\n");
            }

            s.push_str("            else => unreachable,\n");
            s.push_str("        }\n");
            s.push_str("    }\n");
            s.push_str("}\n");

            Ok(s)
        }

        pub fn compile_with_options(
            func: &MirFunction,
            out_dir: &Path,
            opts: &ZigCompileOptions,
        ) -> Result<PathBuf, MirZigBackendError> {
            let zig_source = Self::generate_source_with_options(func, opts)?;
            let zig_file = out_dir.join(format!("{}.zig", func.signature.name));
            fs::write(&zig_file, zig_source).map_err(|e| MirZigBackendError::WriteFile(format!("{e}")))?;

            if opts.emit_c_header {
                crate::mir::c_header_gen::CHeaderGenerator::write_header_for_function(func, out_dir)?;
            }

            Self::build_zig_lib(&func.signature.name, &zig_file, out_dir, opts)
        }

        pub fn compile_module_with_options(
            module: &MirModule,
            out_dir: &Path,
            opts: &ZigCompileOptions,
        ) -> Result<PathBuf, MirZigBackendError> {
            let zig_source = Self::generate_module_source_with_options(module, opts)?;
            let zig_file = out_dir.join(format!("{}.zig", module.name));
            fs::write(&zig_file, zig_source).map_err(|e| MirZigBackendError::WriteFile(format!("{e}")))?;

            if opts.emit_c_header {
                crate::mir::c_header_gen::CHeaderGenerator::write_header_for_module(module, out_dir)?;
            }

            Self::build_zig_lib(&module.name, &zig_file, out_dir, opts)
        }

        fn build_zig_lib(
            base_name: &str,
            zig_file: &Path,
            out_dir: &Path,
            opts: &ZigCompileOptions,
        ) -> Result<PathBuf, MirZigBackendError> {
            let (artifact_flag, out_name) = match opts.artifact_kind {
                ZigArtifactKind::CDylib => {
                    let dylib_name = if cfg!(target_os = "macos") {
                        format!("lib{base_name}.dylib")
                    } else if cfg!(target_os = "windows") {
                        format!("{base_name}.dll")
                    } else {
                        format!("lib{base_name}.so")
                    };
                    ("-dynamic", dylib_name)
                }
                ZigArtifactKind::StaticLib => {
                    let static_name = if cfg!(target_os = "windows") {
                        format!("{base_name}.lib")
                    } else {
                        format!("lib{base_name}.a")
                    };
                    ("-static", static_name)
                }
            };

            let out_path = out_dir.join(&out_name);

            let status = Command::new("zig")
                .args([
                    "build-lib",
                    artifact_flag,
                    zig_file.to_str().unwrap(),
                    "-OReleaseFast",
                    &format!("-femit-bin={}", out_path.to_str().unwrap()),
                ])
                .current_dir(out_dir)
                .status()
                .map_err(|e| MirZigBackendError::InvokeZig(format!("{e}")))?;

            if !status.success() {
                return Err(MirZigBackendError::CompilationFailed);
            }

            Ok(out_path)
        }

        pub fn compile_to_dylib(func: &MirFunction, out_dir: &Path) -> Result<PathBuf, MirZigBackendError> {
            Self::compile_with_options(func, out_dir, &ZigCompileOptions::default())
        }

        pub fn compile_module_to_dylib(module: &MirModule, out_dir: &Path) -> Result<PathBuf, MirZigBackendError> {
            Self::compile_module_with_options(module, out_dir, &ZigCompileOptions::default())
        }
    }

    fn mir_to_zig_type(ty: &MirType) -> &'static str {
        match ty {
            MirType::Int => "i64",
            MirType::Float => "f64",
            MirType::Bool => "bool",
            MirType::Unit => "void",
            MirType::Array(_) | MirType::Tuple(_) | MirType::Struct(_) => "[]i64",
            _ => "i64",
        }
    }

    fn default_zig_value(ty: &MirType) -> &'static str {
        match ty {
            MirType::Int => "0",
            MirType::Float => "0.0",
            MirType::Bool => "false",
            MirType::Unit => "{}",
            MirType::Array(_) | MirType::Tuple(_) | MirType::Struct(_) => "&[_]i64{}",
            _ => "0",
        }
    }

    fn lower_place(place: &Place) -> String {
        match place {
            Place::Local(loc) => format!("_{}", loc.0),
            Place::Index(base, idx) => {
                let b = lower_place(base);
                format!("{b}[@as(usize, @intCast(_{}))]", idx.0)
            }
            Place::Field(base, f) => {
                let b = lower_place(base);
                format!("__get_field_ptr({b}, {f}).*")
            }
        }
    }

    fn lower_operand(op: &Operand) -> Result<String, MirZigBackendError> {
        match op {
            Operand::Copy(place) | Operand::Move(place) => Ok(lower_place(place)),
            Operand::Constant(c) => match c {
                MirConstant::Int(i) => Ok(format!("{i}")),
                MirConstant::Float(f) => Ok(format!("{f:?}")),
                MirConstant::Bool(b) => Ok(format!("{b}")),
                MirConstant::Unit => Ok("{}".into()),
                _ => Err(MirZigBackendError::UnsupportedConstant(c.clone())),
            },
        }
    }

    fn lower_rvalue(rval: &Rvalue) -> Result<String, MirZigBackendError> {
        match rval {
            Rvalue::Use(op) => lower_operand(op),
            Rvalue::BinaryOp(binop, lhs, rhs) => {
                let l = lower_operand(lhs)?;
                let r = lower_operand(rhs)?;
                let s = match binop {
                    MirBinaryOp::Add => format!("{l} + {r}"),
                    MirBinaryOp::Sub => format!("{l} - {r}"),
                    MirBinaryOp::Mul => format!("{l} * {r}"),
                    MirBinaryOp::Div => format!("__lust_mir_div({l}, {r})"),
                    MirBinaryOp::Mod => format!("__lust_mir_rem({l}, {r})"),
                    MirBinaryOp::FloorDiv => format!("__lust_mir_floordiv({l}, {r})"),
                    MirBinaryOp::Pow => format!("__lust_mir_pow({l}, {r})"),
                    MirBinaryOp::BitAnd => format!("({l} & {r})"),
                    MirBinaryOp::BitOr => format!("({l} | {r})"),
                    MirBinaryOp::BitXor => format!("({l} ^ {r})"),
                    MirBinaryOp::Shl => format!("(@as(i64, {l}) << @as(u6, @intCast({r} & 63)))"),
                    MirBinaryOp::Shr => format!("(@as(i64, {l}) >> @as(u6, @intCast({r} & 63)))"),
                    MirBinaryOp::Eq => format!("{l} == {r}"),
                    MirBinaryOp::NotEq => format!("{l} != {r}"),
                    MirBinaryOp::Lt => format!("{l} < {r}"),
                    MirBinaryOp::LtEq => format!("{l} <= {r}"),
                    MirBinaryOp::Gt => format!("{l} > {r}"),
                    MirBinaryOp::GtEq => format!("{l} >= {r}"),
                    MirBinaryOp::And => format!("{l} and {r}"),
                    MirBinaryOp::Or => format!("{l} or {r}"),
                    MirBinaryOp::Concat => format!("__lust_mir_str_concat({l}, {r})"),
                };
                Ok(s)
            }
            Rvalue::UnaryOp(unop, op) => {
                let v = lower_operand(op)?;
                let s = match unop {
                    MirUnaryOp::Neg => format!("-{v}"),
                    MirUnaryOp::Not => format!("!{v}"),
                    MirUnaryOp::BitNot => format!("~{v}"),
                    MirUnaryOp::Len => format!("@as(i64, @intCast({v}.len))"),
                };
                Ok(s)
            }
            Rvalue::Aggregate(MirType::Array(_) | MirType::Tuple(_) | MirType::Struct(_), elements) => {
                let mut elems = String::new();
                for (i, el) in elements.iter().enumerate() {
                    if i > 0 {
                        elems.push_str(", ");
                    }
                    let operand_str = lower_operand(el)?;
                    elems.push_str(&format!("__to_i64({operand_str})"));
                }
                Ok(format!("__alloc_slice(&[_]i64{{{elems}}})"))
            }
            Rvalue::Closure { .. } => Err(MirZigBackendError::UnsupportedConstant(MirConstant::Unit)),
            _ => Err(MirZigBackendError::UnsupportedConstant(MirConstant::Unit)),
        }
    }
}
