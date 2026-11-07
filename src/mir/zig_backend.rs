#[cfg(feature = "std")]
pub mod zig_engine {
    use crate::mir::ir::*;
    use alloc::{format, string::String};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    pub struct MirToZigCompiler;

    impl MirToZigCompiler {
        pub fn generate_source(func: &MirFunction) -> String {
            let mut s = String::new();
            s.push_str("const std = @import(\"std\");\n\n");
            s.push_str("var __arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);\n");
            s.push_str("var __gas_remaining: i64 = 1000000000;\n");
            s.push_str("pub export fn lust_set_gas_limit(gas: i64) void { __gas_remaining = gas; }\n");
            s.push_str("pub export fn lust_get_gas_remaining() i64 { return __gas_remaining; }\n");
            s.push_str("fn __alloc_slice(items: []const i64) []i64 {\n");
            s.push_str("    const allocator = __arena.allocator();\n");
            s.push_str("    const slice = allocator.alloc(i64, items.len) catch unreachable;\n");
            s.push_str("    @memcpy(slice, items);\n");
            s.push_str("    return slice;\n");
            s.push_str("}\n\n");

            let ret_ty = mir_to_zig_type(&func.signature.return_type);
            s.push_str(&format!("pub export fn {}(", func.signature.name));

            for (i, p_id) in func.params.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                let p_ty = mir_to_zig_type(&func.locals[p_id.0].ty);
                s.push_str(&format!("arg{i}: {p_ty}"));
            }

            s.push_str(&format!(") callconv(.c) {ret_ty} {{\n"));

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

            s.push_str("\n    var block_id: usize = 0;\n");
            s.push_str("    while (true) {\n");
            s.push_str("        switch (block_id) {\n");

            for block in &func.blocks {
                s.push_str(&format!("            {} => {{\n", block.id.0));
                for stmt in &block.statements {
                    match stmt {
                        Statement::Assign(place, rval, _) => {
                            let p_expr = lower_place(place);
                            let r_expr = lower_rvalue(rval);
                            s.push_str(&format!("                {p_expr} = {r_expr};\n"));
                        }
                        Statement::ResourceCheck(ResourceKind::Gas, amt, _) => {
                            s.push_str(&format!("                __gas_remaining -= {amt};\n"));
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
                            let c_expr = lower_operand(cond);
                            s.push_str(&format!("                if ({c_expr}) {{\n"));
                            s.push_str(&format!("                    block_id = {};\n", then_block.0));
                            s.push_str("                } else {\n");
                            s.push_str(&format!("                    block_id = {};\n", else_block.0));
                            s.push_str("                }\n");
                            s.push_str("                continue;\n");
                        }
                        Terminator::Return(opt_op, _) => {
                            if func.signature.return_type != MirType::Unit {
                                let ret_expr = if let Some(op) = opt_op {
                                    lower_operand(op)
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

            s
        }

        pub fn compile_to_dylib(func: &MirFunction, out_dir: &Path) -> Result<PathBuf, String> {
            let zig_source = Self::generate_source(func);
            let zig_file = out_dir.join(format!("{}.zig", func.signature.name));
            fs::write(&zig_file, zig_source).map_err(|e| format!("Failed to write zig file: {e}"))?;

            let dylib_name = if cfg!(target_os = "macos") {
                format!("lib{}.dylib", func.signature.name)
            } else if cfg!(target_os = "windows") {
                format!("{}.dll", func.signature.name)
            } else {
                format!("lib{}.so", func.signature.name)
            };
            let dylib_path = out_dir.join(&dylib_name);

            let status = Command::new("zig")
                .args([
                    "build-lib",
                    "-dynamic",
                    zig_file.to_str().unwrap(),
                    "-OReleaseFast",
                    &format!("-femit-bin={}", dylib_path.to_str().unwrap()),
                ])
                .current_dir(out_dir)
                .status()
                .map_err(|e| format!("Failed to invoke zig: {e}"))?;

            if !status.success() {
                return Err("Zig compilation failed".into());
            }

            Ok(dylib_path)
        }
    }

    fn mir_to_zig_type(ty: &MirType) -> &'static str {
        match ty {
            MirType::Int => "i64",
            MirType::Float => "f64",
            MirType::Bool => "bool",
            MirType::Unit => "void",
            MirType::Array(_) => "[]i64",
            _ => "i64",
        }
    }

    fn default_zig_value(ty: &MirType) -> &'static str {
        match ty {
            MirType::Int => "0",
            MirType::Float => "0.0",
            MirType::Bool => "false",
            MirType::Unit => "{}",
            MirType::Array(_) => "&[_]i64{}",
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
                format!("{b}.f{f}")
            }
        }
    }

    fn lower_operand(op: &Operand) -> String {
        match op {
            Operand::Copy(place) | Operand::Move(place) => lower_place(place),
            Operand::Constant(c) => match c {
                MirConstant::Int(i) => format!("{i}"),
                MirConstant::Float(f) => format!("{f:?}"),
                MirConstant::Bool(b) => format!("{b}"),
                MirConstant::Unit => "{}".into(),
                MirConstant::String(s) => format!("\"{s}\""),
            },
        }
    }

    fn lower_rvalue(rval: &Rvalue) -> String {
        match rval {
            Rvalue::Use(op) => lower_operand(op),
            Rvalue::BinaryOp(binop, lhs, rhs) => {
                let l = lower_operand(lhs);
                let r = lower_operand(rhs);
                match binop {
                    MirBinaryOp::Add => format!("{l} + {r}"),
                    MirBinaryOp::Sub => format!("{l} - {r}"),
                    MirBinaryOp::Mul => format!("{l} * {r}"),
                    MirBinaryOp::Div => format!("@divTrunc({l}, {r})"),
                    MirBinaryOp::Mod => format!("@rem({l}, {r})"),
                    MirBinaryOp::Eq => format!("{l} == {r}"),
                    MirBinaryOp::NotEq => format!("{l} != {r}"),
                    MirBinaryOp::Lt => format!("{l} < {r}"),
                    MirBinaryOp::LtEq => format!("{l} <= {r}"),
                    MirBinaryOp::Gt => format!("{l} > {r}"),
                    MirBinaryOp::GtEq => format!("{l} >= {r}"),
                    MirBinaryOp::And => format!("{l} and {r}"),
                    MirBinaryOp::Or => format!("{l} or {r}"),
                    _ => format!("{l} + {r}"),
                }
            }
            Rvalue::UnaryOp(unop, op) => {
                let v = lower_operand(op);
                match unop {
                    MirUnaryOp::Neg => format!("-{v}"),
                    MirUnaryOp::Not => format!("!{v}"),
                    MirUnaryOp::Len => format!("@as(i64, @intCast({v}.len))"),
                }
            }
            Rvalue::Aggregate(MirType::Array(_), elements) => {
                let mut elems = String::new();
                for (i, el) in elements.iter().enumerate() {
                    if i > 0 {
                        elems.push_str(", ");
                    }
                    elems.push_str(&lower_operand(el));
                }
                format!("__alloc_slice(&[_]i64{{{elems}}})")
            }
            _ => "0".into(),
        }
    }
}
