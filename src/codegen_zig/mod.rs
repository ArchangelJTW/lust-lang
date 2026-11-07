pub mod runtime;

use crate::ast::{
    BinaryOp, EnumDef, Expr, ExprKind, FunctionDef, Item, ItemKind, Literal, Pattern, Stmt,
    StmtKind, StructDef, Type, TypeKind, UnaryOp,
};
use crate::error::{LustError, Result};
use runtime::{RuntimeOptions, generate_runtime};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildTargetKind {
    Executable,
    SharedLibrary,
    StaticLibrary,
}

#[derive(Debug, Clone)]
pub struct CompileOptions {
    pub target_kind: BuildTargetKind,
    pub release_mode: String,
    pub emit_zig: bool,
    pub emit_header: Option<PathBuf>,
    pub max_memory: Option<String>,
    pub max_memory_bytes: Option<usize>,
    pub tree_shake: bool,
    pub strip: bool,
    pub target_triple: Option<String>,
    pub cache_dir: Option<PathBuf>,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            target_kind: BuildTargetKind::Executable,
            release_mode: "ReleaseFast".to_string(),
            emit_zig: false,
            emit_header: None,
            max_memory: None,
            max_memory_bytes: None,
            tree_shake: true,
            strip: false,
            target_triple: None,
            cache_dir: None,
        }
    }
}

pub fn parse_memory_size(s: &str) -> Option<usize> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let s_upper = s.to_uppercase();
    if let Some(num_str) = s_upper.strip_suffix("GB").or_else(|| s_upper.strip_suffix('G')) {
        let num: f64 = num_str.trim().parse().ok()?;
        Some((num * 1024.0 * 1024.0 * 1024.0) as usize)
    } else if let Some(num_str) = s_upper.strip_suffix("MB").or_else(|| s_upper.strip_suffix('M')) {
        let num: f64 = num_str.trim().parse().ok()?;
        Some((num * 1024.0 * 1024.0) as usize)
    } else if let Some(num_str) = s_upper.strip_suffix("KB").or_else(|| s_upper.strip_suffix('K')) {
        let num: f64 = num_str.trim().parse().ok()?;
        Some((num * 1024.0) as usize)
    } else if let Some(num_str) = s_upper.strip_suffix('B') {
        let num: usize = num_str.trim().parse().ok()?;
        Some(num)
    } else {
        s.parse::<usize>().ok()
    }
}

pub struct ZigCodegen {
    out: String,
    indent_level: usize,
    known_structs: HashSet<String>,
    known_enums: HashSet<String>,
    temp_var_counter: usize,
    runtime_opts: RuntimeOptions,
}

impl Default for ZigCodegen {
    fn default() -> Self {
        Self::new()
    }
}

impl ZigCodegen {
    pub fn new() -> Self {
        Self::with_options(RuntimeOptions::default())
    }

    pub fn with_options(opts: RuntimeOptions) -> Self {
        Self {
            out: String::new(),
            indent_level: 0,
            known_structs: HashSet::new(),
            known_enums: HashSet::new(),
            temp_var_counter: 0,
            runtime_opts: opts,
        }
    }

    fn next_temp_var(&mut self) -> String {
        self.temp_var_counter += 1;
        format!("__lust_tmp_{}", self.temp_var_counter)
    }

    fn indent(&mut self) {
        self.indent_level += 1;
    }

    fn dedent(&mut self) {
        if self.indent_level > 0 {
            self.indent_level -= 1;
        }
    }

    fn emit_indent(&mut self) {
        for _ in 0..self.indent_level {
            self.out.push_str("    ");
        }
    }

    fn emit_line(&mut self, line: &str) {
        self.emit_indent();
        self.out.push_str(line);
        self.out.push('\n');
    }

    pub fn generate_program(&mut self, items: &[Item]) -> Result<String> {
        self.out.clear();
        self.out.push_str(&generate_runtime(&self.runtime_opts));
        self.out.push('\n');

        // Collect all declared struct and enum names first
        for item in items {
            match &item.kind {
                ItemKind::Struct(s) => {
                    self.known_structs.insert(s.name.clone());
                }
                ItemKind::Enum(e) => {
                    self.known_enums.insert(e.name.clone());
                }
                _ => {}
            }
        }

        // 1. Emit enum definitions (tagged unions)
        for item in items {
            if let ItemKind::Enum(e) = &item.kind {
                self.emit_enum(e)?;
            }
        }

        // 2. Emit struct definitions
        for item in items {
            if let ItemKind::Struct(s) = &item.kind {
                let mut methods = Vec::new();
                for other in items {
                    if let ItemKind::Impl(impl_block) = &other.kind {
                        if let TypeKind::Named(target_name) = &impl_block.target_type.kind {
                            if target_name == &s.name {
                                for m in &impl_block.methods {
                                    methods.push(m);
                                }
                            }
                        }
                    }
                }
                self.emit_struct(s, &methods)?;
            }
        }

        // 3. Emit functions
        for item in items {
            if let ItemKind::Function(f) = &item.kind {
                self.emit_function(f)?;
            }
        }

        // 4. Emit script body (main entrypoint or library init)
        let mut script_stmts: Vec<&Stmt> = Vec::new();
        for item in items {
            if let ItemKind::Script(stmts) = &item.kind {
                for stmt in stmts {
                    script_stmts.push(stmt);
                }
            }
        }

        if self.runtime_opts.is_library {
            self.emit_library_init(&script_stmts)?;
        } else {
            self.emit_main(&script_stmts)?;
        }

        Ok(self.out.clone())
    }

    fn emit_enum(&mut self, e: &EnumDef) -> Result<()> {
        self.emit_line(&format!("pub const {} = union(enum) {{", e.name));
        self.indent();
        for v in &e.variants {
            if let Some(fields) = &v.fields {
                if fields.is_empty() {
                    self.emit_line(&format!("{},", v.name));
                } else if fields.len() == 1 {
                    let ty_str = self.map_type(&fields[0], false)?;
                    self.emit_line(&format!("{}: {},", v.name, ty_str));
                } else {
                    let mut field_tys = Vec::new();
                    for f in fields {
                        field_tys.push(self.map_type(f, false)?);
                    }
                    self.emit_line(&format!(
                        "{}: struct {{ {} }},",
                        v.name,
                        field_tys.join(", ")
                    ));
                }
            } else {
                self.emit_line(&format!("{},", v.name));
            }
        }
        self.dedent();
        self.emit_line("};\n");
        Ok(())
    }

    fn emit_struct(&mut self, s: &StructDef, methods: &[&FunctionDef]) -> Result<()> {
        let modifier = if self.runtime_opts.is_library { "pub const" } else { "pub const" };
        let struct_kw = if self.runtime_opts.is_library { "extern struct" } else { "struct" };
        self.emit_line(&format!("{} {} = {} {{", modifier, s.name, struct_kw));
        self.indent();
        for field in &s.fields {
            let zig_ty = self.map_type(&field.ty, true)?;
            self.emit_line(&format!("{}: {},", field.name, zig_ty));
        }

        for m in methods {
            self.emit_method(m)?;
        }

        self.dedent();
        self.emit_line("};\n");
        Ok(())
    }

    fn emit_method(&mut self, f: &FunctionDef) -> Result<()> {
        let mut params_str = Vec::new();
        for p in &f.params {
            if p.name == "self" {
                params_str.push("self: *@This()".to_string());
            } else {
                let p_ty = self.map_type(&p.ty, false)?;
                params_str.push(format!("__lust_param_{}: {}", p.name, p_ty));
            }
        }

        let ret_ty = if let Some(ret) = &f.return_type {
            let t = self.map_type(ret, false)?;
            format!("!{}", t)
        } else {
            "!void".to_string()
        };

        self.emit_line(&format!(
            "pub fn {}({}) {} {{",
            f.name,
            params_str.join(", "),
            ret_ty
        ));
        self.indent();
        for p in &f.params {
            if p.name != "self" {
                let ident = zig_ident(&p.name);
                self.emit_line(&format!("var {} = __lust_param_{}; _ = &{};", ident, p.name, ident));
            }
        }
        for stmt in &f.body {
            self.emit_stmt(stmt)?;
        }
        self.dedent();
        self.emit_line("}\n");
        Ok(())
    }

    fn map_type(&self, ty: &Type, is_field: bool) -> Result<String> {
        match &ty.kind {
            TypeKind::Int => Ok("i64".to_string()),
            TypeKind::Float => Ok("f64".to_string()),
            TypeKind::Bool => Ok("bool".to_string()),
            TypeKind::String => Ok("[]const u8".to_string()),
            TypeKind::Unit => Ok("void".to_string()),
            TypeKind::Named(name) => {
                if self.known_structs.contains(name) {
                    if is_field {
                        Ok(format!("*{}", name))
                    } else if self.runtime_opts.is_library {
                        Ok(format!("*const {}", name))
                    } else {
                        Ok(format!("*{}", name))
                    }
                } else if self.known_enums.contains(name) {
                    Ok(name.clone())
                } else {
                    Ok(name.clone())
                }
            }
            TypeKind::Option(inner) => {
                let inner_ty = self.map_type(inner, false)?;
                if inner_ty.starts_with('*') {
                    Ok(format!("?{}", inner_ty))
                } else if self.known_structs.contains(inner_ty.as_str()) {
                    Ok(format!("?*{}", inner_ty))
                } else {
                    Ok(format!("?{}", inner_ty))
                }
            }
            TypeKind::Array(inner) => {
                let inner_ty = self.map_type(inner, false)?;
                Ok(format!("LustArray({})", inner_ty))
            }
            TypeKind::Map(k, v) => {
                let k_ty = self.map_type(k, false)?;
                let v_ty = self.map_type(v, false)?;
                Ok(format!("LustMap({}, {})", k_ty, v_ty))
            }
            TypeKind::Tuple(types) => {
                let mut t_strs = Vec::new();
                for t in types {
                    t_strs.push(self.map_type(t, false)?);
                }
                Ok(format!("struct {{ {} }}", t_strs.join(", ")))
            }
            TypeKind::Result(ok_t, err_t) => {
                let ok_ty = self.map_type(ok_t, false)?;
                let err_ty = self.map_type(err_t, false)?;
                Ok(format!("LustResult({}, {})", ok_ty, err_ty))
            }
            TypeKind::Function { params, return_type } => {
                let mut p_strs = Vec::new();
                for p in params {
                    p_strs.push(self.map_type(p, false)?);
                }
                let ret = self.map_type(return_type, false)?;
                Ok(format!("*const fn({}) {}", p_strs.join(", "), ret))
            }
            TypeKind::Ref(inner) | TypeKind::MutRef(inner) => {
                let inner_ty = self.map_type(inner, false)?;
                Ok(format!("*{}", inner_ty))
            }
            TypeKind::Pointer { mutable, pointee } => {
                let inner_ty = self.map_type(pointee, false)?;
                if *mutable {
                    Ok(format!("*{}", inner_ty))
                } else {
                    Ok(format!("*const {}", inner_ty))
                }
            }
            _ => Ok("anyopaque".to_string()),
        }
    }

    fn emit_function(&mut self, f: &FunctionDef) -> Result<()> {
        let mut params_str = Vec::new();
        for p in &f.params {
            let p_ty = self.map_type(&p.ty, false)?;
            params_str.push(format!("__lust_param_{}: {}", p.name, p_ty));
        }

        let ret_base = if let Some(ret) = &f.return_type {
            self.map_type(ret, false)?
        } else {
            "void".to_string()
        };

        let clean_name = f.name.rsplit('.').next().unwrap_or(&f.name);

        if self.runtime_opts.is_library {
            // In library mode, export functions with C ABI
            self.emit_line(&format!(
                "pub export fn {}({}) callconv(.c) {} {{",
                clean_name,
                params_str.join(", "),
                ret_base
            ));
        } else {
            self.emit_line(&format!(
                "fn {}({}) !{} {{",
                clean_name,
                params_str.join(", "),
                ret_base
            ));
        }
        self.indent();

        for p in &f.params {
            let ident = zig_ident(&p.name);
            self.emit_line(&format!("var {} = __lust_param_{}; _ = &{};", ident, p.name, ident));
        }

        for stmt in &f.body {
            self.emit_stmt(stmt)?;
        }

        self.dedent();
        self.emit_line("}\n");
        Ok(())
    }

    fn emit_library_init(&mut self, stmts: &[&Stmt]) -> Result<()> {
        self.emit_line("pub var __lust_initialized: bool = false;\n");
        self.emit_line("pub export fn __lust_init() callconv(.c) void {");
        self.indent();
        self.emit_line("if (__lust_initialized) return;");
        self.emit_line("__lust_initialized = true;");
        if self.runtime_opts.max_memory_bytes.is_some() {
            self.emit_line("__lust_limiter = LimitedAllocator.init(std.heap.page_allocator, __LUST_MAX_MEMORY);");
            self.emit_line("__lust_arena = std.heap.ArenaAllocator.init(__lust_limiter.allocator());");
        } else {
            self.emit_line("__lust_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);");
        }
        self.emit_line("__lust_allocator = __lust_arena.allocator();\n");

        for stmt in stmts {
            self.emit_stmt(stmt)?;
        }

        self.dedent();
        self.emit_line("}\n");
        Ok(())
    }

    fn emit_main(&mut self, stmts: &[&Stmt]) -> Result<()> {
        self.emit_line("pub fn main() !void {");
        self.indent();
        if self.runtime_opts.max_memory_bytes.is_some() {
            self.emit_line("__lust_limiter = LimitedAllocator.init(std.heap.page_allocator, __LUST_MAX_MEMORY);");
            self.emit_line("__lust_arena = std.heap.ArenaAllocator.init(__lust_limiter.allocator());");
        } else {
            self.emit_line("__lust_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);");
        }
        self.emit_line("defer __lust_arena.deinit();");
        self.emit_line("__lust_allocator = __lust_arena.allocator();\n");

        for stmt in stmts {
            self.emit_stmt(stmt)?;
        }

        self.dedent();
        self.emit_line("}\n");
        Ok(())
    }

    fn emit_stmt(&mut self, stmt: &Stmt) -> Result<()> {
        match &stmt.kind {
            StmtKind::Local {
                bindings,
                initializer,
            } => {
                for (idx, b) in bindings.iter().enumerate() {
                    let init_expr = initializer.as_ref().and_then(|inits| inits.get(idx));
                    if let Some(expr) = init_expr {
                        if let ExprKind::Array(elements) = &expr.kind {
                            let elem_type = match &b.type_annotation {
                                Some(Type {
                                    kind: TypeKind::Array(inner),
                                    ..
                                }) => self.map_type(inner, false)?,
                                _ => {
                                    if let Some(first) = elements.first() {
                                        match &first.kind {
                                            ExprKind::Literal(Literal::String(_)) => "[]const u8".to_string(),
                                            ExprKind::Literal(Literal::Float(_)) => "f64".to_string(),
                                            ExprKind::Literal(Literal::Bool(_)) => "bool".to_string(),
                                            _ => "i64".to_string(),
                                        }
                                    } else {
                                        "i64".to_string()
                                    }
                                }
                            };
                            self.emit_line(&format!(
                                "var {} = LustArray({}).init(__lust_allocator);",
                                b.name, elem_type
                            ));
                            for el in elements {
                                let el_str = self.emit_expr_to_string(el)?;
                                self.emit_line(&format!("try {}.push({});", b.name, el_str));
                            }
                        } else {
                            let val_str = self.emit_expr_to_string(expr)?;
                            if let Some(ty) = &b.type_annotation {
                                let zig_ty = self.map_type(ty, false)?;
                                self.emit_line(&format!("var {}: {} = {};", b.name, zig_ty, val_str));
                            } else {
                                let inferred = match &expr.kind {
                                    ExprKind::Literal(Literal::Integer(_)) => Some("i64"),
                                    ExprKind::Literal(Literal::Float(_)) => Some("f64"),
                                    ExprKind::Literal(Literal::Bool(_)) => Some("bool"),
                                    _ => None,
                                };
                                if let Some(ty_str) = inferred {
                                    self.emit_line(&format!("var {}: {} = {};", b.name, ty_str, val_str));
                                } else {
                                    self.emit_line(&format!("var {} = {};", b.name, val_str));
                                }
                            }
                        }
                    } else if let Some(ty) = &b.type_annotation {
                        let zig_ty = self.map_type(ty, false)?;
                        self.emit_line(&format!("var {}: {} = undefined;", b.name, zig_ty));
                    } else {
                        self.emit_line(&format!("var {} = undefined;", b.name));
                    }
                    self.emit_line(&format!("_ = &{};", b.name));
                }
            }
            StmtKind::Assign { targets, values } => {
                for (t, v) in targets.iter().zip(values.iter()) {
                    let val_str = self.emit_expr_to_string(v)?;
                    if let ExprKind::Index { object, index } = &t.kind {
                        let obj_str = self.emit_expr_to_string(object)?;
                        let idx_str = self.emit_expr_to_string(index)?;
                        self.emit_line(&format!("{}.set({}, {});", obj_str, idx_str, val_str));
                    } else {
                        let target_str = self.emit_expr_to_string(t)?;
                        self.emit_line(&format!("{} = {};", target_str, val_str));
                    }
                }
            }
            StmtKind::CompoundAssign { target, op, value } => {
                let target_str = self.emit_expr_to_string(target)?;
                let val_str = self.emit_expr_to_string(value)?;
                match op {
                    BinaryOp::Add => self.emit_line(&format!("{} += {};", target_str, val_str)),
                    BinaryOp::Sub => self.emit_line(&format!("{} -= {};", target_str, val_str)),
                    BinaryOp::Mul => self.emit_line(&format!("{} *= {};", target_str, val_str)),
                    BinaryOp::Div => self.emit_line(&format!("{} = __lust_div({}, {});", target_str, target_str, val_str)),
                    BinaryOp::Mod => self.emit_line(&format!("{} = @mod({}, {});", target_str, target_str, val_str)),
                    BinaryOp::Concat => self.emit_line(&format!("{} = try __lust_str_concat({}, {});", target_str, target_str, val_str)),
                    _ => self.emit_line(&format!("{} = {};", target_str, val_str)),
                }
            }
            StmtKind::While { condition, body } => {
                let cond_str = self.emit_expr_to_string(condition)?;
                self.emit_line(&format!("while ({}) {{", cond_str));
                self.indent();
                for s in body {
                    self.emit_stmt(s)?;
                }
                self.dedent();
                self.emit_line("}");
            }
            StmtKind::ForNumeric {
                variable,
                start,
                end,
                step,
                body,
            } => {
                let s_str = self.emit_expr_to_string(start)?;
                let e_str = self.emit_expr_to_string(end)?;
                let step_str = if let Some(st) = step {
                    self.emit_expr_to_string(st)?
                } else {
                    "1".to_string()
                };
                self.emit_line(&format!("var {}: i64 = {};", variable, s_str));
                self.emit_line(&format!(
                    "while ({} <= {}) : ({} += {}) {{",
                    variable, e_str, variable, step_str
                ));
                self.indent();
                self.emit_line(&format!("_ = &{};", variable));
                for s in body {
                    self.emit_stmt(s)?;
                }
                self.dedent();
                self.emit_line("}");
            }
            StmtKind::ForIn {
                variables,
                iterator,
                body,
            } => {
                let col_expr = match &iterator.kind {
                    ExprKind::Call { callee, args, .. } => {
                        if let ExprKind::Identifier(fn_name) = &callee.kind {
                            if (fn_name == "ipairs" || fn_name == "pairs") && !args.is_empty() {
                                &args[0]
                            } else {
                                iterator
                            }
                        } else {
                            iterator
                        }
                    }
                    _ => iterator,
                };
                let col_str = self.emit_expr_to_string(col_expr)?;

                if variables.len() == 1 {
                    let var_name = zig_ident(&variables[0]);
                    self.emit_line(&format!("for ({}.items) |__{}_val| {{", col_str, var_name));
                    self.indent();
                    self.emit_line(&format!("var {} = __{}_val; _ = &{};", var_name, var_name, var_name));
                    for s in body {
                        self.emit_stmt(s)?;
                    }
                    self.dedent();
                    self.emit_line("}");
                } else if variables.len() >= 2 {
                    let idx_name = zig_ident(&variables[0]);
                    let val_name = zig_ident(&variables[1]);
                    self.emit_line(&format!("for ({}.items, 0..) |__{}_val, __{}_usize| {{", col_str, val_name, idx_name));
                    self.indent();
                    self.emit_line(&format!("var {} = @as(i64, @intCast(__{}_usize)); _ = &{};", idx_name, idx_name, idx_name));
                    self.emit_line(&format!("var {} = __{}_val; _ = &{};", val_name, val_name, val_name));
                    for s in body {
                        self.emit_stmt(s)?;
                    }
                    self.dedent();
                    self.emit_line("}");
                }
            }
            StmtKind::Break => {
                self.emit_line("break;");
            }
            StmtKind::Continue => {
                self.emit_line("continue;");
            }
            StmtKind::Block(stmts) => {
                self.emit_line("{");
                self.indent();
                for s in stmts {
                    self.emit_stmt(s)?;
                }
                self.dedent();
                self.emit_line("}");
            }
            StmtKind::If {
                condition,
                then_block,
                elseif_branches,
                else_block,
            } => {
                self.emit_if_statement(condition, then_block, elseif_branches, else_block.as_deref())?;
            }
            StmtKind::Return(exprs) => {
                if exprs.is_empty() {
                    self.emit_line("return;");
                } else if exprs.len() == 1 {
                    let ret_val = self.emit_expr_to_string(&exprs[0])?;
                    self.emit_line(&format!("return {};", ret_val));
                } else {
                    let mut vals = Vec::new();
                    for e in exprs {
                        vals.push(self.emit_expr_to_string(e)?);
                    }
                    self.emit_line(&format!("return .{{ {} }};", vals.join(", ")));
                }
            }
            StmtKind::Expr(expr) => {
                let expr_str = self.emit_expr_to_string(expr)?;
                self.emit_line(&format!("_ = {};", expr_str));
            }
        }
        Ok(())
    }

    fn emit_if_branch(
        &mut self,
        prefix: &str,
        condition: &Expr,
        body: &[Stmt],
    ) -> Result<()> {
        if let ExprKind::TypeCheck { expr, check_type } = &condition.kind {
            if let TypeKind::Named(type_name) = &check_type.kind {
                let expr_str = self.emit_expr_to_string(expr)?;
                self.emit_line(&format!("{} ({} == .{}) {{", prefix, expr_str, type_name));
                self.indent();
                for s in body {
                    self.emit_stmt(s)?;
                }
                self.dedent();
                return Ok(());
            }
        }

        if let ExprKind::IsPattern { expr, pattern } = &condition.kind {
            match pattern {
                Pattern::Enum {
                    variant,
                    bindings,
                    ..
                } => {
                    let binding_name = bindings
                        .first()
                        .and_then(|b| {
                            if let Pattern::Identifier(name) = b {
                                Some(name.as_str())
                            } else {
                                None
                            }
                        })
                        .unwrap_or("val");

                    if variant == "Some" {
                        let expr_str = self.emit_expr_to_string(expr)?;
                        self.emit_line(&format!("{} ({}) |{}| {{", prefix, expr_str, binding_name));
                    } else if variant == "Ok" {
                        if let ExprKind::Index { object, index } = &expr.kind {
                            let obj_str = self.emit_expr_to_string(object)?;
                            let idx_str = self.emit_expr_to_string(index)?;
                            self.emit_line(&format!(
                                "{} ({}.get({})) |{}| {{",
                                prefix, obj_str, idx_str, binding_name
                            ));
                        } else {
                            let expr_str = self.emit_expr_to_string(expr)?;
                            self.emit_line(&format!("{} ({} == .Ok) {{", prefix, expr_str));
                            if !bindings.is_empty() {
                                self.indent();
                                self.emit_line(&format!("const {} = {}.Ok;", binding_name, expr_str));
                                self.emit_line(&format!("_ = &{};", binding_name));
                                self.dedent();
                            }
                        }
                    } else if variant == "Err" {
                        let esc_binding = zig_ident(binding_name);
                        if let ExprKind::Index { object, index } = &expr.kind {
                            let obj_str = self.emit_expr_to_string(object)?;
                            let idx_str = self.emit_expr_to_string(index)?;
                            self.emit_line(&format!(
                                "{} ({}.get({}) == null) {{",
                                prefix, obj_str, idx_str
                            ));
                            self.indent();
                            self.emit_line(&format!(
                                "const {} = struct {{ index: i64, length: i64 }}{{ .index = {}, .length = {}.len() }};",
                                esc_binding, idx_str, obj_str
                            ));
                            self.emit_line(&format!("_ = &{};", esc_binding));
                            self.dedent();
                        } else {
                            let expr_str = self.emit_expr_to_string(expr)?;
                            self.emit_line(&format!("{} ({} == .Err) {{", prefix, expr_str));
                            if !bindings.is_empty() {
                                self.indent();
                                self.emit_line(&format!("const {} = {}.Err;", esc_binding, expr_str));
                                self.emit_line(&format!("_ = &{};", esc_binding));
                                self.dedent();
                            }
                        }
                    } else {
                        let expr_str = self.emit_expr_to_string(expr)?;
                        self.emit_line(&format!("{} ({} == .{}) {{", prefix, expr_str, variant));
                        if !bindings.is_empty() {
                            if bindings.len() == 1 {
                                self.indent();
                                self.emit_line(&format!(
                                    "const {} = {}.{};",
                                    binding_name, expr_str, variant
                                ));
                                self.emit_line(&format!("_ = &{};", binding_name));
                                self.dedent();
                            }
                        }
                    }
                }
                Pattern::Identifier(variant_name) => {
                    let expr_str = self.emit_expr_to_string(expr)?;
                    self.emit_line(&format!("{} ({} == .{}) {{", prefix, expr_str, variant_name));
                }
                _ => {
                    let cond_str = self.emit_expr_to_string(condition)?;
                    self.emit_line(&format!("{} ({}) {{", prefix, cond_str));
                }
            }
        } else {
            let cond_str = self.emit_expr_to_string(condition)?;
            self.emit_line(&format!("{} ({}) {{", prefix, cond_str));
        }

        self.indent();
        for s in body {
            self.emit_stmt(s)?;
        }
        self.dedent();
        Ok(())
    }

    fn emit_if_statement(
        &mut self,
        condition: &Expr,
        then_block: &[Stmt],
        elseif_branches: &[(Expr, Vec<Stmt>)],
        else_block: Option<&[Stmt]>,
    ) -> Result<()> {
        self.emit_if_branch("if", condition, then_block)?;

        for (elseif_cond, elseif_body) in elseif_branches {
            self.emit_if_branch("} else if", elseif_cond, elseif_body)?;
        }

        if let Some(else_stmts) = else_block {
            self.emit_line("} else {");
            self.indent();
            for s in else_stmts {
                self.emit_stmt(s)?;
            }
            self.dedent();
        }

        self.emit_line("}");
        Ok(())
    }

    fn emit_expr_to_string(&mut self, expr: &Expr) -> Result<String> {
        match &expr.kind {
            ExprKind::Literal(lit) => match lit {
                Literal::Integer(i) => Ok(i.to_string()),
                Literal::Float(f) => Ok(f.to_string()),
                Literal::Bool(b) => Ok(b.to_string()),
                Literal::String(s) => Ok(format!("\"{}\"", s)),
            },
            ExprKind::Identifier(name) => Ok(zig_ident(name)),
            ExprKind::Binary { left, op, right } => {
                let l_str = self.emit_expr_to_string(left)?;
                let r_str = self.emit_expr_to_string(right)?;
                match op {
                    BinaryOp::Add => Ok(format!("({} + {})", l_str, r_str)),
                    BinaryOp::Sub => Ok(format!("({} - {})", l_str, r_str)),
                    BinaryOp::Mul => Ok(format!("({} * {})", l_str, r_str)),
                    BinaryOp::Div => Ok(format!("__lust_div({}, {})", l_str, r_str)),
                    BinaryOp::Mod => Ok(format!("@mod({}, {})", l_str, r_str)),
                    BinaryOp::Pow => Ok(format!("__lust_pow({}, {})", l_str, r_str)),
                    BinaryOp::Eq => Ok(format!("({} == {})", l_str, r_str)),
                    BinaryOp::Ne => Ok(format!("({} != {})", l_str, r_str)),
                    BinaryOp::Lt => Ok(format!("({} < {})", l_str, r_str)),
                    BinaryOp::Le => Ok(format!("({} <= {})", l_str, r_str)),
                    BinaryOp::Gt => Ok(format!("({} > {})", l_str, r_str)),
                    BinaryOp::Ge => Ok(format!("({} >= {})", l_str, r_str)),
                    BinaryOp::And => Ok(format!("({} and {})", l_str, r_str)),
                    BinaryOp::Or => Ok(format!("({} or {})", l_str, r_str)),
                    BinaryOp::Concat => Ok(format!("try __lust_str_concat({}, {})", l_str, r_str)),
                    _ => Ok(format!("({} {} {})", l_str, op, r_str)),
                }
            }
            ExprKind::Unary { op, operand } => {
                let op_str = self.emit_expr_to_string(operand)?;
                match op {
                    UnaryOp::Neg => Ok(format!("(-{})", op_str)),
                    UnaryOp::Not => Ok(format!("(!{})", op_str)),
                }
            }
            ExprKind::FieldAccess { object, field } => {
                if let ExprKind::Identifier(mod_name) = &object.kind {
                    if mod_name == "Option" && field == "None" {
                        return Ok("null".to_string());
                    }
                    if self.known_enums.contains(mod_name) {
                        return Ok(format!("{}{{ .{} = {{}} }}", mod_name, field));
                    }
                }
                let obj_str = self.emit_expr_to_string(object)?;
                Ok(format!("{}.{}", obj_str, zig_ident(field)))
            }
            ExprKind::Index { object, index } => {
                let obj_str = self.emit_expr_to_string(object)?;
                let idx_str = self.emit_expr_to_string(index)?;
                Ok(format!("{}.get_direct({})", obj_str, idx_str))
            }
            ExprKind::MethodCall {
                receiver,
                method,
                args,
                ..
            } => {
                let rec_str = self.emit_expr_to_string(receiver)?;
                match method.as_str() {
                    "try_unwrap" if args.is_empty() => {
                        Ok(format!("(try {}.unwrap_err_to_zig())", rec_str))
                    }
                    "len" if args.is_empty() => Ok(format!("{}.len()", rec_str)),
                    "is_empty" if args.is_empty() => Ok(format!("{}.is_empty()", rec_str)),
                    "push" if args.len() == 1 => {
                        let val = self.emit_expr_to_string(&args[0])?;
                        Ok(format!("try {}.push({})", rec_str, val))
                    }
                    "pop" if args.is_empty() => Ok(format!("{}.pop()", rec_str)),
                    "clear" if args.is_empty() => Ok(format!("{}.clear()", rec_str)),
                    "first" if args.is_empty() => Ok(format!("{}.first()", rec_str)),
                    "last" if args.is_empty() => Ok(format!("{}.last()", rec_str)),
                    "contains" if args.len() == 1 => {
                        let val = self.emit_expr_to_string(&args[0])?;
                        Ok(format!("{}.contains({})", rec_str, val))
                    }
                    "reverse" if args.is_empty() => Ok(format!("{}.reverse()", rec_str)),
                    "insert" if args.len() == 2 => {
                        let idx = self.emit_expr_to_string(&args[0])?;
                        let val = self.emit_expr_to_string(&args[1])?;
                        Ok(format!("try {}.insert({}, {})", rec_str, idx, val))
                    }
                    "remove" if args.len() == 1 => {
                        let idx = self.emit_expr_to_string(&args[0])?;
                        Ok(format!("{}.remove({})", rec_str, idx))
                    }
                    "sort" if args.is_empty() => Ok(format!("{}.sort()", rec_str)),
                    "is_some" if args.is_empty() => Ok(format!("({} != null)", rec_str)),
                    "is_none" if args.is_empty() => Ok(format!("({} == null)", rec_str)),
                    "is_ok" if args.is_empty() => Ok(format!("{}.is_ok()", rec_str)),
                    "is_err" if args.is_empty() => Ok(format!("{}.is_err()", rec_str)),
                    "unwrap" if args.is_empty() => Ok(format!("__lust_unwrap({})", rec_str)),
                    "unwrap_or" if args.len() == 1 => {
                        let default_val = self.emit_expr_to_string(&args[0])?;
                        Ok(format!("({} orelse {})", rec_str, default_val))
                    }
                    _ => {
                        let mut arg_strs = Vec::new();
                        for a in args {
                            arg_strs.push(self.emit_expr_to_string(a)?);
                        }
                        Ok(format!("try {}.{}({})", rec_str, method, arg_strs.join(", ")))
                    }
                }
            }
            ExprKind::Call { callee, args, .. } => {
                if let ExprKind::Identifier(fn_name) = &callee.kind {
                    match fn_name.as_str() {
                        "println" => {
                            if let Some(arg) = args.first() {
                                let arg_str = self.emit_expr_to_string(arg)?;
                                return Ok(format!("__lust_println({})", arg_str));
                            }
                        }
                        "print" => {
                            if let Some(arg) = args.first() {
                                let arg_str = self.emit_expr_to_string(arg)?;
                                return Ok(format!("__lust_print({})", arg_str));
                            }
                        }
                        "tostring" => {
                            if let Some(arg) = args.first() {
                                let arg_str = self.emit_expr_to_string(arg)?;
                                return Ok(format!("try __lust_tostring({})", arg_str));
                            }
                        }
                        "assert" => {
                            if let Some(cond) = args.first() {
                                let cond_str = self.emit_expr_to_string(cond)?;
                                let msg_str = if let Some(m) = args.get(1) {
                                    self.emit_expr_to_string(m)?
                                } else {
                                    "\"assertion failed\"".to_string()
                                };
                                return Ok(format!("__lust_assert({}, {})", cond_str, msg_str));
                            }
                        }
                        _ => {}
                    }
                } else if let ExprKind::FieldAccess { object, field } = &callee.kind {
                    if let ExprKind::Identifier(mod_name) = &object.kind {
                        if self.known_enums.contains(mod_name) {
                            if args.is_empty() {
                                return Ok(format!("{}{{ .{} = {{}} }}", mod_name, field));
                            } else if args.len() == 1 {
                                let val_str = self.emit_expr_to_string(&args[0])?;
                                return Ok(format!("{}{{ .{} = {} }}", mod_name, field, val_str));
                            } else {
                                let mut arg_strs = Vec::new();
                                for a in args {
                                    arg_strs.push(self.emit_expr_to_string(a)?);
                                }
                                return Ok(format!(
                                    "{}{{ .{} = .{{ {} }} }}",
                                    mod_name,
                                    field,
                                    arg_strs.join(", ")
                                ));
                            }
                        }

                        match mod_name.as_str() {
                            "Option" => {
                                if field == "Some" {
                                    if let Some(first_arg) = args.first() {
                                        return self.emit_expr_to_string(first_arg);
                                    }
                                } else if field == "None" {
                                    return Ok("null".to_string());
                                }
                            }
                            "Result" => {
                                if field == "Ok" {
                                    if let Some(first_arg) = args.first() {
                                        let val_str = self.emit_expr_to_string(first_arg)?;
                                        return Ok(format!(".{{ .Ok = {} }}", val_str));
                                    }
                                } else if field == "Err" {
                                    if let Some(first_arg) = args.first() {
                                        let val_str = self.emit_expr_to_string(first_arg)?;
                                        return Ok(format!(".{{ .Err = {} }}", val_str));
                                    }
                                }
                            }
                            "array" => match field.as_str() {
                                "push" if args.len() == 2 => {
                                    let arr_str = self.emit_expr_to_string(&args[0])?;
                                    let val_str = self.emit_expr_to_string(&args[1])?;
                                    return Ok(format!("try {}.push({})", arr_str, val_str));
                                }
                                "pop" if args.len() == 1 => {
                                    let arr_str = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("{}.pop()", arr_str));
                                }
                                "len" if args.len() == 1 => {
                                    let arr_str = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("{}.len()", arr_str));
                                }
                                "clear" if args.len() == 1 => {
                                    let arr_str = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("{}.clear()", arr_str));
                                }
                                "is_empty" if args.len() == 1 => {
                                    let arr_str = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("{}.is_empty()", arr_str));
                                }
                                "first" if args.len() == 1 => {
                                    let arr_str = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("{}.first()", arr_str));
                                }
                                "last" if args.len() == 1 => {
                                    let arr_str = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("{}.last()", arr_str));
                                }
                                "contains" if args.len() == 2 => {
                                    let arr_str = self.emit_expr_to_string(&args[0])?;
                                    let val_str = self.emit_expr_to_string(&args[1])?;
                                    return Ok(format!("{}.contains({})", arr_str, val_str));
                                }
                                "reverse" if args.len() == 1 => {
                                    let arr_str = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("{}.reverse()", arr_str));
                                }
                                "insert" if args.len() == 3 => {
                                    let arr_str = self.emit_expr_to_string(&args[0])?;
                                    let idx_str = self.emit_expr_to_string(&args[1])?;
                                    let val_str = self.emit_expr_to_string(&args[2])?;
                                    return Ok(format!("try {}.insert({}, {})", arr_str, idx_str, val_str));
                                }
                                "remove" if args.len() == 2 => {
                                    let arr_str = self.emit_expr_to_string(&args[0])?;
                                    let idx_str = self.emit_expr_to_string(&args[1])?;
                                    return Ok(format!("{}.remove({})", arr_str, idx_str));
                                }
                                "sort" if args.len() == 1 => {
                                    let arr_str = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("{}.sort()", arr_str));
                                }
                                "get" if args.len() == 2 => {
                                    let arr_str = self.emit_expr_to_string(&args[0])?;
                                    let idx_str = self.emit_expr_to_string(&args[1])?;
                                    return Ok(format!("{}.get({})", arr_str, idx_str));
                                }
                                "slice" if args.len() == 3 => {
                                    let arr_str = self.emit_expr_to_string(&args[0])?;
                                    let s_str = self.emit_expr_to_string(&args[1])?;
                                    let e_str = self.emit_expr_to_string(&args[2])?;
                                    return Ok(format!("try {}.slice({}, {})", arr_str, s_str, e_str));
                                }
                                _ => {}
                            },
                            "map" => match field.as_str() {
                                "set" if args.len() == 3 => {
                                    let m_str = self.emit_expr_to_string(&args[0])?;
                                    let k_str = self.emit_expr_to_string(&args[1])?;
                                    let v_str = self.emit_expr_to_string(&args[2])?;
                                    return Ok(format!("try {}.set({}, {})", m_str, k_str, v_str));
                                }
                                "get" if args.len() == 2 => {
                                    let m_str = self.emit_expr_to_string(&args[0])?;
                                    let k_str = self.emit_expr_to_string(&args[1])?;
                                    return Ok(format!("{}.get({})", m_str, k_str));
                                }
                                "has" if args.len() == 2 => {
                                    let m_str = self.emit_expr_to_string(&args[0])?;
                                    let k_str = self.emit_expr_to_string(&args[1])?;
                                    return Ok(format!("{}.has({})", m_str, k_str));
                                }
                                "delete" if args.len() == 2 => {
                                    let m_str = self.emit_expr_to_string(&args[0])?;
                                    let k_str = self.emit_expr_to_string(&args[1])?;
                                    return Ok(format!("{}.delete({})", m_str, k_str));
                                }
                                "len" if args.len() == 1 => {
                                    let m_str = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("{}.len()", m_str));
                                }
                                "is_empty" if args.len() == 1 => {
                                    let m_str = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("{}.is_empty()", m_str));
                                }
                                "clear" if args.len() == 1 => {
                                    let m_str = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("{}.clear()", m_str));
                                }
                                _ => {}
                            },
                            "io" => match field.as_str() {
                                "read_file" if args.len() == 1 => {
                                    let path_str = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("try io.read_file({})", path_str));
                                }
                                "write_file" if args.len() == 2 => {
                                    let path_str = self.emit_expr_to_string(&args[0])?;
                                    let data_str = self.emit_expr_to_string(&args[1])?;
                                    return Ok(format!("try io.write_file({}, {})", path_str, data_str));
                                }
                                "write_stdout" if args.len() == 1 => {
                                    let data_str = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("io.write_stdout({})", data_str));
                                }
                                "read_stdin" if args.is_empty() => {
                                    return Ok("try io.read_stdin()".to_string());
                                }
                                "read_line" if args.is_empty() => {
                                    return Ok("try io.read_line()".to_string());
                                }
                                _ => {}
                            },
                            "os" => match field.as_str() {
                                "time" if args.is_empty() => {
                                    return Ok("os.time()".to_string());
                                }
                                "sleep" if args.len() == 1 => {
                                    let ms_str = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("os.sleep({})", ms_str));
                                }
                                "get_env" if args.len() == 1 => {
                                    let name_str = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("os.get_env({})", name_str));
                                }
                                "remove_file" if args.len() == 1 => {
                                    let path_str = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("os.remove_file({})", path_str));
                                }
                                "remove_dir" if args.len() == 1 => {
                                    let path_str = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("os.remove_dir({})", path_str));
                                }
                                "rename" if args.len() == 2 => {
                                    let old_str = self.emit_expr_to_string(&args[0])?;
                                    let new_str = self.emit_expr_to_string(&args[1])?;
                                    return Ok(format!("os.rename_file({}, {})", old_str, new_str));
                                }
                                _ => {}
                            },
                            "string" => match field.as_str() {
                                "len" if args.len() == 1 => {
                                    let s_str = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("__lust_string_len({})", s_str));
                                }
                                "sub" if args.len() == 2 => {
                                    let s_str = self.emit_expr_to_string(&args[0])?;
                                    let start_str = self.emit_expr_to_string(&args[1])?;
                                    return Ok(format!("__lust_string_sub({}, {}, null)", s_str, start_str));
                                }
                                "sub" if args.len() == 3 => {
                                    let s_str = self.emit_expr_to_string(&args[0])?;
                                    let start_str = self.emit_expr_to_string(&args[1])?;
                                    let end_str = self.emit_expr_to_string(&args[2])?;
                                    return Ok(format!("__lust_string_sub({}, {}, {})", s_str, start_str, end_str));
                                }
                                "lower" if args.len() == 1 => {
                                    let s_str = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("try __lust_string_lower({})", s_str));
                                }
                                "upper" if args.len() == 1 => {
                                    let s_str = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("try __lust_string_upper({})", s_str));
                                }
                                "is_empty" if args.len() == 1 => {
                                    let s_str = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("__lust_string_is_empty({})", s_str));
                                }
                                "starts_with" if args.len() == 2 => {
                                    let s_str = self.emit_expr_to_string(&args[0])?;
                                    let p_str = self.emit_expr_to_string(&args[1])?;
                                    return Ok(format!("__lust_string_starts_with({}, {})", s_str, p_str));
                                }
                                "ends_with" if args.len() == 2 => {
                                    let s_str = self.emit_expr_to_string(&args[0])?;
                                    let p_str = self.emit_expr_to_string(&args[1])?;
                                    return Ok(format!("__lust_string_ends_with({}, {})", s_str, p_str));
                                }
                                "contains" if args.len() == 2 => {
                                    let s_str = self.emit_expr_to_string(&args[0])?;
                                    let n_str = self.emit_expr_to_string(&args[1])?;
                                    return Ok(format!("__lust_string_contains({}, {})", s_str, n_str));
                                }
                                "trim" if args.len() == 1 => {
                                    let s_str = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("__lust_string_trim({})", s_str));
                                }
                                "trim_start" if args.len() == 1 => {
                                    let s_str = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("__lust_string_trim_start({})", s_str));
                                }
                                "trim_end" if args.len() == 1 => {
                                    let s_str = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("__lust_string_trim_end({})", s_str));
                                }
                                "replace" if args.len() == 3 => {
                                    let s_str = self.emit_expr_to_string(&args[0])?;
                                    let n_str = self.emit_expr_to_string(&args[1])?;
                                    let r_str = self.emit_expr_to_string(&args[2])?;
                                    return Ok(format!("try __lust_string_replace({}, {}, {})", s_str, n_str, r_str));
                                }
                                "split" if args.len() == 2 => {
                                    let s_str = self.emit_expr_to_string(&args[0])?;
                                    let d_str = self.emit_expr_to_string(&args[1])?;
                                    return Ok(format!("try __lust_string_split({}, {})", s_str, d_str));
                                }
                                _ => {}
                            },
                            "math" => match field.as_str() {
                                "abs" if args.len() == 1 => {
                                    let x = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("__lust_math_abs({})", x));
                                }
                                "floor" if args.len() == 1 => {
                                    let x = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("__lust_math_floor({})", x));
                                }
                                "ceil" if args.len() == 1 => {
                                    let x = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("__lust_math_ceil({})", x));
                                }
                                "round" if args.len() == 1 => {
                                    let x = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("__lust_math_round({})", x));
                                }
                                "sqrt" if args.len() == 1 => {
                                    let x = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("__lust_math_sqrt({})", x));
                                }
                                "sin" if args.len() == 1 => {
                                    let x = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("__lust_math_sin({})", x));
                                }
                                "cos" if args.len() == 1 => {
                                    let x = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("__lust_math_cos({})", x));
                                }
                                "tan" if args.len() == 1 => {
                                    let x = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("__lust_math_tan({})", x));
                                }
                                "asin" if args.len() == 1 => {
                                    let x = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("__lust_math_asin({})", x));
                                }
                                "acos" if args.len() == 1 => {
                                    let x = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("__lust_math_acos({})", x));
                                }
                                "atan" if args.len() == 1 => {
                                    let x = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("__lust_math_atan({})", x));
                                }
                                "atan2" if args.len() == 2 => {
                                    let y = self.emit_expr_to_string(&args[0])?;
                                    let x = self.emit_expr_to_string(&args[1])?;
                                    return Ok(format!("__lust_math_atan2({}, {})", y, x));
                                }
                                "clamp" if args.len() == 3 => {
                                    let v = self.emit_expr_to_string(&args[0])?;
                                    let l = self.emit_expr_to_string(&args[1])?;
                                    let u = self.emit_expr_to_string(&args[2])?;
                                    return Ok(format!("__lust_math_clamp({}, {}, {})", v, l, u));
                                }
                                "min" if args.len() == 2 => {
                                    let a = self.emit_expr_to_string(&args[0])?;
                                    let b = self.emit_expr_to_string(&args[1])?;
                                    return Ok(format!("__lust_math_min({}, {})", a, b));
                                }
                                "max" if args.len() == 2 => {
                                    let a = self.emit_expr_to_string(&args[0])?;
                                    let b = self.emit_expr_to_string(&args[1])?;
                                    return Ok(format!("__lust_math_max({}, {})", a, b));
                                }
                                "deg" if args.len() == 1 => {
                                    let x = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("__lust_math_deg({})", x));
                                }
                                "rad" if args.len() == 1 => {
                                    let x = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("__lust_math_rad({})", x));
                                }
                                "exp" if args.len() == 1 => {
                                    let x = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("__lust_math_exp({})", x));
                                }
                                "log" if args.len() == 1 => {
                                    let x = self.emit_expr_to_string(&args[0])?;
                                    return Ok(format!("__lust_math_log({})", x));
                                }
                                _ => {}
                            },
                            _ => {}
                        }
                    }
                }

                let callee_str = self.emit_expr_to_string(callee)?;
                let mut arg_strs = Vec::new();
                for a in args {
                    arg_strs.push(self.emit_expr_to_string(a)?);
                }

                let is_non_failing = match &callee.kind {
                    ExprKind::FieldAccess { field, .. } => matches!(
                        field.as_str(),
                        "sort" | "remove" | "pop" | "clear" | "reverse" | "len" | "is_empty"
                            | "first" | "last" | "contains" | "get" | "has" | "delete"
                            | "time" | "sleep" | "get_env" | "remove_file" | "remove_dir" | "rename"
                            | "write_stdout"
                    ),
                    ExprKind::Identifier(fn_name) => {
                        matches!(fn_name.as_str(), "print" | "println" | "assert" | "type")
                            || fn_name.starts_with("__lust_math_")
                    }
                    _ => false,
                };

                if is_non_failing {
                    Ok(format!("{}({})", callee_str, arg_strs.join(", ")))
                } else {
                    Ok(format!("try {}({})", callee_str, arg_strs.join(", ")))
                }
            }
            ExprKind::StructLiteral { name, fields } => {
                let tmp = self.next_temp_var();
                let mut field_assigns = Vec::new();
                for f in fields {
                    let v_str = self.emit_expr_to_string(&f.value)?;
                    field_assigns.push(format!(".{} = {}", f.name, v_str));
                }

                self.emit_line(&format!(
                    "const {} = try __lust_allocator.create({});",
                    tmp, name
                ));
                self.emit_line(&format!(
                    "{}.* = .{{ {} }};",
                    tmp,
                    field_assigns.join(", ")
                ));
                Ok(tmp)
            }
            ExprKind::EnumConstructor {
                enum_name,
                variant,
                args,
            } => {
                if enum_name == "Option" {
                    if variant == "None" {
                        Ok("null".to_string())
                    } else if variant == "Some" {
                        if let Some(first_arg) = args.first() {
                            self.emit_expr_to_string(first_arg)
                        } else {
                            Ok("null".to_string())
                        }
                    } else {
                        Ok("null".to_string())
                    }
                } else if enum_name == "Result" {
                    if variant == "Ok" {
                        if let Some(first_arg) = args.first() {
                            let val_str = self.emit_expr_to_string(first_arg)?;
                            Ok(format!(".{{ .Ok = {} }}", val_str))
                        } else {
                            Ok(".{ .Ok = {} }".to_string())
                        }
                    } else if variant == "Err" {
                        if let Some(first_arg) = args.first() {
                            let val_str = self.emit_expr_to_string(first_arg)?;
                            Ok(format!(".{{ .Err = {} }}", val_str))
                        } else {
                            Ok(".{ .Err = {} }".to_string())
                        }
                    } else {
                        Ok(".{}".to_string())
                    }
                } else if args.is_empty() {
                    Ok(format!("{}{{ .{} = {{}} }}", enum_name, variant))
                } else if args.len() == 1 {
                    let a_str = self.emit_expr_to_string(&args[0])?;
                    Ok(format!("{}{{ .{} = {} }}", enum_name, variant, a_str))
                } else {
                    let mut arg_strs = Vec::new();
                    for a in args {
                        arg_strs.push(self.emit_expr_to_string(a)?);
                    }
                    Ok(format!(
                        "{}{{ .{} = .{{ {} }} }}",
                        enum_name,
                        variant,
                        arg_strs.join(", ")
                    ))
                }
            }
            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let cond_str = self.emit_expr_to_string(condition)?;
                let then_str = self.emit_expr_to_string(then_branch)?;
                let is_int_literal = matches!(&then_branch.kind, ExprKind::Literal(Literal::Integer(_)));
                let is_float_literal = matches!(&then_branch.kind, ExprKind::Literal(Literal::Float(_)));
                if let Some(else_expr) = else_branch {
                    let else_str = self.emit_expr_to_string(else_expr)?;
                    if is_int_literal {
                        Ok(format!("(if ({}) @as(i64, {}) else @as(i64, {}))", cond_str, then_str, else_str))
                    } else if is_float_literal {
                        Ok(format!("(if ({}) @as(f64, {}) else @as(f64, {}))", cond_str, then_str, else_str))
                    } else {
                        Ok(format!("(if ({}) {} else {})", cond_str, then_str, else_str))
                    }
                } else {
                    Ok(format!("(if ({}) {} else null)", cond_str, then_str))
                }
            }
            ExprKind::Cast { expr, target_type } => {
                let expr_str = self.emit_expr_to_string(expr)?;
                let target_ty_str = self.map_type(target_type, false)?;
                match target_type.kind {
                    TypeKind::Int => Ok(format!("@as(i64, @intCast({}))", expr_str)),
                    TypeKind::Float => Ok(format!("@as(f64, @floatFromInt({}))", expr_str)),
                    _ => Ok(format!("@as({}, {})", target_ty_str, expr_str)),
                }
            }
            ExprKind::Tuple(elements) => {
                let mut elem_strs = Vec::new();
                for el in elements {
                    elem_strs.push(self.emit_expr_to_string(el)?);
                }
                Ok(format!(".{{ {} }}", elem_strs.join(", ")))
            }
            ExprKind::Range { start, end, inclusive } => {
                let s = self.emit_expr_to_string(start)?;
                let e = self.emit_expr_to_string(end)?;
                Ok(format!(".{{ .start = {}, .end = {}, .inclusive = {} }}", s, e, inclusive))
            }
            ExprKind::Array(elements) => {
                let mut elem_strs = Vec::new();
                for el in elements {
                    elem_strs.push(self.emit_expr_to_string(el)?);
                }
                if elem_strs.is_empty() {
                    Ok("LustArray(i64).init(__lust_allocator)".to_string())
                } else {
                    let first = &elem_strs[0];
                    Ok(format!(
                        "(try LustArray(@TypeOf({})).fromSlice(__lust_allocator, &[_]@TypeOf({}){{ {} }}))",
                        first, first, elem_strs.join(", ")
                    ))
                }
            }
            ExprKind::Map(entries) => {
                if entries.is_empty() {
                    Ok("LustMap([]const u8, i64).init(__lust_allocator)".to_string())
                } else {
                    let (first_k, first_v) = &entries[0];
                    let k_type = match &first_k.kind {
                        ExprKind::Literal(Literal::String(_)) => "[]const u8",
                        ExprKind::Literal(Literal::Float(_)) => "f64",
                        ExprKind::Literal(Literal::Bool(_)) => "bool",
                        _ => "i64",
                    };
                    let v_type = match &first_v.kind {
                        ExprKind::Literal(Literal::String(_)) => "[]const u8",
                        ExprKind::Literal(Literal::Float(_)) => "f64",
                        ExprKind::Literal(Literal::Bool(_)) => "bool",
                        _ => "i64",
                    };
                    let mut lines = Vec::new();
                    lines.push(format!("var __m = LustMap({}, {}).init(__lust_allocator);", k_type, v_type));
                    for (k, v) in entries {
                        let ks = self.emit_expr_to_string(k)?;
                        let vs = self.emit_expr_to_string(v)?;
                        lines.push(format!("try __m.set({}, {});", ks, vs));
                    }
                    lines.push("break :blk __m;".to_string());
                    Ok(format!("(blk: {{\n{}\n}})", lines.join("\n")))
                }
            }
            ExprKind::Block(stmts) => {
                let mut block_gen = ZigCodegen::with_options(self.runtime_opts.clone());
                block_gen.indent();
                for s in stmts {
                    block_gen.emit_stmt(s)?;
                }
                Ok(format!("(blk: {{\n{}\n    break :blk {{}};\n}})", block_gen.out))
            }
            ExprKind::Lambda { params, body, .. } => {
                let mut param_decls = Vec::new();
                for (name, ty_opt) in params {
                    let ty_str = if let Some(t) = ty_opt {
                        self.map_type(t, false)?
                    } else {
                        "anytype".to_string()
                    };
                    param_decls.push(format!("__lust_param_{}: {}", name, ty_str));
                }
                let body_str = self.emit_expr_to_string(body)?;
                Ok(format!(
                    "(struct {{ pub fn call({}) @TypeOf({}) {{ return {}; }} }}.call)",
                    param_decls.join(", "),
                    body_str,
                    body_str
                ))
            }
            ExprKind::TypeCheck { expr, check_type } => {
                let expr_str = self.emit_expr_to_string(expr)?;
                match &check_type.kind {
                    TypeKind::Named(name) => Ok(format!("({} == .{})", expr_str, name)),
                    _ => Ok("true".to_string()),
                }
            }
            ExprKind::IsPattern { expr, pattern } => {
                let expr_str = self.emit_expr_to_string(expr)?;
                match pattern {
                    Pattern::Identifier(variant) => {
                        Ok(format!("({} == .{})", expr_str, variant))
                    }
                    Pattern::Enum { variant, bindings, .. } => {
                        if variant == "Some" {
                            Ok(format!("({} != null)", expr_str))
                        } else if variant == "None" {
                            Ok(format!("({} == null)", expr_str))
                        } else if bindings.is_empty() {
                            Ok(format!("({} == .{})", expr_str, variant))
                        } else {
                            Ok(format!("({} == .{})", expr_str, variant))
                        }
                    }
                    _ => Ok("true".to_string()),
                }
            }
            ExprKind::Paren(inner) => {
                let inner_str = self.emit_expr_to_string(inner)?;
                Ok(format!("({})", inner_str))
            }
            _ => Ok("undefined".to_string()),
        }
    }
}

/// AST reachability analysis for tree shaking (prunes unused items).
pub fn tree_shake_items(items: &[Item], is_library: bool) -> Vec<Item> {
    let mut reachable = HashSet::new();

    // 1. Mark roots
    for item in items {
        match &item.kind {
            ItemKind::Script(_) => {
                reachable.insert("__script__".to_string());
            }
            ItemKind::Function(f) => {
                let clean = f.name.rsplit('.').next().unwrap_or(&f.name);
                if is_library || clean == "main" {
                    reachable.insert(f.name.clone());
                    reachable.insert(clean.to_string());
                }
            }
            ItemKind::Struct(s) => {
                if is_library {
                    reachable.insert(s.name.clone());
                }
            }
            ItemKind::Enum(e) => {
                if is_library {
                    reachable.insert(e.name.clone());
                }
            }
            _ => {}
        }
    }

    // 2. Discover reachability
    let mut changed = true;
    while changed {
        let before_count = reachable.len();
        for item in items {
            match &item.kind {
                ItemKind::Function(f) => {
                    let clean = f.name.rsplit('.').next().unwrap_or(&f.name);
                    if reachable.contains(&f.name) || reachable.contains(clean) {
                        for p in &f.params {
                            collect_type_names(&p.ty, &mut reachable);
                        }
                        if let Some(ret) = &f.return_type {
                            collect_type_names(ret, &mut reachable);
                        }
                        for stmt in &f.body {
                            collect_stmt_reachability(stmt, &mut reachable);
                        }
                    }
                }
                ItemKind::Struct(s) => {
                    if reachable.contains(&s.name) {
                        for field in &s.fields {
                            collect_type_names(&field.ty, &mut reachable);
                        }
                    }
                }
                ItemKind::Enum(e) => {
                    if reachable.contains(&e.name) {
                        for v in &e.variants {
                            if let Some(fields) = &v.fields {
                                for f in fields {
                                    collect_type_names(f, &mut reachable);
                                }
                            }
                        }
                    }
                }
                ItemKind::Impl(imp) => {
                    if let TypeKind::Named(t_name) = &imp.target_type.kind {
                        if reachable.contains(t_name) {
                            for m in &imp.methods {
                                for p in &m.params {
                                    collect_type_names(&p.ty, &mut reachable);
                                }
                                if let Some(ret) = &m.return_type {
                                    collect_type_names(ret, &mut reachable);
                                }
                                for stmt in &m.body {
                                    collect_stmt_reachability(stmt, &mut reachable);
                                }
                            }
                        }
                    }
                }
                ItemKind::Script(stmts) => {
                    for stmt in stmts {
                        collect_stmt_reachability(stmt, &mut reachable);
                    }
                }
                _ => {}
            }
        }
        changed = reachable.len() > before_count;
    }

    // 3. Filter retained items
    items
        .iter()
        .filter(|item| match &item.kind {
            ItemKind::Script(_) => true,
            ItemKind::Function(f) => {
                let clean = f.name.rsplit('.').next().unwrap_or(&f.name);
                reachable.contains(&f.name) || reachable.contains(clean)
            }
            ItemKind::Struct(s) => reachable.contains(&s.name),
            ItemKind::Enum(e) => reachable.contains(&e.name),
            ItemKind::Impl(imp) => {
                if let TypeKind::Named(t_name) = &imp.target_type.kind {
                    reachable.contains(t_name)
                } else {
                    true
                }
            }
            _ => true,
        })
        .cloned()
        .collect()
}

fn collect_type_names(ty: &Type, reachable: &mut HashSet<String>) {
    match &ty.kind {
        TypeKind::Named(name) => {
            reachable.insert(name.clone());
        }
        TypeKind::Option(inner)
        | TypeKind::Array(inner)
        | TypeKind::Ref(inner)
        | TypeKind::MutRef(inner) => {
            collect_type_names(inner, reachable);
        }
        TypeKind::Map(k, v) | TypeKind::Result(k, v) => {
            collect_type_names(k, reachable);
            collect_type_names(v, reachable);
        }
        TypeKind::Tuple(types) => {
            for t in types {
                collect_type_names(t, reachable);
            }
        }
        _ => {}
    }
}

fn collect_stmt_reachability(stmt: &Stmt, reachable: &mut HashSet<String>) {
    match &stmt.kind {
        StmtKind::Local { initializer, .. } => {
            if let Some(inits) = initializer {
                for e in inits {
                    collect_expr_reachability(e, reachable);
                }
            }
        }
        StmtKind::Assign { targets, values } => {
            for t in targets {
                collect_expr_reachability(t, reachable);
            }
            for v in values {
                collect_expr_reachability(v, reachable);
            }
        }
        StmtKind::CompoundAssign { target, value, .. } => {
            collect_expr_reachability(target, reachable);
            collect_expr_reachability(value, reachable);
        }
        StmtKind::While { condition, body } => {
            collect_expr_reachability(condition, reachable);
            for s in body {
                collect_stmt_reachability(s, reachable);
            }
        }
        StmtKind::ForNumeric { start, end, step, body, .. } => {
            collect_expr_reachability(start, reachable);
            collect_expr_reachability(end, reachable);
            if let Some(st) = step {
                collect_expr_reachability(st, reachable);
            }
            for s in body {
                collect_stmt_reachability(s, reachable);
            }
        }
        StmtKind::ForIn { iterator, body, .. } => {
            collect_expr_reachability(iterator, reachable);
            for s in body {
                collect_stmt_reachability(s, reachable);
            }
        }
        StmtKind::If { condition, then_block, elseif_branches, else_block } => {
            collect_expr_reachability(condition, reachable);
            for s in then_block {
                collect_stmt_reachability(s, reachable);
            }
            for (c, b) in elseif_branches {
                collect_expr_reachability(c, reachable);
                for s in b {
                    collect_stmt_reachability(s, reachable);
                }
            }
            if let Some(eb) = else_block {
                for s in eb {
                    collect_stmt_reachability(s, reachable);
                }
            }
        }
        StmtKind::Return(exprs) => {
            for e in exprs {
                collect_expr_reachability(e, reachable);
            }
        }
        StmtKind::Expr(e) => {
            collect_expr_reachability(e, reachable);
        }
        StmtKind::Block(stmts) => {
            for s in stmts {
                collect_stmt_reachability(s, reachable);
            }
        }
        _ => {}
    }
}

fn collect_expr_reachability(expr: &Expr, reachable: &mut HashSet<String>) {
    match &expr.kind {
        ExprKind::Identifier(name) => {
            reachable.insert(name.clone());
        }
        ExprKind::Call { callee, args, .. } => {
            collect_expr_reachability(callee, reachable);
            for a in args {
                collect_expr_reachability(a, reachable);
            }
        }
        ExprKind::MethodCall { receiver, args, .. } => {
            collect_expr_reachability(receiver, reachable);
            for a in args {
                collect_expr_reachability(a, reachable);
            }
        }
        ExprKind::Binary { left, right, .. } => {
            collect_expr_reachability(left, reachable);
            collect_expr_reachability(right, reachable);
        }
        ExprKind::Unary { operand, .. } => {
            collect_expr_reachability(operand, reachable);
        }
        ExprKind::FieldAccess { object, .. } => {
            collect_expr_reachability(object, reachable);
        }
        ExprKind::Index { object, index } => {
            collect_expr_reachability(object, reachable);
            collect_expr_reachability(index, reachable);
        }
        ExprKind::StructLiteral { name, fields } => {
            reachable.insert(name.clone());
            for f in fields {
                collect_expr_reachability(&f.value, reachable);
            }
        }
        ExprKind::EnumConstructor { enum_name, args, .. } => {
            reachable.insert(enum_name.clone());
            for a in args {
                collect_expr_reachability(a, reachable);
            }
        }
        ExprKind::Array(elements) | ExprKind::Tuple(elements) => {
            for e in elements {
                collect_expr_reachability(e, reachable);
            }
        }
        ExprKind::Map(entries) => {
            for (k, v) in entries {
                collect_expr_reachability(k, reachable);
                collect_expr_reachability(v, reachable);
            }
        }
        ExprKind::If { condition, then_branch, else_branch } => {
            collect_expr_reachability(condition, reachable);
            collect_expr_reachability(then_branch, reachable);
            if let Some(eb) = else_branch {
                collect_expr_reachability(eb, reachable);
            }
        }
        ExprKind::Cast { expr, target_type } => {
            collect_expr_reachability(expr, reachable);
            collect_type_names(target_type, reachable);
        }
        ExprKind::Paren(inner) => {
            collect_expr_reachability(inner, reachable);
        }
        _ => {}
    }
}

/// Analyze what runtime components are needed across retained items.
pub fn analyze_runtime_requirements(
    items: &[Item],
    is_library: bool,
    max_memory_bytes: Option<usize>,
) -> RuntimeOptions {
    let mut opts = RuntimeOptions {
        needs_array: false,
        needs_map: false,
        needs_string: false,
        needs_math: false,
        needs_pow: false,
        needs_io: false,
        needs_os: false,
        max_memory_bytes,
        is_library,
    };

    for item in items {
        match &item.kind {
            ItemKind::Function(f) => {
                for p in &f.params {
                    scan_type(&p.ty, &mut opts);
                }
                if let Some(ret) = &f.return_type {
                    scan_type(ret, &mut opts);
                }
                for stmt in &f.body {
                    scan_stmt(stmt, &mut opts);
                }
            }
            ItemKind::Struct(s) => {
                for field in &s.fields {
                    scan_type(&field.ty, &mut opts);
                }
            }
            ItemKind::Script(stmts) => {
                for stmt in stmts {
                    scan_stmt(stmt, &mut opts);
                }
            }
            ItemKind::Impl(imp) => {
                for m in &imp.methods {
                    for stmt in &m.body {
                        scan_stmt(stmt, &mut opts);
                    }
                }
            }
            _ => {}
        }
    }

    if opts.needs_string {
        opts.needs_array = true;
    }

    opts
}

fn scan_type(ty: &Type, opts: &mut RuntimeOptions) {
    match &ty.kind {
        TypeKind::Array(inner) => {
            opts.needs_array = true;
            scan_type(inner, opts);
        }
        TypeKind::Map(k, v) => {
            opts.needs_map = true;
            scan_type(k, opts);
            scan_type(v, opts);
        }
        TypeKind::String => {
            opts.needs_string = true;
        }
        _ => {}
    }
}

fn scan_stmt(stmt: &Stmt, opts: &mut RuntimeOptions) {
    match &stmt.kind {
        StmtKind::Local { initializer, .. } => {
            if let Some(inits) = initializer {
                for e in inits {
                    scan_expr(e, opts);
                }
            }
        }
        StmtKind::Assign { targets, values } => {
            for t in targets {
                scan_expr(t, opts);
            }
            for v in values {
                scan_expr(v, opts);
            }
        }
        StmtKind::CompoundAssign { target, op, value } => {
            scan_expr(target, opts);
            scan_expr(value, opts);
            match op {
                BinaryOp::Pow => opts.needs_pow = true,
                BinaryOp::Div | BinaryOp::Mod => opts.needs_math = true,
                BinaryOp::Concat => opts.needs_string = true,
                _ => {}
            }
        }
        StmtKind::While { condition, body } => {
            scan_expr(condition, opts);
            for s in body {
                scan_stmt(s, opts);
            }
        }
        StmtKind::ForNumeric { start, end, step, body, .. } => {
            scan_expr(start, opts);
            scan_expr(end, opts);
            if let Some(st) = step {
                scan_expr(st, opts);
            }
            for s in body {
                scan_stmt(s, opts);
            }
        }
        StmtKind::ForIn { iterator, body, .. } => {
            opts.needs_array = true;
            scan_expr(iterator, opts);
            for s in body {
                scan_stmt(s, opts);
            }
        }
        StmtKind::If { condition, then_block, elseif_branches, else_block } => {
            scan_expr(condition, opts);
            for s in then_block {
                scan_stmt(s, opts);
            }
            for (c, b) in elseif_branches {
                scan_expr(c, opts);
                for s in b {
                    scan_stmt(s, opts);
                }
            }
            if let Some(eb) = else_block {
                for s in eb {
                    scan_stmt(s, opts);
                }
            }
        }
        StmtKind::Return(exprs) => {
            for e in exprs {
                scan_expr(e, opts);
            }
        }
        StmtKind::Expr(e) => {
            scan_expr(e, opts);
        }
        StmtKind::Block(stmts) => {
            for s in stmts {
                scan_stmt(s, opts);
            }
        }
        _ => {}
    }
}

fn scan_expr(expr: &Expr, opts: &mut RuntimeOptions) {
    match &expr.kind {
        ExprKind::Binary { left, op, right } => {
            scan_expr(left, opts);
            scan_expr(right, opts);
            match op {
                BinaryOp::Pow => opts.needs_pow = true,
                BinaryOp::Div | BinaryOp::Mod => opts.needs_math = true,
                BinaryOp::Concat => opts.needs_string = true,
                _ => {}
            }
        }
        ExprKind::Array(elements) => {
            opts.needs_array = true;
            for e in elements {
                scan_expr(e, opts);
            }
        }
        ExprKind::Map(entries) => {
            opts.needs_map = true;
            for (k, v) in entries {
                scan_expr(k, opts);
                scan_expr(v, opts);
            }
        }
        ExprKind::Index { object, index } => {
            opts.needs_array = true;
            scan_expr(object, opts);
            scan_expr(index, opts);
        }
        ExprKind::FieldAccess { object, field } => {
            scan_expr(object, opts);
            if let ExprKind::Identifier(mod_name) = &object.kind {
                match mod_name.as_str() {
                    "io" => opts.needs_io = true,
                    "os" => opts.needs_os = true,
                    "math" => opts.needs_math = true,
                    "string" => opts.needs_string = true,
                    "array" => opts.needs_array = true,
                    "map" => opts.needs_map = true,
                    _ => {}
                }
            }
            if field == "len" || field == "push" || field == "pop" {
                opts.needs_array = true;
            }
        }
        ExprKind::Call { callee, args, .. } => {
            scan_expr(callee, opts);
            for a in args {
                scan_expr(a, opts);
            }
            if let ExprKind::Identifier(name) = &callee.kind {
                match name.as_str() {
                    "tostring" => opts.needs_string = true,
                    "print" | "println" => opts.needs_string = true,
                    _ => {}
                }
            }
        }
        ExprKind::MethodCall { receiver, method, args, .. } => {
            scan_expr(receiver, opts);
            for a in args {
                scan_expr(a, opts);
            }
            match method.as_str() {
                "push" | "pop" | "insert" | "remove" | "sort" | "slice" | "contains" | "reverse" => {
                    opts.needs_array = true;
                }
                "set" | "get" | "has" | "delete" => {
                    opts.needs_map = true;
                }
                _ => {}
            }
        }
        ExprKind::If { condition, then_branch, else_branch } => {
            scan_expr(condition, opts);
            scan_expr(then_branch, opts);
            if let Some(eb) = else_branch {
                scan_expr(eb, opts);
            }
        }
        ExprKind::Paren(inner) | ExprKind::Unary { operand: inner, .. } => {
            scan_expr(inner, opts);
        }
        _ => {}
    }
}

/// Generates a C header file (.h) declaring exported structs and C-ABI functions.
pub fn generate_c_header(items: &[Item]) -> String {
    let mut h = String::new();
    h.push_str("/* Generated by Lust AOT Compiler */\n");
    h.push_str("#ifndef LUST_GENERATED_H\n#define LUST_GENERATED_H\n\n");
    h.push_str("#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n");
    h.push_str("#include <stdint.h>\n#include <stdbool.h>\n#include <stddef.h>\n\n");

    // Forward declare structs
    for item in items {
        if let ItemKind::Struct(s) = &item.kind {
            h.push_str(&format!("typedef struct {0} {0};\n", s.name));
        }
    }
    h.push('\n');

    // Struct definitions
    for item in items {
        if let ItemKind::Struct(s) = &item.kind {
            h.push_str(&format!("struct {} {{\n", s.name));
            for f in &s.fields {
                let c_ty = c_type_from_lust(&f.ty);
                h.push_str(&format!("    {} {};\n", c_ty, f.name));
            }
            h.push_str("};\n\n");
        }
    }

    // Function declarations
    for item in items {
        if let ItemKind::Function(f) = &item.kind {
            let clean_name = f.name.rsplit('.').next().unwrap_or(&f.name);
            let ret_c = if let Some(ret) = &f.return_type {
                c_type_from_lust(ret)
            } else {
                "void".to_string()
            };
            let mut params_c = Vec::new();
            for p in &f.params {
                let p_c = c_type_from_lust(&p.ty);
                params_c.push(format!("{} {}", p_c, p.name));
            }
            let params_str = if params_c.is_empty() {
                "void".to_string()
            } else {
                params_c.join(", ")
            };
            h.push_str(&format!("{} {}({});\n", ret_c, clean_name, params_str));
        }
    }

    h.push_str("\n#ifdef __cplusplus\n}\n#endif\n\n#endif /* LUST_GENERATED_H */\n");
    h
}

fn c_type_from_lust(ty: &Type) -> String {
    match &ty.kind {
        TypeKind::Int => "int64_t".to_string(),
        TypeKind::Float => "double".to_string(),
        TypeKind::Bool => "bool".to_string(),
        TypeKind::String => "const char*".to_string(),
        TypeKind::Unit => "void".to_string(),
        TypeKind::Named(name) => format!("const {}*", name),
        TypeKind::Pointer { mutable, pointee } => {
            if *mutable {
                format!("{}*", c_type_from_lust(pointee))
            } else {
                format!("const {}*", c_type_from_lust(pointee))
            }
        }
        _ => "void*".to_string(),
    }
}

/// Compiles a Zig source file to a native executable or library with full configuration.
pub fn compile_zig(
    zig_path: &Path,
    output_bin: &Path,
    opts: &CompileOptions,
    link_libc: bool,
) -> Result<PathBuf> {
    let mut cmd = Command::new("zig");
    match opts.target_kind {
        BuildTargetKind::Executable => {
            cmd.arg("build-exe");
        }
        BuildTargetKind::SharedLibrary => {
            cmd.arg("build-lib").arg("-dynamic");
        }
        BuildTargetKind::StaticLibrary => {
            cmd.arg("build-lib").arg("-static");
        }
    }
    cmd.arg(format!("-O{}", opts.release_mode));
    cmd.arg(zig_path);
    cmd.arg(format!("-femit-bin={}", output_bin.display()));

    if opts.strip {
        cmd.arg("-fstrip");
    }
    if let Some(target) = &opts.target_triple {
        cmd.arg("-target").arg(target);
    }
    if let Some(cache) = &opts.cache_dir {
        cmd.arg("--cache-dir").arg(cache);
    }
    if link_libc {
        cmd.arg("-lc");
    }

    let output = cmd
        .output()
        .map_err(|e| LustError::Unknown(format!("Failed to invoke zig compiler: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(LustError::Unknown(format!(
            "Zig compilation failed:\n{}",
            stderr
        )));
    }

    Ok(output_bin.to_path_buf())
}

/// Backwards compatibility helper for compiling to executable binary.
pub fn compile_zig_to_binary(
    zig_path: &Path,
    output_bin: &Path,
    release_mode: &str,
) -> Result<PathBuf> {
    let opts = CompileOptions {
        target_kind: BuildTargetKind::Executable,
        release_mode: release_mode.to_string(),
        emit_zig: false,
        emit_header: None,
        max_memory: None,
        max_memory_bytes: None,
        tree_shake: false,
        strip: false,
        target_triple: None,
        cache_dir: None,
    };
    compile_zig(zig_path, output_bin, &opts, false)
}

fn zig_ident(name: &str) -> String {
    match name {
        "error" | "type" | "fn" | "pub" | "struct" | "union" | "enum" | "switch"
        | "test" | "comptime" | "inline" | "opaque" | "try" | "catch" | "defer"
        | "errdefer" | "unreachable" => format!("@\"{}\"", name),
        _ => name.to_string(),
    }
}
