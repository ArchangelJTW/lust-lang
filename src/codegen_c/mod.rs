//! C11 AOT code generator for Lust AST programs.
//!
//! Emits standalone, zero-UB C11 and compiles it with the system C compiler
//! (`$CC`, `cc`, `clang`, or `gcc`) to a native executable or library.

use crate::ast::{
    BinaryOp, Expr, ExprKind, FunctionDef, ImplBlock, Item, ItemKind, Literal, Pattern, Stmt,
    StmtKind, StructDef, Type, TypeKind, UnaryOp,
};
use crate::{LustError, Result};
use alloc::{format, string::String, vec::Vec};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone)]
pub enum CType {
    Int,
    Float,
    Bool,
    Str,
    Unit,
    Struct(String),
    Array(Box<CType>),
    OptionPtr(String), // Option<StructName> represented as nullable arena pointer
    OptionVal(Box<CType>),
}

impl CType {
    fn from_ast(ty: &Type, structs: &HashSet<String>) -> Self {
        match &ty.kind {
            TypeKind::Int => CType::Int,
            TypeKind::Float => CType::Float,
            TypeKind::Bool => CType::Bool,
            TypeKind::String => CType::Str,
            TypeKind::Unit => CType::Unit,
            TypeKind::Named(n) => {
                let short = short_name(n);
                if structs.contains(short) {
                    CType::Struct(short.to_string())
                } else {
                    CType::Int
                }
            }
            TypeKind::Array(inner) => CType::Array(Box::new(Self::from_ast(inner, structs))),
            TypeKind::Option(inner) => match Self::from_ast(inner, structs) {
                CType::Struct(name) => CType::OptionPtr(name),
                other => CType::OptionVal(Box::new(other)),
            },
            TypeKind::GenericInstance { name, type_args } => {
                let short = short_name(name);
                if short == "Array" && type_args.len() == 1 {
                    CType::Array(Box::new(Self::from_ast(&type_args[0], structs)))
                } else if short == "Option" && type_args.len() == 1 {
                    match Self::from_ast(&type_args[0], structs) {
                        CType::Struct(sname) => CType::OptionPtr(sname),
                        other => CType::OptionVal(Box::new(other)),
                    }
                } else {
                    CType::Int
                }
            }
            _ => CType::Int,
        }
    }

    fn c_decl(&self) -> String {
        match self {
            CType::Int => "int64_t".to_string(),
            CType::Float => "double".to_string(),
            CType::Bool => "bool".to_string(),
            CType::Str => "LustStr".to_string(),
            CType::Unit => "void".to_string(),
            CType::Struct(n) => sanitize_ident(n),
            CType::Array(elem) => format!("LustArr_{}*", elem.array_suffix()),
            CType::OptionPtr(n) => format!("{}*", sanitize_ident(n)),
            CType::OptionVal(inner) => format!("LustOpt_{}", inner.array_suffix()),
        }
    }

    fn array_suffix(&self) -> String {
        match self {
            CType::Int => "i64".to_string(),
            CType::Float => "f64".to_string(),
            CType::Bool => "bool".to_string(),
            CType::Str => "str".to_string(),
            CType::Struct(n) => sanitize_ident(n),
            _ => "i64".to_string(),
        }
    }
}

fn short_name(name: &str) -> &str {
    name.rsplit('.').next().unwrap_or(name)
}

fn sanitize_ident(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push_str("__");
        }
    }
    if out.is_empty() || out.as_bytes()[0].is_ascii_digit() {
        format!("__{out}")
    } else if matches!(
        out.as_str(),
        "auto" | "break" | "case" | "char" | "const" | "continue" | "default" | "do" | "double"
            | "else" | "enum" | "extern" | "float" | "for" | "goto" | "if" | "inline" | "int"
            | "long" | "register" | "restrict" | "return" | "short" | "signed" | "sizeof"
            | "static" | "struct" | "switch" | "typedef" | "union" | "unsigned" | "void"
            | "volatile" | "while" | "main"
    ) {
        format!("lust_{out}")
    } else {
        out
    }
}

pub struct CCodegen {
    structs: HashSet<String>,
    struct_defs: HashMap<String, StructDef>,
    fn_sigs: HashMap<String, (Vec<CType>, CType)>,
    methods: HashMap<(String, String), (Vec<CType>, CType)>,
    var_types: Vec<HashMap<String, CType>>,
    current_self_struct: Option<String>,
    tmp_counter: usize,
    indent: usize,
}

impl CCodegen {
    pub fn new() -> Self {
        Self {
            structs: HashSet::new(),
            struct_defs: HashMap::new(),
            fn_sigs: HashMap::new(),
            methods: HashMap::new(),
            var_types: vec![HashMap::new()],
            current_self_struct: None,
            tmp_counter: 0,
            indent: 0,
        }
    }

    fn fresh_tmp(&mut self, prefix: &str) -> String {
        let id = self.tmp_counter;
        self.tmp_counter += 1;
        format!("__{prefix}_{id}")
    }

    fn push_scope(&mut self) {
        self.var_types.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.var_types.pop();
    }

    fn bind_var(&mut self, name: &str, ty: CType) {
        if let Some(scope) = self.var_types.last_mut() {
            scope.insert(name.to_string(), ty);
        }
    }

    fn lookup_var(&self, name: &str) -> Option<&CType> {
        for scope in self.var_types.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(ty);
            }
        }
        None
    }

    fn pad(&self) -> String {
        "    ".repeat(self.indent)
    }

    pub fn generate_program(&mut self, items: &[Item]) -> Result<String> {
        // Pass 1: collect struct names & defs
        let mut ordered_structs: Vec<&StructDef> = Vec::new();
        for item in items {
            if let ItemKind::Struct(sd) = &item.kind {
                let sname = short_name(&sd.name).to_string();
                self.structs.insert(sname.clone());
                self.struct_defs.insert(sname, sd.clone());
                ordered_structs.push(sd);
            }
        }

        // Pass 2: collect function and method signatures
        let mut functions: Vec<&FunctionDef> = Vec::new();
        let mut impls: Vec<&ImplBlock> = Vec::new();
        let mut script_stmts: Vec<&Stmt> = Vec::new();
        for item in items {
            match &item.kind {
                ItemKind::Function(f) => {
                    let params: Vec<CType> = f
                        .params
                        .iter()
                        .map(|p| CType::from_ast(&p.ty, &self.structs))
                        .collect();
                    let ret = f
                        .return_type
                        .as_ref()
                        .map(|t| CType::from_ast(t, &self.structs))
                        .unwrap_or(CType::Unit);
                    let fname = short_name(&f.name).to_string();
                    self.fn_sigs.insert(fname, (params, ret));
                    functions.push(f);
                }
                ItemKind::Impl(ib) => {
                    let target = match &ib.target_type.kind {
                        TypeKind::Named(n) => short_name(n).to_string(),
                        _ => continue,
                    };
                    for m in &ib.methods {
                        let params: Vec<CType> = m
                            .params
                            .iter()
                            .filter(|p| !p.is_self)
                            .map(|p| CType::from_ast(&p.ty, &self.structs))
                            .collect();
                        let ret = m
                            .return_type
                            .as_ref()
                            .map(|t| CType::from_ast(t, &self.structs))
                            .unwrap_or(CType::Unit);
                        let mname = short_name(&m.name).to_string();
                        self.methods.insert((target.clone(), mname), (params, ret));
                    }
                    impls.push(ib);
                }
                ItemKind::Script(stmts) => {
                    script_stmts.extend(stmts.iter());
                }
                _ => {}
            }
        }

        let mut out = String::new();
        out.push_str(Self::runtime_header());

        // Forward declare structs
        for sd in &ordered_structs {
            let cname = sanitize_ident(short_name(&sd.name));
            out.push_str(&format!("typedef struct {cname} {cname};\n"));
        }
        out.push('\n');

        // Define structs + arena constructors
        for sd in &ordered_structs {
            let cname = sanitize_ident(short_name(&sd.name));
            out.push_str(&format!("struct {cname} {{\n"));
            for field in &sd.fields {
                let fty = CType::from_ast(&field.ty, &self.structs);
                let fname = sanitize_ident(&field.name);
                out.push_str(&format!("    {} {};\n", fty.c_decl(), fname));
            }
            out.push_str("};\n");
            out.push_str(&format!(
                "static inline {cname}* __lust_box_{cname}({cname} val) {{\n"
            ));
            out.push_str(&format!(
                "    {cname} *p = ({cname}*)__lust_arena_alloc(sizeof({cname}), _Alignof({cname}));\n"
            ));
            out.push_str("    *p = val;\n");
            out.push_str("    return p;\n");
            out.push_str("}\n\n");
        }

        // Forward declare functions and methods
        for f in &functions {
            out.push_str(&self.fn_prototype(f, None));
            out.push_str(";\n");
        }
        for ib in &impls {
            if let TypeKind::Named(n) = &ib.target_type.kind {
                let target = short_name(n);
                for m in &ib.methods {
                    out.push_str(&self.fn_prototype(m, Some(target)));
                    out.push_str(";\n");
                }
            }
        }
        out.push('\n');

        // Emit functions
        for f in &functions {
            out.push_str(&self.emit_function(f, None)?);
            out.push('\n');
        }

        // Emit impl methods
        for ib in &impls {
            if let TypeKind::Named(n) = &ib.target_type.kind {
                let target = short_name(n);
                for m in &ib.methods {
                    out.push_str(&self.emit_function(m, Some(target))?);
                    out.push('\n');
                }
            }
        }

        // Emit main()
        out.push_str("int main(void) {\n");
        self.push_scope();
        self.indent = 1;
        for stmt in &script_stmts {
            self.emit_stmt(stmt, &mut out)?;
        }
        self.pop_scope();
        out.push_str("    __lust_arena_free();\n");
        out.push_str("    return 0;\n");
        out.push_str("}\n");

        Ok(out)
    }

    fn fn_prototype(&self, f: &FunctionDef, impl_struct: Option<&str>) -> String {
        let ret = f
            .return_type
            .as_ref()
            .map(|t| CType::from_ast(t, &self.structs))
            .unwrap_or(CType::Unit);
        let cname = match impl_struct {
            Some(s) => format!("{}__{}", sanitize_ident(s), sanitize_ident(short_name(&f.name))),
            None => sanitize_ident(short_name(&f.name)),
        };
        let mut params = Vec::new();
        for p in &f.params {
            if p.is_self {
                if let Some(s) = impl_struct {
                    params.push(format!("{}* self", sanitize_ident(s)));
                }
            } else {
                let pty = CType::from_ast(&p.ty, &self.structs);
                params.push(format!("{} {}", pty.c_decl(), sanitize_ident(&p.name)));
            }
        }
        let params_str = if params.is_empty() {
            "void".to_string()
        } else {
            params.join(", ")
        };
        format!("static {} {}({})", ret.c_decl(), cname, params_str)
    }

    fn emit_function(&mut self, f: &FunctionDef, impl_struct: Option<&str>) -> Result<String> {
        let mut out = self.fn_prototype(f, impl_struct);
        out.push_str(" {\n");
        self.push_scope();
        self.current_self_struct = impl_struct.map(|s| s.to_string());
        for p in &f.params {
            if p.is_self {
                if let Some(s) = impl_struct {
                    self.bind_var("self", CType::Struct(s.to_string()));
                }
            } else {
                let pty = CType::from_ast(&p.ty, &self.structs);
                self.bind_var(&p.name, pty);
            }
        }
        self.indent = 1;
        for stmt in &f.body {
            self.emit_stmt(stmt, &mut out)?;
        }
        self.current_self_struct = None;
        self.pop_scope();
        out.push_str("}\n");
        Ok(out)
    }

    fn infer_expr_type(&self, expr: &Expr) -> CType {
        match &expr.kind {
            ExprKind::Literal(Literal::Integer(_)) => CType::Int,
            ExprKind::Literal(Literal::Float(_)) => CType::Float,
            ExprKind::Literal(Literal::Bool(_)) => CType::Bool,
            ExprKind::Literal(Literal::String(_)) => CType::Str,
            ExprKind::Identifier(name) => self
                .lookup_var(short_name(name))
                .cloned()
                .unwrap_or(CType::Int),
            ExprKind::Paren(inner) => self.infer_expr_type(inner),
            ExprKind::StructLiteral { name, .. } => CType::Struct(short_name(name).to_string()),
            ExprKind::FieldAccess { object, field } => {
                if let CType::Struct(sname) = self.infer_expr_type(object) {
                    if let Some(sd) = self.struct_defs.get(&sname) {
                        if let Some(f) = sd.fields.iter().find(|f| f.name == *field) {
                            return CType::from_ast(&f.ty, &self.structs);
                        }
                    }
                }
                CType::Int
            }
            ExprKind::Binary { left, op, right } => match op {
                BinaryOp::Eq
                | BinaryOp::Ne
                | BinaryOp::Lt
                | BinaryOp::Le
                | BinaryOp::Gt
                | BinaryOp::Ge
                | BinaryOp::And
                | BinaryOp::Or => CType::Bool,
                BinaryOp::Concat => CType::Str,
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => {
                    if matches!(self.infer_expr_type(left), CType::Float)
                        || matches!(self.infer_expr_type(right), CType::Float)
                    {
                        CType::Float
                    } else {
                        CType::Int
                    }
                }
                _ => CType::Int,
            },
            ExprKind::Unary { op, operand } => match op {
                UnaryOp::Not => CType::Bool,
                UnaryOp::Neg => self.infer_expr_type(operand),
                UnaryOp::BitNot => CType::Int,
            },
            ExprKind::Call { callee, .. } => {
                if let Some(name) = extract_callee_path(callee) {
                    match name.as_str() {
                        "tostring" => return CType::Str,
                        "array.len" | "string.len" | "table.len" => return CType::Int,
                        _ => {
                            if let Some((_, ret)) = self.fn_sigs.get(short_name(&name)) {
                                return ret.clone();
                            }
                        }
                    }
                }
                CType::Int
            }
            ExprKind::MethodCall {
                receiver, method, ..
            } => {
                if let CType::Struct(sname) = self.infer_expr_type(receiver) {
                    if let Some((_, ret)) = self.methods.get(&(sname, method.clone())) {
                        return ret.clone();
                    }
                }
                CType::Int
            }
            ExprKind::EnumConstructor {
                enum_name,
                variant,
                args,
            } => {
                if short_name(enum_name) == "Option" && variant == "Some" {
                    if let Some(arg) = args.first() {
                        if let CType::Struct(sname) = self.infer_expr_type(arg) {
                            return CType::OptionPtr(sname);
                        }
                    }
                }
                CType::Int
            }
            _ => CType::Int,
        }
    }

    fn emit_stmt(&mut self, stmt: &Stmt, out: &mut String) -> Result<()> {
        let pad = self.pad();
        match &stmt.kind {
            StmtKind::Local {
                bindings,
                initializer,
            } => {
                let inits = initializer.as_deref().unwrap_or(&[]);
                for (i, binding) in bindings.iter().enumerate() {
                    let ty = binding
                        .type_annotation
                        .as_ref()
                        .map(|t| CType::from_ast(t, &self.structs))
                        .or_else(|| inits.get(i).map(|e| self.infer_expr_type(e)))
                        .unwrap_or(CType::Int);
                    let cname = sanitize_ident(&binding.name);
                    let init_str = if let Some(expr) = inits.get(i) {
                        self.emit_expr_with_hint(expr, Some(&ty))?
                    } else {
                        Self::default_c_init(&ty)
                    };
                    out.push_str(&format!("{pad}{} {cname} = {init_str};\n", ty.c_decl()));
                    self.bind_var(&binding.name, ty);
                }
            }
            StmtKind::Assign { targets, values } => {
                for (t, v) in targets.iter().zip(values.iter()) {
                    // Fast path: s = s .. rhs  -->  in-place amortized string append!
                    if let (
                        ExprKind::Identifier(tname),
                        ExprKind::Binary {
                            left,
                            op: BinaryOp::Concat,
                            right,
                        },
                    ) = (&t.kind, &v.kind)
                    {
                        if let ExprKind::Identifier(lname) = &left.kind {
                            if short_name(tname) == short_name(lname)
                                && matches!(self.lookup_var(short_name(tname)), Some(CType::Str))
                            {
                                let tc = sanitize_ident(short_name(tname));
                                // Fast path: s = s .. tostring(int_expr)
                                if let ExprKind::Call { callee, args, .. } = &right.kind {
                                    if extract_callee_path(callee).as_deref() == Some("tostring")
                                        && args.len() == 1
                                        && matches!(self.infer_expr_type(&args[0]), CType::Int)
                                    {
                                        let arg_c = self.emit_expr(&args[0])?;
                                        out.push_str(&format!(
                                            "{pad}__lust_str_append_i64(&{tc}, {arg_c});\n"
                                        ));
                                        continue;
                                    }
                                }
                                let rhs_c = self.emit_expr(right)?;
                                out.push_str(&format!(
                                    "{pad}__lust_str_append(&{tc}, {rhs_c});\n"
                                ));
                                continue;
                            }
                        }
                    }

                    let lhs = self.emit_lvalue(t)?;
                    let lhs_ty = self.infer_expr_type(t);
                    let rhs = self.emit_expr_with_hint(v, Some(&lhs_ty))?;
                    out.push_str(&format!("{pad}{lhs} = {rhs};\n"));
                }
            }
            StmtKind::CompoundAssign { target, op, value } => {
                let lhs = self.emit_lvalue(target)?;
                let rhs = self.emit_expr(value)?;
                let op_str = match op {
                    BinaryOp::Add => "+=",
                    BinaryOp::Sub => "-=",
                    BinaryOp::Mul => "*=",
                    BinaryOp::Div => "/=",
                    BinaryOp::Mod => "%=",
                    _ => {
                        return Err(LustError::RuntimeError {
                            message: format!("Unsupported compound assign op in C backend: {op:?}"),
                        })
                    }
                };
                out.push_str(&format!("{pad}{lhs} {op_str} {rhs};\n"));
            }
            StmtKind::Expr(expr) => {
                if let Some(stmt_code) = self.try_emit_stmt_call(expr)? {
                    out.push_str(&format!("{pad}{stmt_code};\n"));
                } else {
                    let e = self.emit_expr(expr)?;
                    out.push_str(&format!("{pad}(void)({e});\n"));
                }
            }
            StmtKind::If {
                condition,
                then_block,
                elseif_branches,
                else_block,
            } => {
                self.emit_if_chain(condition, then_block, elseif_branches, else_block.as_deref(), out)?;
            }
            StmtKind::While { condition, body } => {
                let cond_c = self.emit_expr(condition)?;
                out.push_str(&format!("{pad}while ({cond_c}) {{\n"));
                self.push_scope();
                self.indent += 1;
                for s in body {
                    self.emit_stmt(s, out)?;
                }
                self.indent -= 1;
                self.pop_scope();
                out.push_str(&format!("{pad}}}\n"));
            }
            StmtKind::ForNumeric {
                variable,
                start,
                end,
                step,
                body,
            } => {
                let v = sanitize_ident(variable);
                let s_c = self.emit_expr(start)?;
                let e_c = self.emit_expr(end)?;
                let end_tmp = self.fresh_tmp("end");
                let step_c = match step {
                    Some(st) => self.emit_expr(st)?,
                    None => "1LL".to_string(),
                };
                out.push_str(&format!("{pad}for (int64_t {v} = {s_c}, {end_tmp} = {e_c}; {v} <= {end_tmp}; {v} += {step_c}) {{\n"));
                self.push_scope();
                self.bind_var(variable, CType::Int);
                self.indent += 1;
                for s in body {
                    self.emit_stmt(s, out)?;
                }
                self.indent -= 1;
                self.pop_scope();
                out.push_str(&format!("{pad}}}\n"));
            }
            StmtKind::Return(vals) => {
                if let Some(first) = vals.first() {
                    let e = self.emit_expr(first)?;
                    out.push_str(&format!("{pad}return {e};\n"));
                } else {
                    out.push_str(&format!("{pad}return;\n"));
                }
            }
            StmtKind::Break => out.push_str(&format!("{pad}break;\n")),
            StmtKind::Continue => out.push_str(&format!("{pad}continue;\n")),
            StmtKind::Block(stmts) | StmtKind::Scoped(stmts) => {
                out.push_str(&format!("{pad}{{\n"));
                self.push_scope();
                self.indent += 1;
                for s in stmts {
                    self.emit_stmt(s, out)?;
                }
                self.indent -= 1;
                self.pop_scope();
                out.push_str(&format!("{pad}}}\n"));
            }
            other => {
                return Err(LustError::RuntimeError {
                    message: format!("Statement not yet supported by C AOT backend: {other:?}"),
                });
            }
        }
        Ok(())
    }

    fn emit_if_chain(
        &mut self,
        condition: &Expr,
        then_block: &[Stmt],
        elseif_branches: &[(Expr, Vec<Stmt>)],
        else_block: Option<&[Stmt]>,
        out: &mut String,
    ) -> Result<()> {
        let pad = self.pad();
        // Pattern-matching `if expr is Ok(v)` / `if expr is Some(v)`
        if let ExprKind::IsPattern { expr, pattern } = &condition.kind {
            if let Pattern::Enum {
                variant, bindings, ..
            } = pattern
            {
                if variant == "Ok" {
                    if let ExprKind::Index { object, index } = &expr.kind {
                        let arr_ty = self.infer_expr_type(object);
                        let elem_ty = match arr_ty {
                            CType::Array(inner) => *inner,
                            _ => CType::Int,
                        };
                        let obj_c = self.emit_expr(object)?;
                        let idx_c = self.emit_expr(index)?;
                        let idx_tmp = self.fresh_tmp("idx");
                        out.push_str(&format!("{pad}{{\n"));
                        out.push_str(&format!("{pad}    int64_t {idx_tmp} = {idx_c};\n"));
                        out.push_str(&format!(
                            "{pad}    if ({idx_tmp} >= 0 && (size_t){idx_tmp} < ({obj_c})->len) {{\n"
                        ));
                        self.push_scope();
                        self.indent += 2;
                        if let Some(Pattern::Identifier(bname)) = bindings.first() {
                            let bc = sanitize_ident(bname);
                            let p2 = self.pad();
                            out.push_str(&format!(
                                "{p2}{} {bc} = ({obj_c})->data[(size_t){idx_tmp}];\n",
                                elem_ty.c_decl()
                            ));
                            self.bind_var(bname, elem_ty);
                        }
                        for s in then_block {
                            self.emit_stmt(s, out)?;
                        }
                        self.indent -= 2;
                        self.pop_scope();
                        if let Some(eb) = else_block {
                            out.push_str(&format!("{pad}    }} else {{\n"));
                            self.push_scope();
                            self.indent += 2;
                            for s in eb {
                                self.emit_stmt(s, out)?;
                            }
                            self.indent -= 2;
                            self.pop_scope();
                        }
                        out.push_str(&format!("{pad}    }}\n"));
                        out.push_str(&format!("{pad}}}\n"));
                        return Ok(());
                    }
                } else if variant == "Some" {
                    let opt_ty = self.infer_expr_type(expr);
                    let opt_c = self.emit_expr(expr)?;
                    let tmp = self.fresh_tmp("opt");
                    if let CType::OptionPtr(sname) = opt_ty {
                        let sc = sanitize_ident(&sname);
                        out.push_str(&format!("{pad}{{\n"));
                        out.push_str(&format!("{pad}    {sc}* {tmp} = {opt_c};\n"));
                        out.push_str(&format!("{pad}    if ({tmp} != NULL) {{\n"));
                        self.push_scope();
                        self.indent += 2;
                        if let Some(Pattern::Identifier(bname)) = bindings.first() {
                            let bc = sanitize_ident(bname);
                            let p2 = self.pad();
                            out.push_str(&format!("{p2}{sc} {bc} = *{tmp};\n"));
                            self.bind_var(bname, CType::Struct(sname));
                        }
                        for s in then_block {
                            self.emit_stmt(s, out)?;
                        }
                        self.indent -= 2;
                        self.pop_scope();
                        if let Some(eb) = else_block {
                            out.push_str(&format!("{pad}    }} else {{\n"));
                            self.push_scope();
                            self.indent += 2;
                            for s in eb {
                                self.emit_stmt(s, out)?;
                            }
                            self.indent -= 2;
                            self.pop_scope();
                        }
                        out.push_str(&format!("{pad}    }}\n"));
                        out.push_str(&format!("{pad}}}\n"));
                        return Ok(());
                    }
                }
            }
        }

        let cond_c = self.emit_expr(condition)?;
        out.push_str(&format!("{pad}if ({cond_c}) {{\n"));
        self.push_scope();
        self.indent += 1;
        for s in then_block {
            self.emit_stmt(s, out)?;
        }
        self.indent -= 1;
        self.pop_scope();
        for (ec, eb) in elseif_branches {
            let ec_c = self.emit_expr(ec)?;
            out.push_str(&format!("{pad}}} else if ({ec_c}) {{\n"));
            self.push_scope();
            self.indent += 1;
            for s in eb {
                self.emit_stmt(s, out)?;
            }
            self.indent -= 1;
            self.pop_scope();
        }
        if let Some(eb) = else_block {
            out.push_str(&format!("{pad}}} else {{\n"));
            self.push_scope();
            self.indent += 1;
            for s in eb {
                self.emit_stmt(s, out)?;
            }
            self.indent -= 1;
            self.pop_scope();
        }
        out.push_str(&format!("{pad}}}\n"));
        Ok(())
    }

    fn try_emit_stmt_call(&mut self, expr: &Expr) -> Result<Option<String>> {
        if let ExprKind::Call { callee, args, .. } = &expr.kind {
            if let Some(path) = extract_callee_path(callee) {
                match path.as_str() {
                    "println" | "print" => {
                        if let Some(arg) = args.first() {
                            let ty = self.infer_expr_type(arg);
                            let arg_c = self.emit_expr(arg)?;
                            let helper = match ty {
                                CType::Float => "__lust_println_f64",
                                CType::Bool => "__lust_println_bool",
                                CType::Str => "__lust_println_str",
                                _ => "__lust_println_i64",
                            };
                            return Ok(Some(format!("{helper}({arg_c})")));
                        }
                        return Ok(Some("puts(\"\")".to_string()));
                    }
                    "array.push" => {
                        if args.len() == 2 {
                            let arr_ty = self.infer_expr_type(&args[0]);
                            let suffix = match arr_ty {
                                CType::Array(elem) => elem.array_suffix(),
                                _ => "i64".to_string(),
                            };
                            let arr_c = self.emit_expr(&args[0])?;
                            let val_c = self.emit_expr(&args[1])?;
                            return Ok(Some(format!("__lust_arr_{suffix}_push({arr_c}, {val_c})")));
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(None)
    }

    fn emit_lvalue(&mut self, expr: &Expr) -> Result<String> {
        match &expr.kind {
            ExprKind::Identifier(name) => Ok(sanitize_ident(short_name(name))),
            ExprKind::FieldAccess { object, field } => {
                let fc = sanitize_ident(field);
                if matches!(&object.kind, ExprKind::Identifier(n) if n == "self")
                    && self.current_self_struct.is_some()
                {
                    Ok(format!("self->{fc}"))
                } else {
                    let oc = self.emit_lvalue(object)?;
                    Ok(format!("{oc}.{fc}"))
                }
            }
            ExprKind::Index { object, index } => {
                let oc = self.emit_expr(object)?;
                let ic = self.emit_expr(index)?;
                Ok(format!("({oc})->data[(size_t)({ic})]"))
            }
            _ => self.emit_expr(expr),
        }
    }

    fn emit_expr(&mut self, expr: &Expr) -> Result<String> {
        self.emit_expr_with_hint(expr, None)
    }

    fn emit_expr_with_hint(&mut self, expr: &Expr, hint: Option<&CType>) -> Result<String> {
        match &expr.kind {
            ExprKind::Literal(Literal::Integer(n)) => {
                if *n == i64::MIN {
                    Ok("INT64_MIN".to_string())
                } else {
                    Ok(format!("{n}LL"))
                }
            }
            ExprKind::Literal(Literal::Float(f)) => {
                let s = format!("{f:?}");
                if s.contains('.') || s.contains('e') || s.contains('E') {
                    Ok(s)
                } else {
                    Ok(format!("{s}.0"))
                }
            }
            ExprKind::Literal(Literal::Bool(b)) => Ok(if *b { "true" } else { "false" }.to_string()),
            ExprKind::Literal(Literal::String(s)) => {
                let escaped = escape_c_string(s);
                Ok(format!("__lust_str_from_lit(\"{escaped}\", {})", s.len()))
            }
            ExprKind::Identifier(name) => Ok(sanitize_ident(short_name(name))),
            ExprKind::Paren(inner) => Ok(format!("({})", self.emit_expr_with_hint(inner, hint)?)),
            ExprKind::Array(elems) => {
                let elem_ty = match hint {
                    Some(CType::Array(inner)) => inner.as_ref().clone(),
                    _ => elems
                        .first()
                        .map(|e| self.infer_expr_type(e))
                        .unwrap_or(CType::Int),
                };
                let suffix = elem_ty.array_suffix();
                if elems.is_empty() {
                    Ok(format!("__lust_arr_{suffix}_new(0)"))
                } else {
                    let mut parts = Vec::with_capacity(elems.len());
                    for e in elems {
                        parts.push(self.emit_expr(e)?);
                    }
                    Ok(format!(
                        "__lust_arr_{suffix}_from((const {}[]){{{}}}, {})",
                        elem_ty.c_decl(),
                        parts.join(", "),
                        elems.len()
                    ))
                }
            }
            ExprKind::StructLiteral { name, fields } => {
                let sname = short_name(name);
                let cname = sanitize_ident(sname);
                let sd_opt = self.struct_defs.get(sname).cloned();
                let mut inits = Vec::with_capacity(fields.len());
                for f in fields {
                    let fty = sd_opt.as_ref().and_then(|sd| {
                        sd.fields
                            .iter()
                            .find(|sf| sf.name == f.name)
                            .map(|sf| CType::from_ast(&sf.ty, &self.structs))
                    });
                    let val_c = self.emit_expr_with_hint(&f.value, fty.as_ref())?;
                    inits.push(format!(".{} = {val_c}", sanitize_ident(&f.name)));
                }
                Ok(format!("({cname}){{{}}}", inits.join(", ")))
            }
            ExprKind::EnumConstructor {
                enum_name,
                variant,
                args,
            } => {
                if short_name(enum_name) == "Option" {
                    if variant == "None" {
                        return Ok("NULL".to_string());
                    }
                    if variant == "Some" && args.len() == 1 {
                        let arg_ty = self.infer_expr_type(&args[0]);
                        let arg_c = self.emit_expr(&args[0])?;
                        if let CType::Struct(sname) = arg_ty {
                            let cname = sanitize_ident(&sname);
                            return Ok(format!("__lust_box_{cname}({arg_c})"));
                        }
                    }
                }
                Err(LustError::RuntimeError {
                    message: format!(
                        "Enum constructor {enum_name}.{variant} not yet supported in C AOT backend"
                    ),
                })
            }
            ExprKind::FieldAccess { object, field } => {
                // Handle `Option.None` parsed as FieldAccess
                if let ExprKind::Identifier(n) = &object.kind {
                    if short_name(n) == "Option" && field == "None" {
                        return Ok("NULL".to_string());
                    }
                }
                let fc = sanitize_ident(field);
                if matches!(&object.kind, ExprKind::Identifier(n) if n == "self")
                    && self.current_self_struct.is_some()
                {
                    Ok(format!("self->{fc}"))
                } else {
                    let oc = self.emit_expr(object)?;
                    Ok(format!("({oc}).{fc}"))
                }
            }
            ExprKind::Index { object, index } => {
                let oc = self.emit_expr(object)?;
                let ic = self.emit_expr(index)?;
                Ok(format!("({oc})->data[(size_t)({ic})]"))
            }
            ExprKind::Binary { left, op, right } => {
                let lt = self.infer_expr_type(left);
                let rt = self.infer_expr_type(right);
                let l = self.emit_expr(left)?;
                let r = self.emit_expr(right)?;
                let is_float = matches!(lt, CType::Float) || matches!(rt, CType::Float);
                match op {
                    BinaryOp::Add => Ok(if is_float {
                        format!("({l} + {r})")
                    } else {
                        format!("(int64_t)((uint64_t)({l}) + (uint64_t)({r}))")
                    }),
                    BinaryOp::Sub => Ok(if is_float {
                        format!("({l} - {r})")
                    } else {
                        format!("(int64_t)((uint64_t)({l}) - (uint64_t)({r}))")
                    }),
                    BinaryOp::Mul => Ok(if is_float {
                        format!("({l} * {r})")
                    } else {
                        format!("(int64_t)((uint64_t)({l}) * (uint64_t)({r}))")
                    }),
                    BinaryOp::Div => Ok(if is_float {
                        format!("((double)({l}) / (double)({r}))")
                    } else {
                        format!("__lust_div_i64({l}, {r})")
                    }),
                    BinaryOp::Mod => Ok(if is_float {
                        format!("fmod((double)({l}), (double)({r}))")
                    } else {
                        format!("__lust_rem_i64({l}, {r})")
                    }),
                    BinaryOp::FloorDiv => Ok(format!("__lust_floordiv_i64({l}, {r})")),
                    BinaryOp::Pow => Ok(if is_float {
                        format!("pow((double)({l}), (double)({r}))")
                    } else {
                        format!("__lust_pow_i64({l}, {r})")
                    }),
                    BinaryOp::Eq => Ok(format!("({l} == {r})")),
                    BinaryOp::Ne => Ok(format!("({l} != {r})")),
                    BinaryOp::Lt => Ok(format!("({l} < {r})")),
                    BinaryOp::Le => Ok(format!("({l} <= {r})")),
                    BinaryOp::Gt => Ok(format!("({l} > {r})")),
                    BinaryOp::Ge => Ok(format!("({l} >= {r})")),
                    BinaryOp::And => Ok(format!("({l} && {r})")),
                    BinaryOp::Or => Ok(format!("({l} || {r})")),
                    BinaryOp::BitAnd => Ok(format!("({l} & {r})")),
                    BinaryOp::BitOr => Ok(format!("({l} | {r})")),
                    BinaryOp::BitXor => Ok(format!("({l} ^ {r})")),
                    BinaryOp::Shl => Ok(format!("(int64_t)((uint64_t)({l}) << ((uint64_t)({r}) & 63ULL))")),
                    BinaryOp::Shr => Ok(format!("(({l}) >> ((uint64_t)({r}) & 63ULL))")),
                    BinaryOp::Concat => Ok(format!("__lust_str_concat({l}, {r})")),
                    BinaryOp::Range => Err(LustError::RuntimeError {
                        message: "Range expression not supported in C AOT backend".to_string(),
                    }),
                }
            }
            ExprKind::Unary { op, operand } => {
                let ot = self.infer_expr_type(operand);
                let v = self.emit_expr(operand)?;
                match op {
                    UnaryOp::Neg => Ok(if matches!(ot, CType::Float) {
                        format!("(-({v}))")
                    } else {
                        format!("(int64_t)(0ULL - (uint64_t)({v}))")
                    }),
                    UnaryOp::Not => Ok(format!("(!({v}))")),
                    UnaryOp::BitNot => Ok(format!("(~({v}))")),
                }
            }
            ExprKind::Call { callee, args, .. } => {
                let Some(path) = extract_callee_path(callee) else {
                    return Err(LustError::RuntimeError {
                        message: "Indirect call not yet supported in C AOT backend".to_string(),
                    });
                };
                match path.as_str() {
                    "Option.Some" if args.len() == 1 => {
                        let arg_ty = self.infer_expr_type(&args[0]);
                        let arg_c = self.emit_expr(&args[0])?;
                        if let CType::Struct(sname) = arg_ty {
                            return Ok(format!("__lust_box_{}({arg_c})", sanitize_ident(&sname)));
                        }
                        Ok(arg_c)
                    }
                    "array.len" | "table.len" if args.len() == 1 => {
                        let a = self.emit_expr(&args[0])?;
                        Ok(format!("(int64_t)(({a})->len)"))
                    }
                    "string.len" if args.len() == 1 => {
                        let a = self.emit_expr(&args[0])?;
                        Ok(format!("(int64_t)(({a}).len)"))
                    }
                    "tostring" if args.len() == 1 => {
                        let ty = self.infer_expr_type(&args[0]);
                        let a = self.emit_expr(&args[0])?;
                        match ty {
                            CType::Int => Ok(format!("__lust_i64_to_str({a})")),
                            CType::Float => Ok(format!("__lust_f64_to_str({a})")),
                            CType::Bool => Ok(format!("__lust_bool_to_str({a})")),
                            CType::Str => Ok(a),
                            _ => Ok(format!("__lust_i64_to_str((int64_t)({a}))")),
                        }
                    }
                    _ => {
                        let cname = sanitize_ident(short_name(&path));
                        let mut cargs = Vec::with_capacity(args.len());
                        for a in args {
                            cargs.push(self.emit_expr(a)?);
                        }
                        Ok(format!("{cname}({})", cargs.join(", ")))
                    }
                }
            }
            ExprKind::MethodCall {
                receiver,
                method,
                args,
                ..
            } => {
                let recv_ty = self.infer_expr_type(receiver);
                if let CType::Struct(sname) = recv_ty {
                    let fname = format!("{}__{}", sanitize_ident(&sname), sanitize_ident(method));
                    let recv_ptr = if matches!(&receiver.kind, ExprKind::Identifier(n) if n == "self")
                        && self.current_self_struct.is_some()
                    {
                        "self".to_string()
                    } else {
                        format!("&({})", self.emit_lvalue(receiver)?)
                    };
                    let mut cargs = Vec::with_capacity(args.len() + 1);
                    cargs.push(recv_ptr);
                    for a in args {
                        cargs.push(self.emit_expr(a)?);
                    }
                    return Ok(format!("{fname}({})", cargs.join(", ")));
                }
                Err(LustError::RuntimeError {
                    message: format!("Method call '{method}' on non-struct not supported in C AOT"),
                })
            }
            other => Err(LustError::RuntimeError {
                message: format!("Expression not yet supported by C AOT backend: {other:?}"),
            }),
        }
    }

    fn default_c_init(ty: &CType) -> String {
        match ty {
            CType::Int => "0LL".to_string(),
            CType::Float => "0.0".to_string(),
            CType::Bool => "false".to_string(),
            CType::Str => "__lust_str_from_lit(\"\", 0)".to_string(),
            CType::OptionPtr(_) => "NULL".to_string(),
            CType::Array(elem) => format!("__lust_arr_{}_new(0)", elem.array_suffix()),
            CType::Struct(n) => format!("({}){{0}}", sanitize_ident(n)),
            _ => "0".to_string(),
        }
    }

    fn runtime_header() -> &'static str {
        r#"/* Generated by Lust AOT compiler (C11 backend) */
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>

/* ---- Fast bump arena for boxed Option<Struct> nodes & temp strings ---- */
typedef struct LustArenaBlock {
    struct LustArenaBlock *next;
    size_t cap;
    size_t used;
    unsigned char data[];
} LustArenaBlock;

static LustArenaBlock *__lust_arena_head = NULL;

static inline void *__lust_arena_alloc(size_t size, size_t align) {
    if (align < 8) align = 8;
    LustArenaBlock *b = __lust_arena_head;
    if (b) {
        size_t aligned = (b->used + (align - 1)) & ~(align - 1);
        if (aligned + size <= b->cap) {
            void *p = b->data + aligned;
            b->used = aligned + size;
            return p;
        }
    }
    size_t block_cap = size > (1024 * 1024) ? size + align : (1024 * 1024);
    LustArenaBlock *nb = (LustArenaBlock*)malloc(sizeof(LustArenaBlock) + block_cap);
    if (!nb) abort();
    nb->next = __lust_arena_head;
    nb->cap = block_cap;
    nb->used = size;
    __lust_arena_head = nb;
    return nb->data;
}

static void __lust_arena_free(void) {
    LustArenaBlock *b = __lust_arena_head;
    while (b) {
        LustArenaBlock *n = b->next;
        free(b);
        b = n;
    }
    __lust_arena_head = NULL;
}

/* ---- Arithmetic helpers (zero UB) ---- */
static inline int64_t __lust_div_i64(int64_t a, int64_t b) {
    if (b == 0) abort();
    if (a == INT64_MIN && b == -1) return INT64_MIN;
    return a / b;
}

static inline int64_t __lust_rem_i64(int64_t a, int64_t b) {
    if (b == 0) abort();
    if (a == INT64_MIN && b == -1) return 0;
    return a % b;
}

static inline int64_t __lust_floordiv_i64(int64_t a, int64_t b) {
    if (b == 0) abort();
    if (a == INT64_MIN && b == -1) return INT64_MIN;
    int64_t q = a / b;
    int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) q -= 1;
    return q;
}

static inline int64_t __lust_pow_i64(int64_t base, int64_t exp) {
    if (exp < 0) abort();
    uint64_t b = (uint64_t)base, e = (uint64_t)exp, res = 1;
    while (e > 0) {
        if (e & 1ULL) res *= b;
        b *= b;
        e >>= 1;
    }
    return (int64_t)res;
}

/* ---- Dynamic strings (amortized O(1) append) ---- */
typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} LustStr;

static inline LustStr __lust_str_from_lit(const char *lit, size_t len) {
    LustStr s = { (char*)lit, len, 0 };
    return s;
}

static inline void __lust_str_reserve(LustStr *s, size_t needed) {
    if (s->cap >= needed) return;
    size_t new_cap = s->cap ? s->cap * 2 : 32;
    while (new_cap < needed) new_cap *= 2;
    if (s->cap == 0) {
        char *buf = (char*)malloc(new_cap);
        if (!buf) abort();
        if (s->len > 0) memcpy(buf, s->ptr, s->len);
        buf[s->len] = '\0';
        s->ptr = buf;
    } else {
        char *buf = (char*)realloc(s->ptr, new_cap);
        if (!buf) abort();
        s->ptr = buf;
    }
    s->cap = new_cap;
}

static inline void __lust_str_append(LustStr *s, LustStr rhs) {
    if (rhs.len == 0) return;
    __lust_str_reserve(s, s->len + rhs.len + 1);
    memcpy(s->ptr + s->len, rhs.ptr, rhs.len);
    s->len += rhs.len;
    s->ptr[s->len] = '\0';
}

static inline void __lust_str_append_i64(LustStr *s, int64_t v) {
    char buf[32];
    int n = snprintf(buf, sizeof(buf), "%" PRId64, v);
    if (n <= 0) return;
    __lust_str_reserve(s, s->len + (size_t)n + 1);
    memcpy(s->ptr + s->len, buf, (size_t)n);
    s->len += (size_t)n;
    s->ptr[s->len] = '\0';
}

static inline LustStr __lust_str_concat(LustStr a, LustStr b) {
    size_t len = a.len + b.len;
    char *buf = (char*)malloc(len + 1);
    if (!buf) abort();
    if (a.len) memcpy(buf, a.ptr, a.len);
    if (b.len) memcpy(buf + a.len, b.ptr, b.len);
    buf[len] = '\0';
    LustStr out = { buf, len, len + 1 };
    return out;
}

static inline LustStr __lust_i64_to_str(int64_t v) {
    char *buf = (char*)__lust_arena_alloc(32, 1);
    int n = snprintf(buf, 32, "%" PRId64, v);
    LustStr s = { buf, n > 0 ? (size_t)n : 0, 0 };
    return s;
}

static inline LustStr __lust_f64_to_str(double v) {
    char *buf = (char*)__lust_arena_alloc(64, 1);
    int n = snprintf(buf, 64, "%.15g", v);
    LustStr s = { buf, n > 0 ? (size_t)n : 0, 0 };
    return s;
}

static inline LustStr __lust_bool_to_str(bool v) {
    return v ? __lust_str_from_lit("true", 4) : __lust_str_from_lit("false", 5);
}

/* ---- Typed dynamic arrays ---- */
#define LUST_DEFINE_ARRAY(SUFFIX, ELEM_T) \
typedef struct { \
    ELEM_T *data; \
    size_t len; \
    size_t cap; \
} LustArr_##SUFFIX; \
static inline LustArr_##SUFFIX *__lust_arr_##SUFFIX##_new(size_t cap) { \
    LustArr_##SUFFIX *a = (LustArr_##SUFFIX*)malloc(sizeof(LustArr_##SUFFIX)); \
    if (!a) abort(); \
    a->len = 0; \
    a->cap = cap; \
    a->data = cap ? (ELEM_T*)malloc(cap * sizeof(ELEM_T)) : NULL; \
    return a; \
} \
static inline LustArr_##SUFFIX *__lust_arr_##SUFFIX##_from(const ELEM_T *items, size_t n) { \
    LustArr_##SUFFIX *a = __lust_arr_##SUFFIX##_new(n); \
    if (n) memcpy(a->data, items, n * sizeof(ELEM_T)); \
    a->len = n; \
    return a; \
} \
static inline void __lust_arr_##SUFFIX##_push(LustArr_##SUFFIX *a, ELEM_T val) { \
    if (a->len == a->cap) { \
        size_t nc = a->cap ? a->cap * 2 : 16; \
        a->data = (ELEM_T*)realloc(a->data, nc * sizeof(ELEM_T)); \
        if (!a->data) abort(); \
        a->cap = nc; \
    } \
    a->data[a->len++] = val; \
}

LUST_DEFINE_ARRAY(i64, int64_t)
LUST_DEFINE_ARRAY(f64, double)
LUST_DEFINE_ARRAY(bool, bool)
LUST_DEFINE_ARRAY(str, LustStr)

/* ---- Print helpers ---- */
static inline void __lust_println_i64(int64_t v) { printf("%" PRId64 "\n", v); }
static inline void __lust_println_f64(double v) { printf("%.15g\n", v); }
static inline void __lust_println_bool(bool v) { puts(v ? "true" : "false"); }
static inline void __lust_println_str(LustStr s) { fwrite(s.ptr, 1, s.len, stdout); putchar('\n'); }

"#
    }
}

fn extract_callee_path(expr: &Expr) -> Option<String> {
    match &expr.kind {
        ExprKind::Identifier(n) => Some(n.clone()),
        ExprKind::FieldAccess { object, field } => {
            let base = extract_callee_path(object)?;
            Some(format!("{base}.{field}"))
        }
        _ => None,
    }
}

fn escape_c_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

pub fn compile_c_executable(
    c_path: &Path,
    output_bin: &Path,
    release_mode: &str,
    strip: bool,
) -> Result<PathBuf> {
    let cc = std::env::var("CC")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| {
            for cand in ["cc", "clang", "gcc"] {
                if Command::new(cand)
                    .arg("--version")
                    .output()
                    .map(|o| o.status.success())
                    .unwrap_or(false)
                {
                    return cand.to_string();
                }
            }
            "cc".to_string()
        });

    let opt = match release_mode {
        "Debug" => "-O0",
        "ReleaseSmall" => "-Os",
        "ReleaseSafe" => "-O2",
        _ => "-O3",
    };

    let mut cmd = Command::new(&cc);
    cmd.args(["-std=c11", opt]);
    if strip && !cfg!(target_os = "macos") {
        cmd.arg("-s");
    }
    cmd.arg(c_path);
    cmd.arg("-o");
    cmd.arg(output_bin);
    if !cfg!(target_os = "windows") {
        cmd.arg("-lm");
    }

    let output = cmd.output().map_err(|e| LustError::RuntimeError {
        message: format!("Failed to invoke C compiler '{cc}': {e}"),
    })?;
    if !output.status.success() {
        return Err(LustError::RuntimeError {
            message: format!(
                "C compilation failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            ),
        });
    }
    Ok(output_bin.to_path_buf())
}
