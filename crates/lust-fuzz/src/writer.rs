//! Writes random, valid, terminating Lust programs. Everything derives from
//! one seed, so `seed -> program` is a pure function and any finding replays
//! from its number alone.
//!
//! The subset is chosen to drive the trace JIT: counted loops (bounded by
//! literals, counters never assigned), int/float/bool arithmetic and
//! comparisons, loop-carried locals, if/elseif/else, break/continue, calls to
//! straight-line helpers (inlinable) and to helpers with their own loops,
//! `Array<int>` push/len/index, a struct with int/float/bool fields, and
//! `Option<int>`. Every observation is appended to a string the entry
//! function returns, so the two engines' outputs can be compared directly.

use crate::rng::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ty {
    Int,
    Float,
    Bool,
    ArrInt,
    Struct,
    OptInt,
}

impl Ty {
    fn name(self) -> &'static str {
        match self {
            Ty::Int => "int",
            Ty::Float => "float",
            Ty::Bool => "bool",
            Ty::ArrInt => "Array<int>",
            Ty::Struct => "P",
            Ty::OptInt => "Option<int>",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
}

impl BinOp {
    fn text(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Mod => "%",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::Eq => "==",
            BinOp::Ne => "~=",
            BinOp::And => "and",
            BinOp::Or => "or",
        }
    }
}

#[derive(Debug, Clone)]
pub enum Expr {
    Int(i64),
    Float(&'static str),
    Bool(bool),
    Var(String),
    Bin(Box<Expr>, BinOp, Box<Expr>),
    Neg(Box<Expr>),
    Not(Box<Expr>),
    /// User function call.
    Call(String, Vec<Expr>),
    /// `math.abs(e)`, `math.min(a, b)`, `math.max(a, b)`
    Math(&'static str, Vec<Expr>),
    ArrLen(String),
    Field(String, &'static str),
    UnwrapOr(String, Box<Expr>),
    IsSome(String),
    /// `[e, e, ...]`
    ArrLit(Vec<Expr>),
    /// `P { a = e, b = e, c = e }`
    StructLit(Box<Expr>, Box<Expr>, Box<Expr>),
    Some(Box<Expr>),
    None,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Local {
        name: String,
        ty: Ty,
        init: Expr,
    },
    Assign {
        name: String,
        expr: Expr,
    },
    SetField {
        obj: String,
        field: &'static str,
        expr: Expr,
    },
    /// `out = out .. tostring(e) .. "\n"`
    Observe(Expr),
    While {
        counter: String,
        bound: i64,
        body: Vec<Stmt>,
    },
    For {
        var: String,
        lo: i64,
        hi: i64,
        body: Vec<Stmt>,
    },
    If {
        branches: Vec<(Expr, Vec<Stmt>)>,
        otherwise: Option<Vec<Stmt>>,
    },
    Break,
    Continue,
    Push {
        arr: String,
        expr: Expr,
    },
    /// `if arr[idx] is Ok(v) then body end`
    IfIndex {
        arr: String,
        idx: Expr,
        var: String,
        body: Vec<Stmt>,
    },
    /// `if opt:is_some() then local v: int = opt:unwrap() body end`
    IfSome {
        opt: String,
        var: String,
        body: Vec<Stmt>,
    },
}

#[derive(Debug, Clone)]
pub struct Func {
    pub name: String,
    pub params: Vec<(String, Ty)>,
    pub ret: Ty,
    pub body: Vec<Stmt>,
    pub ret_expr: Expr,
}

#[derive(Debug, Clone)]
pub struct Program {
    pub funcs: Vec<Func>,
    pub run: Vec<Stmt>,
}

// ── Rendering ────────────────────────────────────────────────────────────

struct Out {
    text: String,
    indent: usize,
}

impl Out {
    fn line(&mut self, s: &str) {
        for _ in 0..self.indent {
            self.text.push_str("    ");
        }
        self.text.push_str(s);
        self.text.push('\n');
    }
}

fn render_expr(e: &Expr, out: &mut String) {
    match e {
        Expr::Int(i) => {
            if *i < 0 {
                out.push_str(&format!("(0 - {})", i.unsigned_abs()));
            } else {
                out.push_str(&i.to_string());
            }
        }
        Expr::Float(f) => out.push_str(f),
        Expr::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Expr::Var(v) => out.push_str(v),
        Expr::Bin(l, op, r) => {
            out.push('(');
            render_expr(l, out);
            out.push(' ');
            out.push_str(op.text());
            out.push(' ');
            render_expr(r, out);
            out.push(')');
        }
        Expr::Neg(x) => {
            out.push_str("(-");
            render_expr(x, out);
            out.push(')');
        }
        Expr::Not(x) => {
            out.push_str("(not ");
            render_expr(x, out);
            out.push(')');
        }
        Expr::Call(name, args) => render_call(name, args, out),
        Expr::Math(name, args) => render_call(name, args, out),
        Expr::ArrLen(a) => out.push_str(&format!("array.len({a})")),
        Expr::Field(o, f) => out.push_str(&format!("{o}.{f}")),
        Expr::UnwrapOr(o, d) => {
            out.push_str(&format!("{o}:unwrap_or("));
            render_expr(d, out);
            out.push(')');
        }
        Expr::IsSome(o) => out.push_str(&format!("{o}:is_some()")),
        Expr::ArrLit(items) => {
            out.push('[');
            for (i, a) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                render_expr(a, out);
            }
            out.push(']');
        }
        Expr::StructLit(a, b, c) => {
            out.push_str("P { a = ");
            render_expr(a, out);
            out.push_str(", b = ");
            render_expr(b, out);
            out.push_str(", c = ");
            render_expr(c, out);
            out.push_str(" }");
        }
        Expr::Some(x) => {
            out.push_str("Option.Some(");
            render_expr(x, out);
            out.push(')');
        }
        Expr::None => out.push_str("Option.None"),
    }
}

fn render_call(name: &str, args: &[Expr], out: &mut String) {
    out.push_str(name);
    out.push('(');
    for (i, a) in args.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        render_expr(a, out);
    }
    out.push(')');
}

fn expr_text(e: &Expr) -> String {
    let mut s = String::new();
    render_expr(e, &mut s);
    s
}

fn render_block(stmts: &[Stmt], out: &mut Out) {
    for s in stmts {
        render_stmt(s, out);
    }
}

fn render_stmt(s: &Stmt, out: &mut Out) {
    match s {
        Stmt::Local { name, ty, init } => {
            out.line(&format!("local {name}: {} = {}", ty.name(), expr_text(init)));
        }
        Stmt::Assign { name, expr } => out.line(&format!("{name} = {}", expr_text(expr))),
        Stmt::SetField { obj, field, expr } => {
            out.line(&format!("{obj}.{field} = {}", expr_text(expr)));
        }
        Stmt::Observe(e) => {
            out.line(&format!("out = out .. tostring({}) .. \"\\n\"", expr_text(e)));
        }
        Stmt::While { counter, bound, body } => {
            // The counter advances before the body so a `continue` can
            // never skip it; every while loop terminates by construction.
            out.line(&format!("local {counter}: int = 0"));
            out.line(&format!("while {counter} < {bound} do"));
            out.indent += 1;
            out.line(&format!("{counter} = {counter} + 1"));
            render_block(body, out);
            out.indent -= 1;
            out.line("end");
        }
        Stmt::For { var, lo, hi, body } => {
            out.line(&format!(
                "for {var} = {}, {} do",
                expr_text(&Expr::Int(*lo)),
                expr_text(&Expr::Int(*hi))
            ));
            out.indent += 1;
            render_block(body, out);
            out.indent -= 1;
            out.line("end");
        }
        Stmt::If { branches, otherwise } => {
            for (i, (cond, body)) in branches.iter().enumerate() {
                let kw = if i == 0 { "if" } else { "elseif" };
                out.line(&format!("{kw} {} then", expr_text(cond)));
                out.indent += 1;
                render_block(body, out);
                out.indent -= 1;
            }
            if let Some(body) = otherwise {
                out.line("else");
                out.indent += 1;
                render_block(body, out);
                out.indent -= 1;
            }
            out.line("end");
        }
        Stmt::Break => out.line("break"),
        Stmt::Continue => out.line("continue"),
        Stmt::Push { arr, expr } => out.line(&format!("array.push({arr}, {})", expr_text(expr))),
        Stmt::IfIndex { arr, idx, var, body } => {
            out.line(&format!("if {arr}[{}] is Ok({var}) then", expr_text(idx)));
            out.indent += 1;
            render_block(body, out);
            out.indent -= 1;
            out.line("end");
        }
        Stmt::IfSome { opt, var, body } => {
            out.line(&format!("if {opt}:is_some() then"));
            out.indent += 1;
            out.line(&format!("local {var}: int = {opt}:unwrap()"));
            render_block(body, out);
            out.indent -= 1;
            out.line("end");
        }
    }
}

pub fn render(p: &Program) -> String {
    let mut out = Out {
        text: String::new(),
        indent: 0,
    };
    out.line("struct P");
    out.indent += 1;
    out.line("a: int");
    out.line("b: float");
    out.line("c: bool");
    out.indent -= 1;
    out.line("end");
    out.line("");
    for f in &p.funcs {
        let params = f
            .params
            .iter()
            .map(|(n, t)| format!("{n}: {}", t.name()))
            .collect::<Vec<_>>()
            .join(", ");
        out.line(&format!("function {}({params}): {}", f.name, f.ret.name()));
        out.indent += 1;
        render_block(&f.body, &mut out);
        out.line(&format!("return {}", expr_text(&f.ret_expr)));
        out.indent -= 1;
        out.line("end");
        out.line("");
    }
    out.line("function run(): string");
    out.indent += 1;
    out.line("local out: string = \"\"");
    render_block(&p.run, &mut out);
    out.line("return out");
    out.indent -= 1;
    out.line("end");
    out.text
}

// ── Generation ───────────────────────────────────────────────────────────

#[derive(Clone)]
struct Var {
    name: String,
    ty: Ty,
    /// Loop counters and pattern-bound names are never assigned.
    fixed: bool,
}

#[derive(Clone)]
struct FuncSig {
    name: String,
    params: Vec<Ty>,
    ret: Ty,
}

struct Gen {
    rng: Rng,
    scopes: Vec<Vec<Var>>,
    funcs: Vec<FuncSig>,
    next_name: u32,
    loop_depth: u32,
    /// Nesting depth of blocks, to bound program size.
    depth: u32,
    size: u32,
    /// Inside a helper function: no `out` to observe.
    in_func: bool,
    /// Remaining statements this program may still write.
    budget: i64,
}

const FLOAT_LITS: &[&str] = &[
    "0.0", "0.5", "1.0", "1.5", "2.25", "3.0", "10.0", "0.1", "100.5", "1234.75", "0.001", "7.0",
];

pub fn program(seed: u64, size: u32) -> Program {
    let size = size.max(1);
    let mut g = Gen {
        rng: Rng::new(seed),
        scopes: Vec::new(),
        funcs: Vec::new(),
        next_name: 0,
        loop_depth: 0,
        depth: 0,
        size,
        in_func: false,
        budget: (size as i64) * 24,
    };
    let mut funcs = Vec::new();
    let func_count = g.rng.below(size as u64 + 1) as usize;
    for _ in 0..func_count {
        funcs.push(g.func());
    }
    g.in_func = false;
    g.scopes.clear();
    g.scopes.push(Vec::new());
    let run = g.block(true);
    Program { funcs, run }
}

impl Gen {
    fn fresh(&mut self, base: &str) -> String {
        self.next_name += 1;
        format!("{base}{}", self.next_name)
    }

    fn vars_of(&self, ty: Ty) -> Vec<Var> {
        self.scopes
            .iter()
            .flatten()
            .filter(|v| v.ty == ty)
            .cloned()
            .collect()
    }

    fn mutable_vars_of(&self, ty: Ty) -> Vec<Var> {
        self.scopes
            .iter()
            .flatten()
            .filter(|v| v.ty == ty && !v.fixed)
            .cloned()
            .collect()
    }

    fn declare(&mut self, name: &str, ty: Ty, fixed: bool) {
        self.scopes.last_mut().unwrap().push(Var {
            name: name.to_string(),
            ty,
            fixed,
        });
    }

    // ── Functions ────────────────────────────────────────────────────────

    fn func(&mut self) -> Func {
        let name = self.fresh("f");
        let scalar = [Ty::Int, Ty::Float, Ty::Bool];
        let param_count = self.rng.below(4) as usize;
        let mut params = Vec::new();
        self.scopes.clear();
        self.scopes.push(Vec::new());
        for _ in 0..param_count {
            let ty = *self.rng.pick(&scalar);
            let pname = self.fresh("p");
            self.declare(&pname, ty, false);
            params.push((pname, ty));
        }
        let ret = *self.rng.pick(&scalar);
        self.in_func = true;
        // Straight-line helpers are inlinable by the trace recorder; helpers
        // with control flow are called out of line. Both are worth having.
        let straight = self.rng.chance(0.5);
        let saved_size = self.size;
        if straight {
            self.size = 1;
        }
        let body = if straight {
            let n = self.rng.below(4) as usize;
            let mut body = Vec::new();
            for _ in 0..n {
                let ty = *self.rng.pick(&scalar);
                let vname = self.fresh("t");
                let init = self.expr(ty, 2);
                body.push(Stmt::Local {
                    name: vname.clone(),
                    ty,
                    init,
                });
                self.declare(&vname, ty, false);
            }
            body
        } else {
            self.block(false)
        };
        let ret_expr = self.expr(ret, 2);
        self.size = saved_size;
        self.in_func = false;
        self.funcs.push(FuncSig {
            name: name.clone(),
            params: params.iter().map(|(_, t)| *t).collect(),
            ret,
        });
        Func {
            name,
            params,
            ret,
            body,
            ret_expr,
        }
    }

    // ── Statements ───────────────────────────────────────────────────────

    /// A block in a new scope. `entry` is the `run` body, which gets more
    /// statements and observations.
    fn block(&mut self, entry: bool) -> Vec<Stmt> {
        self.scopes.push(Vec::new());
        self.depth += 1;
        let max = if entry { self.size * 3 + 2 } else { self.size + 1 };
        let count = self.rng.range(1, max as i64) as usize;
        let mut stmts = Vec::new();
        for _ in 0..count {
            if self.budget <= 0 {
                break;
            }
            self.budget -= 1;
            if let Some(s) = self.stmt() {
                stmts.push(s);
            }
        }
        // Make sure loop bodies and branches do something visible.
        if !self.in_func && self.rng.chance(0.7) {
            if let Some(e) = self.observable() {
                stmts.push(Stmt::Observe(e));
            }
        }
        self.depth -= 1;
        self.scopes.pop();
        stmts
    }

    fn observable(&mut self) -> Option<Expr> {
        let ty = *self.rng.pick(&[Ty::Int, Ty::Int, Ty::Float, Ty::Bool]);
        let vars = self.vars_of(ty);
        if !vars.is_empty() && self.rng.chance(0.75) {
            return Some(Expr::Var(self.rng.pick(&vars).name.clone()));
        }
        let arrays = self.vars_of(Ty::ArrInt);
        if !arrays.is_empty() && self.rng.chance(0.3) {
            return Some(Expr::ArrLen(self.rng.pick(&arrays).name.clone()));
        }
        Some(self.expr(ty, 2))
    }

    fn stmt(&mut self) -> Option<Stmt> {
        let deep = self.depth >= 3;
        let roll = self.rng.below(100);
        match roll {
            0..=17 => Some(self.local()),
            18..=35 => self.assign(),
            36..=45 => {
                if !self.in_func {
                    self.observable().map(Stmt::Observe)
                } else {
                    Some(self.local())
                }
            }
            46..=58 if !deep => Some(self.while_loop()),
            59..=66 if !deep => Some(self.for_loop()),
            67..=78 if !deep => Some(self.if_stmt()),
            79..=81 if self.loop_depth > 0 => Some(self.break_or_continue()),
            82..=86 => self.push(),
            87..=91 if !deep => self.if_index(),
            92..=95 if !deep => self.if_some(),
            96..=99 => self.set_field(),
            _ => Some(self.local()),
        }
    }

    fn local(&mut self) -> Stmt {
        let ty = match self.rng.below(10) {
            0..=3 => Ty::Int,
            4..=5 => Ty::Float,
            6 => Ty::Bool,
            7 => Ty::ArrInt,
            8 => Ty::Struct,
            _ => Ty::OptInt,
        };
        let name = self.fresh(match ty {
            Ty::Int => "n",
            Ty::Float => "x",
            Ty::Bool => "b",
            Ty::ArrInt => "arr",
            Ty::Struct => "p",
            Ty::OptInt => "o",
        });
        let init = self.expr(ty, 2);
        self.declare(&name, ty, false);
        Stmt::Local { name, ty, init }
    }

    fn assign(&mut self) -> Option<Stmt> {
        let ty = *self.rng.pick(&[Ty::Int, Ty::Int, Ty::Float, Ty::Bool, Ty::OptInt]);
        let vars = self.mutable_vars_of(ty);
        if vars.is_empty() {
            return Some(self.local());
        }
        let name = self.rng.pick(&vars).name.clone();
        let expr = self.expr(ty, 3);
        Some(Stmt::Assign { name, expr })
    }

    fn set_field(&mut self) -> Option<Stmt> {
        let vars = self.vars_of(Ty::Struct);
        if vars.is_empty() {
            return Some(self.local());
        }
        let obj = self.rng.pick(&vars).name.clone();
        let (field, ty) = *self.rng.pick(&[("a", Ty::Int), ("b", Ty::Float), ("c", Ty::Bool)]);
        let expr = self.expr(ty, 2);
        Some(Stmt::SetField { obj, field, expr })
    }

    fn while_loop(&mut self) -> Stmt {
        let counter = self.fresh("i");
        // Bias towards enough iterations to make the loop hot.
        let bound = if self.rng.chance(0.8) {
            self.rng.range(6, 40)
        } else {
            self.rng.range(0, 5)
        };
        self.scopes.push(Vec::new());
        self.declare(&counter, Ty::Int, true);
        self.loop_depth += 1;
        let body = self.block(false);
        self.loop_depth -= 1;
        self.scopes.pop();
        Stmt::While {
            counter,
            bound,
            body,
        }
    }

    fn for_loop(&mut self) -> Stmt {
        let var = self.fresh("k");
        let lo = self.rng.range(-5, 5);
        let hi = if self.rng.chance(0.8) {
            lo + self.rng.range(5, 30)
        } else {
            lo + self.rng.range(-2, 4)
        };
        self.scopes.push(Vec::new());
        self.declare(&var, Ty::Int, true);
        self.loop_depth += 1;
        let body = self.block(false);
        self.loop_depth -= 1;
        self.scopes.pop();
        Stmt::For { var, lo, hi, body }
    }

    fn if_stmt(&mut self) -> Stmt {
        let count = self.rng.range(1, 3) as usize;
        let mut branches = Vec::new();
        for _ in 0..count {
            let cond = self.expr(Ty::Bool, 3);
            let body = self.block(false);
            branches.push((cond, body));
        }
        let otherwise = if self.rng.chance(0.5) {
            Some(self.block(false))
        } else {
            None
        };
        Stmt::If {
            branches,
            otherwise,
        }
    }

    fn break_or_continue(&mut self) -> Stmt {
        let cond = self.expr(Ty::Bool, 2);
        let stmt = if self.rng.chance(0.5) {
            Stmt::Break
        } else {
            Stmt::Continue
        };
        Stmt::If {
            branches: vec![(cond, vec![stmt])],
            otherwise: None,
        }
    }

    fn push(&mut self) -> Option<Stmt> {
        let vars = self.vars_of(Ty::ArrInt);
        if vars.is_empty() {
            return Some(self.local());
        }
        let arr = self.rng.pick(&vars).name.clone();
        let expr = self.expr(Ty::Int, 2);
        Some(Stmt::Push { arr, expr })
    }

    fn if_index(&mut self) -> Option<Stmt> {
        let vars = self.vars_of(Ty::ArrInt);
        if vars.is_empty() {
            return Some(self.local());
        }
        let arr = self.rng.pick(&vars).name.clone();
        let idx = self.expr(Ty::Int, 2);
        let var = self.fresh("v");
        self.scopes.push(Vec::new());
        self.declare(&var, Ty::Int, true);
        let body = self.block(false);
        self.scopes.pop();
        Some(Stmt::IfIndex {
            arr,
            idx,
            var,
            body,
        })
    }

    fn if_some(&mut self) -> Option<Stmt> {
        let vars = self.vars_of(Ty::OptInt);
        if vars.is_empty() {
            return Some(self.local());
        }
        let opt = self.rng.pick(&vars).name.clone();
        let var = self.fresh("u");
        self.scopes.push(Vec::new());
        self.declare(&var, Ty::Int, false);
        let body = self.block(false);
        self.scopes.pop();
        Some(Stmt::IfSome { opt, var, body })
    }

    // ── Expressions ──────────────────────────────────────────────────────

    fn expr(&mut self, ty: Ty, depth: u32) -> Expr {
        match ty {
            Ty::Int => self.int_expr(depth),
            Ty::Float => self.float_expr(depth),
            Ty::Bool => self.bool_expr(depth),
            Ty::ArrInt => {
                let n = self.rng.below(5) as usize;
                let items = (0..n).map(|_| self.int_expr(1)).collect();
                Expr::ArrLit(items)
            }
            Ty::Struct => Expr::StructLit(
                Box::new(self.int_expr(1)),
                Box::new(self.float_expr(1)),
                Box::new(self.bool_expr(1)),
            ),
            Ty::OptInt => {
                if self.rng.chance(0.7) {
                    Expr::Some(Box::new(self.int_expr(1)))
                } else {
                    Expr::None
                }
            }
        }
    }

    fn int_lit(&mut self) -> Expr {
        match self.rng.below(20) {
            0 => Expr::Int(i64::MAX),
            1 => Expr::Int(-i64::MAX),
            2 => Expr::Int(self.rng.range(-1_000_000, 1_000_000)),
            3 => Expr::Int(self.rng.range(1 << 40, 1 << 50)),
            _ => Expr::Int(self.rng.range(-20, 40)),
        }
    }

    fn var_or<F: FnOnce(&mut Self) -> Expr>(&mut self, ty: Ty, fallback: F) -> Expr {
        let vars = self.vars_of(ty);
        if vars.is_empty() {
            fallback(self)
        } else {
            Expr::Var(self.rng.pick(&vars).name.clone())
        }
    }

    fn call_returning(&mut self, ty: Ty, depth: u32) -> Option<Expr> {
        let candidates: Vec<FuncSig> = self.funcs.iter().filter(|f| f.ret == ty).cloned().collect();
        if candidates.is_empty() {
            return None;
        }
        let f = self.rng.pick(&candidates).clone();
        let args = f.params.iter().map(|t| self.expr(*t, depth.saturating_sub(1))).collect();
        Some(Expr::Call(f.name, args))
    }

    fn int_expr(&mut self, depth: u32) -> Expr {
        if depth == 0 {
            return if self.rng.chance(0.6) {
                self.var_or(Ty::Int, |g| g.int_lit())
            } else {
                self.int_lit()
            };
        }
        match self.rng.below(100) {
            0..=14 => self.int_lit(),
            15..=39 => self.var_or(Ty::Int, |g| g.int_lit()),
            40..=69 => {
                let op = *self.rng.pick(&[
                    BinOp::Add,
                    BinOp::Add,
                    BinOp::Sub,
                    BinOp::Mul,
                    BinOp::Div,
                    BinOp::Mod,
                ]);
                let l = self.int_expr(depth - 1);
                let r = if matches!(op, BinOp::Div | BinOp::Mod) && self.rng.chance(0.9) {
                    // Keep the divisor a non-zero literal most of the time;
                    // the rest exercises the division-by-zero error path.
                    let d = self.rng.range(-9, 9);
                    Expr::Int(if d == 0 { 7 } else { d })
                } else {
                    self.int_expr(depth - 1)
                };
                Expr::Bin(Box::new(l), op, Box::new(r))
            }
            70..=74 => {
                let inner = self.int_expr(depth - 1);
                negate(inner, Expr::Int(0))
            }
            75..=79 => {
                let a = self.int_expr(depth - 1);
                match self.rng.below(3) {
                    0 => Expr::Math("math.abs", vec![a]),
                    1 => {
                        let b = self.int_expr(depth - 1);
                        Expr::Math("math.min", vec![a, b])
                    }
                    _ => {
                        let b = self.int_expr(depth - 1);
                        Expr::Math("math.max", vec![a, b])
                    }
                }
            }
            80..=84 => {
                let vars = self.vars_of(Ty::ArrInt);
                if vars.is_empty() {
                    self.int_lit()
                } else {
                    Expr::ArrLen(self.rng.pick(&vars).name.clone())
                }
            }
            85..=89 => {
                let vars = self.vars_of(Ty::Struct);
                if vars.is_empty() {
                    self.int_lit()
                } else {
                    Expr::Field(self.rng.pick(&vars).name.clone(), "a")
                }
            }
            90..=93 => {
                let vars = self.vars_of(Ty::OptInt);
                if vars.is_empty() {
                    self.int_lit()
                } else {
                    let d = self.int_expr(depth - 1);
                    Expr::UnwrapOr(self.rng.pick(&vars).name.clone(), Box::new(d))
                }
            }
            _ => self
                .call_returning(Ty::Int, depth)
                .unwrap_or_else(|| self.var_or(Ty::Int, |g| g.int_lit())),
        }
    }

    fn float_lit(&mut self) -> Expr {
        Expr::Float(self.rng.pick(FLOAT_LITS))
    }

    fn float_expr(&mut self, depth: u32) -> Expr {
        if depth == 0 {
            return if self.rng.chance(0.6) {
                self.var_or(Ty::Float, |g| g.float_lit())
            } else {
                self.float_lit()
            };
        }
        match self.rng.below(100) {
            0..=14 => self.float_lit(),
            15..=39 => self.var_or(Ty::Float, |g| g.float_lit()),
            40..=74 => {
                let op = *self.rng.pick(&[
                    BinOp::Add,
                    BinOp::Add,
                    BinOp::Sub,
                    BinOp::Mul,
                    BinOp::Div,
                    BinOp::Mod,
                ]);
                // Mixed int/float arithmetic yields a float; one side stays
                // float so the result type is known.
                let (l, r) = match self.rng.below(4) {
                    0 => (self.int_expr(depth - 1), self.float_expr(depth - 1)),
                    1 => (self.float_expr(depth - 1), self.int_expr(depth - 1)),
                    _ => (self.float_expr(depth - 1), self.float_expr(depth - 1)),
                };
                Expr::Bin(Box::new(l), op, Box::new(r))
            }
            75..=79 => {
                let inner = self.float_expr(depth - 1);
                negate(inner, Expr::Float("0.0"))
            }
            80..=89 => {
                let vars = self.vars_of(Ty::Struct);
                if vars.is_empty() {
                    self.float_lit()
                } else {
                    Expr::Field(self.rng.pick(&vars).name.clone(), "b")
                }
            }
            _ => self
                .call_returning(Ty::Float, depth)
                .unwrap_or_else(|| self.var_or(Ty::Float, |g| g.float_lit())),
        }
    }

    fn bool_expr(&mut self, depth: u32) -> Expr {
        if depth == 0 {
            return if self.rng.chance(0.6) {
                self.var_or(Ty::Bool, |g| Expr::Bool(g.rng.chance(0.5)))
            } else {
                Expr::Bool(self.rng.chance(0.5))
            };
        }
        match self.rng.below(100) {
            0..=7 => Expr::Bool(self.rng.chance(0.5)),
            8..=19 => self.var_or(Ty::Bool, |g| Expr::Bool(g.rng.chance(0.5))),
            20..=59 => {
                // Comparisons need both sides the same type.
                let ty = *self.rng.pick(&[Ty::Int, Ty::Int, Ty::Float]);
                let op = *self.rng.pick(&[
                    BinOp::Lt,
                    BinOp::Le,
                    BinOp::Gt,
                    BinOp::Ge,
                    BinOp::Eq,
                    BinOp::Ne,
                ]);
                let l = self.expr(ty, depth - 1);
                let r = self.expr(ty, depth - 1);
                Expr::Bin(Box::new(l), op, Box::new(r))
            }
            60..=74 => {
                let op = *self.rng.pick(&[BinOp::And, BinOp::Or]);
                let l = self.bool_expr(depth - 1);
                let r = self.bool_expr(depth - 1);
                Expr::Bin(Box::new(l), op, Box::new(r))
            }
            75..=81 => Expr::Not(Box::new(self.bool_expr(depth - 1))),
            82..=86 => {
                let op = *self.rng.pick(&[BinOp::Eq, BinOp::Ne]);
                let l = self.bool_expr(depth - 1);
                let r = self.bool_expr(depth - 1);
                Expr::Bin(Box::new(l), op, Box::new(r))
            }
            87..=90 => {
                let vars = self.vars_of(Ty::Struct);
                if vars.is_empty() {
                    Expr::Bool(self.rng.chance(0.5))
                } else {
                    Expr::Field(self.rng.pick(&vars).name.clone(), "c")
                }
            }
            91..=94 => {
                let vars = self.vars_of(Ty::OptInt);
                if vars.is_empty() {
                    Expr::Bool(self.rng.chance(0.5))
                } else {
                    Expr::IsSome(self.rng.pick(&vars).name.clone())
                }
            }
            _ => self
                .call_returning(Ty::Bool, depth)
                .unwrap_or_else(|| Expr::Bool(self.rng.chance(0.5))),
        }
    }
}

/// Native stdlib calls type as `unknown`, and `unknown` propagates through
/// arithmetic; the checker rejects negating it. `0 - e` is accepted.
fn contains_math(e: &Expr) -> bool {
    match e {
        Expr::Math(..) => true,
        Expr::Bin(l, _, r) => contains_math(l) || contains_math(r),
        Expr::Neg(x) | Expr::Not(x) | Expr::Some(x) => contains_math(x),
        Expr::Call(_, args) | Expr::ArrLit(args) => args.iter().any(contains_math),
        Expr::UnwrapOr(_, d) => contains_math(d),
        Expr::StructLit(a, b, c) => contains_math(a) || contains_math(b) || contains_math(c),
        _ => false,
    }
}

fn negate(inner: Expr, zero: Expr) -> Expr {
    if contains_math(&inner) {
        Expr::Bin(Box::new(zero), BinOp::Sub, Box::new(inner))
    } else {
        Expr::Neg(Box::new(inner))
    }
}

// ── Shrinking support ────────────────────────────────────────────────────

/// Every statement-list in the program, addressed by a path. Used by the
/// shrinker to try removing one statement at a time.
pub fn stmt_paths(p: &Program) -> Vec<Vec<usize>> {
    // Path encoding: [func_index_or_MAX, stmt_index, child..., stmt_index]
    // where each "child" step is the index into the children list of the
    // statement at that point (see `children_mut`).
    let mut paths = Vec::new();
    for (fi, f) in p.funcs.iter().enumerate() {
        collect_paths(&f.body, vec![fi], &mut paths);
    }
    collect_paths(&p.run, vec![usize::MAX], &mut paths);
    paths
}

fn collect_paths(block: &[Stmt], prefix: Vec<usize>, paths: &mut Vec<Vec<usize>>) {
    for (i, s) in block.iter().enumerate() {
        let mut path = prefix.clone();
        path.push(i);
        paths.push(path.clone());
        for (ci, child) in children(s).into_iter().enumerate() {
            let mut cp = path.clone();
            cp.push(ci);
            collect_paths(child, cp, paths);
        }
    }
}

fn children(s: &Stmt) -> Vec<&Vec<Stmt>> {
    match s {
        Stmt::While { body, .. }
        | Stmt::For { body, .. }
        | Stmt::IfIndex { body, .. }
        | Stmt::IfSome { body, .. } => vec![body],
        Stmt::If {
            branches,
            otherwise,
        } => {
            let mut v: Vec<&Vec<Stmt>> = branches.iter().map(|(_, b)| b).collect();
            if let Some(o) = otherwise {
                v.push(o);
            }
            v
        }
        _ => Vec::new(),
    }
}

fn children_mut(s: &mut Stmt) -> Vec<&mut Vec<Stmt>> {
    match s {
        Stmt::While { body, .. }
        | Stmt::For { body, .. }
        | Stmt::IfIndex { body, .. }
        | Stmt::IfSome { body, .. } => vec![body],
        Stmt::If {
            branches,
            otherwise,
        } => {
            let mut v: Vec<&mut Vec<Stmt>> = branches.iter_mut().map(|(_, b)| b).collect();
            if let Some(o) = otherwise {
                v.push(o);
            }
            v
        }
        _ => Vec::new(),
    }
}

/// Remove the statement at `path`. Returns false if the path is stale.
pub fn remove_at(p: &mut Program, path: &[usize]) -> bool {
    let (head, rest) = match path.split_first() {
        Some(x) => x,
        None => return false,
    };
    let block: &mut Vec<Stmt> = if *head == usize::MAX {
        &mut p.run
    } else if let Some(f) = p.funcs.get_mut(*head) {
        &mut f.body
    } else {
        return false;
    };
    remove_in(block, rest)
}

fn remove_in(block: &mut Vec<Stmt>, path: &[usize]) -> bool {
    match path {
        [i] => {
            if *i < block.len() {
                block.remove(*i);
                true
            } else {
                false
            }
        }
        [i, ci, rest @ ..] => {
            let Some(stmt) = block.get_mut(*i) else {
                return false;
            };
            let mut kids = children_mut(stmt);
            match kids.get_mut(*ci) {
                Some(child) => remove_in(child, rest),
                None => false,
            }
        }
        [] => false,
    }
}

/// Halve every loop bound above `min`; returns whether anything changed.
pub fn shrink_loops(p: &mut Program) -> bool {
    let mut changed = false;
    for f in &mut p.funcs {
        changed |= shrink_loops_in(&mut f.body);
    }
    changed |= shrink_loops_in(&mut p.run);
    changed
}

fn shrink_loops_in(block: &mut [Stmt]) -> bool {
    let mut changed = false;
    for s in block.iter_mut() {
        match s {
            Stmt::While { bound, body, .. } => {
                if *bound > 6 {
                    *bound = (*bound / 2).max(6);
                    changed = true;
                }
                changed |= shrink_loops_in(body);
            }
            Stmt::For { lo, hi, body, .. } => {
                if *hi - *lo > 6 {
                    *hi = *lo + ((*hi - *lo) / 2).max(6);
                    changed = true;
                }
                changed |= shrink_loops_in(body);
            }
            _ => {
                for child in children_mut(s) {
                    changed |= shrink_loops_in(child);
                }
            }
        }
    }
    changed
}
