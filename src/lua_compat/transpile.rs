use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use full_moon::{
    ast::{
        Block, Expression, Field, FunctionArgs, FunctionBody, FunctionCall, FunctionDeclaration,
        FunctionName, Index, LastStmt, Parameter, Prefix, Return, Suffix, UnOp, Value,
    },
    parse,
};

fn is_keyword(name: &str) -> bool {
    matches!(
        name.trim(),
        "local"
            | "mut"
            | "function"
            | "return"
            | "if"
            | "then"
            | "else"
            | "elseif"
            | "end"
            | "while"
            | "do"
            | "for"
            | "in"
            | "break"
            | "continue"
            | "struct"
            | "enum"
            | "trait"
            | "impl"
            | "match"
            | "case"
            | "as"
            | "is"
            | "true"
            | "false"
            | "and"
            | "or"
            | "not"
            | "extern"
            | "unsafe"
            | "pub"
            | "use"
            | "module"
            | "const"
            | "static"
            | "type"
    )
}

fn sanitize_identifier(name: &str) -> String {
    let trimmed = name.trim();
    if is_keyword(trimmed) {
        format!("{trimmed}_")
    } else {
        trimmed.to_string()
    }
}

fn var_segments(var: &full_moon::ast::Var) -> Option<Vec<String>> {
    match var {
        full_moon::ast::Var::Name(tok) => Some(vec![sanitize_identifier(&tok.to_string())]),
        full_moon::ast::Var::Expression(expr) => {
            let mut segments = Vec::new();
            match expr.prefix() {
                Prefix::Name(tok) => segments.push(sanitize_identifier(&tok.to_string())),
                _ => return None,
            }
            for suffix in expr.suffixes() {
                if let Suffix::Index(Index::Dot { name, .. }) = suffix {
                    segments.push(sanitize_identifier(&name.to_string()));
                } else {
                    return None;
                }
            }
            Some(segments)
        }
        _ => None,
    }
}

fn expr_segments(expr: &Expression) -> Option<Vec<String>> {
    if let Expression::Value { value, .. } = expr {
        if let Value::Var(var) = &**value {
            return var_segments(var);
        }
    }
    None
}

/// Transpile a Lua 5.1 module into Lust source code.
/// This attempts to mirror the Lua module as closely as possible while mapping Lua `require`
/// to `lua.require(...)` calls and exporting members discovered in a trailing `return { ... }`.
pub fn transpile_lua_stub(source: &str, module_name: &str) -> Result<String, String> {
    let ast = parse(source).map_err(|e| format!("failed to parse Lua: {e}"))?;
    let block = ast.nodes();
    let mut analyzer = Analyzer::new(module_name);
    analyzer.analyze_block(block);
    let mut emitter = Emitter::new(module_name, analyzer);
    emitter.emit_block(block);
    Ok(emitter.finish())
}

#[derive(Default)]
struct Analyzer {
    module_name: String,
    exports: Vec<String>,
    module_decl: Option<ModuleDecl>,
    module_tables: Vec<String>,
}

#[derive(Debug, Clone)]
struct ModuleDecl {
    name: String,
    seeall: bool,
}

impl Analyzer {
    fn new(module_name: &str) -> Self {
        Self {
            module_name: module_name.to_string(),
            module_tables: vec!["_M".to_string()],
            ..Self::default()
        }
    }

    fn analyze_block(&mut self, block: &Block) {
        for stmt in block.stmts() {
            self.analyze_stmt(stmt);
        }
        if let Some(last) = block.last_stmt() {
            self.analyze_last(last);
        }
    }

    fn analyze_stmt(&mut self, stmt: &full_moon::ast::Stmt) {
        match stmt {
            full_moon::ast::Stmt::FunctionDeclaration(func) => {
                if let Some(parts) = self.function_decl_parts(func.name()) {
                    if self.is_module_export(&parts) {
                        if let Some(name) = parts.last() {
                            self.maybe_export_name(name);
                        }
                    }
                }
            }
            full_moon::ast::Stmt::LocalFunction(func) => {
                let name = func.name().to_string();
                self.maybe_export_name(&name);
            }
            full_moon::ast::Stmt::Assignment(assign) => {
                for (idx, var) in assign.variables().iter().enumerate() {
                    self.maybe_export_var(var);
                    if let Some(name) = var_segments(var).and_then(|s| s.first().cloned()) {
                        let expr = assign.expressions().iter().nth(idx);
                        self.maybe_register_module_alias(&name, expr);
                    }
                }
                for expr in assign.expressions().iter() {
                    self.analyze_expr(expr);
                }
            }
            full_moon::ast::Stmt::LocalAssignment(assign) => {
                for (idx, name) in assign.names().iter().enumerate() {
                    let ident = name.to_string().trim().to_string();
                    let expr = assign.expressions().iter().nth(idx);
                    self.maybe_register_module_alias(&ident, expr);
                }
                for expr in assign.expressions().iter() {
                    self.analyze_expr(expr);
                }
            }
            full_moon::ast::Stmt::FunctionCall(call) => {
                self.analyze_function_call(call);
            }
            full_moon::ast::Stmt::If(if_stmt) => {
                self.analyze_block(if_stmt.block());
                if let Some(else_if) = if_stmt.else_if() {
                    for clause in else_if {
                        self.analyze_block(clause.block());
                    }
                }
                if let Some(block) = if_stmt.else_block() {
                    self.analyze_block(block);
                }
            }
            full_moon::ast::Stmt::While(while_stmt) => {
                self.analyze_block(while_stmt.block());
            }
            full_moon::ast::Stmt::Repeat(repeat_stmt) => {
                self.analyze_block(repeat_stmt.block());
            }
            full_moon::ast::Stmt::NumericFor(for_stmt) => {
                self.analyze_block(for_stmt.block());
            }
            full_moon::ast::Stmt::GenericFor(for_stmt) => {
                self.analyze_block(for_stmt.block());
            }
            full_moon::ast::Stmt::Do(do_block) => {
                self.analyze_block(do_block.block());
            }
            _ => {}
        }
    }

    fn analyze_last(&mut self, last: &LastStmt) {
        if let LastStmt::Return(ret) = last {
            for expr in ret.returns().iter() {
                if let Some(exports) = self.extract_exports(expr) {
                    for name in exports {
                        self.exports.push(name);
                    }
                } else {
                    if let Some(parts) = expr_segments(expr) {
                        if self.is_module_export(&parts) {
                            if let Some(name) = parts.last() {
                                self.maybe_export_name(name);
                            }
                        }
                    }
                    self.analyze_expr(expr);
                }
            }
        }
    }

    fn extract_exports(&self, expr: &Expression) -> Option<Vec<String>> {
        if let Expression::Value { value, .. } = expr {
            if let Value::TableConstructor(table) = &**value {
                let mut names = Vec::new();
                for field in table.fields() {
                    if let Field::NameKey { key, .. } = field {
                        let name = sanitize_identifier(&key.token().to_string());
                        names.push(name);
                    }
                }
                if !names.is_empty() {
                    return Some(names);
                }
            }
        }
        None
    }

    fn analyze_expr(&mut self, expr: &Expression) {
        match expr {
            Expression::Value { value, .. } => match &**value {
                Value::FunctionCall(call) => self.analyze_function_call(call),
                Value::Function((_token, body)) => self.analyze_block(body.block()),
                Value::TableConstructor(table) => {
                    for field in table.fields() {
                        match field {
                            Field::ExpressionKey { key, value, .. } => {
                                self.analyze_expr(key);
                                self.analyze_expr(value);
                            }
                            Field::NameKey { value, .. } => self.analyze_expr(value),
                            Field::NoKey(value) => self.analyze_expr(value),
                            _ => {}
                        }
                    }
                }
                Value::Var(var) => {
                    if let full_moon::ast::Var::Expression(var_expr) = var {
                        for suffix in var_expr.suffixes() {
                            if let Suffix::Index(Index::Brackets { expression, .. }) = suffix {
                                self.analyze_expr(expression);
                            }
                        }
                    }
                }
                _ => {}
            },
            Expression::BinaryOperator { lhs, rhs, .. } => {
                self.analyze_expr(lhs);
                self.analyze_expr(rhs);
            }
            Expression::UnaryOperator { expression, .. } => {
                self.analyze_expr(expression);
            }
            Expression::Parentheses { expression, .. } => self.analyze_expr(expression),
            _ => {}
        }
    }

    fn analyze_function_call(&mut self, call: &FunctionCall) {
        // Detect legacy module("name", package.seeall) pattern.
        if let Prefix::Name(tok) = call.prefix() {
            if tok.to_string() == "module" {
                if let Some(Suffix::Call(full_moon::ast::Call::AnonymousCall(args))) =
                    call.suffixes().find(|s| matches!(s, Suffix::Call(_)))
                {
                    if let FunctionArgs::Parentheses { arguments, .. } = args {
                        let mut iter = arguments.iter();
                        if let Some(Expression::Value { value, .. }) = iter.next() {
                            if let Value::String(tok) = &**value {
                                let mod_name = tok.to_string().trim_matches('"').to_string();
                                let mut seeall = false;
                                if let Some(Expression::Value { value, .. }) = iter.next() {
                                    if let Value::Var(var) = &**value {
                                        if let full_moon::ast::Var::Expression(expr) = var {
                                            if let Prefix::Name(pkg) = expr.prefix() {
                                                if pkg.to_string() == "package" {
                                                    if let Some(Suffix::Index(Index::Dot {
                                                        name,
                                                        ..
                                                    })) = expr.suffixes().next()
                                                    {
                                                        if name.to_string() == "seeall" {
                                                            seeall = true;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                self.module_decl = Some(ModuleDecl {
                                    name: mod_name,
                                    seeall,
                                });
                            }
                        }
                    }
                }
            }
        }
        let head = match call.prefix() {
            Prefix::Name(tok) => Some(tok.to_string()),
            _ => None,
        };
        let _ = head;
        for suffix in call.suffixes() {
            if let Suffix::Call(full_moon::ast::Call::AnonymousCall(args)) = suffix {
                if let FunctionArgs::Parentheses { arguments, .. } = args {
                    for expr in arguments.iter() {
                        self.analyze_expr(expr);
                    }
                }
            }
        }
    }

    fn maybe_export_name(&mut self, name: &str) {
        let name = sanitize_identifier(name);
        if self.exports.contains(&name) {
            return;
        }
        self.exports.push(name);
    }

    fn function_decl_parts(&self, name: &FunctionName) -> Option<Vec<String>> {
        let mut parts: Vec<String> = name
            .names()
            .iter()
            .map(|t| sanitize_identifier(&t.to_string()))
            .collect();
        if let Some(method) = name.method_name() {
            parts.push(sanitize_identifier(&method.to_string()));
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts)
        }
    }

    fn module_parts(&self) -> Vec<String> {
        self.module_name
            .split('.')
            .map(|s| sanitize_identifier(s))
            .collect::<Vec<_>>()
    }

    fn module_path_prefix(&self, parts: &[String]) -> bool {
        let module_parts = self.module_parts();
        if parts.len() < module_parts.len() {
            return false;
        }
        parts
            .iter()
            .take(module_parts.len())
            .zip(module_parts.iter())
            .all(|(a, b)| a == b)
    }

    fn is_module_export(&self, parts: &[String]) -> bool {
        if parts.len() > 1 && self.module_path_prefix(parts) {
            return true;
        }
        if let Some(first) = parts.first() {
            if self.module_tables.contains(first) && parts.len() > 1 {
                return true;
            }
        }
        false
    }

    fn maybe_export_var(&mut self, var: &full_moon::ast::Var) {
        if let Some(segments) = var_segments(var) {
            if self.is_module_export(&segments) {
                if let Some(name) = segments.last() {
                    self.maybe_export_name(name);
                }
            }
        }
    }

    fn maybe_register_module_alias(&mut self, name: &str, expr: Option<&Expression>) {
        if let Some(expr) = expr {
            if let Some(path) = expr_segments(expr) {
                if path == self.module_parts() {
                    let ident = sanitize_identifier(name);
                    if !self.module_tables.contains(&ident) {
                        self.module_tables.push(ident);
                    }
                }
            } else if matches!(expr, Expression::Value { value, .. } if matches!(&**value, Value::TableConstructor(_)))
            {
                let ident = sanitize_identifier(name);
                if !self.module_tables.contains(&ident) {
                    self.module_tables.push(ident);
                }
            }
        }
    }
}

struct Emitter {
    module: String,
    analyzer: Analyzer,
    lines: Vec<String>,
    indent: usize,
    exported: Vec<String>,
    module_tables: BTreeSet<String>,
    module_parts: Vec<String>,
    export_bindings: BTreeMap<String, String>,
    exported_functions: BTreeSet<String>,
    export_set: BTreeSet<String>,
    export_wrappers: BTreeMap<String, WrapperSig>,
    local_scopes: Vec<BTreeSet<String>>,
}

const VARARGS_NAME: &str = "__varargs";

#[derive(Clone)]
struct WrapperSig {
    params: Vec<String>,
    args: Vec<String>,
    return_count: usize,
}

impl Emitter {
    fn new(module: &str, analyzer: Analyzer) -> Self {
        let mut module_tables = BTreeSet::new();
        for table in &analyzer.module_tables {
            module_tables.insert(table.clone());
        }
        let mut export_set = BTreeSet::new();
        for name in &analyzer.exports {
            export_set.insert(name.clone());
        }
        Self {
            module: module.to_string(),
            analyzer,
            lines: Vec::new(),
            indent: 0,
            exported: Vec::new(),
            module_tables,
            module_parts: module
                .split('.')
                .map(|s| sanitize_identifier(s))
                .collect::<Vec<_>>(),
            export_bindings: BTreeMap::new(),
            exported_functions: BTreeSet::new(),
            export_set,
            export_wrappers: BTreeMap::new(),
            local_scopes: Vec::new(),
        }
    }

    fn declare_local(&mut self, name: &str) {
        if let Some(scope) = self.local_scopes.last_mut() {
            scope.insert(name.to_string());
        }
    }

    fn is_local(&self, name: &str) -> bool {
        self.local_scopes
            .iter()
            .rev()
            .any(|scope| scope.contains(name))
    }

    fn finish(mut self) -> String {
        if self.module_tables.is_empty() {
            for table in &self.analyzer.module_tables {
                self.module_tables.insert(table.clone());
            }
        }
        for export in &self.analyzer.exports {
            self.export_set.insert(export.clone());
        }
        self.lines.insert(
            0,
            format!(
                "-- Auto-transpiled from Lua module '{module}'",
                module = self.module
            ),
        );
        let insert_at = 1;
        let prelude = vec![
            "local __lua_ret: Array<LuaValue> = []".to_string(),
            "local __lua_pack = function(items: Array<LuaValue>): Array<LuaValue>".to_string(),
            "    return items".to_string(),
            "end".to_string(),
            "local __lua_force_array = function(val: unknown): Array<LuaValue>".to_string(),
            "    if val is Array<LuaValue> then".to_string(),
            "        return val".to_string(),
            "    end".to_string(),
            "    return [lua.to_value(val)]".to_string(),
            "end".to_string(),
            "local __lua_table_from_varargs = function(items: Array<LuaValue>): LuaTable".to_string(),
            "    local t = lua.table()".to_string(),
            "    local idx: int = 0".to_string(),
            "    while idx < items:len() do".to_string(),
            "        local v = items:get(idx)".to_string(),
            "        if v:is_some() then".to_string(),
            "            t[idx + 1] = lua.unwrap(v:unwrap())".to_string(),
            "        end".to_string(),
            "        idx = idx + 1".to_string(),
            "    end".to_string(),
            "    return t".to_string(),
            "end".to_string(),
            "local __lua_truthy = function(val: unknown): bool".to_string(),
            "    return lua.is_truthy(val)".to_string(),
            "end".to_string(),
            "local __lua_first = function(val: unknown): unknown".to_string(),
            "    if val is Array<LuaValue> then".to_string(),
            "        local first = val:get(0)".to_string(),
            "        if first:is_some() then".to_string(),
            "            return first:unwrap()".to_string(),
            "        end".to_string(),
            "        return lua.nil".to_string(),
            "    end".to_string(),
            "    if val is LuaValue then".to_string(),
            "        return val".to_string(),
            "    end".to_string(),
            "    return val".to_string(),
            "end".to_string(),
            "local __lua_nth = function(val: unknown, idx: int): unknown".to_string(),
            "    if val is Array<LuaValue> then".to_string(),
            "        local item = val:get(idx)".to_string(),
            "        if item:is_some() then".to_string(),
            "            return item:unwrap()".to_string(),
            "        end".to_string(),
            "        return lua.nil".to_string(),
            "    end".to_string(),
            "    if idx == 0 then".to_string(),
            "        if val is LuaValue then".to_string(),
            "            return val".to_string(),
            "        end".to_string(),
            "        return val".to_string(),
            "    end".to_string(),
            "    return lua.nil".to_string(),
            "end".to_string(),
            "local __lua_join = function(prefix: Array<LuaValue>, tail: unknown): Array<LuaValue>".to_string(),
            "    local arr = __lua_force_array(tail)".to_string(),
            "    local idx: int = 0".to_string(),
            "    while idx < arr:len() do".to_string(),
            "        local v = arr:get(idx)".to_string(),
            "        if v:is_some() then".to_string(),
            "            prefix:push(v:unwrap())".to_string(),
            "        end".to_string(),
            "        idx = idx + 1".to_string(),
            "    end".to_string(),
            "    return prefix".to_string(),
            "end".to_string(),
            "local __lua_and = function(a: unknown, b: unknown): unknown".to_string(),
            "    if __lua_truthy(a) then".to_string(),
            "        return b".to_string(),
            "    end".to_string(),
            "    return a".to_string(),
            "end".to_string(),
            "local __lua_or = function(a: unknown, b: unknown): unknown".to_string(),
            "    if __lua_truthy(a) then".to_string(),
            "        return a".to_string(),
            "    end".to_string(),
            "    return b".to_string(),
            "end".to_string(),
            "local assert = function(cond: LuaValue, msg: LuaValue): LuaValue".to_string(),
            "    if __lua_truthy(cond) then".to_string(),
            "        return cond".to_string(),
            "    end".to_string(),
            "    error(msg or \"assertion failed\")".to_string(),
            "end".to_string(),
            "local _G: Map<unknown, unknown> = {}".to_string(),
            "_G[\"assert\"] = lua.to_value(assert)".to_string(),
            "_G[\"error\"] = lua.to_value(error)".to_string(),
            "_G[\"ipairs\"] = lua.to_value(ipairs)".to_string(),
            "_G[\"pairs\"] = lua.to_value(pairs)".to_string(),
            "_G[\"setmetatable\"] = lua.to_value(setmetatable)".to_string(),
            "_G[\"select\"] = lua.to_value(select)".to_string(),
            "_G[\"unpack\"] = lua.to_value(unpack)".to_string(),
            "_G[\"tostring\"] = lua.to_value(tostring)".to_string(),
            "_G[\"tonumber\"] = lua.to_value(tonumber)".to_string(),
            "_G[\"type\"] = lua.to_value(type)".to_string(),
            "local type_ = type".to_string(),
            "_G[\"string\"] = lua.to_value(string)".to_string(),
            "local module_ = _G[\"module\"]".to_string(),
        ];
        self.lines
            .splice(insert_at..insert_at, prelude.into_iter());
        if let Some(decl) = &self.analyzer.module_decl {
            let alias = decl.name.replace('.', "_");
            self.lines
                .insert(insert_at, "local __module = {}".to_string());
            if decl.seeall {
                self.lines.insert(
                    insert_at,
                    "-- package.seeall requested; globals fallback not implemented".to_string(),
                );
            }
            self.lines.push(format!("{} = __module", alias));
            self.lines.push("return __module".to_string());
        }
        if !self.export_bindings.is_empty() {
            let mut export_lines = Vec::new();
            for (name, path) in self.export_bindings.iter() {
                let has_wrapper = self.export_wrappers.contains_key(name);
                if let Some(sig) = self.export_wrappers.get(name) {
                    let params = sig.params.join(", ");
                    let args = sig.args.join(", ");
                    export_lines.push(format!("pub function {name}({params}): Array<LuaValue>"));
                    export_lines.push(format!("    return {path}({args})"));
                    export_lines.push("end".to_string());
                }
                if !has_wrapper {
                    export_lines.push(format!("pub const {name}: LuaValue = {path}"));
                }
            }
            self.export_bindings.clear();
            self.export_wrappers.clear();
            if let Some(pos) = self
                .lines
                .iter()
                .rposition(|line| line.trim_start().starts_with("return "))
            {
                self.lines.splice(pos..pos, export_lines);
            } else {
                self.lines.extend(export_lines);
            }
        }
        self.lines.join("\n") + "\n"
    }

    fn emit_block(&mut self, block: &Block) {
        self.local_scopes.push(BTreeSet::new());
        for stmt in block.stmts() {
            self.emit_stmt(stmt);
        }
        if let Some(last) = block.last_stmt() {
            self.emit_last(last);
        }
        self.local_scopes.pop();
    }

    fn emit_stmt(&mut self, stmt: &full_moon::ast::Stmt) {
        match stmt {
            full_moon::ast::Stmt::FunctionDeclaration(func) => {
                let name = self
                    .function_name_parts(func.name())
                    .map(|(parts, _)| parts.join("."));
                let is_exported = name.as_ref().map(|n| self.is_exported(n)).unwrap_or(false);
                self.emit_function_decl(func, is_exported);
            }
            full_moon::ast::Stmt::LocalFunction(func) => {
                let name = self.sanitize_identifier(func.name().to_string());
                let is_exported = self.analyzer.exports.contains(&name);
                self.emit_local_function(func, is_exported);
            }
            full_moon::ast::Stmt::LocalAssignment(assign) => {
                let names: Vec<String> = assign
                    .names()
                    .iter()
                    .map(|t| self.sanitize_identifier(t.to_string()))
                    .collect();
                for name in &names {
                    self.declare_local(name);
                }
                if names.len() > 1 && assign.expressions().len() == 1 {
                    let expr = assign.expressions().first().unwrap().value();
                    if self.is_multi_return_expr(expr) {
                        let temp = "__lua_ret";
                        self.push_line(format!("local {}", names.join(", ")));
                        let raw = self.emit_expr_raw(expr);
                        self.push_line(format!("local {temp} = {raw}"));
                        for (idx, name) in names.iter().enumerate() {
                            self.push_line(format!("{name} = __lua_nth({temp}, {idx})"));
                        }
                        return;
                    }
                }
                let exprs: Vec<String> = assign
                    .expressions()
                    .iter()
                    .map(|e| self.emit_expr(e))
                    .collect();
                if exprs.is_empty() {
                    self.push_line(format!("local {}", names.join(", ")));
                } else {
                    self.push_line(format!("local {} = {}", names.join(", "), exprs.join(", ")));
                }
            }
            full_moon::ast::Stmt::Assignment(assign) => {
                let vars: Vec<String> = assign
                    .variables()
                    .iter()
                    .map(|v| {
                        if self.analyzer.module_decl.is_some() {
                            if let full_moon::ast::Var::Name(tok) = v {
                                return self.format_field(
                                    "__module".to_string(),
                                    &tok.to_string().trim().to_string(),
                                );
                            }
                        }
                        self.emit_var(v)
                    })
                    .collect();
                if vars.len() > 1 && assign.expressions().len() == 1 {
                    let expr = assign.expressions().first().unwrap().value();
                    if self.is_multi_return_expr(expr) {
                        let temp = "__lua_ret";
                        let raw = self.emit_expr_raw(expr);
                        self.push_line(format!("{temp} = {raw}"));
                        for (idx, var) in vars.iter().enumerate() {
                            self.push_line(format!("{var} = __lua_nth({temp}, {idx})"));
                        }
                        return;
                    }
                }
                let expr_sigs: Vec<Option<WrapperSig>> = assign
                    .expressions()
                    .iter()
                    .map(|e| self.extract_function_sig(e))
                    .collect();
                let exprs: Vec<String> = assign
                    .expressions()
                    .iter()
                    .map(|e| {
                        if self.analyzer.module_decl.is_some() {
                            format!("lua.to_value({})", self.emit_expr(e))
                        } else {
                            self.emit_expr(e)
                        }
                    })
                    .collect();
                self.push_line(format!("{} = {}", vars.join(", "), exprs.join(", ")));
                for (idx, var) in assign.variables().iter().enumerate() {
                    if let Some(path) = var_segments(var) {
                        if self.is_module_path(&path) {
                            if let Some(name) = path.last() {
                                let name = sanitize_identifier(name);
                                if self.is_exported(&name)
                                    && !self.exported_functions.contains(&name)
                                {
                                    let path_expr = self.build_path(&path);
                                    self
                                        .export_bindings
                                        .entry(name.clone())
                                        .or_insert(path_expr);
                                    if let Some(sig) = expr_sigs.get(idx).and_then(|s| s.clone()) {
                                        self.export_wrappers.entry(name).or_insert(sig);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            full_moon::ast::Stmt::FunctionCall(call) => {
                if self.analyzer.module_decl.is_some()
                    && matches!(call.prefix(), Prefix::Name(tok) if tok.to_string() == "module")
                {
                    // skip emitting module(...) in transpiled output
                } else {
                    let call_str = self.emit_function_call(call);
                    self.push_line(call_str);
                }
            }
            full_moon::ast::Stmt::If(if_stmt) => {
                let cond = self.emit_condition(if_stmt.condition());
                self.push_line(format!("if {} then", cond));
                self.indent += 1;
                self.emit_block(if_stmt.block());
                self.indent -= 1;
                if let Some(else_if) = if_stmt.else_if() {
                    for clause in else_if {
                        let cond = self.emit_condition(clause.condition());
                        self.push_line(format!("elseif {} then", cond));
                        self.indent += 1;
                        self.emit_block(clause.block());
                        self.indent -= 1;
                    }
                }
                if let Some(block) = if_stmt.else_block() {
                    self.push_line("else".to_string());
                    self.indent += 1;
                    self.emit_block(block);
                    self.indent -= 1;
                }
                self.push_line("end".to_string());
            }
            full_moon::ast::Stmt::While(while_stmt) => {
                let cond = self.emit_condition(while_stmt.condition());
                self.push_line(format!("while {} do", cond));
                self.indent += 1;
                self.emit_block(while_stmt.block());
                self.indent -= 1;
                self.push_line("end".to_string());
            }
            full_moon::ast::Stmt::Repeat(repeat_stmt) => {
                self.push_line("while true do".to_string());
                self.indent += 1;
                self.emit_block(repeat_stmt.block());
                let cond = self.emit_condition(repeat_stmt.until());
                self.push_line(format!("if {} then", cond));
                self.indent += 1;
                self.push_line("break".to_string());
                self.indent -= 1;
                self.push_line("end".to_string());
                self.indent -= 1;
                self.push_line("end".to_string());
            }
            full_moon::ast::Stmt::NumericFor(for_stmt) => {
                let var = self.sanitize_identifier(for_stmt.index_variable().to_string());
                let start = self.emit_expr(for_stmt.start());
                let end = self.emit_expr(for_stmt.end());
                let step = for_stmt
                    .step()
                    .map(|e| self.emit_expr(e))
                    .unwrap_or_else(|| "1".to_string());
                self.push_line(format!("for {var} = {start}, {end}, {step} do"));
                self.indent += 1;
                self.emit_block(for_stmt.block());
                self.indent -= 1;
                self.push_line("end".to_string());
            }
            full_moon::ast::Stmt::GenericFor(for_stmt) => {
                let names: Vec<String> = for_stmt
                    .names()
                    .iter()
                    .map(|t| self.sanitize_identifier(t.to_string()))
                    .collect();
                let exprs: Vec<String> = for_stmt
                    .expressions()
                    .iter()
                    .map(|e| self.emit_expr(e))
                    .collect();
                self.push_line(format!(
                    "for {} in {} do",
                    names.join(", "),
                    exprs.join(", ")
                ));
                self.indent += 1;
                self.emit_block(for_stmt.block());
                self.indent -= 1;
                self.push_line("end".to_string());
            }
            full_moon::ast::Stmt::Do(do_block) => {
                self.push_line("do".to_string());
                self.indent += 1;
                self.emit_block(do_block.block());
                self.indent -= 1;
                self.push_line("end".to_string());
            }
            _ => {}
        }
    }

    fn emit_last(&mut self, last: &LastStmt) {
        if let LastStmt::Return(ret) = last {
            if let Some(exports) = self.extract_exports(ret) {
                for (name, expr) in exports {
                    if self.analyzer.exports.contains(&name) {
                        self.push_line(format!(
                            "pub const {name}: LuaValue = lua.to_value({expr})"
                        ));
                        self.exported.push(name.clone());
                    }
                }
            } else {
                let returns: Vec<&Expression> = ret.returns().iter().collect();
                if returns.is_empty() {
                    self.push_line("return __lua_pack([])".to_string());
                    return;
                }
                if returns.len() == 1 {
                    let expr = returns[0];
                    if self.is_multi_return_expr(expr) {
                        let raw = self.emit_expr_raw(expr);
                        self.push_line(format!("return __lua_force_array({raw})"));
                    } else {
                        let val = self.emit_expr(expr);
                        self.push_line(format!("return __lua_pack([lua.to_value({val})])"));
                    }
                    return;
                }
                let mut prefix_vals = Vec::new();
                for expr in returns.iter().take(returns.len() - 1) {
                    let v = self.emit_expr(expr);
                    prefix_vals.push(format!("lua.to_value({v})"));
                }
                let last_expr = returns.last().unwrap();
                let raw_last = if self.is_multi_return_expr(last_expr) {
                    self.emit_expr_raw(last_expr)
                } else {
                    format!("[lua.to_value({})]", self.emit_expr(last_expr))
                };
                let tmp = "__lua_ret";
                self.push_line(format!("local {tmp}: Array<LuaValue> = [{}]", prefix_vals.join(", ")));
                self.push_line(format!("return __lua_join({tmp}, {raw_last})"));
            }
        }
    }

    fn extract_exports(&mut self, ret: &Return) -> Option<Vec<(String, String)>> {
        let mut out = Vec::new();
        for expr in ret.returns().iter() {
            if let Expression::Value { value, .. } = expr {
                if let Value::TableConstructor(table) = &**value {
                    for field in table.fields() {
                        if let Field::NameKey { key, value, .. } = field {
                            let name = sanitize_identifier(&key.token().to_string());
                            let expr = self.emit_expr(value);
                            out.push((name, expr));
                        }
                    }
                }
            }
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }

    fn emit_function_decl(&mut self, func: &FunctionDeclaration, exported: bool) {
        let (parts, method) = self
            .function_name_parts(func.name())
            .unwrap_or_else(|| (Vec::new(), None));
        let local_binding = method.is_none()
            && parts.len() == 1
            && parts
                .first()
                .map(|name| self.is_local(name))
                .unwrap_or(false);
        let (params_with_types, param_names) = self.emit_function_params(func.body());
        let mut params = params_with_types;
        let mut _call_args = param_names;
        let mut target = self.build_path(&parts);
        let mut needs_assignment = parts.iter().any(|p| self.is_keyword(p));
        if let Some(ref method_name) = method {
            params.insert(0, "self: LuaValue".to_string());
            _call_args.insert(0, "self".to_string());
            target = self.format_field(target, &method_name);
            needs_assignment = true;
        }
        if local_binding {
            needs_assignment = true;
        }
        let belongs_to_module = self.is_module_path(&parts);
        let export_name = method
            .clone()
            .or_else(|| parts.last().cloned())
            .unwrap_or_default();
        let should_export_alias =
            belongs_to_module && self.is_exported(&sanitize_identifier(&export_name));

        if should_export_alias {
            let alias_name = self.export_alias_name(&parts, &export_name);
            let params_str = params.join(", ");
            self.push_line(format!(
                "pub function {alias_name}({params_str}): Array<LuaValue>"
            ));
            self.indent += 1;
            self.emit_block(func.body().block());
            self.indent -= 1;
            self.push_line("end".to_string());
            self.push_line(format!("{target} = {alias_name}"));
            self.exported_functions.insert(alias_name);
            return;
        }
        if !local_binding && self.analyzer.module_decl.is_some() && !target.starts_with("__module.")
        {
            if parts.len() == 1 && method.is_none() {
                target = self.format_field("__module".to_string(), &parts[0]);
            }
        }
        let params = params.join(", ");
        let body_block = func.body().block();
        if needs_assignment {
            self.push_line(format!("{target} = function({params}): Array<LuaValue>"));
        } else {
            let target = if self.analyzer.module_decl.is_some() {
                self.qualify_module_target(&target)
            } else {
                target
            };
            let prefix = if exported && self.analyzer.module_decl.is_none() {
                "pub function"
            } else {
                "function"
            };
            self.push_line(format!("{prefix} {target}({params}): Array<LuaValue>"));
        }
        self.indent += 1;
        self.emit_block(body_block);
        self.indent -= 1;
        self.push_line("end".to_string());
    }

    fn emit_local_function(&mut self, func: &full_moon::ast::LocalFunction, exported: bool) {
        let name = self.sanitize_identifier(func.name().to_string());
        let (params, _) = self.emit_function_params(func.body());
        let params = params.join(", ");
        let body_block = func.body().block();
        let _ = exported;
        self.declare_local(&name);
        self.push_line(format!("local {name} = function({params}): Array<LuaValue>"));
        self.indent += 1;
        self.emit_block(body_block);
        self.indent -= 1;
        self.push_line("end".to_string());
    }

    fn emit_function_params(&self, body: &FunctionBody) -> (Vec<String>, Vec<String>) {
        let mut typed = Vec::new();
        let mut names = Vec::new();
        for p in body.parameters().iter() {
            match p {
                Parameter::Name(name) => {
                    let ident = self.sanitize_identifier(name.to_string());
                    names.push(ident.clone());
                    typed.push(format!("{ident}: LuaValue"));
                }
                Parameter::Ellipse(_) => {
                    names.push(VARARGS_NAME.to_string());
                    typed.push(format!("{VARARGS_NAME}: Array<LuaValue>"));
                }
                &_ => {}
            }
        }
        (typed, names)
    }

    fn emit_expr(&mut self, expr: &Expression) -> String {
        self.emit_expr_mode(expr, true)
    }

    fn emit_expr_raw(&mut self, expr: &Expression) -> String {
        self.emit_expr_mode(expr, false)
    }

    fn emit_expr_mode(&mut self, expr: &Expression, wrap_calls: bool) -> String {
        match expr {
            Expression::BinaryOperator { lhs, binop, rhs } => {
                let left = self.emit_expr_mode(lhs, wrap_calls);
                let right = self.emit_expr_mode(rhs, wrap_calls);
                let op = binop.to_string().trim().to_string();
                match op.as_str() {
                    ".." => format!("lua.op_concat({left}, {right})"),
                    "and" => format!("__lua_and({left}, {right})"),
                    "or" => format!("__lua_or({left}, {right})"),
                    "+" => format!("lua.op_add({left}, {right})"),
                    "-" => format!("lua.op_sub({left}, {right})"),
                    "*" => format!("lua.op_mul({left}, {right})"),
                    "/" => format!("lua.op_div({left}, {right})"),
                    "%" => format!("lua.op_mod({left}, {right})"),
                    "==" => format!("lua.op_eq({left}, {right})"),
                    "~=" => format!("lua.op_ne({left}, {right})"),
                    "<" => format!("lua.op_lt({left}, {right})"),
                    "<=" => format!("lua.op_le({left}, {right})"),
                    ">" => format!("lua.op_gt({left}, {right})"),
                    ">=" => format!("lua.op_ge({left}, {right})"),
                    _ => format!("{left} {op} {right}"),
                }
            }
            Expression::UnaryOperator { unop, expression } => {
                match unop {
                    UnOp::Minus(_) => format!("lua.op_neg({})", self.emit_expr_mode(expression, wrap_calls)),
                    UnOp::Not(_) => format!("not __lua_truthy({})", self.emit_expr_mode(expression, wrap_calls)),
                    UnOp::Hash(_) => format!("({}):len()", self.emit_expr_mode(expression, wrap_calls)),
                    _ => self.emit_expr_mode(expression, wrap_calls),
                }
            }
            Expression::Parentheses { expression, .. } => {
                format!("({})", self.emit_expr_mode(expression, wrap_calls))
            }
            Expression::Value { value, .. } => match &**value {
                Value::Number(tok) => {
                    let num_str = tok.to_string();
                    if wrap_calls {
                        format!("lua.to_value({})", num_str)
                    } else {
                        num_str
                    }
                },
                Value::String(tok) => {
                    let str_val = tok.to_string();
                    if wrap_calls {
                        format!("lua.to_value({})", str_val)
                    } else {
                        str_val
                    }
                },
                Value::Symbol(tok) => match tok.token().to_string().as_str() {
                    "nil" => "lua.nil".to_string(),
                    "..." => {
                        if wrap_calls {
                            format!("__lua_first({VARARGS_NAME})")
                        } else {
                            VARARGS_NAME.to_string()
                        }
                    }
                    "true" => if wrap_calls { "lua.to_value(true)".to_string() } else { "true".to_string() },
                    "false" => if wrap_calls { "lua.to_value(false)".to_string() } else { "false".to_string() },
                    other => other.to_string(),
                },
                Value::Var(var) => self.emit_var_mode(var, wrap_calls),
                Value::TableConstructor(table) => self.emit_table(table),
                Value::Function((_token, body)) => {
                    let (params, _) = self.emit_function_params(body);
                    let params = params.join(", ");
                    let mut inner = Emitter::new(&self.module, Analyzer::new(&self.module));
                    inner.indent = self.indent + 1;
                    inner.emit_block(body.block());
                    let body_src = inner.lines.join("\n");
                    let indent = "    ".repeat(self.indent);
                    format!("function({params}): Array<LuaValue>\n{body_src}\n{indent}end")
                }
                Value::FunctionCall(call) => {
                    let raw = self.emit_function_call_raw(call, wrap_calls);
                    if wrap_calls {
                        format!("__lua_first({raw})")
                    } else {
                        raw
                    }
                }
                Value::ParenthesesExpression(expr) => format!("({})", self.emit_expr_mode(expr, wrap_calls)),
                _ => "lua.nil".to_string(),
            },
            _ => "lua.nil".to_string(),
        }
    }

    fn emit_var(&mut self, var: &full_moon::ast::Var) -> String {
        self.emit_var_mode(var, true)
    }

    fn emit_var_mode(&mut self, var: &full_moon::ast::Var, wrap_calls: bool) -> String {
        match var {
            full_moon::ast::Var::Name(tok) => self.sanitize_identifier(tok.to_string()),
            full_moon::ast::Var::Expression(expr) => {
                let mut out = self.emit_prefix_mode(expr.prefix(), wrap_calls);
                for suffix in expr.suffixes() {
                    match suffix {
                        Suffix::Index(index) => match index {
                            Index::Brackets { expression, .. } => {
                                out = format!("{}[{}]", out, self.emit_expr_mode(expression, wrap_calls));
                            }
                            Index::Dot { name, .. } => {
                                let field = name.to_string().trim().to_string();
                                if out == "table" && field == "unpack" {
                                    out = "unpack".to_string();
                                } else if out == "socket" && field == "protect" {
                                    out = "lua.socket_protect".to_string();
                                } else if out == "socket" && field == "skip" {
                                    out = "lua.socket_skip".to_string();
                                } else {
                                    out = self.format_field(out, &field);
                                }
                            }
                            &_ => {}
                        },
                        Suffix::Call(call) => {
                            let call_rendered = self.emit_call_suffix(out, call, wrap_calls);
                            if wrap_calls {
                                out = format!("__lua_first({call_rendered})");
                            } else {
                                out = call_rendered;
                            }
                        }
                        &_ => {}
                    }
                }
                out
            }
            &_ => "lua.nil".to_string(),
        }
    }

    fn emit_prefix(&mut self, prefix: &Prefix) -> String {
        self.emit_prefix_mode(prefix, true)
    }

    fn emit_prefix_mode(&mut self, prefix: &Prefix, wrap_calls: bool) -> String {
        match prefix {
            Prefix::Expression(expr) => self.emit_expr_mode(expr, wrap_calls),
            Prefix::Name(tok) => self.sanitize_identifier(tok.to_string()),
            &_ => "lua.nil".to_string(),
        }
    }

    fn emit_function_call(&mut self, call: &FunctionCall) -> String {
        let raw = self.emit_function_call_raw(call, true);
        format!("__lua_first({raw})")
    }

    fn emit_function_call_raw(&mut self, call: &FunctionCall, wrap_calls: bool) -> String {
        let mut head = self.emit_prefix_mode(call.prefix(), wrap_calls);
        for suffix in call.suffixes() {
            match suffix {
                Suffix::Call(call_suffix) => {
                    head = self.emit_call_suffix(head, call_suffix, wrap_calls);
                }
                Suffix::Index(index) => match index {
                    Index::Brackets { expression, .. } => {
                        head = format!("{}[{}]", head, self.emit_expr_mode(expression, wrap_calls));
                    }
                    Index::Dot { name, .. } => {
                        let field = name.to_string().trim().to_string();
                        if head == "socket" && field == "protect" {
                            head = "lua.socket_protect".to_string();
                        } else if head == "socket" && field == "skip" {
                            head = "lua.socket_skip".to_string();
                        } else {
                            head = self.format_field(head, &field);
                        }
                    }
                    &_ => {}
                },
                &_ => {}
            }
        }
        head
    }

    fn emit_call_suffix(
        &mut self,
        head: String,
        call: &full_moon::ast::Call,
        wrap_calls: bool,
    ) -> String {
        match call {
            full_moon::ast::Call::AnonymousCall(args) => {
                if head == "require" {
                    if let FunctionArgs::Parentheses { arguments, .. } = args {
                        let first = arguments.iter().next();
                        if let Some(Expression::Value { value, .. }) = first {
                            if let Value::String(tok) = &**value {
                                let module = tok
                                    .to_string()
                                    .trim_matches(|c| c == '"' || c == '\'')
                                    .to_string();
                                if module == "math" || module == "table" {
                                    return "lua.nil".to_string();
                                }
                                return format!("lua.require({})", tok.to_string());
                            }
                        }
                        if let Some(expr) = first {
                            return format!("lua.require({})", self.emit_expr_mode(expr, wrap_calls));
                        }
                    }
                    return "lua.nil".to_string();
                }
                if let Some(rewritten) = self.rewrite_compat_call(&head, args, wrap_calls) {
                    return rewritten;
                }
                let args = self.emit_args_mode(args, wrap_calls);
                format!("{head}({args})")
            }
            full_moon::ast::Call::MethodCall(method) => {
                let args = self.emit_args_mode(method.args(), wrap_calls);
                let name = method.name().to_string().trim().to_string();

                // Known string methods that should be called via string module
                let string_methods = [
                    "byte", "char", "dump", "find", "format", "gmatch", "gsub",
                    "len", "lower", "match", "rep", "reverse", "sub", "upper"
                ];

                // If this is a string method, call it via string module instead of lua.call_method
                if string_methods.contains(&name.as_str()) {
                    let all_args = if args.is_empty() {
                        head.clone()
                    } else {
                        format!("{}, {}", head, args)
                    };
                    format!("__lua_first(string.{}({}))", name, all_args)
                } else {
                    // Use lua.call_method for other methods to handle __index metamethod lookup
                    let arg_list = if args.is_empty() {
                        format!("{}, \"{}\"", head, name)
                    } else {
                        format!("{}, \"{}\", {}", head, name, args)
                    };
                    format!("__lua_first(lua.call_method({}))", arg_list)
                }
            }
            &_ => head,
        }
    }

    fn emit_args(&mut self, args: &FunctionArgs) -> String {
        self.emit_args_mode(args, true)
    }

    fn emit_args_mode(&mut self, args: &FunctionArgs, _wrap_calls: bool) -> String {
        match args {
            FunctionArgs::Parentheses { arguments, .. } => {
                // Lua argument-list semantics:
                // - For all arguments except the last: function calls/varargs are truncated to 1 return.
                // - For the last argument: a function call/varargs passes all returns.
                // This is independent of the parent expression context (Lua's "wrap_calls").
                let last_index = arguments.len().saturating_sub(1);
                arguments
                    .iter()
                    .enumerate()
                    .map(|(idx, e)| {
                        let is_last = idx == last_index;
                        let allow_multi = is_last && self.is_multi_return_expr(e);
                        self.emit_expr_mode(e, !allow_multi)
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            }
            FunctionArgs::String(tok) => tok.to_string(),
            FunctionArgs::TableConstructor(table) => self.emit_table(table),
            &_ => String::new(),
        }
    }

    fn emit_table(&mut self, table: &full_moon::ast::TableConstructor) -> String {
        let fields_vec: Vec<&Field> = table.fields().into_iter().collect();
        if fields_vec.len() == 1 {
            if let Field::NoKey(expr) = fields_vec[0] {
                if self.is_varargs_expr(expr) {
                    return format!("__lua_table_from_varargs({})", VARARGS_NAME);
                }
            }
        }

        let mut entries = Vec::new();
        let mut metamethods = Vec::new();
        let mut list_index = 0usize;
        let mut shared_metamethod_tables = Vec::new();

        // Known Lua metamethods
        let metamethod_names = [
            "__call", "__index", "__newindex", "__mode", "__tostring",
            "__metatable", "__len", "__pairs", "__ipairs", "__gc",
            "__add", "__sub", "__mul", "__div", "__mod", "__pow",
            "__unm", "__idiv", "__band", "__bor", "__bxor", "__bnot",
            "__shl", "__shr", "__concat", "__eq", "__lt", "__le",
            "__name", "__close"
        ];

        for field in fields_vec {
            match field {
                Field::NameKey { key, value, .. } => {
                    let name = key.token().to_string().trim().to_string();
                    let is_metamethod = metamethod_names.contains(&name.as_str());

                    let value_str = self.emit_expr(value);

                    // For metamethods with table values, create a shared reference
                    let (key_str, final_value_str) = if is_metamethod && matches!(value, Expression::Value { value, .. } if matches!(&**value, Value::TableConstructor(_))) {
                        // This is a metamethod with a table literal value - create shared reference
                        let var_name = format!("__shared_{}_{}", name, shared_metamethod_tables.len());
                        shared_metamethod_tables.push(format!("local {} = {}", var_name, value_str));
                        (format!("\"{}\"", name), var_name.clone())
                    } else {
                        (format!("\"{}\"", name), value_str)
                    };

                    entries.push(format!("({}, {})", key_str, final_value_str.clone()));

                    if is_metamethod {
                        let field_str = if self.is_keyword(&name) {
                            format!("[\"{}\"] = {}", name, final_value_str)
                        } else {
                            format!("{} = {}", name, final_value_str)
                        };
                        metamethods.push(field_str);
                    }
                }
                Field::ExpressionKey { key, value, .. } => {
                    // Expression key: key can be any expression, value is wrapped
                    entries.push(format!(
                        "({}, {})",
                        self.emit_expr(key),  // Key expression (might already be wrapped)
                        self.emit_expr(value) // Value wrapped in LuaValue
                    ));
                }
                Field::NoKey(value) => {
                    // Array-style field: emit as tuple (plain index, wrapped value)
                    list_index += 1;
                    let expr = self.emit_expr(value);
                    entries.push(format!("({}, {})", list_index, expr));  // Index as plain int
                }
                &_ => {}
            }
        }

        // Use helper when no metamethods, inline when there are metamethods
        if entries.is_empty() && metamethods.is_empty() {
            // Empty table
            "lua.table()".to_string()
        } else if metamethods.is_empty() {
            // Only regular entries, use helper
            format!("lua.table_from_entries([{}])", entries.join(", "))
        } else {
            // Has metamethods - need to inline the map literal
            // Convert entries from tuples to map entries
            let map_entries: Vec<String> = entries.iter().map(|entry| {
                // entry is like "(lua.to_value("key"), lua.to_value(value))"
                // Convert to "[lua.to_value("key")] = lua.to_value(value)"
                if let Some(content) = entry.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
                    let parts: Vec<&str> = content.splitn(2, ", ").collect();
                    if parts.len() == 2 {
                        format!("[{}] = {}", parts[0], parts[1])
                    } else {
                        entry.clone()
                    }
                } else {
                    entry.clone()
                }
            }).collect();

            let table_literal = if map_entries.is_empty() {
                "{}".to_string()
            } else {
                format!("{{ {} }}", map_entries.join(", "))
            };

            let metamethods_literal = format!("{{ {} }}", metamethods.join(", "));
            let lua_table = format!("LuaTable {{ table = {}, metamethods = {} }}", table_literal, metamethods_literal);

            // If we have shared metamethod tables, wrap in IIFE to create them first
            if !shared_metamethod_tables.is_empty() {
                let var_decls = shared_metamethod_tables.join("\n    ");
                format!("(function()\n    {}\n    return {}\nend)()", var_decls, lua_table)
            } else {
                lua_table
            }
        }
    }

    fn rewrite_compat_call(
        &mut self,
        head: &str,
        args: &FunctionArgs,
        wrap_calls: bool,
    ) -> Option<String> {
        let FunctionArgs::Parentheses { arguments, .. } = args else {
            return None;
        };
        let rendered: Vec<String> = arguments
            .iter()
            .map(|e| self.emit_expr_mode(e, wrap_calls))
            .collect();

        let nil = "lua.nil".to_string();
        match head {
            "table.insert" => {
                if rendered.len() < 2 {
                    return Some(format!("{head}({})", rendered.join(", ")));
                }
                let receiver = rendered[0].clone();
                if rendered.len() == 2 {
                    Some(format!("{receiver}:push({})", rendered[1]))
                } else {
                    Some(format!("{receiver}:insert({}, {})", rendered[1], rendered[2]))
                }
            }
            "table.remove" => {
                if rendered.is_empty() {
                    return Some(format!("{head}()"));
                }
                let receiver = rendered[0].clone();
                let pos = rendered.get(1).cloned().unwrap_or(nil);
                Some(format!("{receiver}:remove({pos})"))
            }
            "table.concat" => {
                if rendered.is_empty() {
                    return Some(format!("{head}()"));
                }
                let receiver = rendered[0].clone();
                let sep = rendered.get(1).cloned().unwrap_or_else(|| "\"\"".to_string());
                let i = rendered.get(2).cloned().unwrap_or_else(|| nil.clone());
                let j = rendered.get(3).cloned().unwrap_or_else(|| nil.clone());
                Some(format!("{receiver}:concat({sep}, {i}, {j})"))
            }
            "table.unpack" => {
                if rendered.is_empty() {
                    return Some(format!("{head}()"));
                }
                let receiver = rendered[0].clone();
                let i = rendered.get(1).cloned().unwrap_or_else(|| nil.clone());
                let j = rendered.get(2).cloned().unwrap_or_else(|| nil.clone());
                Some(format!("{receiver}:unpack({i}, {j})"))
            }
            "table.sort" => {
                if rendered.is_empty() {
                    return Some(format!("{head}()"));
                }
                let receiver = rendered[0].clone();
                let comp = rendered.get(1).cloned().unwrap_or(nil);
                Some(format!("{receiver}:sort({comp})"))
            }
            "table.maxn" => {
                if rendered.is_empty() {
                    return Some(format!("{head}()"));
                }
                Some(format!("{}:maxn()", rendered[0]))
            }
            "math.abs" => {
                if rendered.len() != 1 {
                    return Some(format!("{head}({})", rendered.join(", ")));
                }
                Some(format!("lua.unwrap({}):abs()", rendered[0]))
            }
            "math.mod" => {
                if rendered.len() != 2 {
                    return Some(format!("{head}({})", rendered.join(", ")));
                }
                Some(format!(
                    "(lua.unwrap({}) % lua.unwrap({}))",
                    rendered[0], rendered[1]
                ))
            }
            "math.min" => {
                if rendered.is_empty() {
                    return Some(format!("{head}()"));
                }
                let mut expr = format!("lua.unwrap({})", rendered[0]);
                for arg in rendered.iter().skip(1) {
                    expr = format!("{expr}:min(lua.unwrap({arg}))");
                }
                Some(expr)
            }
            "math.max" => {
                if rendered.is_empty() {
                    return Some(format!("{head}()"));
                }
                let mut expr = format!("lua.unwrap({})", rendered[0]);
                for arg in rendered.iter().skip(1) {
                    expr = format!("{expr}:max(lua.unwrap({arg}))");
                }
                Some(expr)
            }
            "math.random" => {
                let m = rendered.get(0).cloned().unwrap_or_else(|| nil.clone());
                let n = rendered.get(1).cloned().unwrap_or_else(|| nil.clone());
                Some(format!("random({m}, {n})"))
            }
            "math.randomseed" => {
                let seed = rendered.get(0).cloned().unwrap_or(nil);
                Some(format!("randomseed({seed})"))
            }
            _ => None,
        }
    }

    fn function_name_parts(&self, name: &FunctionName) -> Option<(Vec<String>, Option<String>)> {
        let parts: Vec<String> = name
            .names()
            .iter()
            .map(|t| sanitize_identifier(&t.to_string()))
            .collect();
        let method = name
            .method_name()
            .map(|m| sanitize_identifier(&m.to_string()));
        if parts.is_empty() && method.is_none() {
            None
        } else {
            Some((parts, method))
        }
    }

    fn push_line(&mut self, line: String) {
        let indent = "    ".repeat(self.indent);
        self.lines.push(format!("{indent}{line}"));
    }

    fn emit_condition(&mut self, expr: &Expression) -> String {
        format!("__lua_truthy({})", self.emit_expr(expr))
    }

    fn is_multi_return_expr(&self, expr: &Expression) -> bool {
        match expr {
            Expression::Value { value, .. } => match &**value {
                Value::FunctionCall(_) => true,
                Value::Var(var) => {
                    if let full_moon::ast::Var::Expression(v) = var {
                        v.suffixes().any(|s| matches!(s, Suffix::Call(_)))
                    } else {
                        false
                    }
                }
                Value::Symbol(tok) => tok.token().to_string() == "...",
                // Parentheses suppress multiple returns in Lua: `(f())` behaves like a single value.
                Value::ParenthesesExpression(_) => false,
                _ => false,
            },
            // Parentheses suppress multiple returns in Lua: `(f())` behaves like a single value.
            Expression::Parentheses { .. } => false,
            _ => false,
        }
    }

    fn extract_function_sig(&self, expr: &Expression) -> Option<WrapperSig> {
        self.find_function_body(expr).map(|body| {
            let (params, args) = self.emit_function_params(body);
            WrapperSig {
                params,
                args,
                return_count: 1,
            }
        })
    }

    fn find_function_body<'a>(&'a self, expr: &'a Expression) -> Option<&'a FunctionBody> {
        match expr {
            Expression::Value { value, .. } => match &**value {
                Value::Function((_token, body)) => Some(body),
                Value::FunctionCall(call) => {
                    if let Some(body) = self.find_function_body_in_prefix(call.prefix()) {
                        return Some(body);
                    }
                    for suffix in call.suffixes() {
                        if let Suffix::Call(call_suffix) = suffix {
                            if let Some(body) = self.find_function_body_in_call(call_suffix) {
                                return Some(body);
                            }
                        }
                    }
                    None
                }
                Value::TableConstructor(table) => {
                    for field in table.fields() {
                        match field {
                            Field::NameKey { value, .. } => {
                                if let Some(body) = self.find_function_body(value) {
                                    return Some(body);
                                }
                            }
                            Field::ExpressionKey { key, value, .. } => {
                                if let Some(body) =
                                    self.find_function_body(key).or_else(|| self.find_function_body(value))
                                {
                                    return Some(body);
                                }
                            }
                            Field::NoKey(value) => {
                                if let Some(body) = self.find_function_body(value) {
                                    return Some(body);
                                }
                            }
                            &_ => {}
                        }
                    }
                    None
                }
                Value::ParenthesesExpression(inner) => self.find_function_body(inner),
                _ => None,
            },
            Expression::BinaryOperator { lhs, rhs, .. } => {
                self.find_function_body(lhs).or_else(|| self.find_function_body(rhs))
            }
            Expression::UnaryOperator { expression, .. } => self.find_function_body(expression),
            Expression::Parentheses { expression, .. } => self.find_function_body(expression),
            _ => None,
        }
    }

    fn find_function_body_in_prefix<'a>(
        &'a self,
        prefix: &'a Prefix,
    ) -> Option<&'a FunctionBody> {
        match prefix {
            Prefix::Expression(expr) => self.find_function_body(expr),
            _ => None,
        }
    }

    fn find_function_body_in_call<'a>(
        &'a self,
        call: &'a full_moon::ast::Call,
    ) -> Option<&'a FunctionBody> {
        match call {
            full_moon::ast::Call::AnonymousCall(args) => self.find_function_body_in_args(args),
            full_moon::ast::Call::MethodCall(method) => self.find_function_body_in_args(method.args()),
            &_ => None,
        }
    }

    fn find_function_body_in_args<'a>(
        &'a self,
        args: &'a FunctionArgs,
    ) -> Option<&'a FunctionBody> {
        match args {
            FunctionArgs::Parentheses { arguments, .. } => {
                for expr in arguments.iter() {
                    if let Some(body) = self.find_function_body(expr) {
                        return Some(body);
                    }
                }
                None
            }
            FunctionArgs::TableConstructor(table) => {
                for field in table.fields() {
                    match field {
                        Field::NameKey { value, .. } => {
                            if let Some(body) = self.find_function_body(value) {
                                return Some(body);
                            }
                        }
                        Field::ExpressionKey { key, value, .. } => {
                            if let Some(body) =
                                self.find_function_body(key).or_else(|| self.find_function_body(value))
                            {
                                return Some(body);
                            }
                        }
                        Field::NoKey(value) => {
                            if let Some(body) = self.find_function_body(value) {
                                return Some(body);
                            }
                        }
                        &_ => {}
                    }
                }
                None
            }
            FunctionArgs::String(_) => None,
            &_ => None,
        }
    }

    fn is_keyword(&self, name: &str) -> bool {
        matches!(
            name.trim(),
            "local"
                | "mut"
                | "function"
                | "return"
                | "if"
                | "then"
                | "else"
                | "elseif"
                | "end"
                | "while"
                | "do"
                | "for"
                | "in"
                | "break"
                | "continue"
                | "struct"
                | "enum"
                | "trait"
                | "impl"
                | "match"
                | "case"
                | "as"
                | "is"
                | "true"
                | "false"
                | "and"
                | "or"
                | "not"
                | "extern"
                | "unsafe"
                | "pub"
                | "use"
                | "module"
                | "const"
                | "static"
                | "type"
        )
    }

    fn is_varargs_expr(&self, expr: &Expression) -> bool {
        match expr {
            Expression::Value { value, .. } => match &**value {
                Value::Symbol(tok) => tok.token().to_string() == "...",
                _ => false,
            },
            Expression::Parentheses { expression, .. } => self.is_varargs_expr(expression),
            _ => false,
        }
    }

    fn sanitize_identifier(&self, name: String) -> String {
        sanitize_identifier(&name)
    }

    fn format_field(&self, base: String, field: &str) -> String {
        if self.is_keyword(field) {
            format!("{base}[\"{field}\"]")
        } else {
            format!("{base}.{field}")
        }
    }

    fn build_path(&self, parts: &[String]) -> String {
        if parts.is_empty() {
            return String::new();
        }
        let mut iter = parts.iter();
        let mut out = self.sanitize_identifier(iter.next().unwrap().to_string());
        for seg in iter {
            out = self.format_field(out, seg);
        }
        out
    }

    fn qualify_module_target(&self, name: &str) -> String {
        if self.analyzer.module_decl.is_some() && !name.starts_with("__module") {
            format!("__module.{}", name)
        } else {
            name.to_string()
        }
    }

    fn is_module_path(&self, parts: &[String]) -> bool {
        if parts.len() > 1 && self.module_path_prefix(parts) {
            return true;
        }
        if let Some(first) = parts.first() {
            if self.module_tables.contains(first) && parts.len() > 1 {
                return true;
            }
        }
        false
    }

    fn module_path_prefix(&self, parts: &[String]) -> bool {
        if parts.len() < self.module_parts.len() {
            return false;
        }
        parts
            .iter()
            .take(self.module_parts.len())
            .zip(self.module_parts.iter())
            .all(|(a, b)| a == b)
    }

    fn is_exported(&self, name: &str) -> bool {
        let name = sanitize_identifier(name);
        self.export_set.contains(&name) || self.analyzer.exports.contains(&name)
    }

    fn export_alias_name(&mut self, parts: &[String], export_name: &str) -> String {
        let mut base = sanitize_identifier(export_name);
        let needs_prefix = match parts.first().map(|s| s.as_str()) {
            Some("_M") | Some("__module") => false,
            _ => !self.module_path_prefix(parts),
        };
        if needs_prefix {
            if let Some(prefix) = parts.first() {
                base = format!("{}_{}", sanitize_identifier(prefix), base);
            }
        }
        if !self.exported_functions.contains(&base) {
            return base;
        }
        let mut suffix = 2usize;
        loop {
            let candidate = format!("{base}_{suffix}");
            if !self.exported_functions.contains(&candidate) {
                return candidate;
            }
            suffix += 1;
        }
    }
}
