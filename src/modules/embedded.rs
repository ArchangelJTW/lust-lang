use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use hashbrown::{HashMap, HashSet};
#[cfg(feature = "std")]
use std::path::PathBuf;

use crate::{
    ast::{ItemKind, UseTree},
    lexer::Lexer,
    modules::{LoadedModule, ModuleExports, ModuleImports, Program},
    parser::Parser,
    LustError, Result,
};

#[derive(Debug, Clone)]
pub struct EmbeddedModule<'a> {
    pub module: &'a str,
    pub parent: Option<&'a str>,
    pub source: Option<&'a str>,
}

pub fn build_directory_map(entries: &[EmbeddedModule<'_>]) -> HashMap<String, Vec<String>> {
    let mut map: HashMap<String, Vec<String>> = HashMap::new();
    for entry in entries {
        let parent = entry.parent.unwrap_or("");
        map.entry(parent.to_string())
            .or_default()
            .push(entry.module.to_string());
    }

    for children in map.values_mut() {
        children.sort();
    }
    map
}

pub fn load_program_from_embedded(
    entries: &[EmbeddedModule<'_>],
    entry_module: &str,
) -> Result<Program> {
    #[cfg(feature = "esp32c6-logging")]
    log::info!("load_program_from_embedded: starting with {} entries", entries.len());

    let mut module_names: HashSet<String> = entries.iter().map(|e| e.module.to_string()).collect();

    #[cfg(feature = "esp32c6-logging")]
    log::info!("load_program_from_embedded: module names collected");

    let mut registry: HashMap<String, LoadedModule> = HashMap::new();
    for entry in entries.iter() {
        #[cfg(feature = "esp32c6-logging")]
        log::info!("load_program_from_embedded: processing module '{}'", entry.module);

        if let Some(source) = entry.source {
            #[cfg(feature = "esp32c6-logging")]
            log::info!("  parsing module '{}' ({} bytes)", entry.module, source.len());

            let module = parse_module(entry.module, source)?;

            #[cfg(feature = "esp32c6-logging")]
            log::info!("  parsed module '{}' successfully", entry.module);

            registry.insert(entry.module.to_string(), module);
        } else {
            module_names.insert(entry.module.to_string());
        }
    }

    #[cfg(feature = "esp32c6-logging")]
    log::info!("load_program_from_embedded: building dependency map");

    let dependency_map = build_dependency_map(&registry, &module_names);

    #[cfg(feature = "esp32c6-logging")]
    log::info!("load_program_from_embedded: dependency map built");

    let mut ordered = Vec::new();
    let mut visited = HashSet::new();
    let mut stack = HashSet::new();

    for module in registry.keys().cloned().collect::<Vec<_>>() {
        visit_dependencies(
            &module,
            &dependency_map,
            &mut visited,
            &mut stack,
            &mut ordered,
        )?;
    }

    #[cfg(feature = "esp32c6-logging")]
    log::info!("load_program_from_embedded: finalizing {} modules", ordered.len());

    for module in ordered {
        finalize_module(&module_names, &mut registry, &module)?;
    }

    let mut modules: Vec<LoadedModule> = registry.into_values().collect();
    modules.sort_by(|a, b| a.path.cmp(&b.path));

    #[cfg(feature = "esp32c6-logging")]
    log::info!("load_program_from_embedded: complete");

    Ok(Program {
        modules,
        entry_module: entry_module.to_string(),
    })
}

fn parse_module(module: &str, source: &str) -> Result<LoadedModule> {
    #[cfg(feature = "esp32c6-logging")]
    log::info!("parse_module: creating lexer for '{}'", module);

    // Create a string interner for this module
    // Estimate capacity based on source size
    let estimated_strings = (source.len() / 20).max(50);
    let mut interner = crate::intern::Interner::with_capacity(estimated_strings);

    let mut lexer = Lexer::new(source, &mut interner);

    #[cfg(feature = "esp32c6-logging")]
    log::info!("parse_module: tokenizing...");

    let mut parser = Parser::from_lexer(&mut lexer)?;

    #[cfg(feature = "esp32c6-logging")]
    {
        log::info!("parse_module: tokenized {} tokens, parsing...", parser.token_count());
        log::info!("  string interner: {} unique strings, {} bytes",
            interner.len(), interner.total_string_bytes());
    }

    let items = parser.parse()?;

    #[cfg(feature = "esp32c6-logging")]
    log::info!("parse_module: parsed {} items", items.len());

    Ok(LoadedModule {
        path: module.to_string(),
        items,
        imports: ModuleImports::default(),
        exports: ModuleExports::default(),
        init_function: None,
        #[cfg(feature = "std")]
        source_path: PathBuf::new(),
    })
}

fn build_dependency_map(
    modules: &HashMap<String, LoadedModule>,
    module_names: &HashSet<String>,
) -> HashMap<String, Vec<String>> {
    let mut deps = HashMap::new();
    for (name, module) in modules {
        let collected = collect_dependencies(module, module_names);
        deps.insert(name.clone(), collected);
    }

    deps
}

fn collect_dependencies(module: &LoadedModule, module_names: &HashSet<String>) -> Vec<String> {
    let mut deps = HashSet::new();
    for item in &module.items {
        match &item.kind {
            ItemKind::Use { public: _, tree } => {
                collect_deps_from_use(tree, module_names, &mut deps);
            }
            ItemKind::Script(stmts) => {
                for stmt in stmts {
                    collect_deps_from_lua_require_stmt(stmt, &mut deps);
                }
            }
            ItemKind::Function(func) => {
                for stmt in &func.body {
                    collect_deps_from_lua_require_stmt(stmt, &mut deps);
                }
            }
            ItemKind::Const { value, .. } | ItemKind::Static { value, .. } => {
                collect_deps_from_lua_require_expr(value, &mut deps);
            }
            ItemKind::Impl(impl_block) => {
                for method in &impl_block.methods {
                    for stmt in &method.body {
                        collect_deps_from_lua_require_stmt(stmt, &mut deps);
                    }
                }
            }
            ItemKind::Trait(trait_def) => {
                for method in &trait_def.methods {
                    if let Some(default_impl) = &method.default_impl {
                        for stmt in default_impl {
                            collect_deps_from_lua_require_stmt(stmt, &mut deps);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    deps.into_iter().collect()
}

fn collect_deps_from_lua_require_stmt(stmt: &crate::ast::Stmt, deps: &mut HashSet<String>) {
    use crate::ast::StmtKind;
    match &stmt.kind {
        StmtKind::Local { initializer, .. } => {
            if let Some(values) = initializer {
                for expr in values {
                    collect_deps_from_lua_require_expr(expr, deps);
                }
            }
        }
        StmtKind::Assign { targets, values } => {
            for expr in targets {
                collect_deps_from_lua_require_expr(expr, deps);
            }
            for expr in values {
                collect_deps_from_lua_require_expr(expr, deps);
            }
        }
        StmtKind::CompoundAssign { target, value, .. } => {
            collect_deps_from_lua_require_expr(target, deps);
            collect_deps_from_lua_require_expr(value, deps);
        }
        StmtKind::Expr(expr) => collect_deps_from_lua_require_expr(expr, deps),
        StmtKind::If {
            condition,
            then_block,
            elseif_branches,
            else_block,
        } => {
            collect_deps_from_lua_require_expr(condition, deps);
            for stmt in then_block {
                collect_deps_from_lua_require_stmt(stmt, deps);
            }
            for (cond, block) in elseif_branches {
                collect_deps_from_lua_require_expr(cond, deps);
                for stmt in block {
                    collect_deps_from_lua_require_stmt(stmt, deps);
                }
            }
            if let Some(block) = else_block {
                for stmt in block {
                    collect_deps_from_lua_require_stmt(stmt, deps);
                }
            }
        }
        StmtKind::While { condition, body } => {
            collect_deps_from_lua_require_expr(condition, deps);
            for stmt in body {
                collect_deps_from_lua_require_stmt(stmt, deps);
            }
        }
        StmtKind::ForNumeric {
            start, end, step, body, ..
        } => {
            collect_deps_from_lua_require_expr(start, deps);
            collect_deps_from_lua_require_expr(end, deps);
            if let Some(step) = step {
                collect_deps_from_lua_require_expr(step, deps);
            }
            for stmt in body {
                collect_deps_from_lua_require_stmt(stmt, deps);
            }
        }
        StmtKind::ForIn { iterator, body, .. } => {
            collect_deps_from_lua_require_expr(iterator, deps);
            for stmt in body {
                collect_deps_from_lua_require_stmt(stmt, deps);
            }
        }
        StmtKind::Return(values) => {
            for expr in values {
                collect_deps_from_lua_require_expr(expr, deps);
            }
        }
        StmtKind::Block(stmts) => {
            for stmt in stmts {
                collect_deps_from_lua_require_stmt(stmt, deps);
            }
        }
        StmtKind::Break | StmtKind::Continue => {}
    }
}

fn collect_deps_from_lua_require_expr(expr: &crate::ast::Expr, deps: &mut HashSet<String>) {
    use crate::ast::{ExprKind, Literal};
    match &expr.kind {
        ExprKind::Call { callee, args } => {
            if is_lua_require_callee(callee) {
                if let Some(name) = args.get(0).and_then(extract_lua_require_name) {
                    if !is_lua_builtin_module_name(&name) {
                        deps.insert(name);
                    }
                }
            }
            collect_deps_from_lua_require_expr(callee, deps);
            for arg in args {
                collect_deps_from_lua_require_expr(arg, deps);
            }
        }
        ExprKind::MethodCall { receiver, args, .. } => {
            collect_deps_from_lua_require_expr(receiver, deps);
            for arg in args {
                collect_deps_from_lua_require_expr(arg, deps);
            }
        }
        ExprKind::Binary { left, right, .. } => {
            collect_deps_from_lua_require_expr(left, deps);
            collect_deps_from_lua_require_expr(right, deps);
        }
        ExprKind::Unary { operand, .. } => collect_deps_from_lua_require_expr(operand, deps),
        ExprKind::FieldAccess { object, .. } => collect_deps_from_lua_require_expr(object, deps),
        ExprKind::Index { object, index } => {
            collect_deps_from_lua_require_expr(object, deps);
            collect_deps_from_lua_require_expr(index, deps);
        }
        ExprKind::Array(elements) | ExprKind::Tuple(elements) => {
            for element in elements {
                collect_deps_from_lua_require_expr(element, deps);
            }
        }
        ExprKind::Map(entries) => {
            for (k, v) in entries {
                collect_deps_from_lua_require_expr(k, deps);
                collect_deps_from_lua_require_expr(v, deps);
            }
        }
        ExprKind::StructLiteral { fields, .. } => {
            for field in fields {
                collect_deps_from_lua_require_expr(&field.value, deps);
            }
        }
        ExprKind::EnumConstructor { args, .. } => {
            for arg in args {
                collect_deps_from_lua_require_expr(arg, deps);
            }
        }
        ExprKind::Lambda { body, .. } => collect_deps_from_lua_require_expr(body, deps),
        ExprKind::Paren(inner) => collect_deps_from_lua_require_expr(inner, deps),
        ExprKind::Cast { expr, .. } => collect_deps_from_lua_require_expr(expr, deps),
        ExprKind::TypeCheck { expr, .. } => collect_deps_from_lua_require_expr(expr, deps),
        ExprKind::IsPattern { expr, .. } => collect_deps_from_lua_require_expr(expr, deps),
        ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_deps_from_lua_require_expr(condition, deps);
            collect_deps_from_lua_require_expr(then_branch, deps);
            if let Some(other) = else_branch {
                collect_deps_from_lua_require_expr(other, deps);
            }
        }
        ExprKind::Block(stmts) => {
            for stmt in stmts {
                collect_deps_from_lua_require_stmt(stmt, deps);
            }
        }
        ExprKind::Return(values) => {
            for value in values {
                collect_deps_from_lua_require_expr(value, deps);
            }
        }
        ExprKind::Range { start, end, .. } => {
            collect_deps_from_lua_require_expr(start, deps);
            collect_deps_from_lua_require_expr(end, deps);
        }
        ExprKind::Literal(Literal::String(_))
        | ExprKind::Literal(_)
        | ExprKind::Identifier(_) => {}
    }
}

fn is_lua_builtin_module_name(name: &str) -> bool {
    matches!(
        name,
        "math"
            | "table"
            | "string"
            | "io"
            | "os"
            | "package"
            | "coroutine"
            | "debug"
            | "utf8"
    )
}

fn is_lua_require_callee(callee: &crate::ast::Expr) -> bool {
    use crate::ast::ExprKind;
    match &callee.kind {
        ExprKind::Identifier(name) => name == "require",
        ExprKind::FieldAccess { object, field } => {
            field == "require" && matches!(&object.kind, ExprKind::Identifier(name) if name == "lua")
        }
        _ => false,
    }
}

fn extract_lua_require_name(expr: &crate::ast::Expr) -> Option<String> {
    use crate::ast::{ExprKind, Literal};
    match &expr.kind {
        ExprKind::Literal(Literal::String(s)) => Some(s.clone()),
        ExprKind::Call { callee, args } if is_lua_to_value_callee(callee) => args
            .get(0)
            .and_then(|arg| match &arg.kind {
                ExprKind::Literal(Literal::String(s)) => Some(s.clone()),
                _ => None,
            }),
        _ => None,
    }
}

fn is_lua_to_value_callee(callee: &crate::ast::Expr) -> bool {
    use crate::ast::ExprKind;
    matches!(
        &callee.kind,
        ExprKind::FieldAccess { object, field }
            if field == "to_value" && matches!(&object.kind, ExprKind::Identifier(name) if name == "lua")
    )
}

fn collect_deps_from_use(
    tree: &UseTree,
    module_names: &HashSet<String>,
    deps: &mut HashSet<String>,
) {
    match tree {
        UseTree::Path { path, .. } => {
            let full = path.join(".");
            if module_names.contains(&full) {
                deps.insert(full);
            } else if path.len() > 1 {
                deps.insert(path[..path.len() - 1].join("."));
            }
        }
        UseTree::Group { prefix, items } => {
            let module = prefix.join(".");
            if !module.is_empty() {
                deps.insert(module);
            }

            for item in items {
                if item.path.len() > 1 {
                    let mut combined = prefix.clone();
                    combined.extend(item.path[..item.path.len() - 1].iter().cloned());
                    let module_path = combined.join(".");
                    if !module_path.is_empty() {
                        deps.insert(module_path);
                    }
                }
            }
        }
        UseTree::Glob { prefix } => {
            deps.insert(prefix.join("."));
        }
    }
}

fn visit_dependencies(
    module: &str,
    deps: &HashMap<String, Vec<String>>,
    visited: &mut HashSet<String>,
    stack: &mut HashSet<String>,
    ordered: &mut Vec<String>,
) -> Result<()> {
    if visited.contains(module) {
        return Ok(());
    }

    if !stack.insert(module.to_string()) {
        return Err(LustError::Unknown(format!(
            "Cyclic dependency detected while loading module '{}'",
            module
        )));
    }

    if let Some(list) = deps.get(module) {
        for dep in list {
            visit_dependencies(dep, deps, visited, stack, ordered)?;
        }
    }

    stack.remove(module);
    visited.insert(module.to_string());
    ordered.push(module.to_string());
    Ok(())
}

fn finalize_module(
    module_names: &HashSet<String>,
    registry: &mut HashMap<String, LoadedModule>,
    module_name: &str,
) -> Result<()> {
    let mut module = registry
        .remove(module_name)
        .ok_or_else(|| LustError::Unknown(format!("Unknown module '{}'", module_name)))?;

    let registry_ref = ModuleRegistryView { modules: registry };
    for item in &module.items {
        if let ItemKind::Use { tree, .. } = &item.kind {
            process_use_tree(&registry_ref, module_names, tree, &mut module.imports)?;
        }
    }

    for item in &module.items {
        if let ItemKind::Use { public: true, tree } = &item.kind {
            apply_reexport(&registry_ref, module_names, tree, &mut module.exports)?;
        }
    }

    let tail = simple_tail(module_name);
    module
        .imports
        .module_aliases
        .entry(tail.to_string())
        .or_insert_with(|| module_name.to_string());

    registry.insert(module_name.to_string(), module);
    Ok(())
}

struct ModuleRegistryView<'a> {
    modules: &'a HashMap<String, LoadedModule>,
}

impl<'a> ModuleRegistryView<'a> {
    fn get(&self, name: &str) -> Option<&'a LoadedModule> {
        self.modules.get(name)
    }
}

fn process_use_tree(
    registry: &ModuleRegistryView<'_>,
    module_names: &HashSet<String>,
    tree: &UseTree,
    imports: &mut ModuleImports,
) -> Result<()> {
    match tree {
        UseTree::Path { path, alias, .. } => {
            let full = path.join(".");
            if module_names.contains(&full) {
                let alias_name = alias
                    .clone()
                    .unwrap_or_else(|| path.last().unwrap().clone());
                imports.module_aliases.insert(alias_name, full);
            } else if path.len() > 1 {
                let module = path[..path.len() - 1].join(".");
                let item = path.last().unwrap().clone();
                let alias_name = alias.clone().unwrap_or_else(|| item.clone());
                let classification = classify_import_target(registry, &module, &item);
                let fq = format!("{}.{}", module, item);
                if classification.import_value {
                    imports
                        .function_aliases
                        .insert(alias_name.clone(), fq.clone());
                }

                if classification.import_type {
                    imports.type_aliases.insert(alias_name, fq);
                }
            }
        }
        UseTree::Group { prefix, items } => {
            for item in items {
                if item.path.is_empty() {
                    continue;
                }

                let alias_name = item
                    .alias
                    .clone()
                    .unwrap_or_else(|| item.path.last().unwrap().clone());
                let mut full_segments = prefix.clone();
                full_segments.extend(item.path.clone());
                let full = full_segments.join(".");
                if module_names.contains(&full) {
                    imports.module_aliases.insert(alias_name, full);
                    continue;
                }

                let mut module_segments = full_segments.clone();
                let item_name = module_segments.pop().unwrap();
                let module_path = module_segments.join(".");
                let fq_name = if module_path.is_empty() {
                    item_name.clone()
                } else {
                    format!("{}.{}", module_path, item_name)
                };
                let classification = classify_import_target(registry, &module_path, &item_name);
                if classification.import_value {
                    imports
                        .function_aliases
                        .insert(alias_name.clone(), fq_name.clone());
                }

                if classification.import_type {
                    imports.type_aliases.insert(alias_name, fq_name);
                }
            }
        }
        UseTree::Glob { prefix } => {
            let module = prefix.join(".");
            if let Some(loaded) = registry.get(&module) {
                for (name, fq) in &loaded.exports.functions {
                    imports.function_aliases.insert(name.clone(), fq.clone());
                }

                for (name, fq) in &loaded.exports.types {
                    imports.type_aliases.insert(name.clone(), fq.clone());
                }
            }

            if !module.is_empty() {
                let alias_name = prefix.last().cloned().unwrap_or_else(|| module.clone());
                imports.module_aliases.insert(alias_name, module);
            }
        }
    }

    Ok(())
}

fn apply_reexport(
    registry: &ModuleRegistryView<'_>,
    module_names: &HashSet<String>,
    tree: &UseTree,
    exports: &mut ModuleExports,
) -> Result<()> {
    match tree {
        UseTree::Path { path, alias, .. } => {
            if path.len() == 1 {
                return Ok(());
            }

            let module = path[..path.len() - 1].join(".");
            let item = path.last().unwrap().clone();
            let alias_name = alias.clone().unwrap_or_else(|| item.clone());
            let fq = format!("{}.{}", module, item);
            let classification = classify_import_target(registry, &module, &item);
            if classification.import_type {
                exports.types.insert(alias_name.clone(), fq.clone());
            }

            if classification.import_value {
                exports.functions.insert(alias_name, fq);
            }

            Ok(())
        }
        UseTree::Group { prefix, items } => {
            for item in items {
                if item.path.is_empty() {
                    continue;
                }

                let mut full_segments = prefix.clone();
                full_segments.extend(item.path.clone());
                let full = full_segments.join(".");
                if module_names.contains(&full) {
                    continue;
                }

                let mut module_segments = full_segments.clone();
                let item_name = module_segments.pop().unwrap();
                let module_path = module_segments.join(".");
                let fq_name = if module_path.is_empty() {
                    item_name.clone()
                } else {
                    format!("{}.{}", module_path, item_name)
                };
                let alias_name = item
                    .alias
                    .clone()
                    .unwrap_or_else(|| item.path.last().unwrap().clone());
                let classification = classify_import_target(registry, &module_path, &item_name);
                if classification.import_type {
                    exports.types.insert(alias_name.clone(), fq_name.clone());
                }

                if classification.import_value {
                    exports.functions.insert(alias_name, fq_name);
                }
            }

            Ok(())
        }
        UseTree::Glob { prefix } => {
            let module = prefix.join(".");
            if let Some(loaded) = registry.get(&module) {
                for (n, fq) in &loaded.exports.types {
                    exports.types.insert(n.clone(), fq.clone());
                }

                for (n, fq) in &loaded.exports.functions {
                    exports.functions.insert(n.clone(), fq.clone());
                }
            }

            Ok(())
        }
    }
}

#[derive(Clone, Copy)]
struct ImportResolution {
    import_value: bool,
    import_type: bool,
}

impl ImportResolution {
    fn both() -> Self {
        Self {
            import_value: true,
            import_type: true,
        }
    }
}

fn classify_import_target(
    registry: &ModuleRegistryView<'_>,
    module_path: &str,
    item_name: &str,
) -> ImportResolution {
    if module_path.is_empty() {
        return ImportResolution::both();
    }

    if let Some(module) = registry.get(module_path) {
        let has_value = module.exports.functions.contains_key(item_name);
        let has_type = module.exports.types.contains_key(item_name);
        if has_value || has_type {
            return ImportResolution {
                import_value: has_value,
                import_type: has_type,
            };
        }
    }

    ImportResolution::both()
}

fn simple_tail(module_path: &str) -> &str {
    module_path
        .rsplit_once('.')
        .map(|(_, n)| n)
        .unwrap_or(module_path)
}
