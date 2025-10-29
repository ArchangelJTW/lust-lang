use crate::{
    ast::{FunctionDef, Item, ItemKind, UseTree, Visibility},
    error::{LustError, Result},
    lexer::Lexer,
    parser::Parser,
};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};
#[derive(Debug, Clone, Default)]
pub struct ModuleImports {
    pub function_aliases: HashMap<String, String>,
    pub module_aliases: HashMap<String, String>,
    pub type_aliases: HashMap<String, String>,
}

#[derive(Debug, Clone, Default)]
pub struct ModuleExports {
    pub functions: HashMap<String, String>,
    pub types: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct LoadedModule {
    pub path: String,
    pub items: Vec<Item>,
    pub imports: ModuleImports,
    pub exports: ModuleExports,
    pub init_function: Option<String>,
    pub source_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct Program {
    pub modules: Vec<LoadedModule>,
    pub entry_module: String,
}

#[derive(Clone, Copy, Debug, Default)]
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

pub struct ModuleLoader {
    base_dir: PathBuf,
    cache: HashMap<String, LoadedModule>,
    visited: HashSet<String>,
    source_overrides: HashMap<PathBuf, String>,
}

impl ModuleLoader {
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
            cache: HashMap::new(),
            visited: HashSet::new(),
            source_overrides: HashMap::new(),
        }
    }

    pub fn set_source_overrides(&mut self, overrides: HashMap<PathBuf, String>) {
        self.source_overrides = overrides;
    }

    pub fn set_source_override<P: Into<PathBuf>, S: Into<String>>(&mut self, path: P, source: S) {
        self.source_overrides.insert(path.into(), source.into());
    }

    pub fn clear_source_overrides(&mut self) {
        self.source_overrides.clear();
    }

    pub fn load_program_from_entry(&mut self, entry_file: &str) -> Result<Program> {
        let entry_path = Path::new(entry_file);
        let entry_dir = entry_path.parent().unwrap_or_else(|| Path::new("."));
        self.base_dir = entry_dir.to_path_buf();
        let entry_module = Self::module_path_for_file(entry_path);
        let mut order: Vec<String> = Vec::new();
        let mut stack: HashSet<String> = HashSet::new();
        self.load_module_recursive(&entry_module, &mut order, &mut stack, true)?;
        let modules = order
            .into_iter()
            .filter_map(|m| self.cache.get(&m).cloned())
            .collect::<Vec<_>>();
        Ok(Program {
            modules,
            entry_module,
        })
    }

    fn load_module_recursive(
        &mut self,
        module_path: &str,
        order: &mut Vec<String>,
        stack: &mut HashSet<String>,
        is_entry: bool,
    ) -> Result<()> {
        if self.visited.contains(module_path) {
            return Ok(());
        }

        if !stack.insert(module_path.to_string()) {
            return Ok(());
        }

        let mut loaded = self.load_single_module(module_path, is_entry)?;
        self.cache.insert(module_path.to_string(), loaded.clone());
        let deps = self.collect_dependencies(&loaded.items);
        for dep in deps {
            self.load_module_recursive(&dep, order, stack, false)?;
        }

        self.finalize_module(&mut loaded)?;
        self.cache.insert(module_path.to_string(), loaded.clone());
        self.visited.insert(module_path.to_string());
        order.push(module_path.to_string());
        stack.remove(module_path);
        Ok(())
    }

    fn load_single_module(&self, module_path: &str, is_entry: bool) -> Result<LoadedModule> {
        let file = self.file_for_module_path(module_path);
        let source = if let Some(src) = self.source_overrides.get(&file) {
            src.clone()
        } else {
            fs::read_to_string(&file).map_err(|e| {
                LustError::Unknown(format!("Failed to read module '{}': {}", file.display(), e))
            })?
        };
        let mut lexer = Lexer::new(&source);
        let tokens = lexer
            .tokenize()
            .map_err(|err| Self::attach_module_to_error(err, module_path))?;
        let mut parser = Parser::new(tokens);
        let mut items = parser
            .parse()
            .map_err(|err| Self::attach_module_to_error(err, module_path))?;
        let mut imports = ModuleImports::default();
        let mut exports = ModuleExports::default();
        let mut new_items: Vec<Item> = Vec::new();
        let mut init_function: Option<String> = None;
        for item in items.drain(..) {
            match &item.kind {
                ItemKind::Function(func) => {
                    let mut f = func.clone();
                    if !f.is_method && !f.name.contains(':') && !f.name.contains('.') {
                        let fq = format!("{}.{}", module_path, f.name);
                        imports.function_aliases.insert(f.name.clone(), fq.clone());
                        f.name = fq.clone();
                        if matches!(f.visibility, Visibility::Public) {
                            exports
                                .functions
                                .insert(self.simple_name(&f.name).to_string(), f.name.clone());
                        }
                    } else {
                        if matches!(f.visibility, Visibility::Public) {
                            exports
                                .functions
                                .insert(self.simple_name(&f.name).to_string(), f.name.clone());
                        }
                    }

                    new_items.push(Item::new(ItemKind::Function(f), item.span));
                }

                ItemKind::Struct(s) => {
                    if matches!(s.visibility, Visibility::Public) {
                        exports
                            .types
                            .insert(s.name.clone(), format!("{}.{}", module_path, s.name));
                    }

                    new_items.push(item);
                }

                ItemKind::Enum(e) => {
                    if matches!(e.visibility, Visibility::Public) {
                        exports
                            .types
                            .insert(e.name.clone(), format!("{}.{}", module_path, e.name));
                    }

                    new_items.push(item);
                }

                ItemKind::Trait(t) => {
                    if matches!(t.visibility, Visibility::Public) {
                        exports
                            .types
                            .insert(t.name.clone(), format!("{}.{}", module_path, t.name));
                    }

                    new_items.push(item);
                }

                ItemKind::TypeAlias { name, .. } => {
                    exports
                        .types
                        .insert(name.clone(), format!("{}.{}", module_path, name));
                    new_items.push(item);
                }

                ItemKind::Script(stmts) => {
                    if is_entry {
                        new_items.push(Item::new(ItemKind::Script(stmts.clone()), item.span));
                    } else {
                        let init_name = format!("__init@{}", module_path);
                        let func = FunctionDef {
                            name: init_name.clone(),
                            type_params: vec![],
                            trait_bounds: vec![],
                            params: vec![],
                            return_type: None,
                            body: stmts.clone(),
                            is_method: false,
                            visibility: Visibility::Private,
                        };
                        new_items.push(Item::new(ItemKind::Function(func), item.span));
                        init_function = Some(init_name);
                    }
                }

                ItemKind::Extern {
                    abi,
                    items: extern_items,
                } => {
                    let mut rewritten = Vec::new();
                    for extern_item in extern_items {
                        match extern_item {
                            crate::ast::ExternItem::Function {
                                name,
                                params,
                                return_type,
                            } => {
                                let mut new_name = name.clone();
                                if !new_name.contains('.') && !new_name.contains(':') {
                                    new_name = format!("{}.{}", module_path, new_name);
                                }

                                exports.functions.insert(
                                    self.simple_name(&new_name).to_string(),
                                    new_name.clone(),
                                );

                                rewritten.push(crate::ast::ExternItem::Function {
                                    name: new_name,
                                    params: params.clone(),
                                    return_type: return_type.clone(),
                                });
                            }
                        }
                    }
                    new_items.push(Item::new(
                        ItemKind::Extern {
                            abi: abi.clone(),
                            items: rewritten,
                        },
                        item.span,
                    ));
                }

                _ => {
                    new_items.push(item);
                }
            }
        }

        Ok(LoadedModule {
            path: module_path.to_string(),
            items: new_items,
            imports,
            exports,
            init_function,
            source_path: file,
        })
    }

    fn collect_dependencies(&self, items: &[Item]) -> Vec<String> {
        let mut deps = HashSet::new();
        for item in items {
            if let ItemKind::Use { public: _, tree } = &item.kind {
                self.collect_deps_from_use(tree, &mut deps);
            }
        }

        deps.into_iter().collect()
    }

    fn finalize_module(&mut self, module: &mut LoadedModule) -> Result<()> {
        for item in &module.items {
            if let ItemKind::Use { tree, .. } = &item.kind {
                self.process_use_tree(tree, &mut module.imports)?;
            }
        }

        for item in &module.items {
            if let ItemKind::Use { public: true, tree } = &item.kind {
                self.apply_reexport(tree, &mut module.exports)?;
            }
        }

        module
            .imports
            .module_aliases
            .entry(self.simple_tail(&module.path).to_string())
            .or_insert_with(|| module.path.clone());
        Ok(())
    }

    fn collect_deps_from_use(&self, tree: &UseTree, deps: &mut HashSet<String>) {
        match tree {
            UseTree::Path {
                path,
                alias: _,
                import_module: _,
            } => {
                let full = path.join(".");
                let full_file = self.file_for_module_path(&full);
                if self.module_source_known(&full, &full_file) {
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
                        let mut combined: Vec<String> = prefix.clone();
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

    fn process_use_tree(&self, tree: &UseTree, imports: &mut ModuleImports) -> Result<()> {
        match tree {
            UseTree::Path { path, alias, .. } => {
                let full = path.join(".");
                let full_file = self.file_for_module_path(&full);
                if self.module_source_known(&full, &full_file) {
                    let alias_name = alias
                        .clone()
                        .unwrap_or_else(|| path.last().unwrap().clone());
                    imports.module_aliases.insert(alias_name, full);
                } else if path.len() > 1 {
                    let module = path[..path.len() - 1].join(".");
                    let item = path.last().unwrap().clone();
                    let alias_name = alias.clone().unwrap_or_else(|| item.clone());
                    let classification = self.classify_import_target(&module, &item);
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
                    let full_file = self.file_for_module_path(&full);
                    if self.module_source_known(&full, &full_file) {
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
                    let classification = self.classify_import_target(&module_path, &item_name);
                    if classification.import_value {
                        imports
                            .function_aliases
                            .insert(alias_name.clone(), fq_name.clone());
                    }

                    if classification.import_type {
                        imports.type_aliases.insert(alias_name.clone(), fq_name);
                    }
                }
            }

            UseTree::Glob { prefix } => {
                let module = prefix.join(".");
                if let Some(loaded) = self.cache.get(&module) {
                    for (name, fq) in &loaded.exports.functions {
                        imports.function_aliases.insert(name.clone(), fq.clone());
                    }

                    for (name, fq) in &loaded.exports.types {
                        imports.type_aliases.insert(name.clone(), fq.clone());
                    }
                }

                let alias_name = prefix.last().cloned().unwrap_or_else(|| module.clone());
                if !module.is_empty() {
                    imports.module_aliases.insert(alias_name, module);
                }
            }
        }

        Ok(())
    }

    fn attach_module_to_error(error: LustError, module_path: &str) -> LustError {
        match error {
            LustError::LexerError {
                line,
                column,
                message,
                module,
            } => LustError::LexerError {
                line,
                column,
                message,
                module: module.or_else(|| Some(module_path.to_string())),
            },
            LustError::ParserError {
                line,
                column,
                message,
                module,
            } => LustError::ParserError {
                line,
                column,
                message,
                module: module.or_else(|| Some(module_path.to_string())),
            },
            LustError::CompileErrorWithSpan {
                message,
                line,
                column,
                module,
            } => LustError::CompileErrorWithSpan {
                message,
                line,
                column,
                module: module.or_else(|| Some(module_path.to_string())),
            },
            other => other,
        }
    }

    fn apply_reexport(&self, tree: &UseTree, exports: &mut ModuleExports) -> Result<()> {
        match tree {
            UseTree::Path { path, alias, .. } => {
                if path.len() == 1 {
                    return Ok(());
                }

                let module = path[..path.len() - 1].join(".");
                let item = path.last().unwrap().clone();
                let alias_name = alias.clone().unwrap_or_else(|| item.clone());
                let fq = format!("{}.{}", module, item);
                let classification = self.classify_import_target(&module, &item);
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

                    let alias_name = item
                        .alias
                        .clone()
                        .unwrap_or_else(|| item.path.last().unwrap().clone());
                    let mut full_segments = prefix.clone();
                    full_segments.extend(item.path.clone());
                    let full = full_segments.join(".");
                    let full_file = self.file_for_module_path(&full);
                    if self.module_source_known(&full, &full_file) {
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
                    let classification = self.classify_import_target(&module_path, &item_name);
                    if classification.import_type {
                        exports.types.insert(alias_name.clone(), fq_name.clone());
                    }

                    if classification.import_value {
                        exports.functions.insert(alias_name.clone(), fq_name);
                    }
                }

                Ok(())
            }

            UseTree::Glob { prefix } => {
                let module = prefix.join(".");
                if let Some(loaded) = self.cache.get(&module) {
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

    fn simple_name<'a>(&self, qualified: &'a str) -> &'a str {
        qualified
            .rsplit_once('.')
            .map(|(_, n)| n)
            .unwrap_or(qualified)
    }

    fn simple_tail<'a>(&self, module_path: &'a str) -> &'a str {
        module_path
            .rsplit_once('.')
            .map(|(_, n)| n)
            .unwrap_or(module_path)
    }

    fn module_source_known(&self, module_path: &str, file: &Path) -> bool {
        file.exists()
            || self.source_overrides.contains_key(file)
            || self.cache.contains_key(module_path)
    }

    fn classify_import_target(&self, module_path: &str, item_name: &str) -> ImportResolution {
        if module_path.is_empty() {
            return ImportResolution::both();
        }

        if let Some(module) = self.cache.get(module_path) {
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

    fn file_for_module_path(&self, module_path: &str) -> PathBuf {
        let mut p = self.base_dir.clone();
        for seg in module_path.split('.') {
            p.push(seg);
        }

        p.set_extension("lust");
        p
    }

    fn module_path_for_file(path: &Path) -> String {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        stem.to_string()
    }
}
