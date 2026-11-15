use crate::ast::Item;
#[cfg(feature = "std")]
use crate::{
    Span,
    ast::{FunctionDef, ItemKind, Type, TypeKind, UseTree, Visibility},
    error::{LustError, Result},
    lexer::Lexer,
    parser::Parser,
};
#[cfg(feature = "std")]
use alloc::{format, vec};
use alloc::{string::String, vec::Vec};
use hashbrown::{HashMap, HashSet};
#[cfg(feature = "std")]
use std::path::{Path, PathBuf};
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

pub mod embedded;

#[derive(Debug, Clone)]
pub struct LoadedModule {
    pub path: String,
    pub items: Vec<Item>,
    pub imports: ModuleImports,
    pub exports: ModuleExports,
    pub init_function: Option<String>,
    #[cfg(feature = "std")]
    pub source_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct Program {
    pub modules: Vec<LoadedModule>,
    pub entry_module: String,
}

pub use embedded::{EmbeddedModule, build_directory_map, load_program_from_embedded, load_program_from_embedded_with_config};

use crate::LustConfig;
use crate::bytecode::Compiler;
use crate::typechecker::TypeChecker;
use crate::vm::VM;

/// Compiles a Program into a VM with memory optimizations.
/// This is designed for no_std contexts where memory is constrained (e.g., ESP32).
///
/// # Arguments
/// * `program` - The parsed program from `load_program_from_embedded`
/// * `config` - Configuration with low_memory_mode and minimal_runtime_types options
///
/// # Example
/// ```ignore
/// let entries = &[EmbeddedModule { module: "main", parent: None, source: Some(source) }];
/// let program = load_program_from_embedded(entries, "main")?;
/// let mut config = LustConfig::default();
/// config.set_low_memory_mode(true);
/// config.set_minimal_runtime_types(true);
/// let vm = compile_program_with_config(program, &config)?;
/// vm.call_function("__script", &[])?;
/// ```
pub fn compile_program_with_config(program: Program, config: &LustConfig) -> crate::Result<VM> {
    let mut config = config.clone();
    for module in &program.modules {
        for target in module.imports.module_aliases.values() {
            if crate::builtins::is_virtual_std_module(target) {
                let canon = crate::builtins::canonical_std_module_name(target);
                if config.is_std_module_allowed(canon) {
                    config.enable_module(canon);
                }
            }
        }
        for target in module.imports.function_aliases.values() {
            if let Some((mod_part, _)) = target.split_once('.') {
                if crate::builtins::is_virtual_std_module(mod_part) {
                    let canon = crate::builtins::canonical_std_module_name(mod_part);
                    if config.is_std_module_allowed(canon) {
                        config.enable_module(canon);
                    }
                }
            }
        }
    }

    // Build imports map
    let mut imports_map: HashMap<String, ModuleImports> = HashMap::new();
    for module in &program.modules {
        let mut imports = module.imports.clone();
        for target in imports.module_aliases.values() {
            if crate::builtins::canonical_std_module_name(target) == "table" {
                imports
                    .type_aliases
                    .entry("Table".to_string())
                    .or_insert_with(|| "std.table.Table".to_string());
                imports
                    .type_aliases
                    .entry("LuaTable".to_string())
                    .or_insert_with(|| "std.table.LuaTable".to_string());
            }
        }
        for (alias, target) in &imports.function_aliases {
            if target == "std.table.Table" || target == "table.Table" {
                imports
                    .type_aliases
                    .entry(alias.clone())
                    .or_insert_with(|| "std.table.Table".to_string());
            }
            if target == "std.table.LuaTable" || target == "table.LuaTable" {
                imports
                    .type_aliases
                    .entry(alias.clone())
                    .or_insert_with(|| "std.table.LuaTable".to_string());
            }
        }
        imports_map.insert(module.path.clone(), imports);
    }

    // Phase 1: Type check
    let mut typechecker = TypeChecker::with_config(&config);
    typechecker.set_imports_by_module(imports_map.clone());
    typechecker.check_program(&program.modules)?;
    let option_coercions = typechecker.take_option_coercions();
    let checked_array_indices = typechecker.take_checked_array_indices();
    let lambda_return_types = typechecker.take_lambda_return_types();
    let numeric_types = typechecker.take_numeric_types();
    let struct_defs = typechecker.take_struct_definitions();
    let _enum_defs = typechecker.take_enum_definitions();
    let signatures = typechecker.take_function_signatures();
    // Typechecker no longer needed - free its memory
    drop(typechecker);

    // Phase 2: Build wrapped items
    let program_entry_module = program.entry_module;
    let mut wrapped_items: Vec<Item> = Vec::new();
    for module in program.modules {
        wrapped_items.push(Item::new(
            crate::ast::ItemKind::Module {
                name: module.path,
                items: module.items,
            },
            crate::ast::Span::new(0, 0, 0, 0),
        ));
    }

    // Phase 3: Compile
    let mut compiler = Compiler::new();
    compiler.set_option_coercions(option_coercions);
    compiler.set_checked_array_indices(checked_array_indices);
    compiler.set_lambda_return_types(lambda_return_types);
    compiler.set_numeric_types(numeric_types);
    compiler.configure_stdlib(&config);
    compiler.set_imports_by_module(imports_map);
    compiler.set_entry_module(program_entry_module.clone());
    compiler.set_function_signatures(signatures);
    compiler.set_minimal_runtime_types(config.minimal_runtime_types());
    compiler.set_use_mir(config.use_mir());
    compiler.set_strict_mir(config.strict_mir());
    let functions = compiler.compile_module(&wrapped_items)?;
    let mir_functions = compiler.take_mir_functions();
    let trait_impls = compiler.get_trait_impls().to_vec();
    // AST no longer needed
    drop(wrapped_items);

    // Phase 4: Create VM
    let mut vm = VM::with_config(&config);
    vm.load_functions(functions);
    for mir_fn in mir_functions {
        vm.load_mir_function(mir_fn)?;
    }
    vm.register_structs(&struct_defs);
    for (type_name, trait_name) in trait_impls {
        vm.register_trait_impl(type_name, trait_name);
    }

    Ok(vm)
}

#[allow(dead_code)]
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

#[cfg(feature = "std")]
pub struct ModuleLoader {
    base_dir: PathBuf,
    cache: HashMap<String, LoadedModule>,
    visited: HashSet<String>,
    source_overrides: HashMap<PathBuf, String>,
    module_roots: HashMap<String, Vec<ModuleRoot>>,
    config: LustConfig,
    data_cache: HashMap<PathBuf, (u64, crate::data::DataValue)>,
    pub vfs: std::sync::Arc<dyn crate::vfs::VirtualFileSystem>,
}

#[cfg(feature = "std")]
#[derive(Debug, Clone)]
struct ModuleRoot {
    base: PathBuf,
    root_module: Option<PathBuf>,
}

#[cfg(feature = "std")]
impl ModuleLoader {
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
            cache: HashMap::new(),
            visited: HashSet::new(),
            source_overrides: HashMap::new(),
            module_roots: HashMap::new(),
            config: LustConfig::default(),
            data_cache: HashMap::new(),
            vfs: crate::vfs::current_vfs(),
        }
    }

    pub fn with_vfs(mut self, vfs: std::sync::Arc<dyn crate::vfs::VirtualFileSystem>) -> Self {
        self.vfs = vfs;
        self
    }

    pub fn set_vfs(&mut self, vfs: std::sync::Arc<dyn crate::vfs::VirtualFileSystem>) {
        self.vfs = vfs;
    }

    pub fn vfs(&self) -> &std::sync::Arc<dyn crate::vfs::VirtualFileSystem> {
        &self.vfs
    }

    pub fn with_config(mut self, config: LustConfig) -> Self {
        self.config = config;
        self
    }

    pub fn set_config(&mut self, config: LustConfig) {
        self.config = config;
    }

    pub fn config(&self) -> &LustConfig {
        &self.config
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

    pub fn data_cache(&self) -> &HashMap<PathBuf, (u64, crate::data::DataValue)> {
        &self.data_cache
    }

    pub fn clear_data_cache(&mut self) {
        self.data_cache.clear();
    }

    pub fn add_module_root(
        &mut self,
        prefix: impl Into<String>,
        root: impl Into<PathBuf>,
        root_module: Option<PathBuf>,
    ) {
        self.module_roots
            .entry(prefix.into())
            .or_default()
            .push(ModuleRoot {
                base: root.into(),
                root_module,
            });
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

    pub fn digest_bytes(bytes: &[u8]) -> u64 {
        use core::hash::Hasher;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        hasher.write(bytes);
        hasher.finish()
    }

    pub fn load_data_import(&mut self, path: &Path) -> Result<crate::data::DataValue> {
        let resolved = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.base_dir.join(path)
        };

        let bytes = if let Some(src) = self.source_overrides.get(&resolved) {
            src.as_bytes().to_vec()
        } else {
            match self.vfs.read(&resolved) {
                Ok(b) => b,
                Err(e) => {
                    return Err(LustError::CompileError(format!(
                        "Compile error: missing data import file '{}': {}",
                        resolved.display(),
                        e
                    )));
                }
            }
        };

        let digest = Self::digest_bytes(&bytes);
        if let Some((cached_digest, val)) = self.data_cache.get(&resolved) {
            if *cached_digest == digest {
                return Ok(val.clone());
            }
        }

        let s = std::str::from_utf8(&bytes).map_err(|e| {
            LustError::CompileError(format!(
                "Compile error: invalid UTF-8 in data import '{}': {}",
                resolved.display(),
                e
            ))
        })?;

        let val = crate::data::json::from_str(s).map_err(|e| {
            LustError::CompileError(format!(
                "Compile error: invalid JSON in data import '{}': {}",
                resolved.display(),
                e
            ))
        })?;

        Self::verify_finite_numbers(&val, &resolved)?;

        self.data_cache.insert(resolved, (digest, val.clone()));
        Ok(val)
    }

    fn verify_finite_numbers(val: &crate::data::DataValue, path: &Path) -> Result<()> {
        use crate::data::{DataValue, ExactNumber};
        match val {
            DataValue::Number(ExactNumber::Float(f)) => {
                if !f.is_finite() {
                    return Err(LustError::CompileError(format!(
                        "Compile error: non-finite number in data import '{}'",
                        path.display()
                    )));
                }
            }
            DataValue::Array(arr) => {
                for item in arr {
                    Self::verify_finite_numbers(item, path)?;
                }
            }
            DataValue::Document(doc) => {
                for (_k, v) in doc.iter() {
                    Self::verify_finite_numbers(v, path)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub fn data_value_to_lust_code(val: &crate::data::DataValue) -> String {
        use crate::data::{DataValue, ExactNumber};
        match val {
            DataValue::Null => "nil".to_string(),
            DataValue::Bool(b) => if *b { "true".to_string() } else { "false".to_string() },
            DataValue::Number(ExactNumber::Int(i)) => format!("{i}"),
            DataValue::Number(ExactNumber::Float(f)) => format!("{f:?}"),
            DataValue::Number(ExactNumber::UInt(u)) => format!("{u}"),
            DataValue::Number(ExactNumber::Decimal128(_)) => "0".to_string(),
            DataValue::String(s) => format!("{s:?}"),
            DataValue::Binary(bytes) => format!("{bytes:?}"),
            DataValue::ObjectId(oid) => format!("{:?}", oid.to_hex()),
            DataValue::DateTimeUtc(dt) => format!("{}", dt.millis()),
            DataValue::Array(items) => {
                let els: Vec<String> = items.iter().map(Self::data_value_to_lust_code).collect();
                format!("[{}]", els.join(", "))
            }
            DataValue::Document(doc) => {
                let mut parts = Vec::new();
                for (k, v) in doc.iter() {
                    parts.push(format!("[{:?}] = {}", k, Self::data_value_to_lust_code(v)));
                }
                format!("{{ {} }}", parts.join(", "))
            }
        }
    }

    fn parse_import_statement(line: &str) -> Option<(bool, String, String)> {
        let trimmed = line.trim();
        if trimmed.starts_with("//") || trimmed.starts_with("--") {
            return None;
        }
        let (is_export, rest) = if let Some(stripped) = trimmed.strip_prefix("export") {
            if !stripped.chars().next().map_or(false, |c| c.is_whitespace()) {
                return None;
            }
            (true, stripped.trim_start())
        } else {
            (false, trimmed)
        };
        if !rest.starts_with("import") {
            return None;
        }
        let after_import = rest.strip_prefix("import")?;
        if after_import.is_empty() || !after_import.chars().next()?.is_whitespace() {
            return None;
        }
        let after_import = after_import.trim_start();
        let from_idx = after_import.find("from")?;
        let ident = after_import[..from_idx].trim();
        if ident.is_empty() || !ident.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return None;
        }
        let after_from = after_import[from_idx + 4..].trim_start().trim_end_matches(';').trim();
        if (after_from.starts_with('"') && after_from.ends_with('"'))
            || (after_from.starts_with('\'') && after_from.ends_with('\''))
        {
            if after_from.len() >= 2 {
                let path = after_from[1..after_from.len() - 1].to_string();
                return Some((is_export, ident.to_string(), path));
            }
        }
        None
    }

    fn extract_dynamic_import_target(line: &str) -> Option<String> {
        let trimmed = line.trim();
        if trimmed.starts_with("//") || trimmed.starts_with("--") {
            return None;
        }
        for kw in ["import(", "require("] {
            let mut search_from = 0;
            while let Some(pos) = line[search_from..].find(kw) {
                let actual_pos = search_from + pos;
                let after = line[actual_pos + kw.len()..].trim_start();
                if let Some(quote) = after.chars().next() {
                    if quote == '"' || quote == '\'' {
                        let inner = &after[1..];
                        if let Some(end_quote) = inner.find(quote) {
                            let target = &inner[..end_quote];
                            return Some(target.to_string());
                        }
                    }
                }
                search_from = actual_pos + kw.len();
            }
        }
        None
    }

    fn process_data_imports(&mut self, source: &str, current_file: &Path) -> Result<String> {
        let mut result = String::with_capacity(source.len());
        for line in source.lines() {
            let processed_line = if line.trim_start().starts_with("use @std/")
                || line.trim_start().starts_with("pub use @std/")
                || line.trim_start().starts_with("export use @std/")
            {
                line.replace("@std/", "std.")
            } else {
                line.to_string()
            };

            if let Some(target) = Self::extract_dynamic_import_target(&processed_line) {
                if crate::builtins::is_virtual_std_module(&target) {
                    if !self.config.is_std_module_allowed(&target) {
                        return Err(LustError::CompileError(format!(
                            "CapabilityDenied: dynamic import of module '{}' is not enabled in this sandbox",
                            target
                        )));
                    }
                }
            }

            if let Some((is_export, ident, import_path)) = Self::parse_import_statement(&processed_line) {
                if crate::builtins::is_virtual_std_module(&import_path) {
                    if !self.config.is_std_module_allowed(&import_path) {
                        return Err(LustError::CompileError(format!(
                            "CapabilityDenied: module '{}' is not enabled in this sandbox",
                            import_path
                        )));
                    }
                    let canon = crate::builtins::canonical_std_module_name(&import_path);
                    if is_export {
                        result.push_str(&format!("export use std.{} as {}\n", canon, ident));
                    } else {
                        result.push_str(&format!("use std.{} as {}\n", canon, ident));
                    }
                    continue;
                }
                if import_path.ends_with(".json") {
                    let resolved = if import_path.starts_with("./") || import_path.starts_with("../") {
                        current_file.parent().unwrap_or(Path::new(".")).join(&import_path)
                    } else {
                        let p = Path::new(&import_path);
                        if p.is_absolute() {
                            p.to_path_buf()
                        } else {
                            let rel = current_file.parent().unwrap_or(Path::new(".")).join(&import_path);
                            if self.vfs.exists(&rel) || self.source_overrides.contains_key(&rel) {
                                rel
                            } else {
                                self.base_dir.join(&import_path)
                            }
                        }
                    };
                    let val = self.load_data_import(&resolved)?;
                    let code = Self::data_value_to_lust_code(&val);
                    result.push_str(&format!("local {} = {}\n", ident, code));
                    continue;
                }
            }
            result.push_str(&processed_line);
            result.push('\n');
        }
        Ok(result)
    }

    fn load_single_module(&mut self, module_path: &str, is_entry: bool) -> Result<LoadedModule> {
        let file = self.file_for_module_path(module_path);
        let raw_source = if let Some(src) = self.source_overrides.get(&file) {
            src.clone()
        } else {
            match self.vfs.read_to_string(&file) {
                Ok(src) => src,
                Err(e) => {
                    // For non-entry modules, allow missing files and create stub
                    if !is_entry && e.kind() == std::io::ErrorKind::NotFound {
                        #[cfg(feature = "std")]
                        eprintln!(
                            "[WARNING] Module '{}' not found, but present in code",
                            module_path
                        );
                        //return Ok(self.create_nil_stub_module(module_path));
                    }
                    return Err(LustError::Unknown(format!(
                        "Failed to read module '{}': {}",
                        file.display(),
                        e
                    )));
                }
            }
        };
        let source = self.process_data_imports(&raw_source, &file)?;
        let mut interner = crate::intern::Interner::new();
        let mut lexer = Lexer::new(&source, &mut interner);
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
        let mut pending_init_stmts: Vec<crate::ast::Stmt> = Vec::new();
        let mut pending_init_span: Option<Span> = None;
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

                ItemKind::Script(stmts) => {
                    if is_entry {
                        new_items.push(Item::new(ItemKind::Script(stmts.clone()), item.span));
                    } else {
                        if pending_init_span.is_none() {
                            pending_init_span = Some(item.span);
                        }
                        pending_init_stmts.extend(stmts.iter().cloned());
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
                                doc,
                            } => {
                                let mut new_name = name.clone();
                                if let Some((head, tail)) = new_name.split_once(':') {
                                    let qualified_head =
                                        if head.contains('.') || head.contains("::") {
                                            head.to_string()
                                        } else {
                                            format!("{}.{}", module_path, head)
                                        };
                                    new_name = format!("{}:{}", qualified_head, tail);
                                } else if !new_name.contains('.') {
                                    new_name = format!("{}.{}", module_path, new_name);
                                }

                                exports.functions.insert(
                                    self.simple_name(&new_name).to_string(),
                                    new_name.clone(),
                                );
                                imports.function_aliases.insert(
                                    self.simple_name(&new_name).to_string(),
                                    new_name.clone(),
                                );

                                rewritten.push(crate::ast::ExternItem::Function {
                                    name: new_name,
                                    params: params.clone(),
                                    return_type: return_type.clone(),
                                    doc: doc.clone(),
                                });
                            }

                            crate::ast::ExternItem::Const { name, ty, doc } => {
                                let qualified = if name.contains('.') {
                                    name.clone()
                                } else {
                                    format!("{}.{}", module_path, name)
                                };
                                exports.functions.insert(
                                    self.simple_name(&qualified).to_string(),
                                    qualified.clone(),
                                );
                                imports.function_aliases.insert(
                                    self.simple_name(&qualified).to_string(),
                                    qualified.clone(),
                                );
                                rewritten.push(crate::ast::ExternItem::Const {
                                    name: qualified,
                                    ty: ty.clone(),
                                    doc: doc.clone(),
                                });
                            }

                            crate::ast::ExternItem::Struct(def) => {
                                let mut def = def.clone();
                                if !def.name.contains('.') && !def.name.contains("::") {
                                    def.name = format!("{}.{}", module_path, def.name);
                                }
                                exports.types.insert(
                                    self.simple_name(&def.name).to_string(),
                                    def.name.clone(),
                                );
                                rewritten.push(crate::ast::ExternItem::Struct(def));
                            }

                            crate::ast::ExternItem::Enum(def) => {
                                let mut def = def.clone();
                                if !def.name.contains('.') && !def.name.contains("::") {
                                    def.name = format!("{}.{}", module_path, def.name);
                                }
                                exports.types.insert(
                                    self.simple_name(&def.name).to_string(),
                                    def.name.clone(),
                                );
                                rewritten.push(crate::ast::ExternItem::Enum(def));
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

        if !is_entry && !pending_init_stmts.is_empty() {
            let init_name = format!("__init@{}", module_path);
            let func = FunctionDef {
                name: init_name.clone(),
                type_params: vec![],
                trait_bounds: vec![],
                params: vec![],
                // Module initializers may return a Lua module value for the loader to publish.
                return_type: Some(Type::new(TypeKind::Unknown, Span::dummy())),
                body: pending_init_stmts,
                is_method: false,
                visibility: Visibility::Private,
                doc: None,
                effects: None,
            };
            let span = pending_init_span.unwrap_or_else(Span::dummy);
            // Place module init first so the compiler can observe module-level locals
            // (e.g. transpiler prelude helpers) before compiling functions that reference them.
            new_items.insert(0, Item::new(ItemKind::Function(func), span));
            init_function = Some(init_name);
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
            match &item.kind {
                ItemKind::Use { public: _, tree } => {
                    self.collect_deps_from_use(tree, &mut deps);
                }
                ItemKind::Script(stmts) => {
                    for stmt in stmts {
                        self.collect_deps_from_lua_require_stmt(stmt, &mut deps);
                    }
                }
                ItemKind::Function(func) => {
                    for stmt in &func.body {
                        self.collect_deps_from_lua_require_stmt(stmt, &mut deps);
                    }
                }
                ItemKind::Impl(impl_block) => {
                    for method in &impl_block.methods {
                        for stmt in &method.body {
                            self.collect_deps_from_lua_require_stmt(stmt, &mut deps);
                        }
                    }
                }
                ItemKind::Trait(trait_def) => {
                    for method in &trait_def.methods {
                        if let Some(default_impl) = &method.default_impl {
                            for stmt in default_impl {
                                self.collect_deps_from_lua_require_stmt(stmt, &mut deps);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        deps.into_iter().collect()
    }

    fn collect_deps_from_lua_require_stmt(
        &self,
        stmt: &crate::ast::Stmt,
        deps: &mut HashSet<String>,
    ) {
        use crate::ast::StmtKind;
        match &stmt.kind {
            StmtKind::Local { initializer, .. } => {
                if let Some(values) = initializer {
                    for expr in values {
                        self.collect_deps_from_lua_require_expr(expr, deps);
                    }
                }
            }
            StmtKind::Assign { targets, values } => {
                for expr in targets {
                    self.collect_deps_from_lua_require_expr(expr, deps);
                }
                for expr in values {
                    self.collect_deps_from_lua_require_expr(expr, deps);
                }
            }
            StmtKind::CompoundAssign { target, value, .. } => {
                self.collect_deps_from_lua_require_expr(target, deps);
                self.collect_deps_from_lua_require_expr(value, deps);
            }
            StmtKind::Expr(expr) => self.collect_deps_from_lua_require_expr(expr, deps),
            StmtKind::If {
                condition,
                then_block,
                elseif_branches,
                else_block,
            } => {
                self.collect_deps_from_lua_require_expr(condition, deps);
                for stmt in then_block {
                    self.collect_deps_from_lua_require_stmt(stmt, deps);
                }
                for (cond, block) in elseif_branches {
                    self.collect_deps_from_lua_require_expr(cond, deps);
                    for stmt in block {
                        self.collect_deps_from_lua_require_stmt(stmt, deps);
                    }
                }
                if let Some(block) = else_block {
                    for stmt in block {
                        self.collect_deps_from_lua_require_stmt(stmt, deps);
                    }
                }
            }
            StmtKind::While { condition, body } => {
                self.collect_deps_from_lua_require_expr(condition, deps);
                for stmt in body {
                    self.collect_deps_from_lua_require_stmt(stmt, deps);
                }
            }
            StmtKind::ForNumeric {
                start,
                end,
                step,
                body,
                ..
            } => {
                self.collect_deps_from_lua_require_expr(start, deps);
                self.collect_deps_from_lua_require_expr(end, deps);
                if let Some(step) = step {
                    self.collect_deps_from_lua_require_expr(step, deps);
                }
                for stmt in body {
                    self.collect_deps_from_lua_require_stmt(stmt, deps);
                }
            }
            StmtKind::ForIn { iterator, body, .. } => {
                self.collect_deps_from_lua_require_expr(iterator, deps);
                for stmt in body {
                    self.collect_deps_from_lua_require_stmt(stmt, deps);
                }
            }
            StmtKind::Return(values) => {
                for expr in values {
                    self.collect_deps_from_lua_require_expr(expr, deps);
                }
            }
            StmtKind::Block(stmts) => {
                for stmt in stmts {
                    self.collect_deps_from_lua_require_stmt(stmt, deps);
                }
            }
            StmtKind::Break | StmtKind::Continue => {}
            StmtKind::Defer(s) | StmtKind::ErrDefer(s) => {
                self.collect_deps_from_lua_require_stmt(s, deps);
            }
            StmtKind::Scoped(stmts) => {
                for stmt in stmts {
                    self.collect_deps_from_lua_require_stmt(stmt, deps);
                }
            }
        }
    }

    fn collect_deps_from_lua_require_expr(
        &self,
        expr: &crate::ast::Expr,
        deps: &mut HashSet<String>,
    ) {
        use crate::ast::{ExprKind, Literal};
        match &expr.kind {
            ExprKind::Call { callee, args, .. } => {
                if self.is_lua_require_callee(callee)
                    && let Some(name) = args
                        .first()
                        .and_then(|arg| self.extract_lua_require_name(arg))
                    && !Self::is_lua_builtin_module_name(&name)
                {
                    // `lua.require()` calls originate from transpiled Lua stubs. Unlike
                    // Lust `use` imports, these should only pull in modules that we can
                    // actually locate in the current module roots (extern stubs, on-disk
                    // modules, or source overrides). This prevents optional Lua requires
                    // (e.g. `ssl.https`) from becoming hard compile-time dependencies.
                    let file = self.file_for_module_path(&name);
                    if self.module_source_known(&name, &file) {
                        deps.insert(name);
                    }
                }
                self.collect_deps_from_lua_require_expr(callee, deps);
                for arg in args {
                    self.collect_deps_from_lua_require_expr(arg, deps);
                }
            }
            ExprKind::MethodCall { receiver, args, .. } => {
                self.collect_deps_from_lua_require_expr(receiver, deps);
                for arg in args {
                    self.collect_deps_from_lua_require_expr(arg, deps);
                }
            }
            ExprKind::Binary { left, right, .. } => {
                self.collect_deps_from_lua_require_expr(left, deps);
                self.collect_deps_from_lua_require_expr(right, deps);
            }
            ExprKind::Unary { operand, .. } => {
                self.collect_deps_from_lua_require_expr(operand, deps)
            }
            ExprKind::FieldAccess { object, .. } => {
                self.collect_deps_from_lua_require_expr(object, deps)
            }
            ExprKind::Index { object, index } => {
                self.collect_deps_from_lua_require_expr(object, deps);
                self.collect_deps_from_lua_require_expr(index, deps);
            }
            ExprKind::Array(elements) | ExprKind::Tuple(elements) => {
                for element in elements {
                    self.collect_deps_from_lua_require_expr(element, deps);
                }
            }
            ExprKind::Map(entries) => {
                for (k, v) in entries {
                    self.collect_deps_from_lua_require_expr(k, deps);
                    self.collect_deps_from_lua_require_expr(v, deps);
                }
            }
            ExprKind::StructLiteral { fields, .. } => {
                for field in fields {
                    self.collect_deps_from_lua_require_expr(&field.value, deps);
                }
            }
            ExprKind::EnumConstructor { args, .. } => {
                for arg in args {
                    self.collect_deps_from_lua_require_expr(arg, deps);
                }
            }
            ExprKind::Lambda { body, .. } => self.collect_deps_from_lua_require_expr(body, deps),
            ExprKind::Paren(inner) => self.collect_deps_from_lua_require_expr(inner, deps),
            ExprKind::Cast { expr, .. } => self.collect_deps_from_lua_require_expr(expr, deps),
            ExprKind::TypeCheck { expr, .. } => self.collect_deps_from_lua_require_expr(expr, deps),
            ExprKind::IsPattern { expr, .. } => self.collect_deps_from_lua_require_expr(expr, deps),
            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.collect_deps_from_lua_require_expr(condition, deps);
                self.collect_deps_from_lua_require_expr(then_branch, deps);
                if let Some(other) = else_branch {
                    self.collect_deps_from_lua_require_expr(other, deps);
                }
            }
            ExprKind::Block(stmts) => {
                for stmt in stmts {
                    self.collect_deps_from_lua_require_stmt(stmt, deps);
                }
            }
            ExprKind::Return(values) => {
                for value in values {
                    self.collect_deps_from_lua_require_expr(value, deps);
                }
            }
            ExprKind::Range { start, end, .. } => {
                self.collect_deps_from_lua_require_expr(start, deps);
                self.collect_deps_from_lua_require_expr(end, deps);
            }
            ExprKind::Literal(Literal::String(_))
            | ExprKind::Literal(_)
            | ExprKind::Identifier(_)
            | ExprKind::RegexLiteral { .. } => {}
            ExprKind::EffectBlock(stmts) => {
                for stmt in stmts {
                    self.collect_deps_from_lua_require_stmt(stmt, deps);
                }
            }
        }
    }

    fn is_lua_builtin_module_name(name: &str) -> bool {
        matches!(
            name,
            "math" | "table" | "string" | "io" | "os" | "package" | "coroutine" | "debug" | "utf8"
        )
    }

    fn is_lua_require_callee(&self, callee: &crate::ast::Expr) -> bool {
        use crate::ast::ExprKind;
        match &callee.kind {
            ExprKind::Identifier(name) => name == "require",
            ExprKind::FieldAccess { object, field } => {
                field == "require"
                    && matches!(&object.kind, ExprKind::Identifier(name) if name == "lua")
            }
            _ => false,
        }
    }

    fn extract_lua_require_name(&self, expr: &crate::ast::Expr) -> Option<String> {
        use crate::ast::{ExprKind, Literal};
        match &expr.kind {
            ExprKind::Literal(Literal::String(s)) => Some(s.clone()),
            ExprKind::Call { callee, args, .. } if self.is_lua_to_value_callee(callee) => {
                args.first().and_then(|arg| match &arg.kind {
                    ExprKind::Literal(Literal::String(s)) => Some(s.clone()),
                    _ => None,
                })
            }
            _ => None,
        }
    }

    fn is_lua_to_value_callee(&self, callee: &crate::ast::Expr) -> bool {
        use crate::ast::ExprKind;
        matches!(
            &callee.kind,
            ExprKind::FieldAccess { object, field }
                if field == "to_value"
                    && matches!(&object.kind, ExprKind::Identifier(name) if name == "lua")
        )
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

        self.validate_dynamic_imports(&module.items)?;

        module
            .imports
            .module_aliases
            .entry(self.simple_tail(&module.path).to_string())
            .or_insert_with(|| module.path.clone());
        Ok(())
    }

    fn validate_dynamic_imports(&self, items: &[Item]) -> Result<()> {
        for item in items {
            match &item.kind {
                ItemKind::Script(stmts) => {
                    for stmt in stmts {
                        self.validate_dynamic_imports_in_stmt(stmt)?;
                    }
                }
                ItemKind::Function(func) => {
                    for stmt in &func.body {
                        self.validate_dynamic_imports_in_stmt(stmt)?;
                    }
                }
                ItemKind::Impl(impl_block) => {
                    for method in &impl_block.methods {
                        for stmt in &method.body {
                            self.validate_dynamic_imports_in_stmt(stmt)?;
                        }
                    }
                }
                ItemKind::Trait(trait_def) => {
                    for method in &trait_def.methods {
                        if let Some(default_impl) = &method.default_impl {
                            for stmt in default_impl {
                                self.validate_dynamic_imports_in_stmt(stmt)?;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn validate_dynamic_imports_in_stmt(&self, stmt: &crate::ast::Stmt) -> Result<()> {
        use crate::ast::StmtKind;
        match &stmt.kind {
            StmtKind::Local { initializer, .. } => {
                if let Some(values) = initializer {
                    for expr in values {
                        self.validate_dynamic_imports_in_expr(expr)?;
                    }
                }
            }
            StmtKind::Assign { targets, values } => {
                for expr in targets {
                    self.validate_dynamic_imports_in_expr(expr)?;
                }
                for expr in values {
                    self.validate_dynamic_imports_in_expr(expr)?;
                }
            }
            StmtKind::CompoundAssign { target, value, .. } => {
                self.validate_dynamic_imports_in_expr(target)?;
                self.validate_dynamic_imports_in_expr(value)?;
            }
            StmtKind::Expr(expr) => self.validate_dynamic_imports_in_expr(expr)?,
            StmtKind::If {
                condition,
                then_block,
                elseif_branches,
                else_block,
            } => {
                self.validate_dynamic_imports_in_expr(condition)?;
                for s in then_block {
                    self.validate_dynamic_imports_in_stmt(s)?;
                }
                for (cond, block) in elseif_branches {
                    self.validate_dynamic_imports_in_expr(cond)?;
                    for s in block {
                        self.validate_dynamic_imports_in_stmt(s)?;
                    }
                }
                if let Some(block) = else_block {
                    for s in block {
                        self.validate_dynamic_imports_in_stmt(s)?;
                    }
                }
            }
            StmtKind::While { condition, body } => {
                self.validate_dynamic_imports_in_expr(condition)?;
                for s in body {
                    self.validate_dynamic_imports_in_stmt(s)?;
                }
            }
            StmtKind::ForNumeric {
                start,
                end,
                step,
                body,
                ..
            } => {
                self.validate_dynamic_imports_in_expr(start)?;
                self.validate_dynamic_imports_in_expr(end)?;
                if let Some(step) = step {
                    self.validate_dynamic_imports_in_expr(step)?;
                }
                for s in body {
                    self.validate_dynamic_imports_in_stmt(s)?;
                }
            }
            StmtKind::ForIn { iterator, body, .. } => {
                self.validate_dynamic_imports_in_expr(iterator)?;
                for s in body {
                    self.validate_dynamic_imports_in_stmt(s)?;
                }
            }
            StmtKind::Return(values) => {
                for expr in values {
                    self.validate_dynamic_imports_in_expr(expr)?;
                }
            }
            StmtKind::Block(stmts) | StmtKind::Scoped(stmts) => {
                for s in stmts {
                    self.validate_dynamic_imports_in_stmt(s)?;
                }
            }
            StmtKind::Break | StmtKind::Continue => {}
            StmtKind::Defer(s) | StmtKind::ErrDefer(s) => {
                self.validate_dynamic_imports_in_stmt(s)?;
            }
        }
        Ok(())
    }

    fn validate_dynamic_imports_in_expr(&self, expr: &crate::ast::Expr) -> Result<()> {
        use crate::ast::{ExprKind, Literal};
        match &expr.kind {
            ExprKind::Call { callee, args, .. } => {
                let is_dynamic_import = match &callee.kind {
                    ExprKind::Identifier(name) => name == "import" || name == "require",
                    ExprKind::FieldAccess { object, field } => {
                        (field == "import" || field == "require")
                            && matches!(&object.kind, ExprKind::Identifier(name) if name == "lua")
                    }
                    _ => false,
                };
                if is_dynamic_import {
                    if let Some(arg) = args.first() {
                        if let ExprKind::Literal(Literal::String(mod_name)) = &arg.kind {
                            if crate::builtins::is_virtual_std_module(mod_name) {
                                if !self.config.is_std_module_allowed(mod_name) {
                                    return Err(LustError::CompileError(format!(
                                        "CapabilityDenied: dynamic import of module '{}' is not enabled in this sandbox",
                                        mod_name
                                    )));
                                }
                            }
                        }
                    }
                }
                self.validate_dynamic_imports_in_expr(callee)?;
                for arg in args {
                    self.validate_dynamic_imports_in_expr(arg)?;
                }
            }
            ExprKind::MethodCall { receiver, args, .. } => {
                self.validate_dynamic_imports_in_expr(receiver)?;
                for arg in args {
                    self.validate_dynamic_imports_in_expr(arg)?;
                }
            }
            ExprKind::Binary { left, right, .. } => {
                self.validate_dynamic_imports_in_expr(left)?;
                self.validate_dynamic_imports_in_expr(right)?;
            }
            ExprKind::Unary { operand, .. } => self.validate_dynamic_imports_in_expr(operand)?,
            ExprKind::FieldAccess { object, .. } => self.validate_dynamic_imports_in_expr(object)?,
            ExprKind::Index { object, index } => {
                self.validate_dynamic_imports_in_expr(object)?;
                self.validate_dynamic_imports_in_expr(index)?;
            }
            ExprKind::Array(elements) | ExprKind::Tuple(elements) => {
                for element in elements {
                    self.validate_dynamic_imports_in_expr(element)?;
                }
            }
            ExprKind::Map(entries) => {
                for (k, v) in entries {
                    self.validate_dynamic_imports_in_expr(k)?;
                    self.validate_dynamic_imports_in_expr(v)?;
                }
            }
            ExprKind::StructLiteral { fields, .. } => {
                for field in fields {
                    self.validate_dynamic_imports_in_expr(&field.value)?;
                }
            }
            ExprKind::EnumConstructor { args, .. } => {
                for arg in args {
                    self.validate_dynamic_imports_in_expr(arg)?;
                }
            }
            ExprKind::Lambda { body, .. } => self.validate_dynamic_imports_in_expr(body)?,
            ExprKind::Paren(inner) => self.validate_dynamic_imports_in_expr(inner)?,
            ExprKind::Cast { expr, .. } => self.validate_dynamic_imports_in_expr(expr)?,
            ExprKind::TypeCheck { expr, .. } => self.validate_dynamic_imports_in_expr(expr)?,
            ExprKind::IsPattern { expr, .. } => self.validate_dynamic_imports_in_expr(expr)?,
            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.validate_dynamic_imports_in_expr(condition)?;
                self.validate_dynamic_imports_in_expr(then_branch)?;
                if let Some(other) = else_branch {
                    self.validate_dynamic_imports_in_expr(other)?;
                }
            }
            ExprKind::Block(stmts) => {
                for s in stmts {
                    self.validate_dynamic_imports_in_stmt(s)?;
                }
            }
            ExprKind::Return(values) => {
                for value in values {
                    self.validate_dynamic_imports_in_expr(value)?;
                }
            }
            ExprKind::Range { start, end, .. } => {
                self.validate_dynamic_imports_in_expr(start)?;
                self.validate_dynamic_imports_in_expr(end)?;
            }
            ExprKind::Literal(_) | ExprKind::Identifier(_) | ExprKind::RegexLiteral { .. } => {}
            ExprKind::EffectBlock(stmts) => {
                for s in stmts {
                    self.validate_dynamic_imports_in_stmt(s)?;
                }
            }
        }
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
                } else if crate::builtins::is_virtual_std_module(&full) {
                    // Virtual standard module — do not attempt to load from disk
                } else if path.len() > 1 {
                    let parent = path[..path.len() - 1].join(".");
                    if !crate::builtins::is_virtual_std_module(&parent) {
                        deps.insert(parent);
                    }
                }
            }

            UseTree::Group { prefix, items } => {
                let module = prefix.join(".");
                if crate::builtins::is_virtual_std_module(&module) {
                    // Virtual standard module — do not attempt to load from disk
                } else {
                    if !module.is_empty() {
                        deps.insert(module);
                    }

                    for item in items {
                        if item.path.len() > 1 {
                            let mut combined: Vec<String> = prefix.clone();
                            combined.extend(item.path[..item.path.len() - 1].iter().cloned());
                            let module_path = combined.join(".");
                            if !module_path.is_empty()
                                && !crate::builtins::is_virtual_std_module(&module_path)
                            {
                                deps.insert(module_path);
                            }
                        }
                    }
                }
            }

            UseTree::Glob { prefix } => {
                let module = prefix.join(".");
                if !crate::builtins::is_virtual_std_module(&module) {
                    deps.insert(module);
                }
            }
        }
    }

    fn process_use_tree(&mut self, tree: &UseTree, imports: &mut ModuleImports) -> Result<()> {
        match tree {
            UseTree::Path { path, alias, .. } => {
                let full = path.join(".");
                let full_file = self.file_for_module_path(&full);
                if self.module_source_known(&full, &full_file) {
                    let alias_name = alias
                        .clone()
                        .unwrap_or_else(|| path.last().unwrap().clone());
                    imports.module_aliases.insert(alias_name, full);
                } else if crate::builtins::is_virtual_std_module(&full) {
                    if !self.config.is_std_module_allowed(&full) {
                        return Err(LustError::CompileError(format!(
                            "CapabilityDenied: module '{}' is not enabled in this sandbox",
                            full
                        )));
                    }
                    let canon = crate::builtins::canonical_std_module_name(&full);
                    self.config.enable_module(canon);
                    let alias_name = alias.clone().unwrap_or_else(|| canon.to_string());
                    imports
                        .module_aliases
                        .insert(alias_name, canon.to_string());
                    if canon == "table" {
                        imports
                            .type_aliases
                            .insert("Table".to_string(), "std.table.Table".to_string());
                        imports
                            .type_aliases
                            .insert("LuaTable".to_string(), "std.table.LuaTable".to_string());
                    }
                } else if path.len() > 1 {
                    let module = path[..path.len() - 1].join(".");
                    let item = path.last().unwrap().clone();
                    let alias_name = alias.clone().unwrap_or_else(|| item.clone());
                    if crate::builtins::is_virtual_std_module(&module) {
                        if !self.config.is_std_module_allowed(&module) {
                            return Err(LustError::CompileError(format!(
                                "CapabilityDenied: module '{}' is not enabled in this sandbox",
                                module
                            )));
                        }
                        let canon = crate::builtins::canonical_std_module_name(&module);
                        self.config.enable_module(canon);
                        let fq = crate::builtins::resolve_std_function(&module, &item)
                            .map(String::from)
                            .unwrap_or_else(|| {
                                format!(
                                    "{}.{}",
                                    crate::builtins::canonical_std_module_name(&module),
                                    item
                                )
                            });
                        if canon == "table" && (item == "Table" || item == "LuaTable") {
                            imports
                                .type_aliases
                                .insert(alias_name.clone(), format!("std.table.{}", item));
                        }
                        imports.function_aliases.insert(alias_name, fq);
                    } else {
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
            }

            UseTree::Group { prefix, items } => {
                let module = prefix.join(".");
                let is_virtual = crate::builtins::is_virtual_std_module(&module);
                if is_virtual {
                    if !self.config.is_std_module_allowed(&module) {
                        return Err(LustError::CompileError(format!(
                            "CapabilityDenied: module '{}' is not enabled in this sandbox",
                            module
                        )));
                    }
                    let canon = crate::builtins::canonical_std_module_name(&module);
                    self.config.enable_module(canon);
                    for item in items {
                        if item.path.is_empty() {
                            continue;
                        }
                        let item_name = item.path.last().unwrap().clone();
                        let alias_name = item.alias.clone().unwrap_or_else(|| item_name.clone());
                        let fq = crate::builtins::resolve_std_function(&module, &item_name)
                            .map(String::from)
                            .unwrap_or_else(|| {
                                format!(
                                    "{}.{}",
                                    crate::builtins::canonical_std_module_name(&module),
                                    item_name
                                )
                            });
                        if canon == "table" && (item_name == "Table" || item_name == "LuaTable") {
                            imports
                                .type_aliases
                                .insert(alias_name.clone(), format!("std.table.{}", item_name));
                        }
                        imports.function_aliases.insert(alias_name, fq);
                    }
                } else {
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
            }

            UseTree::Glob { prefix } => {
                let module = prefix.join(".");
                if crate::builtins::is_virtual_std_module(&module) {
                    if !self.config.is_std_module_allowed(&module) {
                        return Err(LustError::CompileError(format!(
                            "CapabilityDenied: module '{}' is not enabled in this sandbox",
                            module
                        )));
                    }
                    let canon = crate::builtins::canonical_std_module_name(&module);
                    self.config.enable_module(canon);
                    for (fn_name, fq) in crate::builtins::virtual_std_module_functions(&module) {
                        imports.function_aliases.insert(fn_name.to_string(), fq.to_string());
                    }
                    imports.module_aliases.insert(canon.to_string(), canon.to_string());
                    if canon == "table" {
                        imports
                            .type_aliases
                            .insert("Table".to_string(), "std.table.Table".to_string());
                        imports
                            .type_aliases
                            .insert("LuaTable".to_string(), "std.table.LuaTable".to_string());
                    }
                } else {
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
                if crate::builtins::is_virtual_std_module(&fq) {
                    for (fn_name, fn_fq) in crate::builtins::virtual_std_module_functions(&fq) {
                        exports.functions.insert(format!("{}.{}", alias_name, fn_name), fn_fq.to_string());
                    }
                    exports.functions.insert(alias_name.clone(), fq.clone());
                } else {
                    let classification = self.classify_import_target(&module, &item);
                    if classification.import_type {
                        exports.types.insert(alias_name.clone(), fq.clone());
                    }

                    if classification.import_value {
                        exports.functions.insert(alias_name, fq);
                    }
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
        self.vfs.exists(file)
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
        let segments: Vec<&str> = module_path.split('.').collect();
        let candidates = self.resolve_dependency_roots(&segments);
        if !candidates.is_empty() {
            let mut fallback: Option<PathBuf> = None;
            for (root, consumed) in candidates.iter().rev() {
                let candidate = Self::path_from_root(root, &segments, *consumed);
                if self.vfs.exists(&candidate) {
                    return candidate;
                }
                if fallback.is_none() {
                    fallback = Some(candidate);
                }
            }
            if let Some(path) = fallback {
                return path;
            }
        }

        let mut fallback = self.base_dir.clone();
        for seg in &segments {
            fallback.push(seg);
        }
        fallback.set_extension("lust");
        fallback
    }

    fn resolve_dependency_roots(&self, segments: &[&str]) -> Vec<(&ModuleRoot, usize)> {
        let mut matched: Vec<(&ModuleRoot, usize)> = Vec::new();
        let mut prefix_segments: Vec<&str> = Vec::new();
        for (index, segment) in segments.iter().enumerate() {
            prefix_segments.push(*segment);
            let key = prefix_segments.join(".");
            if let Some(roots) = self.module_roots.get(&key) {
                for root in roots {
                    matched.push((root, index + 1));
                }
            }
        }
        matched
    }

    fn path_from_root(root: &ModuleRoot, segments: &[&str], consumed: usize) -> PathBuf {
        let mut path = root.base.clone();
        if consumed == segments.len() {
            if let Some(relative) = &root.root_module {
                path.push(relative);
            } else if let Some(last) = segments.last() {
                path.push(format!("{last}.lust"));
            }
            return path;
        }
        for seg in &segments[consumed..segments.len() - 1] {
            path.push(seg);
        }
        if let Some(last) = segments.last() {
            path.push(format!("{last}.lust"));
        }
        path
    }

    fn module_path_for_file(path: &Path) -> String {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        stem.to_string()
    }
}

#[cfg(not(feature = "std"))]
pub struct ModuleLoader {
    cache: HashMap<String, LoadedModule>,
    visited: HashSet<String>,
}

#[cfg(not(feature = "std"))]
impl ModuleLoader {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            visited: HashSet::new(),
        }
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.visited.clear();
    }

    pub fn load_program_from_modules(
        &mut self,
        modules: Vec<LoadedModule>,
        entry_module: String,
    ) -> Program {
        Program {
            modules,
            entry_module,
        }
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods, clippy::disallowed_types)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let mut dir = std::env::temp_dir();
        dir.push(format!("{prefix}_{nanos}"));
        dir
    }

    #[test]
    fn merges_multiple_script_chunks_into_single_init() {
        let dir = unique_temp_dir("lust_module_loader_test");
        fs::create_dir_all(&dir).unwrap();
        let entry_path = dir.join("main.lust");
        let module_path = dir.join("m.lust");

        fs::write(&entry_path, "use m as m\n").unwrap();

        // Interleaved script + declarations; parser will produce multiple ItemKind::Script chunks.
        fs::write(
            &module_path,
            r#"
local a: int = 1

function f(): int
    return a
end

local b: int = 2

function g(): int
    return b
end

local c: int = a + b
"#,
        )
        .unwrap();

        let mut loader = ModuleLoader::new(dir.clone());
        let program = loader
            .load_program_from_entry(entry_path.to_str().unwrap())
            .unwrap();

        let module = program.modules.iter().find(|m| m.path == "m").unwrap();
        assert_eq!(module.init_function.as_deref(), Some("__init@m"));

        let init_functions: Vec<&FunctionDef> = module
            .items
            .iter()
            .filter_map(|item| match &item.kind {
                ItemKind::Function(f) if f.name == "__init@m" => Some(f),
                _ => None,
            })
            .collect();
        assert_eq!(init_functions.len(), 1);
        assert_eq!(init_functions[0].body.len(), 3);
        assert!(matches!(
            init_functions[0].return_type.as_ref().map(|ty| &ty.kind),
            Some(TypeKind::Unknown)
        ));

        // Best-effort cleanup.
        let _ = fs::remove_file(entry_path);
        let _ = fs::remove_file(module_path);
        let _ = fs::remove_dir(dir);
    }

    #[test]
    fn lua_require_does_not_force_missing_modules_as_dependencies() {
        let dir = unique_temp_dir("lust_module_loader_lua_require_missing");
        fs::create_dir_all(&dir).unwrap();
        let entry_path = dir.join("main.lust");
        let module_path = dir.join("a.lust");

        fs::write(&entry_path, "use a as a\n").unwrap();

        // `lua.require("missing.module")` appears inside a closure (so it should remain optional).
        // The module loader must not try to resolve it as a hard on-disk dependency.
        fs::write(
            &module_path,
            r#"
use lua as lua

local f = function(): unknown
    return lua.require("missing.module")
end
"#,
        )
        .unwrap();

        let mut loader = ModuleLoader::new(dir.clone());
        let program = loader
            .load_program_from_entry(entry_path.to_str().unwrap())
            .unwrap();

        assert!(program.modules.iter().any(|m| m.path == "a"));
        assert!(!program.modules.iter().any(|m| m.path == "missing.module"));

        // Best-effort cleanup.
        let _ = fs::remove_file(entry_path);
        let _ = fs::remove_file(module_path);
        let _ = fs::remove_dir(dir);
    }

    #[test]
    fn exported_by_default_and_local_privatizes() {
        let dir = unique_temp_dir("lust_module_visibility_test");
        fs::create_dir_all(&dir).unwrap();
        let entry_path = dir.join("main.lust");
        let module_path = dir.join("lib.lust");

        fs::write(
            &module_path,
            r#"
struct ExportedStruct
    x: int
    local secret: int
end

local struct PrivateStruct
    y: int
end

function exported_fn(): int
    return 1
end

local function private_fn(): int
    return 2
end

enum ExportedEnum
    VariantA
end

local enum PrivateEnum
    VariantB
end
"#,
        )
        .unwrap();

        fs::write(
            &entry_path,
            "use lib.{ExportedStruct, exported_fn, ExportedEnum}\n",
        )
        .unwrap();

        let mut loader = ModuleLoader::new(dir.clone());
        let program = loader
            .load_program_from_entry(entry_path.to_str().unwrap())
            .unwrap();

        let lib_module = program.modules.iter().find(|m| m.path == "lib").unwrap();
        assert!(lib_module.exports.types.contains_key("ExportedStruct"));
        assert!(lib_module.exports.types.contains_key("ExportedEnum"));
        assert!(lib_module.exports.functions.contains_key("exported_fn"));
        assert!(!lib_module.exports.types.contains_key("PrivateStruct"));
        assert!(!lib_module.exports.types.contains_key("PrivateEnum"));
        assert!(!lib_module.exports.functions.contains_key("private_fn"));

        // Best-effort cleanup.
        let _ = fs::remove_file(entry_path);
        let _ = fs::remove_file(module_path);
        let _ = fs::remove_dir(dir);
    }

    #[test]
    fn test_compile_time_data_import_json() {
        let dir = unique_temp_dir("lust_data_import_test");
        fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("config.json");
        let entry_path = dir.join("main.lust");

        fs::write(&config_path, r#"{"name": "LustEngine", "port": 8080, "active": true}"#).unwrap();
        fs::write(
            &entry_path,
            r#"
import config from "./config.json"

function get_name(): string
    return config["name"]
end
"#,
        )
        .unwrap();

        let mut loader = ModuleLoader::new(dir.clone());
        let program = loader
            .load_program_from_entry(entry_path.to_str().unwrap())
            .expect("compile-time data import succeeds");

        assert_eq!(program.modules.len(), 1);
        let data_cache = loader.data_cache();
        assert_eq!(data_cache.len(), 1);

        let cached_entry = data_cache.get(&config_path).expect("config.json in data_cache");
        let first_digest = cached_entry.0;

        // Second load: unchanged file uses same digest without reparsing
        let mut loader2 = ModuleLoader::new(dir.clone());
        let val = loader2.load_data_import(&config_path).unwrap();
        assert_eq!(ModuleLoader::digest_bytes(&fs::read(&config_path).unwrap()), first_digest);
        if let crate::data::DataValue::Document(doc) = &val {
            assert_eq!(
                doc.get("name"),
                Some(&crate::data::DataValue::String("LustEngine".to_string()))
            );
        } else {
            panic!("expected document");
        }

        // Cleanup
        let _ = fs::remove_file(config_path);
        let _ = fs::remove_file(entry_path);
        let _ = fs::remove_dir(dir);
    }

    #[test]
    fn test_compile_time_data_import_missing_file_error() {
        let dir = unique_temp_dir("lust_data_missing_test");
        fs::create_dir_all(&dir).unwrap();
        let entry_path = dir.join("main.lust");

        fs::write(
            &entry_path,
            r#"
import config from "./non_existent.json"
"#,
        )
        .unwrap();

        let mut loader = ModuleLoader::new(dir.clone());
        let res = loader.load_program_from_entry(entry_path.to_str().unwrap());
        assert!(res.is_err(), "missing file must be a compile error");

        let _ = fs::remove_file(entry_path);
        let _ = fs::remove_dir(dir);
    }

    #[test]
    fn test_compile_time_data_import_invalid_json_error() {
        let dir = unique_temp_dir("lust_data_invalid_test");
        fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("bad.json");
        let entry_path = dir.join("main.lust");

        fs::write(&config_path, r#"{"name": "broken", invalid_syntax}"#).unwrap();
        fs::write(
            &entry_path,
            r#"
import config from "./bad.json"
"#,
        )
        .unwrap();

        let mut loader = ModuleLoader::new(dir.clone());
        let res = loader.load_program_from_entry(entry_path.to_str().unwrap());
        assert!(res.is_err(), "invalid JSON must be a compile error");

        let _ = fs::remove_file(config_path);
        let _ = fs::remove_file(entry_path);
        let _ = fs::remove_dir(dir);
    }

    #[test]
    fn test_compile_time_data_import_non_finite_number_error() {
        let dir = unique_temp_dir("lust_data_non_finite_test");
        fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("overflow.json");
        let entry_path = dir.join("main.lust");

        // 1e999 overflows to infinity; JSON parser and loader reject non-finite numbers
        fs::write(&config_path, r#"{"rate": 1e999}"#).unwrap();
        fs::write(
            &entry_path,
            r#"
import config from "./overflow.json"
"#,
        )
        .unwrap();

        let mut loader = ModuleLoader::new(dir.clone());
        let res = loader.load_program_from_entry(entry_path.to_str().unwrap());
        assert!(res.is_err(), "non-finite number must be a compile error");

        let _ = fs::remove_file(config_path);
        let _ = fs::remove_file(entry_path);
        let _ = fs::remove_dir(dir);
    }
}
