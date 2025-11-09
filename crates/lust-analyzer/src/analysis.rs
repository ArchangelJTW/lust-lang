use crate::semantic_tokens::collect_semantic_tokens_for_module;
use crate::utils::{
    method_display_name, named_type_name, qualify_type_name, simple_type_name,
    span_contains_position, span_size, span_to_range,
};
use hashbrown::{HashMap, HashSet};
use lust::ast::{EnumDef, FunctionParam, ItemKind, StructDef, TraitDef, Type, Visibility};
use lust::modules::{LoadedModule, Program};
use lust::{Span, TypeCollection};
use std::{
    fmt::Write,
    fs,
    path::{Path, PathBuf},
};
use tower_lsp::lsp_types::{Hover, HoverContents, Location, MarkupContent, MarkupKind, Position};
use url::Url;
#[derive(Clone)]
pub(crate) struct ModuleSnapshot {
    pub(crate) module: LoadedModule,
    pub(crate) expr_types: HashMap<Span, Type>,
    pub(crate) variable_types: HashMap<Span, Type>,
}

impl ModuleSnapshot {
    pub(crate) fn type_for_span(&self, span: &Span) -> Option<&Type> {
        self.expr_types
            .get(span)
            .or_else(|| self.variable_types.get(span))
    }
}

#[derive(Clone)]
pub(crate) struct TypeDefinition {
    pub(crate) name: String,
    pub(crate) qualified_name: String,
    pub(crate) span: Span,
    pub(crate) module_path: String,
    pub(crate) file_path: PathBuf,
    pub(crate) layout: String,
    pub(crate) kind: TypeDefinitionKind,
}

#[derive(Clone, Copy)]
pub(crate) enum TypeDefinitionKind {
    Struct,
    Enum,
    Trait,
}

#[derive(Clone, Default)]
pub(crate) struct SemanticTokenData {
    pub(crate) tokens: Vec<tower_lsp::lsp_types::SemanticToken>,
}

#[derive(Default)]
pub(crate) struct TypeIndex {
    by_simple: HashMap<String, Vec<TypeDefinition>>,
    by_qualified: HashMap<String, TypeDefinition>,
    by_file: HashMap<PathBuf, Vec<TypeDefinition>>,
}

#[derive(Clone)]
pub(crate) struct StructInfo {
    pub(crate) module_path: String,
    pub(crate) def: StructDef,
}

#[derive(Clone)]
pub(crate) struct EnumInfo {
    pub(crate) module_path: String,
    pub(crate) def: EnumDef,
}

#[derive(Clone)]
pub(crate) struct MethodInfo {
    pub(crate) owner: String,
    pub(crate) module_path: String,
    pub(crate) name: String,
    pub(crate) is_instance: bool,
    pub(crate) params: Vec<FunctionParam>,
    pub(crate) return_type: Option<Type>,
    pub(crate) visibility: Visibility,
}

pub(crate) struct AnalysisSnapshot {
    modules_by_path: HashMap<PathBuf, ModuleSnapshot>,
    modules_by_name: HashMap<String, PathBuf>,
    module_children: HashMap<String, HashSet<String>>,
    dependency_roots: HashSet<String>,
    project_module_roots: HashSet<String>,
    type_index: TypeIndex,
    semantic_tokens: HashMap<PathBuf, SemanticTokenData>,
    structs_by_qualified: HashMap<String, StructInfo>,
    structs_by_simple: HashMap<String, Vec<String>>,
    enums_by_qualified: HashMap<String, EnumInfo>,
    enums_by_simple: HashMap<String, Vec<String>>,
    methods_by_type: HashMap<String, Vec<MethodInfo>>,
}

impl AnalysisSnapshot {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        program: &Program,
        mut type_info: TypeCollection,
        source_overrides: &HashMap<PathBuf, String>,
        builtin_structs: HashMap<String, StructDef>,
        builtin_enums: HashMap<String, EnumDef>,
        dependency_roots: HashSet<String>,
    ) -> Self {
        let mut modules_by_path = HashMap::new();
        let mut modules_by_name = HashMap::new();
        let mut module_children: HashMap<String, HashSet<String>> = HashMap::new();
        let mut type_index = TypeIndex::default();
        let mut semantic_tokens = HashMap::new();
        let mut structs_by_qualified = HashMap::new();
        let mut structs_by_simple: HashMap<String, Vec<String>> = HashMap::new();
        let mut enums_by_qualified = HashMap::new();
        let mut enums_by_simple: HashMap<String, Vec<String>> = HashMap::new();
        let mut methods_by_type: HashMap<String, Vec<MethodInfo>> = HashMap::new();
        let mut entry_module_path: Option<PathBuf> = None;
        for module in &program.modules {
            let module_path = module.path.clone();
            let file_path = module.source_path.clone();
            modules_by_name.insert(module_path.clone(), file_path.clone());
            if module_path == program.entry_module {
                entry_module_path = Some(file_path.clone());
            }
            register_module_children(&mut module_children, &module_path);
            register_module_children_from_source(
                &mut module_children,
                &module_path,
                &file_path,
            );
            let source = source_overrides
                .get(&file_path)
                .cloned()
                .unwrap_or_else(|| fs::read_to_string(&file_path).unwrap_or_default());
            let expr_types = type_info
                .expr_types
                .remove(&module_path)
                .unwrap_or_default();
            let variable_types = type_info
                .variable_types
                .remove(&module_path)
                .unwrap_or_default();
            for item in &module.items {
                match &item.kind {
                    ItemKind::Struct(def) => {
                        let simple_name = simple_type_name(&def.name).to_string();
                        let qualified_name = qualify_type_name(&module_path, &def.name);
                        let info = StructInfo {
                            module_path: module_path.clone(),
                            def: def.clone(),
                        };
                        structs_by_simple
                            .entry(simple_name)
                            .or_default()
                            .push(qualified_name.clone());
                        structs_by_qualified.insert(qualified_name, info);
                    }

                    ItemKind::Enum(def) => {
                        let simple_name = simple_type_name(&def.name).to_string();
                        let qualified_name = qualify_type_name(&module_path, &def.name);
                        let info = EnumInfo {
                            module_path: module_path.clone(),
                            def: def.clone(),
                        };
                        enums_by_simple
                            .entry(simple_name)
                            .or_default()
                            .push(qualified_name.clone());
                        enums_by_qualified.insert(qualified_name, info);
                    }

                    ItemKind::Impl(impl_block) => {
                        if let Some(type_name) = named_type_name(&impl_block.target_type) {
                            let qualified_type = qualify_type_name(&module_path, &type_name);
                            let simple_owner = simple_type_name(&qualified_type).to_string();
                            for method in &impl_block.methods {
                                let is_instance =
                                    method.params.iter().any(|p| p.is_self || p.name == "self");
                                let method_name = method_display_name(&method.name);
                                let info = MethodInfo {
                                    owner: qualified_type.clone(),
                                    module_path: module_path.clone(),
                                    name: method_name,
                                    is_instance,
                                    params: method.params.clone(),
                                    return_type: method.return_type.clone(),
                                    visibility: method.visibility,
                                };
                                methods_by_type
                                    .entry(qualified_type.clone())
                                    .or_default()
                                    .push(info.clone());
                                methods_by_type
                                    .entry(simple_owner.clone())
                                    .or_default()
                                    .push(info);
                            }
                        }
                    }

                    ItemKind::Function(func) => {
                        if let Some((type_name, method_name, is_instance)) =
                            crate::utils::split_type_member(&func.name)
                        {
                            let qualified_type = qualify_type_name(&module_path, &type_name);
                            let simple_owner = simple_type_name(&qualified_type).to_string();
                            let info = MethodInfo {
                                owner: qualified_type.clone(),
                                module_path: module_path.clone(),
                                name: method_name,
                                is_instance,
                                params: func.params.clone(),
                                return_type: func.return_type.clone(),
                                visibility: func.visibility,
                            };
                            methods_by_type
                                .entry(qualified_type.clone())
                                .or_default()
                                .push(info.clone());
                            methods_by_type.entry(simple_owner).or_default().push(info);
                        }
                    }

                    _ => {}
                }
            }

            let module_clone = module.clone();
            type_index.add_module(&module_clone, &file_path);
            let semantic = collect_semantic_tokens_for_module(&module_clone, &expr_types, &source);
            semantic_tokens.insert(file_path.clone(), SemanticTokenData { tokens: semantic });
            modules_by_path.insert(
                file_path.clone(),
                ModuleSnapshot {
                    module: module_clone,
                    expr_types,
                    variable_types,
                },
            );
        }

        for (qualified, def) in builtin_structs {
            if structs_by_qualified.contains_key(&qualified) {
                continue;
            }

            let simple_name = simple_type_name(&qualified).to_string();
            let entry = structs_by_simple.entry(simple_name).or_default();
            if !entry.contains(&qualified) {
                entry.push(qualified.clone());
            }

            if type_index.lookup_qualified(&qualified).is_none() {
                type_index.insert(TypeDefinition::from_builtin_struct(&qualified, &def));
            }

            structs_by_qualified.insert(
                qualified,
                StructInfo {
                    module_path: String::new(),
                    def,
                },
            );
        }

        for (qualified, def) in builtin_enums {
            if enums_by_qualified.contains_key(&qualified) {
                continue;
            }

            let simple_name = simple_type_name(&qualified).to_string();
            let entry = enums_by_simple.entry(simple_name).or_default();
            if !entry.contains(&qualified) {
                entry.push(qualified.clone());
            }

            if type_index.lookup_qualified(&qualified).is_none() {
                type_index.insert(TypeDefinition::from_builtin_enum(&qualified, &def));
            }

            enums_by_qualified.insert(
                qualified,
                EnumInfo {
                    module_path: String::new(),
                    def,
                },
            );
        }
        if !dependency_roots.is_empty() {
            module_children
                .entry(String::new())
                .or_default()
                .extend(dependency_roots.iter().cloned());
        }
        let project_module_roots = entry_module_path
            .as_deref()
            .and_then(|path| path.parent())
            .map(|root| collect_project_module_roots(root))
            .unwrap_or_default();
        if !project_module_roots.is_empty() {
            module_children
                .entry(String::new())
                .or_default()
                .extend(project_module_roots.iter().cloned());
        }

        Self {
            modules_by_path,
            modules_by_name,
            module_children,
            dependency_roots,
            project_module_roots,
            type_index,
            semantic_tokens,
            structs_by_qualified,
            structs_by_simple,
            enums_by_qualified,
            enums_by_simple,
            methods_by_type,
        }
    }

    pub(crate) fn module_for_file(&self, path: &Path) -> Option<&ModuleSnapshot> {
        self.modules_by_path.get(path)
    }

    pub(crate) fn module_for_name(&self, name: &str) -> Option<&ModuleSnapshot> {
        self.modules_by_name
            .get(name)
            .and_then(|path| self.modules_by_path.get(path))
    }

    pub(crate) fn module_children(&self, name: &str) -> Option<&HashSet<String>> {
        self.module_children.get(name)
    }

    pub(crate) fn dependency_roots(&self) -> impl Iterator<Item = &String> {
        self.dependency_roots.iter()
    }

    pub(crate) fn has_dependency_root(&self, name: &str) -> bool {
        self.dependency_roots.contains(name)
    }

    pub(crate) fn project_module_roots(&self) -> impl Iterator<Item = &String> {
        self.project_module_roots.iter()
    }

    pub(crate) fn has_struct(&self, qualified: &str) -> bool {
        self.structs_by_qualified.contains_key(qualified)
    }

    pub(crate) fn has_enum(&self, qualified: &str) -> bool {
        self.enums_by_qualified.contains_key(qualified)
    }

    pub(crate) fn module_path_for_file(&self, path: &Path) -> Option<&str> {
        self.modules_by_path
            .get(path)
            .map(|snapshot| snapshot.module.path.as_str())
    }

    pub(crate) fn definitions_by_simple(&self, name: &str) -> Option<&[TypeDefinition]> {
        self.type_index.lookup_simple(name)
    }

    pub(crate) fn definition_by_qualified(&self, name: &str) -> Option<&TypeDefinition> {
        self.type_index.lookup_qualified(name)
    }

    pub(crate) fn definitions_in_file(&self, path: &Path) -> Option<&[TypeDefinition]> {
        self.type_index.definitions_in_file(path)
    }

    pub(crate) fn all_type_definitions(&self) -> impl Iterator<Item = &TypeDefinition> {
        self.type_index.all_definitions()
    }

    pub(crate) fn struct_info_for(
        &self,
        type_name: &str,
        module_path: Option<&str>,
    ) -> Option<&StructInfo> {
        if let Some(info) = self.structs_by_qualified.get(type_name) {
            return Some(info);
        }

        let simple = simple_type_name(type_name);
        if let Some(candidates) = self.structs_by_simple.get(simple) {
            if let Some(module) = module_path {
                if let Some(qualified) = candidates.iter().find(|qualified| {
                    self.structs_by_qualified
                        .get(*qualified)
                        .map(|info| info.module_path == module)
                        .unwrap_or(false)
                }) {
                    return self.structs_by_qualified.get(qualified);
                }
            }

            for qualified in candidates {
                if let Some(info) = self.structs_by_qualified.get(qualified) {
                    return Some(info);
                }
            }
        }

        None
    }

    pub(crate) fn enum_info_for(
        &self,
        type_name: &str,
        module_path: Option<&str>,
    ) -> Option<&EnumInfo> {
        if let Some(info) = self.enums_by_qualified.get(type_name) {
            return Some(info);
        }

        let simple = simple_type_name(type_name);
        if let Some(candidates) = self.enums_by_simple.get(simple) {
            if let Some(module) = module_path {
                if let Some(qualified) = candidates.iter().find(|qualified| {
                    self.enums_by_qualified
                        .get(*qualified)
                        .map(|info| info.module_path == module)
                        .unwrap_or(false)
                }) {
                    return self.enums_by_qualified.get(qualified);
                }
            }

            for qualified in candidates {
                if let Some(info) = self.enums_by_qualified.get(qualified) {
                    return Some(info);
                }
            }
        }

        None
    }

    pub(crate) fn methods_for_type(&self, type_name: &str) -> Option<&[MethodInfo]> {
        if let Some(list) = self.methods_by_type.get(type_name) {
            return Some(list.as_slice());
        }

        let simple = simple_type_name(type_name);
        self.methods_by_type.get(simple).map(|list| list.as_slice())
    }

    pub(crate) fn semantic_tokens_for_path(
        &self,
        path: &Path,
    ) -> Option<Vec<tower_lsp::lsp_types::SemanticToken>> {
        self.semantic_tokens
            .get(path)
            .map(|data| data.tokens.clone())
    }
}

fn register_module_children(map: &mut HashMap<String, HashSet<String>>, module_path: &str) {
    if module_path.is_empty() {
        return;
    }

    let segments: Vec<&str> = module_path.split('.').collect();
    for idx in 0..segments.len() {
        let parent = segments[..idx].join(".");
        let child = segments[idx].to_string();
        map.entry(parent).or_default().insert(child);
    }
}

fn register_module_children_from_source(
    map: &mut HashMap<String, HashSet<String>>,
    module_path: &str,
    source_path: &Path,
) {
    let Some(parent_dir) = source_path.parent() else { return; };
    let Some(stem) = source_path.file_stem().and_then(|s| s.to_str()) else { return; };
    if stem.is_empty() {
        return;
    }

    let segments: Vec<&str> = module_path.split('.').filter(|s| !s.is_empty()).collect();
    let last_segment = segments.last().copied().unwrap_or("");

    let mut register_child = |name: &str| {
        if name.is_empty() || name == stem || !is_valid_module_name(name) {
            return;
        }
        map.entry(module_path.to_string())
            .or_default()
            .insert(name.to_string());
    };

    let scan_lust_files = |dir: &Path, include_dirs: bool, register: &mut dyn FnMut(&str)| {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let Ok(file_type) = entry.file_type() else {
                    continue;
                };
                let name_os = entry.file_name();
                let Some(name) = name_os.to_str() else {
                    continue;
                };
                if should_skip_entry(name) {
                    continue;
                }
                if file_type.is_file() {
                    if entry.path().extension().and_then(|ext| ext.to_str()) != Some("lust") {
                        continue;
                    }
                    if let Some(child) = entry.path().file_stem().and_then(|s| s.to_str()) {
                        register(child);
                    }
                } else if include_dirs && file_type.is_dir() {
                    let candidate = entry.path().join(format!("{name}.lust"));
                    let mod_candidate = entry.path().join("mod.lust");
                    if candidate.exists() || mod_candidate.exists() {
                        register(name);
                    }
                }
            }
        }
    };

    let module_dir = parent_dir.join(stem);
    if module_dir.is_dir() {
        scan_lust_files(&module_dir, true, &mut register_child);
    }

    if !last_segment.is_empty() {
        if let Some(parent_name) = parent_dir.file_name().and_then(|s| s.to_str()) {
            if parent_name == last_segment {
                scan_lust_files(parent_dir, true, &mut register_child);
            }
        }
    }
}

fn is_valid_module_name(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return false;
    }
    chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn collect_project_module_roots(root_dir: &Path) -> HashSet<String> {
    let mut roots = HashSet::new();
    let Ok(entries) = fs::read_dir(root_dir) else {
        return roots;
    };
    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        let name_os = entry.file_name();
        let Some(name) = name_os.to_str() else {
            continue;
        };
        if should_skip_entry(name) {
            continue;
        }
        let path = entry.path();
        if file_type.is_file() {
            if path.extension().and_then(|ext| ext.to_str()) == Some("lust") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    roots.insert(stem.to_string());
                }
            }
        } else if file_type.is_dir() && directory_contains_lust(&path) {
            roots.insert(name.to_string());
        }
    }
    roots
}

fn directory_contains_lust(path: &Path) -> bool {
    let mut stack = vec![PathBuf::from(path)];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            let name_os = entry.file_name();
            let Some(name) = name_os.to_str() else {
                continue;
            };
            if should_skip_entry(name) {
                continue;
            }
            let entry_path = entry.path();
            if file_type.is_file() {
                if entry_path.extension().and_then(|ext| ext.to_str()) == Some("lust") {
                    return true;
                }
            } else if file_type.is_dir() {
                stack.push(entry_path);
            }
        }
    }
    false
}

fn should_skip_entry(name: &str) -> bool {
    matches!(name, "." | "..")
        || name.starts_with('.')
        || matches!(name, "target" | "node_modules" | "__pycache__")
}

impl TypeIndex {
    fn add_module(&mut self, module: &LoadedModule, file_path: &Path) {
        for item in &module.items {
            match &item.kind {
                ItemKind::Struct(def) => {
                    let def = TypeDefinition::from_struct(&module.path, file_path, item.span, def);
                    self.insert(def);
                }

                ItemKind::Enum(def) => {
                    let def = TypeDefinition::from_enum(&module.path, file_path, item.span, def);
                    self.insert(def);
                }

                ItemKind::Trait(def) => {
                    let def = TypeDefinition::from_trait(&module.path, file_path, item.span, def);
                    self.insert(def);
                }

                _ => {}
            }
        }
    }

    fn insert(&mut self, def: TypeDefinition) {
        self.by_simple
            .entry(def.name.clone())
            .or_insert_with(Vec::new)
            .push(def.clone());
        self.by_qualified
            .insert(def.qualified_name.clone(), def.clone());
        self.by_file
            .entry(def.file_path.clone())
            .or_insert_with(Vec::new)
            .push(def);
    }

    fn lookup_simple(&self, name: &str) -> Option<&[TypeDefinition]> {
        self.by_simple.get(name).map(|defs| defs.as_slice())
    }

    fn lookup_qualified(&self, name: &str) -> Option<&TypeDefinition> {
        self.by_qualified.get(name)
    }

    fn definitions_in_file(&self, path: &Path) -> Option<&[TypeDefinition]> {
        self.by_file.get(path).map(|defs| defs.as_slice())
    }

    fn all_definitions(&self) -> impl Iterator<Item = &TypeDefinition> {
        self.by_qualified.values()
    }
}

impl TypeDefinition {
    fn from_struct(module_path: &str, file_path: &Path, span: Span, def: &StructDef) -> Self {
        let simple_name = simple_type_name(&def.name).to_string();
        let qualified_name = qualify_type_name(module_path, &def.name);
        Self {
            name: simple_name.clone(),
            qualified_name,
            span,
            module_path: module_path.to_string(),
            file_path: file_path.to_path_buf(),
            layout: format_struct_layout(&simple_name, def),
            kind: TypeDefinitionKind::Struct,
        }
    }

    fn from_enum(module_path: &str, file_path: &Path, span: Span, def: &EnumDef) -> Self {
        let simple_name = simple_type_name(&def.name).to_string();
        let qualified_name = qualify_type_name(module_path, &def.name);
        Self {
            name: simple_name.clone(),
            qualified_name,
            span,
            module_path: module_path.to_string(),
            file_path: file_path.to_path_buf(),
            layout: format_enum_layout(&simple_name, def),
            kind: TypeDefinitionKind::Enum,
        }
    }

    fn from_trait(module_path: &str, file_path: &Path, span: Span, def: &TraitDef) -> Self {
        let simple_name = simple_type_name(&def.name).to_string();
        let qualified_name = qualify_type_name(module_path, &def.name);
        Self {
            name: simple_name.clone(),
            qualified_name,
            span,
            module_path: module_path.to_string(),
            file_path: file_path.to_path_buf(),
            layout: format_trait_layout(&simple_name, def),
            kind: TypeDefinitionKind::Trait,
        }
    }

    fn from_builtin_struct(qualified_name: &str, def: &StructDef) -> Self {
        let simple_name = simple_type_name(qualified_name).to_string();
        Self {
            name: simple_name.clone(),
            qualified_name: qualified_name.to_string(),
            span: Span::new(0, 0, 0, 0),
            module_path: String::new(),
            file_path: PathBuf::new(),
            layout: format_struct_layout(&simple_name, def),
            kind: TypeDefinitionKind::Struct,
        }
    }

    fn from_builtin_enum(qualified_name: &str, def: &EnumDef) -> Self {
        let simple_name = simple_type_name(qualified_name).to_string();
        Self {
            name: simple_name.clone(),
            qualified_name: qualified_name.to_string(),
            span: Span::new(0, 0, 0, 0),
            module_path: String::new(),
            file_path: PathBuf::new(),
            layout: format_enum_layout(&simple_name, def),
            kind: TypeDefinitionKind::Enum,
        }
    }
}

fn format_type_params(params: &[String]) -> String {
    if params.is_empty() {
        String::new()
    } else {
        format!("<{}>", params.join(", "))
    }
}

fn format_struct_layout(simple_name: &str, def: &StructDef) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "struct {}{}",
        simple_name,
        format_type_params(&def.type_params)
    );
    if def.fields.is_empty() {
        let _ = writeln!(out, "end");
        return out;
    }

    for field in &def.fields {
        let _ = writeln!(out, "  {}: {}", field.name, field.ty);
    }

    let _ = writeln!(out, "end");
    out
}

fn format_enum_layout(simple_name: &str, def: &EnumDef) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "enum {}{}",
        simple_name,
        format_type_params(&def.type_params)
    );
    if def.variants.is_empty() {
        let _ = writeln!(out, "end");
        return out;
    }

    for variant in &def.variants {
        match &variant.fields {
            Some(fields) if !fields.is_empty() => {
                let args = fields
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                let _ = writeln!(out, "  {}({})", variant.name, args);
            }

            _ => {
                let _ = writeln!(out, "  {}", variant.name);
            }
        }
    }

    let _ = writeln!(out, "end");
    out
}

fn format_trait_layout(simple_name: &str, def: &TraitDef) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "trait {}{}",
        simple_name,
        format_type_params(&def.type_params)
    );
    if def.methods.is_empty() {
        let _ = writeln!(out, "end");
        return out;
    }

    for method in &def.methods {
        let method_name = simple_type_name(&method.name);
        let mut signature = String::new();
        let _ = write!(
            signature,
            "fn {}{}(",
            method_name,
            format_type_params(&method.type_params)
        );
        let params = method
            .params
            .iter()
            .map(format_trait_param)
            .collect::<Vec<_>>()
            .join(", ");
        let _ = write!(signature, "{})", params);
        if let Some(ret) = &method.return_type {
            let _ = write!(signature, " -> {}", ret);
        }

        let _ = writeln!(out, "  {}", signature);
    }

    let _ = writeln!(out, "end");
    out
}

fn format_trait_param(param: &FunctionParam) -> String {
    if param.is_self {
        "self".to_string()
    } else {
        format!("{}: {}", param.name, param.ty)
    }
}

pub(crate) fn choose_definition<'a>(
    defs: &'a [TypeDefinition],
    module_path: Option<&str>,
) -> Option<&'a TypeDefinition> {
    if defs.is_empty() {
        return None;
    }

    if let Some(module_path) = module_path {
        if let Some(def) = defs.iter().find(|d| d.module_path == module_path) {
            return Some(def);
        }
    }

    Some(&defs[0])
}

pub(crate) fn find_type_for_position(
    module: &ModuleSnapshot,
    position: Position,
) -> Option<(Span, Type)> {
    find_type_in_map(&module.expr_types, &position)
        .or_else(|| find_type_in_map(&module.variable_types, &position))
}

pub(crate) fn find_type_in_map(
    map: &HashMap<Span, Type>,
    position: &Position,
) -> Option<(Span, Type)> {
    let mut best: Option<(Span, Type, (usize, usize))> = None;
    for (span, ty) in map {
        if span.start_line == 0 {
            continue;
        }

        if span_contains_position(*span, position) {
            let size = span_size(*span);
            let replace = match &best {
                Some((_, _, best_size)) => size < *best_size,
                None => true,
            };
            if replace {
                best = Some((*span, ty.clone(), size));
            }
        }
    }

    best.map(|(span, ty, _)| (span, ty))
}

pub(crate) fn hover_from_definition(def: &TypeDefinition) -> Hover {
    let layout = def.layout.trim_end().to_string();
    let mut body = format!("```lust\n{layout}\n```");
    if def.qualified_name != def.name {
        body.push_str(&format!("\n`{}`", def.qualified_name));
    }

    Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: body,
        }),
        range: Some(span_to_range(def.span)),
    }
}

pub(crate) fn location_from_definition(def: &TypeDefinition) -> Option<Location> {
    let uri = Url::from_file_path(&def.file_path).ok()?;
    Some(Location {
        uri,
        range: span_to_range(def.span),
    })
}
