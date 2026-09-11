use crate::semantic_tokens::collect_semantic_tokens_for_module;
use crate::utils::{
    base_type_name, compute_line_offsets, find_identifier_in_source, find_identifier_in_span,
    method_display_name, named_type_name, qualify_type_name, range_contains_position,
    simple_type_name, span_contains_position, span_size, span_to_range,
};
use hashbrown::{HashMap, HashSet};
use lust::ast::{
    EnumDef, Expr, ExprKind, FunctionDef, FunctionParam, Item, ItemKind, Pattern, Stmt, StmtKind,
    StructDef, TraitDef, Type, TypeKind, UseTree, Visibility,
};
use lust::modules::{LoadedModule, ModuleImports, Program};
use lust::{Span, TypeCollection};
use std::{
    fmt::Write,
    fs,
    path::{Path, PathBuf},
};
use tower_lsp::lsp_types::{
    Hover, HoverContents, Location, MarkupContent, MarkupKind, Position, Range,
};
use url::Url;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum SymbolId {
    Struct(String),
    Enum(String),
    Variant {
        enum_qualified: String,
        variant_name: String,
    },
    Field {
        struct_qualified: String,
        field_name: String,
    },
    Trait(String),
    Function(String),
    Method {
        owner_qualified: String,
        method_name: String,
    },
    Local {
        file: PathBuf,
        def_span: Span,
        name: String,
    },
}

impl SymbolId {
    pub(crate) fn display_name(&self) -> &str {
        match self {
            SymbolId::Struct(s) => simple_type_name(s),
            SymbolId::Enum(e) => simple_type_name(e),
            SymbolId::Variant { variant_name, .. } => variant_name.as_str(),
            SymbolId::Field { field_name, .. } => field_name.as_str(),
            SymbolId::Trait(t) => simple_type_name(t),
            SymbolId::Function(f) => simple_type_name(f),
            SymbolId::Method { method_name, .. } => method_name.as_str(),
            SymbolId::Local { name, .. } => name.as_str(),
        }
    }

    pub(crate) fn is_renameable(&self) -> bool {
        match self {
            SymbolId::Struct(s) | SymbolId::Enum(s) | SymbolId::Trait(s) => {
                let simple = simple_type_name(s);
                !matches!(
                    simple,
                    "int"
                        | "float"
                        | "string"
                        | "bool"
                        | "Option"
                        | "Result"
                        | "Array"
                        | "Map"
                        | "Tuple"
                        | "unknown"
                )
            }
            SymbolId::Function(f) => {
                let simple = simple_type_name(f);
                !matches!(
                    simple,
                    "print" | "println" | "assert" | "panic" | "type_of" | "to_string"
                )
            }
            _ => true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SymbolOccurrence {
    pub(crate) symbol: SymbolId,
    pub(crate) span: Span,
    pub(crate) range: Range,
    pub(crate) file_path: PathBuf,
    pub(crate) is_definition: bool,
}

#[derive(Default)]
pub(crate) struct SymbolIndex {
    pub(crate) occurrences_by_file: HashMap<PathBuf, Vec<SymbolOccurrence>>,
    pub(crate) occurrences_by_symbol: HashMap<SymbolId, Vec<SymbolOccurrence>>,
    pub(crate) definitions_by_symbol: HashMap<SymbolId, SymbolOccurrence>,
}

impl SymbolIndex {
    pub(crate) fn add_occurrence(&mut self, occ: SymbolOccurrence) {
        if let Some(list) = self.occurrences_by_file.get_mut(&occ.file_path) {
            if list
                .iter()
                .any(|existing| existing.span == occ.span && existing.symbol == occ.symbol)
            {
                return;
            }
        }
        if occ.is_definition {
            self.definitions_by_symbol
                .entry(occ.symbol.clone())
                .or_insert_with(|| occ.clone());
        }
        self.occurrences_by_symbol
            .entry(occ.symbol.clone())
            .or_default()
            .push(occ.clone());
        self.occurrences_by_file
            .entry(occ.file_path.clone())
            .or_default()
            .push(occ);
    }

    pub(crate) fn find_symbol_at_position(
        &self,
        file_path: &Path,
        position: &Position,
    ) -> Option<&SymbolOccurrence> {
        let list = self.occurrences_by_file.get(file_path)?;
        let mut best: Option<(&SymbolOccurrence, u32)> = None;
        for occ in list {
            if range_contains_position(&occ.range, position) {
                let len = occ
                    .range
                    .end
                    .character
                    .saturating_sub(occ.range.start.character);
                let replace = match best {
                    Some((_, best_len)) => len < best_len,
                    None => true,
                };
                if replace {
                    best = Some((occ, len));
                }
            }
        }
        best.map(|(occ, _)| occ)
    }

    pub(crate) fn symbol_references(&self, symbol: &SymbolId) -> &[SymbolOccurrence] {
        self.occurrences_by_symbol
            .get(symbol)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub(crate) fn symbol_definition(&self, symbol: &SymbolId) -> Option<&SymbolOccurrence> {
        self.definitions_by_symbol.get(symbol)
    }
}

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
    pub(crate) span: Span,
    pub(crate) file_path: PathBuf,
}

#[derive(Clone)]
pub(crate) struct EnumInfo {
    pub(crate) module_path: String,
    pub(crate) def: EnumDef,
    pub(crate) span: Span,
    pub(crate) file_path: PathBuf,
}

#[derive(Clone)]
pub(crate) struct TraitInfo {
    pub(crate) module_path: String,
    pub(crate) def: TraitDef,
    pub(crate) span: Span,
    pub(crate) file_path: PathBuf,
}

#[derive(Clone)]
pub(crate) struct FunctionInfo {
    pub(crate) module_path: String,
    pub(crate) name: String,
    pub(crate) def: FunctionDef,
    pub(crate) span: Span,
    pub(crate) file_path: PathBuf,
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
    pub(crate) span: Span,
    pub(crate) file_path: PathBuf,
}

pub(crate) struct AnalysisSnapshot {
    modules_by_path: HashMap<PathBuf, ModuleSnapshot>,
    modules_by_name: HashMap<String, PathBuf>,
    module_children: HashMap<String, HashSet<String>>,
    dependency_roots: HashSet<String>,
    #[allow(dead_code)]
    project_module_roots: HashSet<String>,
    type_index: TypeIndex,
    semantic_tokens: HashMap<PathBuf, SemanticTokenData>,
    structs_by_qualified: HashMap<String, StructInfo>,
    structs_by_simple: HashMap<String, Vec<String>>,
    enums_by_qualified: HashMap<String, EnumInfo>,
    enums_by_simple: HashMap<String, Vec<String>>,
    traits_by_qualified: HashMap<String, TraitInfo>,
    #[allow(dead_code)]
    traits_by_simple: HashMap<String, Vec<String>>,
    functions_by_qualified: HashMap<String, FunctionInfo>,
    functions_by_simple: HashMap<String, Vec<String>>,
    methods_by_type: HashMap<String, Vec<MethodInfo>>,
    symbol_index: SymbolIndex,
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
        let mut traits_by_qualified = HashMap::new();
        let mut traits_by_simple: HashMap<String, Vec<String>> = HashMap::new();
        let mut functions_by_qualified = HashMap::new();
        let mut functions_by_simple: HashMap<String, Vec<String>> = HashMap::new();
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
            register_module_children_from_source(&mut module_children, &module_path, &file_path);
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
                            span: item.span,
                            file_path: file_path.clone(),
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
                            span: item.span,
                            file_path: file_path.clone(),
                        };
                        enums_by_simple
                            .entry(simple_name)
                            .or_default()
                            .push(qualified_name.clone());
                        enums_by_qualified.insert(qualified_name, info);
                    }

                    ItemKind::Trait(def) => {
                        let simple_name = simple_type_name(&def.name).to_string();
                        let qualified_name = qualify_type_name(&module_path, &def.name);
                        let info = TraitInfo {
                            module_path: module_path.clone(),
                            def: def.clone(),
                            span: item.span,
                            file_path: file_path.clone(),
                        };
                        traits_by_simple
                            .entry(simple_name)
                            .or_default()
                            .push(qualified_name.clone());
                        traits_by_qualified.insert(qualified_name, info);
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
                                    span: item.span,
                                    file_path: file_path.clone(),
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
                        let unadorned = func
                            .name
                            .strip_prefix(&format!("{module_path}."))
                            .unwrap_or(&func.name);
                        let method_info = if let Some(pos) = unadorned.rfind(':') {
                            Some((&unadorned[..pos], &unadorned[pos + 1..], true))
                        } else if let Some(pos) = unadorned.rfind('.') {
                            let type_part = &unadorned[..pos];
                            let method_part = &unadorned[pos + 1..];
                            let qtype = qualify_type_name(&module_path, type_part);
                            if structs_by_qualified.contains_key(&qtype)
                                || enums_by_qualified.contains_key(&qtype)
                                || structs_by_simple.contains_key(type_part)
                                || enums_by_simple.contains_key(type_part)
                            {
                                Some((type_part, method_part, false))
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                        if let Some((type_name, method_name, is_instance)) = method_info {
                            let qualified_type = qualify_type_name(&module_path, type_name);
                            let simple_owner = simple_type_name(&qualified_type).to_string();
                            let info = MethodInfo {
                                owner: qualified_type.clone(),
                                module_path: module_path.clone(),
                                name: method_name.to_string(),
                                is_instance,
                                params: func.params.clone(),
                                return_type: func.return_type.clone(),
                                visibility: func.visibility,
                                span: item.span,
                                file_path: file_path.clone(),
                            };
                            methods_by_type
                                .entry(qualified_type.clone())
                                .or_default()
                                .push(info.clone());
                            methods_by_type.entry(simple_owner).or_default().push(info);
                        } else {
                            let simple_name = simple_type_name(unadorned).to_string();
                            let qualified_name = qualify_type_name(&module_path, &simple_name);
                            let info = FunctionInfo {
                                module_path: module_path.clone(),
                                name: simple_name.clone(),
                                def: func.clone(),
                                span: item.span,
                                file_path: file_path.clone(),
                            };
                            functions_by_simple
                                .entry(simple_name)
                                .or_default()
                                .push(qualified_name.clone());
                            functions_by_qualified.insert(qualified_name, info);
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
                    span: Span::new(0, 0, 0, 0),
                    file_path: PathBuf::new(),
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
                    span: Span::new(0, 0, 0, 0),
                    file_path: PathBuf::new(),
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

        let mut symbol_index = SymbolIndex::default();
        for module in &program.modules {
            let file_path = &module.source_path;
            let source = source_overrides
                .get(file_path)
                .cloned()
                .unwrap_or_else(|| fs::read_to_string(file_path).unwrap_or_default());
            let line_offsets = compute_line_offsets(&source);
            let empty_map = HashMap::new();
            let expr_types = modules_by_path
                .get(file_path)
                .map(|m| &m.expr_types)
                .unwrap_or(&empty_map);
            let variable_types = modules_by_path
                .get(file_path)
                .map(|m| &m.variable_types)
                .unwrap_or(&empty_map);

            let mut indexer = IndexerContext {
                snapshot_structs: &structs_by_qualified,
                structs_by_simple: &structs_by_simple,
                snapshot_enums: &enums_by_qualified,
                enums_by_simple: &enums_by_simple,
                snapshot_traits: &traits_by_qualified,
                traits_by_simple: &traits_by_simple,
                snapshot_functions: &functions_by_qualified,
                functions_by_simple: &functions_by_simple,
                methods_by_type: &methods_by_type,
                expr_types,
                variable_types,
                module_path: &module.path,
                file_path,
                source: &source,
                line_offsets: &line_offsets,
                imports: &module.imports,
                scopes: Vec::new(),
                index: &mut symbol_index,
            };
            indexer.index_items(&module.items);
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
            traits_by_qualified,
            traits_by_simple,
            functions_by_qualified,
            functions_by_simple,
            methods_by_type,
            symbol_index,
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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    pub(crate) fn symbol_index(&self) -> &SymbolIndex {
        &self.symbol_index
    }

    pub(crate) fn find_symbol_at_position(
        &self,
        file_path: &Path,
        position: &Position,
    ) -> Option<&SymbolOccurrence> {
        self.symbol_index
            .find_symbol_at_position(file_path, position)
    }

    pub(crate) fn symbol_references(&self, symbol: &SymbolId) -> &[SymbolOccurrence] {
        self.symbol_index.symbol_references(symbol)
    }

    pub(crate) fn symbol_definition(&self, symbol: &SymbolId) -> Option<&SymbolOccurrence> {
        self.symbol_index.symbol_definition(symbol)
    }

    pub(crate) fn function_info_for(
        &self,
        func_name: &str,
        module_path: Option<&str>,
    ) -> Option<&FunctionInfo> {
        if let Some(info) = self.functions_by_qualified.get(func_name) {
            return Some(info);
        }
        let simple = simple_type_name(func_name);
        if let Some(candidates) = self.functions_by_simple.get(simple) {
            if let Some(module) = module_path {
                if let Some(qualified) = candidates.iter().find(|qualified| {
                    self.functions_by_qualified
                        .get(*qualified)
                        .map(|info| info.module_path == module)
                        .unwrap_or(false)
                }) {
                    return self.functions_by_qualified.get(qualified);
                }
            }
            for qualified in candidates {
                if let Some(info) = self.functions_by_qualified.get(qualified) {
                    return Some(info);
                }
            }
        }
        None
    }

    #[allow(dead_code)]
    pub(crate) fn trait_info_for(
        &self,
        trait_name: &str,
        module_path: Option<&str>,
    ) -> Option<&TraitInfo> {
        if let Some(info) = self.traits_by_qualified.get(trait_name) {
            return Some(info);
        }
        let simple = simple_type_name(trait_name);
        if let Some(candidates) = self.traits_by_simple.get(simple) {
            if let Some(module) = module_path {
                if let Some(qualified) = candidates.iter().find(|qualified| {
                    self.traits_by_qualified
                        .get(*qualified)
                        .map(|info| info.module_path == module)
                        .unwrap_or(false)
                }) {
                    return self.traits_by_qualified.get(qualified);
                }
            }
            for qualified in candidates {
                if let Some(info) = self.traits_by_qualified.get(qualified) {
                    return Some(info);
                }
            }
        }
        None
    }

    pub(crate) fn all_structs(&self) -> impl Iterator<Item = &StructInfo> {
        self.structs_by_qualified.values()
    }

    pub(crate) fn all_enums(&self) -> impl Iterator<Item = &EnumInfo> {
        self.enums_by_qualified.values()
    }

    pub(crate) fn all_traits(&self) -> impl Iterator<Item = &TraitInfo> {
        self.traits_by_qualified.values()
    }

    pub(crate) fn all_functions(&self) -> impl Iterator<Item = &FunctionInfo> {
        self.functions_by_qualified.values()
    }

    pub(crate) fn all_methods(&self) -> impl Iterator<Item = &MethodInfo> {
        self.methods_by_type.values().flat_map(|v| v.iter())
    }
}

struct IndexerContext<'a> {
    snapshot_structs: &'a HashMap<String, StructInfo>,
    structs_by_simple: &'a HashMap<String, Vec<String>>,
    snapshot_enums: &'a HashMap<String, EnumInfo>,
    enums_by_simple: &'a HashMap<String, Vec<String>>,
    snapshot_traits: &'a HashMap<String, TraitInfo>,
    traits_by_simple: &'a HashMap<String, Vec<String>>,
    snapshot_functions: &'a HashMap<String, FunctionInfo>,
    functions_by_simple: &'a HashMap<String, Vec<String>>,
    #[allow(dead_code)]
    methods_by_type: &'a HashMap<String, Vec<MethodInfo>>,
    expr_types: &'a HashMap<Span, Type>,
    variable_types: &'a HashMap<Span, Type>,
    module_path: &'a str,
    file_path: &'a Path,
    source: &'a str,
    line_offsets: &'a [usize],
    imports: &'a ModuleImports,
    scopes: Vec<HashMap<String, (SymbolId, Option<Type>)>>,
    index: &'a mut SymbolIndex,
}

impl<'a> IndexerContext<'a> {
    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn define_local(&mut self, name: &str, span: Span, range: Range, ty: Option<Type>) {
        let sym = SymbolId::Local {
            file: self.file_path.to_path_buf(),
            def_span: span,
            name: name.to_string(),
        };
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), (sym.clone(), ty));
        }
        self.index.add_occurrence(SymbolOccurrence {
            symbol: sym,
            span,
            range,
            file_path: self.file_path.to_path_buf(),
            is_definition: true,
        });
    }

    fn lookup_local(&self, name: &str) -> Option<SymbolId> {
        for scope in self.scopes.iter().rev() {
            if let Some((sym, _)) = scope.get(name) {
                return Some(sym.clone());
            }
        }
        None
    }

    fn lookup_local_type(&self, name: &str) -> Option<Type> {
        for scope in self.scopes.iter().rev() {
            if let Some((_, Some(ty))) = scope.get(name) {
                return Some(ty.clone());
            }
        }
        None
    }

    fn type_for_expr(&self, expr: &Expr) -> Option<Type> {
        if let ExprKind::Identifier(name) = &expr.kind {
            if let Some(ty) = self.lookup_local_type(name) {
                return Some(ty);
            }
        }
        self.expr_types
            .get(&expr.span)
            .cloned()
            .or_else(|| self.variable_types.get(&expr.span).cloned())
    }

    fn record_occurrence(&mut self, symbol: SymbolId, span: Span, range: Range, is_def: bool) {
        self.index.add_occurrence(SymbolOccurrence {
            symbol,
            span,
            range,
            file_path: self.file_path.to_path_buf(),
            is_definition: is_def,
        });
    }

    fn resolve_type(&self, name: &str) -> Option<SymbolId> {
        if let Some(target) = self.imports.type_aliases.get(name) {
            return self.resolve_type(target);
        }
        if name.contains('.') {
            if self.snapshot_structs.contains_key(name) {
                return Some(SymbolId::Struct(name.to_string()));
            }
            if self.snapshot_enums.contains_key(name) {
                return Some(SymbolId::Enum(name.to_string()));
            }
            if self.snapshot_traits.contains_key(name) {
                return Some(SymbolId::Trait(name.to_string()));
            }
        }
        let qualified = qualify_type_name(self.module_path, name);
        if self.snapshot_structs.contains_key(&qualified) {
            return Some(SymbolId::Struct(qualified));
        }
        if self.snapshot_enums.contains_key(&qualified) {
            return Some(SymbolId::Enum(qualified));
        }
        if self.snapshot_traits.contains_key(&qualified) {
            return Some(SymbolId::Trait(qualified));
        }
        let simple = simple_type_name(name);
        if let Some(candidates) = self.structs_by_simple.get(simple) {
            if let Some(q) = candidates.first() {
                return Some(SymbolId::Struct(q.clone()));
            }
        }
        if let Some(candidates) = self.enums_by_simple.get(simple) {
            if let Some(q) = candidates.first() {
                return Some(SymbolId::Enum(q.clone()));
            }
        }
        if let Some(candidates) = self.traits_by_simple.get(simple) {
            if let Some(q) = candidates.first() {
                return Some(SymbolId::Trait(q.clone()));
            }
        }
        None
    }

    fn resolve_type_qualified(&self, name: &str) -> Option<String> {
        match self.resolve_type(name) {
            Some(SymbolId::Struct(q)) | Some(SymbolId::Enum(q)) | Some(SymbolId::Trait(q)) => {
                Some(q)
            }
            _ => None,
        }
    }

    fn resolve_function(&self, name: &str) -> Option<SymbolId> {
        if let Some(target) = self.imports.function_aliases.get(name) {
            return self.resolve_function(target);
        }
        if name.contains('.') && self.snapshot_functions.contains_key(name) {
            return Some(SymbolId::Function(name.to_string()));
        }
        let qualified = qualify_type_name(self.module_path, name);
        if self.snapshot_functions.contains_key(&qualified) {
            return Some(SymbolId::Function(qualified));
        }
        let simple = simple_type_name(name);
        if let Some(candidates) = self.functions_by_simple.get(simple) {
            if let Some(q) = candidates.first() {
                return Some(SymbolId::Function(q.clone()));
            }
        }
        None
    }

    fn index_items(&mut self, items: &[Item]) {
        for item in items {
            self.index_item(item);
        }
    }

    fn index_item(&mut self, item: &Item) {
        match &item.kind {
            ItemKind::Struct(def) => {
                let simple = simple_type_name(&def.name);
                let qualified = qualify_type_name(self.module_path, &def.name);
                let struct_sym = SymbolId::Struct(qualified.clone());
                if let Some((span, range)) =
                    find_identifier_in_span(self.source, self.line_offsets, item.span, simple)
                {
                    self.record_occurrence(struct_sym, span, range, true);
                }
                for field in &def.fields {
                    let field_sym = SymbolId::Field {
                        struct_qualified: qualified.clone(),
                        field_name: field.name.clone(),
                    };
                    let field_span = Span::new(
                        field.ty.span.start_line,
                        1,
                        field.ty.span.start_line,
                        field.ty.span.start_col,
                    );
                    if let Some((span, range)) = find_identifier_in_span(
                        self.source,
                        self.line_offsets,
                        field_span,
                        &field.name,
                    )
                    .or_else(|| {
                        find_identifier_in_span(
                            self.source,
                            self.line_offsets,
                            item.span,
                            &field.name,
                        )
                    }) {
                        self.record_occurrence(field_sym, span, range, true);
                    }
                    self.walk_type(&field.ty);
                }
            }

            ItemKind::Enum(def) => {
                let simple = simple_type_name(&def.name);
                let qualified = qualify_type_name(self.module_path, &def.name);
                let enum_sym = SymbolId::Enum(qualified.clone());
                if let Some((span, range)) =
                    find_identifier_in_span(self.source, self.line_offsets, item.span, simple)
                {
                    self.record_occurrence(enum_sym, span, range, true);
                }
                for variant in &def.variants {
                    let var_sym = SymbolId::Variant {
                        enum_qualified: qualified.clone(),
                        variant_name: variant.name.clone(),
                    };
                    if let Some((span, range)) = find_identifier_in_span(
                        self.source,
                        self.line_offsets,
                        item.span,
                        &variant.name,
                    ) {
                        self.record_occurrence(var_sym, span, range, true);
                    }
                    if let Some(fields) = &variant.fields {
                        for ty in fields {
                            self.walk_type(ty);
                        }
                    }
                }
            }

            ItemKind::Trait(def) => {
                let simple = simple_type_name(&def.name);
                let qualified = qualify_type_name(self.module_path, &def.name);
                let trait_sym = SymbolId::Trait(qualified.clone());
                if let Some((span, range)) =
                    find_identifier_in_span(self.source, self.line_offsets, item.span, simple)
                {
                    self.record_occurrence(trait_sym, span, range, true);
                }
                for method in &def.methods {
                    let method_sym = SymbolId::Method {
                        owner_qualified: qualified.clone(),
                        method_name: method.name.clone(),
                    };
                    if let Some((span, range)) = find_identifier_in_span(
                        self.source,
                        self.line_offsets,
                        item.span,
                        &method.name,
                    ) {
                        self.record_occurrence(method_sym, span, range, true);
                    }
                    for p in &method.params {
                        self.walk_type(&p.ty);
                    }
                    if let Some(ret) = &method.return_type {
                        self.walk_type(ret);
                    }
                }
            }

            ItemKind::Impl(impl_block) => {
                if let Some(target_name) = named_type_name(&impl_block.target_type) {
                    if let Some(sym) = self.resolve_type(&target_name) {
                        let simple = simple_type_name(&target_name);
                        if let Some((span, range)) = find_identifier_in_span(
                            self.source,
                            self.line_offsets,
                            item.span,
                            simple,
                        ) {
                            self.record_occurrence(sym, span, range, false);
                        }
                    }
                }
                if let Some(trait_name) = &impl_block.trait_name {
                    if let Some(sym) = self.resolve_type(trait_name) {
                        let simple = simple_type_name(trait_name);
                        if let Some((span, range)) = find_identifier_in_span(
                            self.source,
                            self.line_offsets,
                            item.span,
                            simple,
                        ) {
                            self.record_occurrence(sym, span, range, false);
                        }
                    }
                }
                let target_qname = named_type_name(&impl_block.target_type)
                    .and_then(|tn| self.resolve_type_qualified(&tn))
                    .unwrap_or_default();
                for method in &impl_block.methods {
                    let method_name = method_display_name(&method.name);
                    let method_sym = SymbolId::Method {
                        owner_qualified: target_qname.clone(),
                        method_name: method_name.clone(),
                    };
                    let method_header_span = Span::new(
                        item.span.start_line,
                        1,
                        item.span.end_line,
                        item.span.end_col,
                    );
                    if let Some((span, range)) = find_identifier_in_span(
                        self.source,
                        self.line_offsets,
                        method_header_span,
                        &method_name,
                    ) {
                        self.record_occurrence(method_sym, span, range, true);
                    }
                    for p in &method.params {
                        self.walk_type(&p.ty);
                    }
                    if let Some(ret) = &method.return_type {
                        self.walk_type(ret);
                    }
                    self.push_scope();
                    for p in &method.params {
                        let param_ty = if p.is_self || p.name == "self" {
                            Some(Type::new(
                                TypeKind::Named(target_qname.clone()),
                                Span::dummy(),
                            ))
                        } else {
                            Some(p.ty.clone())
                        };
                        if let Some((span, range)) = find_identifier_in_span(
                            self.source,
                            self.line_offsets,
                            method_header_span,
                            &p.name,
                        ) {
                            self.define_local(&p.name, span, range, param_ty);
                        }
                    }
                    self.walk_stmts(&method.body);
                    self.pop_scope();
                }
            }

            ItemKind::Function(func) => {
                let unadorned = func
                    .name
                    .strip_prefix(&format!("{}.", self.module_path))
                    .unwrap_or(&func.name);
                let method_info = if let Some(pos) = unadorned.rfind(':') {
                    Some((&unadorned[..pos], &unadorned[pos + 1..], true))
                } else if let Some(pos) = unadorned.rfind('.') {
                    let type_part = &unadorned[..pos];
                    let method_part = &unadorned[pos + 1..];
                    let qtype = qualify_type_name(self.module_path, type_part);
                    if self.snapshot_structs.contains_key(&qtype)
                        || self.snapshot_enums.contains_key(&qtype)
                        || self.structs_by_simple.contains_key(type_part)
                        || self.enums_by_simple.contains_key(type_part)
                    {
                        Some((type_part, method_part, false))
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some((type_name, method_name, _is_instance)) = method_info {
                    let target_qname = self
                        .resolve_type_qualified(type_name)
                        .unwrap_or_else(|| qualify_type_name(self.module_path, type_name));
                    if let Some(sym) = self.resolve_type(type_name) {
                        let simple = simple_type_name(type_name);
                        if let Some((span, range)) = find_identifier_in_span(
                            self.source,
                            self.line_offsets,
                            item.span,
                            simple,
                        ) {
                            self.record_occurrence(sym, span, range, false);
                        }
                    }
                    let method_sym = SymbolId::Method {
                        owner_qualified: target_qname.clone(),
                        method_name: method_name.to_string(),
                    };
                    if let Some((span, range)) = find_identifier_in_span(
                        self.source,
                        self.line_offsets,
                        item.span,
                        method_name,
                    ) {
                        self.record_occurrence(method_sym, span, range, true);
                    }
                    for p in &func.params {
                        self.walk_type(&p.ty);
                    }
                    if let Some(ret) = &func.return_type {
                        self.walk_type(ret);
                    }
                    self.push_scope();
                    for p in &func.params {
                        let param_ty = if p.is_self || p.name == "self" {
                            Some(Type::new(
                                TypeKind::Named(target_qname.clone()),
                                Span::dummy(),
                            ))
                        } else {
                            Some(p.ty.clone())
                        };
                        if let Some((span, range)) = find_identifier_in_span(
                            self.source,
                            self.line_offsets,
                            item.span,
                            &p.name,
                        ) {
                            self.define_local(&p.name, span, range, param_ty);
                        }
                    }
                    self.walk_stmts(&func.body);
                    self.pop_scope();
                } else {
                    let simple = simple_type_name(unadorned);
                    let qualified = qualify_type_name(self.module_path, simple);
                    let func_sym = SymbolId::Function(qualified);
                    if let Some((span, range)) =
                        find_identifier_in_span(self.source, self.line_offsets, item.span, simple)
                    {
                        self.record_occurrence(func_sym, span, range, true);
                    }
                    for p in &func.params {
                        self.walk_type(&p.ty);
                    }
                    if let Some(ret) = &func.return_type {
                        self.walk_type(ret);
                    }
                    self.push_scope();
                    for p in &func.params {
                        if let Some((span, range)) = find_identifier_in_span(
                            self.source,
                            self.line_offsets,
                            item.span,
                            &p.name,
                        ) {
                            self.define_local(&p.name, span, range, Some(p.ty.clone()));
                        }
                    }
                    self.walk_stmts(&func.body);
                    self.pop_scope();
                }
            }

            ItemKind::Script(stmts) => {
                self.push_scope();
                self.walk_stmts(stmts);
                self.pop_scope();
            }

            ItemKind::Use { tree, .. } => {
                self.index_use_tree(tree, item.span);
            }

            ItemKind::Module { items, .. } => {
                self.index_items(items);
            }

            _ => {}
        }
    }

    fn index_use_tree(&mut self, tree: &UseTree, span: Span) {
        match tree {
            UseTree::Path { path, .. } => {
                if let Some(last) = path.last() {
                    let qualified = path.join(".");
                    let sym = self
                        .resolve_type(&qualified)
                        .or_else(|| self.resolve_function(&qualified));
                    if let Some(sym) = sym {
                        if let Some((span, range)) =
                            find_identifier_in_span(self.source, self.line_offsets, span, last)
                        {
                            self.record_occurrence(sym, span, range, false);
                        }
                    }
                }
            }
            UseTree::Group { prefix, items } => {
                for item in items {
                    if let Some(last) = item.path.last() {
                        let mut full = prefix.clone();
                        full.extend(item.path.clone());
                        let qualified = full.join(".");
                        let sym = self
                            .resolve_type(&qualified)
                            .or_else(|| self.resolve_function(&qualified));
                        if let Some(sym) = sym {
                            if let Some((span, range)) =
                                find_identifier_in_span(self.source, self.line_offsets, span, last)
                            {
                                self.record_occurrence(sym, span, range, false);
                            }
                        }
                    }
                }
            }
            UseTree::Glob { .. } => {}
        }
    }

    fn walk_stmts(&mut self, stmts: &[Stmt]) {
        for stmt in stmts {
            self.walk_stmt(stmt);
        }
    }

    fn walk_stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Local {
                bindings,
                initializer,
            } => {
                if let Some(values) = initializer {
                    for expr in values {
                        self.walk_expr(expr);
                    }
                }
                for (idx, binding) in bindings.iter().enumerate() {
                    if let Some(ty) = &binding.type_annotation {
                        self.walk_type(ty);
                    }
                    let var_ty = binding
                        .type_annotation
                        .clone()
                        .or_else(|| self.variable_types.get(&binding.span).cloned())
                        .or_else(|| self.expr_types.get(&binding.span).cloned())
                        .or_else(|| {
                            initializer
                                .as_ref()
                                .and_then(|vals| vals.get(idx))
                                .and_then(|e| {
                                    self.type_for_expr(e).or_else(|| match &e.kind {
                                        ExprKind::StructLiteral { name, .. } => self
                                            .resolve_type_qualified(name)
                                            .map(|q| Type::new(TypeKind::Named(q), Span::dummy())),
                                        _ => None,
                                    })
                                })
                        });
                    if let Some((span, range)) = find_identifier_in_span(
                        self.source,
                        self.line_offsets,
                        binding.span,
                        &binding.name,
                    ) {
                        self.define_local(&binding.name, span, range, var_ty);
                    } else {
                        let range = span_to_range(binding.span);
                        self.define_local(&binding.name, binding.span, range, var_ty);
                    }
                }
            }
            StmtKind::Assign { targets, values } => {
                for target in targets {
                    self.walk_expr(target);
                }
                for value in values {
                    self.walk_expr(value);
                }
            }
            StmtKind::CompoundAssign { target, value, .. } => {
                self.walk_expr(target);
                self.walk_expr(value);
            }
            StmtKind::Expr(expr) => {
                self.walk_expr(expr);
            }
            StmtKind::If {
                condition,
                then_block,
                elseif_branches,
                else_block,
            } => {
                self.walk_expr(condition);
                self.push_scope();
                self.walk_stmts(then_block);
                self.pop_scope();
                for (cond, block) in elseif_branches {
                    self.walk_expr(cond);
                    self.push_scope();
                    self.walk_stmts(block);
                    self.pop_scope();
                }
                if let Some(block) = else_block {
                    self.push_scope();
                    self.walk_stmts(block);
                    self.pop_scope();
                }
            }
            StmtKind::While { condition, body } => {
                self.walk_expr(condition);
                self.push_scope();
                self.walk_stmts(body);
                self.pop_scope();
            }
            StmtKind::ForNumeric {
                variable,
                start,
                end,
                step,
                body,
            } => {
                self.walk_expr(start);
                self.walk_expr(end);
                if let Some(s) = step {
                    self.walk_expr(s);
                }
                self.push_scope();
                let for_span = Span::new(stmt.span.start_line, 1, stmt.span.start_line, 120);
                if let Some((span, range)) =
                    find_identifier_in_span(self.source, self.line_offsets, for_span, variable)
                {
                    self.define_local(
                        variable,
                        span,
                        range,
                        Some(Type::new(TypeKind::Int, Span::dummy())),
                    );
                }
                self.walk_stmts(body);
                self.pop_scope();
            }
            StmtKind::ForIn {
                variables,
                iterator,
                body,
            } => {
                self.walk_expr(iterator);
                self.push_scope();
                let for_span = Span::new(stmt.span.start_line, 1, stmt.span.start_line, 120);
                for var in variables {
                    if let Some((span, range)) =
                        find_identifier_in_span(self.source, self.line_offsets, for_span, var)
                    {
                        self.define_local(var, span, range, None);
                    }
                }
                self.walk_stmts(body);
                self.pop_scope();
            }
            StmtKind::Block(stmts) => {
                self.push_scope();
                self.walk_stmts(stmts);
                self.pop_scope();
            }
            StmtKind::Return(exprs) => {
                for expr in exprs {
                    self.walk_expr(expr);
                }
            }
            StmtKind::Break | StmtKind::Continue => {}
        }
    }

    fn walk_expr(&mut self, expr: &Expr) {
        match &expr.kind {
            ExprKind::Identifier(name) => {
                if let Some(sym) = self.lookup_local(name) {
                    if let Some((span, range)) =
                        find_identifier_in_span(self.source, self.line_offsets, expr.span, name)
                    {
                        self.record_occurrence(sym, span, range, false);
                    }
                } else if let Some(sym) = self.resolve_type(name) {
                    let simple = simple_type_name(name);
                    if let Some((span, range)) =
                        find_identifier_in_span(self.source, self.line_offsets, expr.span, simple)
                    {
                        self.record_occurrence(sym, span, range, false);
                    }
                } else if let Some(sym) = self.resolve_function(name) {
                    let simple = simple_type_name(name);
                    if let Some((span, range)) =
                        find_identifier_in_span(self.source, self.line_offsets, expr.span, simple)
                    {
                        self.record_occurrence(sym, span, range, false);
                    }
                }
            }
            ExprKind::StructLiteral { name, fields } => {
                let simple = simple_type_name(name);
                let qname_opt = self.resolve_type_qualified(name);
                if let Some(qname) = qname_opt {
                    let sym = SymbolId::Struct(qname.clone());
                    if let Some((span, range)) =
                        find_identifier_in_span(self.source, self.line_offsets, expr.span, simple)
                    {
                        self.record_occurrence(sym, span, range, false);
                    }
                    for field in fields {
                        let field_sym = SymbolId::Field {
                            struct_qualified: qname.clone(),
                            field_name: field.name.clone(),
                        };
                        if let Some((span, range)) = find_identifier_in_span(
                            self.source,
                            self.line_offsets,
                            field.span,
                            &field.name,
                        ) {
                            self.record_occurrence(field_sym, span, range, false);
                        }
                        self.walk_expr(&field.value);
                    }
                } else {
                    for field in fields {
                        self.walk_expr(&field.value);
                    }
                }
            }
            ExprKind::FieldAccess { object, field } => {
                self.walk_expr(object);
                let static_owner = match &object.kind {
                    ExprKind::Identifier(id_name) => self.resolve_type(id_name),
                    _ => None,
                };
                let field_sym = if let Some(SymbolId::Struct(sq)) = static_owner {
                    Some(SymbolId::Method {
                        owner_qualified: sq,
                        method_name: field.clone(),
                    })
                } else if let Some(SymbolId::Enum(eq)) = static_owner {
                    Some(SymbolId::Variant {
                        enum_qualified: eq,
                        variant_name: field.clone(),
                    })
                } else if let Some(ty) = self.type_for_expr(object) {
                    if let Some(base_name) = base_type_name(&ty) {
                        if let Some(SymbolId::Struct(sq)) = self.resolve_type(&base_name) {
                            Some(SymbolId::Field {
                                struct_qualified: sq,
                                field_name: field.clone(),
                            })
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };
                if let Some(sym) = field_sym {
                    let start_col = if object.span.end_line == expr.span.end_line {
                        object.span.end_col
                    } else {
                        1
                    };
                    let access_span = Span::new(
                        expr.span.end_line,
                        start_col,
                        expr.span.end_line,
                        expr.span.end_col,
                    );
                    if let Some((span, range)) =
                        find_identifier_in_span(self.source, self.line_offsets, access_span, field)
                    {
                        self.record_occurrence(sym, span, range, false);
                    }
                }
            }
            ExprKind::MethodCall {
                receiver,
                method,
                type_args,
                args,
            } => {
                self.walk_expr(receiver);
                if let Some(targs) = type_args {
                    for targ in targs {
                        self.walk_type(targ);
                    }
                }
                for arg in args {
                    self.walk_expr(arg);
                }
                if let Some(ty) = self.type_for_expr(receiver) {
                    if let Some(base_name) = base_type_name(&ty) {
                        if let Some(SymbolId::Struct(sq)) = self.resolve_type(&base_name) {
                            let sym = SymbolId::Method {
                                owner_qualified: sq,
                                method_name: method.clone(),
                            };
                            let start_col = if receiver.span.end_line == expr.span.end_line {
                                receiver.span.end_col
                            } else {
                                1
                            };
                            let call_span = Span::new(
                                expr.span.end_line,
                                start_col,
                                expr.span.end_line,
                                expr.span.end_col,
                            );
                            if let Some((span, range)) = find_identifier_in_span(
                                self.source,
                                self.line_offsets,
                                call_span,
                                method,
                            ) {
                                self.record_occurrence(sym, span, range, false);
                            }
                        }
                    }
                }
            }
            ExprKind::Call {
                callee,
                type_args,
                args,
            } => {
                self.walk_expr(callee);
                if let Some(targs) = type_args {
                    for targ in targs {
                        self.walk_type(targ);
                    }
                }
                for arg in args {
                    self.walk_expr(arg);
                }
            }
            ExprKind::Lambda {
                params,
                return_type,
                body,
            } => {
                self.push_scope();
                for (param_name, ty_opt) in params {
                    if let Some(ty) = ty_opt {
                        self.walk_type(ty);
                    }
                    let pspan = Span::new(expr.span.start_line, 1, expr.span.start_line, 120);
                    if let Some((span, range)) =
                        find_identifier_in_span(self.source, self.line_offsets, pspan, param_name)
                    {
                        self.define_local(param_name, span, range, ty_opt.clone());
                    }
                }
                if let Some(ret) = return_type {
                    self.walk_type(ret);
                }
                self.walk_expr(body);
                self.pop_scope();
            }
            ExprKind::Cast { expr, target_type } => {
                self.walk_expr(expr);
                self.walk_type(target_type);
            }
            ExprKind::TypeCheck { expr, check_type } => {
                self.walk_expr(expr);
                self.walk_type(check_type);
            }
            ExprKind::IsPattern { expr, pattern } => {
                self.walk_expr(expr);
                self.walk_pattern(pattern);
            }
            ExprKind::Paren(inner) | ExprKind::Unary { operand: inner, .. } => {
                self.walk_expr(inner);
            }
            ExprKind::Binary { left, right, .. } => {
                self.walk_expr(left);
                self.walk_expr(right);
            }
            ExprKind::Array(elements) | ExprKind::Tuple(elements) => {
                for elem in elements {
                    self.walk_expr(elem);
                }
            }
            ExprKind::Map(entries) => {
                for (k, v) in entries {
                    self.walk_expr(k);
                    self.walk_expr(v);
                }
            }
            ExprKind::Index { object, index } => {
                self.walk_expr(object);
                self.walk_expr(index);
            }
            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.walk_expr(condition);
                self.walk_expr(then_branch);
                if let Some(eb) = else_branch {
                    self.walk_expr(eb);
                }
            }
            ExprKind::Block(stmts) => {
                self.push_scope();
                self.walk_stmts(stmts);
                self.pop_scope();
            }
            ExprKind::Return(values) => {
                for v in values {
                    self.walk_expr(v);
                }
            }
            ExprKind::Range { start, end, .. } => {
                self.walk_expr(start);
                self.walk_expr(end);
            }
            ExprKind::EnumConstructor {
                enum_name,
                variant,
                args,
            } => {
                if let Some(SymbolId::Enum(eq)) = self.resolve_type(enum_name) {
                    let sym = SymbolId::Enum(eq.clone());
                    if let Some((span, range)) = find_identifier_in_span(
                        self.source,
                        self.line_offsets,
                        expr.span,
                        simple_type_name(enum_name),
                    ) {
                        self.record_occurrence(sym, span, range, false);
                    }
                    let var_sym = SymbolId::Variant {
                        enum_qualified: eq,
                        variant_name: variant.clone(),
                    };
                    if let Some((span, range)) =
                        find_identifier_in_span(self.source, self.line_offsets, expr.span, variant)
                    {
                        self.record_occurrence(var_sym, span, range, false);
                    }
                }
                for arg in args {
                    self.walk_expr(arg);
                }
            }
            ExprKind::Literal(_) => {}
        }
    }

    fn walk_type(&mut self, ty: &Type) {
        match &ty.kind {
            TypeKind::Named(name) => {
                let simple = simple_type_name(name);
                if !matches!(
                    simple,
                    "int" | "float" | "string" | "bool" | "unknown" | "nil" | "unit"
                ) {
                    if let Some(sym) = self.resolve_type(name) {
                        if let Some((span, range)) =
                            find_identifier_in_span(self.source, self.line_offsets, ty.span, simple)
                        {
                            self.record_occurrence(sym, span, range, false);
                        }
                    }
                }
            }
            TypeKind::GenericInstance { name, type_args } => {
                let simple = simple_type_name(name);
                if let Some(sym) = self.resolve_type(name) {
                    if let Some((span, range)) =
                        find_identifier_in_span(self.source, self.line_offsets, ty.span, simple)
                    {
                        self.record_occurrence(sym, span, range, false);
                    }
                }
                for targ in type_args {
                    self.walk_type(targ);
                }
            }
            TypeKind::Option(inner)
            | TypeKind::Array(inner)
            | TypeKind::Ref(inner)
            | TypeKind::MutRef(inner) => {
                self.walk_type(inner);
            }
            TypeKind::Pointer { pointee, .. } => {
                self.walk_type(pointee);
            }
            TypeKind::Result(ok, err) => {
                self.walk_type(ok);
                self.walk_type(err);
            }
            TypeKind::Map(k, v) => {
                self.walk_type(k);
                self.walk_type(v);
            }
            TypeKind::Tuple(types) | TypeKind::Union(types) => {
                for t in types {
                    self.walk_type(t);
                }
            }
            TypeKind::Function {
                params,
                return_type,
            } => {
                for p in params {
                    self.walk_type(p);
                }
                self.walk_type(return_type);
            }
            TypeKind::Trait(name) => {
                if let Some(sym) = self.resolve_type(name) {
                    let simple = simple_type_name(name);
                    if let Some((span, range)) =
                        find_identifier_in_span(self.source, self.line_offsets, ty.span, simple)
                    {
                        self.record_occurrence(sym, span, range, false);
                    }
                }
            }
            _ => {}
        }
    }

    fn walk_pattern(&mut self, pattern: &Pattern) {
        match pattern {
            Pattern::Struct { name, fields } => {
                let simple = simple_type_name(name);
                if let Some(SymbolId::Struct(qname)) = self.resolve_type(name) {
                    if let Some((span, range)) =
                        find_identifier_in_source(self.source, self.line_offsets, 0, simple, 0)
                    {
                        self.record_occurrence(SymbolId::Struct(qname.clone()), span, range, false);
                    }
                    for (fname, subpat) in fields {
                        let field_sym = SymbolId::Field {
                            struct_qualified: qname.clone(),
                            field_name: fname.clone(),
                        };
                        if let Some((span, range)) =
                            find_identifier_in_source(self.source, self.line_offsets, 0, fname, 0)
                        {
                            self.record_occurrence(field_sym, span, range, false);
                        }
                        self.walk_pattern(subpat);
                    }
                }
            }
            Pattern::Enum {
                enum_name,
                variant,
                bindings,
            } => {
                if !enum_name.is_empty() {
                    if let Some(SymbolId::Enum(eq)) = self.resolve_type(enum_name) {
                        let simple = simple_type_name(enum_name);
                        if let Some((span, range)) =
                            find_identifier_in_source(self.source, self.line_offsets, 0, simple, 0)
                        {
                            self.record_occurrence(SymbolId::Enum(eq.clone()), span, range, false);
                        }
                        let var_sym = SymbolId::Variant {
                            enum_qualified: eq,
                            variant_name: variant.clone(),
                        };
                        if let Some((span, range)) =
                            find_identifier_in_source(self.source, self.line_offsets, 0, variant, 0)
                        {
                            self.record_occurrence(var_sym, span, range, false);
                        }
                    }
                }
                for b in bindings {
                    self.walk_pattern(b);
                }
            }
            Pattern::TypeCheck(ty) => {
                self.walk_type(ty);
            }
            Pattern::Identifier(id) => {
                let span = Span::new(0, 0, 0, 0);
                let range = Range::default();
                self.define_local(id, span, range, None);
            }
            Pattern::Literal(_) | Pattern::Wildcard => {}
        }
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
    let Some(parent_dir) = source_path.parent() else {
        return;
    };
    let Some(stem) = source_path.file_stem().and_then(|s| s.to_str()) else {
        return;
    };
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
