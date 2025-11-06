use crate::analysis::{
    choose_definition, find_type_for_position, hover_from_definition, location_from_definition,
    AnalysisSnapshot,
};
use crate::diagnostics::error_to_diagnostics;
use crate::semantic_tokens::SEMANTIC_TOKEN_TYPES;
use crate::utils::{
    compute_line_offsets, extract_word_at_position, position_to_offset, prev_char_index,
    span_contains_position, span_to_range,
};
use hashbrown::{HashMap, HashSet};
use lust::ast::{Item, ItemKind, TypeKind};
#[cfg(all(not(target_arch = "wasm32")))]
use lust::{packages::prepare_rust_dependencies, resolve_dependencies};
use lust::{Compiler, LustConfig, ModuleLoader, Span, TypeChecker};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::sync::RwLock;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::{
    CompletionItem, CompletionOptions, CompletionParams, CompletionResponse, CompletionTriggerKind,
    Diagnostic, GotoDefinitionParams, GotoDefinitionResponse, Hover, HoverContents, HoverParams,
    HoverProviderCapability, InitializeParams, InitializeResult, InitializedParams, InlayHint,
    InlayHintOptions, InlayHintParams, InlayHintServerCapabilities, MarkupContent, MarkupKind,
    MessageType, OneOf, SemanticToken, SemanticTokens, SemanticTokensFullOptions,
    SemanticTokensLegend, SemanticTokensOptions, SemanticTokensParams, SemanticTokensRangeParams,
    SemanticTokensRangeResult, SemanticTokensResult, SemanticTokensServerCapabilities,
    ServerCapabilities, TextDocumentSyncCapability, TextDocumentSyncKind, TextDocumentSyncOptions,
};
use tower_lsp::{async_trait, Client, LanguageServer, LspService, Server};
use url::Url;

use super::completions::{
    analyze_identifier_context, analyze_member_method_context, analyze_module_path_context,
    analyze_pattern_context, builtin_global_completions, builtin_instance_method_completions,
    builtin_static_method_completions, enum_variant_completions, identifier_completions,
    instance_method_completions, module_alias_member_completions, module_path_completions,
    resolve_base_type_name_for_context, resolve_type_candidates, static_method_completions,
    struct_field_completions, CompletionKind,
};
use super::hover::hover_for_method_call;
use super::inlay_hints::collect_inlay_hints_for_module;

#[derive(Clone)]
struct DocumentState {
    text: String,
    version: i32,
}

struct Backend {
    client: Client,
    documents: Arc<RwLock<HashMap<Url, DocumentState>>>,
    last_published: Arc<RwLock<HashMap<Url, HashSet<Url>>>>,
    analysis: Arc<RwLock<Option<AnalysisSnapshot>>>,
}

impl Backend {
    fn new(client: Client) -> Self {
        Self {
            client,
            documents: Arc::new(RwLock::new(HashMap::new())),
            last_published: Arc::new(RwLock::new(HashMap::new())),
            analysis: Arc::new(RwLock::new(None)),
        }
    }

    async fn semantic_tokens_for_path(&self, path: &Path) -> Option<Vec<SemanticToken>> {
        let analysis = self.analysis.read().await;
        analysis
            .as_ref()
            .and_then(|snapshot| snapshot.semantic_tokens_for_path(path))
    }

    async fn document_text(&self, uri: &Url) -> Option<String> {
        let cached = {
            let docs = self.documents.read().await;
            docs.get(uri).map(|doc| doc.text.clone())
        };
        if cached.is_some() {
            return cached;
        }

        let path = uri.to_file_path().ok()?;
        std::fs::read_to_string(path).ok()
    }

    async fn analyze(&self, uri: &Url) {
        let version = {
            let docs = self.documents.read().await;
            docs.get(uri).map(|doc| doc.version)
        };
        let Some(version) = version else {
            return;
        };
        let diagnostics = self.compute_diagnostics(uri).await;
        self.publish_diagnostics(uri.clone(), version, diagnostics)
            .await;
    }

    async fn compute_diagnostics(&self, uri: &Url) -> HashMap<Url, Vec<Diagnostic>> {
        let entry_path = match uri.to_file_path() {
            Ok(path) => path,
            Err(_) => {
                self.client
                    .log_message(
                        MessageType::ERROR,
                        format!("Unsupported document URI scheme: {uri}"),
                    )
                    .await;
                return HashMap::new();
            }
        };
        let overrides = {
            let docs = self.documents.read().await;
            let mut map = HashMap::new();
            for (doc_uri, state) in docs.iter() {
                if let Ok(path) = doc_uri.to_file_path() {
                    map.insert(path, state.text.clone());
                }
            }

            map
        };
        let entry_dir = entry_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let entry_path_str = match entry_path.to_str() {
            Some(s) => s.to_string(),
            None => {
                self.client
                    .log_message(
                        MessageType::ERROR,
                        format!("Non-UTF-8 file path not supported: {:?}", entry_path),
                    )
                    .await;
                return HashMap::new();
            }
        };
        let config = match LustConfig::load_for_entry(&entry_path) {
            Ok(cfg) => cfg,
            Err(err) => {
                self.client
                    .log_message(
                        MessageType::ERROR,
                        format!("Failed to load configuration: {}", err),
                    )
                    .await;
                return HashMap::new();
            }
        };
        #[cfg(all(not(target_arch = "wasm32")))]
        let dependency_resolution = match resolve_dependencies(&config, &entry_dir) {
            Ok(res) => res,
            Err(err) => {
                self.client
                    .log_message(
                        MessageType::ERROR,
                        format!("Failed to resolve dependencies: {}", err),
                    )
                    .await;
                return HashMap::new();
            }
        };
        #[cfg(all(not(target_arch = "wasm32")))]
        let prepared_rust = match prepare_rust_dependencies(&dependency_resolution, &entry_dir) {
            Ok(list) => list,
            Err(err) => {
                self.client
                    .log_message(
                        MessageType::ERROR,
                        format!("Failed to prepare extern stubs: {}", err),
                    )
                    .await;
                Vec::new()
            }
        };
        #[cfg(all(not(target_arch = "wasm32")))]
        let dependency_root_set: HashSet<String> = dependency_resolution
            .lust()
            .iter()
            .flat_map(|dep| {
                let mut names = vec![dep.name.clone()];
                if let Some(alias) = &dep.sanitized_name {
                    names.push(alias.clone());
                }
                names
            })
            .collect();
        #[cfg(any(target_arch = "wasm32"))]
        let dependency_root_set = HashSet::new();

        let mut loader = ModuleLoader::new(entry_dir.clone());
        loader.set_source_overrides(overrides.clone());
        #[cfg(all(not(target_arch = "wasm32")))]
        for dependency in dependency_resolution.lust() {
            loader.add_module_root(
                dependency.name.clone(),
                dependency.module_root.clone(),
                dependency.root_module.clone(),
            );
            if let Some(alias) = &dependency.sanitized_name {
                loader.add_module_root(
                    alias.clone(),
                    dependency.module_root.clone(),
                    dependency.root_module.clone(),
                );
            }
        }
        #[cfg(all(not(target_arch = "wasm32")))]
        {
            use hashbrown::HashSet;
            let mut seen: HashSet<(String, PathBuf)> = HashSet::new();
            for entry in &prepared_rust {
                for root in &entry.stub_roots {
                    let key = (root.prefix.clone(), root.directory.clone());
                    if seen.insert(key.clone()) {
                        loader.add_module_root(root.prefix.clone(), root.directory.clone(), None);
                    }
                }
            }
        }
        match loader.load_program_from_entry(&entry_path_str) {
            Ok(program) => {
                let module_path_map: HashMap<String, PathBuf> = program
                    .modules
                    .iter()
                    .map(|m| (m.path.clone(), m.source_path.clone()))
                    .collect();
                let mut imports_map = HashMap::new();
                for module in &program.modules {
                    imports_map.insert(module.path.clone(), module.imports.clone());
                }

                let mut wrapped_items: Vec<Item> = Vec::new();
                for module in &program.modules {
                    wrapped_items.push(Item::new(
                        ItemKind::Module {
                            name: module.path.clone(),
                            items: module.items.clone(),
                        },
                        Span::new(0, 0, 0, 0),
                    ));
                }

                let mut typechecker = TypeChecker::with_config(&config);
                typechecker.set_imports_by_module(imports_map.clone());
                let type_result = typechecker.check_program(&program.modules);
                let option_coercions = typechecker.take_option_coercions();
                let struct_defs = typechecker.struct_definitions();
                let enum_defs = typechecker.enum_definitions();
                let type_info = typechecker.take_type_info();
                let snapshot = AnalysisSnapshot::new(
                    &program,
                    type_info,
                    &overrides,
                    struct_defs,
                    enum_defs,
                    dependency_root_set.clone(),
                );
                {
                    let mut analysis = self.analysis.write().await;
                    *analysis = Some(snapshot);
                }

                if let Err(error) = type_result {
                    return self.convert_path_map_to_url_map(error_to_diagnostics(
                        error,
                        &entry_path,
                        &entry_dir,
                        &module_path_map,
                    ));
                }

                let mut compiler = Compiler::new();
                compiler.set_option_coercions(option_coercions);
                compiler.configure_stdlib(&config);
                compiler.set_imports_by_module(imports_map);
                compiler.set_entry_module(program.entry_module.clone());
                if let Err(error) = compiler.compile_module(&wrapped_items) {
                    return self.convert_path_map_to_url_map(error_to_diagnostics(
                        error,
                        &entry_path,
                        &entry_dir,
                        &module_path_map,
                    ));
                }

                let mut result = HashMap::new();
                if let Ok(url) = Url::from_file_path(&entry_path) {
                    result.insert(url, Vec::new());
                }

                result
            }

            Err(error) => self.convert_path_map_to_url_map(error_to_diagnostics(
                error,
                &entry_path,
                &entry_dir,
                &HashMap::new(),
            )),
        }
    }

    fn convert_path_map_to_url_map(
        &self,
        path_map: HashMap<PathBuf, Vec<Diagnostic>>,
    ) -> HashMap<Url, Vec<Diagnostic>> {
        let mut result = HashMap::new();
        for (path, diagnostics) in path_map {
            if let Ok(url) = Url::from_file_path(&path) {
                result.insert(url, diagnostics);
            }
        }

        result
    }

    async fn publish_diagnostics(
        &self,
        entry_uri: Url,
        entry_version: i32,
        mut new_diagnostics: HashMap<Url, Vec<Diagnostic>>,
    ) {
        new_diagnostics
            .entry(entry_uri.clone())
            .or_insert_with(Vec::new);
        let associated_uris: HashSet<Url> = new_diagnostics.keys().cloned().collect();
        let previous_uris = {
            let mut tracker = self.last_published.write().await;
            tracker
                .insert(entry_uri.clone(), associated_uris.clone())
                .unwrap_or_default()
        };
        let version_lookup = {
            let docs = self.documents.read().await;
            docs.iter()
                .map(|(u, state)| (u.clone(), state.version))
                .collect::<HashMap<_, _>>()
        };
        for (uri, diagnostics) in new_diagnostics {
            let version = version_lookup.get(&uri).copied().or_else(|| {
                if uri == entry_uri {
                    Some(entry_version)
                } else {
                    None
                }
            });
            self.client
                .publish_diagnostics(uri, diagnostics, version)
                .await;
        }

        for uri in previous_uris.difference(&associated_uris) {
            let version = version_lookup.get(uri).copied();
            self.client
                .publish_diagnostics(uri.clone(), Vec::new(), version)
                .await;
        }
    }
}

#[async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _params: InitializeParams) -> Result<InitializeResult> {
        let text_document_sync = TextDocumentSyncOptions {
            open_close: Some(true),
            change: Some(TextDocumentSyncKind::FULL),
            ..Default::default()
        };
        let hover_provider = Some(HoverProviderCapability::Simple(true));
        let definition_provider = Some(OneOf::Left(true));
        let inlay_hint_provider = Some(OneOf::Right(InlayHintServerCapabilities::Options(
            InlayHintOptions::default(),
        )));
        let semantic_tokens_provider = Some(
            SemanticTokensServerCapabilities::SemanticTokensOptions(SemanticTokensOptions {
                legend: SemanticTokensLegend {
                    token_types: SEMANTIC_TOKEN_TYPES.to_vec(),
                    token_modifiers: Vec::new(),
                },
                full: Some(SemanticTokensFullOptions::Bool(true)),
                range: Some(false),
                ..Default::default()
            }),
        );
        let completion_provider = Some(CompletionOptions {
            resolve_provider: Some(false),
            trigger_characters: Some(vec![".".to_string(), ":".to_string()]),
            ..CompletionOptions::default()
        });
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Options(text_document_sync)),
                hover_provider,
                definition_provider,
                completion_provider,
                inlay_hint_provider,
                semantic_tokens_provider,
                ..ServerCapabilities::default()
            },
            server_info: None,
        })
    }

    async fn initialized(&self, _params: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "lust-analyzer initialized")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: tower_lsp::lsp_types::DidOpenTextDocumentParams) {
        let uri = params.text_document.uri;
        let version = params.text_document.version;
        let text = params.text_document.text;
        {
            let mut docs = self.documents.write().await;
            docs.insert(uri.clone(), DocumentState { text, version });
        }
        self.analyze(&uri).await;
    }

    async fn did_change(&self, params: tower_lsp::lsp_types::DidChangeTextDocumentParams) {
        let uri = params.text_document.uri;
        let version = params.text_document.version;
        let mut docs = self.documents.write().await;
        if let Some(doc) = docs.get_mut(&uri) {
            if let Some(change) = params.content_changes.into_iter().last() {
                doc.text = change.text;
                doc.version = version;
            }
        }

        drop(docs);
        self.analyze(&uri).await;
    }

    async fn did_close(&self, params: tower_lsp::lsp_types::DidCloseTextDocumentParams) {
        let uri = params.text_document.uri;
        {
            let mut docs = self.documents.write().await;
            docs.remove(&uri);
        }

        let related = {
            let mut tracker = self.last_published.write().await;
            tracker.remove(&uri).unwrap_or_default()
        };
        for related_uri in related {
            let version = {
                let docs = self.documents.read().await;
                docs.get(&related_uri).map(|state| state.version)
            };
            self.client
                .publish_diagnostics(related_uri.clone(), Vec::new(), version)
                .await;
        }

        self.client.publish_diagnostics(uri, Vec::new(), None).await;
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let uri = params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;
        let mut text = match self.document_text(&uri).await {
            Some(text) => text,
            None => return Ok(None),
        };
        let file_path = match uri.to_file_path() {
            Ok(path) => path,
            Err(_) => return Ok(None),
        };
        let analysis = self.analysis.read().await;
        let snapshot = match analysis.as_ref() {
            Some(snapshot) => snapshot,
            None => return Ok(None),
        };
        let module = match snapshot.module_for_file(&file_path) {
            Some(module) => module,
            None => return Ok(None),
        };
        let module_path = snapshot
            .module_path_for_file(&file_path)
            .map(|s| s.to_string());
        let mut line_offsets = compute_line_offsets(&text);
        let mut offset = match position_to_offset(&text, position, &line_offsets) {
            Some(value) => value,
            None => return Ok(None),
        };
        if let Some(context_info) = params.context.as_ref() {
            if context_info.trigger_kind == CompletionTriggerKind::TRIGGER_CHARACTER {
                if let Some(trigger_str) = context_info.trigger_character.as_ref() {
                    let mut chars = trigger_str.chars();
                    if let Some(trigger_char) = chars.next() {
                        if chars.next().is_none() {
                            let has_trigger = prev_char_index(&text, offset)
                                .map(|(_, ch)| ch == trigger_char)
                                .unwrap_or(false);
                            if !has_trigger {
                                if offset <= text.len() {
                                    text.insert(offset, trigger_char);
                                } else {
                                    text.push(trigger_char);
                                }

                                line_offsets = compute_line_offsets(&text);
                                offset = match position_to_offset(&text, position, &line_offsets) {
                                    Some(value) => value,
                                    None => return Ok(None),
                                };
                            }
                        }
                    }
                }
            }
        }

        let mut context = analyze_member_method_context(&text, offset)
            .or_else(|| analyze_pattern_context(&text, offset));
        if context.is_none() {
            context = analyze_module_path_context(&text, offset, module, &position);
        }

        if context.is_none() {
            context = analyze_identifier_context(&text, offset);
        }

        let Some(context) = context else {
            return Ok(None);
        };
        let mut items: Vec<CompletionItem> = Vec::new();
        match context.kind {
            CompletionKind::Member | CompletionKind::Method | CompletionKind::Pattern => {
                let base_name = resolve_base_type_name_for_context(
                    module,
                    snapshot,
                    module_path.as_deref(),
                    &text,
                    &line_offsets,
                    &context,
                );
                match context.kind {
                    CompletionKind::Member => {
                        if let Some(owner) = base_name.as_deref() {
                            items.extend(struct_field_completions(
                                snapshot,
                                module_path.as_deref(),
                                owner,
                                &context.prefix,
                            ));
                        }

                        if let Some(object_name) = context.object_name.as_ref() {
                            let candidates = resolve_type_candidates(object_name, module);
                            let mut seen_candidates = HashSet::new();
                            for candidate in candidates {
                                if !seen_candidates.insert(candidate.clone()) {
                                    continue;
                                }

                                items.extend(enum_variant_completions(
                                    snapshot,
                                    module_path.as_deref(),
                                    &candidate,
                                    &context.prefix,
                                ));
                                items.extend(static_method_completions(
                                    snapshot,
                                    &candidate,
                                    module_path.as_deref(),
                                    &context.prefix,
                                ));
                            }

                            items.extend(builtin_global_completions(object_name, &context.prefix));
                            items.extend(builtin_static_method_completions(
                                object_name,
                                &context.prefix,
                            ));
                            items.extend(module_alias_member_completions(
                                snapshot,
                                module,
                                &context.prefix,
                                object_name,
                            ));
                        }
                    }

                    CompletionKind::Method => {
                        if let Some(owner) = base_name.as_deref() {
                            items.extend(instance_method_completions(
                                snapshot,
                                owner,
                                module_path.as_deref(),
                                &context.prefix,
                            ));
                            items.extend(builtin_instance_method_completions(
                                owner,
                                &context.prefix,
                            ));
                        }
                    }

                    CompletionKind::Pattern => {
                        if let Some(owner) = base_name.as_deref() {
                            items.extend(enum_variant_completions(
                                snapshot,
                                module_path.as_deref(),
                                owner,
                                &context.prefix,
                            ));
                        }
                    }

                    _ => {}
                }
            }

            CompletionKind::Identifier => {
                items.extend(identifier_completions(
                    module,
                    snapshot,
                    &file_path,
                    position,
                    &context.prefix,
                ));
            }

            CompletionKind::ModulePath => {
                items.extend(module_path_completions(
                    snapshot,
                    module,
                    &context.path_segments,
                    &context.prefix,
                ));
            }
        }

        if items.is_empty() {
            return Ok(None);
        }

        let mut unique = Vec::new();
        let mut seen_labels = HashSet::new();
        for item in items.into_iter() {
            if seen_labels.insert(item.label.clone()) {
                unique.push(item);
            }
        }

        Ok(Some(CompletionResponse::Array(unique)))
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;
        let file_path = match uri.to_file_path() {
            Ok(path) => path,
            Err(_) => return Ok(None),
        };
        let text = self.document_text(&uri).await;
        let word = text
            .as_ref()
            .and_then(|source| extract_word_at_position(source, position));
        let (method_hover, def_opt, type_opt) = {
            let analysis = self.analysis.read().await;
            let Some(snapshot) = analysis.as_ref() else {
                return Ok(None);
            };
            let module_path = snapshot
                .module_path_for_file(&file_path)
                .map(|s| s.to_string());
            let module = snapshot.module_for_file(&file_path);
            let method_hover = if let (Some(module), Some(source), Some(token)) =
                (module, text.as_ref(), word.as_deref())
            {
                hover_for_method_call(
                    snapshot,
                    module,
                    module_path.as_deref(),
                    source,
                    position,
                    token,
                )
            } else {
                None
            };
            let mut def_clone = None;
            if let Some(defs) = snapshot.definitions_in_file(&file_path) {
                if let Some(def) = defs
                    .iter()
                    .find(|d| span_contains_position(d.span, &position))
                {
                    def_clone = Some(def.clone());
                }
            }

            if def_clone.is_none() {
                if let Some(word) = word.as_ref() {
                    if let Some(def) = snapshot.definition_by_qualified(word) {
                        def_clone = Some(def.clone());
                    } else if let Some(defs) = snapshot.definitions_by_simple(word) {
                        if let Some(def) = choose_definition(defs, module_path.as_deref()) {
                            def_clone = Some(def.clone());
                        }
                    }
                }
            }

            let type_opt = if def_clone.is_none() {
                module
                    .and_then(|module| find_type_for_position(module, position))
                    .map(|(span, ty)| {
                        let type_def = match &ty.kind {
                            TypeKind::Named(name) => snapshot
                                .definition_by_qualified(name)
                                .cloned()
                                .or_else(|| {
                                    if let Some(mp) = module_path.as_ref() {
                                        let qualified = format!("{}.{}", mp, name);
                                        snapshot.definition_by_qualified(&qualified).cloned()
                                    } else {
                                        None
                                    }
                                })
                                .or_else(|| {
                                    snapshot
                                        .definitions_by_simple(name)
                                        .and_then(|defs| {
                                            choose_definition(defs, module_path.as_deref())
                                        })
                                        .cloned()
                                }),
                            _ => None,
                        };
                        (span, ty, type_def)
                    })
            } else {
                None
            };
            (method_hover, def_clone, type_opt)
        };
        if let Some(hover) = method_hover {
            return Ok(Some(hover));
        }

        if let Some(def) = def_opt {
            return Ok(Some(hover_from_definition(&def)));
        }

        if let Some((span, ty, type_def)) = type_opt {
            if let Some(def) = type_def {
                return Ok(Some(hover_from_definition(&def)));
            }

            let hover = Hover {
                contents: HoverContents::Markup(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: format!("`{}`", ty),
                }),
                range: Some(span_to_range(span)),
            };
            return Ok(Some(hover));
        }

        Ok(None)
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;
        let file_path = match uri.to_file_path() {
            Ok(path) => path,
            Err(_) => return Ok(None),
        };
        let text = self.document_text(&uri).await;
        let word = text
            .as_ref()
            .and_then(|source| extract_word_at_position(source, position));
        let def_opt = {
            let analysis = self.analysis.read().await;
            let Some(snapshot) = analysis.as_ref() else {
                return Ok(None);
            };
            let mut def_clone = None;
            if let Some(defs) = snapshot.definitions_in_file(&file_path) {
                if let Some(def) = defs
                    .iter()
                    .find(|d| span_contains_position(d.span, &position))
                {
                    def_clone = Some(def.clone());
                }
            }

            if def_clone.is_none() {
                let module_path = snapshot.module_path_for_file(&file_path);
                if let Some(word) = word.as_ref() {
                    if let Some(def) = snapshot.definition_by_qualified(word) {
                        def_clone = Some(def.clone());
                    } else if let Some(defs) = snapshot.definitions_by_simple(word) {
                        if let Some(def) = choose_definition(defs, module_path.as_deref()) {
                            def_clone = Some(def.clone());
                        }
                    }
                }
            }

            def_clone
        };
        if let Some(def) = def_opt {
            if let Some(location) = location_from_definition(&def) {
                return Ok(Some(GotoDefinitionResponse::Scalar(location)));
            }
        }

        Ok(None)
    }

    async fn inlay_hint(&self, params: InlayHintParams) -> Result<Option<Vec<InlayHint>>> {
        let uri = params.text_document.uri;
        let range = params.range;
        let file_path = match uri.to_file_path() {
            Ok(path) => path,
            Err(_) => return Ok(None),
        };
        let hints = {
            let analysis = self.analysis.read().await;
            let Some(snapshot) = analysis.as_ref() else {
                return Ok(None);
            };
            snapshot
                .module_for_file(&file_path)
                .map(|module| collect_inlay_hints_for_module(module, &range))
        };
        Ok(hints.or(Some(Vec::new())))
    }

    async fn semantic_tokens_full(
        &self,
        params: SemanticTokensParams,
    ) -> Result<Option<SemanticTokensResult>> {
        let uri = params.text_document.uri;
        let path = match uri.to_file_path() {
            Ok(path) => path,
            Err(_) => return Ok(None),
        };
        let tokens = self.semantic_tokens_for_path(&path).await;
        Ok(tokens.map(|data| {
            SemanticTokensResult::Tokens(SemanticTokens {
                result_id: None,
                data,
            })
        }))
    }

    async fn semantic_tokens_range(
        &self,
        _params: SemanticTokensRangeParams,
    ) -> Result<Option<SemanticTokensRangeResult>> {
        Ok(None)
    }
}

pub async fn run() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    let (service, socket) = LspService::build(|client| Backend::new(client)).finish();
    Server::new(stdin, stdout, socket).serve(service).await;
}
