//! Rust binding generation for checked Lust modules.
//!
//! Bind public free functions and selected public `impl` methods marked with
//! `---@bindgen` (or `--- @bindgen`). For trait methods, put the tag on the
//! concrete implementation method. An optional `name` setting changes the
//! generated Rust method name. Public nongeneric structs and enums are also
//! emitted as Rust types; add the same tag to a type to override its Rust type
//! name. Struct wrappers expose typed field getters, while enums become Rust
//! enums. Instance methods on generated structs are callable directly on the
//! wrapper. Static methods take the bindings facade explicitly; enum instance
//! methods do too because Rust enum wrappers have no place to retain a program
//! context.
//!
//! ```lust
//! --- Fetch a player by id.
//! ---@bindgen(name = "get_player")
//! function lookup_player(id: int): Player
//!     ...
//! end
//!
//! impl Player
//!     ---@bindgen
//!     function display_label(self): string
//!         return self.name .. "#" .. tostring(self.id)
//!     end
//!
//!     ---@bindgen
//!     function new(id: int): Player
//!         ...
//!     end
//! end
//! ```
//!
//! The generated struct wrapper supports `player.display_label()`. Static
//! Lust methods are called as `Player::new(&mut bindings, id)` so the target
//! embedded program is explicit. Instance methods on Rust enum wrappers also
//! take `&mut bindings`, because the generated enums remain ordinary matchable
//! Rust enums rather than storing a hidden runtime context.
//!
//! The module is available with the `bindgen` Cargo feature. It typechecks the
//! Lust module graph without executing it and emits an inspectable Rust source
//! file. Generated methods call [`crate::EmbeddedProgram::call_typed`], so
//! runtime signature checks remain active. Returned struct wrappers retain a
//! scoped handle to their program, allowing instance methods to be called
//! without passing the program on every call.
//!
//! A build script can generate bindings into Cargo's `OUT_DIR`:
//!
//! ```ignore
//! let generated = lust::bindgen::RustBindingsBuilder::new("scripts/main.lust")
//!     .generate()?;
//! for path in generated.input_files() {
//!     println!("cargo:rerun-if-changed={}", path.display());
//! }
//! let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR")?);
//! generated.write_to(out_dir.join("lust_bindings.rs"))?;
//! ```
//!
//! Then include the generated file from Rust:
//!
//! ```ignore
//! pub mod lust_bindings {
//!     include!(concat!(env!("OUT_DIR"), "/lust_bindings.rs"));
//! }
//! ```

use crate::ast::{
    EnumDef, ExternItem, FunctionDef, Item, ItemKind, StructDef, Type, TypeKind, Visibility,
};
use crate::embed::ExternRegistry;
use crate::modules::{ModuleImports, ModuleLoader, Program};
use crate::typechecker::{FunctionSignature, TypeChecker};
use crate::{LustConfig, LustError, Result};
use hashbrown::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Generates Rust bindings from tagged public Lust functions and impl methods, plus public nominal types.
#[derive(Clone)]
pub struct RustBindingsBuilder {
    entry_file: PathBuf,
    bindings_name: String,
    runtime_crate_path: String,
    extern_registry: ExternRegistry,
}

impl RustBindingsBuilder {
    /// Create a generator for the Lust entry file.
    pub fn new(entry_file: impl Into<PathBuf>) -> Self {
        Self {
            entry_file: entry_file.into(),
            bindings_name: "Bindings".to_string(),
            runtime_crate_path: "::lust".to_string(),
            extern_registry: ExternRegistry::new(),
        }
    }

    /// Set the generated facade type name (default: `Bindings`).
    pub fn bindings_name(mut self, name: impl Into<String>) -> Self {
        self.bindings_name = name.into();
        self
    }

    /// Set the Rust path used to refer to the Lust runtime (default: `::lust`).
    /// This is useful if the application renamed its `lust-rs` dependency.
    pub fn runtime_crate_path(mut self, path: impl Into<String>) -> Self {
        self.runtime_crate_path = path.into();
        self
    }

    /// Supply Rust-declared Lust types/functions used by this project.
    pub fn with_extern_registry(mut self, registry: ExternRegistry) -> Self {
        self.extern_registry = registry;
        self
    }

    /// Typecheck the Lust project and return generated Rust source plus the
    /// source/config files Cargo should watch when used from `build.rs`.
    pub fn generate(&self) -> Result<GeneratedRustBindings> {
        let entry_file = std::fs::canonicalize(&self.entry_file).map_err(|err| {
            LustError::Unknown(format!(
                "failed to resolve Lust entry '{}': {err}",
                self.entry_file.display()
            ))
        })?;
        let entry_text = entry_file.to_str().ok_or_else(|| {
            LustError::Unknown(format!(
                "Lust entry path '{}' is not valid UTF-8",
                entry_file.display()
            ))
        })?;
        let config = LustConfig::load_for_entry(&entry_file)
            .map_err(|err| LustError::Unknown(format!("failed to load Lust config: {err}")))?;

        let mut loader = ModuleLoader::new(entry_file.parent().unwrap_or_else(|| Path::new(".")));
        let program = loader.load_program_from_entry(entry_text)?;

        let mut imports = HashMap::new();
        for module in &program.modules {
            imports.insert(module.path.clone(), module.imports.clone());
        }

        // Run the frontend only. In particular, don't compile/load a VM here:
        // module initializers must not execute during bindgen.
        let mut typechecker = TypeChecker::with_config(&config);
        typechecker.set_imports_by_module(imports);
        self.extern_registry
            .register_with_typechecker(&mut typechecker)?;
        typechecker.check_program(&program.modules)?;

        let signatures = typechecker.function_signatures();
        let mut struct_defs = typechecker.struct_definitions();
        let mut enum_defs = typechecker.enum_definitions();
        for def in self.extern_registry.structs() {
            struct_defs.insert(def.name.clone(), def.clone());
        }
        for def in self.extern_registry.enums() {
            enum_defs.insert(def.name.clone(), def.clone());
        }

        let (declared_structs, declared_enums) =
            collect_declared_nominal_names(&program, &self.extern_registry);
        let (nominal_types, wrapped_types) = collect_nominal_types(
            &struct_defs,
            &enum_defs,
            &declared_structs,
            &declared_enums,
            &self.bindings_name,
        )?;
        let lifetime_types = collect_lifetime_types(&nominal_types, &wrapped_types);
        let (functions, methods) = collect_bindings(
            &program,
            &signatures,
            &wrapped_types,
            &struct_defs,
            &enum_defs,
        )?;
        let source = generate_source(
            &self.bindings_name,
            &self.runtime_crate_path,
            &functions,
            &methods,
            &nominal_types,
            &wrapped_types,
            &lifetime_types,
            &struct_defs,
            &enum_defs,
        )?;

        let mut input_files = vec![entry_file.clone()];
        input_files.extend(program.modules.iter().map(|module| {
            std::fs::canonicalize(&module.source_path)
                .unwrap_or_else(|_| module.source_path.clone())
        }));
        // Watch the config path even if it doesn't exist yet, so adding one
        // later causes build-script users to regenerate the bindings.
        input_files.push(
            entry_file
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join("lust-config.toml"),
        );
        input_files.sort();
        input_files.dedup();

        Ok(GeneratedRustBindings {
            source,
            input_files,
        })
    }
}

/// Generated Rust source and the Lust/config files it depends on.
#[derive(Clone, Debug)]
pub struct GeneratedRustBindings {
    /// Complete, inspectable Rust source for the binding facade.
    pub source: String,
    /// Entry, imported module, and config paths to pass to Cargo's
    /// `cargo:rerun-if-changed` directive from a build script.
    pub input_files: Vec<PathBuf>,
}

impl GeneratedRustBindings {
    pub fn input_files(&self) -> &[PathBuf] {
        &self.input_files
    }

    pub fn write_to(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, &self.source)
    }
}

#[derive(Clone, Debug)]
struct BoundFunction {
    lust_name: String,
    rust_name: String,
    docs: Vec<String>,
    params: Vec<(String, Type)>,
    return_type: Type,
}

#[derive(Clone, Debug)]
struct BoundNominal {
    lust_name: String,
    rust_name: String,
    docs: Vec<String>,
    kind: BoundNominalKind,
}

#[derive(Clone, Debug)]
struct BoundMethod {
    lust_name: String,
    rust_name: String,
    target_lust_name: String,
    target_rust_name: String,
    has_receiver: bool,
    target_is_struct: bool,
    docs: Vec<String>,
    params: Vec<(String, Type)>,
    return_type: Type,
}

#[derive(Clone, Debug)]
enum BoundNominalKind {
    Struct(StructDef),
    Enum(EnumDef),
}

#[derive(Default)]
struct ParsedDoc {
    bindgen: Option<BindgenDirective>,
    prose: Vec<String>,
}

#[derive(Default)]
struct BindgenDirective {
    name: Option<String>,
}

fn collect_nominal_types(
    struct_defs: &HashMap<String, StructDef>,
    enum_defs: &HashMap<String, EnumDef>,
    declared_structs: &HashSet<String>,
    declared_enums: &HashSet<String>,
    bindings_name: &str,
) -> Result<(Vec<BoundNominal>, HashMap<String, String>)> {
    let (_, bindings_key) = rust_identifier(bindings_name, "bindings type name")?;
    let recursive_enums = recursive_enum_names(enum_defs, declared_enums);
    let mut pending = Vec::new();
    for name in declared_structs {
        let def = struct_defs.get(name).ok_or_else(|| {
            bindgen_error(format!(
                "no checked Lust struct definition found for '{name}'"
            ))
        })?;
        pending.push((name.clone(), BoundNominalKind::Struct(def.clone())));
    }
    for name in declared_enums {
        let def = enum_defs.get(name).ok_or_else(|| {
            bindgen_error(format!(
                "no checked Lust enum definition found for '{name}'"
            ))
        })?;
        pending.push((name.clone(), BoundNominalKind::Enum(def.clone())));
    }
    pending.sort_by(|left, right| left.0.cmp(&right.0));

    let mut nominal_types = Vec::new();
    let mut wrapped_types = HashMap::new();
    let mut rust_names = HashSet::new();
    for (lust_name, kind) in pending {
        let (visibility, type_params, doc) = match &kind {
            BoundNominalKind::Struct(def) => (def.visibility, &def.type_params, def.doc.as_deref()),
            BoundNominalKind::Enum(def) => (def.visibility, &def.type_params, def.doc.as_deref()),
        };
        let parsed_doc = parse_doc(doc)?;
        if visibility != Visibility::Public {
            if parsed_doc.bindgen.is_some() {
                return Err(bindgen_error(format!(
                    "type '{lust_name}' is marked @bindgen but is not public"
                )));
            }
            continue;
        }
        if !type_params.is_empty() {
            if parsed_doc.bindgen.is_some() {
                return Err(bindgen_error(format!(
                    "generic type wrappers are not yet supported by bindgen ('{lust_name}')"
                )));
            }
            continue;
        }
        if matches!(&kind, BoundNominalKind::Enum(_)) && recursive_enums.contains(&lust_name) {
            if parsed_doc.bindgen.is_some() {
                return Err(bindgen_error(format!(
                    "recursive enum wrapper '{lust_name}' is not yet supported; use EnumInstance instead"
                )));
            }
            continue;
        }

        let default_name = lust_name.rsplit(['.', ':']).next().unwrap_or(&lust_name);
        let requested_name = parsed_doc
            .bindgen
            .and_then(|directive| directive.name)
            .unwrap_or_else(|| default_name.to_string());
        let (rust_name, collision_key) =
            rust_identifier(&requested_name, "generated nominal type name")?;
        if collision_key == bindings_key {
            return Err(bindgen_error(format!(
                "generated Lust type '{lust_name}' conflicts with bindings type '{bindings_name}'; rename it with @bindgen(name = \"...\")"
            )));
        }
        if !rust_names.insert(collision_key.clone()) {
            return Err(bindgen_error(format!(
                "multiple Lust types map to Rust type '{collision_key}'; rename one with @bindgen(name = \"...\")"
            )));
        }
        wrapped_types.insert(lust_name.clone(), rust_name.clone());
        nominal_types.push(BoundNominal {
            lust_name,
            rust_name,
            docs: parsed_doc.prose,
            kind,
        });
    }

    Ok((nominal_types, wrapped_types))
}

fn collect_lifetime_types(
    nominal_types: &[BoundNominal],
    wrapped_types: &HashMap<String, String>,
) -> HashSet<String> {
    // Struct wrappers retain a scoped program context so their Lust methods can
    // be called directly. Enums need the same lifetime only when their Rust
    // variants contain one of those wrappers.
    let mut lifetime_types = nominal_types
        .iter()
        .filter_map(|nominal| {
            matches!(nominal.kind, BoundNominalKind::Struct(_)).then(|| nominal.lust_name.clone())
        })
        .collect::<HashSet<_>>();

    loop {
        let mut changed = false;
        for nominal in nominal_types {
            if lifetime_types.contains(&nominal.lust_name) {
                continue;
            }
            let BoundNominalKind::Enum(def) = &nominal.kind else {
                continue;
            };
            let contains_context = def
                .variants
                .iter()
                .flat_map(|variant| variant.fields.as_deref().unwrap_or_default())
                .any(|ty| type_contains_lifetime(ty, &lifetime_types, wrapped_types));
            if contains_context {
                changed |= lifetime_types.insert(nominal.lust_name.clone());
            }
        }
        if !changed {
            break;
        }
    }

    // Keep only emitted wrappers. A referenced type may be private/generic and
    // use a runtime handle instead of a generated lifetime-bearing wrapper.
    lifetime_types.retain(|name| wrapped_types.contains_key(name));
    lifetime_types
}

fn type_contains_lifetime(
    ty: &Type,
    lifetime_types: &HashSet<String>,
    wrapped_types: &HashMap<String, String>,
) -> bool {
    match &ty.kind {
        TypeKind::Named(name) | TypeKind::GenericInstance { name, .. } => {
            wrapped_types.contains_key(name) && lifetime_types.contains(name)
        }
        TypeKind::Array(inner) | TypeKind::Option(inner) => {
            type_contains_lifetime(inner, lifetime_types, wrapped_types)
        }
        TypeKind::Result(ok, err) => {
            type_contains_lifetime(ok, lifetime_types, wrapped_types)
                || type_contains_lifetime(err, lifetime_types, wrapped_types)
        }
        _ => false,
    }
}

fn collect_declared_nominal_names(
    program: &Program,
    extern_registry: &ExternRegistry,
) -> (HashSet<String>, HashSet<String>) {
    let mut structs = HashSet::new();
    let mut enums = HashSet::new();
    for module in &program.modules {
        collect_nominal_names_in_items(&module.path, &module.items, &mut structs, &mut enums);
    }
    for def in extern_registry.structs() {
        structs.insert(canonical_nominal_name("", &def.name));
    }
    for def in extern_registry.enums() {
        enums.insert(canonical_nominal_name("", &def.name));
    }
    (structs, enums)
}

fn collect_nominal_names_in_items(
    module: &str,
    items: &[Item],
    structs: &mut HashSet<String>,
    enums: &mut HashSet<String>,
) {
    for item in items {
        match &item.kind {
            ItemKind::Struct(def) => {
                structs.insert(canonical_nominal_name(module, &def.name));
            }
            ItemKind::Enum(def) => {
                enums.insert(canonical_nominal_name(module, &def.name));
            }
            ItemKind::Extern { items, .. } => {
                for item in items {
                    match item {
                        ExternItem::Struct(def) => {
                            structs.insert(canonical_nominal_name(module, &def.name));
                        }
                        ExternItem::Enum(def) => {
                            enums.insert(canonical_nominal_name(module, &def.name));
                        }
                        ExternItem::Function { .. } | ExternItem::Const { .. } => {}
                    }
                }
            }
            ItemKind::Module { name, items } => {
                let child_module = if name.contains('.') || name.contains("::") {
                    name.replace("::", ".")
                } else if module.is_empty() {
                    name.clone()
                } else {
                    format!("{module}.{name}")
                };
                collect_nominal_names_in_items(&child_module, items, structs, enums);
            }
            ItemKind::Script(_)
            | ItemKind::Function(_)
            | ItemKind::Trait(_)
            | ItemKind::Impl(_)
            | ItemKind::Use { .. } => {}
        }
    }
}

fn canonical_nominal_name(module: &str, name: &str) -> String {
    let normalized = name.replace("::", ".");
    if normalized.contains('.') || module.is_empty() {
        normalized
    } else {
        format!("{module}.{normalized}")
    }
}

fn recursive_enum_names(
    enum_defs: &HashMap<String, EnumDef>,
    declared_enums: &HashSet<String>,
) -> HashSet<String> {
    let candidates = declared_enums
        .iter()
        .filter(|name| {
            enum_defs.get(*name).is_some_and(|def| {
                def.visibility == Visibility::Public && def.type_params.is_empty()
            })
        })
        .cloned()
        .collect::<HashSet<_>>();
    let mut graph: HashMap<String, HashSet<String>> = HashMap::new();

    for name in &candidates {
        let mut edges = HashSet::new();
        if let Some(def) = enum_defs.get(name) {
            for variant in &def.variants {
                for field in variant.fields.as_deref().unwrap_or_default() {
                    collect_inline_enum_refs(field, &candidates, &mut edges);
                }
            }
        }
        graph.insert(name.clone(), edges);
    }

    candidates
        .iter()
        .filter(|start| {
            let mut visited = HashSet::from([(*start).clone()]);
            enum_reaches_start(start, start, &graph, &mut visited)
        })
        .cloned()
        .collect()
}

fn collect_inline_enum_refs(ty: &Type, candidates: &HashSet<String>, out: &mut HashSet<String>) {
    match &ty.kind {
        TypeKind::Named(name) | TypeKind::GenericInstance { name, .. }
            if candidates.contains(name) =>
        {
            out.insert(name.clone());
        }
        TypeKind::Option(inner) => collect_inline_enum_refs(inner, candidates, out),
        TypeKind::Result(ok, err) => {
            collect_inline_enum_refs(ok, candidates, out);
            collect_inline_enum_refs(err, candidates, out);
        }
        // Arrays and maps are represented by handles/heap containers and do
        // not create an infinitely sized Rust enum variant.
        _ => {}
    }
}

fn enum_reaches_start(
    start: &str,
    current: &str,
    graph: &HashMap<String, HashSet<String>>,
    visited: &mut HashSet<String>,
) -> bool {
    let Some(neighbors) = graph.get(current) else {
        return false;
    };
    for next in neighbors {
        if next == start {
            return true;
        }
        if visited.insert(next.clone()) && enum_reaches_start(start, next, graph, visited) {
            return true;
        }
    }
    false
}

fn collect_bindings(
    program: &Program,
    signatures: &HashMap<String, FunctionSignature>,
    wrapped_types: &HashMap<String, String>,
    struct_defs: &HashMap<String, StructDef>,
    enum_defs: &HashMap<String, EnumDef>,
) -> Result<(Vec<BoundFunction>, Vec<BoundMethod>)> {
    let mut functions = Vec::new();
    let mut methods = Vec::new();
    let mut function_names = HashSet::new();
    let mut method_names = HashSet::new();

    for module in &program.modules {
        collect_items(
            &module.items,
            &module.path,
            &module.imports,
            signatures,
            wrapped_types,
            struct_defs,
            enum_defs,
            &mut functions,
            &mut methods,
            &mut function_names,
            &mut method_names,
        )?;
    }
    functions.sort_by(|left, right| left.lust_name.cmp(&right.lust_name));
    methods.sort_by(|left, right| {
        (&left.target_lust_name, &left.lust_name).cmp(&(&right.target_lust_name, &right.lust_name))
    });

    Ok((functions, methods))
}

#[allow(clippy::too_many_arguments)]
fn collect_items(
    items: &[Item],
    module_path: &str,
    imports: &ModuleImports,
    signatures: &HashMap<String, FunctionSignature>,
    wrapped_types: &HashMap<String, String>,
    struct_defs: &HashMap<String, StructDef>,
    enum_defs: &HashMap<String, EnumDef>,
    functions: &mut Vec<BoundFunction>,
    methods: &mut Vec<BoundMethod>,
    function_names: &mut HashSet<String>,
    method_names: &mut HashSet<String>,
) -> Result<()> {
    for item in items {
        match &item.kind {
            ItemKind::Function(function) => {
                let parsed_doc = parse_doc(function.doc.as_deref())?;
                if let Some(directive) = parsed_doc.bindgen {
                    add_function_binding(
                        function,
                        directive,
                        parsed_doc.prose,
                        signatures,
                        functions,
                        function_names,
                    )?;
                }
            }
            ItemKind::Impl(impl_block) => {
                for method in &impl_block.methods {
                    let parsed_doc = parse_doc(method.doc.as_deref())?;
                    if let Some(directive) = parsed_doc.bindgen {
                        let target_lust_name =
                            canonical_impl_target(module_path, imports, &impl_block.target_type)?;
                        add_method_binding(
                            method,
                            directive,
                            parsed_doc.prose,
                            &target_lust_name,
                            impl_block,
                            signatures,
                            wrapped_types,
                            struct_defs,
                            enum_defs,
                            methods,
                            method_names,
                        )?;
                    }
                }
            }
            ItemKind::Trait(trait_def) => {
                if parse_doc(trait_def.doc.as_deref())?.bindgen.is_some() {
                    return Err(bindgen_error(format!(
                        "@bindgen is supported on concrete impl methods, not trait '{}'",
                        trait_def.name
                    )));
                }
                for method in &trait_def.methods {
                    if parse_doc(method.doc.as_deref())?.bindgen.is_some() {
                        return Err(bindgen_error(format!(
                            "@bindgen is supported on concrete impl methods, not trait method '{}:{}'",
                            trait_def.name, method.name
                        )));
                    }
                }
            }
            ItemKind::Struct(def) => {
                parse_doc(def.doc.as_deref())?;
            }
            ItemKind::Enum(def) => {
                parse_doc(def.doc.as_deref())?;
            }
            ItemKind::Extern { items, .. } => {
                for extern_item in items {
                    match extern_item {
                        ExternItem::Function { name, doc, .. } => {
                            if parse_doc(doc.as_deref())?.bindgen.is_some() {
                                return Err(bindgen_error(format!(
                                    "@bindgen is currently supported on Lust-defined functions and concrete impl methods, not extern function '{name}'"
                                )));
                            }
                        }
                        ExternItem::Const { name, doc, .. } => {
                            if parse_doc(doc.as_deref())?.bindgen.is_some() {
                                return Err(bindgen_error(format!(
                                    "@bindgen is currently supported on Lust-defined functions and concrete impl methods, not extern constant '{name}'"
                                )));
                            }
                        }
                        ExternItem::Struct(def) => {
                            parse_doc(def.doc.as_deref())?;
                        }
                        ExternItem::Enum(def) => {
                            parse_doc(def.doc.as_deref())?;
                        }
                    }
                }
            }
            ItemKind::Module { name, items } => {
                let child_module = if name.contains('.') || name.contains("::") {
                    name.replace("::", ".")
                } else if module_path.is_empty() {
                    name.clone()
                } else {
                    format!("{module_path}.{name}")
                };
                collect_items(
                    items,
                    &child_module,
                    imports,
                    signatures,
                    wrapped_types,
                    struct_defs,
                    enum_defs,
                    functions,
                    methods,
                    function_names,
                    method_names,
                )?;
            }
            ItemKind::Script(_) | ItemKind::Use { .. } => {}
        }
    }

    Ok(())
}

fn canonical_impl_target(
    module_path: &str,
    imports: &ModuleImports,
    target_type: &Type,
) -> Result<String> {
    let raw_name = match &target_type.kind {
        TypeKind::Named(name) => name,
        TypeKind::GenericInstance { name, .. } => name,
        _ => {
            return Err(bindgen_error(format!(
                "@bindgen methods require a named impl target, found '{target_type}'"
            )));
        }
    };
    let normalized = raw_name.replace("::", ".");
    if let Some((head, tail)) = normalized.split_once('.') {
        if let Some(real_module) = imports.module_aliases.get(head) {
            return Ok(format!("{real_module}.{tail}"));
        }
        return Ok(normalized);
    }
    if let Some(imported_type) = imports.type_aliases.get(&normalized) {
        return Ok(imported_type.clone());
    }
    if module_path.is_empty() {
        Ok(normalized)
    } else {
        Ok(format!("{module_path}.{normalized}"))
    }
}

fn generated_struct_accessor_names(def: &StructDef) -> Result<HashSet<String>> {
    let mut accessors = [
        "from_handle",
        "as_handle",
        "into_handle",
        "LUST_TYPE_NAME",
        "__lust_bind_context",
        "__lust_check_context",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<HashSet<_>>();
    for field in &def.fields {
        if field.visibility != Visibility::Public {
            continue;
        }
        let (accessor, collision_key) =
            rust_identifier(&field.name, "generated struct field accessor")?;
        let (_, collision_key) = if accessors.contains(&collision_key) {
            rust_identifier(
                &format!("get_{}", collision_key.trim_start_matches("r#")),
                "generated struct field accessor",
            )?
        } else {
            (accessor, collision_key)
        };
        if !accessors.insert(collision_key.clone()) {
            return Err(bindgen_error(format!(
                "generated accessor '{collision_key}' conflicts with another field or wrapper method"
            )));
        }
    }
    Ok(accessors)
}

#[allow(clippy::too_many_arguments)]
fn add_method_binding(
    method: &FunctionDef,
    directive: BindgenDirective,
    docs: Vec<String>,
    target_lust_name: &str,
    impl_block: &crate::ast::ImplBlock,
    signatures: &HashMap<String, FunctionSignature>,
    wrapped_types: &HashMap<String, String>,
    struct_defs: &HashMap<String, StructDef>,
    enum_defs: &HashMap<String, EnumDef>,
    methods: &mut Vec<BoundMethod>,
    method_names: &mut HashSet<String>,
) -> Result<()> {
    if !impl_block.type_params.is_empty()
        || !method.type_params.is_empty()
        || !method.trait_bounds.is_empty()
    {
        return Err(bindgen_error(format!(
            "generic impl methods are not yet supported by bindgen ('{target_lust_name}.{}')",
            method.name
        )));
    }
    let Some(target_rust_name) = wrapped_types.get(target_lust_name).cloned() else {
        return Err(bindgen_error(format!(
            "cannot bind method '{}': impl target '{}' has no generated public, nongeneric Rust wrapper",
            method.name, target_lust_name
        )));
    };
    if method.visibility != Visibility::Public {
        return Err(bindgen_error(format!(
            "method '{}.{}' is marked @bindgen but is not public",
            target_lust_name, method.name
        )));
    }

    let has_receiver = method
        .params
        .iter()
        .any(|param| param.is_self || param.name == "self");
    let simple_method_name = method
        .name
        .rsplit([':', '.'])
        .next()
        .unwrap_or(&method.name);
    let lust_name = if method.name.contains(':') || method.name.contains('.') {
        method.name.clone()
    } else if has_receiver {
        format!("{target_lust_name}:{simple_method_name}")
    } else {
        format!("{target_lust_name}.{simple_method_name}")
    };
    let signature = signatures.get(&lust_name).ok_or_else(|| {
        bindgen_error(format!(
            "no checked Lust signature found for tagged method '{}'; expected runtime name '{lust_name}'",
            method.name
        ))
    })?;
    if signature.is_method != has_receiver || !signature.type_params.is_empty() {
        return Err(bindgen_error(format!(
            "generic or inconsistently typed impl methods are not supported by bindgen ('{lust_name}')"
        )));
    }
    if signature.params.len() != method.params.len() {
        return Err(bindgen_error(format!(
            "checked signature for method '{lust_name}' does not match its source parameters"
        )));
    }
    if signature.params.len() > 5 {
        return Err(bindgen_error(format!(
            "method '{lust_name}' has {} parameter(s); generated typed bindings currently support at most five",
            signature.params.len()
        )));
    }

    let receiver_index = method
        .params
        .iter()
        .position(|param| param.is_self || param.name == "self");
    if has_receiver && receiver_index != Some(0) {
        return Err(bindgen_error(format!(
            "method '{lust_name}' must declare self as its first parameter"
        )));
    }
    let target_is_struct = struct_defs.contains_key(target_lust_name);
    if has_receiver && !target_is_struct && !enum_defs.contains_key(target_lust_name) {
        return Err(bindgen_error(format!(
            "instance method '{lust_name}' requires a generated struct or enum wrapper"
        )));
    }

    let mut params = Vec::new();
    let mut seen_params = HashSet::new();
    for (index, (param, ty)) in method.params.iter().zip(&signature.params).enumerate() {
        if Some(index) == receiver_index {
            continue;
        }
        let (rust_param_name, collision_key) =
            rust_identifier(&param.name, "generated method parameter name")?;
        if !seen_params.insert(collision_key.clone()) {
            return Err(bindgen_error(format!(
                "method '{lust_name}' has duplicate Rust parameter name '{collision_key}'"
            )));
        }
        if matches!(ty.kind, TypeKind::Unit) {
            return Err(bindgen_error(format!(
                "unit-valued parameter '{}' in '{lust_name}' cannot be represented by call_typed",
                param.name
            )));
        }
        params.push((rust_param_name, ty.clone()));
    }

    let default_name = simple_method_name;
    let requested_name = directive.name.as_deref().unwrap_or(default_name);
    let (rust_name, collision_key) = rust_identifier(requested_name, "generated Rust method name")?;
    if let Some(def) = struct_defs.get(target_lust_name)
        && generated_struct_accessor_names(def)?.contains(&collision_key)
    {
        return Err(bindgen_error(format!(
            "method '{lust_name}' maps to Rust name '{collision_key}', which conflicts with a generated field accessor or wrapper method; use @bindgen(name = \"...\")"
        )));
    }
    if let Some(def) = enum_defs.get(target_lust_name)
        && def.variants.iter().any(|variant| {
            rust_identifier(&variant.name, "generated enum variant name")
                .is_ok_and(|(_, key)| key == collision_key)
        })
    {
        return Err(bindgen_error(format!(
            "method '{lust_name}' maps to Rust name '{collision_key}', which conflicts with an enum variant; use @bindgen(name = \"...\")"
        )));
    }
    let method_key = format!("{target_rust_name}::{collision_key}");
    if !method_names.insert(method_key.clone()) {
        return Err(bindgen_error(format!(
            "multiple methods on '{target_lust_name}' map to Rust method '{collision_key}'; set distinct names with @bindgen(name = \"...\")"
        )));
    }

    methods.push(BoundMethod {
        lust_name,
        rust_name,
        target_lust_name: target_lust_name.to_string(),
        target_rust_name,
        has_receiver,
        target_is_struct,
        docs,
        params,
        return_type: signature.return_type.clone(),
    });
    Ok(())
}

fn add_function_binding(
    function: &FunctionDef,
    directive: BindgenDirective,
    docs: Vec<String>,
    signatures: &HashMap<String, FunctionSignature>,
    functions: &mut Vec<BoundFunction>,
    method_names: &mut HashSet<String>,
) -> Result<()> {
    if function.visibility != Visibility::Public {
        return Err(bindgen_error(format!(
            "function '{}' is marked @bindgen but is not public",
            function.name
        )));
    }
    if function.is_method || function.name.contains(':') {
        return Err(bindgen_error(format!(
            "@bindgen is currently supported on free functions, not method '{}'",
            function.name
        )));
    }

    let signature = signatures.get(&function.name).ok_or_else(|| {
        bindgen_error(format!(
            "no checked Lust signature found for tagged function '{}'",
            function.name
        ))
    })?;
    if signature.is_method || !signature.type_params.is_empty() {
        return Err(bindgen_error(format!(
            "generic functions and methods are not yet supported by bindgen ('{}')",
            function.name
        )));
    }
    if signature.params.len() != function.params.len() {
        return Err(bindgen_error(format!(
            "checked signature for '{}' does not match its source parameters",
            function.name
        )));
    }
    if signature.params.len() > 5 {
        return Err(bindgen_error(format!(
            "function '{}' has {} parameters; generated typed bindings currently support at most five",
            function.name,
            signature.params.len()
        )));
    }

    let default_name = function
        .name
        .rsplit(['.', ':'])
        .next()
        .unwrap_or(&function.name);
    let rust_name = directive.name.as_deref().unwrap_or(default_name);
    let (rust_name, collision_key) = rust_identifier(rust_name, "generated function name")?;
    if collision_key == "from_program" {
        return Err(bindgen_error(
            "generated function name 'from_program' conflicts with the bindings constructor; use @bindgen(name = \"...\")",
        ));
    }
    if !method_names.insert(collision_key.clone()) {
        return Err(bindgen_error(format!(
            "multiple @bindgen functions map to Rust method '{}'; set distinct names with @bindgen(name = \"...\")",
            collision_key
        )));
    }

    let mut params = Vec::with_capacity(function.params.len());
    let mut seen_params = HashSet::new();
    for (param, ty) in function.params.iter().zip(&signature.params) {
        if param.is_self || param.name == "self" {
            return Err(bindgen_error(format!(
                "@bindgen function '{}' has a receiver parameter",
                function.name
            )));
        }
        let (rust_param_name, collision_key) =
            rust_identifier(&param.name, "generated parameter name")?;
        if !seen_params.insert(collision_key.clone()) {
            return Err(bindgen_error(format!(
                "function '{}' has duplicate Rust parameter name '{collision_key}'",
                function.name
            )));
        }
        if matches!(ty.kind, TypeKind::Unit) {
            return Err(bindgen_error(format!(
                "unit-valued parameter '{}' in '{}' cannot be represented by call_typed",
                param.name, function.name
            )));
        }
        params.push((rust_param_name, ty.clone()));
    }

    functions.push(BoundFunction {
        lust_name: function.name.clone(),
        rust_name,
        docs,
        params,
        return_type: signature.return_type.clone(),
    });
    Ok(())
}

fn parse_doc(doc: Option<&str>) -> Result<ParsedDoc> {
    let mut parsed = ParsedDoc::default();
    let Some(doc) = doc else {
        return Ok(parsed);
    };

    for line in doc.lines() {
        let trimmed = line.trim();
        if trimmed == "@bindgen" || trimmed.starts_with("@bindgen(") {
            if parsed.bindgen.is_some() {
                return Err(bindgen_error(
                    "a doc block may contain only one @bindgen directive",
                ));
            }
            parsed.bindgen = Some(parse_bindgen_directive(trimmed)?);
        } else if trimmed.starts_with("@bindgen") {
            return Err(bindgen_error(format!(
                "malformed bindgen directive '{trimmed}'; expected @bindgen or @bindgen(name = \"rust_name\")"
            )));
        } else {
            parsed.prose.push(line.to_string());
        }
    }

    Ok(parsed)
}

fn parse_bindgen_directive(line: &str) -> Result<BindgenDirective> {
    let suffix = line
        .strip_prefix("@bindgen")
        .expect("directive prefix checked")
        .trim();
    if suffix.is_empty() || suffix == "()" {
        return Ok(BindgenDirective::default());
    }
    if !suffix.starts_with('(') || !suffix.ends_with(')') {
        return Err(bindgen_error(format!(
            "malformed bindgen directive '{line}'; expected @bindgen(name = \"rust_name\")"
        )));
    }

    let settings_text = format!("[bindgen]\n{}\n", &suffix[1..suffix.len() - 1]);
    let settings: toml::Value = toml::from_str(&settings_text)
        .map_err(|err| bindgen_error(format!("invalid settings in '{line}': {err}")))?;
    let table = settings
        .get("bindgen")
        .and_then(toml::Value::as_table)
        .ok_or_else(|| bindgen_error(format!("invalid settings in '{line}'")))?;

    for key in table.keys() {
        if key != "name" {
            return Err(bindgen_error(format!("unknown @bindgen setting '{key}'")));
        }
    }
    let name = table
        .get("name")
        .map(|value| {
            value
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| bindgen_error("@bindgen(name = ...) requires a string value"))
        })
        .transpose()?;
    Ok(BindgenDirective { name })
}

fn generate_source(
    bindings_name: &str,
    runtime_crate_path: &str,
    functions: &[BoundFunction],
    methods: &[BoundMethod],
    nominal_types: &[BoundNominal],
    wrapped_types: &HashMap<String, String>,
    lifetime_types: &HashSet<String>,
    struct_defs: &HashMap<String, StructDef>,
    enum_defs: &HashMap<String, EnumDef>,
) -> Result<String> {
    let (bindings_name, _) = rust_identifier(bindings_name, "bindings type name")?;
    validate_rust_path(runtime_crate_path)?;
    let mut source = String::new();
    let runtime = runtime_crate_path.trim_end_matches("::");
    writeln!(source, "// @generated by `lust bindgen`; do not edit.").unwrap();
    writeln!(
        source,
        "type __LustProgramContext<'a> = ::std::rc::Rc<::std::cell::RefCell<&'a mut {runtime}::EmbeddedProgram>>;"
    )
    .unwrap();
    writeln!(
        source,
        "fn __lust_call_typed<'a, Args, R>(context: &__LustProgramContext<'a>, name: &str, args: Args) -> {runtime}::Result<R>"
    )
    .unwrap();
    writeln!(
        source,
        "where Args: {runtime}::FunctionArgs, R: {runtime}::FromLustValue {{"
    )
    .unwrap();
    writeln!(source, "    let mut program = context.try_borrow_mut().map_err(|_| {runtime}::LustError::RuntimeError {{").unwrap();
    writeln!(
        source,
        "        message: \"the embedded Lust program is already mutably borrowed\".to_string(),"
    )
    .unwrap();
    writeln!(source, "    }})?;").unwrap();
    writeln!(
        source,
        "    let program: &mut {runtime}::EmbeddedProgram = &mut **program;"
    )
    .unwrap();
    writeln!(source, "    program.call_typed(name, args)").unwrap();
    writeln!(source, "}}\n").unwrap();
    for nominal in nominal_types {
        emit_nominal_type(
            &mut source,
            nominal,
            runtime,
            wrapped_types,
            lifetime_types,
            struct_defs,
            enum_defs,
        )?;
    }
    writeln!(source, "pub struct {bindings_name}<'a> {{").unwrap();
    writeln!(source, "    __lust_context: __LustProgramContext<'a>,").unwrap();
    writeln!(source, "}}\n").unwrap();
    writeln!(source, "impl<'a> {bindings_name}<'a> {{").unwrap();
    writeln!(
        source,
        "    pub fn from_program(program: &'a mut {}::EmbeddedProgram) -> Self {{",
        runtime_crate_path.trim_end_matches("::")
    )
    .unwrap();
    writeln!(
        source,
        "        Self {{ __lust_context: ::std::rc::Rc::new(::std::cell::RefCell::new(program)) }}"
    )
    .unwrap();
    writeln!(source, "    }}\n").unwrap();

    for function in functions {
        emit_docs(&mut source, &function.docs);
        let params = function
            .params
            .iter()
            .map(|(name, ty)| {
                Ok(format!(
                    "{name}: {}",
                    rust_type(
                        ty,
                        true,
                        runtime,
                        wrapped_types,
                        lifetime_types,
                        struct_defs,
                        enum_defs,
                    )?
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let return_type = rust_type(
            &function.return_type,
            false,
            runtime,
            wrapped_types,
            lifetime_types,
            struct_defs,
            enum_defs,
        )?;
        writeln!(
            source,
            "    pub fn {}(&mut self{}{}) -> {}::Result<{return_type}> {{",
            function.rust_name,
            if params.is_empty() { "" } else { ", " },
            params.join(", "),
            runtime
        )
        .unwrap();

        let call_args = function
            .params
            .iter()
            .map(|(name, ty)| {
                if matches!(ty.kind, TypeKind::String) {
                    format!("{name}.to_string()")
                } else {
                    name.clone()
                }
            })
            .collect::<Vec<_>>();
        let args = match call_args.as_slice() {
            [] => "()".to_string(),
            [arg] => arg.clone(),
            many => format!("({})", many.join(", ")),
        };

        for (name, ty) in &function.params {
            emit_runtime_validation(
                &mut source,
                ty,
                name,
                wrapped_types,
                struct_defs,
                enum_defs,
                0,
            )?;
            emit_context_validation(
                &mut source,
                ty,
                name,
                "&self.__lust_context",
                "        ",
                runtime,
                wrapped_types,
                lifetime_types,
                struct_defs,
                enum_defs,
                0,
            )?;
        }
        writeln!(
            source,
            "        let mut __lust_result: {return_type} = __lust_call_typed(&self.__lust_context, {}, {args})?;",
            rust_string_literal(&function.lust_name)
        )
        .unwrap();
        emit_runtime_validation(
            &mut source,
            &function.return_type,
            "__lust_result",
            wrapped_types,
            struct_defs,
            enum_defs,
            0,
        )?;
        emit_context_binding(
            &mut source,
            &function.return_type,
            "__lust_result",
            "&self.__lust_context",
            "        ",
            runtime,
            wrapped_types,
            lifetime_types,
            struct_defs,
            enum_defs,
            0,
        )?;
        writeln!(source, "        Ok(__lust_result)").unwrap();
        writeln!(source, "    }}\n").unwrap();
    }

    writeln!(source, "}}\n")
        .map_err(|err| LustError::Unknown(format!("failed to format Rust bindings: {err}")))?;
    for method in methods {
        emit_bound_method(
            &mut source,
            method,
            &bindings_name,
            runtime,
            wrapped_types,
            lifetime_types,
            struct_defs,
            enum_defs,
        )?;
    }
    Ok(source)
}

fn emit_bound_method(
    source: &mut String,
    method: &BoundMethod,
    bindings_name: &str,
    runtime: &str,
    wrapped_types: &HashMap<String, String>,
    lifetime_types: &HashSet<String>,
    struct_defs: &HashMap<String, StructDef>,
    enum_defs: &HashMap<String, EnumDef>,
) -> Result<()> {
    let target_type = nominal_rust_type(
        &method.target_rust_name,
        &method.target_lust_name,
        lifetime_types,
    );
    let mut params = Vec::new();
    if !method.has_receiver {
        params.push(format!("bindings: &mut {bindings_name}<'a>"));
    } else if !method.target_is_struct {
        params.push(format!("bindings: &mut {bindings_name}<'a>"));
    }
    for (name, ty) in &method.params {
        params.push(format!(
            "{name}: {}",
            rust_type(
                ty,
                true,
                runtime,
                wrapped_types,
                lifetime_types,
                struct_defs,
                enum_defs,
            )?
        ));
    }
    let return_type = rust_type(
        &method.return_type,
        false,
        runtime,
        wrapped_types,
        lifetime_types,
        struct_defs,
        enum_defs,
    )?;

    let method_generics =
        if method.target_is_struct || lifetime_types.contains(&method.target_lust_name) {
            ""
        } else {
            "<'a>"
        };
    if method.target_is_struct {
        writeln!(source, "impl<'a> {target_type} {{").unwrap();
    } else if lifetime_types.contains(&method.target_lust_name) {
        writeln!(source, "impl<'a> {target_type} {{").unwrap();
    } else {
        writeln!(source, "impl {target_type} {{").unwrap();
    }
    emit_docs_at(source, &method.docs, "    ");
    writeln!(
        source,
        "    pub fn {}{method_generics}({}) -> {runtime}::Result<{return_type}> {{",
        method.rust_name,
        if method.has_receiver {
            format!(
                "&self{}{}",
                if params.is_empty() { "" } else { ", " },
                params.join(", ")
            )
        } else {
            params.join(", ")
        }
    )
    .unwrap();

    if method.has_receiver && method.target_is_struct {
        writeln!(source, "        let context = self.__lust_context.as_ref().ok_or_else(|| {runtime}::LustError::TypeError {{").unwrap();
        writeln!(source, "            message: \"Lust value is not bound to an embedded program; obtain it through the generated bindings facade\".to_string(),").unwrap();
        writeln!(source, "        }})?.clone();").unwrap();
    } else {
        writeln!(
            source,
            "        let context = bindings.__lust_context.clone();"
        )
        .unwrap();
        if method.has_receiver && lifetime_types.contains(&method.target_lust_name) {
            writeln!(source, "        self.__lust_check_context(&context)?;").unwrap();
        }
    }

    for (name, ty) in &method.params {
        emit_runtime_validation(source, ty, name, wrapped_types, struct_defs, enum_defs, 0)?;
        emit_context_validation(
            source,
            ty,
            name,
            "&context",
            "        ",
            runtime,
            wrapped_types,
            lifetime_types,
            struct_defs,
            enum_defs,
            0,
        )?;
    }

    let mut call_args = Vec::new();
    if method.has_receiver {
        call_args.push("self.clone()".to_string());
    }
    call_args.extend(method.params.iter().map(|(name, ty)| {
        if matches!(ty.kind, TypeKind::String) {
            format!("{name}.to_string()")
        } else {
            name.clone()
        }
    }));
    let args = match call_args.as_slice() {
        [] => "()".to_string(),
        [arg] => arg.clone(),
        many => format!("({})", many.join(", ")),
    };
    writeln!(
        source,
        "        let mut __lust_result: {return_type} = __lust_call_typed(&context, {}, {args})?;",
        rust_string_literal(&method.lust_name)
    )
    .unwrap();
    emit_runtime_validation(
        source,
        &method.return_type,
        "__lust_result",
        wrapped_types,
        struct_defs,
        enum_defs,
        0,
    )?;
    emit_context_binding(
        source,
        &method.return_type,
        "__lust_result",
        "&context",
        "        ",
        runtime,
        wrapped_types,
        lifetime_types,
        struct_defs,
        enum_defs,
        0,
    )?;
    writeln!(source, "        Ok(__lust_result)").unwrap();
    writeln!(source, "    }}").unwrap();
    writeln!(source, "}}\n").unwrap();
    Ok(())
}

fn emit_nominal_type(
    source: &mut String,
    nominal: &BoundNominal,
    runtime: &str,
    wrapped_types: &HashMap<String, String>,
    lifetime_types: &HashSet<String>,
    struct_defs: &HashMap<String, StructDef>,
    enum_defs: &HashMap<String, EnumDef>,
) -> Result<()> {
    match &nominal.kind {
        BoundNominalKind::Struct(def) => emit_struct_wrapper(
            source,
            nominal,
            def,
            runtime,
            wrapped_types,
            lifetime_types,
            struct_defs,
            enum_defs,
        ),
        BoundNominalKind::Enum(def) => emit_enum_wrapper(
            source,
            nominal,
            def,
            runtime,
            wrapped_types,
            lifetime_types,
            struct_defs,
            enum_defs,
        ),
    }
}

fn emit_struct_wrapper(
    source: &mut String,
    nominal: &BoundNominal,
    def: &StructDef,
    runtime: &str,
    wrapped_types: &HashMap<String, String>,
    lifetime_types: &HashSet<String>,
    struct_defs: &HashMap<String, StructDef>,
    enum_defs: &HashMap<String, EnumDef>,
) -> Result<()> {
    emit_docs_at(source, &nominal.docs, "");
    writeln!(source, "#[derive(Clone)]").unwrap();
    writeln!(source, "pub struct {}<'a> {{", nominal.rust_name).unwrap();
    writeln!(source, "    __lust_handle: {runtime}::StructHandle,").unwrap();
    writeln!(
        source,
        "    __lust_context: Option<__LustProgramContext<'a>>,"
    )
    .unwrap();
    writeln!(source, "}}\n").unwrap();
    writeln!(source, "impl<'a> {}<'a> {{", nominal.rust_name).unwrap();
    writeln!(
        source,
        "    pub const LUST_TYPE_NAME: &'static str = {};",
        rust_string_literal(&nominal.lust_name)
    )
    .unwrap();
    writeln!(
        source,
        "    pub fn from_handle(handle: {runtime}::StructHandle) -> {runtime}::Result<Self> {{"
    )
    .unwrap();
    writeln!(
        source,
        "        handle.ensure_exact_type(Self::LUST_TYPE_NAME)?;"
    )
    .unwrap();
    writeln!(
        source,
        "        Ok(Self {{ __lust_handle: handle, __lust_context: None }})"
    )
    .unwrap();
    writeln!(source, "    }}\n").unwrap();
    writeln!(
        source,
        "    pub fn as_handle(&self) -> &{runtime}::StructHandle {{"
    )
    .unwrap();
    writeln!(source, "        &self.__lust_handle").unwrap();
    writeln!(source, "    }}\n").unwrap();
    writeln!(
        source,
        "    pub fn into_handle(self) -> {runtime}::StructHandle {{"
    )
    .unwrap();
    writeln!(source, "        self.__lust_handle").unwrap();
    writeln!(source, "    }}").unwrap();

    let mut accessors = [
        "from_handle",
        "as_handle",
        "into_handle",
        "LUST_TYPE_NAME",
        "__lust_bind_context",
        "__lust_check_context",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<HashSet<_>>();
    for field in &def.fields {
        if field.visibility != Visibility::Public {
            continue;
        }
        let (accessor, collision_key) =
            rust_identifier(&field.name, "generated struct field accessor")?;
        let (accessor, collision_key) = if accessors.contains(&collision_key) {
            rust_identifier(
                &format!("get_{}", collision_key.trim_start_matches("r#")),
                "generated struct field accessor",
            )?
        } else {
            (accessor, collision_key)
        };
        if !accessors.insert(collision_key.clone()) {
            return Err(bindgen_error(format!(
                "generated accessor '{collision_key}' for Lust field '{}.{}' conflicts with another accessor",
                nominal.lust_name, field.name
            )));
        }
        let field_type = rust_type(
            &field.ty,
            false,
            runtime,
            wrapped_types,
            lifetime_types,
            struct_defs,
            enum_defs,
        )?;
        writeln!(
            source,
            "\n    pub fn {accessor}(&self) -> {runtime}::Result<{field_type}> {{"
        )
        .unwrap();
        if type_contains_lifetime(&field.ty, lifetime_types, wrapped_types) {
            writeln!(
                source,
                "        let mut value = self.__lust_handle.field::<{field_type}>({})?;",
                rust_string_literal(&field.name)
            )
            .unwrap();
            writeln!(
                source,
                "        if let Some(context) = &self.__lust_context {{"
            )
            .unwrap();
            emit_context_binding(
                source,
                &field.ty,
                "value",
                "context",
                "            ",
                runtime,
                wrapped_types,
                lifetime_types,
                struct_defs,
                enum_defs,
                0,
            )?;
            writeln!(source, "        }}").unwrap();
            writeln!(source, "        Ok(value)").unwrap();
        } else {
            writeln!(
                source,
                "        self.__lust_handle.field::<{field_type}>({})",
                rust_string_literal(&field.name)
            )
            .unwrap();
        }
        writeln!(source, "    }}").unwrap();
    }
    writeln!(source, "\n    fn __lust_bind_context(&mut self, context: &__LustProgramContext<'a>) -> {runtime}::Result<()> {{").unwrap();
    writeln!(
        source,
        "        if let Some(existing) = &self.__lust_context {{"
    )
    .unwrap();
    writeln!(
        source,
        "            if !::std::rc::Rc::ptr_eq(existing, context) {{"
    )
    .unwrap();
    writeln!(source, "                return Err({runtime}::LustError::TypeError {{ message: \"Lust value belongs to a different embedded program\".to_string() }});").unwrap();
    writeln!(source, "            }}").unwrap();
    writeln!(source, "        }}").unwrap();
    writeln!(
        source,
        "        self.__lust_context = Some(context.clone());"
    )
    .unwrap();
    writeln!(source, "        Ok(())").unwrap();
    writeln!(source, "    }}\n").unwrap();
    writeln!(source, "    fn __lust_check_context(&self, context: &__LustProgramContext<'a>) -> {runtime}::Result<()> {{").unwrap();
    writeln!(source, "        match &self.__lust_context {{").unwrap();
    writeln!(
        source,
        "            Some(existing) if ::std::rc::Rc::ptr_eq(existing, context) => Ok(()),"
    )
    .unwrap();
    writeln!(source, "            _ => Err({runtime}::LustError::TypeError {{ message: \"Lust value is not bound to this embedded program\".to_string() }}),").unwrap();
    writeln!(source, "        }}").unwrap();
    writeln!(source, "    }}").unwrap();
    writeln!(source, "}}\n").unwrap();

    writeln!(
        source,
        "impl<'a> {runtime}::FromLustValue for {}<'a> {{",
        nominal.rust_name
    )
    .unwrap();
    writeln!(
        source,
        "    fn from_value(value: {runtime}::Value) -> {runtime}::Result<Self> {{"
    )
    .unwrap();
    writeln!(
        source,
        "        Self::from_handle({runtime}::StructHandle::from_value(value)?)"
    )
    .unwrap();
    writeln!(source, "    }}").unwrap();
    emit_nominal_type_matcher(source, nominal, runtime);
    writeln!(
        source,
        "    fn type_description() -> &'static str {{ Self::LUST_TYPE_NAME }}"
    )
    .unwrap();
    writeln!(source, "}}\n").unwrap();

    writeln!(
        source,
        "impl<'a> {runtime}::IntoLustValue for {}<'a> {{",
        nominal.rust_name
    )
    .unwrap();
    writeln!(source, "    fn into_value(self) -> {runtime}::Value {{").unwrap();
    writeln!(
        source,
        "        <{runtime}::StructHandle as {runtime}::IntoLustValue>::into_value(self.__lust_handle)"
    )
    .unwrap();
    writeln!(source, "    }}").unwrap();
    emit_nominal_type_matcher(source, nominal, runtime);
    writeln!(
        source,
        "    fn type_description() -> &'static str {{ Self::LUST_TYPE_NAME }}"
    )
    .unwrap();
    writeln!(source, "}}\n").unwrap();
    Ok(())
}

fn emit_enum_wrapper(
    source: &mut String,
    nominal: &BoundNominal,
    def: &EnumDef,
    runtime: &str,
    wrapped_types: &HashMap<String, String>,
    lifetime_types: &HashSet<String>,
    struct_defs: &HashMap<String, StructDef>,
    enum_defs: &HashMap<String, EnumDef>,
) -> Result<()> {
    emit_docs_at(source, &nominal.docs, "");
    let rust_type_name = nominal_rust_type(&nominal.rust_name, &nominal.lust_name, lifetime_types);
    let impl_generics = if lifetime_types.contains(&nominal.lust_name) {
        "<'a>"
    } else {
        ""
    };
    writeln!(source, "#[derive(Clone)]").unwrap();
    writeln!(source, "pub enum {rust_type_name} {{").unwrap();
    let mut variant_names = HashSet::new();
    for variant in &def.variants {
        let (variant_name, collision_key) =
            rust_identifier(&variant.name, "generated enum variant name")?;
        if !variant_names.insert(collision_key.clone()) {
            return Err(bindgen_error(format!(
                "enum '{}' has multiple variants mapping to Rust variant '{collision_key}'",
                nominal.lust_name
            )));
        }
        let fields = variant.fields.as_deref().unwrap_or(&[]);
        if fields.is_empty() {
            writeln!(source, "    {variant_name},").unwrap();
        } else {
            let field_types = fields
                .iter()
                .map(|field| {
                    rust_type(
                        field,
                        false,
                        runtime,
                        wrapped_types,
                        lifetime_types,
                        struct_defs,
                        enum_defs,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            writeln!(source, "    {variant_name}({}),", field_types.join(", ")).unwrap();
        }
    }
    writeln!(source, "}}\n").unwrap();
    if lifetime_types.contains(&nominal.lust_name) {
        emit_enum_context_methods(
            source,
            nominal,
            def,
            runtime,
            wrapped_types,
            lifetime_types,
            struct_defs,
            enum_defs,
        )?;
    }

    if lifetime_types.contains(&nominal.lust_name) {
        writeln!(
            source,
            "impl<'a> {runtime}::FromLustValue for {rust_type_name} {{"
        )
        .unwrap();
    } else {
        writeln!(
            source,
            "impl {runtime}::FromLustValue for {rust_type_name} {{"
        )
        .unwrap();
    }
    writeln!(
        source,
        "    fn from_value(value: {runtime}::Value) -> {runtime}::Result<Self> {{"
    )
    .unwrap();
    writeln!(
        source,
        "        let __lust_enum = <{runtime}::EnumInstance as {runtime}::FromLustValue>::from_value(value)?;"
    )
    .unwrap();
    writeln!(
        source,
        "        __lust_enum.ensure_exact_type({})?;",
        rust_string_literal(&nominal.lust_name)
    )
    .unwrap();
    writeln!(
        source,
        "        match (__lust_enum.variant(), __lust_enum.payload_len()) {{"
    )
    .unwrap();
    for variant in &def.variants {
        let (variant_name, _) = rust_identifier(&variant.name, "generated enum variant name")?;
        let fields = variant.fields.as_deref().unwrap_or(&[]);
        let pattern = rust_string_literal(&variant.name);
        if fields.is_empty() {
            writeln!(
                source,
                "            ({pattern}, 0) => Ok(Self::{variant_name}),"
            )
            .unwrap();
        } else {
            let values = fields
                .iter()
                .enumerate()
                .map(|(index, ty)| {
                    Ok(format!(
                        "__lust_enum.payload::<{}>({index})?",
                        rust_type(
                            ty,
                            false,
                            runtime,
                            wrapped_types,
                            lifetime_types,
                            struct_defs,
                            enum_defs,
                        )?
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            writeln!(
                source,
                "            ({pattern}, {}) => Ok(Self::{variant_name}({})),",
                fields.len(),
                values.join(", ")
            )
            .unwrap();
        }
    }
    writeln!(
        source,
        "            (variant, payload_len) => Err({runtime}::LustError::RuntimeError {{"
    )
    .unwrap();
    writeln!(
        source,
        "                message: ::std::format!(\"Unexpected Lust enum variant '{}.{{}}' with {{}} payload value(s)\", variant, payload_len),",
        nominal.lust_name
    )
    .unwrap();
    writeln!(source, "            }}),").unwrap();
    writeln!(source, "        }}").unwrap();
    writeln!(source, "    }}").unwrap();
    emit_nominal_type_matcher(source, nominal, runtime);
    writeln!(
        source,
        "    fn type_description() -> &'static str {{ {} }}",
        rust_string_literal(&nominal.lust_name)
    )
    .unwrap();
    writeln!(source, "}}\n").unwrap();

    writeln!(
        source,
        "impl{impl_generics} {runtime}::IntoLustValue for {rust_type_name} {{"
    )
    .unwrap();
    writeln!(source, "    fn into_value(self) -> {runtime}::Value {{").unwrap();
    writeln!(source, "        match self {{").unwrap();
    for variant in &def.variants {
        let (variant_name, _) = rust_identifier(&variant.name, "generated enum variant name")?;
        let fields = variant.fields.as_deref().unwrap_or(&[]);
        if fields.is_empty() {
            writeln!(
                source,
                "            Self::{variant_name} => {runtime}::Value::enum_unit({}, {}),",
                rust_string_literal(&nominal.lust_name),
                rust_string_literal(&variant.name)
            )
            .unwrap();
        } else {
            let names = (0..fields.len())
                .map(|index| format!("__lust_field_{index}"))
                .collect::<Vec<_>>();
            let values = fields
                .iter()
                .zip(&names)
                .map(|(ty, name)| {
                    Ok(format!(
                        "<{as_type} as {runtime}::IntoLustValue>::into_value({name})",
                        as_type = rust_type(
                            ty,
                            false,
                            runtime,
                            wrapped_types,
                            lifetime_types,
                            struct_defs,
                            enum_defs,
                        )?
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            writeln!(
                source,
                "            Self::{variant_name}({}) => {runtime}::Value::enum_variant({}, {}, ::std::vec![{}]),",
                names.join(", "),
                rust_string_literal(&nominal.lust_name),
                rust_string_literal(&variant.name),
                values.join(", ")
            )
            .unwrap();
        }
    }
    writeln!(source, "        }}").unwrap();
    writeln!(source, "    }}").unwrap();
    emit_nominal_type_matcher(source, nominal, runtime);
    writeln!(
        source,
        "    fn type_description() -> &'static str {{ {} }}",
        rust_string_literal(&nominal.lust_name)
    )
    .unwrap();
    writeln!(source, "}}\n").unwrap();
    Ok(())
}

fn emit_nominal_type_matcher(source: &mut String, nominal: &BoundNominal, runtime: &str) {
    writeln!(
        source,
        "    fn matches_lust_type(ty: &{runtime}::Type) -> bool {{"
    )
    .unwrap();
    writeln!(
        source,
        "        {runtime}::embed::matches_lust_nominal_type(ty, {})",
        rust_string_literal(&nominal.lust_name)
    )
    .unwrap();
    writeln!(source, "    }}").unwrap();
}

fn emit_docs_at(source: &mut String, docs: &[String], indent: &str) {
    for line in docs {
        if line.is_empty() {
            writeln!(source, "{indent}///").unwrap();
        } else {
            writeln!(source, "{indent}/// {line}").unwrap();
        }
    }
}

fn nominal_rust_type(rust_name: &str, lust_name: &str, lifetime_types: &HashSet<String>) -> String {
    if lifetime_types.contains(lust_name) {
        format!("{rust_name}<'a>")
    } else {
        rust_name.to_string()
    }
}

fn rust_type(
    ty: &Type,
    direct_argument: bool,
    runtime: &str,
    wrapped_types: &HashMap<String, String>,
    lifetime_types: &HashSet<String>,
    struct_defs: &HashMap<String, StructDef>,
    enum_defs: &HashMap<String, EnumDef>,
) -> Result<String> {
    let result = match &ty.kind {
        TypeKind::Int => format!("{runtime}::LustInt"),
        TypeKind::Float => format!("{runtime}::LustFloat"),
        TypeKind::Bool => "bool".to_string(),
        TypeKind::String if direct_argument => "&str".to_string(),
        TypeKind::String => "::std::string::String".to_string(),
        TypeKind::Unit => "()".to_string(),
        TypeKind::Array(inner) => format!(
            "::std::vec::Vec<{}>",
            rust_type(
                inner,
                false,
                runtime,
                wrapped_types,
                lifetime_types,
                struct_defs,
                enum_defs,
            )?
        ),
        TypeKind::Option(inner) => format!(
            "::std::option::Option<{}>",
            rust_type(
                inner,
                false,
                runtime,
                wrapped_types,
                lifetime_types,
                struct_defs,
                enum_defs,
            )?
        ),
        TypeKind::Result(ok, err) => format!(
            "::std::result::Result<{}, {}>",
            rust_type(
                ok,
                false,
                runtime,
                wrapped_types,
                lifetime_types,
                struct_defs,
                enum_defs,
            )?,
            rust_type(
                err,
                false,
                runtime,
                wrapped_types,
                lifetime_types,
                struct_defs,
                enum_defs,
            )?
        ),
        TypeKind::Map(_, _) => format!("{runtime}::MapHandle"),
        TypeKind::Function { .. } => format!("{runtime}::FunctionHandle"),
        TypeKind::Named(name) | TypeKind::GenericInstance { name, .. }
            if wrapped_types.contains_key(name) =>
        {
            nominal_rust_type(&wrapped_types[name], name, lifetime_types)
        }
        TypeKind::Named(name) | TypeKind::GenericInstance { name, .. }
            if struct_defs.contains_key(name) =>
        {
            format!("{runtime}::StructHandle")
        }
        TypeKind::Named(name) | TypeKind::GenericInstance { name, .. }
            if enum_defs.contains_key(name) =>
        {
            format!("{runtime}::EnumInstance")
        }
        // Unknown, unions, and currently unsupported structural types retain
        // their dynamic representation instead of being guessed/coerced.
        _ => format!("{runtime}::Value"),
    };
    Ok(result)
}

fn emit_context_binding(
    source: &mut String,
    ty: &Type,
    expression: &str,
    context: &str,
    indent: &str,
    runtime: &str,
    wrapped_types: &HashMap<String, String>,
    lifetime_types: &HashSet<String>,
    struct_defs: &HashMap<String, StructDef>,
    enum_defs: &HashMap<String, EnumDef>,
    depth: usize,
) -> Result<()> {
    if !type_contains_lifetime(ty, lifetime_types, wrapped_types) {
        return Ok(());
    }
    match &ty.kind {
        TypeKind::Named(name) | TypeKind::GenericInstance { name, .. }
            if wrapped_types.contains_key(name) && lifetime_types.contains(name) =>
        {
            writeln!(
                source,
                "{indent}{expression}.__lust_bind_context({context})?;"
            )
            .unwrap();
        }
        TypeKind::Array(inner) => {
            let item_name = format!("__lust_context_item_{depth}");
            writeln!(source, "{indent}for {item_name} in &mut {expression} {{").unwrap();
            emit_context_binding(
                source,
                inner,
                &item_name,
                context,
                &format!("{indent}    "),
                runtime,
                wrapped_types,
                lifetime_types,
                struct_defs,
                enum_defs,
                depth + 1,
            )?;
            writeln!(source, "{indent}}}").unwrap();
        }
        TypeKind::Option(inner) => {
            let item_name = format!("__lust_context_item_{depth}");
            writeln!(
                source,
                "{indent}if let Some({item_name}) = &mut {expression} {{"
            )
            .unwrap();
            emit_context_binding(
                source,
                inner,
                &item_name,
                context,
                &format!("{indent}    "),
                runtime,
                wrapped_types,
                lifetime_types,
                struct_defs,
                enum_defs,
                depth + 1,
            )?;
            writeln!(source, "{indent}}}").unwrap();
        }
        TypeKind::Result(ok, err) => {
            let ok_name = format!("__lust_context_ok_{depth}");
            let err_name = format!("__lust_context_err_{depth}");
            writeln!(source, "{indent}match &mut {expression} {{").unwrap();
            writeln!(
                source,
                "{indent}    ::std::result::Result::Ok({ok_name}) => {{"
            )
            .unwrap();
            emit_context_binding(
                source,
                ok,
                &ok_name,
                context,
                &format!("{indent}        "),
                runtime,
                wrapped_types,
                lifetime_types,
                struct_defs,
                enum_defs,
                depth + 1,
            )?;
            writeln!(source, "{indent}    }}").unwrap();
            writeln!(
                source,
                "{indent}    ::std::result::Result::Err({err_name}) => {{"
            )
            .unwrap();
            emit_context_binding(
                source,
                err,
                &err_name,
                context,
                &format!("{indent}        "),
                runtime,
                wrapped_types,
                lifetime_types,
                struct_defs,
                enum_defs,
                depth + 1,
            )?;
            writeln!(source, "{indent}    }}").unwrap();
            writeln!(source, "{indent}}}").unwrap();
        }
        _ => {}
    }
    let _ = (runtime, struct_defs, enum_defs);
    Ok(())
}

fn emit_context_validation(
    source: &mut String,
    ty: &Type,
    expression: &str,
    context: &str,
    indent: &str,
    runtime: &str,
    wrapped_types: &HashMap<String, String>,
    lifetime_types: &HashSet<String>,
    struct_defs: &HashMap<String, StructDef>,
    enum_defs: &HashMap<String, EnumDef>,
    depth: usize,
) -> Result<()> {
    if !type_contains_lifetime(ty, lifetime_types, wrapped_types) {
        return Ok(());
    }
    match &ty.kind {
        TypeKind::Named(name) | TypeKind::GenericInstance { name, .. }
            if wrapped_types.contains_key(name) && lifetime_types.contains(name) =>
        {
            writeln!(
                source,
                "{indent}{expression}.__lust_check_context({context})?;"
            )
            .unwrap();
        }
        TypeKind::Array(inner) => {
            let item_name = format!("__lust_context_item_{depth}");
            writeln!(source, "{indent}for {item_name} in &{expression} {{").unwrap();
            emit_context_validation(
                source,
                inner,
                &item_name,
                context,
                &format!("{indent}    "),
                runtime,
                wrapped_types,
                lifetime_types,
                struct_defs,
                enum_defs,
                depth + 1,
            )?;
            writeln!(source, "{indent}}}").unwrap();
        }
        TypeKind::Option(inner) => {
            let item_name = format!("__lust_context_item_{depth}");
            writeln!(
                source,
                "{indent}if let Some({item_name}) = &{expression} {{"
            )
            .unwrap();
            emit_context_validation(
                source,
                inner,
                &item_name,
                context,
                &format!("{indent}    "),
                runtime,
                wrapped_types,
                lifetime_types,
                struct_defs,
                enum_defs,
                depth + 1,
            )?;
            writeln!(source, "{indent}}}").unwrap();
        }
        TypeKind::Result(ok, err) => {
            let ok_name = format!("__lust_context_ok_{depth}");
            let err_name = format!("__lust_context_err_{depth}");
            writeln!(source, "{indent}match &{expression} {{").unwrap();
            writeln!(
                source,
                "{indent}    ::std::result::Result::Ok({ok_name}) => {{"
            )
            .unwrap();
            emit_context_validation(
                source,
                ok,
                &ok_name,
                context,
                &format!("{indent}        "),
                runtime,
                wrapped_types,
                lifetime_types,
                struct_defs,
                enum_defs,
                depth + 1,
            )?;
            writeln!(source, "{indent}    }}").unwrap();
            writeln!(
                source,
                "{indent}    ::std::result::Result::Err({err_name}) => {{"
            )
            .unwrap();
            emit_context_validation(
                source,
                err,
                &err_name,
                context,
                &format!("{indent}        "),
                runtime,
                wrapped_types,
                lifetime_types,
                struct_defs,
                enum_defs,
                depth + 1,
            )?;
            writeln!(source, "{indent}    }}").unwrap();
            writeln!(source, "{indent}}}").unwrap();
        }
        _ => {}
    }
    let _ = runtime;
    Ok(())
}

fn emit_enum_context_methods(
    source: &mut String,
    nominal: &BoundNominal,
    def: &EnumDef,
    runtime: &str,
    wrapped_types: &HashMap<String, String>,
    lifetime_types: &HashSet<String>,
    struct_defs: &HashMap<String, StructDef>,
    enum_defs: &HashMap<String, EnumDef>,
) -> Result<()> {
    let rust_type_name = nominal_rust_type(&nominal.rust_name, &nominal.lust_name, lifetime_types);
    writeln!(source, "impl<'a> {rust_type_name} {{").unwrap();
    writeln!(source, "    fn __lust_bind_context(&mut self, context: &__LustProgramContext<'a>) -> {runtime}::Result<()> {{").unwrap();
    writeln!(source, "        match self {{").unwrap();
    for variant in &def.variants {
        let (variant_name, _) = rust_identifier(&variant.name, "generated enum variant name")?;
        let fields = variant.fields.as_deref().unwrap_or(&[]);
        if fields.is_empty() {
            writeln!(source, "            Self::{variant_name} => {{}},").unwrap();
            continue;
        }
        let names = (0..fields.len())
            .map(|index| format!("__lust_field_{index}"))
            .collect::<Vec<_>>();
        writeln!(
            source,
            "            Self::{variant_name}({}) => {{",
            names.join(", ")
        )
        .unwrap();
        for (ty, name) in fields.iter().zip(&names) {
            emit_context_binding(
                source,
                ty,
                name,
                "context",
                "                ",
                runtime,
                wrapped_types,
                lifetime_types,
                struct_defs,
                enum_defs,
                0,
            )?;
        }
        writeln!(source, "            }},").unwrap();
    }
    writeln!(source, "        }}").unwrap();
    writeln!(source, "        Ok(())").unwrap();
    writeln!(source, "    }}\n").unwrap();

    writeln!(source, "    fn __lust_check_context(&self, context: &__LustProgramContext<'a>) -> {runtime}::Result<()> {{").unwrap();
    writeln!(source, "        match self {{").unwrap();
    for variant in &def.variants {
        let (variant_name, _) = rust_identifier(&variant.name, "generated enum variant name")?;
        let fields = variant.fields.as_deref().unwrap_or(&[]);
        if fields.is_empty() {
            writeln!(source, "            Self::{variant_name} => {{}},").unwrap();
            continue;
        }
        let names = (0..fields.len())
            .map(|index| format!("__lust_field_{index}"))
            .collect::<Vec<_>>();
        writeln!(
            source,
            "            Self::{variant_name}({}) => {{",
            names.join(", ")
        )
        .unwrap();
        for (ty, name) in fields.iter().zip(&names) {
            emit_context_validation(
                source,
                ty,
                name,
                "context",
                "                ",
                runtime,
                wrapped_types,
                lifetime_types,
                struct_defs,
                enum_defs,
                0,
            )?;
        }
        writeln!(source, "            }},").unwrap();
    }
    writeln!(source, "        }}").unwrap();
    writeln!(source, "        Ok(())").unwrap();
    writeln!(source, "    }}").unwrap();
    writeln!(source, "}}\n").unwrap();
    Ok(())
}

fn emit_runtime_validation(
    source: &mut String,
    ty: &Type,
    expression: &str,
    wrapped_types: &HashMap<String, String>,
    struct_defs: &HashMap<String, StructDef>,
    enum_defs: &HashMap<String, EnumDef>,
    depth: usize,
) -> Result<()> {
    if !needs_runtime_validation(ty, wrapped_types, struct_defs, enum_defs) {
        return Ok(());
    }

    match &ty.kind {
        TypeKind::Array(inner) => {
            let item_name = format!("__lust_item_{depth}");
            writeln!(source, "        for {item_name} in &{expression} {{").unwrap();
            emit_runtime_validation(
                source,
                inner,
                &item_name,
                wrapped_types,
                struct_defs,
                enum_defs,
                depth + 1,
            )?;
            writeln!(source, "        }}").unwrap();
        }
        TypeKind::Option(inner) => {
            let item_name = format!("__lust_item_{depth}");
            writeln!(
                source,
                "        if let Some({item_name}) = &{expression} {{"
            )
            .unwrap();
            emit_runtime_validation(
                source,
                inner,
                &item_name,
                wrapped_types,
                struct_defs,
                enum_defs,
                depth + 1,
            )?;
            writeln!(source, "        }}").unwrap();
        }
        TypeKind::Result(ok, err) => {
            let ok_name = format!("__lust_ok_{depth}");
            let err_name = format!("__lust_err_{depth}");
            writeln!(source, "        match &{expression} {{").unwrap();
            writeln!(
                source,
                "            ::std::result::Result::Ok({ok_name}) => {{"
            )
            .unwrap();
            emit_runtime_validation(
                source,
                ok,
                &ok_name,
                wrapped_types,
                struct_defs,
                enum_defs,
                depth + 1,
            )?;
            writeln!(source, "            }}").unwrap();
            writeln!(
                source,
                "            ::std::result::Result::Err({err_name}) => {{"
            )
            .unwrap();
            emit_runtime_validation(
                source,
                err,
                &err_name,
                wrapped_types,
                struct_defs,
                enum_defs,
                depth + 1,
            )?;
            writeln!(source, "            }}").unwrap();
            writeln!(source, "        }}").unwrap();
        }
        TypeKind::Named(name) | TypeKind::GenericInstance { name, .. }
            if !wrapped_types.contains_key(name) && struct_defs.contains_key(name) =>
        {
            writeln!(
                source,
                "        {expression}.ensure_exact_type({})?;",
                rust_string_literal(name)
            )
            .unwrap();
        }
        TypeKind::Named(name) | TypeKind::GenericInstance { name, .. }
            if !wrapped_types.contains_key(name) && enum_defs.contains_key(name) =>
        {
            writeln!(
                source,
                "        {expression}.ensure_exact_type({})?;",
                rust_string_literal(name)
            )
            .unwrap();
        }
        _ => {}
    }
    Ok(())
}

fn needs_runtime_validation(
    ty: &Type,
    wrapped_types: &HashMap<String, String>,
    struct_defs: &HashMap<String, StructDef>,
    enum_defs: &HashMap<String, EnumDef>,
) -> bool {
    match &ty.kind {
        TypeKind::Array(inner) | TypeKind::Option(inner) => {
            needs_runtime_validation(inner, wrapped_types, struct_defs, enum_defs)
        }
        TypeKind::Result(ok, err) => {
            needs_runtime_validation(ok, wrapped_types, struct_defs, enum_defs)
                || needs_runtime_validation(err, wrapped_types, struct_defs, enum_defs)
        }
        TypeKind::Named(name) | TypeKind::GenericInstance { name, .. } => {
            !wrapped_types.contains_key(name)
                && (struct_defs.contains_key(name) || enum_defs.contains_key(name))
        }
        _ => false,
    }
}

fn emit_docs(source: &mut String, docs: &[String]) {
    for line in docs {
        if line.is_empty() {
            writeln!(source, "    ///").unwrap();
        } else {
            writeln!(source, "    /// {line}").unwrap();
        }
    }
}

fn rust_identifier(name: &str, description: &str) -> Result<(String, String)> {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return Err(bindgen_error(format!("{description} cannot be empty")));
    };
    if !(first == '_' || first.is_ascii_alphabetic())
        || !chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
    {
        return Err(bindgen_error(format!(
            "{description} '{name}' is not a valid Rust identifier"
        )));
    }

    const KEYWORDS: &[&str] = &[
        "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum",
        "extern", "false", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move",
        "mut", "pub", "ref", "return", "self", "Self", "static", "struct", "super", "trait",
        "true", "type", "unsafe", "use", "where", "while", "abstract", "become", "box", "do",
        "final", "macro", "override", "priv", "typeof", "unsized", "union", "virtual", "yield",
        "try", "gen",
    ];
    if KEYWORDS.contains(&name) {
        if matches!(name, "self" | "Self" | "super" | "crate") {
            return Err(bindgen_error(format!(
                "{description} '{name}' cannot be used as a Rust identifier"
            )));
        }
        Ok((format!("r#{name}"), name.to_string()))
    } else {
        Ok((name.to_string(), name.to_string()))
    }
}

fn validate_rust_path(path: &str) -> Result<()> {
    let stripped = path.trim_start_matches(':');
    if stripped.is_empty()
        || stripped.split("::").any(|segment| {
            segment.is_empty() || rust_identifier(segment, "runtime crate path").is_err()
        })
    {
        return Err(bindgen_error(format!(
            "runtime crate path '{path}' is not a valid Rust path"
        )));
    }
    Ok(())
}

fn rust_string_literal(value: &str) -> String {
    format!("{value:?}")
}

fn bindgen_error(message: impl Into<String>) -> LustError {
    LustError::TypeError {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn generates_checked_wrappers_for_tagged_functions() {
        let dir = TempDir::new().expect("temp dir");
        let entry = dir.path().join("main.lust");
        fs::write(
            &entry,
            r#"
error("bindgen must not execute module code")

struct Player
    id: int
end

---@bindgen(name = "RustStatus")
enum Status
    Ready
end

--- Adds two values.
---@bindgen
function add(a: int, b: int): int
    return a + b
end

--- @bindgen(name = "get_player")
function lookup_player(id: int): Player
    return Player { id = id }
end

---@bindgen
function maybe_player(): Option<Player>
    return Option.None
end

---@bindgen
function echo_name(name: string): string
    return name
end

---@bindgen
function current_status(): Status
    return Status.Ready
end

function internal_helper(): int
    return 1
end
"#,
        )
        .expect("write Lust source");

        let generated = RustBindingsBuilder::new(&entry)
            .bindings_name("GameBindings")
            .generate()
            .expect("generate bindings");

        assert!(generated.source.contains("pub struct GameBindings<'a>"));
        assert!(
            generated
                .source
                .contains("pub fn add(&mut self, a: ::lust::LustInt, b: ::lust::LustInt)")
        );
        assert!(generated.source.contains("\"main.add\""));
        assert!(generated.source.contains(
            "pub fn get_player(&mut self, id: ::lust::LustInt) -> ::lust::Result<Player<'a>>"
        ));
        assert!(generated.source.contains("pub struct Player<'a>"));
        assert!(
            generated
                .source
                .contains("pub fn id(&self) -> ::lust::Result<::lust::LustInt>")
        );
        assert!(generated.source.contains("::lust::StructHandle"));
        assert!(
            generated
                .source
                .contains("LUST_TYPE_NAME: &'static str = \"main.Player\"")
        );
        assert!(
            generated
                .source
                .contains("handle.ensure_exact_type(Self::LUST_TYPE_NAME)?")
        );
        assert!(generated.source.contains("::lust::EnumInstance"));
        assert!(
            generated
                .source
                .contains("ensure_exact_type(\"main.Status\")?")
        );
        assert!(
            generated
                .source
                .contains("::std::option::Option<Player<'a>>")
        );
        assert!(generated.source.contains("pub enum RustStatus"));
        assert!(generated.source.contains("::lust::Result<RustStatus>"));
        assert!(!generated.source.contains("pub struct IndexError"));
        assert!(
            generated
                .source
                .contains("pub fn echo_name(&mut self, name: &str)")
        );
        assert!(generated.source.contains(
            "__lust_call_typed(&self.__lust_context, \"main.echo_name\", name.to_string())"
        ));
        assert!(!generated.source.contains("internal_helper"));
        assert!(generated.source.contains("/// Adds two values."));
        assert!(
            generated
                .input_files
                .contains(&entry.canonicalize().unwrap())
        );
    }

    #[test]
    fn generates_instance_static_and_trait_impl_method_wrappers() {
        let dir = TempDir::new().expect("temp dir");
        let entry = dir.path().join("main.lust");
        fs::write(
            &entry,
            r#"
struct Player
    id: int
end

enum Level
    Low
    High
end

impl Level
    --- Return a numeric representation of this level.
    ---@bindgen
    function ordinal(self): int
        return 1
    end
end

trait Named
    function display_name(self): string
end

impl Player
    --- Return this player's id doubled.
    ---@bindgen
    function double_id(self): int
        return self.id * 2
    end

    --- Create a player from an id.
    ---@bindgen
    function from_id(id: int): Player
        return Player { id = id }
    end
end

impl Named for Player
    --- Describe a player via its trait implementation.
    ---@bindgen(name = "trait_display_name")
    function display_name(self): string
        return "player-" .. tostring(self.id)
    end
end
"#,
        )
        .expect("write Lust source");

        let generated = RustBindingsBuilder::new(&entry)
            .bindings_name("GameBindings")
            .generate()
            .expect("generate method bindings");

        assert!(
            generated
                .source
                .contains("pub fn double_id(&self) -> ::lust::Result<::lust::LustInt>")
        );
        assert!(
            generated
                .source
                .contains("__lust_call_typed(&context, \"main.Player:double_id\", self.clone())")
        );
        assert!(generated.source.contains(
            "pub fn from_id(bindings: &mut GameBindings<'a>, id: ::lust::LustInt) -> ::lust::Result<Player<'a>>"
        ));
        assert!(
            generated
                .source
                .contains("__lust_call_typed(&context, \"main.Player.from_id\", id)")
        );
        assert!(
            generated.source.contains(
                "pub fn trait_display_name(&self) -> ::lust::Result<::std::string::String>"
            )
        );
        assert!(
            generated.source.contains(
                "__lust_call_typed(&context, \"main.Player:display_name\", self.clone())"
            )
        );
        assert!(generated.source.contains(
            "pub fn ordinal<'a>(&self, bindings: &mut GameBindings<'a>) -> ::lust::Result<::lust::LustInt>"
        ));
        assert!(
            generated
                .source
                .contains("__lust_call_typed(&context, \"main.Level:ordinal\", self.clone())")
        );
    }

    #[test]
    fn malformed_bindgen_settings_are_rejected() {
        let dir = TempDir::new().expect("temp dir");
        let entry = dir.path().join("main.lust");
        fs::write(
            &entry,
            "---@bindgen(unknown = \"value\")\nfunction f(): int\n    return 1\nend\n",
        )
        .expect("write Lust source");

        let err = RustBindingsBuilder::new(&entry)
            .generate()
            .expect_err("unknown setting should fail");
        assert!(
            err.to_string()
                .contains("unknown @bindgen setting 'unknown'")
        );
    }

    #[test]
    fn bindgen_requires_public_non_generic_free_functions() {
        let dir = TempDir::new().expect("temp dir");
        let entry = dir.path().join("main.lust");
        fs::write(
            &entry,
            "---@bindgen\nlocal function private_api(): int\n    return 1\nend\n",
        )
        .expect("write Lust source");

        let err = RustBindingsBuilder::new(&entry)
            .generate()
            .expect_err("private function should fail");
        assert!(
            err.to_string()
                .contains("is marked @bindgen but is not public")
        );
    }
}
