use super::async_runtime::{
    AsyncRegistry, AsyncTaskEntry, AsyncTaskQueue, AsyncTaskTarget, AsyncValueFuture,
    PendingAsyncTask, signal_pair,
};
use super::conversions::{
    FromLustArgs, FromLustValue, FunctionArgs, IntoLustValue, IntoTypedValue,
};
use super::native_types::ExternRegistry;
use super::values::{EnumInstance, FunctionHandle, StructField, StructInstance, TypedValue};
use crate::ast::{
    EnumDef, FieldOwnership, FunctionDef, ImplBlock, Item, ItemKind, Span, StructDef, TraitDef,
    Type, TypeKind,
};
use crate::bytecode::{Compiler, EnumObject, NativeCallResult, Value, native_fn};
use crate::modules::{ModuleImports, ModuleLoader};
use crate::typechecker::{FunctionSignature, TypeChecker};
use crate::vm::{NativeExport, NativeExportParam, VM};
use crate::{LustConfig, LustError, Result};
use hashbrown::HashMap;
use std::cell::RefCell;
use std::fmt;
use std::future::Future;
use std::io;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_PROGRAM_ID: AtomicU64 = AtomicU64::new(1);
use std::task::{Context, Poll};

pub struct EmbeddedBuilder {
    base_dir: PathBuf,
    base_dir_provided: bool,
    modules: HashMap<String, String>,
    entry_module: Option<String>,
    config: LustConfig,
    extern_registry: ExternRegistry,
}

impl Default for EmbeddedBuilder {
    fn default() -> Self {
        Self {
            base_dir: PathBuf::from("__embedded__"),
            base_dir_provided: false,
            modules: HashMap::new(),
            entry_module: None,
            config: LustConfig::default(),
            extern_registry: ExternRegistry::new(),
        }
    }
}

impl EmbeddedBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the root directory used to resolve Lust module files.
    ///
    /// Modules registered in memory with [`Self::module`] or [`Self::add_module`]
    /// override files under this directory. If the selected entry module has no
    /// registered source, [`Self::compile`] loads it from the corresponding path under
    /// this directory.
    pub fn with_base_dir(self, base_dir: impl Into<PathBuf>) -> Self {
        self.set_base_dir(base_dir)
    }

    /// Set the root directory used to resolve Lust module files.
    ///
    /// Setting a base directory also allows [`Self::compile`] to load an
    /// unregistered entry module from disk.
    pub fn set_base_dir(mut self, base_dir: impl Into<PathBuf>) -> Self {
        self.base_dir = base_dir.into();
        self.base_dir_provided = true;
        self
    }

    /// Register an in-memory source for a module.
    ///
    /// This source takes precedence over the corresponding file under the
    /// configured base directory.
    pub fn module(mut self, module_path: impl Into<String>, source: impl Into<String>) -> Self {
        self.modules.insert(module_path.into(), source.into());
        self
    }

    pub fn enable_stdlib_module<S: AsRef<str>>(mut self, module: S) -> Self {
        self.config.enable_module(module);
        self
    }

    pub fn add_stdlib_module<S: AsRef<str>>(mut self, module: S) -> Self {
        self.config.enable_module(module);
        self
    }

    pub fn declare_struct(mut self, def: StructDef) -> Self {
        self.extern_registry.add_struct(def);
        self
    }

    pub fn declare_enum(mut self, def: EnumDef) -> Self {
        self.extern_registry.add_enum(def);
        self
    }

    pub fn declare_trait(mut self, def: TraitDef) -> Self {
        self.extern_registry.add_trait(def);
        self
    }

    pub fn declare_impl(mut self, impl_block: ImplBlock) -> Self {
        self.extern_registry.add_impl(impl_block);
        self
    }

    pub fn declare_function(mut self, func: FunctionDef) -> Self {
        self.extern_registry.add_function(func);
        self
    }

    pub fn with_extern_registry(mut self, registry: ExternRegistry) -> Self {
        self.extern_registry = registry;
        self
    }

    pub fn extern_registry_mut(&mut self) -> &mut ExternRegistry {
        &mut self.extern_registry
    }

    pub fn with_config(mut self, config: LustConfig) -> Self {
        self.config = config;
        self
    }

    pub fn set_config(mut self, config: LustConfig) -> Self {
        self.config = config;
        self
    }

    /// Enable low memory mode for constrained environments (e.g., ESP32).
    /// Reduces compilation memory by not storing expression/variable type info.
    pub fn low_memory_mode(mut self) -> Self {
        self.config.set_low_memory_mode(true);
        self
    }

    /// Enable minimal runtime types to reduce memory in compiled functions.
    /// Strips register type info from Function objects.
    pub fn minimal_runtime_types(mut self) -> Self {
        self.config.set_minimal_runtime_types(true);
        self
    }

    pub fn add_module(
        &mut self,
        module_path: impl Into<String>,
        source: impl Into<String>,
    ) -> &mut Self {
        self.modules.insert(module_path.into(), source.into());
        self
    }

    /// Select the entry module to compile and execute.
    ///
    /// Its source must either be registered with [`Self::module`] / [`Self::add_module`]
    /// or be available under a base directory set with [`Self::with_base_dir`].
    pub fn entry_module(mut self, module_path: impl Into<String>) -> Self {
        self.set_entry_module(module_path);
        self
    }

    pub fn set_entry_module(&mut self, module_path: impl Into<String>) -> &mut Self {
        self.entry_module = Some(module_path.into());
        self
    }

    pub fn compile(self) -> Result<EmbeddedProgram> {
        let entry_module = self
            .entry_module
            .ok_or_else(|| LustError::Unknown("No entry module configured for embedding".into()))?;
        let has_entry_source = self.modules.contains_key(&entry_module);
        if !has_entry_source && !self.base_dir_provided {
            return Err(LustError::Unknown(format!(
                "Entry module '{}' was not provided via EmbeddedBuilder::module and no base directory was set",
                entry_module
            )));
        }

        let overrides: HashMap<PathBuf, String> = self
            .modules
            .into_iter()
            .map(|(module, source)| (module_path_to_file(&self.base_dir, &module), source))
            .collect();
        compile_in_memory(
            self.base_dir,
            entry_module,
            overrides,
            self.config,
            self.extern_registry,
        )
    }

    /// OPT-14: Precompile modules into in-memory bytecode without instantiating the VM.
    /// The resulting [`CompiledBytecode`] can be used to spawn multiple [`EmbeddedProgram`]
    /// instances with near-zero (< 2 µs) cold start latency.
    pub fn precompile(self) -> Result<CompiledBytecode> {
        let entry_module = self
            .entry_module
            .ok_or_else(|| LustError::Unknown("No entry module configured for embedding".into()))?;
        let has_entry_source = self.modules.contains_key(&entry_module);
        if !has_entry_source && !self.base_dir_provided {
            return Err(LustError::Unknown(format!(
                "Entry module '{}' was not provided via EmbeddedBuilder::module and no base directory was set",
                entry_module
            )));
        }

        let overrides: HashMap<PathBuf, String> = self
            .modules
            .into_iter()
            .map(|(module, source)| (module_path_to_file(&self.base_dir, &module), source))
            .collect();
        compile_to_bytecode(
            self.base_dir,
            entry_module,
            overrides,
            self.config,
            self.extern_registry,
        )
    }
}

/// OPT-14: Precompiled bytecode and metadata cache for instant (< 2 µs) cold start.
#[derive(Clone, Debug)]
pub struct CompiledBytecode {
    pub(crate) entry_module: String,
    pub(crate) entry_script: Option<String>,
    pub(crate) functions: Vec<crate::bytecode::Function>,
    pub(crate) signatures: HashMap<String, FunctionSignature>,
    pub(crate) struct_defs: HashMap<String, StructDef>,
    pub(crate) enum_defs: HashMap<String, EnumDef>,
    pub(crate) trait_impls: Vec<(String, String)>,
    pub(crate) init_funcs: Vec<(String, String)>,
    pub(crate) config: LustConfig,
    pub(crate) extern_registry: ExternRegistry,
}

impl CompiledBytecode {
    pub fn entry_module(&self) -> &str {
        &self.entry_module
    }

    pub fn function_count(&self) -> usize {
        self.functions.len()
    }

    pub fn functions(&self) -> &[crate::bytecode::Function] {
        &self.functions
    }
}

pub struct EmbeddedProgram {
    #[allow(dead_code)]
    program_id: u64,
    #[allow(dead_code)]
    program_token: Rc<()>,
    vm: VM,
    signatures: HashMap<String, FunctionSignature>,
    struct_defs: HashMap<String, StructDef>,
    enum_defs: HashMap<String, EnumDef>,
    entry_script: Option<String>,
    entry_module: String,
    async_registry: Rc<RefCell<AsyncRegistry>>,
    last_typed_target: Option<(String, usize)>,
}

/// OPT-06: A pre-resolved typed function handle for zero-overhead host calls.
pub struct TypedHandle<Args, Ret> {
    pub(crate) func_idx: usize,
    pub(crate) expected_name: String,
    pub(crate) _marker: std::marker::PhantomData<(Args, Ret)>,
}

pub struct AsyncDriver<'a> {
    program: &'a mut EmbeddedProgram,
}

impl<'a> AsyncDriver<'a> {
    pub fn new(program: &'a mut EmbeddedProgram) -> Self {
        Self { program }
    }

    pub fn poll(&mut self) -> Result<()> {
        self.program.poll_async_tasks()
    }

    pub fn pump_until_idle(&mut self) -> Result<()> {
        while self.program.has_pending_async_tasks() {
            self.program.poll_async_tasks()?;
        }
        Ok(())
    }

    pub fn has_pending(&self) -> bool {
        self.program.has_pending_async_tasks()
    }
}

impl EmbeddedProgram {
    pub fn builder() -> EmbeddedBuilder {
        EmbeddedBuilder::default()
    }

    /// OPT-14: Instantiate an [`EmbeddedProgram`] from precompiled bytecode in < 2 µs.
    pub fn from_compiled(bytecode: &CompiledBytecode) -> Result<Self> {
        let mut vm = VM::with_config(&bytecode.config);
        vm.load_functions(bytecode.functions.clone());
        vm.register_structs(&bytecode.struct_defs);
        bytecode.extern_registry.register_type_stubs(&mut vm);
        for (type_name, trait_name) in &bytecode.trait_impls {
            vm.register_trait_impl(type_name.clone(), trait_name.clone());
        }
        bytecode.extern_registry.register_trait_impls(&mut vm);

        for (module_path, init) in &bytecode.init_funcs {
            let value = vm.call(init, Vec::new())?;
            vm.set_global(module_path, value);
        }

        let program_id = {
            let id = NEXT_PROGRAM_ID.fetch_add(1, Ordering::Relaxed);
            if id == 0 {
                panic!("Fatal: EmbeddedProgram program_id counter overflowed u64");
            }
            id
        };

        Ok(EmbeddedProgram {
            program_id,
            program_token: Rc::new(()),
            vm,
            signatures: bytecode.signatures.clone(),
            struct_defs: bytecode.struct_defs.clone(),
            enum_defs: bytecode.enum_defs.clone(),
            entry_script: bytecode.entry_script.clone(),
            entry_module: bytecode.entry_module.clone(),
            async_registry: Rc::new(RefCell::new(AsyncRegistry::new())),
            last_typed_target: None,
        })
    }

    /// Alias for [`Self::from_compiled`].
    pub fn from_compiled_bytecode(bytecode: &CompiledBytecode) -> Result<Self> {
        Self::from_compiled(bytecode)
    }

    /// OPT-14: Precompile using an [`EmbeddedBuilder`].
    pub fn precompile(builder: EmbeddedBuilder) -> Result<CompiledBytecode> {
        builder.precompile()
    }

    pub fn vm_mut(&mut self) -> &mut VM {
        &mut self.vm
    }

    pub fn jit_stats(&self) -> crate::jit::JitStats {
        self.vm.jit_stats()
    }

    pub fn set_gas_budget(&mut self, limit: u64) {
        self.vm.set_gas_budget(limit);
    }

    pub fn clear_gas_budget(&mut self) {
        self.vm.clear_gas_budget();
    }

    pub fn reset_gas_counter(&mut self) {
        self.vm.reset_gas_counter();
    }

    pub fn gas_used(&self) -> u64 {
        self.vm.gas_used()
    }

    pub fn gas_remaining(&self) -> Option<u64> {
        self.vm.gas_remaining()
    }

    /// Set a memory budget in bytes.
    ///
    /// While a memory budget is active, effective execution safely switches to the
    /// interpreter (Stage A) so existing logical array and growth checks apply.
    /// This does not charge all allocations or guarantee RSS/physical memory containment;
    /// other allocation classes and complete process containment are out of scope.
    /// Configured JIT preferences are preserved and will resume if the budget is cleared.
    ///
    /// # Errors
    /// Returns `Err` if called while VM execution is active (e.g. inside a native callback,
    /// reentrant call, or executing JIT/bytecode frame). An error leaves the previous
    /// budget policy and counters completely unchanged.
    #[must_use = "memory budget mutations are fallible and must not be ignored; an ignored error leaves the previous budget policy and counters unchanged"]
    pub fn set_memory_budget_bytes(&mut self, limit_bytes: usize) -> Result<()> {
        self.vm.set_memory_budget_bytes(limit_bytes)
    }

    /// Set a memory budget in kilobytes (1 KB = 1024 bytes).
    ///
    /// While a memory budget is active, effective execution safely switches to the
    /// interpreter (Stage A) so existing logical array and growth checks apply.
    /// This does not charge all allocations or guarantee RSS/physical memory containment;
    /// other allocation classes and complete process containment are out of scope.
    ///
    /// # Errors
    /// Returns `Err` if called while VM execution is active (e.g. inside a native callback,
    /// reentrant call, or executing JIT/bytecode frame). An error leaves the previous
    /// budget policy and counters completely unchanged.
    #[must_use = "memory budget mutations are fallible and must not be ignored; an ignored error leaves the previous budget policy and counters unchanged"]
    pub fn set_memory_budget_kb(&mut self, limit_kb: u64) -> Result<()> {
        self.vm.set_memory_budget_kb(limit_kb)
    }

    /// Clear the active memory budget and reset the memory counter.
    ///
    /// When cleared, execution transitions back to normal unmetered execution
    /// (with JIT enabled if configured), and previously compiled code is invalidated
    /// so stale entries are never reused.
    ///
    /// # Errors
    /// Returns `Err` if called while VM execution is active (e.g. inside a native callback,
    /// reentrant call, or executing JIT/bytecode frame). An error leaves the previous
    /// budget policy and counters completely unchanged.
    #[must_use = "memory budget mutations are fallible and must not be ignored; an ignored error leaves the previous budget policy and counters unchanged"]
    pub fn clear_memory_budget(&mut self) -> Result<()> {
        self.vm.clear_memory_budget()
    }

    /// Reset the cumulative memory usage counter to zero without clearing the limit.
    ///
    /// # Errors
    /// Returns `Err` if called while VM execution is active (e.g. inside a native callback,
    /// reentrant call, or executing JIT/bytecode frame). An error leaves the previous
    /// budget policy and counters completely unchanged.
    #[must_use = "memory budget mutations are fallible and must not be ignored; an ignored error leaves the previous budget policy and counters unchanged"]
    pub fn reset_memory_counter(&mut self) -> Result<()> {
        self.vm.reset_memory_counter()
    }

    /// Whether JIT compilation and native execution are configured to be enabled.
    #[inline]
    pub fn jit_configured_enabled(&self) -> bool {
        self.vm.jit_configured_enabled()
    }

    /// Whether JIT execution is effectively active.
    ///
    /// Returns true only when JIT is configured enabled AND no memory budget is active.
    #[inline]
    pub fn effective_jit_enabled(&self) -> bool {
        self.vm.effective_jit_enabled()
    }

    pub fn memory_used_bytes(&self) -> usize {
        self.vm.memory_used_bytes()
    }

    pub fn memory_remaining_bytes(&self) -> Option<usize> {
        self.vm.memory_remaining_bytes()
    }

    pub fn dump_externs_to_dir(&self, output_root: impl AsRef<Path>) -> io::Result<Vec<PathBuf>> {
        self.vm.dump_externs_to_dir(output_root)
    }

    pub(crate) fn vm(&self) -> &VM {
        &self.vm
    }

    pub fn global_names(&self) -> Vec<String> {
        self.vm.global_names()
    }

    pub fn globals(&self) -> Vec<(String, Value)> {
        self.vm.globals_snapshot()
    }

    pub fn signature(&self, function_name: &str) -> Option<&FunctionSignature> {
        self.find_signature(function_name).map(|(_, sig)| sig)
    }

    pub fn typed_functions(&self) -> impl Iterator<Item = (&String, &FunctionSignature)> {
        self.signatures.iter()
    }

    pub fn struct_definition(&self, type_name: &str) -> Option<&StructDef> {
        self.struct_defs.get(type_name)
    }

    pub fn enum_definition(&self, type_name: &str) -> Option<&EnumDef> {
        self.enum_defs.get(type_name)
    }

    fn find_signature(&self, name: &str) -> Option<(String, &FunctionSignature)> {
        if let Some(sig) = self.signatures.get(name) {
            return Some((name.to_string(), sig));
        }

        for candidate in self.signature_lookup_candidates(name) {
            if let Some(sig) = self.signatures.get(&candidate) {
                return Some((candidate, sig));
            }
        }

        let matches = self
            .signatures
            .iter()
            .filter_map(|(key, sig)| {
                if Self::simple_name(key) == name {
                    Some((key, sig))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        if matches.len() == 1 {
            let (key, sig) = matches[0];
            return Some((key.clone(), sig));
        }

        None
    }

    fn resolve_signature(&self, name: &str) -> Result<(String, &FunctionSignature)> {
        if let Some(found) = self.find_signature(name) {
            return Ok(found);
        }

        let matches = self
            .signatures
            .keys()
            .filter(|key| Self::simple_name(key) == name)
            .count();
        if matches > 1 {
            return Err(LustError::TypeError {
                message: format!(
                    "Cannot register native '{}': multiple matching functions found; specify a fully qualified name",
                    name
                ),
            });
        }

        Err(LustError::TypeError {
            message: format!(
                "Cannot register native '{}': function not declared in Lust source",
                name
            ),
        })
    }

    fn signature_lookup_candidates(&self, name: &str) -> Vec<String> {
        let mut candidates: Vec<String> = Vec::new();
        if name.contains("::") {
            candidates.push(name.replace("::", "."));
        }

        if name.contains('.') {
            candidates.push(name.replace('.', "::"));
        }

        if !name.contains('.') && !name.contains("::") {
            let module = &self.entry_module;
            candidates.push(format!("{}.{}", module, name));
            candidates.push(format!("{}::{}", module, name));
        }

        candidates
    }

    fn simple_name(name: &str) -> &str {
        name.rsplit(['.', ':']).next().unwrap_or(name)
    }

    fn register_native_with_aliases<F>(&mut self, requested_name: &str, canonical: String, func: F)
    where
        F: Fn(&[Value]) -> std::result::Result<NativeCallResult, String> + 'static,
    {
        let value = Value::NativeFunction(native_fn(func));
        let mut aliases: Vec<String> = Vec::new();
        aliases.push(canonical.clone());
        let canonical_normalized = normalize_global_name(&canonical);
        if canonical_normalized != canonical {
            aliases.push(canonical_normalized);
        }

        if requested_name != canonical {
            aliases.push(requested_name.to_string());
            let normalized = normalize_global_name(requested_name);
            if normalized != requested_name {
                aliases.push(normalized);
            }
        }

        aliases.sort();
        aliases.dedup();
        #[cfg(debug_assertions)]
        eprintln!(
            "register_native_with_aliases requested='{}' canonical='{}' aliases={:?}",
            requested_name, canonical, aliases
        );
        for key in aliases {
            self.vm.register_native(key, value.clone());
        }
    }

    pub fn get_global_value(&self, name: &str) -> Option<Value> {
        let normalized = normalize_global_name(name);
        self.vm.get_global(&normalized)
    }

    pub fn get_typed_global<T: FromLustValue>(&self, name: &str) -> Result<Option<T>> {
        let normalized = normalize_global_name(name);
        match self.vm.get_global(&normalized) {
            Some(value) => T::from_value(value).map(Some),
            None => Ok(None),
        }
    }

    pub fn set_global_value<V: IntoTypedValue>(&mut self, name: impl Into<String>, value: V) {
        let name_string = name.into();
        let normalized = normalize_global_name(&name_string);
        let value = value.into_typed_value().into_value();
        self.vm.set_global(normalized, value);
    }

    pub fn struct_instance<I>(
        &self,
        type_name: impl Into<String>,
        fields: I,
    ) -> Result<StructInstance>
    where
        I: IntoIterator,
        I::Item: Into<StructField>,
    {
        let type_name = type_name.into();
        let def = self
            .struct_defs
            .get(&type_name)
            .ok_or_else(|| LustError::TypeError {
                message: format!("Unknown struct '{}'", type_name),
            })?;
        let mut provided: HashMap<String, TypedValue> = fields
            .into_iter()
            .map(|field| {
                let field: StructField = field.into();
                field.into_parts()
            })
            .collect();
        let mut ordered_fields: Vec<(Rc<String>, Value)> = Vec::with_capacity(def.fields.len());
        for field in &def.fields {
            let typed_value = provided
                .remove(&field.name)
                .ok_or_else(|| LustError::TypeError {
                    message: format!(
                        "Struct '{}' is missing required field '{}'",
                        type_name, field.name
                    ),
                })?;

            let matches_ref_inner = matches!(field.ownership, FieldOwnership::Weak)
                && matches!(typed_value.as_value(), Value::Struct { .. });
            if !typed_value.matches(&field.ty) && !matches_ref_inner {
                return Err(LustError::TypeError {
                    message: format!(
                        "Struct field '{}' expects Rust value of type '{}' but received '{}'",
                        field.name,
                        field.ty,
                        typed_value.description()
                    ),
                });
            }

            ordered_fields.push((Rc::new(field.name.clone()), typed_value.into_value()));
        }

        if !provided.is_empty() {
            let mut unexpected: Vec<String> = provided.into_keys().collect();
            unexpected.sort();
            return Err(LustError::TypeError {
                message: format!(
                    "Struct '{}' received unexpected field(s): {}",
                    type_name,
                    unexpected.join(", ")
                ),
            });
        }

        let value = self.vm.instantiate_struct(&type_name, ordered_fields)?;
        Ok(StructInstance::new(type_name.clone(), value))
    }

    pub fn enum_variant(
        &self,
        type_name: impl Into<String>,
        variant: impl Into<String>,
    ) -> Result<EnumInstance> {
        self.enum_variant_with(type_name, variant, std::iter::empty::<Value>())
    }

    pub fn enum_variant_with<I, V>(
        &self,
        type_name: impl Into<String>,
        variant: impl Into<String>,
        payload: I,
    ) -> Result<EnumInstance>
    where
        I: IntoIterator<Item = V>,
        V: IntoTypedValue,
    {
        let type_name = type_name.into();
        let variant_name = variant.into();
        let def = self
            .enum_defs
            .get(&type_name)
            .ok_or_else(|| LustError::TypeError {
                message: format!("Unknown enum '{}'", type_name),
            })?;
        let enum_variant = def
            .variants
            .iter()
            .find(|v| v.name == variant_name)
            .ok_or_else(|| LustError::TypeError {
                message: format!(
                    "Enum '{}' has no variant named '{}'",
                    type_name, variant_name
                ),
            })?;
        let mut values: Vec<TypedValue> =
            payload.into_iter().map(|v| v.into_typed_value()).collect();
        let coerced_values: Option<Vec<Value>> = match &enum_variant.fields {
            None => {
                if !values.is_empty() {
                    return Err(LustError::TypeError {
                        message: format!(
                            "Enum variant '{}.{}' does not accept payload values",
                            type_name, variant_name
                        ),
                    });
                }

                None
            }

            Some(field_types) => {
                if values.len() != field_types.len() {
                    return Err(LustError::TypeError {
                        message: format!(
                            "Enum variant '{}.{}' expects {} value(s) but {} were supplied",
                            type_name,
                            variant_name,
                            field_types.len(),
                            values.len()
                        ),
                    });
                }

                let mut collected = Vec::with_capacity(field_types.len());
                for (idx, (typed_value, field_ty)) in
                    values.drain(..).zip(field_types.iter()).enumerate()
                {
                    if !typed_value.matches(field_ty) {
                        return Err(LustError::TypeError {
                            message: format!(
                                "Enum variant '{}.{}' field {} expects Lust type '{}' but Rust provided '{}'",
                                type_name,
                                variant_name,
                                idx + 1,
                                field_ty,
                                typed_value.description()
                            ),
                        });
                    }

                    collected.push(typed_value.into_value());
                }

                Some(collected)
            }
        };
        Ok(EnumInstance::new(
            type_name.clone(),
            variant_name.clone(),
            Value::Enum(EnumObject::new(
                type_name.clone(),
                variant_name.clone(),
                coerced_values,
            )),
        ))
    }

    pub fn register_native_fn<F>(&mut self, name: impl Into<String>, func: F)
    where
        F: Fn(&[Value]) -> std::result::Result<NativeCallResult, String> + 'static,
    {
        let native = Value::NativeFunction(native_fn(func));
        self.vm.register_native(name, native);
    }

    pub fn register_async_native<F, Fut>(&mut self, name: impl Into<String>, func: F) -> Result<()>
    where
        F: Fn(Vec<Value>) -> Fut + 'static,
        Fut: Future<Output = std::result::Result<Value, String>> + 'static,
    {
        let registry = self.async_registry.clone();
        let name_string = name.into();
        let handler = move |values: &[Value]| -> std::result::Result<NativeCallResult, String> {
            let args: Vec<Value> = values.to_vec();
            let future: AsyncValueFuture = Box::pin(func(args));
            VM::with_current(|vm| {
                let handle = vm
                    .current_task_handle()
                    .ok_or_else(|| "Async native functions require a running task".to_string())?;
                let mut registry = registry.borrow_mut();
                if registry.has_pending_for(handle) {
                    return Err(format!(
                        "Task {} already has a pending async native call",
                        handle.id()
                    ));
                }

                registry.register(AsyncTaskEntry::new(
                    AsyncTaskTarget::ScriptTask(handle),
                    future,
                ));
                Ok(NativeCallResult::Yield(Value::Nil))
            })
        };
        self.register_native_fn(name_string, handler);
        Ok(())
    }

    pub fn register_typed_native<Args, R, F>(&mut self, name: &str, func: F) -> Result<()>
    where
        Args: FromLustArgs,
        R: IntoLustValue + FromLustValue,
        F: Fn(Args) -> std::result::Result<R, String> + 'static,
    {
        let (canonical, signature) = self.resolve_signature(name)?;
        if !Args::matches_signature(&signature.params) {
            return Err(LustError::TypeError {
                message: format!(
                    "Native '{}' argument types do not match Lust signature",
                    name
                ),
            });
        }

        ensure_return_type::<R>(name, &signature.return_type)?;
        let return_type = signature.return_type.clone();
        self.vm
            .record_exported_native(native_export_from_signature(&canonical, signature));
        let native_name = name.to_string();
        let handler = move |values: &[Value]| -> std::result::Result<NativeCallResult, String> {
            let args = Args::from_values(values)?;
            let result = func(args)?;
            let value = result.into_value();
            VM::with_current(|vm| {
                if vm.value_matches_type(&value, &return_type) {
                    Ok(NativeCallResult::Return(value))
                } else {
                    Err(format!(
                        "Native '{}' must return {}, got {:?}",
                        native_name,
                        return_type,
                        value.type_of()
                    ))
                }
            })
        };
        self.register_native_with_aliases(name, canonical, handler);
        Ok(())
    }

    pub fn register_async_typed_native<Args, R, F, Fut>(
        &mut self,
        name: &str,
        func: F,
    ) -> Result<()>
    where
        Args: FromLustArgs,
        R: IntoLustValue + FromLustValue,
        F: Fn(Args) -> Fut + 'static,
        Fut: Future<Output = std::result::Result<R, String>> + 'static,
    {
        let (canonical, signature) = self.resolve_signature(name)?;
        let signature = signature.clone();
        self.vm
            .record_exported_native(native_export_from_signature(&canonical, &signature));
        if !Args::matches_signature(&signature.params) {
            return Err(LustError::TypeError {
                message: format!(
                    "Native '{}' argument types do not match Lust signature",
                    name
                ),
            });
        }
        ensure_return_type::<R>(name, &signature.return_type)?;

        let registry = self.async_registry.clone();
        let return_type = signature.return_type.clone();
        let handler = move |values: &[Value]| -> std::result::Result<NativeCallResult, String> {
            let args = Args::from_values(values)?;
            let future = func(args);
            let mapped = async move {
                match future.await {
                    Ok(result) => Ok(result.into_value()),
                    Err(err) => Err(err),
                }
            };
            let future: AsyncValueFuture = Box::pin(mapped);
            VM::with_current(|vm| {
                let handle = vm
                    .current_task_handle()
                    .ok_or_else(|| "Async native functions require a running task".to_string())?;
                let mut registry = registry.borrow_mut();
                if registry.has_pending_for(handle) {
                    return Err(format!(
                        "Task {} already has a pending async native call",
                        handle.id()
                    ));
                }

                registry.register(
                    AsyncTaskEntry::new(AsyncTaskTarget::ScriptTask(handle), future)
                        .with_expected_type(return_type.clone()),
                );
                Ok(NativeCallResult::Yield(Value::Nil))
            })
        };
        self.register_native_with_aliases(name, canonical, handler);
        Ok(())
    }

    pub fn register_async_task_native<Args, R, F, Fut>(&mut self, name: &str, func: F) -> Result<()>
    where
        Args: FromLustArgs,
        R: IntoLustValue + FromLustValue,
        F: Fn(Args) -> Fut + 'static,
        Fut: Future<Output = std::result::Result<R, String>> + 'static,
    {
        let (canonical, signature) = self.resolve_signature(name)?;
        let signature = signature.clone();
        self.vm
            .record_exported_native(native_export_from_signature(&canonical, &signature));
        if !Args::matches_signature(&signature.params) {
            return Err(LustError::TypeError {
                message: format!(
                    "Native '{}' argument types do not match Lust signature",
                    name
                ),
            });
        }

        let registry = self.async_registry.clone();
        let handler = move |values: &[Value]| -> std::result::Result<NativeCallResult, String> {
            let args = Args::from_values(values)?;
            let future = func(args);
            let mapped = async move {
                match future.await {
                    Ok(result) => Ok(result.into_value()),
                    Err(err) => Err(err),
                }
            };
            let future: AsyncValueFuture = Box::pin(mapped);
            VM::with_current(|vm| {
                let mut registry = registry.borrow_mut();
                let handle = vm.create_native_future_task();
                let entry = AsyncTaskEntry::new(AsyncTaskTarget::NativeTask(handle), future);
                registry.register(entry);
                Ok(NativeCallResult::Return(Value::task(handle)))
            })
        };
        self.register_native_with_aliases(name, canonical, handler);
        Ok(())
    }

    pub fn register_async_task_queue<Args, R>(
        &mut self,
        name: &str,
        queue: AsyncTaskQueue<Args, R>,
    ) -> Result<()>
    where
        Args: FromLustArgs + 'static,
        R: IntoLustValue + FromLustValue + 'static,
    {
        let (canonical, signature) = self.resolve_signature(name)?;
        let signature = signature.clone();
        self.vm
            .record_exported_native(native_export_from_signature(&canonical, &signature));
        if !Args::matches_signature(&signature.params) {
            return Err(LustError::TypeError {
                message: format!(
                    "Native '{}' argument types do not match Lust signature",
                    name
                ),
            });
        }

        let registry = self.async_registry.clone();
        let queue_clone = queue.clone();
        let handler = move |values: &[Value]| -> std::result::Result<NativeCallResult, String> {
            let args = Args::from_values(values)?;
            let (completer, signal_future) = signal_pair::<R>();
            let future: AsyncValueFuture = Box::pin(async move {
                match signal_future.await {
                    Ok(result) => Ok(result.into_value()),
                    Err(err) => Err(err),
                }
            });

            VM::with_current(|vm| {
                let mut registry = registry.borrow_mut();
                let handle = vm.create_native_future_task();
                let entry = AsyncTaskEntry::new(AsyncTaskTarget::NativeTask(handle), future);
                registry.register(entry);
                queue_clone.push(PendingAsyncTask::new(handle, args, completer));
                Ok(NativeCallResult::Return(Value::task(handle)))
            })
        };
        self.register_native_with_aliases(name, canonical, handler);
        Ok(())
    }

    pub fn get_typed_handle<Args, Ret>(&mut self, function_name: &str) -> Result<TypedHandle<Args, Ret>>
    where
        Args: FunctionArgs,
        Ret: FromLustValue,
    {
        let signature = self
            .signatures
            .get(function_name)
            .ok_or_else(|| LustError::TypeError {
                message: format!(
                    "No type information available for function '{}'; \
                     use call_raw if the function is dynamically typed",
                    function_name
                ),
            })?;
        Args::validate_signature(function_name, &signature.params)?;
        ensure_return_type::<Ret>(function_name, &signature.return_type)?;

        let func_idx = self
            .vm
            .functions()
            .iter()
            .position(|f| f.name == function_name)
            .ok_or_else(|| LustError::RuntimeError {
                message: format!("Function not found: {}", function_name),
            })?;

        Ok(TypedHandle {
            func_idx,
            expected_name: function_name.to_string(),
            _marker: std::marker::PhantomData,
        })
    }

    pub fn call_handle<Args, Ret>(
        &mut self,
        handle: &TypedHandle<Args, Ret>,
        args: Args,
    ) -> Result<Ret>
    where
        Args: FunctionArgs,
        Ret: FromLustValue,
    {
        let mut stack_args = [const { Value::Nil }; 16];
        let arg_count = args.write_args(&mut stack_args);
        let value = self.vm.call_retained_target_slice(
            handle.func_idx,
            &handle.expected_name,
            &stack_args[..arg_count],
        )?;
        Ret::from_value(value)
    }

    pub fn call_typed<Args, R>(&mut self, function_name: &str, args: Args) -> Result<R>
    where
        Args: FunctionArgs,
        R: FromLustValue,
    {
        let signature = self
            .signatures
            .get(function_name)
            .ok_or_else(|| LustError::TypeError {
                message: format!(
                    "No type information available for function '{}'; \
                     use call_raw if the function is dynamically typed",
                    function_name
                ),
            })?;
        Args::validate_signature(function_name, &signature.params)?;
        ensure_return_type::<R>(function_name, &signature.return_type)?;
        let func_idx = if let Some((cached_name, cached_idx)) = &self.last_typed_target
            && cached_name == function_name
        {
            *cached_idx
        } else {
            let idx = self
                .vm
                .functions()
                .iter()
                .position(|f| f.name == function_name)
                .ok_or_else(|| LustError::RuntimeError {
                    message: format!("Function not found: {}", function_name),
                })?;
            self.last_typed_target = Some((function_name.to_string(), idx));
            idx
        };

        let mut stack_args = [const { Value::Nil }; 16];
        let arg_count = args.write_args(&mut stack_args);
        let value = self.vm.call_retained_target_slice(
            func_idx,
            function_name,
            &stack_args[..arg_count],
        )?;
        R::from_value(value)
    }

    pub fn call_raw(&mut self, function_name: &str, args: Vec<Value>) -> Result<Value> {
        self.vm.call(function_name, args)
    }

    pub fn function_handle(&self, function_name: &str) -> Result<FunctionHandle> {
        let mut candidates = Vec::new();
        candidates.push(function_name.to_string());
        candidates.extend(self.signature_lookup_candidates(function_name));
        for name in candidates {
            if let Some(value) = self.vm.function_value(&name) {
                return FunctionHandle::from_value(value);
            }
        }
        Err(LustError::RuntimeError {
            message: format!("Function '{}' not found in embedded program", function_name),
        })
    }

    pub fn run_entry_script(&mut self) -> Result<()> {
        let Some(entry) = &self.entry_script else {
            return Err(LustError::RuntimeError {
                message: "Embedded program has no entry script".into(),
            });
        };
        let result = self.vm.call(entry, Vec::new())?;
        match result {
            Value::Nil => Ok(()),
            other => Err(LustError::RuntimeError {
                message: format!(
                    "Entry script '{}' returned non-unit value: {:?}",
                    entry, other
                ),
            }),
        }
    }

    pub fn poll_async_tasks(&mut self) -> Result<()> {
        let pending_ids: Vec<u64> = {
            let registry = self.async_registry.borrow();
            registry.pending.keys().copied().collect()
        };

        for id in pending_ids {
            let mut completion: Option<(
                AsyncTaskTarget,
                Option<Type>,
                std::result::Result<Value, String>,
            )> = None;
            {
                let mut registry = self.async_registry.borrow_mut();
                let entry = match registry.pending.get_mut(&id) {
                    Some(entry) => entry,
                    None => continue,
                };

                if !entry.take_should_poll() {
                    continue;
                }

                let waker = entry.make_waker();
                let mut cx = Context::from_waker(&waker);
                if let Poll::Ready(result) = entry.future.as_mut().poll(&mut cx) {
                    completion = Some((entry.target, entry.expected_type.clone(), result));
                }
            }

            if let Some((target, expected_type, mut outcome)) = completion {
                self.async_registry.borrow_mut().pending.remove(&id);
                if let (Some(expected), Ok(value)) = (&expected_type, &outcome)
                    && !self.vm.value_matches_type(value, expected)
                {
                    outcome = Err(format!(
                        "Async native must return {}, got {:?}",
                        expected,
                        value.type_of()
                    ));
                }
                match target {
                    AsyncTaskTarget::ScriptTask(handle) => match outcome {
                        Ok(value) => {
                            self.vm.resume_task_handle(handle, Some(value))?;
                        }

                        Err(message) => {
                            self.vm
                                .fail_task_handle(handle, LustError::RuntimeError { message })?;
                        }
                    },

                    AsyncTaskTarget::NativeTask(handle) => {
                        self.vm.complete_native_future_task(handle, outcome)?;
                    }
                }
            }
        }

        Ok(())
    }

    pub fn has_pending_async_tasks(&self) -> bool {
        !self.async_registry.borrow().is_empty()
    }

    #[allow(dead_code)]
    pub(crate) fn last_dispatched_target(&self) -> Option<(usize, String)> {
        self.vm.last_dispatched_target()
    }

    #[allow(dead_code)]
    pub(crate) fn resolve_canonical_target(
        &self,
        selector: &str,
    ) -> std::result::Result<PreflightTarget, PreflightError> {
        if selector.is_empty() {
            return Err(PreflightError::call("Empty function selector"));
        }
        if selector.contains("::") {
            return Err(PreflightError::call(format!(
                "Noncanonical selector '{}': alternate normalization with '::' is rejected; use exact canonical name with '.'",
                selector
            )));
        }
        if selector.starts_with('.') || selector.ends_with('.') || selector.contains("..") {
            return Err(PreflightError::call(format!(
                "Noncanonical selector '{}': invalid dot positioning",
                selector
            )));
        }

        let mut exact_matches = Vec::new();
        for (idx, func) in self.vm.functions().iter().enumerate() {
            if func.name == selector {
                exact_matches.push((idx, func));
            }
        }

        if exact_matches.is_empty() {
            let simple_name_matches: Vec<&str> = self
                .vm
                .functions()
                .iter()
                .filter_map(|f| {
                    if f.name.rsplit('.').next() == Some(selector) && f.name != selector {
                        Some(f.name.as_str())
                    } else {
                        None
                    }
                })
                .collect();

            if !simple_name_matches.is_empty() {
                return Err(PreflightError::call(format!(
                    "Noncanonical selector '{}': simple-name fallback is rejected; use exact canonical target (candidates: {:?})",
                    selector, simple_name_matches
                )));
            }

            return Err(PreflightError::call(format!(
                "Target function '{}' not found in program",
                selector
            )));
        }

        if exact_matches.len() > 1 {
            return Err(PreflightError::call(format!(
                "Ambiguous target function '{}': {} functions match in table",
                selector,
                exact_matches.len()
            )));
        }

        let (func_idx, func) = exact_matches[0];

        if func.is_method {
            return Err(PreflightError::call(format!(
                "Target function '{}' is a method; method invocation via JSON entrypoint is rejected because instance receiver semantics are not proven safe",
                selector
            )));
        }

        let sig = func.signature.as_ref().ok_or_else(|| {
            PreflightError::type_error(format!(
                "Target function '{}' has no type signature (unresolved/absent signature)",
                selector
            ))
        })?;

        if !sig.type_params.is_empty() {
            return Err(PreflightError::type_error(format!(
                "Target function '{}' has uninstantiated type parameters: {:?}",
                selector, sig.type_params
            )));
        }

        if !sig.trait_bounds.is_empty() {
            return Err(PreflightError::type_error(format!(
                "Target function '{}' has generic trait bounds",
                selector
            )));
        }

        let mut schema_budget = WorkBudget::new(64 * 1024, 64);
        for (idx, param) in sig.params.iter().enumerate() {
            self.validate_schema(param, 0, &mut schema_budget)
                .map_err(|mut e| {
                    e.message = format!(
                        "Parameter {} schema for '{}' is invalid: {}",
                        idx + 1,
                        selector,
                        e.message
                    );
                    e
                })?;
        }

        let executable_sig = (
            func.name.clone(),
            func.param_count,
            func.signature.as_ref().map(|s| s.to_string()),
            func.is_method,
        );

        Ok(PreflightTarget {
            program_id: self.program_id,
            program_token: self.program_token.clone(),
            func_idx,
            canonical_name: func.name.clone(),
            signature: sig.clone(),
            executable_sig,
        })
    }

    pub(crate) fn validate_schema(
        &self,
        ty: &Type,
        depth: usize,
        budget: &mut WorkBudget,
    ) -> std::result::Result<(), PreflightError> {
        budget.check_depth(depth)?;
        budget.consume_work(1)?;

        match &ty.kind {
            TypeKind::Int
            | TypeKind::Float
            | TypeKind::String
            | TypeKind::Bool
            | TypeKind::Unit => Ok(()),
            TypeKind::Array(inner) => self.validate_schema(inner, depth + 1, budget),
            TypeKind::Map(key, val) => {
                match &key.kind {
                    TypeKind::String => {}
                    _ => {
                        return Err(PreflightError::type_error(format!(
                            "Unsupported map key schema '{}': JSON objects require string keys",
                            key
                        )));
                    }
                }
                self.validate_schema(key, depth + 1, budget)?;
                self.validate_schema(val, depth + 1, budget)
            }
            TypeKind::Tuple(types) => Err(PreflightError::type_error(format!(
                "Unsupported schema: Tuple with {} element(s) is rejected",
                types.len()
            ))),
            TypeKind::Option(inner) => Err(PreflightError::type_error(format!(
                "Unsupported schema: Option<{}> is rejected",
                inner
            ))),
            TypeKind::Result(ok, err) => Err(PreflightError::type_error(format!(
                "Unsupported schema: Result<{}, {}> is rejected",
                ok, err
            ))),
            TypeKind::Unknown => Err(PreflightError::type_error(
                "Unresolved schema: TypeKind::Unknown is strictly rejected",
            )),
            TypeKind::Infer => Err(PreflightError::type_error(
                "Unresolved schema: TypeKind::Infer is strictly rejected",
            )),
            TypeKind::Generic(name) => Err(PreflightError::type_error(format!(
                "Uninstantiated generic type parameter '{}' is strictly rejected",
                name
            ))),
            TypeKind::GenericInstance { name, .. } => Err(PreflightError::type_error(format!(
                "Uninstantiated generic instance '{}' is strictly rejected",
                name
            ))),
            TypeKind::Named(name) => {
                if name == "Option" || name.starts_with("Option<") {
                    Err(PreflightError::type_error(format!(
                        "Unsupported schema: Option ({}) is rejected",
                        name
                    )))
                } else if name == "Result" || name.starts_with("Result<") {
                    Err(PreflightError::type_error(format!(
                        "Unsupported schema: Result ({}) is rejected",
                        name
                    )))
                } else {
                    Err(PreflightError::type_error(format!(
                        "Unsupported nominal schema '{}'",
                        name
                    )))
                }
            }
            TypeKind::Trait(name) => Err(PreflightError::type_error(format!(
                "Unsupported trait schema '{}'",
                name
            ))),
            TypeKind::TraitBound(_) => Err(PreflightError::type_error(
                "Unsupported trait bound in schema",
            )),
            TypeKind::Function { .. } => Err(PreflightError::type_error(
                "Function schema is strictly rejected",
            )),
            TypeKind::Ref(_) | TypeKind::MutRef(_) | TypeKind::Pointer { .. } => Err(
                PreflightError::type_error("Pointer/reference schema is strictly rejected"),
            ),
            TypeKind::Union(_) => Err(PreflightError::type_error(
                "Union schema is strictly rejected",
            )),
        }
    }

    pub(crate) fn validate_argument_value(
        &self,
        value: &Value,
        expected: &Type,
        depth: usize,
        budget: &mut WorkBudget,
    ) -> std::result::Result<(), PreflightError> {
        budget.check_depth(depth)?;
        budget.consume_work(1)?;

        // CRITICAL: Always validate full schema first even for empty containers
        self.validate_schema(expected, depth, budget)?;

        match &expected.kind {
            TypeKind::Int => {
                if matches!(value, Value::Int(_)) {
                    Ok(())
                } else {
                    Err(PreflightError::type_error(format!(
                        "Type mismatch: expected int, got {:?}",
                        value.type_of()
                    )))
                }
            }
            TypeKind::Float => {
                if let Value::Float(f) = value {
                    if !f.is_finite() {
                        return Err(PreflightError::type_error(
                            "Non-finite float (NaN or Infinity) is not supported in JSON subset",
                        ));
                    }
                    Ok(())
                } else {
                    Err(PreflightError::type_error(format!(
                        "Type mismatch: expected float, got {:?}",
                        value.type_of()
                    )))
                }
            }
            TypeKind::String => {
                if matches!(value, Value::String(_)) {
                    Ok(())
                } else {
                    Err(PreflightError::type_error(format!(
                        "Type mismatch: expected string, got {:?}",
                        value.type_of()
                    )))
                }
            }
            TypeKind::Bool => {
                if matches!(value, Value::Bool(_)) {
                    Ok(())
                } else {
                    Err(PreflightError::type_error(format!(
                        "Type mismatch: expected bool, got {:?}",
                        value.type_of()
                    )))
                }
            }
            TypeKind::Unit => {
                if matches!(value, Value::Nil) {
                    Ok(())
                } else {
                    Err(PreflightError::type_error(format!(
                        "Type mismatch: expected unit/nil, got {:?}",
                        value.type_of()
                    )))
                }
            }
            TypeKind::Array(elem_ty) => match value {
                Value::Array(arr) => {
                    let borrowed = arr.borrow();
                    for (i, elem) in borrowed.iter().enumerate() {
                        self.validate_argument_value(elem, elem_ty, depth + 1, budget)
                            .map_err(|mut e| {
                                e.message = format!("Array element [{}]: {}", i, e.message);
                                e
                            })?;
                    }
                    Ok(())
                }
                _ => Err(PreflightError::type_error(format!(
                    "Type mismatch: expected Array<{}>, got {:?}",
                    elem_ty,
                    value.type_of()
                ))),
            },
            TypeKind::Map(k_ty, v_ty) => match value {
                Value::Map(m) => {
                    let borrowed = m.borrow();
                    for (k, v) in borrowed.iter() {
                        let k_val = k.to_value();
                        self.validate_argument_value(&k_val, k_ty, depth + 1, budget)
                            .map_err(|mut e| {
                                e.message = format!("Map key: {}", e.message);
                                e
                            })?;
                        self.validate_argument_value(v, v_ty, depth + 1, budget)
                            .map_err(|mut e| {
                                e.message = format!("Map value: {}", e.message);
                                e
                            })?;
                    }
                    Ok(())
                }
                _ => Err(PreflightError::type_error(format!(
                    "Type mismatch: expected Map<{}, {}>, got {:?}",
                    k_ty,
                    v_ty,
                    value.type_of()
                ))),
            },
            TypeKind::Tuple(types) => Err(PreflightError::type_error(format!(
                "Unsupported schema: Tuple with {} element(s) is rejected",
                types.len()
            ))),
            TypeKind::Option(inner) => Err(PreflightError::type_error(format!(
                "Unsupported schema: Option<{}> is rejected",
                inner
            ))),
            TypeKind::Result(ok_ty, err_ty) => Err(PreflightError::type_error(format!(
                "Unsupported schema: Result<{}, {}> is rejected",
                ok_ty, err_ty
            ))),
            _ => Err(PreflightError::type_error(format!(
                "Unsupported schema type: {}",
                expected
            ))),
        }
    }

    // Bounded traversal: Depth and work budget bound recursion and traversal effort.
    // Active ancestors tracking detects cyclic containers (CYCLE status) without rejecting
    // acyclic shared aliases traversed by value.
    pub(crate) fn validate_return_value(
        &self,
        value: &Value,
        expected: &Type,
        depth: usize,
        budget: &mut WorkBudget,
        active_ancestors: &mut Vec<usize>,
    ) -> std::result::Result<(), PreflightError> {
        // Cycle check: If this container is an active ancestor, report CYCLE immediately
        // before any type/depth check.
        match value {
            Value::Array(arr) => {
                let ptr_id = Rc::as_ptr(arr) as usize;
                if active_ancestors.contains(&ptr_id) {
                    return Err(PreflightError::cycle(
                        "Cyclic container detected: array contains an active ancestor reference",
                    ));
                }
            }
            Value::Map(m) => {
                let ptr_id = Rc::as_ptr(m) as usize;
                if active_ancestors.contains(&ptr_id) {
                    return Err(PreflightError::cycle(
                        "Cyclic container detected: map contains an active ancestor reference",
                    ));
                }
            }
            _ => {}
        }

        budget.check_depth(depth)?;
        budget.consume_work(1)?;

        // CRITICAL: Always validate full return schema first even for empty containers / unit
        self.validate_schema(expected, depth, budget)?;

        match &expected.kind {
            TypeKind::Int => {
                if matches!(value, Value::Int(_)) {
                    Ok(())
                } else {
                    Err(PreflightError::type_error(format!(
                        "Return type mismatch: expected int, got {:?}",
                        value.type_of()
                    )))
                }
            }
            TypeKind::Float => {
                if let Value::Float(f) = value {
                    if !f.is_finite() {
                        return Err(PreflightError::type_error(
                            "Non-finite float (NaN or Infinity) is not supported in JSON subset",
                        ));
                    }
                    Ok(())
                } else {
                    Err(PreflightError::type_error(format!(
                        "Return type mismatch: expected float, got {:?}",
                        value.type_of()
                    )))
                }
            }
            TypeKind::String => {
                if matches!(value, Value::String(_)) {
                    Ok(())
                } else {
                    Err(PreflightError::type_error(format!(
                        "Return type mismatch: expected string, got {:?}",
                        value.type_of()
                    )))
                }
            }
            TypeKind::Bool => {
                if matches!(value, Value::Bool(_)) {
                    Ok(())
                } else {
                    Err(PreflightError::type_error(format!(
                        "Return type mismatch: expected bool, got {:?}",
                        value.type_of()
                    )))
                }
            }
            TypeKind::Unit => {
                if matches!(value, Value::Nil) {
                    Ok(())
                } else {
                    Err(PreflightError::type_error(format!(
                        "Return type mismatch: expected unit/nil, got {:?}",
                        value.type_of()
                    )))
                }
            }
            TypeKind::Array(elem_ty) => match value {
                Value::Array(arr) => {
                    let ptr_id = Rc::as_ptr(arr) as usize;
                    if active_ancestors.contains(&ptr_id) {
                        return Err(PreflightError::cycle(
                            "Cyclic container detected: array contains an active ancestor reference",
                        ));
                    }
                    let borrowed = arr.try_borrow().map_err(|_| {
                        PreflightError::borrow("Cannot borrow array: value is mutably borrowed")
                    })?;
                    active_ancestors.push(ptr_id);
                    let res = (|| {
                        for (i, elem) in borrowed.iter().enumerate() {
                            self.validate_return_value(
                                elem,
                                elem_ty,
                                depth + 1,
                                budget,
                                active_ancestors,
                            )
                            .map_err(|mut e| {
                                e.message = format!("Array element [{}]: {}", i, e.message);
                                e
                            })?;
                        }
                        Ok(())
                    })();
                    active_ancestors.pop();
                    res
                }
                _ => Err(PreflightError::type_error(format!(
                    "Return type mismatch: expected Array<{}>, got {:?}",
                    elem_ty,
                    value.type_of()
                ))),
            },
            TypeKind::Map(k_ty, v_ty) => match value {
                Value::Map(m) => {
                    let ptr_id = Rc::as_ptr(m) as usize;
                    if active_ancestors.contains(&ptr_id) {
                        return Err(PreflightError::cycle(
                            "Cyclic container detected: map contains an active ancestor reference",
                        ));
                    }
                    let borrowed = m.try_borrow().map_err(|_| {
                        PreflightError::borrow("Cannot borrow map: value is mutably borrowed")
                    })?;
                    active_ancestors.push(ptr_id);
                    let res = (|| {
                        for (k, v) in borrowed.iter() {
                            let k_val = k.to_value();
                            self.validate_return_value(
                                &k_val,
                                k_ty,
                                depth + 1,
                                budget,
                                active_ancestors,
                            )
                            .map_err(|mut e| {
                                e.message = format!("Map key: {}", e.message);
                                e
                            })?;
                            self.validate_return_value(
                                v,
                                v_ty,
                                depth + 1,
                                budget,
                                active_ancestors,
                            )
                            .map_err(|mut e| {
                                e.message = format!("Map value: {}", e.message);
                                e
                            })?;
                        }
                        Ok(())
                    })();
                    active_ancestors.pop();
                    res
                }
                _ => Err(PreflightError::type_error(format!(
                    "Return type mismatch: expected Map<{}, {}>, got {:?}",
                    k_ty,
                    v_ty,
                    value.type_of()
                ))),
            },
            TypeKind::Tuple(types) => Err(PreflightError::type_error(format!(
                "Unsupported schema: Tuple with {} element(s) is rejected",
                types.len()
            ))),
            TypeKind::Option(inner) => Err(PreflightError::type_error(format!(
                "Unsupported schema: Option<{}> is rejected",
                inner
            ))),
            TypeKind::Result(ok_ty, err_ty) => Err(PreflightError::type_error(format!(
                "Unsupported schema: Result<{}, {}> is rejected",
                ok_ty, err_ty
            ))),
            _ => Err(PreflightError::type_error(format!(
                "Unsupported return schema type: {}",
                expected
            ))),
        }
    }

    pub(crate) fn call_preflight_spike(
        &mut self,
        selector: &str,
        args: Vec<Value>,
        limits: &PreflightLimits,
    ) -> std::result::Result<Value, PreflightError> {
        let target = self.resolve_canonical_target(selector)?;
        self.call_retained_target_spike(&target, args, limits)
    }

    pub(crate) fn call_retained_target_spike(
        &mut self,
        target: &PreflightTarget,
        args: Vec<Value>,
        limits: &PreflightLimits,
    ) -> std::result::Result<Value, PreflightError> {
        if !Rc::ptr_eq(&self.program_token, &target.program_token)
            || target.program_id != self.program_id
        {
            return Err(PreflightError::call(format!(
                "Cross-program target dispatch rejected: target was bound to originating EmbeddedProgram {} but invoked on program {}",
                target.program_id, self.program_id
            )));
        }

        let func = self.vm.functions().get(target.func_idx).ok_or_else(|| {
            PreflightError::call(format!(
                "Target function index {} out of bounds",
                target.func_idx
            ))
        })?;

        if target.canonical_name != func.name {
            return Err(PreflightError::call(format!(
                "Target canonical name mismatch: target has '{}', function in table has '{}'",
                target.canonical_name, func.name
            )));
        }

        let current_sig = (
            func.name.clone(),
            func.param_count,
            func.signature.as_ref().map(|s| s.to_string()),
            func.is_method,
        );
        if current_sig != target.executable_sig {
            return Err(PreflightError::call(format!(
                "Target executable signature mismatch for '{}': target is staled or mismatched",
                target.canonical_name
            )));
        }

        let live_sig = func.signature.as_ref().ok_or_else(|| {
            PreflightError::call(format!(
                "Target function '{}' has no live type signature",
                target.canonical_name
            ))
        })?;

        // Retained signature integrity snapshot equality check (Part A)
        if !function_signatures_equal(&target.signature, live_sig) {
            return Err(PreflightError::call(format!(
                "Target retained signature integrity mismatch for '{}': retained signature does not match live executable signature",
                target.canonical_name
            )));
        }

        if args.len() != live_sig.params.len() {
            return Err(PreflightError::type_error(format!(
                "Wrong arity for '{}': expected {} arguments, got {}",
                target.canonical_name,
                live_sig.params.len(),
                args.len()
            )));
        }

        let mut budget = WorkBudget::new(limits.max_work, limits.max_depth);
        for (idx, (arg, param_ty)) in args.iter().zip(&live_sig.params).enumerate() {
            self.validate_argument_value(arg, param_ty, 0, &mut budget)
                .map_err(|mut e| {
                    e.message = format!(
                        "Argument {} for '{}' failed validation: {}",
                        idx + 1,
                        target.canonical_name,
                        e.message
                    );
                    e
                })?;
        }

        let return_type = live_sig.return_type.clone();

        let raw_return_value = self
            .vm
            .call_retained_target(target.func_idx, &target.canonical_name, args)
            .map_err(|err| PreflightError::call(format!("Runtime execution error: {}", err)))?;

        // Post-call return validation (Part B)
        // Validates returned Value against the same function's declared return Type
        // and supported JSON-subset type/value rules, with bounded depth/work traversal
        // and active-ancestor cycle detection.
        // NOTE: Serialization/output quotas remain deferred to future gate work (Gate 3).
        let mut return_budget = WorkBudget::new(limits.max_work, limits.max_depth);
        let mut active_ancestors = Vec::new();
        self.validate_return_value(
            &raw_return_value,
            &return_type,
            0,
            &mut return_budget,
            &mut active_ancestors,
        )
        .map_err(|mut e| {
            e.message = format!(
                "Post-call return validation error for '{}': {}",
                target.canonical_name, e.message
            );
            e
        })?;

        Ok(raw_return_value)
    }
}

fn function_signatures_equal(a: &FunctionSignature, b: &FunctionSignature) -> bool {
    a.params == b.params
        && a.return_type == b.return_type
        && a.is_method == b.is_method
        && a.type_params == b.type_params
        && a.trait_bounds == b.trait_bounds
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PreflightStatus {
    Call,
    Type,
    Limit,
    Borrow,
    Cycle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PreflightError {
    pub(crate) status: PreflightStatus,
    pub(crate) message: String,
}

impl PreflightError {
    pub(crate) fn call(msg: impl Into<String>) -> Self {
        Self {
            status: PreflightStatus::Call,
            message: msg.into(),
        }
    }

    pub(crate) fn type_error(msg: impl Into<String>) -> Self {
        Self {
            status: PreflightStatus::Type,
            message: msg.into(),
        }
    }

    pub(crate) fn limit(msg: impl Into<String>) -> Self {
        Self {
            status: PreflightStatus::Limit,
            message: msg.into(),
        }
    }

    pub(crate) fn borrow(msg: impl Into<String>) -> Self {
        Self {
            status: PreflightStatus::Borrow,
            message: msg.into(),
        }
    }

    pub(crate) fn cycle(msg: impl Into<String>) -> Self {
        Self {
            status: PreflightStatus::Cycle,
            message: msg.into(),
        }
    }
}

impl fmt::Display for PreflightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.status, self.message)
    }
}

impl std::error::Error for PreflightError {}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PreflightLimits {
    pub(crate) max_depth: usize,
    pub(crate) max_work: usize,
}

impl Default for PreflightLimits {
    fn default() -> Self {
        Self {
            max_depth: 64,
            max_work: 65_536,
        }
    }
}

#[derive(Debug)]
pub(crate) struct WorkBudget {
    max_depth: usize,
    remaining_work: usize,
}

impl WorkBudget {
    pub(crate) fn new(max_work: usize, max_depth: usize) -> Self {
        Self {
            max_depth,
            remaining_work: max_work,
        }
    }

    pub(crate) fn check_depth(&self, depth: usize) -> std::result::Result<(), PreflightError> {
        if depth > self.max_depth {
            return Err(PreflightError::limit(format!(
                "Max depth exceeded: depth {} exceeds limit {}",
                depth, self.max_depth
            )));
        }
        Ok(())
    }

    pub(crate) fn consume_work(&mut self, units: usize) -> std::result::Result<(), PreflightError> {
        if self.remaining_work < units {
            return Err(PreflightError::limit(format!(
                "Work budget exhausted: requested {} units, remaining {}",
                units, self.remaining_work
            )));
        }
        self.remaining_work -= units;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PreflightTarget {
    program_id: u64,
    program_token: Rc<()>,
    func_idx: usize,
    canonical_name: String,
    signature: FunctionSignature,
    executable_sig: (String, u8, Option<String>, bool),
}

#[allow(dead_code)]
impl PreflightTarget {
    pub(crate) fn canonical_name(&self) -> &str {
        &self.canonical_name
    }

    pub(crate) fn func_idx(&self) -> usize {
        self.func_idx
    }

    pub(crate) fn signature(&self) -> &FunctionSignature {
        &self.signature
    }

    pub(crate) fn executable_sig(&self) -> &(String, u8, Option<String>, bool) {
        &self.executable_sig
    }

    #[cfg(test)]
    pub(crate) fn tamper_signature_for_test(&mut self, tampered: FunctionSignature) {
        self.signature = tampered;
    }

    #[cfg(test)]
    pub(crate) fn tamper_program_token_for_test(&mut self, token: Rc<()>, id: u64) {
        self.program_token = token;
        self.program_id = id;
    }
}

#[cfg(feature = "std")]
thread_local! {
    static BYTECODE_CACHE: RefCell<HashMap<String, CompiledBytecode>> =
        RefCell::new(HashMap::new());
}

pub(crate) fn compile_to_bytecode(
    base_dir: PathBuf,
    entry_module: String,
    overrides: HashMap<PathBuf, String>,
    config: LustConfig,
    extern_registry: ExternRegistry,
) -> Result<CompiledBytecode> {
    #[cfg(feature = "std")]
    let cache_key = if extern_registry.is_empty() {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        base_dir.hash(&mut hasher);
        entry_module.hash(&mut hasher);
        let mut sorted: Vec<(&PathBuf, &String)> = overrides.iter().collect();
        sorted.sort_by_key(|(p, _)| *p);
        for (p, s) in sorted {
            p.hash(&mut hasher);
            s.hash(&mut hasher);
        }
        config.jit_enabled().hash(&mut hasher);
        config.minimal_runtime_types().hash(&mut hasher);
        config.low_memory_mode().hash(&mut hasher);
        let key = format!("{:016x}", hasher.finish());
        let cached = BYTECODE_CACHE.with(|cache| cache.borrow().get(&key).cloned());
        if let Some(cached) = cached {
            return Ok(cached);
        }
        Some(key)
    } else {
        None
    };

    let mut loader = ModuleLoader::new(base_dir.clone());
    loader.set_source_overrides(overrides);
    let entry_path = module_path_to_file(&base_dir, &entry_module);
    let entry_path_str = entry_path
        .to_str()
        .ok_or_else(|| LustError::Unknown("Entry path contained invalid UTF-8".into()))?
        .to_string();
    let program = loader.load_program_from_entry(&entry_path_str)?;

    // Build imports map (small: just alias/path strings, not AST)
    let mut imports_map: HashMap<String, ModuleImports> = HashMap::new();
    for module in &program.modules {
        imports_map.insert(module.path.clone(), module.imports.clone());
    }

    // Phase 1: type check while program is still alive (borrows module items)
    let mut typechecker = TypeChecker::with_config(&config);
    typechecker.set_imports_by_module(imports_map.clone());
    extern_registry.register_with_typechecker(&mut typechecker)?;
    typechecker.check_program(&program.modules)?;
    let option_coercions = typechecker.take_option_coercions();
    let checked_array_indices = typechecker.take_checked_array_indices();
    let lambda_return_types = typechecker.take_lambda_return_types();
    let numeric_types = typechecker.take_numeric_types();
    // Use take_ to move data out instead of cloning
    let mut struct_defs = typechecker.take_struct_definitions();
    for def in extern_registry.structs() {
        struct_defs.insert(def.name.clone(), def.clone());
    }
    let mut enum_defs = typechecker.take_enum_definitions();
    for def in extern_registry.enums() {
        enum_defs.insert(def.name.clone(), def.clone());
    }
    let signatures = typechecker.take_function_signatures();
    // Typechecker no longer needed — free its memory (expr_types, variable_types, scopes, etc.)
    drop(typechecker);

    // Phase 2: consume program.modules to build wrapped_items without cloning AST items
    let program_entry_module = program.entry_module;
    let mut init_funcs: Vec<(String, String)> = Vec::new();
    let mut wrapped_items: Vec<Item> = Vec::new();
    for module in program.modules {
        if module.path != program_entry_module
            && let Some(ref init) = module.init_function
        {
            let init_name = module
                .imports
                .function_aliases
                .get(init)
                .cloned()
                .unwrap_or_else(|| init.clone());
            init_funcs.push((module.path.clone(), init_name));
        }
        wrapped_items.push(Item::new(
            ItemKind::Module {
                name: module.path,
                items: module.items,
            },
            Span::new(0, 0, 0, 0),
        ));
    }

    // Phase 3: compile — move signatures into compiler to avoid a second copy
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
    let functions = compiler.compile_module(&wrapped_items)?;
    let trait_impls = compiler.get_trait_impls().to_vec();
    // Recover signatures from compiler (avoids keeping two copies alive simultaneously)
    let mut signatures = compiler.take_function_signatures();
    // AST no longer needed — free it before setting up the VM
    drop(wrapped_items);

    let entry_script = functions
        .iter()
        .find(|f| f.name == "__script")
        .map(|f| f.name.clone());
    if let Some(script_name) = &entry_script {
        signatures
            .entry(script_name.clone())
            .or_insert_with(|| FunctionSignature {
                params: Vec::new(),
                return_type: Type::new(TypeKind::Unit, Span::new(0, 0, 0, 0)),
                is_method: false,
                type_params: Vec::new(),
                trait_bounds: Vec::new(),
            });
    }

    let compiled = CompiledBytecode {
        entry_module: program_entry_module,
        entry_script,
        functions,
        signatures,
        struct_defs,
        enum_defs,
        trait_impls,
        init_funcs,
        config,
        extern_registry,
    };

    #[cfg(feature = "std")]
    if let Some(key) = cache_key {
        BYTECODE_CACHE.with(|cache| {
            cache.borrow_mut().insert(key, compiled.clone());
        });
    }

    Ok(compiled)
}

fn compile_in_memory(
    base_dir: PathBuf,
    entry_module: String,
    overrides: HashMap<PathBuf, String>,
    config: LustConfig,
    extern_registry: ExternRegistry,
) -> Result<EmbeddedProgram> {
    let bytecode = compile_to_bytecode(
        base_dir,
        entry_module,
        overrides,
        config,
        extern_registry,
    )?;
    EmbeddedProgram::from_compiled(&bytecode)
}

fn module_path_to_file(base_dir: &Path, module_path: &str) -> PathBuf {
    let mut path = base_dir.to_path_buf();
    for segment in module_path.split('.') {
        path.push(segment);
    }

    path.set_extension("lust");
    path
}

pub(crate) fn normalize_global_name(name: &str) -> String {
    if name.contains("::") {
        name.to_string()
    } else if let Some((module, identifier)) = name.rsplit_once('.') {
        format!("{}::{}", module, identifier)
    } else {
        name.to_string()
    }
}

pub(crate) fn ensure_return_type<R: FromLustValue>(function_name: &str, ty: &Type) -> Result<()> {
    if matches!(ty.kind, TypeKind::Unknown) || R::matches_lust_type(ty) {
        return Ok(());
    }

    Err(LustError::TypeError {
        message: format!(
            "Function '{}' reports return type '{}' which is incompatible with Rust receiver '{}'",
            function_name,
            ty,
            R::type_description()
        ),
    })
}

fn normalize_extern_type_string(mut ty: String) -> String {
    if ty.starts_with("function(") && ty.ends_with(": ()") {
        ty.truncate(ty.len().saturating_sub(4));
    }
    ty
}

fn native_export_from_signature(
    canonical_name: &str,
    signature: &FunctionSignature,
) -> NativeExport {
    let name = canonical_name.replace("::", ".");
    let params = signature
        .params
        .iter()
        .enumerate()
        .map(|(idx, ty)| {
            NativeExportParam::new(
                format!("arg{idx}"),
                normalize_extern_type_string(ty.to_string()),
            )
        })
        .collect::<Vec<_>>();
    let return_type = normalize_extern_type_string(signature.return_type.to_string());
    NativeExport::new(name, params, return_type)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn base_dir_loads_entry_and_imported_modules_without_module_sources() {
        let dir = tempdir().expect("temporary directory");
        fs::write(
            dir.path().join("main.lust"),
            "use helper.{answer}\nfunction result(): int\n    return answer()\nend\n",
        )
        .expect("write entry script");
        fs::write(
            dir.path().join("helper.lust"),
            "function answer(): int\n    return 41\nend\n",
        )
        .expect("write helper module");

        let mut program = EmbeddedProgram::builder()
            .with_base_dir(dir.path())
            .entry_module("main")
            .compile()
            .expect("load entry and imported module from base directory");
        let result: crate::LustInt = program
            .call_typed("main.result", ())
            .expect("call imported function");

        assert_eq!(result, 41);
    }

    #[test]
    fn test_b11_fast_path_and_typed_handle() {
        let source = r#"
            function b11_target(val: int): int
                return (val * 3 + 7) % 10007
            end
        "#;
        let mut program = EmbeddedProgram::builder()
            .module("main", source)
            .entry_module("main")
            .compile()
            .expect("compile");

        let handle = program
            .get_typed_handle::<i64, i64>("main.b11_target")
            .expect("get_typed_handle");

        for i in 0..100 {
            let res: i64 = program.call_handle(&handle, i).expect("call_handle");
            assert_eq!(res, (i * 3 + 7) % 10007);
        }

        let t0 = std::time::Instant::now();
        let mut st = 1i64;
        for i in 0..10_000i64 {
            st = program.call_typed("main.b11_target", st + i % 10).expect("call_typed");
        }
        std::hint::black_box(st);
        let elapsed_typed = t0.elapsed();

        let t1 = std::time::Instant::now();
        let mut st2 = 1i64;
        for i in 0..10_000i64 {
            st2 = program.call_handle(&handle, st2 + i % 10).expect("call_handle");
        }
        std::hint::black_box(st2);
        let elapsed_handle = t1.elapsed();

        println!("call_typed: {} ns/call", elapsed_typed.as_nanos() / 10_000);
        println!("call_handle: {} ns/call", elapsed_handle.as_nanos() / 10_000);
        println!("JIT stats: {:?}", program.jit_stats());
    }

    #[test]
    fn registered_entry_source_overrides_file_under_base_dir() {
        let dir = tempdir().expect("temporary directory");
        fs::write(
            dir.path().join("main.lust"),
            "function result(): int\n    return 1\nend\n",
        )
        .expect("write on-disk entry script");

        let mut program = EmbeddedProgram::builder()
            .with_base_dir(dir.path())
            .module("main", "function result(): int\n    return 42\nend\n")
            .entry_module("main")
            .compile()
            .expect("compile with in-memory entry override");
        let result: crate::LustInt = program
            .call_typed("main.result", ())
            .expect("call overridden entry function");

        assert_eq!(result, 42);
    }

    #[test]
    fn registered_module_source_overrides_file_under_base_dir() {
        let dir = tempdir().expect("temporary directory");
        fs::write(
            dir.path().join("main.lust"),
            "use helper.{answer}\nfunction result(): int\n    return answer()\nend\n",
        )
        .expect("write entry script");
        fs::write(
            dir.path().join("helper.lust"),
            "function answer(): int\n    return 1\nend\n",
        )
        .expect("write on-disk helper module");

        let mut program = EmbeddedProgram::builder()
            .with_base_dir(dir.path())
            .module("helper", "function answer(): int\n    return 42\nend\n")
            .entry_module("main")
            .compile()
            .expect("compile with in-memory module override");
        let result: crate::LustInt = program
            .call_typed("main.result", ())
            .expect("call overridden helper function");

        assert_eq!(result, 42);
    }

    #[test]
    fn gate2_exact_target_identity_and_witness() {
        use std::sync::Arc;
        use std::sync::Mutex;

        let audit_records = Arc::new(Mutex::new(Vec::new()));
        let audit_clone = audit_records.clone();

        let mut program = EmbeddedProgram::builder()
            .module(
                "audit",
                r#"
                extern
                    function record_call(int): int
                end
                "#,
            )
            .module(
                "alpha",
                r#"
                use audit.{record_call}
                function marker(): int
                    record_call(101)
                    return 101
                end
                "#,
            )
            .module(
                "beta",
                r#"
                use audit.{record_call}
                function marker(): int
                    record_call(202)
                    return 202
                end
                "#,
            )
            .module(
                "main",
                r#"
                use alpha.{marker as a_marker}
                use beta.{marker as b_marker}
                function entry(): int
                    return 0
                end
                "#,
            )
            .entry_module("main")
            .compile()
            .expect("compile program with alpha and beta");

        let audit_cb1 = audit_clone.clone();
        program.register_native_fn("record_call", move |args: &[Value]| {
            let code = match args.first() {
                Some(Value::Int(n)) => *n,
                _ => 0,
            };
            let frame_info = VM::with_current(|vm| {
                Ok(vm
                    .current_frame_target()
                    .map(|(idx, name)| (idx, name.to_string())))
            })
            .unwrap();
            audit_cb1.lock().unwrap().push((code, frame_info));
            Ok(NativeCallResult::Return(Value::Int(0)))
        });
        let audit_cb2 = audit_clone.clone();
        program.register_native_fn("audit.record_call", move |args: &[Value]| {
            let code = match args.first() {
                Some(Value::Int(n)) => *n,
                _ => 0,
            };
            let frame_info = VM::with_current(|vm| {
                Ok(vm
                    .current_frame_target()
                    .map(|(idx, name)| (idx, name.to_string())))
            })
            .unwrap();
            audit_cb2.lock().unwrap().push((code, frame_info));
            Ok(NativeCallResult::Return(Value::Int(0)))
        });

        let limits = PreflightLimits::default();

        // 1. Resolve and invoke alpha.marker
        let target_a = program
            .resolve_canonical_target("alpha.marker")
            .expect("resolve alpha.marker");
        assert_eq!(target_a.canonical_name, "alpha.marker");

        let res_a = program
            .call_retained_target_spike(&target_a, Vec::new(), &limits)
            .expect("call alpha.marker");
        assert_eq!(res_a, Value::Int(101));

        // Verify witness of equality: resolved identity == VM last dispatched == invoked frame
        assert_eq!(
            program.last_dispatched_target(),
            Some((target_a.func_idx, "alpha.marker".to_string()))
        );
        let records = audit_records.lock().unwrap().clone();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].0, 101);
        assert_eq!(
            records[0].1,
            Some((target_a.func_idx, "alpha.marker".to_string()))
        );

        // 2. Resolve and invoke beta.marker
        let target_b = program
            .resolve_canonical_target("beta.marker")
            .expect("resolve beta.marker");
        assert_eq!(target_b.canonical_name, "beta.marker");
        assert_ne!(target_a.func_idx, target_b.func_idx);

        let res_b = program
            .call_retained_target_spike(&target_b, Vec::new(), &limits)
            .expect("call beta.marker");
        assert_eq!(res_b, Value::Int(202));

        // Verify witness for beta
        assert_eq!(
            program.last_dispatched_target(),
            Some((target_b.func_idx, "beta.marker".to_string()))
        );
        let records2 = audit_records.lock().unwrap().clone();
        assert_eq!(records2.len(), 2);
        assert_eq!(records2[1].0, 202);
        assert_eq!(
            records2[1].1,
            Some((target_b.func_idx, "beta.marker".to_string()))
        );

        // 3. Negative selector checks on same program
        // Simple-name fallback rejected
        let err_simple = program
            .resolve_canonical_target("marker")
            .expect_err("simple name fallback must fail");
        assert_eq!(err_simple.status, PreflightStatus::Call);
        assert!(
            err_simple
                .message
                .contains("simple-name fallback is rejected")
        );

        // Alternate normalization with :: rejected
        let err_colon = program
            .resolve_canonical_target("alpha::marker")
            .expect_err("alternate normalization :: must fail");
        assert_eq!(err_colon.status, PreflightStatus::Call);
        assert!(
            err_colon
                .message
                .contains("alternate normalization with '::' is rejected")
        );

        // Missing target rejected
        let err_missing = program
            .resolve_canonical_target("nonexistent.func")
            .expect_err("missing target must fail");
        assert_eq!(err_missing.status, PreflightStatus::Call);
        assert!(err_missing.message.contains("not found"));
    }

    #[test]
    fn gate2_preflight_failure_categories_preserve_all_counters() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let callback_count = Arc::new(AtomicUsize::new(0));

        let mut program = EmbeddedProgram::builder()
            .module(
                "audit",
                r#"
                extern
                    function record_call(int): int
                end
                "#,
            )
            .module(
                "test_mod",
                r#"
                use audit.{record_call}
                function valid_fn(x: int, y: int): int
                    record_call(1)
                    return x + y
                end
                function array_fn(items: Array<int>): int
                    record_call(2)
                    return 1
                end
                function nested_fn(nested: Array<Array<int>>): int
                    record_call(3)
                    return 1
                end
                function test_mod:method_fn(self: int): int
                    record_call(4)
                    return 1
                end
                "#,
            )
            .entry_module("test_mod")
            .compile()
            .expect("compile test_mod");

        let cb1 = callback_count.clone();
        program.register_native_fn("record_call", move |_args: &[Value]| {
            cb1.fetch_add(1, Ordering::SeqCst);
            Ok(NativeCallResult::Return(Value::Int(0)))
        });
        let cb2 = callback_count.clone();
        program.register_native_fn("audit.record_call", move |_args: &[Value]| {
            cb2.fetch_add(1, Ordering::SeqCst);
            Ok(NativeCallResult::Return(Value::Int(0)))
        });

        // Initialize baseline global and init markers
        program.set_global_value("test_marker", Value::Int(777));
        program.set_global_value("init_counter", Value::Int(42));

        let limits = PreflightLimits {
            max_depth: 3,
            max_work: 20,
        };

        fn sorted_global_names(p: &EmbeddedProgram) -> Vec<String> {
            let mut names = p.global_names();
            names.sort();
            names
        }

        // Baseline snapshots after builder construction
        let baseline_cb = callback_count.load(Ordering::SeqCst);
        let baseline_marker = program
            .get_typed_global::<i64>("test_marker")
            .expect("read test_marker");
        let baseline_init = program
            .get_typed_global::<i64>("init_counter")
            .expect("read init_counter");
        let baseline_names = sorted_global_names(&program);
        let baseline_dispatched = program.last_dispatched_target();

        let assert_snapshots_unmodified = |program: &EmbeddedProgram, label: &str| {
            assert_eq!(
                callback_count.load(Ordering::SeqCst),
                baseline_cb,
                "[{label}] callback counter modified during preflight failure"
            );
            assert_eq!(
                program
                    .get_typed_global::<i64>("test_marker")
                    .expect("read test_marker"),
                baseline_marker,
                "[{label}] test_marker modified during preflight failure"
            );
            assert_eq!(
                program
                    .get_typed_global::<i64>("init_counter")
                    .expect("read init_counter"),
                baseline_init,
                "[{label}] init_counter modified during preflight failure"
            );
            assert_eq!(
                sorted_global_names(program),
                baseline_names,
                "[{label}] global names modified during preflight failure"
            );
            assert_eq!(
                program.last_dispatched_target(),
                baseline_dispatched,
                "[{label}] last dispatched target modified during preflight failure"
            );
        };

        // Category 1: Unknown target
        let err = program
            .call_preflight_spike("test_mod.unknown_func", Vec::new(), &limits)
            .expect_err("unknown target must fail");
        assert_eq!(err.status, PreflightStatus::Call);
        assert_snapshots_unmodified(&program, "unknown_target");

        // Category 2: Noncanonical selector (alternate normalization ::)
        let err = program
            .call_preflight_spike(
                "test_mod::valid_fn",
                vec![Value::Int(1), Value::Int(2)],
                &limits,
            )
            .expect_err(":: normalization must fail");
        assert_eq!(err.status, PreflightStatus::Call);
        assert_snapshots_unmodified(&program, "noncanonical_colon");

        // Category 3: Noncanonical selector (simple-name fallback)
        let err = program
            .call_preflight_spike("valid_fn", vec![Value::Int(1), Value::Int(2)], &limits)
            .expect_err("simple name fallback must fail");
        assert_eq!(err.status, PreflightStatus::Call);
        assert_snapshots_unmodified(&program, "noncanonical_simple_name");

        // Category 4: Method target rejected
        let err = program
            .call_preflight_spike("test_mod:method_fn", Vec::new(), &limits)
            .expect_err("method target must fail");
        assert_eq!(err.status, PreflightStatus::Call);
        assert_snapshots_unmodified(&program, "method_target");

        // Category 5: Wrong arity (too few)
        let err = program
            .call_preflight_spike("test_mod.valid_fn", vec![Value::Int(1)], &limits)
            .expect_err("wrong arity (under) must fail");
        assert_eq!(err.status, PreflightStatus::Type);
        assert_snapshots_unmodified(&program, "wrong_arity_under");

        // Category 6: Wrong arity (too many)
        let err = program
            .call_preflight_spike(
                "test_mod.valid_fn",
                vec![Value::Int(1), Value::Int(2), Value::Int(3)],
                &limits,
            )
            .expect_err("wrong arity (over) must fail");
        assert_eq!(err.status, PreflightStatus::Type);
        assert_snapshots_unmodified(&program, "wrong_arity_over");

        // Category 7: Wrong scalar type
        let err = program
            .call_preflight_spike(
                "test_mod.valid_fn",
                vec![Value::Int(1), Value::String(Rc::new("bad".into()))],
                &limits,
            )
            .expect_err("wrong scalar type must fail");
        assert_eq!(err.status, PreflightStatus::Type);
        assert_snapshots_unmodified(&program, "wrong_scalar_type");

        // Category 8: Wrong nested type
        let err = program
            .call_preflight_spike(
                "test_mod.array_fn",
                vec![Value::Array(Rc::new(RefCell::new(vec![Value::String(
                    Rc::new("bad".into()),
                )])))],
                &limits,
            )
            .expect_err("wrong nested type must fail");
        assert_eq!(err.status, PreflightStatus::Type);
        assert_snapshots_unmodified(&program, "wrong_nested_type");

        // Category 9: Absent / unresolved signature
        let func_idx = program
            .vm
            .functions
            .iter()
            .position(|f| f.name == "test_mod.valid_fn")
            .unwrap();
        let orig_sig = program.vm.functions[func_idx].signature.take();
        let err = program
            .resolve_canonical_target("test_mod.valid_fn")
            .expect_err("absent signature must fail");
        assert_eq!(err.status, PreflightStatus::Type);
        assert!(err.message.contains("no type signature"));
        program.vm.functions[func_idx].signature = orig_sig;
        assert_snapshots_unmodified(&program, "absent_signature");

        // Category 10: Retained signature integrity mismatch (tampered target signature)
        let mut generic_sig = program
            .resolve_canonical_target("test_mod.valid_fn")
            .unwrap();
        generic_sig.signature.params[0] =
            Type::new(TypeKind::Generic("T".into()), Span::new(0, 0, 0, 0));
        let err = program
            .call_retained_target_spike(&generic_sig, vec![Value::Int(1), Value::Int(2)], &limits)
            .expect_err("tampered signature must fail");
        assert_eq!(err.status, PreflightStatus::Call);
        assert!(
            err.message
                .contains("retained signature integrity mismatch")
        );
        assert_snapshots_unmodified(&program, "tampered_generic_schema");

        // Category 11: Empty-container unsupported schema (TypeKind::Unknown in Array)
        // In naive value_matches_type, empty array passes vacuously. In Gate 2, schema is validated first!
        let empty_bad_type = Type::new(
            TypeKind::Array(Box::new(Type::new(
                TypeKind::Unknown,
                Span::new(0, 0, 0, 0),
            ))),
            Span::new(0, 0, 0, 0),
        );
        let empty_array = Value::Array(Rc::new(RefCell::new(Vec::new())));
        let mut empty_budget = WorkBudget::new(limits.max_work, limits.max_depth);
        let err = program
            .validate_argument_value(&empty_array, &empty_bad_type, 0, &mut empty_budget)
            .expect_err("empty container with unsupported schema must fail");
        assert_eq!(err.status, PreflightStatus::Type);
        assert_snapshots_unmodified(&program, "empty_container_unsupported_schema");

        // Category 12: Depth limit exceeded
        // Array<Array<int>> has depth 2, while tight_limits allows max_depth: 1
        let depth1 = Value::Array(Rc::new(RefCell::new(vec![Value::Int(1)])));
        let depth2 = Value::Array(Rc::new(RefCell::new(vec![depth1])));
        let tight_limits = PreflightLimits {
            max_depth: 1,
            max_work: 100,
        };
        let err = program
            .call_preflight_spike("test_mod.nested_fn", vec![depth2], &tight_limits)
            .expect_err("depth limit must fail");
        assert_eq!(err.status, PreflightStatus::Limit);
        assert_snapshots_unmodified(&program, "depth_limit");

        // Category 13: Work budget limit exceeded
        // Array with 30 items while max_work is 5
        let large_items: Vec<Value> = (0..30).map(Value::Int).collect();
        let large_array = Value::Array(Rc::new(RefCell::new(large_items)));
        let work_limits = PreflightLimits {
            max_depth: 64,
            max_work: 5,
        };
        let err = program
            .call_preflight_spike("test_mod.array_fn", vec![large_array], &work_limits)
            .expect_err("work limit must fail");
        assert_eq!(err.status, PreflightStatus::Limit);
        assert_snapshots_unmodified(&program, "work_limit");

        // Positive control: after proving all 13 failure categories preserved baseline snapshots,
        // execute a valid call to prove that actual execution advances counters as expected.
        let ok_res = program
            .call_preflight_spike(
                "test_mod.valid_fn",
                vec![Value::Int(10), Value::Int(20)],
                &limits,
            )
            .expect("valid call must succeed");
        assert_eq!(ok_res, Value::Int(30));
        assert_eq!(callback_count.load(Ordering::SeqCst), baseline_cb + 1);
        assert_eq!(
            program
                .last_dispatched_target()
                .as_ref()
                .map(|(_, name)| name.as_str()),
            Some("test_mod.valid_fn")
        );
    }

    #[test]
    fn gate2_positive_valid_call_and_runtime_error_boundary() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let callback_count = Arc::new(AtomicUsize::new(0));

        let mut program = EmbeddedProgram::builder()
            .module(
                "audit",
                r#"
                extern
                    function record_call(int): int
                end
                "#,
            )
            .module(
                "math_mod",
                r#"
                use audit.{record_call}
                function add(a: int, b: int): int
                    record_call(10)
                    return a + b
                end
                function divide(a: int, b: int): int
                    record_call(20)
                    return a / b
                end
                "#,
            )
            .entry_module("math_mod")
            .compile()
            .expect("compile math_mod");

        let cb1 = callback_count.clone();
        program.register_native_fn("record_call", move |_args: &[Value]| {
            cb1.fetch_add(1, Ordering::SeqCst);
            Ok(NativeCallResult::Return(Value::Int(0)))
        });
        let cb2 = callback_count.clone();
        program.register_native_fn("audit.record_call", move |_args: &[Value]| {
            cb2.fetch_add(1, Ordering::SeqCst);
            Ok(NativeCallResult::Return(Value::Int(0)))
        });

        let limits = PreflightLimits::default();

        // 1. Positive valid call
        let res = program
            .call_preflight_spike(
                "math_mod.add",
                vec![Value::Int(15), Value::Int(27)],
                &limits,
            )
            .expect("valid call must succeed");
        assert_eq!(res, Value::Int(42));
        assert_eq!(callback_count.load(Ordering::SeqCst), 1);
        assert_eq!(
            program
                .last_dispatched_target()
                .as_ref()
                .map(|(_, name)| name.as_str()),
            Some("math_mod.add")
        );

        // 2. Post-invocation runtime error (division by zero)
        // Preflight validation succeeds, but execution triggers a runtime error.
        let err = program
            .call_preflight_spike(
                "math_mod.divide",
                vec![Value::Int(10), Value::Int(0)],
                &limits,
            )
            .expect_err("runtime error must fail");
        assert_eq!(err.status, PreflightStatus::Call);
        assert!(err.message.contains("Runtime execution error"));

        // Callback WAS called because preflight succeeded and execution began
        assert_eq!(
            callback_count.load(Ordering::SeqCst),
            2,
            "Callback was executed once for add and once for divide before division-by-zero"
        );

        // 3. Subsequent valid call on the same program after runtime error
        // Verifies VM and program state was cleanly unwound and is fully reusable.
        let target_add = program
            .resolve_canonical_target("math_mod.add")
            .expect("resolve math_mod.add");
        let res_subsequent = program
            .call_retained_target_spike(
                &target_add,
                vec![Value::Int(100), Value::Int(200)],
                &limits,
            )
            .expect("subsequent retained call on same program after runtime error must succeed");
        assert_eq!(res_subsequent, Value::Int(300));
        assert_eq!(
            callback_count.load(Ordering::SeqCst),
            3,
            "Callback was executed third time during successful subsequent call"
        );
        assert_eq!(
            program
                .last_dispatched_target()
                .as_ref()
                .map(|(_, name)| name.as_str()),
            Some("math_mod.add")
        );

        let res_subsequent_spike = program
            .call_preflight_spike("math_mod.add", vec![Value::Int(1), Value::Int(2)], &limits)
            .expect("subsequent preflight spike on same program must succeed");
        assert_eq!(res_subsequent_spike, Value::Int(3));
        assert_eq!(callback_count.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn gate2_cross_program_target_and_unsupported_schemas_rejected() {
        use std::rc::Rc;

        let mut program1 = EmbeddedProgram::builder()
            .module(
                "mod1",
                r#"
                function compute(x: int): int
                    return x + 1
                end
                "#,
            )
            .entry_module("mod1")
            .compile()
            .expect("compile program1");

        let mut program2 = EmbeddedProgram::builder()
            .module(
                "mod1",
                r#"
                function compute(x: string): string
                    return x
                end
                "#,
            )
            .entry_module("mod1")
            .compile()
            .expect("compile program2");

        let target1 = program1
            .resolve_canonical_target("mod1.compute")
            .expect("resolve target1");
        let limits = PreflightLimits::default();

        // Cross-program invocation with target from program1 on program2 must be rejected
        let err = program2
            .call_retained_target_spike(&target1, vec![Value::Int(10)], &limits)
            .expect_err("cross-program target invocation must be rejected");
        assert_eq!(err.status, PreflightStatus::Call);
        assert!(err.message.contains("originating EmbeddedProgram"));

        // Verify that program tokens are distinct pointers
        assert!(!Rc::ptr_eq(
            &program1.program_token,
            &program2.program_token
        ));

        // Tamper test 1: Forging program_id alone does NOT bypass Rc::ptr_eq check
        let mut target_tampered_id = target1.clone();
        target_tampered_id.program_id = program2.program_id;
        let err = program2
            .call_retained_target_spike(&target_tampered_id, vec![Value::Int(10)], &limits)
            .expect_err("forged program_id must still be rejected by Rc::ptr_eq");
        assert_eq!(err.status, PreflightStatus::Call);
        assert!(err.message.contains("originating EmbeddedProgram"));

        // Tamper test 2: Forging program_token to match program2 is caught by executable signature mismatch
        let mut target_tampered_token = target1.clone();
        target_tampered_token.program_id = program2.program_id;
        target_tampered_token.program_token = program2.program_token.clone();
        let err = program2
            .call_retained_target_spike(&target_tampered_token, vec![Value::Int(10)], &limits)
            .expect_err("forged token with differing signature must be rejected");
        assert_eq!(err.status, PreflightStatus::Call);
        assert!(err.message.contains("Target executable signature mismatch"));

        // Same-name, same-index, same-signature foreign program collision test:
        // program3 defines the exact same module name, function name, and signature as program1.
        let mut program3 = EmbeddedProgram::builder()
            .module(
                "mod1",
                r#"
                function compute(x: int): int
                    return x + 100
                end
                "#,
            )
            .entry_module("mod1")
            .compile()
            .expect("compile program3");

        let target3 = program3
            .resolve_canonical_target("mod1.compute")
            .expect("resolve target3");

        // Verify exact structural collision: name, index, signature, and executable_sig all match
        assert_eq!(target1.canonical_name, target3.canonical_name);
        assert_eq!(target1.func_idx, target3.func_idx);
        assert_eq!(target1.signature.to_string(), target3.signature.to_string());
        assert_eq!(target1.executable_sig, target3.executable_sig);
        // But program tokens/ids are strictly distinct:
        assert_ne!(target1.program_id, target3.program_id);
        assert!(!Rc::ptr_eq(&target1.program_token, &target3.program_token));

        // Executing target1 on foreign program3 MUST be rejected even though name, index, and signature match identically!
        let err_same_sig_foreign = program3
            .call_retained_target_spike(&target1, vec![Value::Int(10)], &limits)
            .expect_err("same-name/index/signature foreign target invocation must be rejected");
        assert_eq!(err_same_sig_foreign.status, PreflightStatus::Call);
        assert!(
            err_same_sig_foreign
                .message
                .contains("originating EmbeddedProgram")
        );

        // Executing target3 on foreign program1 MUST be rejected
        let err_same_sig_foreign_rev = program1
            .call_retained_target_spike(&target3, vec![Value::Int(10)], &limits)
            .expect_err("same-name/index/signature foreign target invocation must be rejected");
        assert_eq!(err_same_sig_foreign_rev.status, PreflightStatus::Call);
        assert!(
            err_same_sig_foreign_rev
                .message
                .contains("originating EmbeddedProgram")
        );

        // Valid executions on their originating programs succeed
        let res1 = program1
            .call_retained_target_spike(&target1, vec![Value::Int(10)], &limits)
            .expect("target1 on program1 succeeds");
        assert_eq!(res1, Value::Int(11));

        let res3 = program3
            .call_retained_target_spike(&target3, vec![Value::Int(10)], &limits)
            .expect("target3 on program3 succeeds");
        assert_eq!(res3, Value::Int(110));

        // Tamper test 3: Forging a fresh Rc::new(()) program_token with target's own program_id fails
        let mut target_fresh_token = target1.clone();
        target_fresh_token.program_token = Rc::new(());
        let err_fresh_token = program1
            .call_retained_target_spike(&target_fresh_token, vec![Value::Int(10)], &limits)
            .expect_err("freshly allocated forged token must be rejected by Rc::ptr_eq");
        assert_eq!(err_fresh_token.status, PreflightStatus::Call);
        assert!(
            err_fresh_token
                .message
                .contains("originating EmbeddedProgram")
        );

        // Program with direct Tuple, Option, Result schemas in declarations
        let program_schemas = EmbeddedProgram::builder()
            .module(
                "schemas",
                r#"
                function opt_fn(x: Option<int>): int
                    return 1
                end
                function tuple_fn(x: (int, int)): int
                    return 2
                end
                function result_fn(x: Result<int, string>): int
                    return 3
                end
                function nested_opt_fn(x: Array<Option<int>>): int
                    return 4
                end
                "#,
            )
            .entry_module("schemas")
            .compile()
            .expect("compile program_schemas");

        // Schema rejection directly during resolve_canonical_target
        let err = program_schemas
            .resolve_canonical_target("schemas.opt_fn")
            .expect_err("Option parameter schema must be rejected on resolve");
        assert_eq!(err.status, PreflightStatus::Type);
        assert!(err.message.contains("Unsupported schema: Option"));

        let err = program_schemas
            .resolve_canonical_target("schemas.tuple_fn")
            .expect_err("Tuple parameter schema must be rejected on resolve");
        assert_eq!(err.status, PreflightStatus::Type);
        assert!(err.message.contains("Unsupported schema: Tuple"));

        let err = program_schemas
            .resolve_canonical_target("schemas.result_fn")
            .expect_err("Result parameter schema must be rejected on resolve");
        assert_eq!(err.status, PreflightStatus::Type);
        assert!(err.message.contains("Unsupported schema: Result"));

        let err = program_schemas
            .resolve_canonical_target("schemas.nested_opt_fn")
            .expect_err("Nested Option parameter schema must be rejected on resolve");
        assert_eq!(err.status, PreflightStatus::Type);
        assert!(err.message.contains("Unsupported schema: Option"));

        // Retained signature tampering rejection on call_retained_target_spike (Part A)
        // 1. Tuple signature tampering
        let mut target_tuple = target1.clone();
        target_tuple.signature.params[0] = Type::new(
            TypeKind::Tuple(vec![Type::new(TypeKind::Int, Span::new(0, 0, 0, 0))]),
            Span::new(0, 0, 0, 0),
        );
        let err = program1
            .call_retained_target_spike(
                &target_tuple,
                vec![Value::Tuple(Rc::new(vec![Value::Int(1)]))],
                &limits,
            )
            .expect_err("tampered tuple signature must be rejected before invocation");
        assert_eq!(err.status, PreflightStatus::Call);
        assert!(
            err.message
                .contains("retained signature integrity mismatch")
        );

        // 2. Option signature tampering
        let mut target_opt = target1.clone();
        target_opt.signature.params[0] = Type::new(
            TypeKind::Option(Box::new(Type::new(TypeKind::Int, Span::new(0, 0, 0, 0)))),
            Span::new(0, 0, 0, 0),
        );
        let err = program1
            .call_retained_target_spike(&target_opt, vec![Value::Nil], &limits)
            .expect_err("tampered option signature must be rejected before invocation");
        assert_eq!(err.status, PreflightStatus::Call);
        assert!(
            err.message
                .contains("retained signature integrity mismatch")
        );

        // 3. Result signature tampering
        let mut target_res = target1.clone();
        target_res.signature.params[0] = Type::new(
            TypeKind::Result(
                Box::new(Type::new(TypeKind::Int, Span::new(0, 0, 0, 0))),
                Box::new(Type::new(TypeKind::String, Span::new(0, 0, 0, 0))),
            ),
            Span::new(0, 0, 0, 0),
        );
        let err = program1
            .call_retained_target_spike(&target_res, vec![Value::Nil], &limits)
            .expect_err("tampered result signature must be rejected before invocation");
        assert_eq!(err.status, PreflightStatus::Call);
        assert!(
            err.message
                .contains("retained signature integrity mismatch")
        );

        // Valid untampered target1 still succeeds
        let valid_res = program1
            .call_retained_target_spike(&target1, vec![Value::Int(41)], &limits)
            .expect("valid retained target must succeed");
        assert_eq!(valid_res, Value::Int(42));

        // Recursive schema rejection on empty containers via validate_argument_value
        let empty_arr = Value::Array(Rc::new(RefCell::new(Vec::new())));
        let empty_map = Value::Map(Rc::new(RefCell::new(crate::bytecode::LustMap::default())));

        // 1. Empty array with Option schema
        let type_empty_arr_opt = Type::new(
            TypeKind::Array(Box::new(Type::new(
                TypeKind::Option(Box::new(Type::new(TypeKind::Int, Span::new(0, 0, 0, 0)))),
                Span::new(0, 0, 0, 0),
            ))),
            Span::new(0, 0, 0, 0),
        );
        let mut budget = WorkBudget::new(limits.max_work, limits.max_depth);
        let err = program1
            .validate_argument_value(&empty_arr, &type_empty_arr_opt, 0, &mut budget)
            .expect_err("empty array with Option schema must be rejected");
        assert_eq!(err.status, PreflightStatus::Type);
        assert!(err.message.contains("Unsupported schema: Option"));

        // 2. Empty array with Tuple schema
        let type_empty_arr_tuple = Type::new(
            TypeKind::Array(Box::new(Type::new(
                TypeKind::Tuple(vec![Type::new(TypeKind::Int, Span::new(0, 0, 0, 0))]),
                Span::new(0, 0, 0, 0),
            ))),
            Span::new(0, 0, 0, 0),
        );
        let mut budget = WorkBudget::new(limits.max_work, limits.max_depth);
        let err = program1
            .validate_argument_value(&empty_arr, &type_empty_arr_tuple, 0, &mut budget)
            .expect_err("empty array with Tuple schema must be rejected");
        assert_eq!(err.status, PreflightStatus::Type);
        assert!(err.message.contains("Unsupported schema: Tuple"));

        // 3. Empty array with Result schema
        let type_empty_arr_res = Type::new(
            TypeKind::Array(Box::new(Type::new(
                TypeKind::Result(
                    Box::new(Type::new(TypeKind::Int, Span::new(0, 0, 0, 0))),
                    Box::new(Type::new(TypeKind::String, Span::new(0, 0, 0, 0))),
                ),
                Span::new(0, 0, 0, 0),
            ))),
            Span::new(0, 0, 0, 0),
        );
        let mut budget = WorkBudget::new(limits.max_work, limits.max_depth);
        let err = program1
            .validate_argument_value(&empty_arr, &type_empty_arr_res, 0, &mut budget)
            .expect_err("empty array with Result schema must be rejected");
        assert_eq!(err.status, PreflightStatus::Type);
        assert!(err.message.contains("Unsupported schema: Result"));

        // 4. Empty map with Option schema
        let type_empty_map_opt = Type::new(
            TypeKind::Map(
                Box::new(Type::new(TypeKind::String, Span::new(0, 0, 0, 0))),
                Box::new(Type::new(
                    TypeKind::Option(Box::new(Type::new(TypeKind::Int, Span::new(0, 0, 0, 0)))),
                    Span::new(0, 0, 0, 0),
                )),
            ),
            Span::new(0, 0, 0, 0),
        );
        let mut budget = WorkBudget::new(limits.max_work, limits.max_depth);
        let err = program1
            .validate_argument_value(&empty_map, &type_empty_map_opt, 0, &mut budget)
            .expect_err("empty map with Option schema must be rejected");
        assert_eq!(err.status, PreflightStatus::Type);
        assert!(err.message.contains("Unsupported schema: Option"));

        // 5. Nested empty array: [[Option<int>]] with [[]]
        let type_nested_empty = Type::new(
            TypeKind::Array(Box::new(Type::new(
                TypeKind::Array(Box::new(Type::new(
                    TypeKind::Option(Box::new(Type::new(TypeKind::Int, Span::new(0, 0, 0, 0)))),
                    Span::new(0, 0, 0, 0),
                ))),
                Span::new(0, 0, 0, 0),
            ))),
            Span::new(0, 0, 0, 0),
        );
        let nested_empty = Value::Array(Rc::new(RefCell::new(vec![empty_arr])));
        let mut budget = WorkBudget::new(limits.max_work, limits.max_depth);
        let err = program1
            .validate_argument_value(&nested_empty, &type_nested_empty, 0, &mut budget)
            .expect_err("nested empty array with Option schema must be rejected");
        assert_eq!(err.status, PreflightStatus::Type);
        assert!(err.message.contains("Unsupported schema: Option"));

        // 6. Empty map with Tuple schema
        let type_empty_map_tuple = Type::new(
            TypeKind::Map(
                Box::new(Type::new(TypeKind::String, Span::new(0, 0, 0, 0))),
                Box::new(Type::new(
                    TypeKind::Tuple(vec![Type::new(TypeKind::Int, Span::new(0, 0, 0, 0))]),
                    Span::new(0, 0, 0, 0),
                )),
            ),
            Span::new(0, 0, 0, 0),
        );
        let mut budget = WorkBudget::new(limits.max_work, limits.max_depth);
        let err = program1
            .validate_argument_value(&empty_map, &type_empty_map_tuple, 0, &mut budget)
            .expect_err("empty map with Tuple schema must be rejected");
        assert_eq!(err.status, PreflightStatus::Type);
        assert!(err.message.contains("Unsupported schema: Tuple"));

        // 7. Empty map with Result schema
        let type_empty_map_res = Type::new(
            TypeKind::Map(
                Box::new(Type::new(TypeKind::String, Span::new(0, 0, 0, 0))),
                Box::new(Type::new(
                    TypeKind::Result(
                        Box::new(Type::new(TypeKind::Int, Span::new(0, 0, 0, 0))),
                        Box::new(Type::new(TypeKind::String, Span::new(0, 0, 0, 0))),
                    ),
                    Span::new(0, 0, 0, 0),
                )),
            ),
            Span::new(0, 0, 0, 0),
        );
        let mut budget = WorkBudget::new(limits.max_work, limits.max_depth);
        let err = program1
            .validate_argument_value(&empty_map, &type_empty_map_res, 0, &mut budget)
            .expect_err("empty map with Result schema must be rejected");
        assert_eq!(err.status, PreflightStatus::Type);
        assert!(err.message.contains("Unsupported schema: Result"));

        // 8. Empty map with non-string key schema
        let type_empty_map_bad_key = Type::new(
            TypeKind::Map(
                Box::new(Type::new(TypeKind::Int, Span::new(0, 0, 0, 0))),
                Box::new(Type::new(TypeKind::Int, Span::new(0, 0, 0, 0))),
            ),
            Span::new(0, 0, 0, 0),
        );
        let mut budget = WorkBudget::new(limits.max_work, limits.max_depth);
        let err = program1
            .validate_argument_value(&empty_map, &type_empty_map_bad_key, 0, &mut budget)
            .expect_err("empty map with int key schema must be rejected");
        assert_eq!(err.status, PreflightStatus::Type);
        assert!(err.message.contains("require string keys"));

        // 9. Nested container in map: Map<string, Array<Tuple<int>>> with empty map
        let type_empty_map_nested = Type::new(
            TypeKind::Map(
                Box::new(Type::new(TypeKind::String, Span::new(0, 0, 0, 0))),
                Box::new(Type::new(
                    TypeKind::Array(Box::new(Type::new(
                        TypeKind::Tuple(vec![Type::new(TypeKind::Int, Span::new(0, 0, 0, 0))]),
                        Span::new(0, 0, 0, 0),
                    ))),
                    Span::new(0, 0, 0, 0),
                )),
            ),
            Span::new(0, 0, 0, 0),
        );
        let mut budget = WorkBudget::new(limits.max_work, limits.max_depth);
        let err = program1
            .validate_argument_value(&empty_map, &type_empty_map_nested, 0, &mut budget)
            .expect_err("nested container with Tuple schema in empty map must be rejected");
        assert_eq!(err.status, PreflightStatus::Type);
        assert!(err.message.contains("Unsupported schema: Tuple"));
    }

    #[test]
    fn gate2_part_a_retained_signature_tampering_and_part_b_return_validation() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let host_counter = Arc::new(AtomicUsize::new(0));
        let c1 = host_counter.clone();

        let mut program = EmbeddedProgram::builder()
            .module(
                "audit",
                r#"
                extern
                    function record_call(int): int
                end
                "#,
            )
            .module(
                "test_pkg",
                r#"
                use audit.{record_call}

                function side_effect_fn(x: int): int
                    record_call(1)
                    return x + 10
                end

                function return_opt_fn(x: int): Option<int>
                    record_call(1)
                    return Option.Some(x)
                end

                function return_tuple_fn(x: int): (int, int)
                    record_call(1)
                    return (x, 1)
                end

                function return_result_fn(x: int): Result<int, string>
                    record_call(1)
                    return Result.Ok(x)
                end

                function return_nan_fn(): float
                    record_call(1)
                    return 0.0 / 0.0
                end

                function return_deep_array(x: int): Array<Array<int>>
                    record_call(1)
                    return [[x]]
                end

                function divide_zero(x: int): int
                    record_call(1)
                    return x / 0
                end
                "#,
            )
            .entry_module("test_pkg")
            .compile()
            .expect("compile test_pkg");

        let c1_dup = host_counter.clone();
        program.register_native_fn("record_call", move |_args: &[Value]| {
            c1_dup.fetch_add(1, Ordering::SeqCst);
            Ok(NativeCallResult::Return(Value::Int(0)))
        });
        program.register_native_fn("audit.record_call", move |_args: &[Value]| {
            c1.fetch_add(1, Ordering::SeqCst);
            Ok(NativeCallResult::Return(Value::Int(0)))
        });

        let limits = PreflightLimits::default();

        // -------------------------------------------------------------------
        // Part A: Retained Signature Integrity & Tampering Rejection
        // -------------------------------------------------------------------
        let valid_target = program
            .resolve_canonical_target("test_pkg.side_effect_fn")
            .expect("resolve valid_target");

        // Tamper with parameter type
        let mut tampered_param_target = valid_target.clone();
        tampered_param_target.signature.params[0] =
            Type::new(TypeKind::String, Span::new(0, 0, 0, 0));

        let err_param = program
            .call_retained_target_spike(
                &tampered_param_target,
                vec![Value::String(Rc::new("tampered".into()))],
                &limits,
            )
            .expect_err("tampered param target must be rejected before VM invocation");
        assert_eq!(err_param.status, PreflightStatus::Call);
        assert!(
            err_param
                .message
                .contains("Target retained signature integrity mismatch")
        );
        // PROOF: Zero side effects occurred!
        assert_eq!(
            host_counter.load(Ordering::SeqCst),
            0,
            "Rejection before VM dispatch must produce zero side effects"
        );

        // Tamper with return type
        let mut tampered_ret_target = valid_target.clone();
        tampered_ret_target.signature.return_type =
            Type::new(TypeKind::String, Span::new(0, 0, 0, 0));

        let err_ret = program
            .call_retained_target_spike(&tampered_ret_target, vec![Value::Int(42)], &limits)
            .expect_err("tampered return type target must be rejected before VM invocation");
        assert_eq!(err_ret.status, PreflightStatus::Call);
        assert!(
            err_ret
                .message
                .contains("Target retained signature integrity mismatch")
        );
        // PROOF: Still zero side effects!
        assert_eq!(
            host_counter.load(Ordering::SeqCst),
            0,
            "Rejection before VM dispatch must produce zero side effects"
        );

        // Valid retained target succeeds and advances counter by exactly 1
        let res_ok = program
            .call_retained_target_spike(&valid_target, vec![Value::Int(32)], &limits)
            .expect("valid retained target must succeed");
        assert_eq!(res_ok, Value::Int(42));
        assert_eq!(host_counter.load(Ordering::SeqCst), 1);

        // -------------------------------------------------------------------
        // Part B: Post-Call Return Validation
        // For unsupported return schema, error happens ONLY after VM execution.
        // Side-effect counter must advance exactly once and not roll back.
        // -------------------------------------------------------------------

        // 1. Return schema: Option<int>
        let target_ret_opt = program
            .resolve_canonical_target("test_pkg.return_opt_fn")
            .expect("resolve return_opt_fn succeeds because parameter schema is valid int");
        let err_opt = program
            .call_retained_target_spike(&target_ret_opt, vec![Value::Int(7)], &limits)
            .expect_err("post-call return validation must reject Option return schema");
        assert_eq!(err_opt.status, PreflightStatus::Type);
        assert!(
            err_opt
                .message
                .contains("Post-call return validation error")
        );
        assert!(err_opt.message.contains("Unsupported schema: Option"));
        // PROOF: Counter increased exactly once (1 -> 2) and was NOT rolled back!
        assert_eq!(
            host_counter.load(Ordering::SeqCst),
            2,
            "Invocation occurred exactly once before post-call validation failure"
        );

        // 2. Return schema: Tuple (int, int)
        let target_ret_tuple = program
            .resolve_canonical_target("test_pkg.return_tuple_fn")
            .expect("resolve return_tuple_fn succeeds");
        let err_tuple = program
            .call_retained_target_spike(&target_ret_tuple, vec![Value::Int(8)], &limits)
            .expect_err("post-call return validation must reject Tuple return schema");
        assert_eq!(err_tuple.status, PreflightStatus::Type);
        assert!(
            err_tuple
                .message
                .contains("Post-call return validation error")
        );
        assert!(err_tuple.message.contains("Unsupported schema: Tuple"));
        // PROOF: Counter increased exactly once (2 -> 3) and was NOT rolled back!
        assert_eq!(
            host_counter.load(Ordering::SeqCst),
            3,
            "Invocation occurred exactly once before post-call validation failure"
        );

        // 3. Return schema: Result<int, string>
        let target_ret_res = program
            .resolve_canonical_target("test_pkg.return_result_fn")
            .expect("resolve return_result_fn succeeds");
        let err_res = program
            .call_retained_target_spike(&target_ret_res, vec![Value::Int(9)], &limits)
            .expect_err("post-call return validation must reject Result return schema");
        assert_eq!(err_res.status, PreflightStatus::Type);
        assert!(
            err_res
                .message
                .contains("Post-call return validation error")
        );
        assert!(err_res.message.contains("Unsupported schema: Result"));
        // PROOF: Counter increased exactly once (3 -> 4) and was NOT rolled back!
        assert_eq!(
            host_counter.load(Ordering::SeqCst),
            4,
            "Invocation occurred exactly once before post-call validation failure"
        );

        // 4. Return value: Non-finite float (NaN)
        let target_ret_nan = program
            .resolve_canonical_target("test_pkg.return_nan_fn")
            .expect("resolve return_nan_fn succeeds");
        let err_nan = program
            .call_retained_target_spike(&target_ret_nan, Vec::new(), &limits)
            .expect_err("post-call return validation must reject non-finite float");
        assert_eq!(err_nan.status, PreflightStatus::Type);
        assert!(err_nan.message.contains("Non-finite float"));
        // PROOF: Counter increased exactly once (4 -> 5) and was NOT rolled back!
        assert_eq!(host_counter.load(Ordering::SeqCst), 5);

        // 5. Return value: Traversal depth limit exceeded
        let target_ret_deep = program
            .resolve_canonical_target("test_pkg.return_deep_array")
            .expect("resolve return_deep_array succeeds");
        let tight_depth_limits = PreflightLimits {
            max_depth: 1,
            max_work: 100,
        };
        let err_depth = program
            .call_retained_target_spike(&target_ret_deep, vec![Value::Int(1)], &tight_depth_limits)
            .expect_err("post-call return validation must enforce depth limit");
        assert_eq!(err_depth.status, PreflightStatus::Limit);
        assert!(
            err_depth
                .message
                .contains("Post-call return validation error")
        );
        assert!(err_depth.message.contains("Max depth exceeded"));
        // PROOF: Counter increased exactly once (5 -> 6) and was NOT rolled back!
        assert_eq!(host_counter.load(Ordering::SeqCst), 6);

        // -------------------------------------------------------------------
        // Distinct Error Classification & Clean VM Recovery
        // -------------------------------------------------------------------

        // A. Preflight error: wrong argument type produces zero side effects
        let err_preflight = program
            .call_retained_target_spike(
                &valid_target,
                vec![Value::String(Rc::new("bad".into()))],
                &limits,
            )
            .expect_err("preflight argument validation must fail");
        assert_eq!(err_preflight.status, PreflightStatus::Type);
        assert!(err_preflight.message.contains("Argument 1"));
        assert_eq!(
            host_counter.load(Ordering::SeqCst),
            6,
            "Preflight failure must NOT invoke VM and must have zero side effects"
        );

        // B. Runtime execution error: status is Call, distinct from post-call validation
        let target_divide = program
            .resolve_canonical_target("test_pkg.divide_zero")
            .expect("resolve divide_zero succeeds");
        let err_runtime = program
            .call_retained_target_spike(&target_divide, vec![Value::Int(10)], &limits)
            .expect_err("division by zero must trigger runtime error");
        assert_eq!(err_runtime.status, PreflightStatus::Call);
        assert!(err_runtime.message.contains("Runtime execution error"));
        assert_eq!(
            host_counter.load(Ordering::SeqCst),
            7,
            "Runtime execution error advances counter once during execution"
        );

        // C. Clean subsequent invocation on the same program succeeds cleanly
        let subsequent_ok = program
            .call_retained_target_spike(&valid_target, vec![Value::Int(90)], &limits)
            .expect("subsequent invocation on same program must succeed");
        assert_eq!(subsequent_ok, Value::Int(100));
        assert_eq!(host_counter.load(Ordering::SeqCst), 8);
    }

    #[test]
    fn gate2_return_validation_borrow_remediation_and_nested_containers() {
        use crate::bytecode::{LustMap, ValueKey};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let host_counter = Arc::new(AtomicUsize::new(0));

        let shared_array = Rc::new(RefCell::new(vec![Value::Int(10), Value::Int(20)]));
        let shared_array_cb = shared_array.clone();

        let mut base_map = LustMap::default();
        base_map.insert(ValueKey::from("k1"), Value::Int(100));
        let shared_map = Rc::new(RefCell::new(base_map));
        let shared_map_cb = shared_map.clone();

        let mut program = EmbeddedProgram::builder()
            .module(
                "audit",
                r#"
                extern
                    function record_call(int): int
                end
                "#,
            )
            .module(
                "borrow_pkg",
                r#"
                use audit.{record_call}

                extern
                    function get_borrowed_array(): Array<int>
                    function get_borrowed_map(): Map<string, int>
                    function get_nested_array(): Array<Array<int>>
                    function get_nested_map(): Map<string, Map<string, int>>
                end

                function return_borrowed_array(): Array<int>
                    record_call(1)
                    return get_borrowed_array()
                end

                function return_borrowed_map(): Map<string, int>
                    record_call(1)
                    return get_borrowed_map()
                end

                function return_nested_array(): Array<Array<int>>
                    record_call(1)
                    return get_nested_array()
                end

                function return_nested_map(): Map<string, Map<string, int>>
                    record_call(1)
                    return get_nested_map()
                end
                "#,
            )
            .entry_module("borrow_pkg")
            .compile()
            .expect("compile borrow_pkg");

        let c1 = host_counter.clone();
        program.register_native_fn("record_call", move |_args: &[Value]| {
            c1.fetch_add(1, Ordering::SeqCst);
            Ok(NativeCallResult::Return(Value::Int(0)))
        });
        let c2 = host_counter.clone();
        program.register_native_fn("audit.record_call", move |_args: &[Value]| {
            c2.fetch_add(1, Ordering::SeqCst);
            Ok(NativeCallResult::Return(Value::Int(0)))
        });

        let sa1 = shared_array_cb.clone();
        program.register_native_fn("get_borrowed_array", move |_args: &[Value]| {
            Ok(NativeCallResult::Return(Value::Array(sa1.clone())))
        });
        let sa2 = shared_array_cb.clone();
        program.register_native_fn("borrow_pkg.get_borrowed_array", move |_args: &[Value]| {
            Ok(NativeCallResult::Return(Value::Array(sa2.clone())))
        });

        let sm1 = shared_map_cb.clone();
        program.register_native_fn("get_borrowed_map", move |_args: &[Value]| {
            Ok(NativeCallResult::Return(Value::Map(sm1.clone())))
        });
        let sm2 = shared_map_cb.clone();
        program.register_native_fn("borrow_pkg.get_borrowed_map", move |_args: &[Value]| {
            Ok(NativeCallResult::Return(Value::Map(sm2.clone())))
        });

        program.register_native_fn("get_nested_array", move |_args: &[Value]| {
            let inner1 = Rc::new(RefCell::new(vec![Value::Int(1), Value::Int(2)]));
            let inner2 = Rc::new(RefCell::new(vec![Value::Int(3)]));
            let outer = Rc::new(RefCell::new(vec![
                Value::Array(inner1),
                Value::Array(inner2),
            ]));
            Ok(NativeCallResult::Return(Value::Array(outer)))
        });
        program.register_native_fn("borrow_pkg.get_nested_array", move |_args: &[Value]| {
            let inner1 = Rc::new(RefCell::new(vec![Value::Int(1), Value::Int(2)]));
            let inner2 = Rc::new(RefCell::new(vec![Value::Int(3)]));
            let outer = Rc::new(RefCell::new(vec![
                Value::Array(inner1),
                Value::Array(inner2),
            ]));
            Ok(NativeCallResult::Return(Value::Array(outer)))
        });

        program.register_native_fn("get_nested_map", move |_args: &[Value]| {
            let mut inner_m = LustMap::default();
            inner_m.insert(ValueKey::from("inner_key"), Value::Int(777));
            let mut outer_m = LustMap::default();
            outer_m.insert(
                ValueKey::from("outer_key"),
                Value::Map(Rc::new(RefCell::new(inner_m))),
            );
            Ok(NativeCallResult::Return(Value::Map(Rc::new(RefCell::new(
                outer_m,
            )))))
        });
        program.register_native_fn("borrow_pkg.get_nested_map", move |_args: &[Value]| {
            let mut inner_m = LustMap::default();
            inner_m.insert(ValueKey::from("inner_key"), Value::Int(777));
            let mut outer_m = LustMap::default();
            outer_m.insert(
                ValueKey::from("outer_key"),
                Value::Map(Rc::new(RefCell::new(inner_m))),
            );
            Ok(NativeCallResult::Return(Value::Map(Rc::new(RefCell::new(
                outer_m,
            )))))
        });

        let limits = PreflightLimits::default();

        let target_arr = program
            .resolve_canonical_target("borrow_pkg.return_borrowed_array")
            .expect("resolve return_borrowed_array");
        let target_map = program
            .resolve_canonical_target("borrow_pkg.return_borrowed_map")
            .expect("resolve return_borrowed_map");
        let target_nested_arr = program
            .resolve_canonical_target("borrow_pkg.return_nested_array")
            .expect("resolve return_nested_array");
        let target_nested_map = program
            .resolve_canonical_target("borrow_pkg.return_nested_map")
            .expect("resolve return_nested_map");

        // -------------------------------------------------------------------
        // Preflight Zero-Effect Control
        // Invocations rejected during preflight produce zero host side-effects.
        // -------------------------------------------------------------------
        assert_eq!(host_counter.load(Ordering::SeqCst), 0);
        let err_preflight_arr = program
            .call_retained_target_spike(&target_arr, vec![Value::Int(99)], &limits)
            .expect_err("preflight arity mismatch must fail before VM dispatch");
        assert_eq!(err_preflight_arr.status, PreflightStatus::Type);
        assert_eq!(
            host_counter.load(Ordering::SeqCst),
            0,
            "Preflight rejection on array target must produce 0 side effects"
        );

        let err_preflight_map = program
            .call_retained_target_spike(&target_map, vec![Value::Int(99)], &limits)
            .expect_err("preflight arity mismatch must fail before VM dispatch");
        assert_eq!(err_preflight_map.status, PreflightStatus::Type);
        assert_eq!(
            host_counter.load(Ordering::SeqCst),
            0,
            "Preflight rejection on map target must produce 0 side effects"
        );

        // -------------------------------------------------------------------
        // Deterministic Array Borrow Remediation Test
        // Hold an outstanding RefMut across target call.
        // Target side effect counter must increment exactly once before post-call
        // BORROW error, and no panic escapes.
        // -------------------------------------------------------------------
        {
            let _held_arr_refmut = shared_array.borrow_mut();
            let err_borrow_arr = program
                .call_retained_target_spike(&target_arr, Vec::new(), &limits)
                .expect_err("post-call return validation must reject borrowed array");
            assert_eq!(err_borrow_arr.status, PreflightStatus::Borrow);
            assert!(err_borrow_arr.message.contains(
                "Post-call return validation error for 'borrow_pkg.return_borrowed_array'"
            ));
            assert!(err_borrow_arr.message.contains("Cannot borrow array"));
            assert_eq!(
                host_counter.load(Ordering::SeqCst),
                1,
                "Target side effect counter must increment exactly once before BORROW error"
            );
        } // _held_arr_refmut drops here, releasing borrow

        // Successful reuse of array target after releasing borrow
        let ok_arr = program
            .call_retained_target_spike(&target_arr, Vec::new(), &limits)
            .expect("array target call must succeed after borrow is released");
        assert_eq!(
            host_counter.load(Ordering::SeqCst),
            2,
            "Successful array call increments counter (1 -> 2)"
        );
        match ok_arr {
            Value::Array(arr) => {
                let borrowed = arr.borrow();
                assert_eq!(borrowed.len(), 2);
                assert_eq!(borrowed[0], Value::Int(10));
                assert_eq!(borrowed[1], Value::Int(20));
            }
            other => panic!("expected Value::Array, got {:?}", other),
        }

        // Valid nested Array success
        let ok_nested_arr = program
            .call_retained_target_spike(&target_nested_arr, Vec::new(), &limits)
            .expect("nested array return validation must succeed");
        assert_eq!(
            host_counter.load(Ordering::SeqCst),
            3,
            "Successful nested array call increments counter (2 -> 3)"
        );
        match ok_nested_arr {
            Value::Array(arr) => {
                let borrowed = arr.borrow();
                assert_eq!(borrowed.len(), 2);
            }
            other => panic!("expected Value::Array, got {:?}", other),
        }

        // -------------------------------------------------------------------
        // Deterministic Map Borrow Remediation Test
        // Hold an outstanding RefMut across target call.
        // Target side effect counter must increment exactly once before post-call
        // BORROW error, and no panic escapes.
        // -------------------------------------------------------------------
        {
            let _held_map_refmut = shared_map.borrow_mut();
            let err_borrow_map = program
                .call_retained_target_spike(&target_map, Vec::new(), &limits)
                .expect_err("post-call return validation must reject borrowed map");
            assert_eq!(err_borrow_map.status, PreflightStatus::Borrow);
            assert!(err_borrow_map.message.contains(
                "Post-call return validation error for 'borrow_pkg.return_borrowed_map'"
            ));
            assert!(err_borrow_map.message.contains("Cannot borrow map"));
            assert_eq!(
                host_counter.load(Ordering::SeqCst),
                4,
                "Target side effect counter must increment exactly once before BORROW error"
            );
        } // _held_map_refmut drops here, releasing borrow

        // Successful reuse of map target after releasing borrow
        let ok_map = program
            .call_retained_target_spike(&target_map, Vec::new(), &limits)
            .expect("map target call must succeed after borrow is released");
        assert_eq!(
            host_counter.load(Ordering::SeqCst),
            5,
            "Successful map call increments counter (4 -> 5)"
        );
        match ok_map {
            Value::Map(m) => {
                let borrowed = m.borrow();
                assert_eq!(borrowed.len(), 1);
                assert_eq!(borrowed.get(&ValueKey::from("k1")), Some(&Value::Int(100)));
            }
            other => panic!("expected Value::Map, got {:?}", other),
        }

        // Valid nested Map success
        let ok_nested_map = program
            .call_retained_target_spike(&target_nested_map, Vec::new(), &limits)
            .expect("nested map return validation must succeed");
        assert_eq!(
            host_counter.load(Ordering::SeqCst),
            6,
            "Successful nested map call increments counter (5 -> 6)"
        );
        match ok_nested_map {
            Value::Map(m) => {
                let borrowed = m.borrow();
                assert_eq!(borrowed.len(), 1);
                let inner = borrowed.get(&ValueKey::from("outer_key")).unwrap();
                match inner {
                    Value::Map(im) => {
                        assert_eq!(
                            im.borrow().get(&ValueKey::from("inner_key")),
                            Some(&Value::Int(777))
                        );
                    }
                    other => panic!("expected nested Value::Map, got {:?}", other),
                }
            }
            other => panic!("expected Value::Map, got {:?}", other),
        }

        // Preflight zero-effect control after all invocations
        let err_preflight_final = program
            .call_retained_target_spike(&target_nested_map, vec![Value::Int(1)], &limits)
            .expect_err("final preflight rejection must fail before VM dispatch");
        assert_eq!(err_preflight_final.status, PreflightStatus::Type);
        assert_eq!(
            host_counter.load(Ordering::SeqCst),
            6,
            "Final preflight rejection must produce 0 side effects"
        );
    }

    #[test]
    fn gate2_return_validation_cycle_remediation_and_acyclic_aliases() {
        use crate::bytecode::value::NativeCallResult;
        use crate::bytecode::{LustMap, ValueKey};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let host_counter = Arc::new(AtomicUsize::new(0));

        // Shared containers for returning
        // Self-cyclic array: [cyclic_arr]
        let cyclic_arr = Rc::new(RefCell::new(Vec::new()));
        cyclic_arr
            .borrow_mut()
            .push(Value::Array(Rc::clone(&cyclic_arr)));

        // Two-container cyclic array: A -> [B], B -> [A]
        let two_node_a = Rc::new(RefCell::new(Vec::new()));
        let two_node_b = Rc::new(RefCell::new(vec![Value::Array(Rc::clone(&two_node_a))]));
        two_node_a
            .borrow_mut()
            .push(Value::Array(Rc::clone(&two_node_b)));

        let acyclic_leaf = Rc::new(RefCell::new(vec![Value::Int(42)]));
        // Array containing two references to the same acyclic leaf (DAG / diamond alias)
        let acyclic_diamond_arr = Rc::new(RefCell::new(vec![
            Value::Array(Rc::clone(&acyclic_leaf)),
            Value::Array(Rc::clone(&acyclic_leaf)),
        ]));

        // Self-cyclic map
        let base_cyclic_map = LustMap::default();
        let cyclic_map = Rc::new(RefCell::new(base_cyclic_map));
        cyclic_map.borrow_mut().insert(
            ValueKey::from("self_ref"),
            Value::Map(Rc::clone(&cyclic_map)),
        );

        // Two-container cyclic map: M1 -> {"next": M2}, M2 -> {"back": M1}
        let two_map_1 = Rc::new(RefCell::new(LustMap::default()));
        let mut base_map_2 = LustMap::default();
        base_map_2.insert(ValueKey::from("back"), Value::Map(Rc::clone(&two_map_1)));
        let two_map_2 = Rc::new(RefCell::new(base_map_2));
        two_map_1
            .borrow_mut()
            .insert(ValueKey::from("next"), Value::Map(Rc::clone(&two_map_2)));

        // Acyclic shared map
        let mut base_acyclic_leaf = LustMap::default();
        base_acyclic_leaf.insert(ValueKey::from("answer"), Value::Int(42));
        let acyclic_leaf_map = Rc::new(RefCell::new(base_acyclic_leaf));

        let mut base_parent_map = LustMap::default();
        base_parent_map.insert(
            ValueKey::from("left"),
            Value::Map(Rc::clone(&acyclic_leaf_map)),
        );
        base_parent_map.insert(
            ValueKey::from("right"),
            Value::Map(Rc::clone(&acyclic_leaf_map)),
        );
        let acyclic_parent_map = Rc::new(RefCell::new(base_parent_map));

        let mut program = EmbeddedProgram::builder()
            .module(
                "cycle_pkg",
                r#"
                extern
                    function get_cyclic_array(): Array<int>
                    function get_two_node_cyclic_array(): Array<Array<int>>
                    function get_acyclic_diamond(): Array<Array<int>>
                    function get_cyclic_map(): Map<string, int>
                    function get_two_node_cyclic_map(): Map<string, Map<string, int>>
                    function get_acyclic_map(): Map<string, Map<string, int>>
                end

                function return_cyclic_array(): Array<int>
                    return get_cyclic_array()
                end

                function return_two_node_cyclic_array(): Array<Array<int>>
                    return get_two_node_cyclic_array()
                end

                function return_acyclic_diamond(): Array<Array<int>>
                    return get_acyclic_diamond()
                end

                function return_cyclic_map(): Map<string, int>
                    return get_cyclic_map()
                end

                function return_two_node_cyclic_map(): Map<string, Map<string, int>>
                    return get_two_node_cyclic_map()
                end

                function return_acyclic_map(): Map<string, Map<string, int>>
                    return get_acyclic_map()
                end
                "#,
            )
            .entry_module("cycle_pkg")
            .compile()
            .expect("compile cycle_pkg");

        let c1 = host_counter.clone();
        let ret_arr1 = cyclic_arr.clone();
        program.register_native_fn("get_cyclic_array", move |_args: &[Value]| {
            c1.fetch_add(1, Ordering::SeqCst);
            Ok(NativeCallResult::Return(Value::Array(ret_arr1.clone())))
        });
        let c1b = host_counter.clone();
        let ret_arr1b = cyclic_arr.clone();
        program.register_native_fn("cycle_pkg.get_cyclic_array", move |_args: &[Value]| {
            c1b.fetch_add(1, Ordering::SeqCst);
            Ok(NativeCallResult::Return(Value::Array(ret_arr1b.clone())))
        });

        let c2 = host_counter.clone();
        let ret_arr2 = acyclic_diamond_arr.clone();
        program.register_native_fn("get_acyclic_diamond", move |_args: &[Value]| {
            c2.fetch_add(1, Ordering::SeqCst);
            Ok(NativeCallResult::Return(Value::Array(ret_arr2.clone())))
        });
        let c2b = host_counter.clone();
        let ret_arr2b = acyclic_diamond_arr.clone();
        program.register_native_fn("cycle_pkg.get_acyclic_diamond", move |_args: &[Value]| {
            c2b.fetch_add(1, Ordering::SeqCst);
            Ok(NativeCallResult::Return(Value::Array(ret_arr2b.clone())))
        });

        let c2c = host_counter.clone();
        let ret_arr_two = two_node_a.clone();
        program.register_native_fn("get_two_node_cyclic_array", move |_args: &[Value]| {
            c2c.fetch_add(1, Ordering::SeqCst);
            Ok(NativeCallResult::Return(Value::Array(ret_arr_two.clone())))
        });
        let c2d = host_counter.clone();
        let ret_arr_two_b = two_node_a.clone();
        program.register_native_fn(
            "cycle_pkg.get_two_node_cyclic_array",
            move |_args: &[Value]| {
                c2d.fetch_add(1, Ordering::SeqCst);
                Ok(NativeCallResult::Return(Value::Array(
                    ret_arr_two_b.clone(),
                )))
            },
        );

        let c3 = host_counter.clone();
        let ret_map1 = cyclic_map.clone();
        program.register_native_fn("get_cyclic_map", move |_args: &[Value]| {
            c3.fetch_add(1, Ordering::SeqCst);
            Ok(NativeCallResult::Return(Value::Map(ret_map1.clone())))
        });
        let c3b = host_counter.clone();
        let ret_map1b = cyclic_map.clone();
        program.register_native_fn("cycle_pkg.get_cyclic_map", move |_args: &[Value]| {
            c3b.fetch_add(1, Ordering::SeqCst);
            Ok(NativeCallResult::Return(Value::Map(ret_map1b.clone())))
        });

        let c3c = host_counter.clone();
        let ret_map_two = two_map_1.clone();
        program.register_native_fn("get_two_node_cyclic_map", move |_args: &[Value]| {
            c3c.fetch_add(1, Ordering::SeqCst);
            Ok(NativeCallResult::Return(Value::Map(ret_map_two.clone())))
        });
        let c3d = host_counter.clone();
        let ret_map_two_b = two_map_1.clone();
        program.register_native_fn(
            "cycle_pkg.get_two_node_cyclic_map",
            move |_args: &[Value]| {
                c3d.fetch_add(1, Ordering::SeqCst);
                Ok(NativeCallResult::Return(Value::Map(ret_map_two_b.clone())))
            },
        );

        let c4 = host_counter.clone();
        let ret_map2 = acyclic_parent_map.clone();
        program.register_native_fn("get_acyclic_map", move |_args: &[Value]| {
            c4.fetch_add(1, Ordering::SeqCst);
            Ok(NativeCallResult::Return(Value::Map(ret_map2.clone())))
        });
        let c4b = host_counter.clone();
        let ret_map2b = acyclic_parent_map.clone();
        program.register_native_fn("cycle_pkg.get_acyclic_map", move |_args: &[Value]| {
            c4b.fetch_add(1, Ordering::SeqCst);
            Ok(NativeCallResult::Return(Value::Map(ret_map2b.clone())))
        });

        let limits = PreflightLimits {
            max_depth: 32,
            max_work: 100,
        };

        let target_cyclic_arr = program
            .resolve_canonical_target("cycle_pkg.return_cyclic_array")
            .expect("resolve return_cyclic_array");
        let target_two_node_arr = program
            .resolve_canonical_target("cycle_pkg.return_two_node_cyclic_array")
            .expect("resolve return_two_node_cyclic_array");
        let target_acyclic_arr = program
            .resolve_canonical_target("cycle_pkg.return_acyclic_diamond")
            .expect("resolve return_acyclic_diamond");
        let target_cyclic_map = program
            .resolve_canonical_target("cycle_pkg.return_cyclic_map")
            .expect("resolve return_cyclic_map");
        let target_two_node_map = program
            .resolve_canonical_target("cycle_pkg.return_two_node_cyclic_map")
            .expect("resolve return_two_node_cyclic_map");
        let target_acyclic_map = program
            .resolve_canonical_target("cycle_pkg.return_acyclic_map")
            .expect("resolve return_acyclic_map");

        // -------------------------------------------------------------------
        // Test 1: Cyclic Array Rejection
        // Must reject with PreflightStatus::Cycle before depth exhaustion.
        // Side effect must increment exactly once.
        // -------------------------------------------------------------------
        assert_eq!(host_counter.load(Ordering::SeqCst), 0);
        let err_cyclic_arr = program
            .call_retained_target_spike(&target_cyclic_arr, Vec::new(), &limits)
            .expect_err("cyclic array return validation must fail");
        println!("ACTUAL err_cyclic_arr: {:?}", err_cyclic_arr);
        assert_eq!(
            err_cyclic_arr.status,
            PreflightStatus::Cycle,
            "Must return CYCLE status for cyclic array"
        );
        assert!(
            err_cyclic_arr
                .message
                .contains("Cyclic container detected: array contains an active ancestor reference")
        );
        assert_eq!(
            host_counter.load(Ordering::SeqCst),
            1,
            "Target side-effect executes exactly once before cycle error"
        );

        // -------------------------------------------------------------------
        // Test 2: Cycle vs Limit Ordering Proof
        // With max_depth: 1, validating the root container at depth 0 succeeds,
        // and its child element at depth 1 is an active ancestor reference.
        // It must be detected as CYCLE, taking precedence over depth exhaustion
        // (which would trigger at depth > 1). Furthermore, even if max_work is 1,
        // encountering the active ancestor at depth 0 on subsequent call triggers
        // CYCLE before work budget is exhausted.
        // -------------------------------------------------------------------
        let strict_depth_limits = PreflightLimits {
            max_depth: 1,
            max_work: 50,
        };
        let err_cycle_ordering = program
            .call_retained_target_spike(&target_cyclic_arr, Vec::new(), &strict_depth_limits)
            .expect_err("cyclic array with strict depth must fail as CYCLE");
        assert_eq!(
            err_cycle_ordering.status,
            PreflightStatus::Cycle,
            "Cycle at depth 1 takes precedence over depth exhaustion"
        );
        assert_eq!(host_counter.load(Ordering::SeqCst), 2);

        // -------------------------------------------------------------------
        // Test 3: Acyclic Diamond / Shared Aliases MUST SUCCEED
        // Multiple references to the same acyclic container are traversed by value
        // and must NOT be rejected as cycles!
        // -------------------------------------------------------------------
        let ok_diamond = program
            .call_retained_target_spike(&target_acyclic_arr, Vec::new(), &limits)
            .expect("acyclic diamond array must succeed validation");
        assert_eq!(host_counter.load(Ordering::SeqCst), 3);
        match ok_diamond {
            Value::Array(arr) => {
                let borrowed = arr.borrow();
                assert_eq!(borrowed.len(), 2);
            }
            other => panic!("expected Value::Array, got {:?}", other),
        }

        // -------------------------------------------------------------------
        // Test 4: Cyclic Map Rejection
        // Must reject with PreflightStatus::Cycle. Side-effect executes once.
        // -------------------------------------------------------------------
        let err_cyclic_map = program
            .call_retained_target_spike(&target_cyclic_map, Vec::new(), &limits)
            .expect_err("cyclic map return validation must fail");
        assert_eq!(
            err_cyclic_map.status,
            PreflightStatus::Cycle,
            "Must return CYCLE status for cyclic map"
        );
        assert!(
            err_cyclic_map
                .message
                .contains("Cyclic container detected: map contains an active ancestor reference")
        );
        assert_eq!(host_counter.load(Ordering::SeqCst), 4);

        // -------------------------------------------------------------------
        // Test 5: Acyclic Shared Map MUST SUCCEED
        // -------------------------------------------------------------------
        let ok_map = program
            .call_retained_target_spike(&target_acyclic_map, Vec::new(), &limits)
            .expect("acyclic shared map must succeed validation");
        assert_eq!(host_counter.load(Ordering::SeqCst), 5);
        match ok_map {
            Value::Map(m) => {
                let borrowed = m.borrow();
                assert_eq!(borrowed.len(), 2);
            }
            other => panic!("expected Value::Map, got {:?}", other),
        }

        // -------------------------------------------------------------------
        // Test 6: Target reuse after cycle error
        // -------------------------------------------------------------------
        let err_cyclic_arr_again = program
            .call_retained_target_spike(&target_cyclic_arr, Vec::new(), &limits)
            .expect_err("second cyclic array call must also fail cleanly");
        assert_eq!(err_cyclic_arr_again.status, PreflightStatus::Cycle);
        assert_eq!(host_counter.load(Ordering::SeqCst), 6);

        // -------------------------------------------------------------------
        // Test 7: Multi-container (two-node) cyclic array A -> B -> A
        // Under declared return schema Array<Array<int>>, traversal visits A,
        // pushes A to ancestors, visits B, pushes B to ancestors, then encounters
        // back-edge to A. Must return PreflightStatus::Cycle.
        // -------------------------------------------------------------------
        let err_two_node_arr = program
            .call_retained_target_spike(&target_two_node_arr, Vec::new(), &limits)
            .expect_err("two-node cyclic array must fail as CYCLE");
        assert_eq!(
            err_two_node_arr.status,
            PreflightStatus::Cycle,
            "Two-node cyclic array returns CYCLE"
        );
        assert_eq!(host_counter.load(Ordering::SeqCst), 7);

        // -------------------------------------------------------------------
        // Test 8: Multi-container (two-node) cyclic map M1 -> M2 -> M1
        // Under declared return schema Map<string, Map<string, int>>, traversal visits
        // M1, pushes M1, visits M2, pushes M2, then encounters back-edge to M1.
        // Must return PreflightStatus::Cycle.
        // -------------------------------------------------------------------
        let err_two_node_map = program
            .call_retained_target_spike(&target_two_node_map, Vec::new(), &limits)
            .expect_err("two-node cyclic map must fail as CYCLE");
        assert_eq!(
            err_two_node_map.status,
            PreflightStatus::Cycle,
            "Two-node cyclic map returns CYCLE"
        );
        assert_eq!(host_counter.load(Ordering::SeqCst), 8);
    }
}
