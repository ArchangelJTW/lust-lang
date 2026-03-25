use super::async_runtime::{
    signal_pair, AsyncRegistry, AsyncTaskEntry, AsyncTaskQueue, AsyncTaskTarget, AsyncValueFuture,
    PendingAsyncTask,
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
use crate::bytecode::{Compiler, NativeCallResult, Value};
use crate::modules::{ModuleImports, ModuleLoader};
use crate::typechecker::{FunctionSignature, TypeChecker};
use crate::vm::{NativeExport, NativeExportParam, VM};
use crate::{LustConfig, LustError, Result};
use hashbrown::HashMap;
use std::cell::RefCell;
use std::future::Future;
use std::io;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::task::{Context, Poll};

pub struct EmbeddedBuilder {
    base_dir: PathBuf,
    modules: HashMap<String, String>,
    entry_module: Option<String>,
    config: LustConfig,
    extern_registry: ExternRegistry,
}

impl Default for EmbeddedBuilder {
    fn default() -> Self {
        Self {
            base_dir: PathBuf::from("__embedded__"),
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

    pub fn with_base_dir(self, base_dir: impl Into<PathBuf>) -> Self {
        self.set_base_dir(base_dir)
    }

    pub fn set_base_dir(mut self, base_dir: impl Into<PathBuf>) -> Self {
        self.base_dir = base_dir.into();
        self
    }

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
        let has_entry = self.modules.contains_key(&entry_module);
        if !has_entry {
            return Err(LustError::Unknown(format!(
                "Entry module '{}' was not provided via EmbeddedBuilder::module",
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
}

pub struct EmbeddedProgram {
    vm: VM,
    signatures: HashMap<String, FunctionSignature>,
    struct_defs: HashMap<String, StructDef>,
    enum_defs: HashMap<String, EnumDef>,
    entry_script: Option<String>,
    entry_module: String,
    async_registry: Rc<RefCell<AsyncRegistry>>,
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

    pub fn vm_mut(&mut self) -> &mut VM {
        &mut self.vm
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

    pub fn set_memory_budget_bytes(&mut self, limit_bytes: usize) {
        self.vm.set_memory_budget_bytes(limit_bytes);
    }

    pub fn set_memory_budget_kb(&mut self, limit_kb: u64) {
        self.vm.set_memory_budget_kb(limit_kb);
    }

    pub fn clear_memory_budget(&mut self) {
        self.vm.clear_memory_budget();
    }

    pub fn reset_memory_counter(&mut self) {
        self.vm.reset_memory_counter();
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
        name.rsplit(|c| c == '.' || c == ':').next().unwrap_or(name)
    }

    fn register_native_with_aliases<F>(&mut self, requested_name: &str, canonical: String, func: F)
    where
        F: Fn(&[Value]) -> std::result::Result<NativeCallResult, String> + 'static,
    {
        let native_fn: Rc<dyn Fn(&[Value]) -> std::result::Result<NativeCallResult, String>> =
            Rc::new(func);
        let value = Value::NativeFunction(native_fn);
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
        let coerced_values: Option<Rc<Vec<Value>>> = match &enum_variant.fields {
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

                Some(Rc::new(collected))
            }
        };
        Ok(EnumInstance::new(
            type_name.clone(),
            variant_name.clone(),
            Value::Enum {
                enum_name: type_name,
                variant: variant_name,
                values: coerced_values,
            },
        ))
    }

    pub fn register_native_fn<F>(&mut self, name: impl Into<String>, func: F)
    where
        F: Fn(&[Value]) -> std::result::Result<NativeCallResult, String> + 'static,
    {
        let native = Value::NativeFunction(Rc::new(func));
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
            let args: Vec<Value> = values.iter().cloned().collect();
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
        self.vm
            .record_exported_native(native_export_from_signature(&canonical, &signature));
        let handler = move |values: &[Value]| -> std::result::Result<NativeCallResult, String> {
            let args = Args::from_values(values)?;
            let result = func(args)?;
            Ok(NativeCallResult::Return(result.into_value()))
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
        let values = args.into_values();
        let value = self.vm.call(function_name, values)?;
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
            let mut completion: Option<(AsyncTaskTarget, std::result::Result<Value, String>)> =
                None;
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
                    completion = Some((entry.target, result));
                }
            }

            if let Some((target, outcome)) = completion {
                self.async_registry.borrow_mut().pending.remove(&id);
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
}

fn compile_in_memory(
    base_dir: PathBuf,
    entry_module: String,
    overrides: HashMap<PathBuf, String>,
    config: LustConfig,
    extern_registry: ExternRegistry,
) -> Result<EmbeddedProgram> {
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
        if module.path != program_entry_module {
            if let Some(ref init) = module.init_function {
                let init_name = module
                    .imports
                    .function_aliases
                    .get(init)
                    .cloned()
                    .unwrap_or_else(|| init.clone());
                init_funcs.push((module.path.clone(), init_name));
            }
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
            });
    }

    let mut vm = VM::with_config(&config);
    vm.load_functions(functions);
    vm.register_structs(&struct_defs);
    extern_registry.register_type_stubs(&mut vm);
    for (type_name, trait_name) in trait_impls {
        vm.register_trait_impl(type_name, trait_name);
    }

    for (module_path, init) in init_funcs {
        let value = vm.call(&init, Vec::new())?;
        vm.set_global(module_path, value);
    }

    Ok(EmbeddedProgram {
        vm,
        signatures,
        struct_defs,
        enum_defs,
        entry_script,
        entry_module: program_entry_module,
        async_registry: Rc::new(RefCell::new(AsyncRegistry::new())),
    })
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

fn native_export_from_signature(canonical_name: &str, signature: &FunctionSignature) -> NativeExport {
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
