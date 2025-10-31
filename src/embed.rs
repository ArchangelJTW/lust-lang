use crate::ast::{EnumDef, FieldOwnership, Item, ItemKind, Span, StructDef, Type, TypeKind};
use crate::bytecode::{Compiler, NativeCallResult, TaskHandle, Value};
use crate::modules::{ModuleImports, ModuleLoader};
use crate::number::{LustFloat, LustInt};
use crate::typechecker::{FunctionSignature, TypeChecker};
use crate::vm::VM;
use crate::{LustConfig, LustError, Result};
use std::cell::RefCell;
use hashbrown::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

type AsyncValueFuture = Pin<Box<dyn Future<Output = std::result::Result<Value, String>>>>;

struct AsyncRegistry {
    pending: HashMap<u64, AsyncTaskEntry>,
}

impl AsyncRegistry {
    fn new() -> Self {
        Self {
            pending: HashMap::new(),
        }
    }

    fn register(
        &mut self,
        handle: TaskHandle,
        future: AsyncValueFuture,
    ) -> std::result::Result<(), String> {
        let key = handle.id();
        if self.pending.contains_key(&key) {
            return Err(format!(
                "Task {} already has a pending async native call",
                key
            ));
        }

        self.pending
            .insert(key, AsyncTaskEntry::new(handle, future));
        Ok(())
    }

    fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

struct AsyncTaskEntry {
    handle: TaskHandle,
    future: AsyncValueFuture,
    wake_flag: Arc<WakeFlag>,
    immediate_poll: bool,
}

impl AsyncTaskEntry {
    fn new(handle: TaskHandle, future: AsyncValueFuture) -> Self {
        Self {
            handle,
            future,
            wake_flag: Arc::new(WakeFlag::new()),
            immediate_poll: true,
        }
    }

    fn take_should_poll(&mut self) -> bool {
        if self.immediate_poll {
            self.immediate_poll = false;
            true
        } else {
            self.wake_flag.take()
        }
    }

    fn make_waker(&self) -> Waker {
        make_async_waker(&self.wake_flag)
    }
}

struct WakeFlag {
    pending: AtomicBool,
}

impl WakeFlag {
    fn new() -> Self {
        Self {
            pending: AtomicBool::new(true),
        }
    }

    fn take(&self) -> bool {
        self.pending.swap(false, Ordering::SeqCst)
    }

    fn wake(&self) {
        self.pending.store(true, Ordering::SeqCst);
    }
}

fn make_async_waker(flag: &Arc<WakeFlag>) -> Waker {
    unsafe {
        Waker::from_raw(RawWaker::new(
            Arc::into_raw(flag.clone()) as *const (),
            &ASYNC_WAKER_VTABLE,
        ))
    }
}

unsafe fn async_waker_clone(ptr: *const ()) -> RawWaker {
    let arc = Arc::<WakeFlag>::from_raw(ptr as *const WakeFlag);
    let cloned = arc.clone();
    std::mem::forget(arc);
    RawWaker::new(Arc::into_raw(cloned) as *const (), &ASYNC_WAKER_VTABLE)
}

unsafe fn async_waker_wake(ptr: *const ()) {
    let arc = Arc::<WakeFlag>::from_raw(ptr as *const WakeFlag);
    arc.wake();
}

unsafe fn async_waker_wake_by_ref(ptr: *const ()) {
    let arc = Arc::<WakeFlag>::from_raw(ptr as *const WakeFlag);
    arc.wake();
    std::mem::forget(arc);
}

unsafe fn async_waker_drop(ptr: *const ()) {
    let _ = Arc::<WakeFlag>::from_raw(ptr as *const WakeFlag);
}

static ASYNC_WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(
    async_waker_clone,
    async_waker_wake,
    async_waker_wake_by_ref,
    async_waker_drop,
);
pub struct EmbeddedBuilder {
    base_dir: PathBuf,
    modules: HashMap<String, String>,
    entry_module: Option<String>,
    config: LustConfig,
}

impl Default for EmbeddedBuilder {
    fn default() -> Self {
        Self {
            base_dir: PathBuf::from("__embedded__"),
            modules: HashMap::new(),
            entry_module: None,
            config: LustConfig::default(),
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

    pub fn with_config(mut self, config: LustConfig) -> Self {
        self.config = config;
        self
    }

    pub fn set_config(mut self, config: LustConfig) -> Self {
        self.config = config;
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
        compile_in_memory(self.base_dir, entry_module, overrides, self.config)
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

impl EmbeddedProgram {
    pub fn builder() -> EmbeddedBuilder {
        EmbeddedBuilder::default()
    }

    pub fn vm_mut(&mut self) -> &mut VM {
        &mut self.vm
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

    fn register_native_with_aliases<F>(
        &mut self,
        requested_name: &str,
        canonical: String,
        func: F,
    ) where
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

    pub fn struct_instance<I, K, V>(
        &self,
        type_name: impl Into<String>,
        fields: I,
    ) -> Result<StructInstance>
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: IntoTypedValue,
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
            .map(|(name, value)| (name.into(), value.into_typed_value()))
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
            let matches_declared = typed_value.matches(&field.ty);
            let matches_ref_inner = matches!(field.ownership, FieldOwnership::Weak)
                && field
                    .weak_target
                    .as_ref()
                    .map(|inner| typed_value.matches(inner))
                    .unwrap_or(false);
            if !(matches_declared || matches_ref_inner) {
                return Err(LustError::TypeError {
                    message: format!(
                        "Struct '{}' field '{}' expects Lust type '{}' but Rust provided '{}'",
                        type_name,
                        field.name,
                        field.ty,
                        typed_value.description()
                    ),
                });
            }

            ordered_fields.push((Rc::new(field.name.clone()), typed_value.into_value()));
        }

        if !provided.is_empty() {
            let extra = provided.keys().cloned().collect::<Vec<_>>().join(", ");
            return Err(LustError::TypeError {
                message: format!(
                    "Struct '{}' received unknown field(s): {}",
                    type_name, extra
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

    pub fn register_async_native<F, Fut>(
        &mut self,
        name: impl Into<String>,
        func: F,
    ) -> Result<()>
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
                registry.borrow_mut().register(handle, future)?;
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
                registry.borrow_mut().register(handle, future)?;
                Ok(NativeCallResult::Yield(Value::Nil))
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

        let mut completions: Vec<(TaskHandle, std::result::Result<Value, String>)> = Vec::new();
        for id in pending_ids {
            let handle = TaskHandle(id);
            if self.vm.get_task_instance(handle).is_err() {
                self.async_registry.borrow_mut().pending.remove(&id);
                continue;
            }

            let maybe_outcome = {
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
                match entry.future.as_mut().poll(&mut cx) {
                    Poll::Ready(result) => {
                        let handle = entry.handle;
                        registry.pending.remove(&id);
                        Some((handle, result))
                    }

                    Poll::Pending => None,
                }
            };

            if let Some(outcome) = maybe_outcome {
                completions.push(outcome);
            }
        }

        for (handle, outcome) in completions {
            match outcome {
                Ok(value) => {
                    self.vm.resume_task_handle(handle, Some(value))?;
                }

                Err(message) => {
                    self.vm.fail_task_handle(
                        handle,
                        LustError::RuntimeError { message },
                    )?;
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
) -> Result<EmbeddedProgram> {
    let mut loader = ModuleLoader::new(base_dir.clone());
    loader.set_source_overrides(overrides);
    let entry_path = module_path_to_file(&base_dir, &entry_module);
    let entry_path_str = entry_path
        .to_str()
        .ok_or_else(|| LustError::Unknown("Entry path contained invalid UTF-8".into()))?
        .to_string();
    let program = loader.load_program_from_entry(&entry_path_str)?;
    let mut imports_map: HashMap<String, ModuleImports> = HashMap::new();
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
    typechecker.check_program(&program.modules)?;
    let struct_defs = typechecker.struct_definitions();
    let enum_defs = typechecker.enum_definitions();
    let mut signatures = typechecker.function_signatures();
    let mut compiler = Compiler::new();
    compiler.configure_stdlib(&config);
    compiler.set_imports_by_module(imports_map);
    compiler.set_entry_module(program.entry_module.clone());
    let functions = compiler.compile_module(&wrapped_items)?;
    let trait_impls = compiler.get_trait_impls().to_vec();
    let mut init_funcs = Vec::new();
    for module in &program.modules {
        if module.path != program.entry_module {
            if let Some(init) = &module.init_function {
                init_funcs.push(init.clone());
            }
        }
    }

    let function_names: Vec<String> = functions.iter().map(|f| f.name.clone()).collect();
    let entry_script = function_names
        .iter()
        .find(|name| name.as_str() == "__script")
        .cloned();
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
    for (type_name, trait_name) in trait_impls {
        vm.register_trait_impl(type_name, trait_name);
    }

    for init in init_funcs {
        vm.call(&init, Vec::new())?;
    }

    Ok(EmbeddedProgram {
        vm,
        signatures,
        struct_defs,
        enum_defs,
        entry_script,
        entry_module: program.entry_module,
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

fn normalize_global_name(name: &str) -> String {
    if name.contains("::") {
        name.to_string()
    } else if let Some((module, identifier)) = name.rsplit_once('.') {
        format!("{}::{}", module, identifier)
    } else {
        name.to_string()
    }
}

fn ensure_return_type<R: FromLustValue>(function_name: &str, ty: &Type) -> Result<()> {
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

pub struct TypedValue {
    value: Value,
    matcher: Box<dyn Fn(&Value, &Type) -> bool>,
    description: &'static str,
}

impl TypedValue {
    fn new<F>(value: Value, matcher: F, description: &'static str) -> Self
    where
        F: Fn(&Value, &Type) -> bool + 'static,
    {
        Self {
            value,
            matcher: Box::new(matcher),
            description,
        }
    }

    fn matches(&self, ty: &Type) -> bool {
        match &ty.kind {
            TypeKind::Union(types) => types.iter().any(|alt| (self.matcher)(&self.value, alt)),
            _ => (self.matcher)(&self.value, ty),
        }
    }

    fn description(&self) -> &'static str {
        self.description
    }

    fn into_value(self) -> Value {
        self.value
    }
}

#[derive(Clone)]
pub struct StructInstance {
    type_name: String,
    value: Value,
}

impl StructInstance {
    fn new(type_name: String, value: Value) -> Self {
        debug_assert!(matches!(value, Value::Struct { .. }));
        Self { type_name, value }
    }

    pub fn type_name(&self) -> &str {
        &self.type_name
    }

    pub fn field<T: FromLustValue>(&self, field: &str) -> Result<T> {
        match &self.value {
            Value::Struct { layout, fields, .. } => {
                let index = layout
                    .index_of_str(field)
                    .ok_or_else(|| LustError::RuntimeError {
                        message: format!(
                            "Struct '{}' has no field named '{}'",
                            self.type_name, field
                        ),
                    })?;
                let stored =
                    fields
                        .borrow()
                        .get(index)
                        .cloned()
                        .ok_or_else(|| LustError::RuntimeError {
                            message: format!(
                                "Struct '{}' field '{}' is unavailable",
                                self.type_name, field
                            ),
                        })?;
                let materialized = layout.materialize_field_value(index, stored);
                T::from_value(materialized)
            }

            _ => Err(LustError::RuntimeError {
                message: "StructInstance does not contain a struct value".to_string(),
            }),
        }
    }

    pub fn as_value(&self) -> &Value {
        &self.value
    }
}

#[derive(Clone)]
pub struct EnumInstance {
    type_name: String,
    variant: String,
    value: Value,
}

impl EnumInstance {
    fn new(type_name: String, variant: String, value: Value) -> Self {
        debug_assert!(matches!(value, Value::Enum { .. }));
        Self {
            type_name,
            variant,
            value,
        }
    }

    pub fn type_name(&self) -> &str {
        &self.type_name
    }

    pub fn variant(&self) -> &str {
        &self.variant
    }

    pub fn payload_len(&self) -> usize {
        match &self.value {
            Value::Enum { values, .. } => values.as_ref().map(|v| v.len()).unwrap_or(0),
            _ => 0,
        }
    }

    pub fn payload<T: FromLustValue>(&self, index: usize) -> Result<T> {
        match &self.value {
            Value::Enum { values, .. } => {
                let values = values.as_ref().ok_or_else(|| LustError::RuntimeError {
                    message: format!(
                        "Enum variant '{}.{}' carries no payload",
                        self.type_name, self.variant
                    ),
                })?;
                let stored = values
                    .get(index)
                    .cloned()
                    .ok_or_else(|| LustError::RuntimeError {
                        message: format!(
                            "Enum variant '{}.{}' payload index {} is out of bounds",
                            self.type_name, self.variant, index
                        ),
                    })?;
                T::from_value(stored)
            }

            _ => Err(LustError::RuntimeError {
                message: "EnumInstance does not contain an enum value".to_string(),
            }),
        }
    }

    pub fn as_value(&self) -> &Value {
        &self.value
    }
}

pub trait IntoTypedValue {
    fn into_typed_value(self) -> TypedValue;
}

impl IntoTypedValue for Value {
    fn into_typed_value(self) -> TypedValue {
        TypedValue::new(self, |_value, _ty| true, "Value")
    }
}

impl IntoTypedValue for StructInstance {
    fn into_typed_value(self) -> TypedValue {
        let StructInstance {
            type_name: _,
            value,
        } = self;
        TypedValue::new(value, |v, ty| matches_lust_struct(v, ty), "struct")
    }
}

impl IntoTypedValue for EnumInstance {
    fn into_typed_value(self) -> TypedValue {
        let EnumInstance {
            type_name: _,
            variant: _,
            value,
        } = self;
        TypedValue::new(value, |v, ty| matches_lust_enum(v, ty), "enum")
    }
}

macro_rules! impl_into_typed_for_primitive {
    ($ty:ty, $desc:expr, $matcher:expr) => {
        impl IntoTypedValue for $ty {
            fn into_typed_value(self) -> TypedValue {
                let value = self.into_value();
                TypedValue::new(value, $matcher, $desc)
            }
        }
    };
}

impl_into_typed_for_primitive!(LustInt, "int", |_, ty: &Type| match &ty.kind {
    TypeKind::Int | TypeKind::Unknown => true,
    TypeKind::Union(types) => types
        .iter()
        .any(|alt| matches!(&alt.kind, TypeKind::Int | TypeKind::Unknown)),
    _ => false,
});
impl_into_typed_for_primitive!(LustFloat, "float", |_, ty: &Type| match &ty.kind {
    TypeKind::Float | TypeKind::Unknown => true,
    TypeKind::Union(types) => types
        .iter()
        .any(|alt| matches!(&alt.kind, TypeKind::Float | TypeKind::Unknown)),
    _ => false,
});
impl_into_typed_for_primitive!(bool, "bool", |_, ty: &Type| match &ty.kind {
    TypeKind::Bool | TypeKind::Unknown => true,
    TypeKind::Union(types) => types
        .iter()
        .any(|alt| matches!(&alt.kind, TypeKind::Bool | TypeKind::Unknown)),
    _ => false,
});
impl IntoTypedValue for String {
    fn into_typed_value(self) -> TypedValue {
        let value = self.into_value();
        TypedValue::new(value, string_matcher, "string")
    }
}

impl<'a> IntoTypedValue for &'a str {
    fn into_typed_value(self) -> TypedValue {
        let value = self.into_value();
        TypedValue::new(value, string_matcher, "string")
    }
}

impl<'a> IntoTypedValue for &'a String {
    fn into_typed_value(self) -> TypedValue {
        let value = self.into_value();
        TypedValue::new(value, string_matcher, "string")
    }
}

impl IntoTypedValue for () {
    fn into_typed_value(self) -> TypedValue {
        TypedValue::new(
            Value::Nil,
            |_, ty| matches!(ty.kind, TypeKind::Unit | TypeKind::Unknown),
            "unit",
        )
    }
}

impl<T> IntoTypedValue for Vec<T>
where
    T: IntoLustValue,
{
    fn into_typed_value(self) -> TypedValue {
        let values = self.into_iter().map(|item| item.into_value()).collect();
        TypedValue::new(
            Value::array(values),
            |_, ty| matches_array_type(ty, &T::matches_lust_type),
            "array",
        )
    }
}

fn string_matcher(_: &Value, ty: &Type) -> bool {
    match &ty.kind {
        TypeKind::String | TypeKind::Unknown => true,
        TypeKind::Union(types) => types
            .iter()
            .any(|alt| matches!(&alt.kind, TypeKind::String | TypeKind::Unknown)),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};
    use std::task::{Context, Poll, Waker};

    #[derive(Default)]
    struct ManualAsyncState {
        result: Mutex<Option<std::result::Result<LustInt, String>>>,
        waker: Mutex<Option<Waker>>,
    }

    impl ManualAsyncState {
        fn new() -> Arc<Self> {
            Arc::new(Self::default())
        }

        fn future(self: &Arc<Self>) -> ManualFuture {
            ManualFuture {
                state: Arc::clone(self),
            }
        }

        fn complete_ok(&self, value: LustInt) {
            self.complete(Ok(value));
        }

        fn complete_err(&self, message: impl Into<String>) {
            self.complete(Err(message.into()));
        }

        fn complete(&self, value: std::result::Result<LustInt, String>) {
            {
                let mut slot = self.result.lock().unwrap();
                *slot = Some(value);
            }

            if let Some(waker) = self.waker.lock().unwrap().take() {
                waker.wake();
            }
        }
    }

    struct ManualFuture {
        state: Arc<ManualAsyncState>,
    }

    impl Future for ManualFuture {
        type Output = std::result::Result<LustInt, String>;

        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            {
                let mut slot = self.state.result.lock().unwrap();
                if let Some(result) = slot.take() {
                    return Poll::Ready(result);
                }
            }

            let mut waker_slot = self.state.waker.lock().unwrap();
            *waker_slot = Some(cx.waker().clone());
            Poll::Pending
        }
    }

    fn build_program(source: &str) -> EmbeddedProgram {
        EmbeddedProgram::builder()
            .module("main", source)
            .entry_module("main")
            .compile()
            .expect("compile embedded program")
    }

    #[test]
    fn async_native_resumes_task_on_completion() {
        let source = r#"
            extern {
                function fetch_value(): int
            }

            function compute(): int
                return fetch_value()
            end
        "#;

        let mut program = build_program(source);

        let state = ManualAsyncState::new();
        let register_state = Arc::clone(&state);
        program
            .register_async_typed_native::<(), LustInt, _, _>("fetch_value", move |_| {
                register_state.future()
            })
            .expect("register async native");

        let handle = {
            let vm = program.vm_mut();
            let compute_fn = vm
                .function_value("main.compute")
                .expect("compute function");
            vm.spawn_task_value(compute_fn, Vec::new())
                .expect("spawn task")
        };

        assert!(program.has_pending_async_tasks());
        program.poll_async_tasks().expect("initial poll");
        assert!(program.has_pending_async_tasks());

        state.complete_ok(123);
        program
            .poll_async_tasks()
            .expect("resume after completion");

        {
            let vm = program.vm_mut();
            let task = vm.get_task_instance(handle).expect("task exists");
            let result = task
                .last_result
                .as_ref()
                .and_then(|value| value.as_int())
                .expect("task produced result");
            assert_eq!(result, 123);
            assert!(task.error.is_none());
        }

        assert!(!program.has_pending_async_tasks());
    }

    #[test]
    fn async_native_failure_marks_task_failed() {
        let source = r#"
            extern {
                function fetch_value(): int
            }

            function compute(): int
                return fetch_value()
            end
        "#;

        let mut program = build_program(source);

        let state = ManualAsyncState::new();
        let register_state = Arc::clone(&state);
        program
            .register_async_typed_native::<(), LustInt, _, _>("fetch_value", move |_| {
                register_state.future()
            })
            .expect("register async native");

        let handle = {
            let vm = program.vm_mut();
            let compute_fn = vm
                .function_value("main.compute")
                .expect("compute function");
            vm.spawn_task_value(compute_fn, Vec::new())
                .expect("spawn task")
        };

        program.poll_async_tasks().expect("initial poll");
        state.complete_err("boom");
        let err = program
            .poll_async_tasks()
            .expect_err("poll should propagate failure");
        match err {
            LustError::RuntimeError { message } => assert_eq!(message, "boom"),
            other => panic!("unexpected error: {other:?}"),
        }

        {
            let vm = program.vm_mut();
            let task = vm.get_task_instance(handle).expect("task exists");
            assert!(task.last_result.is_none());
            let error_message = task
                .error
                .as_ref()
                .map(|e| e.to_string())
                .expect("task should record error");
            assert!(error_message.contains("boom"));
        }

        assert!(!program.has_pending_async_tasks());
    }
}

fn matches_lust_struct(value: &Value, ty: &Type) -> bool {
    match (value, &ty.kind) {
        (Value::Struct { name, .. }, TypeKind::Named(expected)) => name == expected,
        (Value::Struct { name, .. }, TypeKind::GenericInstance { name: expected, .. }) => {
            name == expected
        }

        (value, TypeKind::Union(types)) => types.iter().any(|alt| matches_lust_struct(value, alt)),
        (_, TypeKind::Unknown) => true,
        _ => false,
    }
}

fn matches_lust_enum(value: &Value, ty: &Type) -> bool {
    match (value, &ty.kind) {
        (Value::Enum { enum_name, .. }, TypeKind::Named(expected)) => enum_name == expected,
        (Value::Enum { enum_name, .. }, TypeKind::GenericInstance { name: expected, .. }) => {
            enum_name == expected
        }

        (value, TypeKind::Union(types)) => types.iter().any(|alt| matches_lust_enum(value, alt)),
        (_, TypeKind::Unknown) => true,
        _ => false,
    }
}

fn matches_array_type<F>(ty: &Type, matcher: &F) -> bool
where
    F: Fn(&Type) -> bool,
{
    match &ty.kind {
        TypeKind::Array(inner) => matcher(inner),
        TypeKind::Unknown => true,
        TypeKind::Union(types) => types.iter().any(|alt| matches_array_type(alt, matcher)),
        _ => false,
    }
}

pub trait FromLustArgs: Sized {
    fn from_values(values: &[Value]) -> std::result::Result<Self, String>;
    fn matches_signature(params: &[Type]) -> bool;
}

macro_rules! impl_from_lust_args_tuple {
    ($( $name:ident ),+) => {
        impl<$($name),+> FromLustArgs for ($($name,)+)
        where
            $($name: FromLustValue,)+
        {
            fn from_values(values: &[Value]) -> std::result::Result<Self, String> {
                let expected = count_idents!($($name),+);
                if values.len() != expected {
                    return Err(format!(
                        "Native function expected {} argument(s) but received {}",
                        expected,
                        values.len()
                    ));
                }

                let mut idx = 0;
                let result = (
                    $(
                        {
                            let value = $name::from_value(values[idx].clone()).map_err(|e| e.to_string())?;
                            idx += 1;
                            value
                        },
                    )+
                );
                let _ = idx;
                Ok(result)
            }

            fn matches_signature(params: &[Type]) -> bool {
                let expected = count_idents!($($name),+);
                params.len() == expected && {
                    let mut idx = 0;
                    let mut ok = true;
                    $(
                        if ok && !$name::matches_lust_type(&params[idx]) {
                            ok = false;
                        }

                        idx += 1;
                    )+
                    let _ = idx;
                    ok
                }

            }

        }

    };
}

macro_rules! count_idents {
    ($($name:ident),*) => {
        <[()]>::len(&[$(count_idents!(@sub $name)),*])
    };
    (@sub $name:ident) => { () };
}

impl_from_lust_args_tuple!(A);
impl_from_lust_args_tuple!(A, B);
impl_from_lust_args_tuple!(A, B, C);
impl_from_lust_args_tuple!(A, B, C, D);
impl_from_lust_args_tuple!(A, B, C, D, E);
impl<T> FromLustArgs for T
where
    T: FromLustValue,
{
    fn from_values(values: &[Value]) -> std::result::Result<Self, String> {
        match values.len() {
            0 => T::from_value(Value::Nil).map_err(|e| e.to_string()),
            1 => T::from_value(values[0].clone()).map_err(|e| e.to_string()),
            count => Err(format!(
                "Native function expected 1 argument but received {}",
                count
            )),
        }
    }

    fn matches_signature(params: &[Type]) -> bool {
        if params.is_empty() {
            let unit = Type::new(TypeKind::Unit, Span::new(0, 0, 0, 0));
            return T::matches_lust_type(&unit);
        }

        params.len() == 1 && T::matches_lust_type(&params[0])
    }
}

pub trait IntoLustValue: Sized {
    fn into_value(self) -> Value;
    fn matches_lust_type(ty: &Type) -> bool;
    fn type_description() -> &'static str;
}

pub trait FromLustValue: Sized {
    fn from_value(value: Value) -> Result<Self>;
    fn matches_lust_type(ty: &Type) -> bool;
    fn type_description() -> &'static str;
}

pub trait FunctionArgs {
    fn into_values(self) -> Vec<Value>;
    fn validate_signature(function_name: &str, params: &[Type]) -> Result<()>;
}

impl IntoLustValue for Value {
    fn into_value(self) -> Value {
        self
    }

    fn matches_lust_type(_: &Type) -> bool {
        true
    }

    fn type_description() -> &'static str {
        "Value"
    }
}

impl FromLustValue for Value {
    fn from_value(value: Value) -> Result<Self> {
        Ok(value)
    }

    fn matches_lust_type(_: &Type) -> bool {
        true
    }

    fn type_description() -> &'static str {
        "Value"
    }
}

impl IntoLustValue for LustInt {
    fn into_value(self) -> Value {
        Value::Int(self)
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches!(ty.kind, TypeKind::Int | TypeKind::Unknown)
    }

    fn type_description() -> &'static str {
        "int"
    }
}

impl FromLustValue for LustInt {
    fn from_value(value: Value) -> Result<Self> {
        match value {
            Value::Int(v) => Ok(v),
            other => Err(LustError::RuntimeError {
                message: format!("Expected Lust value 'int' but received '{:?}'", other),
            }),
        }
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches!(ty.kind, TypeKind::Int | TypeKind::Unknown)
    }

    fn type_description() -> &'static str {
        "int"
    }
}

impl IntoLustValue for LustFloat {
    fn into_value(self) -> Value {
        Value::Float(self)
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches!(ty.kind, TypeKind::Float | TypeKind::Unknown)
    }

    fn type_description() -> &'static str {
        "float"
    }
}

impl FromLustValue for LustFloat {
    fn from_value(value: Value) -> Result<Self> {
        match value {
            Value::Float(v) => Ok(v),
            other => Err(LustError::RuntimeError {
                message: format!("Expected Lust value 'float' but received '{:?}'", other),
            }),
        }
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches!(ty.kind, TypeKind::Float | TypeKind::Unknown)
    }

    fn type_description() -> &'static str {
        "float"
    }
}

impl IntoLustValue for bool {
    fn into_value(self) -> Value {
        Value::Bool(self)
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches!(ty.kind, TypeKind::Bool | TypeKind::Unknown)
    }

    fn type_description() -> &'static str {
        "bool"
    }
}

impl FromLustValue for bool {
    fn from_value(value: Value) -> Result<Self> {
        match value {
            Value::Bool(b) => Ok(b),
            other => Err(LustError::RuntimeError {
                message: format!("Expected Lust value 'bool' but received '{:?}'", other),
            }),
        }
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches!(ty.kind, TypeKind::Bool | TypeKind::Unknown)
    }

    fn type_description() -> &'static str {
        "bool"
    }
}

impl IntoLustValue for String {
    fn into_value(self) -> Value {
        Value::String(Rc::new(self))
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches!(ty.kind, TypeKind::String | TypeKind::Unknown)
    }

    fn type_description() -> &'static str {
        "string"
    }
}

impl IntoLustValue for StructInstance {
    fn into_value(self) -> Value {
        self.value
    }

    fn matches_lust_type(ty: &Type) -> bool {
        match &ty.kind {
            TypeKind::Unknown | TypeKind::Named(_) | TypeKind::GenericInstance { .. } => true,
            TypeKind::Union(types) => types
                .iter()
                .any(|alt| <Self as IntoLustValue>::matches_lust_type(alt)),
            _ => false,
        }
    }

    fn type_description() -> &'static str {
        "struct"
    }
}

impl FromLustValue for StructInstance {
    fn from_value(value: Value) -> Result<Self> {
        match &value {
            Value::Struct { name, .. } => Ok(StructInstance {
                type_name: name.clone(),
                value,
            }),
            other => Err(LustError::RuntimeError {
                message: format!("Expected Lust value 'struct' but received '{:?}'", other),
            }),
        }
    }

    fn matches_lust_type(ty: &Type) -> bool {
        match &ty.kind {
            TypeKind::Unknown | TypeKind::Named(_) | TypeKind::GenericInstance { .. } => true,
            TypeKind::Union(types) => types
                .iter()
                .any(|alt| <Self as FromLustValue>::matches_lust_type(alt)),
            _ => false,
        }
    }

    fn type_description() -> &'static str {
        "struct"
    }
}

impl IntoLustValue for EnumInstance {
    fn into_value(self) -> Value {
        self.value
    }

    fn matches_lust_type(ty: &Type) -> bool {
        match &ty.kind {
            TypeKind::Unknown | TypeKind::Named(_) | TypeKind::GenericInstance { .. } => true,
            TypeKind::Union(types) => types
                .iter()
                .any(|alt| <Self as IntoLustValue>::matches_lust_type(alt)),
            _ => false,
        }
    }

    fn type_description() -> &'static str {
        "enum"
    }
}

impl FromLustValue for EnumInstance {
    fn from_value(value: Value) -> Result<Self> {
        match &value {
            Value::Enum {
                enum_name, variant, ..
            } => Ok(EnumInstance {
                type_name: enum_name.clone(),
                variant: variant.clone(),
                value,
            }),
            other => Err(LustError::RuntimeError {
                message: format!("Expected Lust value 'enum' but received '{:?}'", other),
            }),
        }
    }

    fn matches_lust_type(ty: &Type) -> bool {
        match &ty.kind {
            TypeKind::Unknown | TypeKind::Named(_) | TypeKind::GenericInstance { .. } => true,
            TypeKind::Union(types) => types
                .iter()
                .any(|alt| <Self as FromLustValue>::matches_lust_type(alt)),
            _ => false,
        }
    }

    fn type_description() -> &'static str {
        "enum"
    }
}

impl<T> IntoLustValue for Vec<T>
where
    T: IntoLustValue,
{
    fn into_value(self) -> Value {
        let values = self.into_iter().map(|item| item.into_value()).collect();
        Value::array(values)
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches_array_type(ty, &T::matches_lust_type)
    }

    fn type_description() -> &'static str {
        "array"
    }
}

impl<T> FromLustValue for Vec<T>
where
    T: FromLustValue,
{
    fn from_value(value: Value) -> Result<Self> {
        match value {
            Value::Array(items) => {
                let borrowed = items.borrow();
                let mut result = Vec::with_capacity(borrowed.len());
                for item in borrowed.iter() {
                    result.push(T::from_value(item.clone())?);
                }

                Ok(result)
            }

            other => Err(LustError::RuntimeError {
                message: format!("Expected Lust value 'array' but received '{:?}'", other),
            }),
        }
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches_array_type(ty, &T::matches_lust_type)
    }

    fn type_description() -> &'static str {
        "array"
    }
}

impl<'a> IntoLustValue for &'a str {
    fn into_value(self) -> Value {
        Value::String(Rc::new(self.to_owned()))
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches!(ty.kind, TypeKind::String | TypeKind::Unknown)
    }

    fn type_description() -> &'static str {
        "string"
    }
}

impl<'a> IntoLustValue for &'a String {
    fn into_value(self) -> Value {
        Value::String(Rc::new(self.clone()))
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches!(ty.kind, TypeKind::String | TypeKind::Unknown)
    }

    fn type_description() -> &'static str {
        "string"
    }
}

impl FromLustValue for String {
    fn from_value(value: Value) -> Result<Self> {
        match value {
            Value::String(s) => Ok((*s).clone()),
            other => Err(LustError::RuntimeError {
                message: format!("Expected Lust value 'string' but received '{:?}'", other),
            }),
        }
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches!(ty.kind, TypeKind::String | TypeKind::Unknown)
    }

    fn type_description() -> &'static str {
        "string"
    }
}

impl FromLustValue for () {
    fn from_value(value: Value) -> Result<Self> {
        match value {
            Value::Nil => Ok(()),
            other => Err(LustError::RuntimeError {
                message: format!("Expected Lust value 'unit' but received '{:?}'", other),
            }),
        }
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches!(ty.kind, TypeKind::Unit | TypeKind::Unknown)
    }

    fn type_description() -> &'static str {
        "unit"
    }
}

impl FunctionArgs for () {
    fn into_values(self) -> Vec<Value> {
        Vec::new()
    }

    fn validate_signature(function_name: &str, params: &[Type]) -> Result<()> {
        ensure_arity(function_name, params, 0)
    }
}

impl<T> FunctionArgs for T
where
    T: IntoLustValue,
{
    fn into_values(self) -> Vec<Value> {
        vec![self.into_value()]
    }

    fn validate_signature(function_name: &str, params: &[Type]) -> Result<()> {
        ensure_arity(function_name, params, 1)?;
        ensure_arg_type::<T>(function_name, params, 0)
    }
}

impl<A, B> FunctionArgs for (A, B)
where
    A: IntoLustValue,
    B: IntoLustValue,
{
    fn into_values(self) -> Vec<Value> {
        vec![self.0.into_value(), self.1.into_value()]
    }

    fn validate_signature(function_name: &str, params: &[Type]) -> Result<()> {
        ensure_arity(function_name, params, 2)?;
        ensure_arg_type::<A>(function_name, params, 0)?;
        ensure_arg_type::<B>(function_name, params, 1)?;
        Ok(())
    }
}

impl<A, B, C> FunctionArgs for (A, B, C)
where
    A: IntoLustValue,
    B: IntoLustValue,
    C: IntoLustValue,
{
    fn into_values(self) -> Vec<Value> {
        vec![
            self.0.into_value(),
            self.1.into_value(),
            self.2.into_value(),
        ]
    }

    fn validate_signature(function_name: &str, params: &[Type]) -> Result<()> {
        ensure_arity(function_name, params, 3)?;
        ensure_arg_type::<A>(function_name, params, 0)?;
        ensure_arg_type::<B>(function_name, params, 1)?;
        ensure_arg_type::<C>(function_name, params, 2)?;
        Ok(())
    }
}

impl<A, B, C, D> FunctionArgs for (A, B, C, D)
where
    A: IntoLustValue,
    B: IntoLustValue,
    C: IntoLustValue,
    D: IntoLustValue,
{
    fn into_values(self) -> Vec<Value> {
        vec![
            self.0.into_value(),
            self.1.into_value(),
            self.2.into_value(),
            self.3.into_value(),
        ]
    }

    fn validate_signature(function_name: &str, params: &[Type]) -> Result<()> {
        ensure_arity(function_name, params, 4)?;
        ensure_arg_type::<A>(function_name, params, 0)?;
        ensure_arg_type::<B>(function_name, params, 1)?;
        ensure_arg_type::<C>(function_name, params, 2)?;
        ensure_arg_type::<D>(function_name, params, 3)?;
        Ok(())
    }
}

impl<A, B, C, D, E> FunctionArgs for (A, B, C, D, E)
where
    A: IntoLustValue,
    B: IntoLustValue,
    C: IntoLustValue,
    D: IntoLustValue,
    E: IntoLustValue,
{
    fn into_values(self) -> Vec<Value> {
        vec![
            self.0.into_value(),
            self.1.into_value(),
            self.2.into_value(),
            self.3.into_value(),
            self.4.into_value(),
        ]
    }

    fn validate_signature(function_name: &str, params: &[Type]) -> Result<()> {
        ensure_arity(function_name, params, 5)?;
        ensure_arg_type::<A>(function_name, params, 0)?;
        ensure_arg_type::<B>(function_name, params, 1)?;
        ensure_arg_type::<C>(function_name, params, 2)?;
        ensure_arg_type::<D>(function_name, params, 3)?;
        ensure_arg_type::<E>(function_name, params, 4)?;
        Ok(())
    }
}

fn ensure_arity(function_name: &str, params: &[Type], provided: usize) -> Result<()> {
    if params.len() == provided {
        Ok(())
    } else {
        Err(LustError::TypeError {
            message: format!(
                "Function '{}' expects {} argument(s) but {} were supplied",
                function_name,
                params.len(),
                provided
            ),
        })
    }
}

fn ensure_arg_type<T: IntoLustValue>(
    function_name: &str,
    params: &[Type],
    index: usize,
) -> Result<()> {
    if <T as IntoLustValue>::matches_lust_type(&params[index]) {
        Ok(())
    } else {
        Err(argument_type_mismatch(
            function_name,
            index,
            <T as IntoLustValue>::type_description(),
            &params[index],
        ))
    }
}

fn argument_type_mismatch(
    function_name: &str,
    index: usize,
    rust_type: &str,
    lust_type: &Type,
) -> LustError {
    LustError::TypeError {
        message: format!(
            "Function '{}' parameter {} expects Lust type '{}' but Rust provided '{}'",
            function_name,
            index + 1,
            lust_type,
            rust_type
        ),
    }
}
