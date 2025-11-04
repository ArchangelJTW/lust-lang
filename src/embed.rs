use crate::ast::{EnumDef, FieldOwnership, Item, ItemKind, Span, StructDef, Type, TypeKind};
use crate::bytecode::{
    Compiler, FieldStorage, NativeCallResult, StructLayout, TaskHandle, Value, ValueKey,
};
use crate::modules::{ModuleImports, ModuleLoader};
use crate::number::{LustFloat, LustInt};
use crate::typechecker::{FunctionSignature, TypeChecker};
use crate::vm::VM;
use crate::{LustConfig, LustError, Result};
use hashbrown::HashMap;
use std::cell::{Ref, RefCell, RefMut};
use std::collections::VecDeque;
use std::future::Future;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

type AsyncValueFuture = Pin<Box<dyn Future<Output = std::result::Result<Value, String>>>>;

struct AsyncRegistry {
    next_id: u64,
    pending: HashMap<u64, AsyncTaskEntry>,
}

impl AsyncRegistry {
    fn new() -> Self {
        Self {
            next_id: 1,
            pending: HashMap::new(),
        }
    }

    fn register(&mut self, entry: AsyncTaskEntry) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.pending.insert(id, entry);
        id
    }

    fn has_pending_for(&self, handle: TaskHandle) -> bool {
        self.pending.values().any(|entry| match entry.target {
            AsyncTaskTarget::ScriptTask(existing) | AsyncTaskTarget::NativeTask(existing) => {
                existing == handle
            }
        })
    }

    fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

struct AsyncTaskEntry {
    target: AsyncTaskTarget,
    future: AsyncValueFuture,
    wake_flag: Arc<WakeFlag>,
    immediate_poll: bool,
}

#[derive(Clone, Copy)]
enum AsyncTaskTarget {
    ScriptTask(TaskHandle),
    NativeTask(TaskHandle),
}

impl AsyncTaskEntry {
    fn new(target: AsyncTaskTarget, future: AsyncValueFuture) -> Self {
        Self {
            target,
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

struct AsyncTaskQueueInner<Args, R> {
    queue: Mutex<VecDeque<PendingAsyncTask<Args, R>>>,
    condvar: Condvar,
}

pub struct AsyncTaskQueue<Args, R> {
    inner: Arc<AsyncTaskQueueInner<Args, R>>,
}

impl<Args, R> Clone for AsyncTaskQueue<Args, R> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<Args, R> AsyncTaskQueue<Args, R> {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(AsyncTaskQueueInner {
                queue: Mutex::new(VecDeque::new()),
                condvar: Condvar::new(),
            }),
        }
    }

    pub fn push(&self, task: PendingAsyncTask<Args, R>) {
        let mut guard = self.inner.queue.lock().unwrap();
        guard.push_back(task);
        self.inner.condvar.notify_one();
    }

    pub fn pop(&self) -> Option<PendingAsyncTask<Args, R>> {
        let mut guard = self.inner.queue.lock().unwrap();
        guard.pop_front()
    }

    pub fn pop_blocking(&self) -> PendingAsyncTask<Args, R> {
        let mut guard = self.inner.queue.lock().unwrap();
        loop {
            if let Some(task) = guard.pop_front() {
                return task;
            }
            guard = self.inner.condvar.wait(guard).unwrap();
        }
    }

    pub fn len(&self) -> usize {
        let guard = self.inner.queue.lock().unwrap();
        guard.len()
    }

    pub fn is_empty(&self) -> bool {
        let guard = self.inner.queue.lock().unwrap();
        guard.is_empty()
    }
}

pub struct PendingAsyncTask<Args, R> {
    task: TaskHandle,
    args: Args,
    completer: AsyncCompleter<R>,
}

impl<Args, R> PendingAsyncTask<Args, R> {
    fn new(task: TaskHandle, args: Args, completer: AsyncCompleter<R>) -> Self {
        Self {
            task,
            args,
            completer,
        }
    }

    pub fn task(&self) -> TaskHandle {
        self.task
    }

    pub fn args(&self) -> &Args {
        &self.args
    }

    pub fn complete_ok(self, value: R) {
        self.completer.complete_ok(value);
    }

    pub fn complete_err(self, message: impl Into<String>) {
        self.completer.complete_err(message);
    }
}

struct AsyncSignalState<T> {
    result: Mutex<Option<std::result::Result<T, String>>>,
    waker: Mutex<Option<Waker>>,
}

struct AsyncCompleter<T> {
    inner: Arc<AsyncSignalState<T>>,
}

impl<T> Clone for AsyncCompleter<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> AsyncCompleter<T> {
    fn complete_ok(&self, value: T) {
        self.set_result(Ok(value));
    }

    fn complete_err(&self, message: impl Into<String>) {
        self.set_result(Err(message.into()));
    }

    fn set_result(&self, value: std::result::Result<T, String>) {
        {
            let mut guard = self.inner.result.lock().unwrap();
            if guard.is_some() {
                return;
            }
            *guard = Some(value);
        }

        if let Some(waker) = self.inner.waker.lock().unwrap().take() {
            waker.wake();
        }
    }
}

struct AsyncSignalFuture<T> {
    inner: Arc<AsyncSignalState<T>>,
}

impl<T> Future for AsyncSignalFuture<T> {
    type Output = std::result::Result<T, String>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        {
            let mut guard = self.inner.result.lock().unwrap();
            if let Some(result) = guard.take() {
                return Poll::Ready(result);
            }
        }

        let mut waker_slot = self.inner.waker.lock().unwrap();
        *waker_slot = Some(cx.waker().clone());
        Poll::Pending
    }
}

fn signal_pair<T>() -> (AsyncCompleter<T>, AsyncSignalFuture<T>) {
    let inner = Arc::new(AsyncSignalState {
        result: Mutex::new(None),
        waker: Mutex::new(None),
    });
    (
        AsyncCompleter {
            inner: Arc::clone(&inner),
        },
        AsyncSignalFuture { inner },
    )
}
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
    let option_coercions = typechecker.take_option_coercions();
    let struct_defs = typechecker.struct_definitions();
    let enum_defs = typechecker.enum_definitions();
    let mut signatures = typechecker.function_signatures();
    let mut compiler = Compiler::new();
    compiler.set_option_coercions(option_coercions);
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

pub struct StructField {
    name: String,
    value: TypedValue,
}

impl StructField {
    pub fn new(name: impl Into<String>, value: impl IntoTypedValue) -> Self {
        Self {
            name: name.into(),
            value: value.into_typed_value(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    fn into_parts(self) -> (String, TypedValue) {
        (self.name, self.value)
    }
}

pub fn struct_field(name: impl Into<String>, value: impl IntoTypedValue) -> StructField {
    StructField::new(name, value)
}

impl<K, V> From<(K, V)> for StructField
where
    K: Into<String>,
    V: IntoTypedValue,
{
    fn from((name, value): (K, V)) -> Self {
        StructField::new(name, value)
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
        let value_ref = self.borrow_field(field)?;
        T::from_value(value_ref.into_owned())
    }

    pub fn borrow_field(&self, field: &str) -> Result<ValueRef<'_>> {
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
                match layout.field_storage(index) {
                    FieldStorage::Strong => {
                        let slots = fields.borrow();
                        if slots.get(index).is_none() {
                            return Err(LustError::RuntimeError {
                                message: format!(
                                    "Struct '{}' field '{}' is unavailable",
                                    self.type_name, field
                                ),
                            });
                        }

                        Ok(ValueRef::borrowed(Ref::map(slots, move |values| {
                            &values[index]
                        })))
                    }

                    FieldStorage::Weak => {
                        let stored = {
                            let slots = fields.borrow();
                            slots
                                .get(index)
                                .cloned()
                                .ok_or_else(|| LustError::RuntimeError {
                                    message: format!(
                                        "Struct '{}' field '{}' is unavailable",
                                        self.type_name, field
                                    ),
                                })?
                        };
                        let materialized = layout.materialize_field_value(index, stored);
                        Ok(ValueRef::owned(materialized))
                    }
                }
            }

            _ => Err(LustError::RuntimeError {
                message: "StructInstance does not contain a struct value".to_string(),
            }),
        }
    }

    pub fn set_field<V: IntoTypedValue>(&self, field: &str, value: V) -> Result<()> {
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
                let typed_value = value.into_typed_value();
                let matches_declared = typed_value.matches(layout.field_type(index));
                let matches_ref_inner = layout.is_weak(index)
                    && layout
                        .weak_target(index)
                        .map(|inner| typed_value.matches(inner))
                        .unwrap_or(false);
                if !(matches_declared || matches_ref_inner) {
                    return Err(LustError::TypeError {
                        message: format!(
                            "Struct '{}' field '{}' expects Lust type '{}' but Rust provided '{}'",
                            self.type_name,
                            field,
                            layout.field_type(index),
                            typed_value.description()
                        ),
                    });
                }

                let canonical_value = layout
                    .canonicalize_field_value(index, typed_value.into_value())
                    .map_err(|message| LustError::TypeError { message })?;
                fields.borrow_mut()[index] = canonical_value;
                Ok(())
            }

            _ => Err(LustError::RuntimeError {
                message: "StructInstance does not contain a struct value".to_string(),
            }),
        }
    }

    pub fn update_field<F, V>(&self, field: &str, update: F) -> Result<()>
    where
        F: FnOnce(Value) -> Result<V>,
        V: IntoTypedValue,
    {
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
                let mut slots = fields.borrow_mut();
                let slot = slots
                    .get_mut(index)
                    .ok_or_else(|| LustError::RuntimeError {
                        message: format!(
                            "Struct '{}' field '{}' is unavailable",
                            self.type_name, field
                        ),
                    })?;
                let fallback = slot.clone();
                let current_canonical = std::mem::replace(slot, Value::Nil);
                let current_materialized = layout.materialize_field_value(index, current_canonical);
                let updated = match update(current_materialized) {
                    Ok(value) => value,
                    Err(err) => {
                        *slot = fallback;
                        return Err(err);
                    }
                };
                let typed_value = updated.into_typed_value();
                let matches_declared = typed_value.matches(layout.field_type(index));
                let matches_ref_inner = layout.is_weak(index)
                    && layout
                        .weak_target(index)
                        .map(|inner| typed_value.matches(inner))
                        .unwrap_or(false);
                if !(matches_declared || matches_ref_inner) {
                    *slot = fallback;
                    return Err(LustError::TypeError {
                        message: format!(
                            "Struct '{}' field '{}' expects Lust type '{}' but Rust provided '{}'",
                            self.type_name,
                            field,
                            layout.field_type(index),
                            typed_value.description()
                        ),
                    });
                }

                let canonical_value = layout
                    .canonicalize_field_value(index, typed_value.into_value())
                    .map_err(|message| LustError::TypeError { message })?;
                *slot = canonical_value;
                Ok(())
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
pub struct FunctionHandle {
    value: Value,
}

impl FunctionHandle {
    fn is_callable_value(value: &Value) -> bool {
        matches!(
            value,
            Value::Function(_) | Value::Closure { .. } | Value::NativeFunction(_)
        )
    }

    fn new_unchecked(value: Value) -> Self {
        Self { value }
    }

    pub fn from_value(value: Value) -> Result<Self> {
        if Self::is_callable_value(&value) {
            Ok(Self::new_unchecked(value))
        } else {
            Err(LustError::RuntimeError {
                message: format!("Expected Lust value 'function' but received '{:?}'", value),
            })
        }
    }

    pub fn as_value(&self) -> &Value {
        &self.value
    }

    pub fn into_value(self) -> Value {
        self.value
    }

    fn function_index(&self) -> Option<usize> {
        match &self.value {
            Value::Function(idx) => Some(*idx),
            Value::Closure { function_idx, .. } => Some(*function_idx),
            _ => None,
        }
    }

    fn function_name<'a>(&'a self, program: &'a EmbeddedProgram) -> Option<&'a str> {
        let idx = self.function_index()?;
        program.vm.function_name(idx)
    }

    pub fn signature<'a>(
        &'a self,
        program: &'a EmbeddedProgram,
    ) -> Option<(&'a str, &'a FunctionSignature)> {
        let name = self.function_name(program)?;
        program.signature(name).map(|sig| (name, sig))
    }

    pub fn matches_signature(
        &self,
        program: &EmbeddedProgram,
        expected: &FunctionSignature,
    ) -> bool {
        match self.signature(program) {
            Some((_, actual)) => signatures_match(actual, expected),
            None => false,
        }
    }

    pub fn validate_signature(
        &self,
        program: &EmbeddedProgram,
        expected: &FunctionSignature,
    ) -> Result<()> {
        let (name, actual) = self.signature(program).ok_or_else(|| LustError::TypeError {
            message: "No type information available for function value; use call_raw if the function is dynamically typed"
                .into(),
        })?;
        if signatures_match(actual, expected) {
            Ok(())
        } else {
            Err(LustError::TypeError {
                message: format!(
                    "Function '{}' signature mismatch: expected {}, found {}",
                    name,
                    signature_to_string(expected),
                    signature_to_string(actual)
                ),
            })
        }
    }

    pub fn call_raw(&self, program: &mut EmbeddedProgram, args: Vec<Value>) -> Result<Value> {
        program.vm.call_value(self.as_value(), args)
    }

    pub fn call_typed<Args, R>(
        &self,
        program: &mut EmbeddedProgram,
        args: Args,
    ) -> Result<R>
    where
        Args: FunctionArgs,
        R: FromLustValue,
    {
        let program_ref: &EmbeddedProgram = &*program;
        let values = args.into_values();
        if let Some((name, signature)) = self.signature(program_ref) {
            Args::validate_signature(name, &signature.params)?;
            ensure_return_type::<R>(name, &signature.return_type)?;
            let value = program.vm.call_value(self.as_value(), values)?;
            R::from_value(value)
        } else {
            let value = program.vm.call_value(self.as_value(), values)?;
            R::from_value(value)
        }
    }

}

#[derive(Clone)]
pub struct StructHandle {
    instance: StructInstance,
}

impl StructHandle {
    fn from_instance(instance: StructInstance) -> Self {
        Self { instance }
    }

    fn from_parts(
        name: &String,
        layout: &Rc<StructLayout>,
        fields: &Rc<RefCell<Vec<Value>>>,
    ) -> Self {
        let value = Value::Struct {
            name: name.clone(),
            layout: layout.clone(),
            fields: fields.clone(),
        };
        Self::from_instance(StructInstance::new(name.clone(), value))
    }

    pub fn from_value(value: Value) -> Result<Self> {
        <StructInstance as FromLustValue>::from_value(value).map(StructHandle::from)
    }

    pub fn type_name(&self) -> &str {
        self.instance.type_name()
    }

    pub fn field<T: FromLustValue>(&self, field: &str) -> Result<T> {
        self.instance.field(field)
    }

    pub fn borrow_field(&self, field: &str) -> Result<ValueRef<'_>> {
        self.instance.borrow_field(field)
    }

    pub fn set_field<V: IntoTypedValue>(&self, field: &str, value: V) -> Result<()> {
        self.instance.set_field(field, value)
    }

    pub fn update_field<F, V>(&self, field: &str, update: F) -> Result<()>
    where
        F: FnOnce(Value) -> Result<V>,
        V: IntoTypedValue,
    {
        self.instance.update_field(field, update)
    }

    pub fn as_value(&self) -> &Value {
        self.instance.as_value()
    }

    pub fn to_instance(&self) -> StructInstance {
        self.instance.clone()
    }

    pub fn into_instance(self) -> StructInstance {
        self.instance
    }

    pub fn matches_type(&self, expected: &str) -> bool {
        lust_type_names_match(self.type_name(), expected)
    }

    pub fn ensure_type(&self, expected: &str) -> Result<()> {
        if self.matches_type(expected) {
            Ok(())
        } else {
            Err(LustError::TypeError {
                message: format!(
                    "Struct '{}' does not match expected type '{}'",
                    self.type_name(),
                    expected
                ),
            })
        }
    }
}

impl StructInstance {
    pub fn to_handle(&self) -> StructHandle {
        StructHandle::from_instance(self.clone())
    }

    pub fn into_handle(self) -> StructHandle {
        StructHandle::from_instance(self)
    }
}

impl From<StructInstance> for StructHandle {
    fn from(instance: StructInstance) -> Self {
        StructHandle::from_instance(instance)
    }
}

impl From<StructHandle> for StructInstance {
    fn from(handle: StructHandle) -> Self {
        handle.into_instance()
    }
}

pub enum ValueRef<'a> {
    Borrowed(Ref<'a, Value>),
    Owned(Value),
}

impl<'a> ValueRef<'a> {
    fn borrowed(inner: Ref<'a, Value>) -> Self {
        Self::Borrowed(inner)
    }

    fn owned(value: Value) -> Self {
        Self::Owned(value)
    }

    pub fn as_value(&self) -> &Value {
        match self {
            ValueRef::Borrowed(inner) => &*inner,
            ValueRef::Owned(value) => value,
        }
    }

    pub fn to_owned(&self) -> Value {
        self.as_value().clone()
    }

    pub fn into_owned(self) -> Value {
        match self {
            ValueRef::Borrowed(inner) => inner.clone(),
            ValueRef::Owned(value) => value,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self.as_value() {
            Value::Bool(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<LustInt> {
        self.as_value().as_int()
    }

    pub fn as_float(&self) -> Option<LustFloat> {
        self.as_value().as_float()
    }

    pub fn as_string(&self) -> Option<&str> {
        self.as_value().as_string()
    }

    pub fn as_rc_string(&self) -> Option<Rc<String>> {
        match self.as_value() {
            Value::String(value) => Some(value.clone()),
            _ => None,
        }
    }

    pub fn as_array_handle(&self) -> Option<ArrayHandle> {
        match self.as_value() {
            Value::Array(items) => Some(ArrayHandle::from_rc(items.clone())),
            _ => None,
        }
    }

    pub fn as_map_handle(&self) -> Option<MapHandle> {
        match self.as_value() {
            Value::Map(map) => Some(MapHandle::from_rc(map.clone())),
            _ => None,
        }
    }

    pub fn as_struct_handle(&self) -> Option<StructHandle> {
        match self.as_value() {
            Value::Struct {
                name,
                layout,
                fields,
            } => Some(StructHandle::from_parts(name, layout, fields)),
            Value::WeakStruct(weak) => weak
                .upgrade()
                .and_then(|value| StructHandle::from_value(value).ok()),
            _ => None,
        }
    }
}

pub struct StringRef<'a> {
    value: ValueRef<'a>,
}

impl<'a> StringRef<'a> {
    fn new(value: ValueRef<'a>) -> Self {
        Self { value }
    }

    pub fn as_str(&self) -> &str {
        self.value
            .as_string()
            .expect("StringRef must wrap a Lust string")
    }

    pub fn as_rc(&self) -> Rc<String> {
        self.value
            .as_rc_string()
            .expect("StringRef must wrap a Lust string")
    }

    pub fn to_value(&self) -> &Value {
        self.value.as_value()
    }

    pub fn into_value_ref(self) -> ValueRef<'a> {
        self.value
    }
}

impl<'a> Deref for StringRef<'a> {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

#[derive(Clone)]
pub struct ArrayHandle {
    inner: Rc<RefCell<Vec<Value>>>,
}

impl ArrayHandle {
    fn from_rc(inner: Rc<RefCell<Vec<Value>>>) -> Self {
        Self { inner }
    }

    pub fn len(&self) -> usize {
        self.inner.borrow().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn borrow(&self) -> Ref<'_, [Value]> {
        Ref::map(self.inner.borrow(), |values| values.as_slice())
    }

    pub fn borrow_mut(&self) -> RefMut<'_, Vec<Value>> {
        self.inner.borrow_mut()
    }

    pub fn push(&self, value: Value) {
        self.inner.borrow_mut().push(value);
    }

    pub fn extend<I>(&self, iter: I)
    where
        I: IntoIterator<Item = Value>,
    {
        self.inner.borrow_mut().extend(iter);
    }

    pub fn get(&self, index: usize) -> Option<ValueRef<'_>> {
        {
            let values = self.inner.borrow();
            if values.get(index).is_none() {
                return None;
            }
        }

        let values = self.inner.borrow();
        Some(ValueRef::borrowed(Ref::map(values, move |items| {
            &items[index]
        })))
    }

    pub fn with_ref<R>(&self, f: impl FnOnce(&[Value]) -> R) -> R {
        let values = self.inner.borrow();
        f(values.as_slice())
    }

    pub fn with_mut<R>(&self, f: impl FnOnce(&mut Vec<Value>) -> R) -> R {
        let mut values = self.inner.borrow_mut();
        f(&mut values)
    }
}

#[derive(Clone)]
pub struct MapHandle {
    inner: Rc<RefCell<HashMap<ValueKey, Value>>>,
}

impl MapHandle {
    fn from_rc(inner: Rc<RefCell<HashMap<ValueKey, Value>>>) -> Self {
        Self { inner }
    }

    pub fn len(&self) -> usize {
        self.inner.borrow().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn borrow(&self) -> Ref<'_, HashMap<ValueKey, Value>> {
        self.inner.borrow()
    }

    pub fn borrow_mut(&self) -> RefMut<'_, HashMap<ValueKey, Value>> {
        self.inner.borrow_mut()
    }

    pub fn contains_key<K>(&self, key: K) -> bool
    where
        K: Into<ValueKey>,
    {
        self.inner.borrow().contains_key(&key.into())
    }

    pub fn get<K>(&self, key: K) -> Option<ValueRef<'_>>
    where
        K: Into<ValueKey>,
    {
        let key = key.into();
        {
            if !self.inner.borrow().contains_key(&key) {
                return None;
            }
        }
        let lookup = key.clone();
        let map = self.inner.borrow();
        Some(ValueRef::borrowed(Ref::map(map, move |values| {
            values
                .get(&lookup)
                .expect("lookup key should be present after contains_key")
        })))
    }

    pub fn insert<K>(&self, key: K, value: Value) -> Option<Value>
    where
        K: Into<ValueKey>,
    {
        self.inner.borrow_mut().insert(key.into(), value)
    }

    pub fn remove<K>(&self, key: K) -> Option<Value>
    where
        K: Into<ValueKey>,
    {
        self.inner.borrow_mut().remove(&key.into())
    }

    pub fn with_ref<R>(&self, f: impl FnOnce(&HashMap<ValueKey, Value>) -> R) -> R {
        let map = self.inner.borrow();
        f(&map)
    }

    pub fn with_mut<R>(&self, f: impl FnOnce(&mut HashMap<ValueKey, Value>) -> R) -> R {
        let mut map = self.inner.borrow_mut();
        f(&mut map)
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

fn struct_field_type_error(field: &str, expected: &str, actual: &Value) -> LustError {
    LustError::TypeError {
        message: format!(
            "Struct field '{}' expects '{}' but received value of type '{:?}'",
            field,
            expected,
            actual.type_of()
        ),
    }
}

pub trait FromStructField<'a>: Sized {
    fn from_value(field: &str, value: ValueRef<'a>) -> Result<Self>;
}

impl<'a> FromStructField<'a> for ValueRef<'a> {
    fn from_value(_field: &str, value: ValueRef<'a>) -> Result<Self> {
        Ok(value)
    }
}

impl<'a> FromStructField<'a> for StructHandle {
    fn from_value(field: &str, value: ValueRef<'a>) -> Result<Self> {
        value
            .as_struct_handle()
            .ok_or_else(|| struct_field_type_error(field, "struct", value.as_value()))
    }
}

impl<'a> FromStructField<'a> for StructInstance {
    fn from_value(field: &str, value: ValueRef<'a>) -> Result<Self> {
        value
            .as_struct_handle()
            .map(|handle| handle.to_instance())
            .ok_or_else(|| struct_field_type_error(field, "struct", value.as_value()))
    }
}

impl<'a> FromStructField<'a> for ArrayHandle {
    fn from_value(field: &str, value: ValueRef<'a>) -> Result<Self> {
        value
            .as_array_handle()
            .ok_or_else(|| struct_field_type_error(field, "array", value.as_value()))
    }
}

impl<'a> FromStructField<'a> for MapHandle {
    fn from_value(field: &str, value: ValueRef<'a>) -> Result<Self> {
        value
            .as_map_handle()
            .ok_or_else(|| struct_field_type_error(field, "map", value.as_value()))
    }
}

impl<'a> FromStructField<'a> for FunctionHandle {
    fn from_value(field: &str, value: ValueRef<'a>) -> Result<Self> {
        let owned = value.to_owned();
        FunctionHandle::from_value(owned)
            .map_err(|_| struct_field_type_error(field, "function", value.as_value()))
    }
}

impl<'a> FromStructField<'a> for LustInt {
    fn from_value(field: &str, value: ValueRef<'a>) -> Result<Self> {
        value
            .as_int()
            .ok_or_else(|| struct_field_type_error(field, "int", value.as_value()))
    }
}

impl<'a> FromStructField<'a> for LustFloat {
    fn from_value(field: &str, value: ValueRef<'a>) -> Result<Self> {
        value
            .as_float()
            .ok_or_else(|| struct_field_type_error(field, "float", value.as_value()))
    }
}

impl<'a> FromStructField<'a> for bool {
    fn from_value(field: &str, value: ValueRef<'a>) -> Result<Self> {
        value
            .as_bool()
            .ok_or_else(|| struct_field_type_error(field, "bool", value.as_value()))
    }
}

impl<'a> FromStructField<'a> for Rc<String> {
    fn from_value(field: &str, value: ValueRef<'a>) -> Result<Self> {
        value
            .as_rc_string()
            .ok_or_else(|| struct_field_type_error(field, "string", value.as_value()))
    }
}

impl<'a> FromStructField<'a> for Value {
    fn from_value(_field: &str, value: ValueRef<'a>) -> Result<Self> {
        Ok(value.into_owned())
    }
}

impl<'a, T> FromStructField<'a> for Option<T>
where
    T: FromStructField<'a>,
{
    fn from_value(field: &str, value: ValueRef<'a>) -> Result<Self> {
        if matches!(value.as_value(), Value::Nil) {
            Ok(None)
        } else {
            T::from_value(field, value).map(Some)
        }
    }
}

impl<'a> FromStructField<'a> for StringRef<'a> {
    fn from_value(field: &str, value: ValueRef<'a>) -> Result<Self> {
        if value.as_string().is_some() {
            Ok(StringRef::new(value))
        } else {
            Err(struct_field_type_error(field, "string", value.as_value()))
        }
    }
}

impl<'a> FromStructField<'a> for EnumInstance {
    fn from_value(field: &str, value: ValueRef<'a>) -> Result<Self> {
        match value.as_value() {
            Value::Enum { .. } => <EnumInstance as FromLustValue>::from_value(value.into_owned()),
            other => Err(struct_field_type_error(field, "enum", other)),
        }
    }
}

pub trait LustStructView<'a>: Sized {
    const TYPE_NAME: &'static str;

    fn from_handle(handle: &'a StructHandle) -> Result<Self>;
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

impl IntoTypedValue for StructHandle {
    fn into_typed_value(self) -> TypedValue {
        <StructInstance as IntoTypedValue>::into_typed_value(self.into_instance())
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

impl IntoTypedValue for FunctionHandle {
    fn into_typed_value(self) -> TypedValue {
        let value = self.into_value();
        TypedValue::new(value, |_v, ty| matches_function_handle_type(ty), "function")
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

impl IntoTypedValue for ArrayHandle {
    fn into_typed_value(self) -> TypedValue {
        let value = self.into_value();
        TypedValue::new(value, |_, ty| matches_array_handle_type(ty), "array")
    }
}

impl IntoTypedValue for MapHandle {
    fn into_typed_value(self) -> TypedValue {
        let value = self.into_value();
        TypedValue::new(value, |_, ty| matches_map_handle_type(ty), "map")
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
    use crate::embed::LustStructView;
    use std::rc::Rc;

    fn serial_guard() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|err| err.into_inner())
    }

    fn build_program(source: &str) -> EmbeddedProgram {
        EmbeddedProgram::builder()
            .module("main", source)
            .entry_module("main")
            .compile()
            .expect("compile embedded program")
    }

    #[test]
    fn struct_instance_supports_mixed_field_types() {
        let _guard = serial_guard();
        let source = r#"
            struct Mixed
                count: int
                label: string
                enabled: bool
            end
        "#;

        let program = build_program(source);
        let mixed = program
            .struct_instance(
                "main.Mixed",
                [
                    struct_field("count", 7_i64),
                    struct_field("label", "hi"),
                    struct_field("enabled", true),
                ],
            )
            .expect("struct instance");

        assert_eq!(mixed.field::<i64>("count").expect("count field"), 7);
        assert_eq!(mixed.field::<String>("label").expect("label field"), "hi");
        assert!(mixed.field::<bool>("enabled").expect("enabled field"));
    }

    #[test]
    fn struct_instance_borrow_field_provides_reference_view() {
        let _guard = serial_guard();
        let source = r#"
            struct Sample
                name: string
            end
        "#;

        let program = build_program(source);
        let sample = program
            .struct_instance("main.Sample", [struct_field("name", "Borrowed")])
            .expect("struct instance");

        let name_ref = sample.borrow_field("name").expect("borrow name field");
        assert_eq!(name_ref.as_string().unwrap(), "Borrowed");
        assert!(name_ref.as_array_handle().is_none());
    }

    #[test]
    fn array_handle_allows_in_place_mutation() {
        let _guard = serial_guard();
        let value = Value::array(vec![Value::Int(1)]);
        let handle = <ArrayHandle as FromLustValue>::from_value(value).expect("array handle");

        {
            let mut slots = handle.borrow_mut();
            slots.push(Value::Int(2));
            slots.push(Value::Int(3));
        }

        let snapshot: Vec<_> = handle
            .borrow()
            .iter()
            .map(|value| value.as_int().expect("int value"))
            .collect();
        assert_eq!(snapshot, vec![1, 2, 3]);
    }

    #[test]
    fn struct_instance_allows_setting_fields() {
        let _guard = serial_guard();
        let source = r#"
            struct Mixed
                count: int
                label: string
                enabled: bool
            end
        "#;

        let program = build_program(source);
        let mixed = program
            .struct_instance(
                "main.Mixed",
                [
                    struct_field("count", 1_i64),
                    struct_field("label", "start"),
                    struct_field("enabled", false),
                ],
            )
            .expect("struct instance");

        mixed
            .set_field("count", 11_i64)
            .expect("update count field");
        assert_eq!(mixed.field::<i64>("count").expect("count field"), 11);

        let err = mixed
            .set_field("count", "oops")
            .expect_err("type mismatch should fail");
        match err {
            LustError::TypeError { message } => {
                assert!(message.contains("count"));
                assert!(message.contains("int"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert_eq!(mixed.field::<i64>("count").expect("count field"), 11);

        mixed
            .set_field("label", String::from("updated"))
            .expect("update label");
        assert_eq!(
            mixed.field::<String>("label").expect("label field"),
            "updated"
        );

        mixed.set_field("enabled", true).expect("update enabled");
        assert!(mixed.field::<bool>("enabled").expect("enabled field"));
    }

    #[test]
    fn struct_instance_accepts_nested_structs() {
        let _guard = serial_guard();
        let source = r#"
            struct Child
                value: int
            end

            struct Parent
                child: main.Child
            end
        "#;

        let program = build_program(source);
        let child = program
            .struct_instance("main.Child", [struct_field("value", 42_i64)])
            .expect("child struct");
        let parent = program
            .struct_instance("main.Parent", [struct_field("child", child.clone())])
            .expect("parent struct");

        let nested: StructInstance = parent.field("child").expect("child field");
        assert_eq!(nested.field::<i64>("value").expect("value field"), 42);
    }

    #[test]
    fn struct_handle_allows_field_mutation() {
        let _guard = serial_guard();
        let source = r#"
            struct Counter
                value: int
            end
        "#;

        let program = build_program(source);
        let counter = program
            .struct_instance("main.Counter", [struct_field("value", 1_i64)])
            .expect("counter struct");
        let handle = counter.to_handle();

        handle
            .set_field("value", 7_i64)
            .expect("update through handle");
        assert_eq!(handle.field::<i64>("value").expect("value field"), 7);
        assert_eq!(counter.field::<i64>("value").expect("value field"), 7);

        handle
            .update_field("value", |current| match current {
                Value::Int(v) => Ok(v + 1),
                other => Err(LustError::RuntimeError {
                    message: format!("unexpected value {other:?}"),
                }),
            })
            .expect("increment value");
        assert_eq!(counter.field::<i64>("value").expect("value field"), 8);
    }

    #[test]
    fn value_ref_can_materialize_struct_handle() {
        let _guard = serial_guard();
        let source = r#"
            struct Child
                value: int
            end

            struct Parent
                child: main.Child
            end
        "#;

        let program = build_program(source);
        let child = program
            .struct_instance("main.Child", [struct_field("value", 10_i64)])
            .expect("child struct");
        let parent = program
            .struct_instance("main.Parent", [struct_field("child", child)])
            .expect("parent struct");

        let handle = {
            let child_ref = parent.borrow_field("child").expect("child field borrow");
            child_ref
                .as_struct_handle()
                .expect("struct handle from value ref")
        };
        handle
            .set_field("value", 55_i64)
            .expect("update nested value");

        let nested = parent
            .field::<StructInstance>("child")
            .expect("child field");
        assert_eq!(nested.field::<i64>("value").expect("value field"), 55);
    }

    #[derive(crate::LustStructView)]
    #[lust(type = "main.Child", crate = "crate")]
    struct ChildView<'a> {
        #[lust(field = "value")]
        value: ValueRef<'a>,
    }

    #[derive(crate::LustStructView)]
    #[lust(type = "main.Parent", crate = "crate")]
    struct ParentView<'a> {
        #[lust(field = "child")]
        child: StructHandle,
        #[lust(field = "label")]
        label: StringRef<'a>,
    }

    #[test]
    fn derive_struct_view_zero_copy() {
        let _guard = serial_guard();
        let source = r#"
            struct Child
                value: int
            end

            struct Parent
                child: main.Child
                label: string
            end
        "#;

        let program = build_program(source);
        let child = program
            .struct_instance("main.Child", [struct_field("value", 7_i64)])
            .expect("child struct");
        let parent = program
            .struct_instance(
                "main.Parent",
                [
                    struct_field("child", child.clone()),
                    struct_field("label", "parent label"),
                ],
            )
            .expect("parent struct");

        let handle = parent.to_handle();
        let view = ParentView::from_handle(&handle).expect("construct view");
        assert_eq!(view.child.field::<i64>("value").expect("child value"), 7);
        let label_rc_from_view = view.label.as_rc();
        assert_eq!(&*label_rc_from_view, "parent label");

        let label_ref = parent.borrow_field("label").expect("borrow label");
        let label_rc = label_ref.as_rc_string().expect("label rc");
        assert!(Rc::ptr_eq(&label_rc_from_view, &label_rc));

        let child_view = ChildView::from_handle(&view.child).expect("child view");
        assert_eq!(child_view.value.as_int().expect("child value"), 7);

        match ParentView::from_handle(&child.to_handle()) {
            Ok(_) => panic!("expected type mismatch"),
            Err(LustError::TypeError { message }) => {
                assert!(message.contains("Parent"), "unexpected message: {message}");
            }
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn globals_snapshot_exposes_lust_values() {
        let _guard = serial_guard();
        let source = r#"
            struct Child
                value: int
            end

            struct Parent
                child: unknown
            end

            function make_parent(): Parent
                return Parent { child = Child { value = 3 } }
            end
        "#;

        let mut program = build_program(source);
        program.run_entry_script().expect("run entry script");
        let parent: StructInstance = program
            .call_typed("main.make_parent", ())
            .expect("call make_parent");
        program.set_global_value("main.some_nested_structure", parent.clone());

        let globals = program.globals();
        let (_, value) = globals
            .into_iter()
            .find(|(name, _)| name.ends_with("some_nested_structure"))
            .expect("global binding present");
        let stored =
            <StructInstance as FromLustValue>::from_value(value).expect("convert to struct");
        let child_value = stored
            .field::<StructInstance>("child")
            .expect("nested child");
        assert_eq!(child_value.field::<i64>("value").expect("child value"), 3);
    }

    #[test]
    fn function_handle_supports_typed_and_raw_calls() {
        let _guard = serial_guard();
        let source = r#"
            pub function add(a: int, b: int): int
                return a + b
            end
        "#;

        let mut program = build_program(source);
        let handle = program
            .function_handle("main.add")
            .expect("function handle");

        let typed: i64 = handle
            .call_typed(&mut program, (2_i64, 3_i64))
            .expect("typed call");
        assert_eq!(typed, 5);

        let raw = handle
            .call_raw(&mut program, vec![Value::Int(4_i64), Value::Int(6_i64)])
            .expect("raw call");
        assert_eq!(raw.as_int(), Some(10));

        let signature = program
            .signature("main.add")
            .expect("signature")
            .clone();
        handle
            .validate_signature(&program, &signature)
            .expect("matching signature");

        let mut mismatched = signature.clone();
        mismatched.return_type = Type::new(TypeKind::Bool, Span::new(0, 0, 0, 0));
        assert!(
            handle.validate_signature(&program, &mismatched).is_err(),
            "expected signature mismatch"
        );
    }

    #[test]
    fn async_task_native_returns_task_handle() {
        let _guard = serial_guard();
        let source = r#"
            extern {
                function fetch_value(): Task
            }

            pub function start(): Task
                return fetch_value()
            end
        "#;

        let mut program = build_program(source);
        program
            .register_async_task_native::<(), LustInt, _, _>("fetch_value", move |_| async move {
                Ok(42_i64)
            })
            .expect("register async task native");

        let task_value = program
            .call_raw("main.start", Vec::new())
            .expect("call start");
        let handle = match task_value {
            Value::Task(handle) => handle,
            other => panic!("expected task handle, found {other:?}"),
        };

        {
            let mut driver = AsyncDriver::new(&mut program);
            driver.pump_until_idle().expect("poll async");
        }

        let (state_label, last_result, err) = {
            let vm = program.vm_mut();
            let task = vm.get_task_instance(handle).expect("task instance");
            (
                task.state.as_str().to_string(),
                task.last_result.clone(),
                task.error.clone(),
            )
        };
        assert_eq!(state_label, "completed");
        assert!(err.is_none());
        let int_value = last_result
            .and_then(|value| value.as_int())
            .expect("int result");
        assert_eq!(int_value, 42);
    }

    #[test]
    fn update_field_modifies_value_in_place() {
        let _guard = serial_guard();
        let source = r#"
            struct Counter
                value: int
            end
        "#;

        let program = build_program(source);
        let counter = program
            .struct_instance("main.Counter", [struct_field("value", 10_i64)])
            .expect("counter struct");

        counter
            .update_field("value", |current| match current {
                Value::Int(v) => Ok(v + 5),
                other => Err(LustError::RuntimeError {
                    message: format!("unexpected value {other:?}"),
                }),
            })
            .expect("update in place");
        assert_eq!(counter.field::<i64>("value").expect("value field"), 15);

        let err = counter
            .update_field("value", |_| Ok(String::from("oops")))
            .expect_err("string should fail type check");
        match err {
            LustError::TypeError { message } => {
                assert!(message.contains("value"));
                assert!(message.contains("int"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert_eq!(counter.field::<i64>("value").expect("value field"), 15);

        let err = counter
            .update_field("value", |_| -> Result<i64> {
                Err(LustError::RuntimeError {
                    message: "closure failure".to_string(),
                })
            })
            .expect_err("closure error should propagate");
        match err {
            LustError::RuntimeError { message } => assert_eq!(message, "closure failure"),
            other => panic!("unexpected error: {other:?}"),
        }
        assert_eq!(counter.field::<i64>("value").expect("value field"), 15);
    }
}

fn matches_lust_struct(value: &Value, ty: &Type) -> bool {
    match (value, &ty.kind) {
        (Value::Struct { name, .. }, TypeKind::Named(expected)) => {
            lust_type_names_match(name, expected)
        }
        (Value::Struct { name, .. }, TypeKind::GenericInstance { name: expected, .. }) => {
            lust_type_names_match(name, expected)
        }

        (value, TypeKind::Union(types)) => types.iter().any(|alt| matches_lust_struct(value, alt)),
        (_, TypeKind::Unknown) => true,
        _ => false,
    }
}

fn matches_lust_enum(value: &Value, ty: &Type) -> bool {
    match (value, &ty.kind) {
        (Value::Enum { enum_name, .. }, TypeKind::Named(expected)) => {
            lust_type_names_match(enum_name, expected)
        }
        (Value::Enum { enum_name, .. }, TypeKind::GenericInstance { name: expected, .. }) => {
            lust_type_names_match(enum_name, expected)
        }

        (value, TypeKind::Union(types)) => types.iter().any(|alt| matches_lust_enum(value, alt)),
        (_, TypeKind::Unknown) => true,
        _ => false,
    }
}

fn lust_type_names_match(value: &str, expected: &str) -> bool {
    if value == expected {
        return true;
    }

    let normalized_value = normalize_global_name(value);
    let normalized_expected = normalize_global_name(expected);
    if normalized_value == normalized_expected {
        return true;
    }

    simple_type_name(&normalized_value) == simple_type_name(&normalized_expected)
}

fn simple_type_name(name: &str) -> &str {
    name.rsplit(|c| c == '.' || c == ':').next().unwrap_or(name)
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

fn matches_array_handle_type(ty: &Type) -> bool {
    match &ty.kind {
        TypeKind::Array(_) | TypeKind::Unknown => true,
        TypeKind::Union(types) => types.iter().any(|alt| matches_array_handle_type(alt)),
        _ => false,
    }
}

fn matches_map_handle_type(ty: &Type) -> bool {
    match &ty.kind {
        TypeKind::Map(_, _) | TypeKind::Unknown => true,
        TypeKind::Union(types) => types.iter().any(|alt| matches_map_handle_type(alt)),
        _ => false,
    }
}

fn matches_function_handle_type(ty: &Type) -> bool {
    match &ty.kind {
        TypeKind::Function { .. } | TypeKind::Unknown => true,
        TypeKind::Union(types) => types.iter().any(|alt| matches_function_handle_type(alt)),
        _ => false,
    }
}

fn signatures_match(a: &FunctionSignature, b: &FunctionSignature) -> bool {
    if a.is_method != b.is_method || a.params.len() != b.params.len() {
        return false;
    }

    if a.return_type != b.return_type {
        return false;
    }

    a.params.iter().zip(&b.params).all(|(left, right)| left == right)
}

fn signature_to_string(signature: &FunctionSignature) -> String {
    let params = signature
        .params
        .iter()
        .map(|param| param.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!("function({}) -> {}", params, signature.return_type)
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

impl IntoLustValue for Rc<String> {
    fn into_value(self) -> Value {
        Value::String(self)
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

impl IntoLustValue for StructHandle {
    fn into_value(self) -> Value {
        <StructInstance as IntoLustValue>::into_value(self.into_instance())
    }

    fn matches_lust_type(ty: &Type) -> bool {
        <StructInstance as IntoLustValue>::matches_lust_type(ty)
    }

    fn type_description() -> &'static str {
        <StructInstance as IntoLustValue>::type_description()
    }
}

impl IntoLustValue for FunctionHandle {
    fn into_value(self) -> Value {
        self.into_value()
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches_function_handle_type(ty)
    }

    fn type_description() -> &'static str {
        "function"
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

impl FromLustValue for StructHandle {
    fn from_value(value: Value) -> Result<Self> {
        <StructInstance as FromLustValue>::from_value(value).map(StructHandle::from)
    }

    fn matches_lust_type(ty: &Type) -> bool {
        <StructInstance as FromLustValue>::matches_lust_type(ty)
    }

    fn type_description() -> &'static str {
        <StructInstance as FromLustValue>::type_description()
    }
}

impl FromLustValue for FunctionHandle {
    fn from_value(value: Value) -> Result<Self> {
        if FunctionHandle::is_callable_value(&value) {
            Ok(FunctionHandle::new_unchecked(value))
        } else {
            Err(LustError::RuntimeError {
                message: format!("Expected Lust value 'function' but received '{:?}'", value),
            })
        }
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches_function_handle_type(ty)
    }

    fn type_description() -> &'static str {
        "function"
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

impl IntoLustValue for ArrayHandle {
    fn into_value(self) -> Value {
        Value::Array(self.inner)
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches_array_handle_type(ty)
    }

    fn type_description() -> &'static str {
        "array"
    }
}

impl IntoLustValue for MapHandle {
    fn into_value(self) -> Value {
        Value::Map(self.inner)
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches_map_handle_type(ty)
    }

    fn type_description() -> &'static str {
        "map"
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

impl FromLustValue for ArrayHandle {
    fn from_value(value: Value) -> Result<Self> {
        match value {
            Value::Array(items) => Ok(ArrayHandle::from_rc(items)),
            other => Err(LustError::RuntimeError {
                message: format!("Expected Lust value 'array' but received '{:?}'", other),
            }),
        }
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches_array_handle_type(ty)
    }

    fn type_description() -> &'static str {
        "array"
    }
}

impl FromLustValue for MapHandle {
    fn from_value(value: Value) -> Result<Self> {
        match value {
            Value::Map(map) => Ok(MapHandle::from_rc(map)),
            other => Err(LustError::RuntimeError {
                message: format!("Expected Lust value 'map' but received '{:?}'", other),
            }),
        }
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches_map_handle_type(ty)
    }

    fn type_description() -> &'static str {
        "map"
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

impl FromLustValue for Rc<String> {
    fn from_value(value: Value) -> Result<Self> {
        match value {
            Value::String(s) => Ok(s),
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
