use super::*;
use crate::ast::Type;
use crate::bytecode::{LustMap, ValueKey};
use crate::config::LustConfig;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::result::Result as CoreResult;
use core::{array, mem};
use hashbrown::HashMap;

#[derive(Debug, Clone)]
pub struct NativeExportParam {
    name: String,
    ty: String,
}

impl NativeExportParam {
    pub fn new(name: impl Into<String>, ty: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ty: ty.into(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn ty(&self) -> &str {
        &self.ty
    }
}

#[derive(Debug, Clone)]
pub struct NativeExport {
    name: String,
    params: Vec<NativeExportParam>,
    return_type: String,
    doc: Option<String>,
}

impl NativeExport {
    pub fn new(
        name: impl Into<String>,
        params: Vec<NativeExportParam>,
        return_type: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            params,
            return_type: return_type.into(),
            doc: None,
        }
    }

    pub fn with_doc(mut self, doc: impl Into<String>) -> Self {
        self.doc = Some(doc.into());
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn params(&self) -> &[NativeExportParam] {
        &self.params
    }

    pub fn return_type(&self) -> &str {
        &self.return_type
    }

    pub fn doc(&self) -> Option<&str> {
        self.doc.as_deref()
    }
}
impl VM {
    pub fn new() -> Self {
        Self::with_config(&LustConfig::default())
    }

    pub fn with_config(config: &LustConfig) -> Self {
        let mut vm = Self {
            jit: JitState::new(),
            functions: Vec::new(),
            natives: HashMap::new(),
            globals: HashMap::new(),
            map_hasher: DefaultHashBuilder::default(),
            call_stack: Vec::new(),
            max_stack_depth: 1000,
            pending_return_value: None,
            pending_return_dest: None,
            trace_recorder: None,
            side_trace_context: None,
            skip_next_trace_record: false,
            trait_impls: HashMap::new(),
            struct_tostring_cache: HashMap::new(),
            struct_metadata: HashMap::new(),
            call_until_depth: None,
            task_manager: TaskManager::new(),
            current_task: None,
            pending_task_signal: None,
            last_task_signal: None,
            cycle_collector: cycle::CycleCollector::new(),
            exported_natives: Vec::new(),
            export_prefix_stack: Vec::new(),
            exported_type_stubs: Vec::new(),
        };
        vm.jit.enabled = vm.jit.enabled && config.jit_enabled();
        vm.trait_impls
            .insert(("int".to_string(), "ToString".to_string()), true);
        vm.trait_impls
            .insert(("float".to_string(), "ToString".to_string()), true);
        vm.trait_impls
            .insert(("string".to_string(), "ToString".to_string()), true);
        vm.trait_impls
            .insert(("bool".to_string(), "ToString".to_string()), true);
        super::corelib::install_core_builtins(&mut vm);
        #[cfg(feature = "std")]
        for (name, func) in super::stdlib::create_stdlib(config, &vm) {
            vm.register_native(name, func);
        }

        vm
    }

    pub(crate) fn new_map(&self) -> LustMap {
        HashMap::with_hasher(self.map_hasher.clone())
    }

    pub(crate) fn map_with_entries(
        &self,
        entries: impl IntoIterator<Item = (ValueKey, Value)>,
    ) -> Value {
        let mut map = self.new_map();
        map.extend(entries);
        Value::map(map)
    }

    pub(crate) fn new_map_value(&self) -> Value {
        Value::map(self.new_map())
    }

    pub(super) fn observe_value(&mut self, value: &Value) {
        self.cycle_collector.register_value(value);
    }

    pub(super) fn maybe_collect_cycles(&mut self) {
        let mut collector = mem::take(&mut self.cycle_collector);
        collector.maybe_collect(self);
        self.cycle_collector = collector;
    }

    pub fn with_current<F, R>(f: F) -> CoreResult<R, String>
    where
        F: FnOnce(&mut VM) -> CoreResult<R, String>,
    {
        let ptr_opt = super::with_vm_stack(|stack| stack.last().copied());
        if let Some(ptr) = ptr_opt {
            let vm = unsafe { &mut *ptr };
            f(vm)
        } else {
            Err("task API requires a running VM".to_string())
        }
    }

    pub fn load_functions(&mut self, functions: Vec<Function>) {
        self.functions = functions;
    }

    pub fn register_structs(&mut self, defs: &HashMap<String, StructDef>) {
        for (name, def) in defs {
            let field_names: Vec<Rc<String>> = def
                .fields
                .iter()
                .map(|field| Rc::new(field.name.clone()))
                .collect();
            let field_storage: Vec<FieldStorage> = def
                .fields
                .iter()
                .map(|field| match field.ownership {
                    FieldOwnership::Weak => FieldStorage::Weak,
                    FieldOwnership::Strong => FieldStorage::Strong,
                })
                .collect();
            let field_types: Vec<Type> = def.fields.iter().map(|field| field.ty.clone()).collect();
            let weak_targets: Vec<Option<Type>> = def
                .fields
                .iter()
                .map(|field| field.weak_target.clone())
                .collect();
            let layout = Rc::new(StructLayout::new(
                def.name.clone(),
                field_names,
                field_storage,
                field_types,
                weak_targets,
            ));
            self.struct_metadata.insert(
                name.clone(),
                RuntimeStructInfo {
                    layout: layout.clone(),
                },
            );
            if let Some(simple) = name.rsplit('.').next() {
                self.struct_metadata.insert(
                    simple.to_string(),
                    RuntimeStructInfo {
                        layout: layout.clone(),
                    },
                );
            }
        }
    }

    pub fn instantiate_struct(
        &self,
        struct_name: &str,
        fields: Vec<(Rc<String>, Value)>,
    ) -> Result<Value> {
        let info =
            self.struct_metadata
                .get(struct_name)
                .ok_or_else(|| LustError::RuntimeError {
                    message: format!("Unknown struct '{}'", struct_name),
                })?;
        Self::build_struct_value(struct_name, info, fields)
    }

    fn build_struct_value(
        struct_name: &str,
        info: &RuntimeStructInfo,
        mut fields: Vec<(Rc<String>, Value)>,
    ) -> Result<Value> {
        let layout = info.layout.clone();
        let field_count = layout.field_names().len();
        let mut ordered = vec![Value::Nil; field_count];
        let mut filled = vec![false; field_count];
        for (field_name, field_value) in fields.drain(..) {
            let index_opt = layout
                .index_of_rc(&field_name)
                .or_else(|| layout.index_of_str(field_name.as_str()));
            let index = match index_opt {
                Some(i) => i,
                None => {
                    return Err(LustError::RuntimeError {
                        message: format!("Struct '{}' has no field '{}'", struct_name, field_name),
                    })
                }
            };
            let canonical = layout
                .canonicalize_field_value(index, field_value)
                .map_err(|msg| LustError::RuntimeError { message: msg })?;
            ordered[index] = canonical;
            filled[index] = true;
        }

        if filled.iter().any(|slot| !*slot) {
            let missing: Vec<String> = layout
                .field_names()
                .iter()
                .enumerate()
                .filter_map(|(idx, name)| (!filled[idx]).then(|| (**name).clone()))
                .collect();
            return Err(LustError::RuntimeError {
                message: format!(
                    "Struct '{}' is missing required field(s): {}",
                    struct_name,
                    missing.join(", ")
                ),
            });
        }

        Ok(Value::Struct {
            name: struct_name.to_string(),
            layout,
            fields: Rc::new(RefCell::new(ordered)),
        })
    }

    pub fn register_trait_impl(&mut self, type_name: String, trait_name: String) {
        self.trait_impls.insert((type_name, trait_name), true);
    }

    pub fn register_native(&mut self, name: impl Into<String>, value: Value) {
        let name = name.into();
        match value {
            Value::NativeFunction(_) => {
                let cloned = value.clone();
                self.natives.insert(name.clone(), value);
                self.globals.insert(name, cloned);
            }

            other => {
                self.globals.insert(name, other);
            }
        }
    }

    pub(crate) fn push_export_prefix(&mut self, crate_name: &str) {
        let sanitized = crate_name.replace('-', "_");
        self.export_prefix_stack.push(sanitized);
    }

    pub(crate) fn pop_export_prefix(&mut self) {
        self.export_prefix_stack.pop();
    }

    fn current_export_prefix(&self) -> Option<&str> {
        self.export_prefix_stack.last().map(|s| s.as_str())
    }

    pub fn export_prefix(&self) -> Option<String> {
        self.current_export_prefix().map(|s| s.to_string())
    }

    fn canonicalize_export_name(&self, export: &mut NativeExport) {
        if let Some(prefix) = self.current_export_prefix() {
            let needs_prefix = match export.name.strip_prefix(prefix) {
                Some(rest) => {
                    if rest.is_empty() {
                        false
                    } else {
                        !matches!(rest.chars().next(), Some('.') | Some(':'))
                    }
                }
                None => true,
            };
            if needs_prefix {
                export.name = if export.name.is_empty() {
                    prefix.to_string()
                } else {
                    format!("{prefix}.{}", export.name)
                };
            }
        }
    }

    fn push_export_metadata(&mut self, export: NativeExport) {
        if self.exported_natives.iter().any(|existing| existing.name == export.name) {
            return;
        }
        self.exported_natives.push(export);
    }

    pub fn record_exported_native(&mut self, mut export: NativeExport) {
        self.canonicalize_export_name(&mut export);
        self.push_export_metadata(export);
    }

    pub fn register_exported_native<F>(&mut self, export: NativeExport, func: F)
    where
        F: Fn(&[Value]) -> CoreResult<NativeCallResult, String> + 'static,
    {
        let mut export = export;
        self.canonicalize_export_name(&mut export);
        let name = export.name.clone();
        self.push_export_metadata(export);
        let native = Value::NativeFunction(Rc::new(func));
        self.register_native(name, native);
    }

    pub fn register_type_stubs(&mut self, stubs: Vec<ModuleStub>) {
        if stubs.is_empty() {
            return;
        }
        self.exported_type_stubs.extend(stubs);
    }

    pub fn exported_type_stubs(&self) -> &[ModuleStub] {
        &self.exported_type_stubs
    }

    pub fn take_type_stubs(&mut self) -> Vec<ModuleStub> {
        mem::take(&mut self.exported_type_stubs)
    }

    pub fn exported_natives(&self) -> &[NativeExport] {
        &self.exported_natives
    }

    pub fn take_exported_natives(&mut self) -> Vec<NativeExport> {
        mem::take(&mut self.exported_natives)
    }

    pub fn clear_native_functions(&mut self) {
        self.natives.clear();
        self.exported_type_stubs.clear();
    }

    #[cfg(feature = "std")]
    pub fn dump_externs_to_dir(
        &self,
        output_root: impl AsRef<std::path::Path>,
    ) -> std::io::Result<Vec<std::path::PathBuf>> {
        self.dump_externs_to_dir_with_options(
            output_root,
            &crate::externs::DumpExternsOptions::default(),
        )
    }

    #[cfg(feature = "std")]
    pub fn dump_externs_to_dir_with_options(
        &self,
        output_root: impl AsRef<std::path::Path>,
        options: &crate::externs::DumpExternsOptions,
    ) -> std::io::Result<Vec<std::path::PathBuf>> {
        let files = crate::externs::extern_files_from_vm(self, options);
        crate::externs::write_extern_files(output_root, &files)
    }

    pub fn get_global(&self, name: &str) -> Option<Value> {
        if let Some(value) = self.globals.get(name) {
            Some(value.clone())
        } else {
            self.natives.get(name).cloned()
        }
    }

    pub fn global_names(&self) -> Vec<String> {
        self.globals.keys().cloned().collect()
    }

    pub fn globals_snapshot(&self) -> Vec<(String, Value)> {
        self.globals
            .iter()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect()
    }

    pub fn set_global(&mut self, name: impl Into<String>, value: Value) {
        let name = name.into();
        self.observe_value(&value);
        self.globals.insert(name.clone(), value);
        self.natives.remove(&name);
        self.maybe_collect_cycles();
    }

    pub fn call(&mut self, function_name: &str, args: Vec<Value>) -> Result<Value> {
        let func_idx = self
            .functions
            .iter()
            .position(|f| f.name == function_name)
            .ok_or_else(|| LustError::RuntimeError {
                message: format!("Function not found: {}", function_name),
            })?;
        let mut frame = CallFrame {
            function_idx: func_idx,
            ip: 0,
            registers: array::from_fn(|_| Value::Nil),
            base_register: 0,
            return_dest: None,
            upvalues: Vec::new(),
        };
        let func = &self.functions[func_idx];
        if args.len() != func.param_count as usize {
            return Err(LustError::RuntimeError {
                message: format!(
                    "Function {} expects {} arguments, got {}",
                    function_name,
                    func.param_count,
                    args.len()
                ),
            });
        }

        for (i, arg) in args.into_iter().enumerate() {
            frame.registers[i] = arg;
        }

        self.call_stack.push(frame);
        match self.run() {
            Ok(v) => Ok(v),
            Err(e) => Err(self.annotate_runtime_error(e)),
        }
    }

    pub fn function_value(&self, function_name: &str) -> Option<Value> {
        let canonical = if function_name.contains("::") {
            function_name.replace("::", ".")
        } else {
            function_name.to_string()
        };
        self.functions
            .iter()
            .position(|f| f.name == canonical)
            .map(Value::Function)
    }

    pub fn function_name(&self, index: usize) -> Option<&str> {
        self.functions.get(index).map(|f| f.name.as_str())
    }

    pub fn fail_task_handle(&mut self, handle: TaskHandle, error: LustError) -> Result<()> {
        let task_id = self.task_id_from_handle(handle)?;
        let mut task =
            self.task_manager
                .detach(task_id)
                .ok_or_else(|| LustError::RuntimeError {
                    message: format!("Invalid task handle {}", handle.id()),
                })?;
        task.state = TaskState::Failed;
        task.error = Some(error.clone());
        task.last_yield = None;
        task.last_result = None;
        task.yield_dest = None;
        task.call_stack.clear();
        task.pending_return_value = None;
        task.pending_return_dest = None;
        self.task_manager.attach(task);
        Err(error)
    }
}
