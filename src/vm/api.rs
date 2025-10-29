use super::*;
use crate::config::LustConfig;
use std::rc::Rc;

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
            functions: Vec::new(),
            natives: HashMap::new(),
            globals: HashMap::new(),
            call_stack: Vec::new(),
            max_stack_depth: 1000,
            pending_return_value: None,
            pending_return_dest: None,
            jit: JitState::new(),
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
        vm.trait_impls
            .insert(("int".to_string(), "Hashable".to_string()), true);
        vm.trait_impls
            .insert(("float".to_string(), "Hashable".to_string()), true);
        vm.trait_impls
            .insert(("string".to_string(), "Hashable".to_string()), true);
        vm.trait_impls
            .insert(("bool".to_string(), "Hashable".to_string()), true);
        for (name, func) in stdlib::create_stdlib(config) {
            vm.register_native(name, func);
        }

        vm
    }

    pub(super) fn observe_value(&mut self, value: &Value) {
        self.cycle_collector.register_value(value);
    }

    pub(super) fn maybe_collect_cycles(&mut self) {
        let mut collector = std::mem::take(&mut self.cycle_collector);
        collector.maybe_collect(self);
        self.cycle_collector = collector;
    }

    pub fn with_current<F, R>(f: F) -> std::result::Result<R, String>
    where
        F: FnOnce(&mut VM) -> std::result::Result<R, String>,
    {
        CURRENT_VM_STACK.with(|stack_cell| {
            let ptr_opt = {
                let stack = stack_cell.borrow();
                stack.last().copied()
            };
            if let Some(ptr) = ptr_opt {
                let vm = unsafe { &mut *ptr };
                f(vm)
            } else {
                Err("task API requires a running VM".to_string())
            }
        })
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
            let layout = Rc::new(StructLayout::new(
                def.name.clone(),
                field_names,
                field_storage,
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
                self.natives.insert(name, value);
            }

            other => {
                self.globals.insert(name, other);
            }
        }
    }

    pub(crate) fn push_export_prefix(&mut self, crate_name: &str) {
        let sanitized = crate_name.replace('-', "_");
        let prefix = format!("externs.{sanitized}");
        self.export_prefix_stack.push(prefix);
    }

    pub(crate) fn pop_export_prefix(&mut self) {
        self.export_prefix_stack.pop();
    }

    fn current_export_prefix(&self) -> Option<&str> {
        self.export_prefix_stack.last().map(|s| s.as_str())
    }

    pub fn register_exported_native<F>(&mut self, export: NativeExport, func: F)
    where
        F: Fn(&[Value]) -> std::result::Result<NativeCallResult, String> + 'static,
    {
        let mut export = export;
        if let Some(prefix) = self.current_export_prefix() {
            if !export.name.starts_with("externs.") {
                let name = if export.name.is_empty() {
                    prefix.to_string()
                } else {
                    format!("{prefix}.{}", export.name)
                };
                export.name = name;
            }
        }
        let name = export.name.clone();
        self.exported_natives.push(export);
        let native = Value::NativeFunction(Rc::new(func));
        self.register_native(name, native);
    }

    pub fn exported_natives(&self) -> &[NativeExport] {
        &self.exported_natives
    }

    pub fn take_exported_natives(&mut self) -> Vec<NativeExport> {
        std::mem::take(&mut self.exported_natives)
    }

    pub fn clear_native_functions(&mut self) {
        self.natives.clear();
    }

    pub fn get_global(&self, name: &str) -> Option<Value> {
        if let Some(value) = self.globals.get(name) {
            Some(value.clone())
        } else {
            self.natives.get(name).cloned()
        }
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
            registers: std::array::from_fn(|_| Value::Nil),
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
}
